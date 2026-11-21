"""Check vector persistence, resumability, integrity and actual cosine ranking."""

import hashlib
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from src.dense import Siglip2Encoder, generate_embeddings
from src.dense_store import (
    connect, get_config, load_vectors, save_vectors, search, set_config,
)


class DenseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "dense.sqlite"
        self.db = connect(self.path)
        self.addCleanup(self.db.close)
        self.config = {"model": "test", "revision": "test", "dimension": 3}
        self.rows = []
        for i, content in enumerate((b"first", b"second", b"first")):
            path = self.root / f"{i}.jpg"
            path.write_bytes(content)
            self.rows.append({"imageID": f"id{i}", "file_path": path.name,
                              "sha256": hashlib.sha256(content).hexdigest()})
        self.encoder = Mock()
        self.encoder.image_features.side_effect = lambda paths: [
            [3, 0, 0] if p.read_bytes() == b"first" else [0, 2, 0] for p in paths]

    def generate(self, **kwargs):
        return generate_embeddings(self.rows, self.root, self.db, self.config,
                                   lambda: self.encoder, **kwargs)

    def test_resume_hash_reuse_and_persistence(self):
        self.assertEqual(self.generate(limit=1), 1)
        self.assertEqual(self.generate(batch_size=2), 2)
        self.assertEqual([len(call.args[0]) for call in self.encoder.image_features.call_args_list], [1, 1])
        self.assertEqual(generate_embeddings(self.rows, self.root, self.db, self.config,
                                            lambda: self.fail("Do not load a completed model")), 0)
        with closing(connect(self.path, readonly=True)) as reopened:
            documents, vectors = load_vectors(reopened, self.rows, 3)
            self.assertEqual(get_config(reopened), self.config)
        self.assertEqual([r["imageID"] for r in documents], ["id0", "id1", "id2"])
        np.testing.assert_allclose(vectors, [[1, 0, 0], [0, 1, 0], [1, 0, 0]])

    def test_duplicate_hash_in_same_batch_is_encoded_once(self):
        self.assertEqual(self.generate(batch_size=3), 3)
        self.assertEqual(len(self.encoder.image_features.call_args.args[0]), 2)

    def test_failed_batch_preserves_previous_batches(self):
        self.encoder.image_features.side_effect = [[[1, 0, 0]], RuntimeError("inference failed")]
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            self.generate(batch_size=1)
        self.assertEqual(self.db.execute("SELECT count(*) FROM embeddings").fetchone()[0], 1)

    def test_changed_config_is_rejected_without_overwriting(self):
        self.generate(limit=1)
        with self.assertRaisesRegex(ValueError, "settings changed"):
            set_config(self.db, dict(self.config, revision="different"))
        self.assertEqual(get_config(self.db), self.config)

    def test_completed_images_are_revalidated_against_actual_bytes(self):
        self.generate()
        (self.root / "0.jpg").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Image changed"):
            self.generate()

    def test_path_cannot_escape_data_root(self):
        for bad in ("../outside.jpg", str((self.root.parent / "outside.jpg").resolve())):
            with self.subTest(path=bad), self.assertRaisesRegex(ValueError, "outside data-root"):
                generate_embeddings([dict(self.rows[0], file_path=bad)], self.root, self.db,
                                    self.config, lambda: self.encoder)

    def test_cosine_ranking_ties_and_top_k(self):
        self.generate()
        documents, vectors = load_vectors(self.db, self.rows, 3)
        results = search(documents, vectors, [8, 0, 0], 20)
        self.assertEqual([r["imageID"] for r in results], ["id0", "id2", "id1"])
        self.assertEqual(results[0]["cosine_score"], 1.0)
        self.assertEqual(search(documents, vectors, [0, 9, 0], 1)[0]["file_path"], "1.jpg")
        with self.assertRaisesRegex(ValueError, "positive"):
            search(documents, vectors, [1, 0, 0], 0)

    def test_partial_stale_and_removed_manifest_rows(self):
        self.generate(limit=1)
        with self.assertRaisesRegex(ValueError, "Missing/stale"):
            load_vectors(self.db, self.rows, 3)
        documents, _ = load_vectors(self.db, self.rows, 3, allow_partial=True)
        self.assertEqual(len(documents), 1)
        with self.assertRaisesRegex(ValueError, "Missing/stale"):
            load_vectors(self.db, [dict(self.rows[0], sha256="changed")], 3)
        self.generate()
        documents, _ = load_vectors(self.db, self.rows[1:2], 3)
        self.assertEqual([r["imageID"] for r in documents], ["id1"])

    def test_invalid_vectors_do_not_partially_commit_a_batch(self):
        set_config(self.db, self.config)
        for bad in ([[1, 0, 0], [float("nan"), 1, 0]],
                    [[1, 0, 0], [0, 0, 0]], [[1, 2], [3, 4]], [[1, 0, 0]]):
            with self.subTest(vectors=bad), self.assertRaises(ValueError):
                save_vectors(self.db, self.rows[:2], bad, 3)
        self.assertEqual(self.db.execute("SELECT count(*) FROM embeddings").fetchone()[0], 0)

    def test_corrupt_persisted_vector_is_rejected(self):
        self.generate()
        with self.db:
            self.db.execute("UPDATE embeddings SET vector=? WHERE imageID='id0'", (b"broken",))
        with self.assertRaisesRegex(ValueError, "Corrupt"):
            load_vectors(self.db, self.rows, 3)

    def test_lexical_database_cannot_be_used_for_dense_build(self):
        path = self.root / "lexical.sqlite"
        with closing(sqlite3.connect(path)) as db:
            db.execute("CREATE TABLE captions (caption TEXT)")
            db.execute("INSERT INTO captions VALUES ('keep this caption')")
            db.commit()
        with self.assertRaisesRegex(ValueError, "separate"):
            connect(path)
        with closing(sqlite3.connect(path)) as db:
            self.assertEqual(db.execute("SELECT caption FROM captions").fetchone()[0], "keep this caption")

    def test_encoder_text_preprocessing_and_both_feature_apis(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Install requirements-dense.txt for encoder tests")
        encoder = Siglip2Encoder.__new__(Siglip2Encoder)
        encoder.torch = torch
        encoder.device = "cpu"
        encoder.processor = Mock()
        encoder.processor.return_value.to.return_value = {"input_ids": torch.ones((1, 64), dtype=torch.long)}
        encoder.model = Mock()
        for output in (torch.tensor([[1., 2., 3.]]),
                       SimpleNamespace(pooler_output=torch.tensor([[1., 2., 3.]]))):
            encoder.model.get_text_features.return_value = output
            np.testing.assert_allclose(encoder.text_features(["  A RED SHIRT  "]), [[1, 2, 3]])
            self.assertEqual(encoder.processor.call_args.kwargs,
                             {"text": ["a red shirt"], "padding": "max_length", "max_length": 64,
                              "truncation": True, "return_tensors": "pt"})
        with self.assertRaisesRegex(ValueError, "empty"):
            encoder.text_features([" "])


if __name__ == "__main__":
    unittest.main()
