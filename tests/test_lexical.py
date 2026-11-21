"""Test persistence, resume, integrity and actual SQLite BM25 without a VLM download."""

import hashlib
import json
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock, patch

from src.bm25 import build_index, search
from src.caption import QwenCaptioner, generate_captions
from src.lexical_store import connect, read_manifest, save_caption, set_config


class LexicalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = connect(self.root / "captions.sqlite")
        self.addCleanup(self.db.close)
        self.config = {"model": "test", "prompt": "test"}
        self.rows = []
        for index, data in enumerate((b"first image", b"second image", b"first image")):
            path = self.root / f"{index}.jpg"
            path.write_bytes(data)
            self.rows.append({"imageID": f"id{index}", "file_path": path.name,
                              "sha256": hashlib.sha256(data).hexdigest()})

    def test_resume_and_hash_reuse_across_ids(self):
        calls = []

        def factory():
            def caption(path):
                calls.append(path.name)
                return "Người mặc áo đỏ cạnh tủ lạnh."
            return caption

        self.assertEqual(generate_captions(self.rows, self.root, self.db, self.config, factory, 1), 1)
        self.assertEqual(generate_captions(self.rows, self.root, self.db, self.config, factory), 2)
        self.assertEqual(calls, ["0.jpg", "1.jpg"])
        self.assertEqual(generate_captions(
            self.rows, self.root, self.db, self.config,
            lambda: self.fail("Completed run must not load model")), 0)

    def test_interruption_keeps_successful_captions(self):
        def caption(path):
            if path.name == "1.jpg":
                raise RuntimeError("simulated inference failure")
            return "người áo đỏ"
        with self.assertRaisesRegex(RuntimeError, "simulated"):
            generate_captions(self.rows, self.root, self.db, self.config, lambda: caption)
        self.assertEqual(self.db.execute("SELECT count(*) FROM captions").fetchone()[0], 1)

    def test_changed_config_or_image_is_rejected(self):
        set_config(self.db, self.config)
        with self.assertRaisesRegex(ValueError, "changed"):
            set_config(self.db, {"model": "another"})
        (self.root / "0.jpg").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Image changed"):
            generate_captions(self.rows, self.root, self.db, self.config, lambda: None)

    def test_path_cannot_escape_data_root(self):
        row = dict(self.rows[0], file_path="../outside.jpg")
        with self.assertRaisesRegex(ValueError, "outside data-root"):
            generate_captions([row], self.root, self.db, self.config, lambda: None)

    def test_actual_bm25_persistence_and_literal_query(self):
        captions = ["người áo đỏ cạnh tủ lạnh", "người áo xanh ngồi trên ghế", "bồn rửa nhà bếp"]
        for row, caption in zip(self.rows, captions):
            save_caption(self.db, row, caption)
        self.assertEqual(build_index(self.db, self.rows), (3, 0))
        results = search(self.db, "tủ lạnh", 2)
        self.assertEqual(results[0]["imageID"], "id0")
        self.assertEqual(search(self.db, "!!!"), [])
        self.assertEqual(search(self.db, "khôngtồntại"), [])
        self.assertTrue(search(self.db, '"tủ" OR (lạnh*)'))
        self.assertEqual(search(self.db, "tu\u0309 la\u0323nh")[0]["imageID"], "id0")
        with closing(sqlite3.connect(self.root / "captions.sqlite")) as reopened:
            self.assertEqual(search(reopened, "tủ lạnh")[0]["file_path"], "0.jpg")

    def test_incomplete_or_stale_captions_require_explicit_partial_index(self):
        save_caption(self.db, self.rows[0], "người áo đỏ")
        with self.assertRaisesRegex(ValueError, "Missing/stale"):
            build_index(self.db, self.rows)
        self.assertEqual(build_index(self.db, self.rows, allow_partial=True), (1, 2))
        save_caption(self.db, self.rows[1], "người áo xanh")
        with self.assertRaisesRegex(ValueError, "No current BM25"):
            search(self.db, "người")
        changed = [dict(self.rows[0], sha256="different")]
        with self.assertRaisesRegex(ValueError, "Missing/stale"):
            build_index(self.db, changed)

    def test_bad_manifest_is_rejected(self):
        path = self.root / "bad.csv"
        path.write_text("imageID,file_path,sha256\nx,x.jpg,a\nx,y.jpg,b\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            read_manifest(path)

    def test_token_budget_change_preserves_captions_and_config_history(self):
        old = dict(self.config, max_new_tokens=320)
        new = dict(self.config, max_new_tokens=768)
        set_config(self.db, old)
        save_caption(self.db, self.rows[0], "caption already complete")
        set_config(self.db, new)
        set_config(self.db, new)
        self.assertEqual(self.db.execute("SELECT count(*) FROM captions").fetchone()[0], 1)
        history = json.loads(self.db.execute(
            "SELECT value FROM metadata WHERE key='caption_config_history'").fetchone()[0])
        self.assertEqual(history, [old])
        with self.assertRaisesRegex(ValueError, "settings changed"):
            set_config(self.db, dict(new, prompt="different prompt"))

    def test_truncated_generation_retries_and_saves_diagnostics_on_failure(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Install requirements-caption.txt for generation tests")
        captioner = QwenCaptioner.__new__(QwenCaptioner)
        captioner.torch = torch
        captioner.max_new_tokens = 2
        captioner.model = Mock()
        captioner.model.generation_config.eos_token_id = [9]
        captioner.processor = Mock()
        captioner.processor.batch_decode.return_value = ["complete caption"]
        inputs = {"input_ids": torch.tensor([[1]])}
        captioner.model.generate.side_effect = [torch.tensor([[1, 3, 4]]),
                                               torch.tensor([[1, 3, 4, 5, 9]])]
        self.assertEqual(captioner.generate_text(inputs), "complete caption")
        self.assertEqual([c.kwargs["max_new_tokens"] for c in captioner.model.generate.call_args_list], [2, 4])
        captioner.processor.reset_mock()
        captioner.model.generate.side_effect = [torch.tensor([[1, 3, 4]]),
                                               torch.tensor([[1, 3, 4, 5, 6]])]
        with self.assertRaisesRegex(ValueError, "same --db"):
            captioner.generate_text(inputs, image_path=self.root / "0.jpg", error_dir=self.root / "errors")
        files = list((self.root / "errors").glob("*.json"))
        self.assertEqual(len(files), 1)
        diagnostic = json.loads(files[0].read_text(encoding="utf-8"))
        self.assertEqual(diagnostic["image_path"], str((self.root / "0.jpg").resolve()))
        self.assertEqual(diagnostic["eos_token_ids"], [9])
        self.assertEqual([a["generated_tokens"] for a in diagnostic["attempts"]], [2, 4])
        self.assertEqual(diagnostic["attempts"][-1]["token_ids"], [3, 4, 5, 6])
        self.assertEqual(diagnostic["attempts"][-1]["raw_text"], "complete caption")
        self.assertTrue(all(not c.kwargs["skip_special_tokens"]
                            for c in captioner.processor.batch_decode.call_args_list))
        self.assertEqual(self.db.execute("SELECT count(*) FROM captions").fetchone()[0], 0)

    def test_model_load_uses_bf16_on_cuda_and_fp32_on_cpu(self):
        torch = Mock()
        torch.cuda.is_available.return_value = True
        transformers = Mock()
        with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}):
            for device, expected in (("cuda", torch.bfloat16), ("cpu", torch.float32)):
                with self.subTest(device=device):
                    QwenCaptioner("test", "main", device, 401408, 768, self.root)
                    self.assertEqual(
                        transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained.call_args.kwargs["dtype"],
                        expected,
                    )
            transformers.reset_mock()
            torch.cuda.is_bf16_supported.return_value = False
            with self.assertRaisesRegex(ValueError, "does not support BF16"):
                QwenCaptioner("test", "main", "cuda", 401408, 768, self.root)
            transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained.assert_not_called()

    def test_prompt_change_requires_opt_in_and_preserves_old_captions(self):
        set_config(self.db, self.config)
        save_caption(self.db, self.rows[0], "old caption")
        new_config = dict(self.config, prompt="new prompt")
        with self.assertRaisesRegex(ValueError, "allow-prompt-change"):
            set_config(self.db, new_config)
        captioner = Mock(return_value="new caption")
        count = generate_captions(self.rows, self.root, self.db, new_config,
                                  lambda: captioner, allow_prompt_change=True)
        self.assertEqual(count, 2)
        self.assertEqual(captioner.call_count, 2)  # even a duplicate hash uses the new prompt
        self.assertEqual(dict(self.db.execute("SELECT imageID, caption FROM captions")),
                         {"id0": "old caption", "id1": "new caption", "id2": "new caption"})
        history = json.loads(self.db.execute(
            "SELECT value FROM metadata WHERE key='caption_config_history'").fetchone()[0])
        self.assertEqual(history, [self.config])
        set_config(self.db, new_config)  # subsequent runs do not need the flag again
        for change in ({"model": "another"}, {"language": "another"}, {"max_pixels": 999}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "settings changed"):
                set_config(self.db, dict(new_config, **change), allow_prompt_change=True)
        self.assertEqual(json.loads(self.db.execute(
            "SELECT value FROM metadata WHERE key='caption_config'").fetchone()[0]), new_config)


if __name__ == "__main__":
    unittest.main()
