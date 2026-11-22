"""Check manual-QA handoff and original-image preview without models or DRES."""

import hashlib
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.qa import evidence_image, load_draft, make_draft, save_draft


class ManualQATests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "08_102.jpg"
        Image.new("RGB", (100, 60), "red").save(self.path)
        self.rows = [{"imageID": "day1_Kitchen_08_102", "file_path": self.path.name,
                      "sha256": hashlib.sha256(self.path.read_bytes()).hexdigest()}]

    def draft(self, **kwargs):
        args = {"question": "Có bao nhiêu người?", "query": "people near refrigerator",
                "image_id": self.rows[0]["imageID"], "answer": " 3 ", "task_label": "QA01"}
        args.update(kwargs)
        return make_draft(rows=self.rows, **args)

    def test_save_and_load_handoff_preserves_question_answer_and_evidence(self):
        path = self.root / "draft.json"
        save_draft(path, self.draft(), self.rows)
        loaded = load_draft(path, self.rows)
        self.assertEqual(loaded["answer"], "3")
        self.assertEqual(loaded["question"], "Có bao nhiêu người?")
        self.assertEqual(loaded["imageID"], self.rows[0]["imageID"])
        self.assertEqual(loaded["task_label"], "QA01")
        changed_rows = [dict(self.rows[0], sha256="changed")]
        with self.assertRaisesRegex(ValueError, "manifest"):
            load_draft(path, changed_rows)

    def test_empty_fields_or_nonexistent_image_cannot_be_saved(self):
        for key in ("question", "query", "answer", "image_id"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.draft(**{key: " "})

    def test_preview_and_crop_do_not_change_original(self):
        original = self.path.read_bytes()
        with evidence_image(self.root, self.rows[0]) as image:
            self.assertEqual(image.size, (100, 60))
        with evidence_image(self.root, self.rows[0], (10, 10, 50, 40)) as image:
            self.assertEqual(image.size, (40, 30))
        self.assertEqual(self.path.read_bytes(), original)
        with self.assertRaises(ValueError):
            evidence_image(self.root, self.rows[0], (0, 0, 200, 60))
        with self.assertRaises(ValueError):
            evidence_image(self.root, dict(self.rows[0], file_path="../outside.jpg"))


if __name__ == "__main__":
    unittest.main()
