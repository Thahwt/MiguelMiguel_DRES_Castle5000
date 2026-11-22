"""Exercise notebook buttons without a live DRES server or an image model."""

import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from src.dres_client import PreparedSubmission
from src.notebook_ui import dres_panel, manual_qa_panel
from src.qa import load_draft, make_draft, save_draft


def control(panel, description):
    if getattr(panel, "description", None) == description:
        return panel
    for child in getattr(panel, "children", ()):
        result = control(child, description)
        if result is not None:
            return result
    return None


@unittest.skipUnless(importlib.util.find_spec("ipywidgets"), "Install requirements-client.txt for widget tests")
class NotebookUITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.draft_path = self.root / "qa_draft.json"
        self.rows = []
        for index in (1, 2):
            name = f"08_{index}.jpg"
            Image.new("RGB", (100, 60), "red").save(self.root / name)
            self.rows.append({"imageID": f"day1_Kitchen_08_{index}", "file_path": name, "sha256": f"hash{index}"})
        self.addCleanup(patch.stopall)
        patch("IPython.display.display").start()
        self.output = io.StringIO()
        patch("sys.stdout", self.output).start()

    def test_manual_draft_button_and_evidence_change(self):
        retrieval = {"query": "person near refrigerator", "hybrid": self.rows}
        panel = manual_qa_panel(retrieval, self.rows, self.root, self.draft_path)
        self.addCleanup(panel.close)
        self.assertFalse(self.draft_path.exists())
        control(panel, "Câu hỏi:").value = "Áo màu gì?"
        control(panel, "Đáp án:").value = "đỏ"
        control(panel, "Tên task:").value = "QA01"
        control(panel, "Lưu bản nháp QA").click()
        self.assertEqual(load_draft(self.draft_path, self.rows)["answer"], "đỏ")
        control(panel, "Ảnh:").value = self.rows[1]["imageID"]
        self.assertEqual(control(panel, "Đáp án:").value, "")

    def test_dres_panel_never_sends_without_preview_and_click(self):
        client = Mock(history_path=self.root / "history.jsonl")
        client.list_evaluations.return_value = [{"id": "evaluation-1", "name": "Practice", "status": "ACTIVE"}]
        client.prepare.side_effect = lambda evaluation, label, kind, value: PreparedSubmission(
            evaluation, label, kind, '{"taskType":"LSC Question & Answer Text","taskId":"qa-1"}', value)
        client.submit.return_value = {"http_status": 202, "submission": "INDETERMINATE", "state": "RECEIVED"}
        save_draft(self.draft_path, make_draft("Áo màu gì?", "person", self.rows[0]["imageID"],
                                              "đỏ", self.rows, "QA01"), self.rows)
        panel = dres_panel(client, self.rows, self.root, self.draft_path)
        self.addCleanup(panel.close)
        client.submit.assert_not_called()
        self.assertTrue(control(panel, "Nộp lên DRES").disabled)
        control(panel, "Tải buổi thi").click()
        control(panel, "Buổi thi:").value = "evaluation-1"
        control(panel, "Nạp bản nháp QA").click()
        control(panel, "Xem trước bài nộp").click()
        client.submit.assert_not_called()
        self.assertFalse(control(panel, "Nộp lên DRES").disabled)
        control(panel, "Đáp án QA:").value = "xanh"
        self.assertTrue(control(panel, "Nộp lên DRES").disabled)
        control(panel, "Xem trước bài nộp").click()
        control(panel, "Nộp lên DRES").click()
        client.submit.assert_called_once()
        self.assertEqual(client.submit.call_args.args[0].value, "xanh")
        self.assertTrue(control(panel, "Nộp lên DRES").disabled)
        self.assertIn("đang chờ trọng tài", self.output.getvalue())
        # Even a programmatic second click cannot re-use the prepared submission.
        control(panel, "Nộp lên DRES").click()
        client.submit.assert_called_once()


if __name__ == "__main__":
    unittest.main()
