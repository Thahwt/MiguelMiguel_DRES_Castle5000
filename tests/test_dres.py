"""Verify the supplied DRES protocol without contacting the competition server."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import requests

from src.dres_client import DresClient, DresError, to_item_name


def response(status, data):
    result = Mock(status_code=status, text=json.dumps(data, ensure_ascii=False))
    result.json.return_value = data
    return result


class DresTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.history = Path(self.temp.name) / "history.jsonl"
        self.http = Mock()
        self.task = {"taskType": "LSC Question & Answer Text", "taskId": "qa-1", "name": "QA01"}
        self.posts = []
        self.submit_response = response(202, {"submission": "INDETERMINATE"})
        self.active = True

        def request(method, url, **kwargs):
            if url.endswith("/login"):
                return response(200, {"sessionId": "private-session"})
            self.assertEqual(kwargs["params"], {"session": "private-session"})
            self.assertFalse(kwargs["allow_redirects"])
            self.assertEqual(kwargs["timeout"], 15)
            if url.endswith("/client/evaluation/list"):
                return response(200, [{"id": "evaluation-1", "status": "ACTIVE" if self.active else "TERMINATED"}])
            if "/currentTask/" in url:
                return response(200, dict(self.task))
            if "/submit/" in url:
                self.posts.append(kwargs["json"])
                if isinstance(self.submit_response, Exception):
                    raise self.submit_response
                return self.submit_response
            self.fail(f"Unexpected request: {method}")

        self.http.request.side_effect = request
        self.client = DresClient(history_path=self.history, session=self.http)
        self.client.login("member", "private-password")

    def prepare(self, kind="QA", value=" 3 "):
        return self.client.prepare("evaluation-1", "QA01", kind, value)

    def test_qa_payload_pending_and_history_have_no_credentials(self):
        prepared = self.prepare()
        self.assertEqual(self.posts, [])
        result = self.client.submit(prepared)
        self.assertEqual(result["http_status"], 202)
        self.assertEqual(result["submission"], "INDETERMINATE")
        self.assertEqual(self.posts, [{"answerSets": [{"answers": [{"text": "3"}]}]}])
        text = self.history.read_text(encoding="utf-8")
        self.assertNotIn("private-session", text)
        self.assertNotIn("private-password", text)
        self.assertEqual([json.loads(line)["state"] for line in text.splitlines()], ["ATTEMPTED", "RECEIVED"])

    def test_kis_preserves_leading_zero_and_collection(self):
        self.task = {"taskType": "LSC Known-Item Search", "taskId": "kis-1"}
        prepared = self.prepare("KIS", r"C:\data\08_00102.jpg")
        self.client.submit(prepared)
        self.assertEqual(self.posts[0]["answerSets"][0]["answers"], [{
            "mediaItemName": "day1_Kitchen_08_00102", "mediaItemCollectionName": "castle5000"}])
        self.assertEqual(to_item_name("day1_Kitchen_08_00102"), "day1_Kitchen_08_00102")

    def test_wrong_type_unknown_type_empty_answer_and_inactive_are_not_posted(self):
        with self.assertRaises(DresError):
            self.prepare("KIS", "08_102.jpg")
        with self.assertRaises(ValueError):
            self.prepare(value="  ")
        with self.assertRaises(ValueError):
            self.client.prepare("evaluation-1", "", "QA", "3")
        self.task["taskType"] = "Unexpected task"
        with self.assertRaises(DresError):
            self.prepare()
        self.task["taskType"] = "LSC Question & Answer Text"
        self.active = False
        with self.assertRaises(DresError):
            self.prepare()
        self.assertEqual(self.posts, [])

    def test_same_type_task_change_is_detected_before_submit(self):
        prepared = self.prepare()
        self.task["taskId"] = "qa-2"
        with self.assertRaisesRegex(DresError, "đã đổi"):
            self.client.submit(prepared)
        self.assertEqual(self.posts, [])

    def test_countdown_change_does_not_change_task_identity(self):
        self.task["timeLeft"] = 100
        prepared = self.prepare()
        self.task["timeLeft"] = 90
        self.client.submit(prepared)
        self.assertEqual(len(self.posts), 1)

    def test_duplicate_is_blocked_after_kernel_restart(self):
        prepared = self.prepare()
        self.client.submit(prepared)
        restarted = DresClient(history_path=self.history, session=self.http)
        restarted.login("member", "private-password")
        with self.assertRaisesRegex(DresError, "đã gửi"):
            restarted.submit(prepared)
        self.assertEqual(len(self.posts), 1)

    def test_timeout_is_unknown_and_never_retried(self):
        prepared = self.prepare()
        self.submit_response = requests.Timeout("private-session should not be echoed")
        result = self.client.submit(prepared)
        self.assertEqual(result["state"], "UNKNOWN")
        with self.assertRaises(DresError):
            self.client.submit(prepared)
        self.assertEqual(len(self.posts), 1)
        self.assertNotIn("private-session", self.history.read_text(encoding="utf-8"))

    def test_401_reauthentication_requires_another_explicit_submit(self):
        prepared = self.prepare()
        self.submit_response = response(401, {"message": "Unauthorized"})
        self.assertEqual(self.client.submit(prepared)["state"], "REJECTED")
        self.assertEqual(len(self.posts), 1)
        self.client.login("member", "private-password")
        self.submit_response = response(200, {"submission": "CORRECT"})
        self.assertEqual(self.client.submit(prepared)["submission"], "CORRECT")
        self.assertEqual(len(self.posts), 2)

    def test_412_is_reported_and_500_is_not_retried(self):
        prepared = self.prepare()
        self.submit_response = response(412, {"message": "Duplicate submission"})
        self.assertEqual(self.client.submit(prepared)["http_status"], 412)
        self.submit_response = response(500, {"message": "Server error"})
        self.assertEqual(self.client.submit(prepared)["state"], "UNKNOWN")
        with self.assertRaises(DresError):
            self.client.submit(prepared)
        self.assertEqual(len(self.posts), 2)

    def test_invalid_history_stops_before_post(self):
        prepared = self.prepare()
        self.history.write_text("incomplete JSON", encoding="utf-8")
        with self.assertRaisesRegex(DresError, "lịch sử"):
            self.client.submit(prepared)
        self.assertEqual(self.posts, [])

    def test_rejected_http_and_connection_errors_do_not_leak_session(self):
        self.http.request.side_effect = requests.ConnectionError("https://host?session=private-session")
        with self.assertRaises(DresError) as caught:
            self.client.list_evaluations()
        self.assertNotIn("private-session", str(caught.exception))
        self.http.request.side_effect = None
        self.http.request.return_value = response(401, {"message": "private-session"})
        with self.assertRaises(DresError) as caught:
            self.client.list_evaluations()
        self.assertNotIn("private-session", str(caught.exception))

    def test_no_current_task(self):
        self.http.request.side_effect = None
        self.http.request.return_value = response(404, {"message": "No task"})
        self.assertIsNone(self.client.current_task("evaluation-1"))

    def test_bad_media_names_are_rejected(self):
        for name in ("24_100.jpg", "8_102.jpg", "08_102.txt", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                to_item_name(name)


if __name__ == "__main__":
    unittest.main()
