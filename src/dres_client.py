"""Small DRES v2 client with explicit preparation and single-attempt submission."""

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlsplit

import requests

DEFAULT_URL = "https://dres.2hbread.com/api/v2"
COLLECTION = "castle5000"
TASK_TYPES = {
    "LSC Known-Item Search": "KIS",
    "LSC Question & Answer Text": "QA",
}


class DresError(RuntimeError):
    """User-facing error that never includes an authenticated request URL."""


def to_item_name(value):
    stem = str(value).strip().replace("\\", "/").rsplit("/", 1)[-1]
    stem = re.sub(r"\.(jpg|jpeg|png)$", "", stem, flags=re.IGNORECASE)
    if not stem.startswith("day1_Kitchen_"):
        stem = "day1_Kitchen_" + stem
    match = re.fullmatch(r"day1_Kitchen_([0-2][0-9])_([0-9]+)", stem)
    if not match or int(match[1]) > 23:
        raise ValueError("Tên ảnh phải có dạng HH_frame.jpg hoặc day1_Kitchen_HH_frame.")
    return stem


def task_kind(task):
    kind = TASK_TYPES.get(task.get("taskType")) if isinstance(task, dict) else None
    if kind is None:
        raise DresError("Không có task KIS/QA được hỗ trợ. Hãy làm mới task.")
    return kind


def task_marker(task):
    # Do not include changing countdown fields in task identity. Some DRES
    # deployments expose only taskType; the notebook also requires a task label.
    fields = ("taskType", "id", "taskId", "name", "taskName")
    return json.dumps({key: task[key] for key in fields if key in task},
                      sort_keys=True, ensure_ascii=False)


@dataclass(frozen=True)
class PreparedSubmission:
    evaluation_id: str
    task_label: str
    kind: str
    marker: str
    value: str

    @property
    def payload(self):
        if self.kind == "QA":
            answer = {"text": self.value}
        elif self.kind == "KIS":
            answer = {"mediaItemName": self.value, "mediaItemCollectionName": COLLECTION}
        else:
            raise ValueError("Chỉ hỗ trợ KIS hoặc QA.")
        return {"answerSets": [{"answers": [answer]}]}


class DresClient:
    def __init__(self, base_url=DEFAULT_URL, history_path=None, timeout=15, session=None):
        parsed = urlsplit(base_url)
        if (parsed.scheme not in {"https", "http"} or not parsed.netloc
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("DRES URL không hợp lệ; không đặt tài khoản/session trong URL.")
        self.base_url = base_url.rstrip("/")
        self.history_path = Path(history_path) if history_path else None
        self.timeout = timeout
        self.http = session if session is not None else requests.Session()
        self._session_id = None
        self._attempted = set()

    def _redact(self, text):
        text = str(text)
        return text.replace(self._session_id, "[SESSION]") if self._session_id else text

    def _request(self, method, path, authenticated=True, **kwargs):
        params = {}
        if authenticated:
            if not self._session_id:
                raise DresError("Chưa đăng nhập DRES.")
            params["session"] = self._session_id
        try:
            # No retry and no redirects: POST must be sent at most once.
            return self.http.request(method, self.base_url + path, params=params,
                                     timeout=self.timeout, allow_redirects=False, **kwargs)
        except requests.RequestException:
            raise DresError("Lỗi kết nối DRES. Yêu cầu không được tự động gửi lại.") from None

    def _json(self, response):
        if not 200 <= response.status_code < 300:
            detail = self._redact(response.text)[:2000]
            raise DresError(f"DRES HTTP {response.status_code}: {detail}")
        try:
            return response.json()
        except ValueError:
            raise DresError("DRES trả dữ liệu không phải JSON.") from None

    def login(self, username, password):
        self._session_id = None
        if not username.strip() or not password:
            raise ValueError("Cần username và password.")
        response = self._request("POST", "/login", authenticated=False,
                                 json={"username": username.strip(), "password": password})
        # Do not echo login response bodies, which may contain credentials.
        if response.status_code != 200:
            raise DresError(f"Đăng nhập thất bại: HTTP {response.status_code}.")
        data = self._json(response)
        if not isinstance(data, dict) or not isinstance(data.get("sessionId"), str) or not data["sessionId"]:
            raise DresError("Phản hồi đăng nhập thiếu sessionId.")
        self._session_id = data["sessionId"]

    def list_evaluations(self):
        data = self._json(self._request("GET", "/client/evaluation/list"))
        if not isinstance(data, list) or any(not isinstance(item, dict) for item in data):
            raise DresError("Danh sách evaluation không hợp lệ.")
        return data

    def current_task(self, evaluation_id):
        response = self._request("GET", "/client/evaluation/currentTask/" + quote(evaluation_id, safe=""))
        if response.status_code == 404:
            return None
        data = self._json(response)
        if not isinstance(data, dict):
            raise DresError("Phản hồi currentTask không hợp lệ.")
        return data

    def _active_task(self, evaluation_id):
        active = any(item.get("id") == evaluation_id and item.get("status") == "ACTIVE"
                     for item in self.list_evaluations())
        if not active:
            raise DresError("Evaluation đã chọn không ACTIVE hoặc không thuộc tài khoản.")
        task = self.current_task(evaluation_id)
        task_kind(task)
        return task

    def prepare(self, evaluation_id, task_label, kind, value):
        evaluation_id, task_label = evaluation_id.strip(), task_label.strip()
        if not evaluation_id or not task_label:
            raise ValueError("Chọn evaluation và nhập đúng tên task trên Viewer, ví dụ QA01.")
        value = value.strip()
        if not value:
            raise ValueError("Bài nộp không được rỗng.")
        task = self._active_task(evaluation_id)
        if kind != task_kind(task):
            raise DresError("Loại bài nộp không khớp task đang chạy.")
        if kind == "KIS":
            value = to_item_name(value)
        return PreparedSubmission(evaluation_id, task_label, kind, task_marker(task), value)

    def _key(self, prepared):
        content = [self.base_url, prepared.evaluation_id, prepared.task_label,
                   prepared.marker, prepared.kind, prepared.value]
        return hashlib.sha256(json.dumps(content, ensure_ascii=False).encode("utf-8")).hexdigest()

    def _blocked_keys(self):
        keys = set(self._attempted)
        if self.history_path and self.history_path.exists():
            try:
                with self.history_path.open(encoding="utf-8") as stream:
                    latest = {}
                    for line in stream:
                        entry = json.loads(line)
                        latest[entry["key"]] = entry["state"]
                keys.update(key for key, state in latest.items()
                            if state in {"ATTEMPTED", "RECEIVED", "UNKNOWN"})
            except (OSError, ValueError, KeyError, TypeError):
                raise DresError("Không đọc được lịch sử nộp. Kiểm tra file trước khi gửi tiếp.") from None
        return keys

    def _record(self, prepared, key, state, **extra):
        if self.history_path is None:
            return
        entry = {"time_utc": datetime.now(timezone.utc).isoformat(), "key": key,
                 "evaluation_id": prepared.evaluation_id, "task_label": prepared.task_label,
                 "kind": prepared.kind, "payload": prepared.payload, "state": state, **extra}
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with self.history_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
                stream.flush()
        except OSError:
            raise DresError("Không ghi được lịch sử nộp. Kiểm tra trạng thái bài trên DRES trước khi thử lại.") from None

    def submit(self, prepared):
        if not isinstance(prepared, PreparedSubmission):
            raise ValueError("Cần xem trước bài bằng prepare() trước khi nộp.")
        # Revalidate content, evaluation and task immediately before POST.
        current = self.prepare(prepared.evaluation_id, prepared.task_label,
                               prepared.kind, prepared.value)
        if current != prepared:
            raise DresError("Task hoặc nội dung đã đổi. Hãy xem trước bài nộp lại.")
        key = self._key(prepared)
        if key in self._blocked_keys():
            raise DresError("Bài này đã gửi hoặc chưa rõ kết quả; không gửi lại. Kiểm tra DRES/lịch sử.")
        self._record(prepared, key, "ATTEMPTED")
        self._attempted.add(key)
        try:
            response = self._request("POST", "/submit/" + quote(prepared.evaluation_id, safe=""),
                                     json=prepared.payload)
        except DresError:
            result = {"http_status": None, "submission": None, "state": "UNKNOWN",
                      "detail": "Lỗi mạng: chưa biết DRES đã nhận bài hay chưa. Không tự nộp lại."}
        else:
            try:
                data = response.json()
            except ValueError:
                data = {}
            state = ("RECEIVED" if response.status_code in {200, 202} else
                     "REJECTED" if 400 <= response.status_code < 500 else "UNKNOWN")
            submission = data.get("submission") if isinstance(data, dict) else None
            result = {"http_status": response.status_code, "submission": submission,
                      "state": state, "detail": self._redact(response.text)[:2000]}
        self._record(prepared, key, **result)
        if result["state"] == "REJECTED":
            self._attempted.discard(key)
        return result
