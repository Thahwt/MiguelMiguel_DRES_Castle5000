"""Manual QA evidence previews and a local draft shared by two notebooks."""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def validate_draft(draft, rows):
    if not isinstance(draft, dict) or draft.get("schema_version") != 1:
        raise ValueError("Bản nháp QA không đúng schema_version=1.")
    for key in ("question", "query", "imageID", "answer"):
        if not isinstance(draft.get(key), str) or not draft[key].strip():
            raise ValueError(f"Bản nháp thiếu {key}.")
    if not isinstance(draft.get("task_label", ""), str):
        raise ValueError("task_label phải là chuỗi.")
    row = next((row for row in rows if row["imageID"] == draft["imageID"]), None)
    if row is None:
        raise ValueError("Ảnh bằng chứng không có trong manifest.")
    if draft.get("file_path") != row["file_path"] or draft.get("sha256") != row["sha256"]:
        raise ValueError("Bản nháp không còn khớp manifest; hãy chọn lại ảnh bằng chứng.")
    return draft


def make_draft(question, query, image_id, answer, rows, task_label=""):
    row = next((row for row in rows if row["imageID"] == image_id.strip()), None)
    if row is None:
        raise ValueError("Chọn imageID có trong manifest.")
    draft = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
             "task_label": task_label.strip(), "question": question.strip(),
             "query": query.strip(), "imageID": row["imageID"],
             "file_path": row["file_path"], "sha256": row["sha256"], "answer": answer.strip()}
    return validate_draft(draft, rows)


def save_draft(path, draft, rows):
    validate_draft(draft, rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        json.dump(draft, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        temporary = stream.name
    os.replace(temporary, path)


def load_draft(path, rows):
    with Path(path).open(encoding="utf-8") as stream:
        return validate_draft(json.load(stream), rows)


def evidence_image(data_root, row, crop=None):
    from PIL import Image

    data_root = Path(data_root).resolve()
    path = (data_root / row["file_path"]).resolve()
    if not path.is_relative_to(data_root):
        raise ValueError("Ảnh nằm ngoài data-root.")
    with Image.open(path) as source:
        image = source.convert("RGB")
    if crop is not None:
        if (len(crop) != 4 or any(type(value) is not int for value in crop)
                or not 0 <= crop[0] < crop[2] <= image.width
                or not 0 <= crop[1] < crop[3] <= image.height):
            image.close()
            raise ValueError("Crop phải là (left, top, right, bottom) trong kích thước ảnh.")
        cropped = image.crop(crop)
        image.close()
        image = cropped
    return image
