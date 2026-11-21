"""Shared manifest validation and persistent caption storage (standard library)."""

import csv
import json
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "artifacts/manifest.csv"
DEFAULT_DB = PROJECT_ROOT / "artifacts/lexical.sqlite"


def read_manifest(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"imageID", "file_path", "sha256"}.issubset(reader.fieldnames or []):
            raise ValueError("Manifest needs imageID, file_path, sha256 columns")
        rows = list(reader)
    if not rows or any(not r["imageID"] or not r["file_path"] or not r["sha256"] for r in rows):
        raise ValueError("Manifest is empty or has empty required fields")
    if len({r["imageID"] for r in rows}) != len(rows):
        raise ValueError("Duplicate imageID in manifest")
    return rows


def connect(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute("""CREATE TABLE IF NOT EXISTS captions (
        imageID TEXT PRIMARY KEY, file_path TEXT NOT NULL,
        sha256 TEXT NOT NULL, caption TEXT NOT NULL)""")
    db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    db.execute("CREATE INDEX IF NOT EXISTS captions_sha256 ON captions(sha256)")
    db.commit()
    return db


def set_config(db, config, allow_prompt_change=False):
    serialized = json.dumps(config, ensure_ascii=False, sort_keys=True)
    previous = db.execute("SELECT value FROM metadata WHERE key='caption_config'").fetchone()
    if previous and previous[0] != serialized:
        old_settings = json.loads(previous[0])
        new_settings = dict(config)
        # Accepted captions ended naturally, so changing only the output budget
        # does not invalidate them. Model/prompt/preprocessing changes still do.
        old_settings.pop("max_new_tokens", None)
        new_settings.pop("max_new_tokens", None)
        if allow_prompt_change:
            old_settings.pop("prompt", None)
            new_settings.pop("prompt", None)
        if old_settings != new_settings:
            raise ValueError(
                "Caption settings changed. Use a different --db, or "
                "--allow-prompt-change if only the prompt changed."
            )
    with db:
        if previous and previous[0] != serialized:
            history = db.execute("SELECT value FROM metadata WHERE key='caption_config_history'").fetchone()
            values = json.loads(history[0]) if history else []
            values.append(json.loads(previous[0]))
            db.execute("INSERT OR REPLACE INTO metadata VALUES ('caption_config_history', ?)",
                       (json.dumps(values, ensure_ascii=False),))
        db.execute("INSERT OR REPLACE INTO metadata VALUES ('caption_config', ?)", (serialized,))


def find_caption(db, row):
    result = db.execute(
        "SELECT caption FROM captions WHERE imageID=? AND sha256=? AND file_path=?",
        (row["imageID"], row["sha256"], row["file_path"]),
    ).fetchone()
    return result[0] if result else None


def save_caption(db, row, caption):
    if not isinstance(caption, str):
        raise ValueError(f"Invalid caption for {row['imageID']}")
    caption = caption.strip()
    if not caption:
        raise ValueError(f"Empty caption for {row['imageID']}")
    with db:
        # A previous index must never silently serve captions that have changed.
        db.execute("BEGIN")
        db.execute("DROP TABLE IF EXISTS caption_fts")
        db.execute("INSERT OR REPLACE INTO captions VALUES (?, ?, ?, ?)",
                   (row["imageID"], row["file_path"], row["sha256"], caption))


def cached_by_hash(db, sha256):
    result = db.execute("SELECT caption FROM captions WHERE sha256=? LIMIT 1", (sha256,)).fetchone()
    return result[0] if result else None
