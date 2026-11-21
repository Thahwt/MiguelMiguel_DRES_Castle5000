"""Build and query a SQLite FTS5 BM25 index over Vietnamese captions."""

import argparse
import json
import re
import sqlite3
import sys
import unicodedata
from pathlib import Path

from src.lexical_store import DEFAULT_DB, DEFAULT_MANIFEST, connect, find_caption, read_manifest


def build_index(db, rows, allow_partial=False):
    documents = []
    missing = []
    for row in rows:
        caption = find_caption(db, row)
        if caption is None:
            missing.append(row["imageID"])
        else:
            documents.append((row["imageID"], row["file_path"], unicodedata.normalize("NFC", caption)))
    if missing and not allow_partial:
        raise ValueError(f"Missing/stale captions: {len(missing)}. Run captioning first or use --allow-partial.")
    if not documents:
        raise ValueError("No captions to index")
    with db:
        db.execute("BEGIN")
        db.execute("DROP TABLE IF EXISTS caption_fts")
        db.execute("""CREATE VIRTUAL TABLE caption_fts USING fts5(
            imageID UNINDEXED, file_path UNINDEXED, caption,
            tokenize='unicode61 remove_diacritics 0')""")
        db.executemany("INSERT INTO caption_fts VALUES (?, ?, ?)", documents)
    return len(documents), len(missing)


def search(db, query, top_k=10):
    if top_k < 1:
        raise ValueError("top_k must be positive")
    # Literal OR terms: no user-supplied FTS operators; allow partial word overlap.
    terms = list(dict.fromkeys(re.findall(r"[^\W_]+", unicodedata.normalize("NFC", query).casefold())))
    if not terms:
        return []
    expression = " OR ".join('"' + term + '"' for term in terms)
    if not db.execute("SELECT 1 FROM sqlite_master WHERE name='caption_fts'").fetchone():
        raise ValueError("No current BM25 index. Run: python -m src.bm25 build")
    matches = db.execute("""SELECT imageID, file_path, caption, bm25(caption_fts) AS score
        FROM caption_fts WHERE caption_fts MATCH ? ORDER BY score, imageID LIMIT ?""",
        (expression, top_k),
    )
    return [dict(zip(("imageID", "file_path", "caption", "bm25_score"), row)) for row in matches]


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    build.add_argument("--allow-partial", action="store_true")
    query = sub.add_parser("search")
    query.add_argument("query")
    query.add_argument("--top-k", type=int, default=10)
    for child in (build, query):
        child.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = parser.parse_args(argv)
    db = None
    try:
        if not args.db.is_file():
            raise ValueError("Caption database not found. Run captioning first.")
        db = connect(args.db)
        if args.command == "build":
            count, missing = build_index(db, read_manifest(args.manifest), args.allow_partial)
            print(f"Indexed {count} images; missing/stale captions: {missing}")
        else:
            print(json.dumps(search(db, args.query, args.top_k), ensure_ascii=False, indent=2))
    except (OSError, ValueError, sqlite3.Error) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        if db is not None:
            db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
