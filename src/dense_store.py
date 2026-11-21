"""Resumable float32 image vectors and exact cosine retrieval using SQLite/NumPy."""

import json
import sqlite3
from pathlib import Path

import numpy as np

from src.lexical_store import PROJECT_ROOT

DEFAULT_DENSE_DB = PROJECT_ROOT / "artifacts/dense.sqlite"


def connect(path, readonly=False):
    path = Path(path).resolve()
    if readonly:
        return sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if tables - {"embeddings", "dense_metadata"}:
        db.close()
        raise ValueError("Not a dense database. Choose a separate --db; do not use lexical.sqlite.")
    with db:
        db.execute("""CREATE TABLE IF NOT EXISTS embeddings (
            imageID TEXT PRIMARY KEY, file_path TEXT NOT NULL,
            sha256 TEXT NOT NULL, vector BLOB NOT NULL)""")
        db.execute("CREATE TABLE IF NOT EXISTS dense_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        db.execute("CREATE INDEX IF NOT EXISTS embeddings_sha256 ON embeddings(sha256)")
    return db


def get_config(db):
    saved = db.execute("SELECT value FROM dense_metadata WHERE key='embedding_config'").fetchone()
    if saved is None:
        raise ValueError("No embedding configuration. Run: python -m src.dense build")
    return json.loads(saved[0])


def set_config(db, config):
    serialized = json.dumps(config, sort_keys=True)
    saved = db.execute("SELECT value FROM dense_metadata WHERE key='embedding_config'").fetchone()
    if saved and saved[0] != serialized:
        raise ValueError("Embedding settings changed. Use another --db to avoid mixing vectors.")
    if not saved and db.execute("SELECT 1 FROM embeddings LIMIT 1").fetchone():
        raise ValueError("Embedding configuration is missing from a nonempty database.")
    with db:
        db.execute("INSERT OR REPLACE INTO dense_metadata VALUES ('embedding_config', ?)", (serialized,))


def unit_vectors(values, dimension):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != dimension:
        raise ValueError(f"Expected a matrix with {dimension} columns, got {values.shape}")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or not np.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError("Embedding contains nonfinite values or a zero vector")
    return values / norms


def decode_vector(blob, dimension):
    if len(blob) != dimension * 4:
        raise ValueError("Corrupt embedding: wrong vector byte length")
    return unit_vectors(np.frombuffer(blob, dtype="<f4")[None, :], dimension)[0]


def save_vectors(db, rows, vectors, dimension):
    vectors = unit_vectors(vectors, dimension)
    if len(rows) != len(vectors):
        raise ValueError("Embedding batch size does not match its image IDs")
    # Validate the entire batch before committing any row.
    records = [(row["imageID"], row["file_path"], row["sha256"], vector.astype("<f4").tobytes())
               for row, vector in zip(rows, vectors)]
    with db:
        db.executemany("INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?)", records)


def cached_by_hash(db, digest, dimension):
    saved = db.execute("SELECT vector FROM embeddings WHERE sha256=? LIMIT 1", (digest,)).fetchone()
    return decode_vector(saved[0], dimension) if saved else None


def load_vectors(db, rows, dimension, allow_partial=False):
    saved = {record[0]: record[1:] for record in db.execute(
        "SELECT imageID, file_path, sha256, vector FROM embeddings")}
    documents, vectors = [], []
    for row in rows:
        record = saved.get(row["imageID"])
        if record and record[:2] == (row["file_path"], row["sha256"]):
            documents.append(row)
            vectors.append(decode_vector(record[2], dimension))
    missing = len(rows) - len(documents)
    if missing and not allow_partial:
        raise ValueError(f"Missing/stale embeddings: {missing}. Run dense build or use --allow-partial.")
    if not documents:
        raise ValueError("No current image embeddings to search")
    return documents, np.stack(vectors)


def search(documents, vectors, query_vector, top_k=10):
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if vectors.ndim != 2 or len(documents) != len(vectors):
        raise ValueError("Vector matrix does not match its image IDs")
    query = unit_vectors(np.asarray(query_vector)[None, :], vectors.shape[1])[0]
    scores = np.clip(vectors @ query, -1.0, 1.0)
    # Stable ties by imageID; 5,000 vectors do not need an approximate index.
    order = sorted(range(len(documents)), key=lambda i: (-float(scores[i]), documents[i]["imageID"]))
    return [{"imageID": documents[i]["imageID"], "file_path": documents[i]["file_path"],
             "cosine_score": float(scores[i])} for i in order[:top_k]]
