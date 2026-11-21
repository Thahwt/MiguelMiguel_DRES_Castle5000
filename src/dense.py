"""Build SigLIP 2 Large image embeddings and retrieve images with text queries."""

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

from src.dense_store import (
    DEFAULT_DENSE_DB, cached_by_hash, connect, get_config, load_vectors,
    save_vectors, search, set_config,
)
from src.lexical_store import DEFAULT_MANIFEST, PROJECT_ROOT, read_manifest

MODEL = "google/siglip2-large-patch16-384"
REVISION = "1b426889ea62b5a72bf9839009a1b184bfc9c178"
DIMENSION = 1024


def resolve_device(device):
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable. Use CUDA-enabled PyTorch or --device cpu.")
    dtype = "float32" if device == "cpu" else (
        "bfloat16" if torch.cuda.is_bf16_supported() else "float16")
    return device, dtype


def embedding_config(revision, dtype):
    return {"model": MODEL, "revision": revision, "dimension": DIMENSION,
            "storage_dtype": "<f4", "normalize": "l2", "image_mode": "RGB",
            "processor_use_fast": True, "text_lowercase": True,
            "text_max_length": 64, "text_padding": "max_length", "text_truncation": True,
            "inference_dtype": dtype, "transformers_version": version("transformers"),
            "schema_version": 1}


class Siglip2Encoder:
    def __init__(self, revision, device, cache_dir, local_files_only=False):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.device, dtype = resolve_device(device)
        self.dtype = getattr(torch, dtype)
        print(f"Loading {MODEL}@{revision} on {self.device} ({dtype})", file=sys.stderr, flush=True)
        self.processor = AutoProcessor.from_pretrained(
            MODEL, revision=revision, cache_dir=str(cache_dir), use_fast=True,
            local_files_only=local_files_only,
        )
        # FixRes SigLIP 2 checkpoints use the SiglipModel architecture. AutoModel
        # selects it from config.json; the NaFlex Siglip2Model is a different API.
        self.model = AutoModel.from_pretrained(
            MODEL, revision=revision, cache_dir=str(cache_dir), dtype=self.dtype,
            attn_implementation="sdpa", local_files_only=local_files_only,
        ).to(self.device).eval()

    def _features(self, output):
        # Transformers 4 returns a Tensor; Transformers 5 returns a ModelOutput.
        features = output if isinstance(output, self.torch.Tensor) else output.pooler_output
        return features.float().cpu().numpy()

    def image_features(self, paths):
        from PIL import Image

        images = []
        try:
            for path in paths:
                with Image.open(path) as source:
                    images.append(source.convert("RGB"))
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = inputs.to(device=self.device, dtype=self.dtype)
            with self.torch.inference_mode():
                return self._features(self.model.get_image_features(**inputs))
        finally:
            for image in images:
                image.close()

    def text_features(self, texts):
        texts = [text.strip().lower() for text in texts]
        if not texts or any(not text for text in texts):
            raise ValueError("Text query must not be empty")
        inputs = self.processor(text=texts, padding="max_length", max_length=64,
                                truncation=True, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            return self._features(self.model.get_text_features(**inputs))


def generate_embeddings(rows, data_root, db, config, factory, batch_size=8, limit=0):
    """Check bytes, lazily load the encoder and commit each completed batch."""
    if batch_size < 1 or limit < 0:
        raise ValueError("batch_size must be positive and limit nonnegative")
    set_config(db, config)
    dimension = config["dimension"]
    data_root = Path(data_root).resolve()
    saved = {r[0]: r[1:] for r in db.execute("SELECT imageID, file_path, sha256 FROM embeddings")}
    pending = []
    print(f"Validating {len(rows)} images against manifest SHA-256...", file=sys.stderr, flush=True)
    for row in rows:
        relative = Path(row["file_path"])
        path = (data_root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(data_root):
            raise ValueError(f"Path outside data-root: {row['file_path']}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != row["sha256"]:
            raise ValueError(f"Image changed; regenerate manifest: {row['imageID']}")
        if saved.get(row["imageID"]) != (row["file_path"], digest):
            pending.append((row, path))
    if limit:
        pending = pending[:limit]
    encoder = None
    completed = 0
    started = time.monotonic()
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset:offset + batch_size]
        by_hash, to_encode = {}, {}
        for row, path in batch:
            digest = row["sha256"]
            if digest not in by_hash:
                vector = cached_by_hash(db, digest, dimension)
                if vector is not None:
                    by_hash[digest] = vector
                else:
                    to_encode[digest] = path
        if to_encode:
            if encoder is None:
                encoder = factory()
            features = np.asarray(encoder.image_features(list(to_encode.values())))
            if features.shape != (len(to_encode), dimension):
                raise ValueError(f"Unexpected encoder feature shape: {features.shape}")
            by_hash.update(zip(to_encode, features))
        save_vectors(db, [row for row, _ in batch],
                     [by_hash[row["sha256"]] for row, _ in batch], dimension)
        completed += len(batch)
        print(f"Embedded {completed}/{len(pending)} new images; "
              f"{time.monotonic() - started:.1f}s", file=sys.stderr, flush=True)
    return completed


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="Create image vectors; rerun to resume")
    build.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    build.add_argument("--revision", default=REVISION, help="Full Hugging Face model commit hash")
    build.add_argument("--batch-size", type=int, default=8)
    build.add_argument("--limit", type=int, default=0, help="Max NEW embeddings; 0 = all remaining")
    query = sub.add_parser("search", help="Text-to-image cosine retrieval")
    query.add_argument("query")
    query.add_argument("--top-k", type=int, default=10)
    query.add_argument("--allow-partial", action="store_true")
    for child in (build, query):
        child.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
        child.add_argument("--db", type=Path, default=DEFAULT_DENSE_DB)
        child.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
        child.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "artifacts/hf_cache")
        child.add_argument("--local-files-only", action="store_true", help="Use downloaded weights without network access")
    args = parser.parse_args(argv)
    if args.command == "build":
        if args.batch_size < 1 or args.limit < 0:
            parser.error("batch-size > 0 and limit >= 0 required")
        if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
            parser.error("--revision must be a full 40-character commit hash, not a mutable branch")
    elif args.top_k < 1 or not args.query.strip():
        parser.error("top-k > 0 and a nonempty query required")
    db = None
    try:
        rows = read_manifest(args.manifest)
        device, dtype = resolve_device(args.device)
        db = connect(args.db, readonly=args.command == "search")
        if args.command == "build":
            config = embedding_config(args.revision, dtype)
            count = generate_embeddings(
                rows, args.data_root, db, config,
                lambda: Siglip2Encoder(args.revision, device, args.cache_dir, args.local_files_only),
                args.batch_size, args.limit,
            )
            matched = db.execute("SELECT imageID, file_path, sha256 FROM embeddings").fetchall()
            present = set(matched)
            ready = sum((r["imageID"], r["file_path"], r["sha256"]) in present for r in rows)
            print(f"Saved {count} new embeddings; current: {ready}/{len(rows)}; db: {args.db}")
        else:
            config = get_config(db)
            expected = embedding_config(config["revision"], config["inference_dtype"])
            if config != expected:
                raise ValueError("Embedding configuration/library differs; use the original environment.")
            documents, vectors = load_vectors(db, rows, config["dimension"], args.allow_partial)
            encoder = Siglip2Encoder(config["revision"], device, args.cache_dir, args.local_files_only)
            query_vector = encoder.text_features([args.query])[0]
            print(json.dumps(search(documents, vectors, query_vector, args.top_k),
                             ensure_ascii=False, indent=2))
    except (OSError, ValueError, RuntimeError, ImportError, sqlite3.Error) as exc:
        print(f"Dense pipeline failed: {exc}", file=sys.stderr)
        if args.command == "build":
            print("Completed batches are saved; rerun the same command to resume. "
                  "For CUDA OOM, reduce --batch-size.", file=sys.stderr)
        return 1
    finally:
        if db is not None:
            db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
