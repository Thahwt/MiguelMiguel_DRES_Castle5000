"""Generate resumable English captions locally with Qwen2.5-VL."""

import argparse
import hashlib
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

from src.lexical_store import (
    DEFAULT_DB, DEFAULT_MANIFEST, PROJECT_ROOT, cached_by_hash, connect,
    find_caption, read_manifest, save_caption, set_config,
)

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
PROMPT = """Describe this camera image in English for image retrieval.
Write only 2-3 SHORT sentences, at most 50 words. State the number of visible
people, clothing colors, and the visible actions and spatial relationships of
one or two prominent people and nearby objects. Write it SHORT and SIMPLE, DO NOT describe the surroundings in detail. Do not guess identities, intentions,
or events before or after this image. Do not infer opening versus closing from
a static pose. Omit uncertain details. Output only the caption, with no introduction
or commentary."""


class QwenCaptioner:
    def __init__(self, model, revision, device, max_pixels, max_new_tokens, cache_dir):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA unavailable. Install CUDA-enabled PyTorch or use --device cpu.")
        if device == "cuda" and not torch.cuda.is_bf16_supported():
            raise ValueError("GPU does not support BF16. Use --device cpu for FP32 inference.")
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.device = device
        self.max_new_tokens = max_new_tokens
        print(f"Loading {model} on {device} ({dtype})", flush=True)
        self.processor = AutoProcessor.from_pretrained(
            model, revision=revision, min_pixels=256 * 28 * 28,
            max_pixels=max_pixels, cache_dir=str(cache_dir),
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model, revision=revision, cache_dir=str(cache_dir),
            dtype=dtype,
        ).to(device).eval()

    def __call__(self, path):
        from PIL import Image

        with Image.open(path) as source:
            image = source.convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": PROMPT},
        ]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)
        return self.generate_text(inputs, image_path=path)

    def generate_text(self, inputs, image_path=None, error_dir=None):
        """Retry once with more tokens; never save an unfinished caption."""
        eos_ids = self.model.generation_config.eos_token_id
        eos_ids = [eos_ids] if isinstance(eos_ids, int) else (eos_ids or [])
        attempts = []
        for budget in (self.max_new_tokens, self.max_new_tokens * 2):
            with self.torch.inference_mode():
                outputs = self.model.generate(**inputs, max_new_tokens=budget, do_sample=False)
            generated = outputs[:, inputs["input_ids"].shape[1]:]
            if generated.shape[1] < budget or generated[0, -1].item() in eos_ids:
                return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
            attempts.append({
                "budget": budget,
                "generated_tokens": generated.shape[1],
                "token_ids": generated[0].tolist(),
                "raw_text": self.processor.batch_decode(generated, skip_special_tokens=False)[0],
            })
            if budget == self.max_new_tokens:
                print(f"Caption reached {budget} tokens; retrying with {budget * 2}.", flush=True)
        error_dir = Path(error_dir) if error_dir is not None else PROJECT_ROOT / "artifacts/caption_errors"
        error_dir.mkdir(parents=True, exist_ok=True)
        # A unique file per failure preserves previous runs for comparison.
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=error_dir,
            prefix="caption_", suffix=".json", delete=False,
        ) as stream:
            json.dump({
                "image_path": str(Path(image_path).resolve()) if image_path is not None else None,
                "prompt": PROMPT,
                "eos_token_ids": eos_ids,
                "attempts": attempts,
            }, stream, ensure_ascii=False, indent=2)
            diagnostic_path = Path(stream.name)
        raise ValueError(
            f"Caption still unfinished after {budget} tokens. "
            f"Inspect raw output: {diagnostic_path}. "
            "The unfinished caption was not saved to the database; resume with the same --db."
        )


def generate_captions(rows, data_root, db, config, factory, limit=0, allow_prompt_change=False):
    """factory is lazy: a completed run never loads the VLM. Commit each image."""
    set_config(db, config, allow_prompt_change=allow_prompt_change)
    history = db.execute("SELECT value FROM metadata WHERE key='caption_config_history'").fetchone()
    # On a mixed-prompt database, an old caption must not be copied to a new ID.
    reuse_by_hash = not any(
        old.get("prompt") != config.get("prompt") for old in (json.loads(history[0]) if history else [])
    )
    data_root = Path(data_root).resolve()
    captioner = None
    completed = 0
    for row in rows:
        if find_caption(db, row) is not None:
            continue
        if limit and completed >= limit:
            break
        path = (data_root / row["file_path"]).resolve()
        if not path.is_relative_to(data_root):
            raise ValueError(f"Path outside data-root: {row['file_path']}")
        # Check image bytes before trusting the manifest or reusing a caption.
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != row["sha256"]:
            raise ValueError(f"Image changed; regenerate manifest: {row['imageID']}")
        caption = cached_by_hash(db, digest) if reuse_by_hash else None
        if caption is None:
            if captioner is None:
                captioner = factory()
            caption = captioner(path)
        save_caption(db, row, caption)
        completed += 1
        print(f"[{completed}] {row['imageID']}", flush=True)
    return completed


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--model", default=MODEL, help="Qwen2.5-VL-compatible checkpoint or local path")
    parser.add_argument("--revision", default="main", help="Pin a model commit for reproducibility")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "artifacts/hf_cache")
    parser.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    parser.add_argument("--max-new-tokens", type=int, default=768,
                        help="Initial token budget; retry once with twice this value if truncated")
    parser.add_argument("--limit", type=int, default=0, help="Max NEW captions this run; 0 = all remaining")
    parser.add_argument("--allow-prompt-change", action="store_true",
                        help="Accept a new prompt in the same database; keep completed captions")
    args = parser.parse_args(argv)
    if args.limit < 0 or args.max_new_tokens <= 0 or args.max_pixels < 256 * 28 * 28:
        parser.error("limit >= 0, max-new-tokens > 0, max-pixels >= 200704 required")
    config = {"model": args.model, "revision": args.revision, "prompt": PROMPT,
              "max_pixels": args.max_pixels, "max_new_tokens": args.max_new_tokens,
              "language": "en", "do_sample": False}
    db = None
    try:
        rows = read_manifest(args.manifest)
        db = connect(args.db)
        count = generate_captions(
            rows, args.data_root, db, config,
            lambda: QwenCaptioner(args.model, args.revision, args.device,
                                  args.max_pixels, args.max_new_tokens, args.cache_dir),
            args.limit,
            allow_prompt_change=args.allow_prompt_change,
        )
        print(f"Saved {count} new captions to {args.db}")
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        print(f"Captioning failed: {exc}\nCompleted captions are saved; rerun to resume.", file=sys.stderr)
        return 1
    finally:
        if db is not None:
            db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
