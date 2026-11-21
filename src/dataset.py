"""Validate Castle5000 images and build a portable CSV manifest.

Run from any directory: python /path/to/project/src/dataset.py
file_path values in the CSV are relative to --data-root, not the CSV directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
import tempfile
import warnings
from collections import Counter
from pathlib import Path

from PIL import Image, ImageFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLLECTION = "castle5000"
IMAGE_PREFIX = "day1_Kitchen_"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
NAME_PATTERN = re.compile(r"(?P<hour>[0-2][0-9])_(?P<frame>[0-9]+)")
FIELDS = (
    "imageID", "file_path", "collection", "batch", "hour", "frame_index",
    "width", "height", "size_bytes", "sha256",
)


class DatasetValidationError(ValueError):
    """The dataset cannot safely be exported as a manifest."""


def scan_images(data_root: Path, expected_count: int | None = 5000) -> list[dict]:
    """Return deterministic rows, or raise if any candidate image is invalid.

    IDs retain the exact filename stem (including leading zeros). Hashes expose
    identical file contents; only duplicate IDs are forbidden, since distinct
    frames can legitimately have identical pixels or file contents.
    """
    data_root = Path(data_root).resolve()
    if not data_root.is_dir():
        raise DatasetValidationError(f"Data directory does not exist: {data_root}")

    paths = sorted(
        (p for p in data_root.rglob("*") if p.is_file()
         and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: p.relative_to(data_root).as_posix(),
    )
    errors = []
    if not paths:
        errors.append("No supported images found (.jpg, .jpeg, .png).")
    if expected_count is not None and len(paths) != expected_count:
        errors.append(f"Expected {expected_count} images, found {len(paths)}.")

    rows = []
    seen_ids: dict[str, str] = {}
    # Never silently accept truncated JPEGs, even if another caller enables it.
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    for path in paths:
        relative_path = path.relative_to(data_root).as_posix()
        match = NAME_PATTERN.fullmatch(path.stem)
        if match is None or int(match["hour"]) > 23:
            errors.append(f"{relative_path}: expected filename HH_frame with hour 00-23.")
            continue
        image_id = IMAGE_PREFIX + path.stem
        if image_id in seen_ids:
            errors.append(
                f"Duplicate imageID {image_id}: {seen_ids[image_id]} and {relative_path}"
            )
            continue
        seen_ids[image_id] = relative_path

        try:
            if path.is_symlink() or not path.resolve().is_relative_to(data_root):
                raise ValueError("image must be a regular file inside data-root")
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(path) as image:
                    image.verify()
                # verify() checks structure; load() also decodes the pixels.
                with Image.open(path) as image:
                    image.load()
                    width, height = image.size
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            size_bytes = path.stat().st_size
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError,
                Image.DecompressionBombWarning) as exc:
            errors.append(f"{relative_path}: cannot validate image: {exc}")
            continue

        rows.append({
            "imageID": image_id,
            "file_path": relative_path,
            "collection": COLLECTION,
            "batch": path.relative_to(data_root).parts[0]
            if len(path.relative_to(data_root).parts) > 1 else "",
            "hour": int(match["hour"]),
            "frame_index": int(match["frame"]),
            "width": width,
            "height": height,
            "size_bytes": size_bytes,
            "sha256": digest,
        })

    if errors:
        details = "\n".join(f"- {error}" for error in errors[:20])
        if len(errors) > 20:
            details += f"\n- ... {len(errors) - 20} more errors."
        raise DatasetValidationError(f"Validation failed ({len(errors)} errors):\n{details}")
    return rows


def write_manifest(rows: list[dict], output: Path) -> None:
    """Atomically replace the CSV only after the complete scan succeeds."""
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=output.parent,
            prefix=output.name + ".", suffix=".tmp", delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(output)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "manifest.csv")
    parser.add_argument(
        "--expected-count", type=int, default=5000,
        help="Required number of images (default: 5000); 0 disables the count check.",
    )
    args = parser.parse_args(argv)
    if args.expected_count < 0:
        parser.error("--expected-count must be nonnegative")
    try:
        rows = scan_images(args.data_root, args.expected_count or None)
        write_manifest(rows, args.output)
    except (DatasetValidationError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Manifest: {args.output.resolve()}")
    print(f"Data root: {args.data_root.resolve()}")
    print(f"Validated images: {len(rows)}")
    for batch, count in sorted(Counter(row["batch"] for row in rows).items()):
        print(f"  {batch or '(root)'}: {count}")
    repeated = len(rows) - len({row["sha256"] for row in rows})
    print(f"Repeated file contents beyond first occurrence: {repeated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
