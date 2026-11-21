"""Focused checks for dataset integrity and portable DRES mappings."""

import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.dataset import DatasetValidationError, main, scan_images, write_manifest


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.data.mkdir()

    def add_image(self, relative):
        path = self.data / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 12), "red").save(path)
        return path

    def test_nested_paths_ids_and_csv_round_trip(self):
        self.add_image("Batch2/Batch2/08_001.jpg")
        self.add_image("Batch1/10_1528.jpg")
        rows = scan_images(self.data, expected_count=2)
        self.assertEqual([r["imageID"] for r in rows],
                         ["day1_Kitchen_10_1528", "day1_Kitchen_08_001"])
        self.assertEqual(rows[1]["file_path"], "Batch2/Batch2/08_001.jpg")
        self.assertEqual((rows[1]["hour"], rows[1]["frame_index"]), (8, 1))
        self.assertEqual((rows[1]["width"], rows[1]["height"]), (16, 12))
        self.assertEqual(rows, scan_images(self.data, expected_count=2))
        output = self.root / "artifacts" / "manifest.csv"
        write_manifest(rows, output)
        with output.open(encoding="utf-8", newline="") as stream:
            saved = list(csv.DictReader(stream))
        self.assertEqual(len(saved), 2)
        for row in saved:
            self.assertTrue((self.data / row["file_path"]).is_file())

    def test_duplicate_id_across_batches_is_rejected(self):
        self.add_image("Batch1/08_102.jpg")
        self.add_image("Batch2/08_102.jpg")
        with self.assertRaisesRegex(DatasetValidationError, "Duplicate imageID"):
            scan_images(self.data, 2)

    def test_invalid_name_and_corrupt_image_are_reported_together(self):
        self.add_image("Batch1/24_1.jpg")
        corrupt = self.data / "Batch1/08_2.jpg"
        corrupt.write_bytes(b"not an image")
        with self.assertRaises(DatasetValidationError) as caught:
            scan_images(self.data, 2)
        self.assertIn("expected filename", str(caught.exception))
        self.assertIn("cannot validate image", str(caught.exception))

    def test_failed_scan_preserves_existing_manifest(self):
        self.add_image("Batch1/08_1.jpg")
        output = self.root / "manifest.csv"
        output.write_text("previous manifest", encoding="utf-8")
        result = main(["--data-root", str(self.data), "--output", str(output),
                       "--expected-count", "2"])
        self.assertEqual(result, 1)
        self.assertEqual(output.read_text(encoding="utf-8"), "previous manifest")

    def test_empty_or_missing_directory_is_rejected(self):
        for root in (self.data, self.root / "missing"):
            with self.subTest(root=root), self.assertRaises(DatasetValidationError):
                scan_images(root, expected_count=None)


if __name__ == "__main__":
    unittest.main()
