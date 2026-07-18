import tempfile
import unittest
import json
from pathlib import Path

from tools.create_voc_tuning_split import (
    build_split,
    parse_class_rows,
    write_split,
)
from tools.create_voc_tuning_split import sha256_file


class VocTuningSplitTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            ("img01", (0, 1)),
            ("img02", (0, 2)),
            ("img03", (0, 3)),
            ("img04", (0, 4)),
            ("img05", (0, 5)),
            ("img06", (0, 1, 2)),
            ("img07", (0, 3, 4)),
            ("img08", (0, 5, 6)),
            ("img09", (0, 6)),
            ("img10", (0, 1, 6)),
        ]

    def test_parser_reads_image_id_and_zero_based_class_tokens(self):
        parsed = parse_class_rows(["a 0 2", "b 3"])
        self.assertEqual(parsed, [("a", (0, 2)), ("b", (3,))])

    def test_split_is_deterministic_disjoint_and_has_exact_size(self):
        first = build_split(self.rows, fraction=0.3, seed=7)
        second = build_split(list(reversed(self.rows)), fraction=0.3, seed=7)

        self.assertEqual(first.holdout_ids, second.holdout_ids)
        self.assertEqual(len(first.holdout_ids), 3)
        self.assertFalse(set(first.holdout_ids) & set(first.train_ids))
        self.assertEqual(set(first.holdout_ids) | set(first.train_ids), {row[0] for row in self.rows})

    def test_split_covers_each_observed_foreground_class_when_capacity_allows(self):
        split = build_split(self.rows, fraction=0.7, seed=11)
        heldout_rows = {image_id: classes for image_id, classes in self.rows if image_id in split.holdout_ids}
        heldout_classes = {cls for classes in heldout_rows.values() for cls in classes}

        self.assertTrue({1, 2, 3, 4, 5, 6}.issubset(heldout_classes))

    def test_tight_split_repairs_greedy_choice_for_maximum_class_coverage(self):
        rows = [
            ("a", (1, 2, 4, 5)),
            ("b", (1, 2, 3)),
            ("c", (4, 5, 6)),
        ]
        split = build_split(rows, fraction=2 / 3, seed=5)
        heldout_rows = {image_id: classes for image_id, classes in rows if image_id in split.holdout_ids}
        heldout_classes = {cls for classes in heldout_rows.values() for cls in classes}

        self.assertEqual(heldout_classes, {1, 2, 3, 4, 5, 6})

    def test_write_split_creates_plain_id_list_and_metadata(self):
        split = build_split(self.rows, fraction=0.3, seed=7)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "holdout.txt"
            metadata_path = Path(tmpdir) / "holdout.json"
            write_split(split, output_path, metadata_path, source_path="fixture.txt")

            self.assertEqual(output_path.read_text(encoding="utf-8").splitlines(), list(split.holdout_ids))
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["seed"], 7)
            self.assertEqual(metadata["source_path"], "fixture.txt")
            self.assertEqual(metadata["holdout_sha256"], sha256_file(output_path))

if __name__ == "__main__":
    unittest.main()
