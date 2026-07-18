import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from datasets.init_dataset import validate_clean_validation_lists
from datasets.voc import exclude_image_ids


class VocValidationContractTests(unittest.TestCase):
    def test_validation_list_requires_training_exclusion(self):
        opts = SimpleNamespace(validation_list="holdout.txt", train_exclude_list=None)
        with self.assertRaisesRegex(ValueError, "requires --train_exclude_list"):
            validate_clean_validation_lists(opts)

    def test_validation_ids_must_be_excluded_from_training(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            validation_path = Path(tmpdir) / "validation.txt"
            exclusion_path = Path(tmpdir) / "exclusion.txt"
            validation_path.write_text("a\nb\n", encoding="utf-8")
            exclusion_path.write_text("a\n", encoding="utf-8")
            opts = SimpleNamespace(
                validation_list=str(validation_path),
                train_exclude_list=str(exclusion_path),
            )
            with self.assertRaisesRegex(ValueError, "does not cover"):
                validate_clean_validation_lists(opts)

    def test_training_exclusion_removes_only_holdout_ids(self):
        remaining = exclude_image_ids(["img01", "img02", "img03"], ["img02", "img99"])
        self.assertEqual(remaining, ["img01", "img03"])


if __name__ == "__main__":
    unittest.main()
