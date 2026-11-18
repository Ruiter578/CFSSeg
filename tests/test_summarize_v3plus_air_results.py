import json
import tempfile
import unittest
from pathlib import Path

from tools.summarize_v3plus_air_results import build_table


class SummarizeAirResultsTests(unittest.TestCase):
    def test_complete_and_pending_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_dir = root / "run_decoder" / "voc" / "15-5" / "sequential" / "step1"
            result_dir.mkdir(parents=True)
            result = {
                "Mean IoU": 0.69,
                "0 to 15 mIoU": 0.78,
                "16 to 20 mIoU": 0.42,
                "Class IoU": {"16": 0.2, "19": 0.7, "20": 0.3},
            }
            (result_dir / "test_results_1.json").write_text(json.dumps(result), encoding="utf-8")

            table = build_table(root, "run", ("decoder", "aspp"))

        self.assertIn("| decoder | complete | 0.6900 | 0.7800 | 0.4200", table)
        self.assertIn("| aspp | pending |", table)


if __name__ == "__main__":
    unittest.main()
