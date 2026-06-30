import tempfile
import unittest
from pathlib import Path

from tools.summarize_pseudo_label_grid import group_miou, read_grid


class PseudoLabelGridSummaryTests(unittest.TestCase):
    def test_read_grid_accepts_phase_a_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            grid = Path(tmpdir) / "grid.tsv"
            grid.write_text(
                "\t".join(
                    [
                        "name",
                        "subpath",
                        "task",
                        "setting",
                        "strategy",
                        "confidence",
                        "quantile",
                        "min_conf",
                        "max_conf",
                        "min_pixels",
                        "shrinkage",
                        "margin_min",
                        "base_subpath",
                        "skip_step0",
                        "batch_size",
                        "step0_batch_size",
                        "buffer",
                        "gamma",
                        "random_seed",
                        "model",
                        "air_feature_source",
                    ]
                )
                + "\n"
                + "fixed0p6\tsub\t15-5\toverlap\tfixed\t0.6\t0.7\t0.0\t1.0\t1\t0.0\t0.0\tbase\t1\t32\t32\t8196\t1\t1\tdeeplabv3_resnet101\tauto\n",
                encoding="utf-8",
            )

            rows = read_grid(grid)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["name"], "fixed0p6")
        self.assertEqual(rows[0]["confidence"], "0.6")

    def test_group_miou_finds_old_and_new_keys(self):
        old, new = group_miou(
            {
                "Mean IoU": 0.7,
                "0 to 15 mIoU": 0.8,
                "16 to 20 mIoU": 0.4,
            }
        )

        self.assertEqual(old, 0.8)
        self.assertEqual(new, 0.4)


if __name__ == "__main__":
    unittest.main()
