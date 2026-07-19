import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.summarize_pseudo_label_grid import group_miou, read_grid


REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE_A_HEADER = [
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
PHASE_B_HEADER = PHASE_A_HEADER + [
    "threshold_artifact",
    "threshold_max_batches",
]
WEIGHTED_HEADER = PHASE_B_HEADER + ["weighting"]


class PseudoLabelGridSummaryTests(unittest.TestCase):
    def test_read_grid_accepts_phase_a_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            grid = Path(tmpdir) / "grid.tsv"
            grid.write_text(
                "\t".join(PHASE_A_HEADER)
                + "\n"
                + "fixed0p6\tsub\t15-5\toverlap\tfixed\t0.6\t0.7\t0.0\t1.0\t1\t0.0\t0.0\tbase\t1\t32\t32\t8196\t1\t1\tdeeplabv3_resnet101\tauto\n",
                encoding="utf-8",
            )

            rows = read_grid(grid)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["name"], "fixed0p6")
        self.assertEqual(rows[0]["confidence"], "0.6")
        self.assertEqual(rows[0]["threshold_artifact"], "")
        self.assertEqual(rows[0]["threshold_max_batches"], "")

    def test_read_grid_accepts_phase_b_artifact_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            grid = Path(tmpdir) / "grid.tsv"
            grid.write_text(
                "\t".join(PHASE_B_HEADER)
                + "\n"
                + "artifact_q0p1\tsub\t15-5\toverlap\tartifact_class\t0.7\t0.1\t0.0\t1.0\t1\t0.0\t0.0\tbase\t1\t32\t32\t8196\t1\t1\tdeeplabv3_resnet101\tauto\tartifacts/pseudo_label/q0p1.json\t0\n",
                encoding="utf-8",
            )

            rows = read_grid(grid)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["strategy"], "artifact_class")
        self.assertEqual(rows[0]["threshold_artifact"], "artifacts/pseudo_label/q0p1.json")
        self.assertEqual(rows[0]["threshold_max_batches"], "0")
        self.assertEqual(rows[0]["weighting"], "")

    def test_read_grid_accepts_weighting_and_runner_forwards_it(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            grid = Path(tmpdir) / "grid.tsv"
            grid.write_text(
                "\t".join(WEIGHTED_HEADER)
                + "\n"
                + "weighted_conf\tsub\t15-5\toverlap\tfixed\t0.447265625\t0.7\t0.0\t1.0\t1\t0.0\t0.0\tbase\t1\t32\t32\t8196\t1\t1\tdeeplabv3_resnet101\tauto\t\t0\tconfidence\n",
                encoding="utf-8",
            )

            rows = read_grid(grid)
            result = subprocess.run(
                [
                    "bash",
                    "tools/run_pseudo_label_grid.sh",
                    "--grid",
                    str(grid),
                    "--mode",
                    "dry-run",
                ],
                cwd=REPO_ROOT,
                check=False,
                text=True,
                capture_output=True,
            )

        self.assertEqual(rows[0]["weighting"], "confidence")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PSEUDO_LABEL_WEIGHTING=confidence", result.stdout)

    def test_grid_runner_dry_run_passes_artifact_environment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            grid = Path(tmpdir) / "grid.tsv"
            grid.write_text(
                "\t".join(PHASE_B_HEADER)
                + "\n"
                + "artifact_q0p1\tsub\t15-5\toverlap\tartifact_class\t0.7\t0.1\t0.0\t1.0\t1\t0.0\t0.0\tbase\t1\t32\t32\t8196\t1\t1\tdeeplabv3_resnet101\tauto\tartifacts/pseudo_label/q0p1.json\t0\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    "bash",
                    "tools/run_pseudo_label_grid.sh",
                    "--grid",
                    str(grid),
                    "--mode",
                    "dry-run",
                ],
                cwd=REPO_ROOT,
                check=False,
                text=True,
                capture_output=True,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "PSEUDO_LABEL_THRESHOLD_ARTIFACT=artifacts/pseudo_label/q0p1.json",
            result.stdout,
        )
        self.assertIn("PSEUDO_LABEL_THRESHOLD_MAX_BATCHES=0", result.stdout)

    def test_grid_runner_rejects_artifact_class_without_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            grid = Path(tmpdir) / "grid.tsv"
            grid.write_text(
                "\t".join(PHASE_A_HEADER)
                + "\n"
                + "artifact_missing\tsub\t15-5\toverlap\tartifact_class\t0.7\t0.1\t0.0\t1.0\t1\t0.0\t0.0\tbase\t1\t32\t32\t8196\t1\t1\tdeeplabv3_resnet101\tauto\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    "bash",
                    "tools/run_pseudo_label_grid.sh",
                    "--grid",
                    str(grid),
                    "--mode",
                    "dry-run",
                ],
                cwd=REPO_ROOT,
                check=False,
                text=True,
                capture_output=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("artifact_class", result.stderr)
        self.assertIn("threshold_artifact", result.stderr)

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
