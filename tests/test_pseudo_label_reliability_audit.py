import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tools.audit_pseudo_label_reliability import (
    RawMaskAuditDataset,
    build_audit_manifest,
    compute_candidate_outcomes,
    compute_reliability_bins,
    expected_calibration_error,
    signal_gate,
    verify_checkpoint_sha256,
    write_audit_outputs,
)
from utils.pseudo_label import PseudoLabelCandidates


class _SyntheticVocDataset:
    def __init__(self, mask_path):
        self.masks = [str(mask_path)]
        self.ordering_map = np.full(256, 255, dtype=np.uint8)
        self.ordering_map[0] = 0
        self.ordering_map[3] = 1
        self.ordering_map[7] = 2

    def __getitem__(self, index):
        image = torch.zeros(3, 2, 3)
        incremental = torch.tensor([[0, 1, 0], [2, 0, 255]])
        return image, incremental, "synthetic"

    def __len__(self):
        return 1


class PseudoLabelReliabilityAuditTests(unittest.TestCase):
    def test_formal_runner_dry_run_lists_both_settings_without_writes(self):
        runner = (
            Path(__file__).resolve().parents[1]
            / "tools/run_pseudo_label_reliability_w0_20260719.sh"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "audit"
            env = dict(os.environ)
            env.update(
                {
                    "DRY_RUN": "1",
                    "OUTPUT_ROOT": str(output_root),
                    "PYTHON": sys.executable,
                }
            )
            completed = subprocess.run(
                ["bash", str(runner)],
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--setting overlap", completed.stdout)
        self.assertIn("--setting disjoint", completed.stdout)
        self.assertIn("--max-samples 0", completed.stdout)
        self.assertFalse(output_root.exists())

    def test_raw_mask_wrapper_keeps_pixel_alignment_and_voc_ordering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mask_path = Path(tmpdir) / "mask.png"
            Image.fromarray(
                np.array([[0, 3, 7], [7, 3, 255]], dtype=np.uint8)
            ).save(mask_path)
            dataset = RawMaskAuditDataset(_SyntheticVocDataset(mask_path))

            image, incremental, raw_ordered, file_name = dataset[0]

        self.assertEqual(tuple(image.shape), (3, 2, 3))
        self.assertEqual(file_name, "synthetic")
        torch.testing.assert_close(
            raw_ordered,
            torch.tensor([[0, 1, 2], [2, 1, 255]], dtype=torch.long),
        )
        self.assertEqual(tuple(raw_ordered.shape), tuple(incremental.shape))

    def test_raw_mask_wrapper_rejects_spatial_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mask_path = Path(tmpdir) / "mask.png"
            Image.fromarray(np.zeros((1, 1), dtype=np.uint8)).save(mask_path)
            dataset = RawMaskAuditDataset(_SyntheticVocDataset(mask_path))

            with self.assertRaisesRegex(ValueError, "spatial"):
                dataset[0]

    def test_exact_class_and_error_decomposition_are_exclusive_and_conserved(self):
        incremental = torch.zeros(1, 2, 3, dtype=torch.long)
        raw = torch.tensor([[[1, 0, 2], [3, 255, 1]]])
        candidates = PseudoLabelCandidates(
            scores=torch.tensor([[[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]]]),
            labels=torch.tensor([[[1, 1, 1], [1, 1, 2]]]),
            margins=torch.full((1, 2, 3), 0.2),
            mask=torch.ones(1, 2, 3, dtype=torch.bool),
        )

        outcomes = compute_candidate_outcomes(
            candidates,
            incremental,
            raw,
            old_class_ids=[1, 2],
        )

        self.assertEqual(int(outcomes.correct.sum()), 1)
        self.assertEqual(int(outcomes.false_old_on_background.sum()), 1)
        self.assertEqual(int(outcomes.wrong_old_class.sum()), 2)
        self.assertEqual(int(outcomes.false_old_on_current_or_future.sum()), 1)
        self.assertEqual(int(outcomes.ignored_raw.sum()), 1)
        classified = (
            outcomes.correct
            | outcomes.false_old_on_background
            | outcomes.wrong_old_class
            | outcomes.false_old_on_current_or_future
            | outcomes.ignored_raw
        )
        self.assertTrue(torch.equal(classified, candidates.mask))
        pairwise = torch.stack(
            [
                outcomes.correct,
                outcomes.false_old_on_background,
                outcomes.wrong_old_class,
                outcomes.false_old_on_current_or_future,
                outcomes.ignored_raw,
            ]
        ).sum(dim=0)
        self.assertEqual(int(pairwise.max()), 1)

    def test_hidden_old_recall_denominator_uses_incremental_background(self):
        incremental = torch.tensor([[[0, 1, 0], [0, 255, 0]]])
        raw = torch.tensor([[[1, 1, 2], [3, 1, 0]]])
        candidates = PseudoLabelCandidates(
            scores=torch.ones(1, 2, 3),
            labels=torch.ones(1, 2, 3, dtype=torch.long),
            margins=torch.ones(1, 2, 3),
            mask=torch.ones(1, 2, 3, dtype=torch.bool),
        )

        outcomes = compute_candidate_outcomes(
            candidates,
            incremental,
            raw,
            old_class_ids=[1, 2],
        )

        expected = torch.tensor([[[True, False, True], [False, False, False]]])
        self.assertTrue(torch.equal(outcomes.hidden_old, expected))

    def test_equal_width_bins_include_zero_and_one(self):
        bins = compute_reliability_bins(
            np.array([0.0, 0.1, 0.5, 0.9, 1.0]),
            np.array([1, 0, 1, 0, 1], dtype=bool),
            mode="equal_width",
            num_bins=10,
        )

        self.assertEqual(sum(row["count"] for row in bins), 5)
        self.assertEqual(bins[0]["lower"], 0.0)
        self.assertEqual(bins[-1]["upper"], 1.0)
        self.assertEqual(bins[0]["count"], 2)
        self.assertEqual(bins[-1]["count"], 1)

    def test_equal_count_bins_conserve_candidate_count(self):
        scores = np.linspace(0.0, 1.0, 23)
        bins = compute_reliability_bins(
            scores,
            scores > 0.4,
            mode="equal_count",
            num_bins=10,
        )

        self.assertEqual(sum(row["count"] for row in bins), 23)
        self.assertEqual(sum(row["count"] > 0 for row in bins), 10)

    def test_ece_matches_hand_calculation(self):
        bins = [
            {"count": 2, "mean_score": 0.25, "precision": 0.5},
            {"count": 2, "mean_score": 0.75, "precision": 1.0},
        ]

        self.assertAlmostEqual(expected_calibration_error(bins), 0.25)

    def test_empty_bins_do_not_produce_nan(self):
        bins = compute_reliability_bins(
            np.array([], dtype=np.float64),
            np.array([], dtype=bool),
            mode="equal_width",
            num_bins=10,
        )

        self.assertEqual(len(bins), 10)
        self.assertTrue(all(row["count"] == 0 for row in bins))
        self.assertTrue(all(row["precision"] is None for row in bins))
        self.assertEqual(expected_calibration_error(bins), 0.0)

    def test_signal_gate_enforces_both_settings(self):
        passing = {
            "nonempty_equal_count_bins": 10,
            "spearman_rho": 0.8,
            "top_bottom_precision_gap": 0.1,
            "extreme_mass_ratio": 0.2,
        }
        failing = dict(passing, spearman_rho=0.5)

        result = signal_gate({"overlap": passing, "disjoint": passing})
        failed = signal_gate({"overlap": passing, "disjoint": failing})

        self.assertTrue(result["passed"])
        self.assertFalse(failed["passed"])
        self.assertIn("spearman_rho", failed["settings"]["disjoint"]["failed"])

    def test_manifest_is_explicitly_audit_only(self):
        manifest = build_audit_manifest(
            dataset="voc",
            task="15-5",
            step=1,
            setting="overlap",
            file_count=42,
            teacher_checkpoint="/tmp/teacher.pth",
            teacher_sha256="abc",
            git_commit="deadbeef",
            git_dirty=False,
            loss_type="bce_loss",
            expected_teacher_classes=16,
            old_class_ids=list(range(1, 16)),
            matched_global_threshold=0.4,
            max_samples=1,
            seed=1,
        )

        self.assertTrue(manifest["audit_only"])
        self.assertEqual(manifest["transform"]["geometry"], "original")

    def test_checkpoint_sha_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "teacher.pth"
            checkpoint.write_bytes(b"teacher")

            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                verify_checkpoint_sha256(checkpoint, "0" * 64)

    def test_output_writer_creates_five_formal_artifacts(self):
        manifest = {"audit_only": True, "setting": "overlap"}
        summary = {
            "setting": "overlap",
            "signals": {
                "confidence": {
                    "candidate_count": 2,
                    "equal_width_bins": [
                        {
                            "mode": "equal_width",
                            "bin_index": 0,
                            "lower": 0.0,
                            "upper": 1.0,
                            "count": 2,
                            "mean_score": 0.5,
                            "precision": 0.5,
                        }
                    ],
                    "equal_count_bins": [],
                }
            },
            "per_class": [{"class_id": 1, "candidate_count": 2}],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_audit_outputs(tmpdir, manifest, summary)

            expected = {
                "audit_manifest.json",
                "reliability_summary.json",
                "reliability_bins.csv",
                "per_class.csv",
                "report.md",
            }
            self.assertEqual({Path(path).name for path in paths}, expected)
            loaded = json.loads(
                (Path(tmpdir) / "audit_manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(loaded["audit_only"])
            with (Path(tmpdir) / "reliability_bins.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 1)


if __name__ == "__main__":
    unittest.main()
