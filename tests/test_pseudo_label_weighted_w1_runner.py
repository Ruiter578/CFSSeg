import fcntl
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.summarize_pseudo_label_weighted_w1 import (
    _weight_extreme_mass_ratio,
    evaluate_weighting_gate,
    load_and_verify_baselines,
    read_and_validate_grid,
    summarize_candidate_row,
    verify_step0_checkpoints,
    verify_output_paths_absent,
    write_outputs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
GRID = REPO_ROOT / "configs/pseudo_label_weighted_w1_20260719.tsv"
BASELINES = (
    REPO_ROOT / "configs/pseudo_label_weighted_w1_baselines_20260719.json"
)
RUNNER = REPO_ROOT / "tools/run_pseudo_label_weighted_w1_20260719.sh"
ADAPTIVE_RUNNER = REPO_ROOT / "tools/run_adaptive_pseudo_label.sh"


class PseudoLabelWeightedW1Tests(unittest.TestCase):
    def test_summarizer_cli_is_runnable_from_repo_root(self):
        result = subprocess.run(
            [
                "/home/linyichen/miniconda3/envs/segacil/bin/python",
                "tools/summarize_pseudo_label_weighted_w1.py",
                "--help",
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--baselines", result.stdout)

    def test_incomplete_summary_is_written_as_review_instead_of_crashing(self):
        summary = {
            "schema_version": 1,
            "rows": [
                {
                    "name": "missing",
                    "setting": "overlap",
                    "weighting": "confidence",
                    "status": "missing_output",
                }
            ],
            "gates": {},
            "recommendation": "review",
            "preferred_weighting": None,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_outputs(summary, Path(tmpdir) / "summary")
            markdown = paths[0].read_text(encoding="utf-8")

        self.assertIn("missing_output", markdown)
        self.assertIn("recommendation: `review`", markdown)

    def test_grid_has_exact_four_pre_registered_rows(self):
        rows = read_and_validate_grid(GRID)

        self.assertEqual(len(rows), 4)
        self.assertEqual(
            {(row["setting"], row["weighting"]) for row in rows},
            {
                ("overlap", "confidence"),
                ("overlap", "confidence_margin"),
                ("disjoint", "confidence"),
                ("disjoint", "confidence_margin"),
            },
        )
        self.assertEqual({row["strategy"] for row in rows}, {"fixed"})
        self.assertEqual({row["random_seed"] for row in rows}, {"1"})
        self.assertEqual({row["batch_size"] for row in rows}, {"32"})
        self.assertEqual({row["quantile"] for row in rows}, {"0.01"})
        self.assertEqual(len({row["subpath"] for row in rows}), 4)
        thresholds = {
            row["setting"]: row["confidence"]
            for row in rows
        }
        self.assertEqual(thresholds["overlap"], "0.447265625")
        self.assertEqual(thresholds["disjoint"], "0.029296875")

    def test_baseline_registry_verifies_real_json_hashes_and_metrics(self):
        baselines = load_and_verify_baselines(BASELINES, repo_root=REPO_ROOT)

        self.assertAlmostEqual(
            baselines["overlap"]["metrics"]["all_miou"],
            0.7080382155336873,
        )
        self.assertAlmostEqual(
            baselines["disjoint"]["metrics"]["all_miou"],
            0.6948731452045601,
        )
        self.assertEqual(
            baselines["overlap"]["teacher_sha256"],
            "6505c26b567cefbb9eb099af174961cbd4596d139b30a7f92f952ad494a9a913",
        )
        self.assertTrue(
            baselines["overlap"]["w0_direction"]["confidence"]["passed"]
        )
        self.assertTrue(
            baselines["disjoint"]["w0_direction"]["confidence_margin"]["passed"]
        )

    def test_real_step0_paths_and_hashes_match_each_grid_row(self):
        rows = read_and_validate_grid(GRID)
        baselines = load_and_verify_baselines(BASELINES, repo_root=REPO_ROOT)

        verified = verify_step0_checkpoints(
            rows,
            repo_root=REPO_ROOT,
            baselines=baselines,
        )

        self.assertEqual(set(verified), {"overlap", "disjoint"})

    def test_any_existing_subpath_root_is_rejected(self):
        rows = read_and_validate_grid(GRID)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stale_root = root / "checkpoints" / rows[0]["subpath"]
            stale_root.mkdir(parents=True)

            with self.assertRaisesRegex(FileExistsError, "output root"):
                verify_output_paths_absent(rows, repo_root=root)

    def test_extreme_mass_uses_larger_pre_registered_endpoint_region(self):
        counts = [49] + [0] * 18 + [49]
        stats = {
            "weighted_pixel_count": 100,
            "weight_histogram": {"bins": 20, "counts": counts},
        }

        self.assertAlmostEqual(_weight_extreme_mass_ratio(stats), 0.49)

    def test_gate_uses_both_settings_and_all_pre_registered_limits(self):
        passing_rows = [
            {
                "setting": "overlap",
                "status": "done",
                "delta_all_pp": 0.15,
                "delta_old_pp": 0.20,
                "delta_new_pp": -0.01,
                "mask_counts_match": True,
                "manifest_ok": True,
                "result_ok": True,
                "weight_stats_ok": True,
                "weight_extreme_mass_ratio": 0.4,
                "w0_direction_ok": True,
            },
            {
                "setting": "disjoint",
                "status": "done",
                "delta_all_pp": 0.07,
                "delta_old_pp": 0.10,
                "delta_new_pp": -0.02,
                "mask_counts_match": True,
                "manifest_ok": True,
                "result_ok": True,
                "weight_stats_ok": True,
                "weight_extreme_mass_ratio": 0.5,
                "w0_direction_ok": True,
            },
        ]

        passed = evaluate_weighting_gate("confidence", passing_rows)
        failed = evaluate_weighting_gate(
            "confidence",
            [dict(passing_rows[0]), dict(passing_rows[1], delta_all_pp=-0.06)],
        )
        direction_failed = evaluate_weighting_gate(
            "confidence",
            [dict(passing_rows[0]), dict(passing_rows[1], w0_direction_ok=False)],
        )

        self.assertTrue(passed["passed"])
        self.assertAlmostEqual(passed["mean_delta_all_pp"], 0.11)
        self.assertFalse(failed["passed"])
        self.assertIn("per_setting_all_delta", failed["failed"])
        self.assertFalse(direction_failed["passed"])
        self.assertIn("w0_direction", direction_failed["failed"])

    def test_candidate_summary_checks_per_class_counts_and_provenance(self):
        row = next(
            row
            for row in read_and_validate_grid(GRID)
            if row["setting"] == "overlap"
            and row["weighting"] == "confidence"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = (
                root
                / row["subpath"]
                / "voc"
                / row["task"]
                / row["setting"]
                / "step1"
            )
            output_dir.mkdir(parents=True)
            status_path = root / "source_status.txt"
            patch_path = root / "source_patch.diff"
            status_path.write_text("", encoding="utf-8")
            patch_path.write_text("", encoding="utf-8")
            result = {
                "Mean IoU": 0.71,
                "0 to 15 mIoU": 0.80,
                "16 to 20 mIoU": 0.41,
                "Class IoU": {
                    str(class_id): 0.5 + class_id / 100
                    for class_id in range(21)
                },
            }
            (output_dir / "test_results_20260719_000000.json").write_text(
                json.dumps(result),
                encoding="utf-8",
            )
            stats = {
                "schema_version": 2,
                "candidate_count": 2,
                "accepted_count": 2,
                "weighted_pixel_count": 2,
                "per_class_candidates": {"1": 2},
                "per_class_accepted": {"1": 2},
                "weighting": "confidence",
                "task": row["task"],
                "setting": row["setting"],
                "step": 1,
                "strategy": "fixed",
                "fixed_confidence": float(row["confidence"]),
                "teacher_sha256": "a" * 64,
                "weight_histogram": {"counts": [0] * 19 + [2]},
                "weight_mean": 0.8,
                "weight_std": 0.1,
                "weight_p10": 0.6,
                "weight_p25": 0.7,
                "weight_p50": 0.8,
                "weight_p75": 0.9,
                "weight_p90": 0.95,
                "weight_min": 0.5,
                "weight_max": 1.0,
            }
            (output_dir / "pseudo_label_stats.json").write_text(
                json.dumps(stats),
                encoding="utf-8",
            )
            manifest = {
                "task": row["task"],
                "setting": row["setting"],
                "curr_step": 1,
                "subpath": row["subpath"],
                "base_subpath": row["base_subpath"],
                "model": row["model"],
                "batch_size": 32,
                "buffer": 8196,
                "gamma": 1.0,
                "requested_air_feature_source": "auto",
                "resolved_air_feature_source": "decoder",
                "rhl_norm": "none",
                "rhl_seed": -1,
                "pseudo_label_strategy": "fixed",
                "pseudo_label_weighting": "confidence",
                "pseudo_label_confidence": float(row["confidence"]),
                "pseudo_label_quantile": float(row["quantile"]),
                "pseudo_label_min_conf": 0.0,
                "pseudo_label_max_conf": 1.0,
                "pseudo_label_min_pixels": 1,
                "pseudo_label_shrinkage": 0.0,
                "pseudo_label_margin_min": 0.0,
                "pseudo_label_threshold_artifact": None,
                "pseudo_label_threshold_max_batches": 0,
                "random_seed": 1,
                "base_checkpoint_sha256": "a" * 64,
                "teacher_sha256": "a" * 64,
                "source_commit": "f" * 40,
                "source_dirty": False,
                "source_status_path": str(status_path),
                "source_patch_path": str(patch_path),
                "baseline_registry_path": "registry.json",
                "baseline_registry_sha256": "c" * 64,
                "pin_memory": "0",
                "baseline_result_path": "baseline.json",
                "baseline_result_sha256": "b" * 64,
            }
            (output_dir / "run_manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            baseline = {
                "metrics": {
                    "all_miou": 0.70,
                    "old_miou": 0.79,
                    "new_miou": 0.40,
                },
                "class_iou": {
                    str(class_id): 0.49 + class_id / 100
                    for class_id in range(21)
                },
                "candidate_count": 2,
                "accepted_count": 2,
                "per_class_candidates": {"1": 2},
                "per_class_accepted": {"1": 2},
                "teacher_sha256": "a" * 64,
                "result_path_record": "baseline.json",
                "result_sha256": "b" * 64,
                "registry_path_record": "registry.json",
                "registry_sha256": "c" * 64,
                "w0_direction": {"confidence": {"passed": True}},
            }

            summary = summarize_candidate_row(
                row,
                checkpoints_root=root,
                baselines={"overlap": baseline},
            )
            stats["per_class_accepted"] = {"1": 1}
            (output_dir / "pseudo_label_stats.json").write_text(
                json.dumps(stats),
                encoding="utf-8",
            )
            mismatched = summarize_candidate_row(
                row,
                checkpoints_root=root,
                baselines={"overlap": baseline},
            )
            stats["per_class_accepted"] = {"1": 2}
            (output_dir / "pseudo_label_stats.json").write_text(
                json.dumps(stats),
                encoding="utf-8",
            )
            result["Mean IoU"] = float("nan")
            (output_dir / "test_results_20260719_000000.json").write_text(
                json.dumps(result),
                encoding="utf-8",
            )
            nonfinite = summarize_candidate_row(
                row,
                checkpoints_root=root,
                baselines={"overlap": baseline},
            )

        self.assertTrue(summary["manifest_ok"])
        self.assertTrue(summary["result_ok"])
        self.assertTrue(summary["mask_counts_match"])
        self.assertAlmostEqual(summary["per_class_iou_delta"]["1"], 0.01)
        self.assertFalse(mismatched["mask_counts_match"])
        self.assertFalse(nonfinite["result_ok"])

    def test_baseline_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = Path(tmpdir) / "result.json"
            result.write_text(
                json.dumps(
                    {
                        "Mean IoU": 0.7,
                        "0 to 15 mIoU": 0.8,
                        "16 to 20 mIoU": 0.4,
                    }
                ),
                encoding="utf-8",
            )
            registry = Path(tmpdir) / "baselines.json"
            registry.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "settings": {
                            setting: {
                                "result_path": str(result),
                                "result_sha256": "0" * 64,
                                "teacher_sha256": "a" * 64,
                                "candidate_count": 1,
                                "accepted_count": 1,
                                "metrics": {
                                    "all_miou": 0.7,
                                    "old_miou": 0.8,
                                    "new_miou": 0.4,
                                },
                            }
                            for setting in ("overlap", "disjoint")
                        },
                        "w0_audit": {
                            setting: {
                                "summary_path": str(result),
                                "summary_sha256": "0" * 64,
                            }
                            for setting in ("overlap", "disjoint")
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_and_verify_baselines(registry, repo_root=Path("/"))

    def test_runner_dry_run_prints_four_weighted_commands(self):
        env = dict(os.environ)
        env.update(
            {
                "DRY_RUN": "1",
                "PYTHON": "/home/linyichen/miniconda3/envs/segacil/bin/python",
                "RHL_NORM": "l2",
                "RHL_SEED": "99",
                "SEGACIL_PIN_MEMORY": "1",
            }
        )
        result = subprocess.run(
            ["bash", str(RUNNER)],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("bash tools/run_adaptive_pseudo_label.sh"), 4)
        self.assertEqual(result.stdout.count("PSEUDO_LABEL_WEIGHTING=confidence "), 2)
        self.assertEqual(
            result.stdout.count("PSEUDO_LABEL_WEIGHTING=confidence_margin "),
            2,
        )
        self.assertIn("[weighted-w1] dry-run complete", result.stdout)
        self.assertIn(
            "locked_runtime=RHL_NORM=none RHL_SEED=-1 SEGACIL_PIN_MEMORY=0",
            result.stdout,
        )
        self.assertIn("NVIDIA A100", result.stdout)

    def test_runner_rejects_a_concurrent_launch_lock(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = Path(tmpdir) / "weighted-w1.lock"
            with lock_path.open("w") as lock_handle:
                fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                env = dict(os.environ)
                env.update(
                    {
                        "DRY_RUN": "1",
                        "LOCK_PATH": str(lock_path),
                        "PYTHON": (
                            "/home/linyichen/miniconda3/envs/segacil/bin/python"
                        ),
                    }
                )
                result = subprocess.run(
                    ["bash", str(RUNNER)],
                    cwd=REPO_ROOT,
                    env=env,
                    text=True,
                    capture_output=True,
                    check=False,
                )

        self.assertEqual(result.returncode, 3)
        self.assertIn("another W1 runner", result.stderr)

    def test_step0_forces_unweighted_mode(self):
        script = ADAPTIVE_RUNNER.read_text(encoding="utf-8")
        step0 = script.split('echo "[step0]', 1)[1].split('echo "[step1]', 1)[0]

        self.assertIn("PSEUDO_LABEL_WEIGHTING=none", step0)
        self.assertIn("export ALLOW_INCOMPLETE=0", RUNNER.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
