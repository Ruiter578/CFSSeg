import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from trainer.trainer import Trainer
from utils.parser import Config
from utils.run_manifest import (
    current_git_commit,
    current_git_dirty,
    safe_write_run_manifest,
    write_run_manifest,
)


class RunManifestTests(unittest.TestCase):
    def test_git_commit_falls_back_when_repository_metadata_is_unavailable(self):
        with patch(
            "utils.run_manifest.subprocess.run",
            side_effect=subprocess.CalledProcessError(128, ["git"]),
        ):
            self.assertEqual(current_git_commit(), "unknown")

    def test_git_dirty_probe_uses_timeout_and_falls_back_when_unavailable(self):
        completed = subprocess.CompletedProcess(
            args=["git"],
            returncode=0,
            stdout=" M utils/run_manifest.py\n",
        )
        with patch("utils.run_manifest.subprocess.run", return_value=completed) as run:
            self.assertTrue(current_git_dirty())

        self.assertEqual(run.call_args.kwargs["timeout"], 5)

        with patch(
            "utils.run_manifest.subprocess.run",
            side_effect=subprocess.TimeoutExpired(["git"], timeout=5),
        ):
            self.assertIsNone(current_git_dirty())

    def test_step0_loader_options_do_not_mutate_live_step(self):
        opts = Config(curr_step=1, task="15-5", setting="sequential")

        step0_opts = Trainer.make_step0_loader_opts(opts)

        self.assertEqual(opts.curr_step, 1)
        self.assertEqual(step0_opts.curr_step, 0)
        self.assertEqual(step0_opts.task, opts.task)
        self.assertIsNot(step0_opts, opts)

    def test_manifest_records_model_source_rhl_and_checkpoint_hash(self):
        opts = replace(
            Config(),
            model="deeplabv3plus_resnet101",
            dataset="voc",
            task="15-5",
            setting="sequential",
            curr_step=1,
            subpath="integration_replay",
            base_subpath="v3plus_step0",
            batch_size=16,
            output_stride=8,
            buffer=8196,
            gamma=1,
            random_seed=1,
            method="acil",
            loss_type="bce_loss",
            lr=0.01,
            rhl_norm="none",
            rhl_seed=-1,
            air_feature_source="auto",
            use_pseudo_label=True,
            pseudo_label_strategy="batch_class",
            pseudo_label_quantile=0.7,
            pseudo_label_min_conf=0.5,
            pseudo_label_max_conf=0.95,
            pseudo_label_min_pixels=16,
            pseudo_label_shrinkage=32,
            pseudo_label_margin_min=0.05,
            pseudo_label_threshold_artifact="thresholds.json",
            pseudo_label_threshold_max_batches=12,
            pseudo_label_stats=True,
            pseudo_label_weighting="confidence_margin",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "step0.pth"
            checkpoint.write_bytes(b"checkpoint fixture")

            manifest_path = write_run_manifest(
                output_dir=tmpdir,
                opts=opts,
                requested_air_feature_source="auto",
                resolved_air_feature_source="aspp_up",
                base_checkpoint_path=str(checkpoint),
                git_commit="abc123",
            )

            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

        self.assertEqual(manifest["model"], "deeplabv3plus_resnet101")
        self.assertEqual(manifest["requested_air_feature_source"], "auto")
        self.assertEqual(manifest["resolved_air_feature_source"], "aspp_up")
        self.assertEqual(manifest["buffer"], 8196)
        self.assertEqual(manifest["data_root"], opts.data_root)
        self.assertEqual(manifest["method"], "acil")
        self.assertEqual(manifest["loss_type"], "bce_loss")
        self.assertEqual(manifest["lr"], 0.01)
        self.assertIn("use_pseudo_label", manifest)
        self.assertEqual(manifest["pseudo_label_strategy"], "batch_class")
        self.assertEqual(manifest["pseudo_label_quantile"], 0.7)
        self.assertEqual(manifest["pseudo_label_min_conf"], 0.5)
        self.assertEqual(manifest["pseudo_label_max_conf"], 0.95)
        self.assertEqual(manifest["pseudo_label_min_pixels"], 16)
        self.assertEqual(manifest["pseudo_label_shrinkage"], 32)
        self.assertEqual(manifest["pseudo_label_margin_min"], 0.05)
        self.assertEqual(
            manifest["pseudo_label_threshold_artifact"],
            "thresholds.json",
        )
        self.assertEqual(manifest["pseudo_label_threshold_max_batches"], 12)
        self.assertTrue(manifest["pseudo_label_stats"])
        self.assertEqual(manifest["pseudo_label_weighting"], "confidence_margin")
        self.assertEqual(
            manifest["args"]["pseudo_label_weighting"],
            "confidence_margin",
        )
        self.assertEqual(manifest["rhl_norm"], "none")
        self.assertEqual(manifest["git_commit"], "abc123")
        self.assertEqual(
            manifest["base_checkpoint_sha256"],
            "f2c14f9cb881ed2ced49b06273c5478f4dbc97d5f6760c9fa074e9517a4cd6d8",
        )

    def test_manifest_records_complete_args_command_git_runtime_and_paths(self):
        opts = replace(
            Config(),
            model="deeplabv3plus_resnet101",
            dataset="voc",
            task="15-5",
            setting="sequential",
            curr_step=1,
            subpath="manifest_complete",
            base_subpath="base_step0",
            gpu_id=[2],
            batch_size=32,
            buffer=8208,
            gamma=1,
            air_feature_source="auto",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "step0.pth"
            checkpoint.write_bytes(b"checkpoint fixture")

            with patch.dict(
                os.environ,
                {"CUDA_VISIBLE_DEVICES": "2"},
                clear=False,
            ), patch.object(
                sys,
                "argv",
                [
                    "train.py",
                    "--subpath",
                    "manifest_complete",
                    "--buffer",
                    "8208",
                ],
            ), patch(
                "utils.run_manifest.socket.gethostname",
                return_value="trs-test-host",
            ), patch(
                "utils.run_manifest.current_git_dirty",
                return_value=True,
            ):
                manifest_path = write_run_manifest(
                    output_dir=tmpdir,
                    opts=opts,
                    requested_air_feature_source="auto",
                    resolved_air_feature_source="aspp_up",
                    base_checkpoint_path=str(checkpoint),
                    git_commit="abc123",
                )

            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(manifest["hostname"], "trs-test-host")
        self.assertEqual(
            manifest["command"],
            ["train.py", "--subpath", "manifest_complete", "--buffer", "8208"],
        )
        self.assertEqual(manifest["git"], {"commit": "abc123", "dirty": True})
        self.assertEqual(manifest["runtime"]["cuda_visible_devices"], "2")
        self.assertIn("python", manifest["runtime"])
        self.assertIn("pytorch", manifest["runtime"])
        self.assertIn("cuda_available", manifest["runtime"])
        self.assertEqual(manifest["resolved_paths"]["output_dir"], tmpdir)
        self.assertEqual(
            manifest["resolved_paths"]["base_checkpoint_path"],
            str(checkpoint),
        )
        self.assertEqual(
            manifest["resolved_paths"]["base_checkpoint_sha256"],
            "f2c14f9cb881ed2ced49b06273c5478f4dbc97d5f6760c9fa074e9517a4cd6d8",
        )
        self.assertEqual(
            manifest["air"],
            {
                "requested_feature_source": "auto",
                "resolved_feature_source": "aspp_up",
            },
        )
        for field in fields(Config):
            self.assertIn(field.name, manifest["args"])
        self.assertEqual(manifest["args"]["test_only"], False)
        self.assertEqual(manifest["args"]["gpu_id"], [2])
        self.assertEqual(manifest["args"]["air_feature_source"], "auto")

        # Keep the old flat keys readable for existing reports and scripts.
        self.assertEqual(manifest["model"], "deeplabv3plus_resnet101")
        self.assertEqual(manifest["buffer"], 8208)
        self.assertEqual(manifest["git_commit"], "abc123")

    def test_manifest_records_w1_source_and_baseline_provenance(self):
        opts = replace(
            Config(),
            setting="overlap",
            pseudo_label_weighting="confidence",
        )
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ,
            {
                "SEGACIL_SOURCE_COMMIT": "source123",
                "SEGACIL_SOURCE_DIRTY": "false",
                "SEGACIL_SOURCE_STATUS_PATH": "logs/source_status.txt",
                "SEGACIL_SOURCE_PATCH_PATH": "logs/source_patch.diff",
                "SEGACIL_BASELINE_REGISTRY_PATH": "configs/baselines.json",
                "SEGACIL_BASELINE_REGISTRY_SHA256": "c" * 64,
                "SEGACIL_PIN_MEMORY": "0",
                "SEGACIL_OVERLAP_BASELINE_RESULT_PATH": "baseline.json",
                "SEGACIL_OVERLAP_BASELINE_RESULT_SHA256": "b" * 64,
            },
            clear=False,
        ):
            checkpoint = Path(tmpdir) / "step0.pth"
            checkpoint.write_bytes(b"teacher")
            manifest_path = write_run_manifest(
                output_dir=tmpdir,
                opts=opts,
                requested_air_feature_source="auto",
                resolved_air_feature_source="decoder",
                base_checkpoint_path=checkpoint,
            )
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

        self.assertEqual(manifest["source_commit"], "source123")
        self.assertFalse(manifest["source_dirty"])
        self.assertEqual(
            manifest["source_status_path"],
            "logs/source_status.txt",
        )
        self.assertEqual(
            manifest["source_patch_path"],
            "logs/source_patch.diff",
        )
        self.assertEqual(
            manifest["baseline_registry_path"],
            "configs/baselines.json",
        )
        self.assertEqual(manifest["baseline_registry_sha256"], "c" * 64)
        self.assertEqual(manifest["pin_memory"], "0")
        self.assertEqual(manifest["baseline_result_path"], "baseline.json")
        self.assertEqual(manifest["baseline_result_sha256"], "b" * 64)
        self.assertEqual(
            manifest["teacher_sha256"],
            manifest["base_checkpoint_sha256"],
        )

    def test_manifest_accepts_plain_namespace_options(self):
        opts = SimpleNamespace(
            model="plain_model",
            buffer=2048,
            air_feature_source="auto",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = write_run_manifest(
                output_dir=tmpdir,
                opts=opts,
                requested_air_feature_source="auto",
                resolved_air_feature_source="decoder",
                git_commit="abc123",
            )

            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

        self.assertEqual(manifest["args"]["model"], "plain_model")
        self.assertEqual(manifest["args"]["buffer"], 2048)
        self.assertEqual(manifest["model"], "plain_model")
        self.assertEqual(manifest["buffer"], 2048)

    def test_safe_manifest_write_returns_none_when_metadata_write_fails(self):
        with patch(
            "utils.run_manifest.write_run_manifest",
            side_effect=OSError("read-only output"),
        ):
            result = safe_write_run_manifest(
                output_dir="/unwritable",
                opts=Config(),
                requested_air_feature_source="auto",
                resolved_air_feature_source=None,
            )

        self.assertIsNone(result)

    def test_manifest_preserves_existing_file_when_output_dir_is_reused(self):
        first_opts = replace(Config(), model="deeplabv3_resnet101", buffer=8196)
        second_opts = replace(Config(), model="deeplabv3_resnet101", buffer=8200)

        with tempfile.TemporaryDirectory() as tmpdir:
            first_path = write_run_manifest(
                output_dir=tmpdir,
                opts=first_opts,
                requested_air_feature_source="auto",
                resolved_air_feature_source="decoder",
                git_commit="abc123",
            )
            second_path = write_run_manifest(
                output_dir=tmpdir,
                opts=second_opts,
                requested_air_feature_source="auto",
                resolved_air_feature_source="decoder",
                git_commit="abc123",
            )

            first_manifest = json.loads(Path(first_path).read_text(encoding="utf-8"))
            second_manifest = json.loads(Path(second_path).read_text(encoding="utf-8"))

        self.assertNotEqual(first_path, second_path)
        self.assertEqual(Path(first_path).name, "run_manifest.json")
        self.assertRegex(Path(second_path).name, r"run_manifest_\d{8}_\d{6}.*\.json")
        self.assertEqual(first_manifest["args"]["buffer"], 8196)
        self.assertEqual(second_manifest["args"]["buffer"], 8200)


if __name__ == "__main__":
    unittest.main()
