import json
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from trainer.trainer import Trainer
from utils.parser import Config
from utils.run_manifest import current_git_commit, write_run_manifest


class RunManifestTests(unittest.TestCase):
    def test_git_commit_falls_back_when_repository_metadata_is_unavailable(self):
        with patch(
            "utils.run_manifest.subprocess.run",
            side_effect=subprocess.CalledProcessError(128, ["git"]),
        ):
            self.assertEqual(current_git_commit(), "unknown")

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
            rhl_norm="none",
            rhl_seed=-1,
            air_feature_source="auto",
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
        self.assertEqual(manifest["rhl_norm"], "none")
        self.assertEqual(manifest["git_commit"], "abc123")
        self.assertEqual(
            manifest["base_checkpoint_sha256"],
            "f2c14f9cb881ed2ced49b06273c5478f4dbc97d5f6760c9fa074e9517a4cd6d8",
        )


if __name__ == "__main__":
    unittest.main()
