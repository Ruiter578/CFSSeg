import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from network.AnalyticLinear import RecursiveLinear
from trainer.trainer import Trainer
from utils.parser import Config
from utils.run_manifest import write_run_manifest


class HyperparameterSearchPlumbingTests(unittest.TestCase):
    def test_config_defaults_preserve_existing_optimizer_and_tail_behavior(self):
        opts = Config()

        self.assertEqual(opts.backbone_lr, 0.001)
        self.assertEqual(opts.classifier_lr, 0.01)
        self.assertEqual(opts.analytic_tail_epsilon, 1e-3)
        self.assertEqual(opts.evaluation_mode, "test")

    def test_optimizer_uses_explicit_backbone_and_classifier_learning_rates(self):
        trainer = Trainer.__new__(Trainer)
        trainer.opts = Config(
            backbone_lr=0.002,
            classifier_lr=0.02,
            weight_decay=1e-4,
        )
        trainer.model = SimpleNamespace(
            backbone=nn.Linear(2, 2),
            classifier=nn.Linear(2, 2),
        )

        optimizer = trainer.init_optimizer()

        self.assertEqual([group["lr"] for group in optimizer.param_groups], [0.002, 0.02])

    def test_recursive_linear_tail_epsilon_controls_new_class_prior(self):
        layer = RecursiveLinear(2, gamma=1, tail_epsilon=0.0, dtype=torch.double)
        X = torch.zeros((1, 1, 2), dtype=torch.double)
        y = torch.ones((1, 1), dtype=torch.long)

        with patch("network.AnalyticLinear.torch.randn", return_value=torch.ones((2, 2))):
            layer.fit(X, y)

        self.assertTrue(torch.equal(layer.weight, torch.zeros_like(layer.weight)))

    def test_evaluation_mode_dispatch_and_result_prefix(self):
        self.assertEqual(Trainer.evaluation_modes("val"), ("val",))
        self.assertEqual(Trainer.evaluation_modes("test"), ("test",))
        self.assertEqual(Trainer.evaluation_modes("both"), ("val", "test"))
        self.assertEqual(Trainer.evaluation_result_prefix("val"), "val_results")
        self.assertEqual(Trainer.evaluation_result_prefix("test"), "test_results")
        with self.assertRaisesRegex(ValueError, "evaluation mode"):
            Trainer.evaluation_modes("invalid")

    def test_manifest_records_search_plumbing_values(self):
        opts = Config(
            model="deeplabv3_resnet101",
            subpath="manifest_search_plumbing",
            backbone_lr=0.002,
            classifier_lr=0.02,
            analytic_tail_epsilon=1e-4,
            evaluation_mode="val",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_run_manifest(
                output_dir=tmpdir,
                opts=opts,
                requested_air_feature_source="aspp",
                resolved_air_feature_source="aspp",
                git_commit="abc123",
            )
            manifest = __import__("json").loads(Path(path).read_text(encoding="utf-8"))

        self.assertEqual(manifest["backbone_lr"], 0.002)
        self.assertEqual(manifest["classifier_lr"], 0.02)
        self.assertEqual(manifest["analytic_tail_epsilon"], 1e-4)
        self.assertEqual(manifest["evaluation_mode"], "val")


if __name__ == "__main__":
    unittest.main()
