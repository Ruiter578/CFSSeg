import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn

from network._deeplab import DeepLabHead, DeepLabHeadBgA, DeepLabHeadV3Plus
from network.modeling import DeepLabModelFactory
from network.utils import _SimpleSegmentationModel
from trainer.trainer import AIR, Trainer
from utils.parser import Config, get_argparser


class FakeBackbone(nn.Module):
    def forward(self, x):
        batch = x.shape[0]
        return {
            "out": torch.ones(batch, 2048, 5, 5, device=x.device),
            "low_level": torch.ones(batch, 256, 9, 9, device=x.device),
        }


class AirFeatureIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.images = torch.randn(2, 3, 33, 33)

    def make_model(self, classifier):
        return _SimpleSegmentationModel(
            FakeBackbone(), classifier, bn_freeze=False
        ).eval()

    def test_v3plus_standard_forward_and_feature_shapes(self):
        model = self.make_model(DeepLabHeadV3Plus(2048, 256, [1, 15]))

        logits, details = model(self.images)
        self.assertEqual(tuple(logits.shape), (2, 16, 9, 9))
        self.assertEqual(tuple(details["decoder_feature"].shape), (2, 256, 9, 9))

        expected = {
            "decoder": (2, 256, 9, 9),
            "decoder_stride8": (2, 256, 5, 5),
            "aspp": (2, 256, 5, 5),
            "aspp_up": (2, 256, 9, 9),
        }
        for source, shape in expected.items():
            with self.subTest(source=source):
                feature = model.forward_air_features(self.images, source)
                self.assertEqual(tuple(feature.shape), shape)
                self.assertTrue(torch.isfinite(feature).all())

    def test_auto_source_is_model_described(self):
        v3 = self.make_model(DeepLabHead(2048, [1, 15]))
        v3plus = self.make_model(DeepLabHeadV3Plus(2048, 256, [1, 15]))

        self.assertEqual(v3.resolve_air_feature_source("auto"), "decoder")
        self.assertEqual(v3plus.resolve_air_feature_source("auto"), "aspp_up")

        torch.testing.assert_close(
            v3.forward_air_features(self.images, "auto"),
            v3.forward_air_features(self.images, "decoder"),
        )
        torch.testing.assert_close(
            v3plus.forward_air_features(self.images, "auto"),
            v3plus.forward_air_features(self.images, "aspp_up"),
        )

    def test_v3_forward_matches_legacy_decoder_math(self):
        model = self.make_model(DeepLabHead(2048, [1, 15]))

        with torch.no_grad():
            backbone_features = model.backbone(self.images)
            aspp = model.classifier.aspp(backbone_features["out"])
            decoder = model.classifier.head_pre(aspp)
            batch, channels, height, width = decoder.shape
            flattened = decoder.view(batch, channels, -1).permute(0, 2, 1)
            expected_logits = model.classifier.head(flattened)
            expected_logits = expected_logits.permute(0, 2, 1).reshape(
                batch,
                -1,
                height,
                width,
            )
            actual_logits, details = model(self.images)

        torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
        torch.testing.assert_close(details["decoder_feature"], decoder, rtol=0, atol=0)

    def test_unsupported_source_fails_instead_of_aliasing(self):
        v3 = self.make_model(DeepLabHead(2048, [1, 15]))

        with self.assertRaisesRegex(ValueError, "does not support AIR feature source"):
            v3.resolve_air_feature_source("aspp_up")
        with self.assertRaisesRegex(ValueError, "does not support AIR feature source"):
            v3.resolve_air_feature_source("not-a-source")

    def test_bga_preserves_decoder_air_contract(self):
        model = self.make_model(DeepLabHeadBgA(2048, [1, 15]))

        self.assertEqual(model.resolve_air_feature_source("auto"), "decoder")
        feature = model.forward_air_features(self.images, "auto")
        self.assertEqual(tuple(feature.shape), (2, 256, 5, 5))

    def test_missing_air_interface_fails_clearly(self):
        model = self.make_model(nn.Identity())

        with self.assertRaisesRegex(TypeError, "does not expose the AIR feature interface"):
            model.resolve_air_feature_source("auto")

    def test_train_returns_model_when_bn_is_not_frozen(self):
        model = self.make_model(DeepLabHead(2048, [1, 15]))
        self.assertIs(model.train(), model)

    def test_air_uses_resolved_source_and_preserves_rhl_options(self):
        backbone = self.make_model(DeepLabHeadV3Plus(2048, 256, [1, 15]))
        air = AIR(
            backbone_output=256,
            backbone=backbone,
            buffer_size=32,
            gamma=1,
            feature_source="aspp_up",
            rhl_norm="l2_sqrt",
            rhl_norm_eps=1e-5,
            rhl_seed=7,
            rhl_stats=False,
            device="cpu",
            dtype=torch.double,
        ).eval()

        expanded = air.feature_expansion(self.images)
        self.assertEqual(air.feature_source, "aspp_up")
        self.assertEqual(tuple(expanded.shape), (2, 81, 32))
        self.assertEqual(air.buffer.rhl_norm, "l2_sqrt")
        self.assertEqual(air.buffer.rhl_norm_eps, 1e-5)

    def test_parser_defaults_to_auto_and_accepts_explicit_source(self):
        with patch.object(sys, "argv", ["train.py"]):
            default_config = get_argparser()
        self.assertEqual(default_config.air_feature_source, "auto")

        with patch.object(
            sys,
            "argv",
            ["train.py", "--air_feature_source", "decoder_stride8"],
        ):
            explicit_config = get_argparser()
        self.assertEqual(explicit_config.air_feature_source, "decoder_stride8")

    def test_resnet_v3plus_factory_and_separable_conv_path(self):
        trainer = Trainer.__new__(Trainer)
        trainer.opts = Config(
            model="deeplabv3plus_resnet50",
            num_classes=[1, 15],
            output_stride=8,
            pretrained_backbone=False,
            bn_freeze=False,
            separable_conv=True,
        )
        trainer.model_factory = DeepLabModelFactory()
        trainer.device = "cpu"

        trainer.init_models()

        self.assertEqual(
            trainer.model.resolve_air_feature_source("auto"),
            "aspp_up",
        )
        self.assertEqual(
            trainer.model.backbone.return_layers["layer1"],
            "low_level",
        )

    def test_factory_defaults_accept_integer_num_classes(self):
        factory = DeepLabModelFactory()

        v3 = factory.deeplabv3_resnet50(pretrained_backbone=False)
        v3plus = factory.deeplabv3plus_resnet50(pretrained_backbone=False)

        self.assertEqual(v3.classifier.head[0].out_features, 21)
        self.assertEqual(v3plus.classifier.head[0].out_features, 21)

    def test_resumed_air_source_rejects_explicit_mismatch(self):
        checkpoint_model = nn.Module()
        checkpoint_model.feature_source = "aspp_up"

        self.assertEqual(
            Trainer.resolve_resumed_air_feature_source(checkpoint_model, "auto"),
            "aspp_up",
        )
        self.assertEqual(
            Trainer.resolve_resumed_air_feature_source(
                checkpoint_model,
                "aspp_up",
            ),
            "aspp_up",
        )
        with self.assertRaisesRegex(ValueError, "checkpoint uses 'aspp_up'"):
            Trainer.resolve_resumed_air_feature_source(
                checkpoint_model,
                "decoder",
            )

    def test_legacy_air_checkpoint_defaults_to_decoder(self):
        checkpoint_model = nn.Module()
        self.assertEqual(
            Trainer.resolve_resumed_air_feature_source(checkpoint_model, "auto"),
            "decoder",
        )


if __name__ == "__main__":
    unittest.main()
