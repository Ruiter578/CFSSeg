import unittest

import torch
from torch import nn

from network._deeplab import DeepLabHead, DeepLabHeadV3Plus
from network.utils import _SimpleSegmentationModel
from trainer.trainer import AIR


class FakeBackbone(nn.Module):
    def forward(self, x):
        batch = x.shape[0]
        device = x.device
        return {
            "out": torch.ones(batch, 2048, 5, 5, device=device),
            "low_level": torch.ones(batch, 256, 9, 9, device=device),
        }


class AirFeatureSourceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        classifier = DeepLabHeadV3Plus(2048, 256, [1, 15])
        self.model = _SimpleSegmentationModel(FakeBackbone(), classifier, bn_freeze=False).eval()
        self.images = torch.randn(2, 3, 33, 33)

    def test_standard_forward_keeps_logits_contract(self):
        logits, details = self.model(self.images)
        self.assertEqual(tuple(logits.shape), (2, 16, 9, 9))
        self.assertEqual(tuple(details["decoder_feature"].shape), (2, 256, 9, 9))

    def test_air_feature_shapes(self):
        expected = {
            "decoder": (2, 256, 9, 9),
            "decoder_stride8": (2, 256, 5, 5),
            "aspp": (2, 256, 5, 5),
            "aspp_up": (2, 256, 9, 9),
        }
        for source, shape in expected.items():
            with self.subTest(source=source):
                feature = self.model.forward_air_features(self.images, source)
                self.assertEqual(tuple(feature.shape), shape)
                self.assertTrue(torch.isfinite(feature).all())

    def test_default_air_feature_matches_decoder_feature(self):
        with torch.no_grad():
            backbone_features = self.model.backbone(self.images)
            expected = self.model.classifier.extract_features(backbone_features)["decoder"]
            actual = self.model.forward_air_features(self.images)
        torch.testing.assert_close(actual, expected)

    def test_invalid_source_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "Unknown AIR feature source"):
            self.model.forward_air_features(self.images, "not-a-source")

    def test_deeplabv3_exposes_same_interface(self):
        classifier = DeepLabHead(2048, [1, 15])
        model = _SimpleSegmentationModel(FakeBackbone(), classifier, bn_freeze=False).eval()
        feature = model.forward_air_features(self.images, "decoder")
        self.assertEqual(tuple(feature.shape), (2, 256, 5, 5))

    def test_air_uses_explicit_feature_interface(self):
        air = AIR(
            backbone_output=256,
            backbone=self.model,
            buffer_size=32,
            gamma=1,
            feature_source="aspp",
            device="cpu",
            dtype=torch.double,
        ).eval()
        expanded = air.feature_expansion(self.images)
        self.assertEqual(tuple(expanded.shape), (2, 25, 32))
        self.assertTrue(torch.isfinite(expanded).all())


if __name__ == "__main__":
    unittest.main()
