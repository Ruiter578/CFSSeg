import unittest
from copy import deepcopy

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
        air = self._make_air(feature_source="aspp")
        expanded = air.feature_expansion(self.images)
        self.assertEqual(tuple(expanded.shape), (2, 25, 32))
        self.assertTrue(torch.isfinite(expanded).all())

    def _make_air(self, **kwargs):
        return AIR(
            backbone_output=256,
            backbone=self.model,
            buffer_size=32,
            gamma=1,
            device="cpu",
            dtype=torch.double,
            **kwargs,
        ).eval()

    def test_class_cap_requires_positive_limit(self):
        with self.assertRaisesRegex(ValueError, "requires max_pixels_per_class > 0"):
            self._make_air(pixel_balance="class_cap")

    def test_class_cap_is_deterministic_bounded_and_ordered(self):
        air = self._make_air(pixel_balance="class_cap", max_pixels_per_class=3)
        feature_map = torch.arange(24).reshape(1, 2, 1, 12).float()
        labels = torch.tensor([[[[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 255, 255]]]])
        air.B, air.channle, air.H, air.W = feature_map.shape

        torch.manual_seed(7)
        features1, labels1 = air._select_fit_pixels(feature_map, labels)
        torch.manual_seed(7)
        features2, labels2 = air._select_fit_pixels(feature_map, labels)

        torch.testing.assert_close(features1, features2)
        torch.testing.assert_close(labels1, labels2)
        self.assertLessEqual(int((labels1 == 0).sum()), 3)
        self.assertLessEqual(int((labels1 == 1).sum()), 3)
        self.assertTrue(torch.all(features1[1:, 0] > features1[:-1, 0]))
        self.assertNotIn(255, labels1.tolist())

    def test_class_cap_rejects_batch_without_valid_pixels(self):
        air = self._make_air(pixel_balance="class_cap", max_pixels_per_class=3)
        feature_map = torch.zeros(1, 256, 2, 2)
        labels = torch.full((1, 1, 2, 2), 255)
        air.B, air.channle, air.H, air.W = feature_map.shape
        with self.assertRaisesRegex(ValueError, "contains no valid pixels"):
            air._select_fit_pixels(feature_map, labels)

    def test_no_balance_fit_matches_previous_fit_path(self):
        air = self._make_air(feature_source="aspp", pixel_balance="none")
        reference = deepcopy(air)
        labels = torch.randint(0, 3, (2, 33, 33))

        torch.manual_seed(11)
        air.fit(self.images, labels)
        torch.manual_seed(11)
        expanded = reference.feature_expansion(self.images)
        resized_labels = reference._resize_labels(labels)
        reference.analytic_linear.fit(expanded, resized_labels)

        torch.testing.assert_close(air.analytic_linear.R, reference.analytic_linear.R)
        torch.testing.assert_close(air.analytic_linear.weight, reference.analytic_linear.weight)

    def test_class_cap_fit_updates_finite_state(self):
        air = self._make_air(
            feature_source="aspp",
            pixel_balance="class_cap",
            max_pixels_per_class=4,
        )
        labels = torch.randint(0, 3, (2, 33, 33))
        air.fit(self.images, labels)
        self.assertTrue(torch.isfinite(air.analytic_linear.R).all())
        self.assertTrue(torch.isfinite(air.analytic_linear.weight).all())


if __name__ == "__main__":
    unittest.main()
