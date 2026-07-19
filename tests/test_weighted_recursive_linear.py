import copy
import unittest

import torch
from torch.nn import functional as F

from network.AnalyticLinear import RecursiveLinear


class WeightedRecursiveLinearTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.dtype = torch.double
        self.gamma = 0.75

    def make_linear(self, in_features=3, out_features=3, bias=False):
        model = RecursiveLinear(
            in_features,
            gamma=self.gamma,
            bias=bias,
            device="cpu",
            dtype=self.dtype,
        )
        rows = in_features + int(bias)
        model.weight = torch.zeros(
            rows, out_features, dtype=self.dtype
        )
        return model

    def fixture(self):
        X = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.5],
                    [0.0, 1.0, 0.5],
                    [1.0, 1.0, 0.0],
                    [0.5, 0.5, 1.0],
                ]
            ],
            dtype=self.dtype,
        )
        y = torch.tensor([[[[0, 1], [2, 1]]]], dtype=torch.long)
        return X, y

    def direct_weighted_solution(self, X, y, sample_weight, out_features=3):
        X_flat = X.reshape(-1, X.shape[-1])
        y_flat = y.reshape(-1)
        weight_flat = sample_weight.reshape(-1).to(self.dtype)
        valid = (y_flat != 255) & (weight_flat > 0)
        X_flat = X_flat[valid]
        y_flat = y_flat[valid]
        weight_flat = weight_flat[valid]
        Y = F.one_hot(y_flat, num_classes=out_features).to(self.dtype)
        sqrt_weight = torch.sqrt(weight_flat).unsqueeze(1)
        Xw = X_flat * sqrt_weight
        Yw = Y * sqrt_weight
        ridge = self.gamma * torch.eye(X.shape[-1], dtype=self.dtype)
        return torch.linalg.solve(ridge + Xw.T @ Xw, Xw.T @ Yw)

    def test_none_argument_is_bitwise_identical_to_frozen_legacy_formula(self):
        X, y = self.fixture()
        model = self.make_linear()
        initial_R = model.R.clone()
        initial_weight = model.weight.clone()
        X_flat = X.reshape(-1, X.shape[-1])
        y_flat = y.reshape(-1)
        Y = F.one_hot(y_flat, num_classes=3).to(self.dtype)
        expected_R = torch.inverse(
            torch.inverse(initial_R) + X_flat.T @ X_flat
        )
        expected_weight = initial_weight + (
            expected_R
            @ X_flat.T
            @ (Y - X_flat @ initial_weight)
        )

        model.fit(X, y, sample_weight=None)

        torch.testing.assert_close(model.R, expected_R, rtol=0, atol=0)
        torch.testing.assert_close(
            model.weight, expected_weight, rtol=0, atol=0
        )

    def test_all_one_weights_match_unweighted_update(self):
        X, y = self.fixture()
        unweighted = self.make_linear()
        weighted = copy.deepcopy(unweighted)

        unweighted.fit(X, y)
        weighted.fit(X, y, sample_weight=torch.ones(1, 2, 2))

        torch.testing.assert_close(unweighted.R, weighted.R, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(
            unweighted.weight, weighted.weight, rtol=1e-12, atol=1e-12
        )

    def test_zero_weight_row_equals_deleting_that_sample(self):
        X, y = self.fixture()
        weighted = self.make_linear()
        deleted = copy.deepcopy(weighted)
        sample_weight = torch.tensor([[[1.0, 0.0], [1.0, 1.0]]])

        weighted.fit(X, y, sample_weight=sample_weight)
        keep = torch.tensor([True, False, True, True])
        deleted.fit(
            X.reshape(1, 4, 3)[:, keep],
            y.reshape(1, 1, 1, 4)[:, :, :, keep],
        )

        torch.testing.assert_close(weighted.R, deleted.R, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(
            weighted.weight, deleted.weight, rtol=1e-12, atol=1e-12
        )

    def test_fractional_weights_match_direct_weighted_ridge(self):
        X, y = self.fixture()
        sample_weight = torch.tensor(
            [[[1.0, 0.25], [0.5, 0.8]]], dtype=self.dtype
        )
        model = self.make_linear()

        model.fit(X, y, sample_weight=sample_weight)

        expected = self.direct_weighted_solution(X, y, sample_weight)
        torch.testing.assert_close(model.weight, expected, rtol=1e-11, atol=1e-11)

    def test_multiple_recursive_batches_match_one_shot_sufficient_statistics(self):
        X, y = self.fixture()
        sample_weight = torch.tensor(
            [[[1.0, 0.25], [0.5, 0.8]]], dtype=self.dtype
        )
        recursive = self.make_linear()
        one_shot = self.make_linear()

        recursive.fit(
            X[:, :2],
            y.reshape(1, 1, 1, 4)[:, :, :, :2],
            sample_weight=sample_weight.reshape(1, 1, 4)[:, :, :2],
        )
        recursive.fit(
            X[:, 2:],
            y.reshape(1, 1, 1, 4)[:, :, :, 2:],
            sample_weight=sample_weight.reshape(1, 1, 4)[:, :, 2:],
        )
        one_shot.fit(X, y, sample_weight=sample_weight)

        torch.testing.assert_close(
            recursive.R, one_shot.R, rtol=1e-11, atol=1e-11
        )
        torch.testing.assert_close(
            recursive.weight, one_shot.weight, rtol=1e-11, atol=1e-11
        )

    def test_ignore_and_zero_weight_are_both_excluded(self):
        X, y = self.fixture()
        y = y.clone()
        y.reshape(-1)[1] = 255
        sample_weight = torch.tensor(
            [[[1.0, 1.0], [0.0, 1.0]]], dtype=self.dtype
        )
        weighted = self.make_linear()
        retained = copy.deepcopy(weighted)

        weighted.fit(X, y, sample_weight=sample_weight)
        keep = torch.tensor([True, False, False, True])
        retained.fit(
            X[:, keep],
            torch.tensor([[[[0, 1]]]], dtype=torch.long),
        )

        torch.testing.assert_close(weighted.R, retained.R, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(
            weighted.weight, retained.weight, rtol=1e-12, atol=1e-12
        )

    def test_invalid_weight_inputs_raise(self):
        X, y = self.fixture()
        cases = [
            (torch.ones(1, 2, 1), "shape"),
            (torch.tensor([[[1.0, float("nan")], [1.0, 1.0]]]), "finite"),
            (torch.tensor([[[1.0, -0.1], [1.0, 1.0]]]), "non-negative"),
        ]
        for sample_weight, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                self.make_linear().fit(X, y, sample_weight=sample_weight)

    def test_all_zero_valid_weights_leave_state_unchanged(self):
        X, y = self.fixture()
        model = self.make_linear()
        before_R = model.R.clone()
        before_weight = model.weight.clone()

        model.fit(X, y, sample_weight=torch.zeros(1, 2, 2))

        torch.testing.assert_close(model.R, before_R, rtol=0, atol=0)
        torch.testing.assert_close(model.weight, before_weight, rtol=0, atol=0)

    def test_bias_false_weighted_path(self):
        X, y = self.fixture()
        model = self.make_linear(bias=False)

        model.fit(X, y, sample_weight=torch.full((1, 2, 2), 0.5))

        self.assertFalse(model.bias)
        self.assertEqual(tuple(model.weight.shape), (3, 3))
        self.assertTrue(torch.isfinite(model.weight).all())

    def test_weighted_path_preserves_class_expansion(self):
        X, y = self.fixture()
        model = RecursiveLinear(
            3,
            gamma=self.gamma,
            bias=False,
            device="cpu",
            dtype=self.dtype,
        )

        model.fit(X[:, :2], torch.tensor([[[[0, 1]]]]), sample_weight=torch.ones(1, 1, 2))
        self.assertEqual(model.out_features, 2)
        model.fit(X[:, 2:], torch.tensor([[[[2, 1]]]]), sample_weight=torch.ones(1, 1, 2))

        self.assertEqual(model.out_features, 3)
        self.assertTrue(torch.isfinite(model.weight).all())


if __name__ == "__main__":
    unittest.main()
