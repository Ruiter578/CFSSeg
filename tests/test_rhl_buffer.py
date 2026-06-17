import math
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from network.Buffer import RandomBuffer
from trainer.trainer import Trainer
from utils.ckpt import save_ckpt
from utils.parser import get_argparser


class RandomBufferBOATest(unittest.TestCase):
    def test_parser_accepts_boa_rhl_arguments(self):
        argv = [
            "prog",
            "--rhl_init",
            "orthogonal_antithetic",
            "--rhl_scale_mode",
            "kaiming",
        ]
        with patch.object(sys, "argv", argv):
            opts = get_argparser()

        self.assertEqual(opts.rhl_init, "orthogonal_antithetic")
        self.assertEqual(opts.rhl_scale_mode, "kaiming")

    def test_orthogonal_unit_blocks_have_unit_rows_and_block_orthogonality(self):
        buffer = RandomBuffer(
            4,
            10,
            dtype=torch.double,
            rhl_seed=17,
            rhl_init="orthogonal",
            rhl_scale_mode="unit",
        )

        weight = buffer.weight.double()
        self.assertTrue(torch.allclose(weight.norm(dim=1), torch.ones(10, dtype=torch.double), atol=1e-6))

        for start in range(0, weight.shape[0], buffer.in_features):
            block = weight[start : start + buffer.in_features]
            gram = block @ block.T
            eye = torch.eye(block.shape[0], dtype=torch.double)
            self.assertTrue(torch.allclose(gram, eye, atol=1e-6))

    def test_orthogonal_antithetic_pairs_are_negatives(self):
        out_features = 7
        buffer = RandomBuffer(
            4,
            out_features,
            dtype=torch.double,
            rhl_seed=23,
            rhl_init="orthogonal_antithetic",
            rhl_scale_mode="unit",
        )

        half = math.ceil(out_features / 2)
        paired_rows = out_features - half
        self.assertTrue(
            torch.allclose(
                buffer.weight[half : half + paired_rows],
                -buffer.weight[:paired_rows],
                atol=1e-6,
            )
        )

    def test_orthogonal_scale_modes_set_expected_row_norms(self):
        expected = {
            "legacy": 1.0 / math.sqrt(3.0),
            "kaiming": math.sqrt(2.0),
            "unit": 1.0,
        }

        for scale_mode, row_norm in expected.items():
            with self.subTest(scale_mode=scale_mode):
                buffer = RandomBuffer(
                    4,
                    8,
                    dtype=torch.double,
                    rhl_seed=31,
                    rhl_init="orthogonal",
                    rhl_scale_mode=scale_mode,
                )
                norms = buffer.weight.double().norm(dim=1)
                target = torch.full_like(norms, row_norm)
                self.assertTrue(torch.allclose(norms, target, atol=1e-6))

    def test_boa_forward_preserves_shape_and_relu_nonnegativity(self):
        buffer = RandomBuffer(
            4,
            8,
            dtype=torch.double,
            rhl_seed=31,
            rhl_init="orthogonal_antithetic",
            rhl_scale_mode="kaiming",
        )
        features = torch.randn(2, 3, 4, dtype=torch.double)

        expanded = buffer(features)

        self.assertEqual(tuple(expanded.shape), (2, 3, 8))
        self.assertFalse(torch.isnan(expanded).any().item())
        self.assertTrue((expanded >= 0).all().item())

    def test_seeded_boa_init_does_not_consume_outer_rng(self):
        torch.manual_seed(99)
        _ = torch.rand(3)
        _ = RandomBuffer(
            4,
            8,
            dtype=torch.double,
            rhl_seed=37,
            rhl_init="orthogonal",
            rhl_scale_mode="unit",
        )
        after_buffer = torch.rand(3)

        torch.manual_seed(99)
        _ = torch.rand(3)
        expected_after = torch.rand(3)

        self.assertTrue(torch.allclose(after_buffer, expected_after))

    def test_save_ckpt_persists_training_config(self):
        model = torch.nn.Linear(2, 1)
        config = {
            "rhl_init": "orthogonal_antithetic",
            "rhl_scale_mode": "kaiming",
        }

        with tempfile.NamedTemporaryFile(suffix=".pth") as tmp:
            save_ckpt(tmp.name, model, config=config)
            checkpoint = torch.load(tmp.name, map_location="cpu")

        self.assertEqual(checkpoint["training_config"], config)

    def test_write_run_config_accepts_explicit_config_snapshot(self):
        trainer = Trainer.__new__(Trainer)
        explicit_config = {"curr_step": 0, "rhl_init": "orthogonal"}

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer._write_run_config(tmpdir, config=explicit_config)
            with open(f"{tmpdir}/run_config.json") as f:
                saved = json.load(f)

        self.assertEqual(saved, explicit_config)


if __name__ == "__main__":
    unittest.main()
