import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from utils.ckpt import load_ckpt, save_ckpt


class CheckpointTests(unittest.TestCase):
    def test_save_ckpt_creates_missing_parent_directories(self):
        model = nn.Linear(2, 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "missing" / "step0" / "final.pth"

            save_ckpt(str(ckpt_path), model)

            self.assertTrue(ckpt_path.exists())
            loaded_model, optimizer_state, best_score = load_ckpt(str(ckpt_path))

        self.assertIsInstance(loaded_model, nn.Linear)
        self.assertIsNone(optimizer_state)
        self.assertIsNone(best_score)


if __name__ == "__main__":
    unittest.main()
