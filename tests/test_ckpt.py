import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from utils.ckpt import load_ckpt, save_ckpt


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_load_deserializes_project_model_on_cpu(self):
        model = nn.Linear(2, 1)
        with patch('utils.ckpt.torch.load', return_value={
            'model_architecture': model, 'model_state': model.state_dict(),
        }) as load:
            loaded_model, _, _ = load_ckpt('project.pth')
        self.assertIs(loaded_model, model)
        load.assert_called_once_with('project.pth', map_location='cpu', weights_only=False)

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
