import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from trainer.trainer import Trainer
from utils.parser import Config
from utils.scheduler import build_scheduler


class Step0ResumeTests(unittest.TestCase):
    def test_resume_step0_checkpoint_restores_model_optimizer_and_best_score(self):
        source_model = nn.Linear(2, 1)
        target_model = nn.Linear(2, 1)

        with torch.no_grad():
            source_model.weight.fill_(2.0)
            source_model.bias.fill_(0.5)
            target_model.weight.fill_(-1.0)
            target_model.bias.fill_(-0.5)

        source_optimizer = torch.optim.SGD(
            source_model.parameters(),
            lr=0.123,
            momentum=0.9,
        )
        target_optimizer = torch.optim.SGD(
            target_model.parameters(),
            lr=0.01,
            momentum=0.9,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "step0.pth"
            torch.save(
                {
                    "model_state": source_model.state_dict(),
                    "model_architecture": source_model,
                    "optimizer_state": source_optimizer.state_dict(),
                    "best_score": 0.42,
                },
                ckpt_path,
            )

            trainer = Trainer.__new__(Trainer)
            trainer.model = target_model
            trainer.optimizer = target_optimizer
            trainer.device = "cpu"
            trainer.best_score = -1

            trainer.resume_step0_checkpoint(str(ckpt_path))

        torch.testing.assert_close(target_model.weight, source_model.weight)
        torch.testing.assert_close(target_model.bias, source_model.bias)
        self.assertEqual(target_optimizer.param_groups[0]["lr"], 0.123)
        self.assertEqual(trainer.best_score, 0.42)

    def test_step0_scheduler_resume_uses_completed_iterations(self):
        model = nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        opts = Config(lr_policy="poly", curr_itrs=5)
        total_remaining_itrs = 10

        scheduler = build_scheduler(
            opts,
            optimizer,
            Trainer.step0_scheduler_total_itrs(total_remaining_itrs, opts.curr_itrs),
        )

        Trainer.sync_step0_scheduler_for_resume(
            scheduler,
            optimizer,
            completed_itrs=opts.curr_itrs,
        )

        self.assertEqual(scheduler.max_iters, 15)
        self.assertEqual(scheduler.last_epoch, 5)
        self.assertLess(optimizer.param_groups[0]["lr"], 0.1)

    def test_step0_resume_rejects_completed_iterations_without_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "requires --ckpt"):
            Trainer.validate_step0_resume_options(
                curr_step=0,
                ckpt_path=None,
                completed_itrs=5,
            )

    def test_step0_resume_rejects_negative_completed_iterations(self):
        with self.assertRaisesRegex(ValueError, "must be >= 0"):
            Trainer.step0_scheduler_total_itrs(
                total_itrs=10,
                completed_itrs=-1,
            )

        model = nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = build_scheduler(Config(lr_policy="poly"), optimizer, 10)
        with self.assertRaisesRegex(ValueError, "must be >= 0"):
            Trainer.sync_step0_scheduler_for_resume(
                scheduler,
                optimizer,
                completed_itrs=-1,
            )


if __name__ == "__main__":
    unittest.main()
