import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from trainer.trainer import Trainer
from utils.parser import Config
from utils.scheduler import build_scheduler
from utils.utils import AverageMeter


class Step0ResumeTests(unittest.TestCase):
    def test_step1_uses_best_step0_or_available_final_checkpoint(self):
        root = 'checkpoints/base/voc/15-5/sequential/step0/'
        named = root + 'deeplabv3plus_resnet101_voc_15-5_step_0_sequential.pth'
        for best_exists in [True, False]:
            with self.subTest(best_checkpoint=best_exists):
                opts = Config(model='deeplabv3plus_resnet101', curr_step=1, task='15-5',
                              setting='sequential', num_classes=[1, 15, 5],
                              subpath='new', base_subpath='base')
                def exists(path):
                    return str(path).endswith('/final.pth') or best_exists
                with patch('trainer.trainer.init_dataloader', return_value=([None], [], [])), \
                     patch('trainer.trainer.mkdir'), patch('trainer.trainer.Logger'), \
                     patch('trainer.trainer.os.path.isfile', side_effect=exists):
                    trainer = Trainer(opts)
                self.assertEqual(trainer.ckpt, named if best_exists else root + 'final.pth')

    def test_short_step0_saves_before_first_validation(self):
        class SmallSegmentationModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 2, 1)

            def forward(self, images):
                return self.conv(images), {}

        trainer = Trainer.__new__(Trainer)
        trainer.opts = Config(curr_step=0, train_epoch=1, task='15-5')
        trainer.device = 'cpu'
        trainer.model = SmallSegmentationModel()
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
        trainer.scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, 1)
        trainer.criterion = nn.CrossEntropyLoss()
        trainer.train_loader = [(torch.randn(2, 3, 4, 4), torch.zeros(2, 4, 4, dtype=torch.long), None)]
        trainer.avg_loss, trainer.avg_time, trainer.avg_loss_std = (AverageMeter() for _ in range(3))
        trainer.logger = SimpleNamespace(write_loss=lambda *args: None)
        trainer.root_path = '/unused/'
        trainer.best_score = -1
        with patch('trainer.trainer.save_ckpt') as save:
            trainer.train()
        self.assertEqual(save.call_count, 1)
        self.assertEqual(save.call_args.args[0], '/unused/final.pth')
        self.assertEqual(save.call_args.args[3], -1)

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
