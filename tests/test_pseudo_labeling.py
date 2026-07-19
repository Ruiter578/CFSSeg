import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from trainer.trainer import Trainer
from utils.parser import Config, get_argparser
from utils.pseudo_label import (
    PseudoLabelConfig,
    PseudoLabelBatchStats,
    PseudoLabelCandidates,
    apply_pseudo_labels,
    build_pseudo_label_sample_weight,
    compute_pseudo_label_candidates,
    extract_teacher_probabilities,
    load_threshold_artifact,
    resolve_pseudo_label_strategy,
    resolve_thresholds,
    resolve_thresholds_with_fallbacks,
    validate_pseudo_label_config,
)


class PseudoLabelingTests(unittest.TestCase):
    def test_trainer_off_returns_labels_and_no_weight(self):
        trainer = Trainer.__new__(Trainer)
        trainer.device = "cpu"
        trainer.pseudo_labeler = None
        labels = torch.tensor([[[0, 1]]])

        result_labels, sample_weight = trainer.apply_pseudo_labels_if_enabled(
            torch.zeros(1, 3, 1, 2),
            labels,
        )

        torch.testing.assert_close(result_labels, labels)
        self.assertIsNone(sample_weight)

    def test_trainer_returns_weighted_pseudo_labels(self):
        trainer = Trainer.__new__(Trainer)
        trainer.device = "cpu"
        trainer.model_prev = MagicMock(return_value=torch.zeros(1, 2, 1, 2))
        expected_labels = torch.tensor([[[1, 0]]])
        expected_weight = torch.tensor([[[0.8, 1.0]]])
        stats = PseudoLabelBatchStats(
            strategy="fixed",
            candidate_count=1,
            accepted_count=1,
            thresholds={"1": 0.5},
            per_class_candidates={"1": 1},
            per_class_accepted={"1": 1},
            weighting="confidence",
        )
        trainer.pseudo_labeler = SimpleNamespace(
            enabled=True,
            apply=MagicMock(
                return_value=SimpleNamespace(
                    labels=expected_labels,
                    sample_weight=expected_weight,
                    stats=stats,
                )
            ),
        )
        trainer.pseudo_label_stats_records = []

        labels, sample_weight = trainer.apply_pseudo_labels_if_enabled(
            torch.zeros(1, 3, 1, 2),
            torch.zeros(1, 1, 2, dtype=torch.long),
        )

        torch.testing.assert_close(labels, expected_labels)
        torch.testing.assert_close(sample_weight, expected_weight)
        self.assertEqual(trainer.pseudo_label_stats_records, [stats])

    def test_incremental_fit_forwards_weight_to_model(self):
        trainer = Trainer.__new__(Trainer)
        trainer.device = "cpu"
        trainer.model = MagicMock()
        labels = torch.zeros(1, 2, 2, dtype=torch.long)
        sample_weight = torch.full((1, 2, 2), 0.5)
        trainer.apply_pseudo_labels_if_enabled = MagicMock(
            return_value=(labels, sample_weight)
        )
        images = torch.zeros(1, 3, 2, 2)

        trainer.fit_incremental_batch(images, labels)

        trainer.model.fit.assert_called_once()
        args, kwargs = trainer.model.fit.call_args
        torch.testing.assert_close(args[0], images)
        torch.testing.assert_close(args[1], labels)
        torch.testing.assert_close(kwargs["sample_weight"], sample_weight)

    def test_step0_realign_loader_stays_unweighted(self):
        trainer = Trainer.__new__(Trainer)
        trainer.device = "cpu"
        trainer.model = MagicMock()
        trainer.train_loader0 = [
            (
                torch.zeros(1, 3, 2, 2),
                torch.zeros(1, 2, 2, dtype=torch.long),
                None,
            )
        ]

        trainer.fit_step0_realign_loader()

        trainer.model.fit.assert_called_once()
        self.assertNotIn("sample_weight", trainer.model.fit.call_args.kwargs)

    def test_step1_and_later_loader_use_incremental_weighted_path(self):
        trainer = Trainer.__new__(Trainer)
        trainer.fit_incremental_batch = MagicMock()
        batch = (
            torch.zeros(1, 3, 2, 2),
            torch.zeros(1, 2, 2, dtype=torch.long),
            None,
        )
        trainer.train_loader = [batch]

        for curr_step in (1, 2):
            with self.subTest(curr_step=curr_step):
                trainer.opts = SimpleNamespace(curr_step=curr_step)
                trainer.fit_incremental_batch.reset_mock()
                trainer.fit_incremental_loader()
                trainer.fit_incremental_batch.assert_called_once_with(
                    batch[0],
                    batch[1],
                )

    def test_parser_defaults_to_no_weighting_and_accepts_both_signals(self):
        with patch("sys.argv", ["train.py"]):
            defaults = get_argparser()
        self.assertEqual(defaults.pseudo_label_weighting, "none")

        for weighting in ("confidence", "confidence_margin"):
            with self.subTest(weighting=weighting), patch(
                "sys.argv",
                ["train.py", "--pseudo_label_weighting", weighting],
            ):
                opts = get_argparser()
            self.assertEqual(opts.pseudo_label_weighting, weighting)

    def test_invalid_weighting_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "weighting"):
            validate_pseudo_label_config(
                PseudoLabelConfig(strategy="fixed", weighting="temperature")
            )

    def test_none_weighting_returns_none(self):
        labels = torch.tensor([[[0, 255]]])
        candidates = PseudoLabelCandidates(
            scores=torch.tensor([[[0.8, 0.9]]]),
            labels=torch.tensor([[[1, 1]]]),
            margins=torch.tensor([[[0.5, 0.4]]]),
            mask=torch.tensor([[[True, False]]]),
        )

        sample_weight = build_pseudo_label_sample_weight(
            labels,
            candidates,
            torch.tensor([[[True, False]]]),
            weighting="none",
        )

        self.assertIsNone(sample_weight)

    def test_weight_map_uses_signal_only_for_accepted_pseudo_pixels(self):
        labels = torch.tensor([[[0, 0, 2], [255, 0, 3]]])
        candidates = PseudoLabelCandidates(
            scores=torch.tensor([[[0.8, 0.7, 0.6], [0.9, 0.4, 0.5]]]),
            labels=torch.tensor([[[1, 1, 1], [1, 1, 1]]]),
            margins=torch.tensor([[[0.5, 0.2, 0.1], [0.8, 0.3, 0.4]]]),
            mask=torch.tensor(
                [[[True, True, False], [False, True, False]]]
            ),
        )
        accepted = torch.tensor(
            [[[True, False, False], [False, True, False]]]
        )

        confidence = build_pseudo_label_sample_weight(
            labels, candidates, accepted, weighting="confidence"
        )
        confidence_margin = build_pseudo_label_sample_weight(
            labels, candidates, accepted, weighting="confidence_margin"
        )

        torch.testing.assert_close(
            confidence,
            torch.tensor([[[0.8, 1.0, 1.0], [0.0, 0.4, 1.0]]]),
        )
        torch.testing.assert_close(
            confidence_margin,
            torch.tensor([[[0.4, 1.0, 1.0], [0.0, 0.12, 1.0]]]),
        )

    def test_weight_map_rejects_invalid_shapes_masks_and_values(self):
        labels = torch.zeros(1, 1, 2, dtype=torch.long)
        candidates = PseudoLabelCandidates(
            scores=torch.tensor([[[0.8, float("nan")]]]),
            labels=torch.ones(1, 1, 2, dtype=torch.long),
            margins=torch.ones(1, 1, 2),
            mask=torch.tensor([[[True, False]]]),
        )
        with self.assertRaisesRegex(ValueError, "subset"):
            build_pseudo_label_sample_weight(
                labels,
                candidates,
                torch.tensor([[[True, True]]]),
                weighting="confidence",
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            build_pseudo_label_sample_weight(
                labels,
                candidates,
                torch.tensor([[[True, False]]]),
                weighting="confidence",
            )
        with self.assertRaisesRegex(ValueError, "shape"):
            build_pseudo_label_sample_weight(
                labels[:, :, :1],
                candidates,
                torch.tensor([[[True, False]]]),
                weighting="confidence",
            )

    def test_apply_pseudo_labels_preserves_acceptance_and_attaches_weight(self):
        labels = torch.zeros(1, 1, 2, dtype=torch.long)
        candidates = PseudoLabelCandidates(
            scores=torch.tensor([[[0.9, 0.4]]]),
            labels=torch.tensor([[[1, 1]]]),
            margins=torch.tensor([[[0.5, 0.5]]]),
            mask=torch.ones(1, 1, 2, dtype=torch.bool),
        )

        result = apply_pseudo_labels(
            labels,
            candidates,
            thresholds={"1": 0.5},
            margin_min=0.0,
            weighting="confidence",
        )

        self.assertTrue(torch.equal(result.mask, torch.tensor([[[True, False]]])))
        torch.testing.assert_close(
            result.sample_weight,
            torch.tensor([[[0.9, 1.0]]]),
        )

    def test_extract_teacher_probabilities_handles_deeplab_tuple_and_air_nhwc(self):
        nchw_logits = torch.tensor(
            [[[[0.0, 2.0]], [[2.0, 0.0]], [[-2.0, -2.0]]]]
        )
        nhwc_logits = nchw_logits.permute(0, 2, 3, 1).contiguous()

        from_tuple = extract_teacher_probabilities(
            (nchw_logits, {"decoder_feature": torch.empty(0)}),
            loss_type="ce_loss",
            expected_classes=3,
        )
        from_air = extract_teacher_probabilities(
            nhwc_logits,
            loss_type="ce_loss",
            expected_classes=3,
        )

        torch.testing.assert_close(from_tuple, from_air)
        self.assertEqual(tuple(from_tuple.shape), (1, 3, 1, 2))

    def test_extract_teacher_probabilities_rejects_ambiguous_layout(self):
        logits = torch.zeros(1, 3, 2, 3)

        with self.assertRaisesRegex(ValueError, "ambiguous"):
            extract_teacher_probabilities(
                logits,
                loss_type="bce_loss",
                expected_classes=3,
            )

    def test_candidates_only_use_background_old_class_pixels(self):
        probabilities = torch.zeros(1, 5, 2, 3)
        labels = torch.tensor([[[0, 2, 255], [0, 0, 0]]])
        probabilities[0, 1, 0, 0] = 0.9
        probabilities[0, 1, 0, 1] = 0.9
        probabilities[0, 1, 0, 2] = 0.9
        probabilities[0, 4, 1, 0] = 0.95
        probabilities[0, 2, 1, 1] = 0.8
        probabilities[0, 3, 1, 2] = 0.7

        candidates = compute_pseudo_label_candidates(
            probabilities,
            labels,
            old_class_ids=[1, 2],
        )

        self.assertTrue(candidates.mask[0, 0, 0])
        self.assertFalse(candidates.mask[0, 0, 1])
        self.assertFalse(candidates.mask[0, 0, 2])
        self.assertFalse(candidates.mask[0, 1, 0])
        self.assertTrue(candidates.mask[0, 1, 1])
        self.assertFalse(candidates.mask[0, 1, 2])

    def test_fixed_and_batch_class_thresholds_apply_labels_and_preserve_gt(self):
        probabilities = torch.zeros(1, 4, 2, 3)
        labels = torch.tensor([[[0, 2, 255], [0, 0, 0]]])
        probabilities[0, 1, 0, 0] = 0.9
        probabilities[0, 1, 1, 0] = 0.5
        probabilities[0, 2, 1, 1] = 0.8
        probabilities[0, 2, 1, 2] = 0.2

        candidates = compute_pseudo_label_candidates(
            probabilities,
            labels,
            old_class_ids=[1, 2],
        )
        thresholds = resolve_thresholds(
            candidates,
            PseudoLabelConfig(
                strategy="batch_class",
                fixed_confidence=0.7,
                quantile=0.5,
                min_conf=0.0,
                max_conf=1.0,
                min_pixels=1,
                shrinkage=0.0,
            ),
            old_class_ids=[1, 2],
        )
        result = apply_pseudo_labels(
            labels,
            candidates,
            thresholds,
            margin_min=0.0,
        )

        self.assertEqual(result.labels[0, 0, 0].item(), 1)
        self.assertEqual(result.labels[0, 0, 1].item(), 2)
        self.assertEqual(result.labels[0, 0, 2].item(), 255)
        self.assertEqual(result.stats.accepted_count, 2)
        self.assertIn("1", result.stats.thresholds)
        self.assertIn("2", result.stats.thresholds)

    def test_batch_class_records_global_fallback_for_sparse_classes(self):
        probabilities = torch.zeros(1, 3, 1, 3)
        labels = torch.zeros(1, 1, 3, dtype=torch.long)
        probabilities[0, 1, 0, 0] = 0.9
        probabilities[0, 1, 0, 1] = 0.8
        probabilities[0, 2, 0, 2] = 0.7
        candidates = compute_pseudo_label_candidates(
            probabilities,
            labels,
            old_class_ids=[1, 2],
        )

        thresholds, fallbacks = resolve_thresholds_with_fallbacks(
            candidates,
            PseudoLabelConfig(
                strategy="batch_class",
                fixed_confidence=0.5,
                quantile=0.5,
                min_pixels=2,
            ),
            old_class_ids=[1, 2],
        )

        self.assertEqual(fallbacks["1"], "none")
        self.assertEqual(fallbacks["2"], "global")
        self.assertEqual(thresholds["2"], resolve_thresholds(
            candidates,
            PseudoLabelConfig(strategy="batch_global", fixed_confidence=0.5, quantile=0.5),
            old_class_ids=[1, 2],
        )["2"])

    def test_legacy_use_pseudo_label_maps_to_fixed_strategy(self):
        self.assertEqual(
            resolve_pseudo_label_strategy(
                use_pseudo_label=True,
                pseudo_label_strategy=None,
            ),
            "fixed",
        )
        self.assertEqual(
            resolve_pseudo_label_strategy(
                use_pseudo_label=False,
                pseudo_label_strategy="batch_class",
            ),
            "off",
        )

    def test_parser_accepts_adaptive_pseudo_label_options(self):
        with patch(
            "sys.argv",
            [
                "train.py",
                "--use_pseudo_label",
                "--pseudo_label_strategy",
                "batch_class",
                "--pseudo_label_quantile",
                "0.75",
                "--pseudo_label_min_conf",
                "0.4",
                "--pseudo_label_max_conf",
                "0.95",
                "--pseudo_label_min_pixels",
                "16",
                "--pseudo_label_shrinkage",
                "32",
                "--pseudo_label_margin_min",
                "0.05",
                "--pseudo_label_threshold_artifact",
                "thresholds.json",
                "--pseudo_label_threshold_max_batches",
                "12",
                "--pseudo_label_stats",
            ],
        ):
            opts = get_argparser()

        self.assertTrue(opts.use_pseudo_label)
        self.assertEqual(opts.pseudo_label_strategy, "batch_class")
        self.assertEqual(opts.pseudo_label_quantile, 0.75)
        self.assertEqual(opts.pseudo_label_min_conf, 0.4)
        self.assertEqual(opts.pseudo_label_max_conf, 0.95)
        self.assertEqual(opts.pseudo_label_min_pixels, 16)
        self.assertEqual(opts.pseudo_label_shrinkage, 32)
        self.assertEqual(opts.pseudo_label_margin_min, 0.05)
        self.assertEqual(opts.pseudo_label_threshold_artifact, "thresholds.json")
        self.assertEqual(opts.pseudo_label_threshold_max_batches, 12)
        self.assertTrue(opts.pseudo_label_stats)

    def test_trainer_rejects_pseudo_labels_in_sequential_setting(self):
        opts = Config(
            curr_step=1,
            setting="sequential",
            use_pseudo_label=True,
            pseudo_label_strategy="batch_global",
        )

        with self.assertRaisesRegex(ValueError, "sequential"):
            Trainer.validate_pseudo_label_protocol(opts)

    def test_trainer_computes_old_class_ids_and_teacher_class_count(self):
        opts = Config(
            dataset="voc",
            task="15-5",
            curr_step=1,
            num_classes=[1, 15, 5],
        )

        self.assertEqual(
            Trainer.old_class_ids_for_step(opts),
            list(range(1, 16)),
        )
        self.assertEqual(Trainer.teacher_class_count_for_step(opts), 16)

    def test_trainer_writes_pseudo_label_stats_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer.__new__(Trainer)
            trainer.pseudo_labeler = SimpleNamespace(enabled=True)
            trainer.pseudo_label_config = PseudoLabelConfig(
                strategy="batch_class",
                fixed_confidence=0.7,
                quantile=0.7,
                min_conf=0.5,
                max_conf=0.95,
                min_pixels=4,
                shrinkage=16.0,
                margin_min=0.05,
                stats=True,
            )
            trainer.pseudo_label_old_class_ids = [1, 2]
            trainer.opts = Config(
                dataset="voc",
                task="15-5",
                curr_step=1,
                setting="overlap",
            )
            trainer.root_path = tmpdir
            trainer.pseudo_label_stats_records = [
                PseudoLabelBatchStats(
                    strategy="batch_class",
                    candidate_count=10,
                    accepted_count=4,
                    thresholds={"1": 0.7, "2": 0.8},
                    per_class_candidates={"1": 6, "2": 4},
                    per_class_accepted={"1": 3, "2": 1},
                ),
                PseudoLabelBatchStats(
                    strategy="batch_class",
                    candidate_count=5,
                    accepted_count=3,
                    thresholds={"1": 0.6, "2": 0.9},
                    per_class_candidates={"1": 2, "2": 3},
                    per_class_accepted={"1": 1, "2": 2},
                ),
            ]

            stats_path = Path(trainer.save_pseudo_label_stats())
            summary = json.loads(stats_path.read_text(encoding="utf-8"))

        self.assertEqual(summary["strategy"], "batch_class")
        self.assertEqual(summary["candidate_count"], 15)
        self.assertEqual(summary["accepted_count"], 7)
        self.assertAlmostEqual(summary["accepted_ratio"], 7 / 15)
        self.assertEqual(summary["per_class_accepted"], {"1": 4, "2": 3})
        self.assertAlmostEqual(summary["mean_thresholds"]["1"], 0.65)
        self.assertEqual(summary["fallback_counts"], {"1": {}, "2": {}})
        self.assertEqual(len(summary["batch_stats"]), 2)
        self.assertEqual(summary["schema_version"], 2)
        self.assertEqual(summary["weighting"], "none")
        self.assertEqual(summary["weighted_pixel_count"], 0)
        self.assertIsNone(summary["weight_mean"])

    def test_trainer_aggregates_weighted_pseudo_label_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer.__new__(Trainer)
            trainer.pseudo_labeler = SimpleNamespace(enabled=True)
            trainer.pseudo_label_config = PseudoLabelConfig(
                strategy="fixed",
                weighting="confidence",
                stats=True,
            )
            trainer.pseudo_label_old_class_ids = [1, 2]
            trainer.opts = Config(
                dataset="voc",
                task="15-5",
                curr_step=1,
                setting="overlap",
                pseudo_label_threshold_max_batches=0,
            )
            trainer.root_path = tmpdir
            trainer.pseudo_label_stats_records = [
                PseudoLabelBatchStats(
                    strategy="fixed",
                    candidate_count=3,
                    accepted_count=2,
                    thresholds={"1": 0.4, "2": 0.4},
                    per_class_candidates={"1": 2, "2": 1},
                    per_class_accepted={"1": 1, "2": 1},
                    weighting="confidence",
                    weighted_pixel_count=2,
                    weight_sum=1.4,
                    weight_sq_sum=1.0,
                    weight_min=0.6,
                    weight_max=0.8,
                    weight_histogram=[0] * 12 + [1, 0, 0, 1] + [0] * 4,
                    per_class_weight_sum={"1": 0.8, "2": 0.6},
                    per_class_weight_count={"1": 1, "2": 1},
                )
            ]

            stats_path = Path(trainer.save_pseudo_label_stats())
            summary = json.loads(stats_path.read_text(encoding="utf-8"))

        self.assertEqual(summary["schema_version"], 2)
        self.assertEqual(summary["weighting"], "confidence")
        self.assertEqual(summary["weighted_pixel_count"], 2)
        self.assertAlmostEqual(summary["weight_sum"], 1.4)
        self.assertAlmostEqual(summary["weight_mean"], 0.7)
        self.assertAlmostEqual(summary["weight_std"], 0.1)
        self.assertEqual(summary["weight_min"], 0.6)
        self.assertEqual(summary["weight_max"], 0.8)
        self.assertEqual(sum(summary["weight_histogram"]["counts"]), 2)
        self.assertAlmostEqual(summary["per_class_weight_mean"]["1"], 0.8)
        self.assertAlmostEqual(summary["per_class_weight_sum"]["2"], 0.6)

    def test_artifact_thresholds_validate_metadata_and_load_thresholds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = Path(tmpdir) / "thresholds.json"
            artifact.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "dataset": "voc",
                        "task": "15-5",
                        "step": 1,
                        "setting": "overlap",
                        "teacher_sha256": "abc",
                        "loss_type": "bce_loss",
                        "random_seed": 1,
                        "max_batches": 0,
                        "old_class_ids": [1, 2],
                        "global_threshold": 0.6,
                        "classes": {
                            "1": {"final_threshold": 0.7, "candidate_count": 12},
                            "2": {"final_threshold": 0.8, "candidate_count": 8},
                        },
                    }
                ),
                encoding="utf-8",
            )

            thresholds = load_threshold_artifact(
                artifact,
                dataset="voc",
                task="15-5",
                step=1,
                setting="overlap",
                old_class_ids=[1, 2],
                teacher_sha256="abc",
                quantile=0.7,
                loss_type="bce_loss",
                random_seed=1,
                max_batches=0,
            )

        self.assertEqual(thresholds["1"], 0.7)
        self.assertEqual(thresholds["2"], 0.8)

    def test_artifact_thresholds_reject_hyperparameter_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = Path(tmpdir) / "thresholds.json"
            artifact.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "dataset": "voc",
                        "task": "15-5",
                        "step": 1,
                        "setting": "overlap",
                        "teacher_sha256": "abc",
                        "old_class_ids": [1],
                        "quantile": 0.7,
                        "global_threshold": 0.6,
                        "classes": {"1": {"final_threshold": 0.7}},
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "quantile"):
                load_threshold_artifact(
                    artifact,
                    dataset="voc",
                    task="15-5",
                    step=1,
                    setting="overlap",
                    old_class_ids=[1],
                    teacher_sha256="abc",
                    quantile=0.8,
                )

    def test_artifact_strategy_rejects_missing_class_thresholds(self):
        probabilities = torch.zeros(1, 3, 1, 1)
        labels = torch.zeros(1, 1, 1, dtype=torch.long)
        candidates = compute_pseudo_label_candidates(
            probabilities,
            labels,
            old_class_ids=[1, 2],
        )

        with self.assertRaisesRegex(ValueError, "missing entries"):
            resolve_thresholds(
                candidates,
                PseudoLabelConfig(
                    strategy="artifact_class",
                    artifact_thresholds={"1": 0.7},
                ),
                old_class_ids=[1, 2],
            )


if __name__ == "__main__":
    unittest.main()
