#!/usr/bin/env python
"""Read-only raw-mask audit for pseudo-label reliability.

The raw VOC masks are used only to measure how well teacher confidence ranks
pseudo-label correctness. They never enter training or threshold fitting.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import VOCSegmentation
from utils import ext_transforms as et
from utils.parser import Config
from utils.pseudo_label import (
    PseudoLabelCandidates,
    compute_pseudo_label_candidates,
    extract_teacher_probabilities,
    resize_probabilities_to_labels,
)
from utils.tasks import get_tasks


SIGNAL_NAMES = ("confidence", "margin", "confidence_margin")


@dataclass
class CandidateOutcomes:
    correct: torch.Tensor
    false_old_on_background: torch.Tensor
    wrong_old_class: torch.Tensor
    false_old_on_current_or_future: torch.Tensor
    ignored_raw: torch.Tensor
    hidden_old: torch.Tensor


class RawMaskAuditDataset(Dataset):
    """Add the ordered full VOC mask to a deterministic incremental dataset."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, incremental_label, file_name = self.dataset[index]
        raw_mask = np.asarray(Image.open(self.dataset.masks[index]), dtype=np.uint8)
        raw_ordered = torch.from_numpy(
            self.dataset.ordering_map[raw_mask].copy()
        ).long()
        if tuple(raw_ordered.shape) != tuple(incremental_label.shape):
            raise ValueError(
                "Raw and incremental labels must have identical spatial shape: "
                f"{tuple(raw_ordered.shape)} vs {tuple(incremental_label.shape)}"
            )
        return image, incremental_label, raw_ordered, file_name


def _membership_mask(values: torch.Tensor, class_ids: Iterable[int]):
    mask = torch.zeros_like(values, dtype=torch.bool)
    for class_id in class_ids:
        mask |= values == int(class_id)
    return mask


def compute_candidate_outcomes(
    candidates: PseudoLabelCandidates,
    incremental_labels: torch.Tensor,
    raw_ordered_labels: torch.Tensor,
    old_class_ids: Iterable[int],
):
    if candidates.mask.shape != incremental_labels.shape:
        raise ValueError("Candidate and incremental-label shapes do not match.")
    if raw_ordered_labels.shape != incremental_labels.shape:
        raise ValueError("Raw and incremental-label shapes do not match.")

    old_raw = _membership_mask(raw_ordered_labels, old_class_ids)
    candidate = candidates.mask
    correct = candidate & (candidates.labels == raw_ordered_labels)
    false_background = candidate & (raw_ordered_labels == 0)
    wrong_old = (
        candidate
        & old_raw
        & (candidates.labels != raw_ordered_labels)
    )
    current_or_future = (
        candidate
        & (raw_ordered_labels != 0)
        & (raw_ordered_labels != 255)
        & ~old_raw
    )
    ignored_raw = candidate & (raw_ordered_labels == 255)
    hidden_old = (incremental_labels == 0) & old_raw

    classified = (
        correct
        | false_background
        | wrong_old
        | current_or_future
        | ignored_raw
    )
    if not torch.equal(classified, candidate):
        raise RuntimeError("Candidate outcome decomposition is not exhaustive.")
    overlap_count = torch.stack(
        [
            correct,
            false_background,
            wrong_old,
            current_or_future,
            ignored_raw,
        ]
    ).sum(dim=0)
    if bool((overlap_count > 1).any()):
        raise RuntimeError("Candidate outcome decomposition is not exclusive.")

    return CandidateOutcomes(
        correct=correct,
        false_old_on_background=false_background,
        wrong_old_class=wrong_old,
        false_old_on_current_or_future=current_or_future,
        ignored_raw=ignored_raw,
        hidden_old=hidden_old,
    )


def _nullable_float(value):
    value = float(value)
    return value if np.isfinite(value) else None


def distribution_summary(values: np.ndarray):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            key: None
            for key in ("mean", "std", "min", "p10", "p25", "p50", "p75", "p90", "max")
        }
    quantiles = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "mean": _nullable_float(values.mean()),
        "std": _nullable_float(values.std()),
        "min": _nullable_float(values.min()),
        "p10": _nullable_float(quantiles[0]),
        "p25": _nullable_float(quantiles[1]),
        "p50": _nullable_float(quantiles[2]),
        "p75": _nullable_float(quantiles[3]),
        "p90": _nullable_float(quantiles[4]),
        "max": _nullable_float(values.max()),
    }


def _bin_record(mode, index, lower, upper, scores, correct):
    count = int(scores.size)
    return {
        "mode": mode,
        "bin_index": int(index),
        "lower": _nullable_float(lower),
        "upper": _nullable_float(upper),
        "count": count,
        "mean_score": _nullable_float(scores.mean()) if count else None,
        "precision": _nullable_float(correct.mean()) if count else None,
    }


def compute_reliability_bins(
    scores: np.ndarray,
    correct: np.ndarray,
    *,
    mode: str,
    num_bins: int = 10,
):
    scores = np.asarray(scores, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    if scores.ndim != 1 or correct.ndim != 1 or scores.size != correct.size:
        raise ValueError("scores and correct must be same-length 1D arrays.")
    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")
    if scores.size and (not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any()):
        raise ValueError("Reliability scores must be finite and in [0, 1].")

    if mode == "equal_width":
        edges = np.linspace(0.0, 1.0, num_bins + 1)
        assignments = np.searchsorted(edges[1:-1], scores, side="left")
        return [
            _bin_record(
                mode,
                index,
                edges[index],
                edges[index + 1],
                scores[assignments == index],
                correct[assignments == index],
            )
            for index in range(num_bins)
        ]

    if mode != "equal_count":
        raise ValueError(f"Unsupported bin mode: {mode}")
    order = np.argsort(scores, kind="stable")
    chunks = np.array_split(order, num_bins)
    rows = []
    for index, chunk in enumerate(chunks):
        chunk_scores = scores[chunk]
        chunk_correct = correct[chunk]
        lower = chunk_scores.min() if chunk_scores.size else 0.0
        upper = chunk_scores.max() if chunk_scores.size else 1.0
        rows.append(
            _bin_record(
                mode,
                index,
                lower,
                upper,
                chunk_scores,
                chunk_correct,
            )
        )
    return rows


def expected_calibration_error(bins: Sequence[dict]):
    total = sum(int(row["count"]) for row in bins)
    if total == 0:
        return 0.0
    error = 0.0
    for row in bins:
        if not row["count"]:
            continue
        error += (
            int(row["count"])
            / total
            * abs(float(row["precision"]) - float(row["mean_score"]))
        )
    return float(error)


def _rank_with_ties(values: np.ndarray):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    cursor = 0
    while cursor < values.size:
        end = cursor + 1
        while end < values.size and values[order[end]] == values[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = (cursor + end - 1) / 2.0
        cursor = end
    return ranks


def spearman_rho(x: Sequence[float], y: Sequence[float]):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size != y.size or x.size < 2:
        return None
    x_rank = _rank_with_ties(x)
    y_rank = _rank_with_ties(y)
    if np.std(x_rank) == 0 or np.std(y_rank) == 0:
        return 0.0
    return _nullable_float(np.corrcoef(x_rank, y_rank)[0, 1])


def _quartile_precisions(scores: np.ndarray, correct: np.ndarray):
    if scores.size == 0:
        return None, None
    order = np.argsort(scores, kind="stable")
    quartile_size = max(1, int(np.ceil(scores.size / 4.0)))
    bottom = correct[order[:quartile_size]]
    top = correct[order[-quartile_size:]]
    return float(top.mean()), float(bottom.mean())


def summarize_signal(scores, correct):
    scores = np.asarray(scores, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    equal_width = compute_reliability_bins(
        scores, correct, mode="equal_width", num_bins=10
    )
    equal_count = compute_reliability_bins(
        scores, correct, mode="equal_count", num_bins=10
    )
    nonempty_deciles = [row for row in equal_count if row["count"] > 0]
    rho = spearman_rho(
        [row["mean_score"] for row in nonempty_deciles],
        [row["precision"] for row in nonempty_deciles],
    )
    top_precision, bottom_precision = _quartile_precisions(scores, correct)
    gap = (
        None
        if top_precision is None or bottom_precision is None
        else float(top_precision - bottom_precision)
    )
    low_mass = float((scores <= 0.05).mean()) if scores.size else 0.0
    high_mass = float((scores >= 0.95).mean()) if scores.size else 0.0
    return {
        "candidate_count": int(scores.size),
        "distribution": distribution_summary(scores),
        "equal_width_bins": equal_width,
        "equal_count_bins": equal_count,
        "ece": expected_calibration_error(equal_width),
        "nonempty_equal_count_bins": len(nonempty_deciles),
        "spearman_rho": rho,
        "top_quartile_precision": top_precision,
        "bottom_quartile_precision": bottom_precision,
        "top_bottom_precision_gap": gap,
        "low_extreme_mass_ratio": low_mass,
        "high_extreme_mass_ratio": high_mass,
        "extreme_mass_ratio": max(low_mass, high_mass),
    }


def signal_gate(setting_metrics: Dict[str, dict]):
    required_settings = ("overlap", "disjoint")
    result = {"passed": True, "settings": {}}
    for setting in required_settings:
        metrics = setting_metrics.get(setting)
        failed = []
        if metrics is None:
            failed.append("missing_setting")
            result["settings"][setting] = {"passed": False, "failed": failed}
            result["passed"] = False
            continue
        if int(metrics.get("nonempty_equal_count_bins", 0)) < 8:
            failed.append("nonempty_equal_count_bins")
        rho = metrics.get("spearman_rho")
        if rho is None or float(rho) < 0.6:
            failed.append("spearman_rho")
        gap = metrics.get("top_bottom_precision_gap")
        if gap is None or float(gap) < 0.05:
            failed.append("top_bottom_precision_gap")
        if float(metrics.get("extreme_mass_ratio", 1.0)) > 0.95:
            failed.append("extreme_mass_ratio")
        passed = not failed
        result["settings"][setting] = {"passed": passed, "failed": failed}
        result["passed"] &= passed
    return result


def verify_checkpoint_sha256(path, expected_sha256):
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint}")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if expected_sha256 and actual.lower() != str(expected_sha256).lower():
        raise ValueError(
            "Teacher checkpoint SHA256 mismatch: "
            f"expected={expected_sha256}, actual={actual}"
        )
    return actual


def _git_output(args, default):
    try:
        return subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return default


def build_audit_manifest(
    *,
    dataset,
    task,
    step,
    setting,
    file_count,
    teacher_checkpoint,
    teacher_sha256,
    git_commit,
    git_dirty,
    loss_type,
    expected_teacher_classes,
    old_class_ids,
    matched_global_threshold,
    max_samples,
    seed,
):
    return {
        "schema_version": 1,
        "audit_only": True,
        "raw_gt_used_for_training": False,
        "dataset": dataset,
        "task": task,
        "step": int(step),
        "setting": setting,
        "split": "train",
        "file_count": int(file_count),
        "teacher_checkpoint": str(teacher_checkpoint),
        "teacher_sha256": str(teacher_sha256),
        "git_commit": str(git_commit),
        "git_dirty": bool(git_dirty),
        "transform": {
            "geometry": "original",
            "operations": ["ExtToTensor", "ExtNormalize"],
            "random_augmentation": False,
        },
        "loss_type": loss_type,
        "expected_teacher_classes": int(expected_teacher_classes),
        "old_class_ids": [int(value) for value in old_class_ids],
        "matched_global_threshold": float(matched_global_threshold),
        "max_samples": int(max_samples),
        "seed": int(seed),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def _safe_ratio(numerator, denominator):
    if int(denominator) == 0:
        return None
    return float(numerator) / float(denominator)


def _matched_metrics(
    confidence,
    correct,
    predicted,
    hidden_old_by_class,
    threshold,
    old_class_ids,
):
    accepted = confidence >= float(threshold)
    accepted_correct = accepted & correct
    total_hidden = sum(hidden_old_by_class.values())
    global_metrics = {
        "threshold": float(threshold),
        "accepted_count": int(accepted.sum()),
        "precision": _safe_ratio(accepted_correct.sum(), accepted.sum()),
        "hidden_old_recall": _safe_ratio(accepted_correct.sum(), total_hidden),
        "coverage": _safe_ratio(accepted.sum(), confidence.size),
    }
    per_class = {}
    for class_id in old_class_ids:
        class_mask = predicted == int(class_id)
        class_accepted = accepted & class_mask
        class_correct = accepted_correct & class_mask
        per_class[str(class_id)] = {
            "accepted_count": int(class_accepted.sum()),
            "precision": _safe_ratio(class_correct.sum(), class_accepted.sum()),
            "hidden_old_recall": _safe_ratio(
                class_correct.sum(), hidden_old_by_class[str(class_id)]
            ),
            "coverage": _safe_ratio(class_accepted.sum(), class_mask.sum()),
        }
    return global_metrics, per_class


def summarize_audit(
    *,
    setting,
    confidence,
    margin,
    predicted,
    correct,
    error_masks,
    hidden_old_by_class,
    old_class_ids,
    matched_global_threshold,
    processed_samples,
):
    confidence = np.asarray(confidence, dtype=np.float32)
    margin = np.asarray(margin, dtype=np.float32)
    predicted = np.asarray(predicted, dtype=np.int16)
    correct = np.asarray(correct, dtype=bool)
    if not (
        confidence.size
        == margin.size
        == predicted.size
        == correct.size
    ):
        raise ValueError("Candidate arrays have inconsistent lengths.")
    signal_values = {
        "confidence": confidence,
        "margin": margin,
        "confidence_margin": confidence * margin,
    }
    signals = {
        name: summarize_signal(values, correct)
        for name, values in signal_values.items()
    }

    candidate_count = int(correct.size)
    correct_count = int(correct.sum())
    hidden_total = int(sum(hidden_old_by_class.values()))
    errors = {
        name: {
            "count": int(np.asarray(mask, dtype=bool).sum()),
            "share_of_candidates": _safe_ratio(
                np.asarray(mask, dtype=bool).sum(), candidate_count
            ),
        }
        for name, mask in error_masks.items()
    }
    matched_global, matched_per_class = _matched_metrics(
        confidence,
        correct,
        predicted,
        hidden_old_by_class,
        matched_global_threshold,
        old_class_ids,
    )

    per_class = []
    for class_id in old_class_ids:
        class_mask = predicted == int(class_id)
        class_correct = correct & class_mask
        row = {
            "class_id": int(class_id),
            "candidate_count": int(class_mask.sum()),
            "correct_count": int(class_correct.sum()),
            "precision": _safe_ratio(class_correct.sum(), class_mask.sum()),
            "hidden_old_count": int(hidden_old_by_class[str(class_id)]),
            "hidden_old_recall": _safe_ratio(
                class_correct.sum(), hidden_old_by_class[str(class_id)]
            ),
            "confidence": distribution_summary(confidence[class_mask]),
            "margin": distribution_summary(margin[class_mask]),
            "confidence_margin": distribution_summary(
                (confidence * margin)[class_mask]
            ),
            "matched_global": matched_per_class[str(class_id)],
        }
        per_class.append(row)

    return {
        "schema_version": 1,
        "setting": setting,
        "processed_samples": int(processed_samples),
        "candidate_count": candidate_count,
        "correct_count": correct_count,
        "exact_class_precision": _safe_ratio(correct_count, candidate_count),
        "hidden_old_count": hidden_total,
        "hidden_old_recall": _safe_ratio(correct_count, hidden_total),
        "errors": errors,
        "matched_global": matched_global,
        "signals": signals,
        "per_class": per_class,
    }


def _json_text(payload):
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _flatten_per_class(row):
    flat = {
        "class_id": row.get("class_id"),
        "candidate_count": row.get("candidate_count"),
        "correct_count": row.get("correct_count"),
        "precision": row.get("precision"),
        "hidden_old_count": row.get("hidden_old_count"),
        "hidden_old_recall": row.get("hidden_old_recall"),
    }
    for signal in SIGNAL_NAMES:
        for key, value in row.get(signal, {}).items():
            flat[f"{signal}_{key}"] = value
    for key, value in row.get("matched_global", {}).items():
        flat[f"matched_global_{key}"] = value
    return flat


def _render_report(manifest, summary):
    lines = [
        f"# W0 pseudo-label reliability audit: {manifest.get('setting')}",
        "",
        "> 这是只读诊断。完整 raw mask 仅用于审计，没有进入训练或参数拟合。",
        "",
        "## 运行口径",
        "",
        f"- teacher: `{manifest.get('teacher_checkpoint')}`",
        f"- teacher SHA256: `{manifest.get('teacher_sha256')}`",
        f"- processed samples: {summary.get('processed_samples')}",
        f"- deterministic transform: `{manifest.get('transform')}`",
        f"- matched-global threshold: {manifest.get('matched_global_threshold')}",
        "",
        "## 全局结果",
        "",
        f"- candidates: {summary.get('candidate_count')}",
        f"- exact-class precision: {summary.get('exact_class_precision')}",
        f"- hidden-old recall: {summary.get('hidden_old_recall')}",
        "",
        "## 排序信号",
        "",
        "| signal | bins | rho | top-bottom gap | extreme mass | ECE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in summary.get("signals", {}).items():
        lines.append(
            "| {name} | {bins} | {rho} | {gap} | {mass} | {ece} |".format(
                name=name,
                bins=metrics.get("nonempty_equal_count_bins"),
                rho=metrics.get("spearman_rho"),
                gap=metrics.get("top_bottom_precision_gap"),
                mass=metrics.get("extreme_mass_ratio"),
                ece=metrics.get("ece"),
            )
        )
    lines.extend(
        [
            "",
            "## Matched-global",
            "",
            "```json",
            json.dumps(
                summary.get("matched_global", {}),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            ),
            "```",
            "",
            "## 错误分解",
            "",
            "```json",
            json.dumps(
                summary.get("errors", {}),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            ),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_audit_outputs(output_dir, manifest, summary):
    output = Path(output_dir)
    if output.exists():
        if any(output.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite non-empty audit directory: {output}"
            )
    else:
        output.mkdir(parents=True)
    manifest_path = output / "audit_manifest.json"
    summary_path = output / "reliability_summary.json"
    bins_path = output / "reliability_bins.csv"
    per_class_path = output / "per_class.csv"
    report_path = output / "report.md"

    manifest_path.write_text(_json_text(manifest), encoding="utf-8")
    summary_path.write_text(_json_text(summary), encoding="utf-8")

    bin_rows = []
    for signal, metrics in summary.get("signals", {}).items():
        for mode_key in ("equal_width_bins", "equal_count_bins"):
            for row in metrics.get(mode_key, []):
                bin_rows.append({"signal": signal, **row})
    bin_fields = [
        "signal",
        "mode",
        "bin_index",
        "lower",
        "upper",
        "count",
        "mean_score",
        "precision",
    ]
    with bins_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=bin_fields)
        writer.writeheader()
        writer.writerows(bin_rows)

    per_class_rows = [
        _flatten_per_class(row) for row in summary.get("per_class", [])
    ]
    per_class_fields = list(per_class_rows[0]) if per_class_rows else [
        "class_id",
        "candidate_count",
        "correct_count",
        "precision",
        "hidden_old_count",
        "hidden_old_recall",
    ]
    with per_class_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_class_fields)
        writer.writeheader()
        writer.writerows(per_class_rows)

    report_path.write_text(_render_report(manifest, summary), encoding="utf-8")
    return [
        str(manifest_path),
        str(summary_path),
        str(bins_path),
        str(per_class_path),
        str(report_path),
    ]


def _deterministic_transform():
    return et.ExtCompose(
        [
            et.ExtToTensor(),
            et.ExtNormalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def _build_dataset(args):
    opts = Config(
        data_root=args.data_root,
        dataset="voc",
        task=args.task,
        curr_step=args.curr_step,
        setting=args.setting,
        overlap=args.setting == "overlap",
        unknown=0,
    )
    base = VOCSegmentation(
        opts=opts,
        image_set="train",
        transform=_deterministic_transform(),
        cil_step=args.curr_step,
    )
    return RawMaskAuditDataset(base)


def _old_class_ids(task, curr_step):
    class_ids = []
    for step in range(int(curr_step)):
        for class_id in get_tasks("voc", task, step):
            class_id = int(class_id)
            if class_id != 0 and class_id not in class_ids:
                class_ids.append(class_id)
    return sorted(class_ids)


def _select_device(gpu_id):
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        device = torch.device("cuda:0")
    else:
        device = torch.device(f"cuda:{int(gpu_id)}")
    torch.cuda.set_device(device)
    return device


def _load_teacher(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    model = checkpoint["model_architecture"]
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval()


def _concat(chunks, dtype):
    if not chunks:
        return np.array([], dtype=dtype)
    return np.concatenate(chunks).astype(dtype, copy=False)


def run_audit(args):
    if args.curr_step < 1:
        raise ValueError("W0 audit requires --curr-step >= 1.")
    if args.max_samples < 0:
        raise ValueError("--max-samples must be >= 0.")
    if not 0.0 <= args.matched_global_threshold <= 1.0:
        raise ValueError("--matched-global-threshold must be in [0, 1].")
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse audit output directory: {output_dir}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    teacher_sha = verify_checkpoint_sha256(
        args.teacher_checkpoint,
        args.expected_teacher_sha256,
    )
    dataset = _build_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    old_class_ids = _old_class_ids(args.task, args.curr_step)
    expected_teacher_classes = len(old_class_ids) + 1
    if (
        args.expected_teacher_classes is not None
        and int(args.expected_teacher_classes) != expected_teacher_classes
    ):
        raise ValueError(
            "--expected-teacher-classes conflicts with task-derived count: "
            f"{args.expected_teacher_classes} vs {expected_teacher_classes}"
        )

    device = _select_device(args.gpu_id)
    teacher = _load_teacher(args.teacher_checkpoint, device)
    chunks = {
        "confidence": [],
        "margin": [],
        "predicted": [],
        "correct": [],
        "false_old_on_background": [],
        "wrong_old_class": [],
        "false_old_on_current_or_future": [],
        "ignored_raw": [],
    }
    hidden_old_by_class = {str(class_id): 0 for class_id in old_class_ids}
    processed = 0

    with torch.no_grad():
        for images, incremental, raw_ordered, _ in loader:
            if args.max_samples and processed >= args.max_samples:
                break
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            incremental = incremental.to(device, dtype=torch.long, non_blocking=True)
            raw_ordered = raw_ordered.to(device, dtype=torch.long, non_blocking=True)
            probabilities = extract_teacher_probabilities(
                teacher(images),
                loss_type=args.loss_type,
                expected_classes=expected_teacher_classes,
            )
            probabilities = resize_probabilities_to_labels(
                probabilities, incremental
            )
            candidates = compute_pseudo_label_candidates(
                probabilities,
                incremental,
                old_class_ids,
            )
            outcomes = compute_candidate_outcomes(
                candidates,
                incremental,
                raw_ordered,
                old_class_ids,
            )
            candidate_mask = candidates.mask
            chunks["confidence"].append(
                candidates.scores[candidate_mask].float().cpu().numpy()
            )
            chunks["margin"].append(
                candidates.margins[candidate_mask].float().cpu().numpy()
            )
            chunks["predicted"].append(
                candidates.labels[candidate_mask].short().cpu().numpy()
            )
            for name in (
                "correct",
                "false_old_on_background",
                "wrong_old_class",
                "false_old_on_current_or_future",
                "ignored_raw",
            ):
                chunks[name].append(
                    getattr(outcomes, name)[candidate_mask].cpu().numpy()
                )
            for class_id in old_class_ids:
                hidden_old_by_class[str(class_id)] += int(
                    (
                        outcomes.hidden_old
                        & (raw_ordered == int(class_id))
                    ).sum().item()
                )
            processed += 1
            if processed == 1 or processed % 25 == 0:
                print(
                    f"[W0] setting={args.setting} samples={processed}/{len(dataset)} "
                    f"candidates={sum(array.size for array in chunks['confidence'])}",
                    flush=True,
                )

    confidence = _concat(chunks["confidence"], np.float32)
    margin = _concat(chunks["margin"], np.float32)
    predicted = _concat(chunks["predicted"], np.int16)
    correct = _concat(chunks["correct"], bool)
    error_masks = {
        name: _concat(chunks[name], bool)
        for name in (
            "false_old_on_background",
            "wrong_old_class",
            "false_old_on_current_or_future",
            "ignored_raw",
        )
    }
    summary = summarize_audit(
        setting=args.setting,
        confidence=confidence,
        margin=margin,
        predicted=predicted,
        correct=correct,
        error_masks=error_masks,
        hidden_old_by_class=hidden_old_by_class,
        old_class_ids=old_class_ids,
        matched_global_threshold=args.matched_global_threshold,
        processed_samples=processed,
    )
    manifest = build_audit_manifest(
        dataset="voc",
        task=args.task,
        step=args.curr_step,
        setting=args.setting,
        file_count=len(dataset),
        teacher_checkpoint=str(Path(args.teacher_checkpoint).resolve()),
        teacher_sha256=teacher_sha,
        git_commit=_git_output(["rev-parse", "HEAD"], "unknown"),
        git_dirty=bool(_git_output(["status", "--porcelain"], "")),
        loss_type=args.loss_type,
        expected_teacher_classes=expected_teacher_classes,
        old_class_ids=old_class_ids,
        matched_global_threshold=args.matched_global_threshold,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    paths = write_audit_outputs(output_dir, manifest, summary)
    print(
        f"[W0] completed setting={args.setting} samples={processed} "
        f"candidates={summary['candidate_count']} output={output_dir}",
        flush=True,
    )
    return paths


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Audit pseudo-label reliability against ordered raw VOC masks."
    )
    parser.add_argument(
        "--data-root",
        "--data_root",
        default=str(REPO_ROOT / "data_root/VOC2012"),
    )
    parser.add_argument("--task", default="15-5")
    parser.add_argument(
        "--setting", required=True, choices=["overlap", "disjoint"]
    )
    parser.add_argument("--curr-step", "--curr_step", type=int, default=1)
    parser.add_argument(
        "--teacher-checkpoint",
        "--teacher_ckpt",
        required=True,
    )
    parser.add_argument("--expected-teacher-sha256", required=True)
    parser.add_argument("--expected-teacher-classes", type=int)
    parser.add_argument("--matched-global-threshold", type=float, required=True)
    parser.add_argument("--loss-type", default="bce_loss", choices=["bce_loss", "ce_loss", "focal_loss"])
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="0 audits the entire step train split.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main():
    run_audit(parse_args())


if __name__ == "__main__":
    main()
