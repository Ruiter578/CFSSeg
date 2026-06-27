#!/usr/bin/env python
"""Calibrate per-class pseudo-label thresholds from a teacher checkpoint."""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import init_dataloader
from utils import load_ckpt
from utils.parser import Config
from utils.pseudo_label import (
    compute_pseudo_label_candidates,
    extract_teacher_probabilities,
    resize_probabilities_to_labels,
)
from utils.run_manifest import file_sha256
from utils.tasks import get_tasks


def build_opts(args):
    opts = Config(
        data_root=args.data_root,
        dataset=args.dataset,
        task=args.task,
        curr_step=args.curr_step,
        setting=args.setting,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        crop_size=args.crop_size,
        loss_type=args.loss_type,
        random_seed=args.random_seed,
        gpu_id=[args.gpu_id],
    )
    num_classes = [
        len(get_tasks(opts.dataset, opts.task, step))
        for step in range(opts.curr_step + 1)
    ]
    opts.target_cls = [
        get_tasks(opts.dataset, opts.task, step)
        for step in range(opts.curr_step + 1)
    ]
    opts.num_classes = [1, num_classes[0] - 1] + num_classes[1:]
    return opts


def old_class_ids_for_step(opts):
    class_ids = []
    for step in range(opts.curr_step):
        for class_id in get_tasks(opts.dataset, opts.task, step):
            if int(class_id) != 0 and int(class_id) not in class_ids:
                class_ids.append(int(class_id))
    return sorted(class_ids)


def teacher_class_count_for_step(opts):
    return int(sum(opts.num_classes[: opts.curr_step + 1]))


def add_histogram(hist, scores, bins):
    if scores.numel() == 0:
        return hist
    return hist + torch.histc(
        scores.float().cpu(),
        bins=bins,
        min=0.0,
        max=1.0,
    ).long()


def threshold_from_hist(hist, quantile):
    total = int(hist.sum().item())
    if total <= 0:
        return None
    rank = max(1, int(math.ceil(float(quantile) * total)))
    cumsum = torch.cumsum(hist, dim=0)
    bin_idx = int(torch.searchsorted(cumsum, torch.tensor(rank)).item())
    bin_idx = min(max(bin_idx, 0), hist.numel() - 1)
    return float((bin_idx + 0.5) / hist.numel())


def select_device(gpu_id):
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        device = torch.device("cuda:0")
    else:
        device = torch.device(f"cuda:{int(gpu_id)}")
    torch.cuda.set_device(device)
    return device


def clip_threshold(value, min_conf, max_conf):
    return float(max(min_conf, min(max_conf, float(value))))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=str(REPO_ROOT / "data_root/VOC2012"))
    parser.add_argument("--dataset", default="voc", choices=["voc", "ade", "cityscapes_domain"])
    parser.add_argument("--task", default="15-5")
    parser.add_argument("--setting", default="overlap", choices=["overlap", "disjoint"])
    parser.add_argument("--curr_step", type=int, default=1)
    parser.add_argument("--teacher_ckpt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--loss_type", default="bce_loss", choices=["bce_loss", "ce_loss", "focal_loss"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=1)
    parser.add_argument("--crop_size", type=int, default=513)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--quantile", type=float, default=0.7)
    parser.add_argument("--min_conf", type=float, default=0.0)
    parser.add_argument("--max_conf", type=float, default=1.0)
    parser.add_argument("--min_pixels", type=int, default=1)
    parser.add_argument("--shrinkage", type=float, default=0.0)
    parser.add_argument("--bins", type=int, default=256)
    parser.add_argument("--max_batches", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.curr_step < 1:
        raise ValueError("Calibration requires curr_step >= 1.")
    if not 0.0 <= args.quantile <= 1.0:
        raise ValueError("--quantile must be in [0, 1].")
    if args.min_conf > args.max_conf:
        raise ValueError("--min_conf must be <= --max_conf.")
    if args.bins <= 0:
        raise ValueError("--bins must be > 0.")

    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    device = select_device(args.gpu_id)
    opts = build_opts(args)
    train_loader, _, _ = init_dataloader(opts)
    old_class_ids = old_class_ids_for_step(opts)
    teacher_classes = teacher_class_count_for_step(opts)
    teacher = load_ckpt(args.teacher_ckpt)[0].to(device).eval()

    global_hist = torch.zeros(args.bins, dtype=torch.long)
    class_hists = {
        str(class_id): torch.zeros(args.bins, dtype=torch.long)
        for class_id in old_class_ids
    }

    with torch.no_grad():
        for batch_idx, (images, labels, _) in enumerate(train_loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            probabilities = extract_teacher_probabilities(
                teacher(images),
                loss_type=args.loss_type,
                expected_classes=teacher_classes,
            )
            probabilities = resize_probabilities_to_labels(probabilities, labels)
            candidates = compute_pseudo_label_candidates(
                probabilities,
                labels,
                old_class_ids,
            )
            candidate_scores = candidates.scores[candidates.mask]
            global_hist = add_histogram(global_hist, candidate_scores, args.bins)
            for class_id in old_class_ids:
                class_mask = candidates.mask & (candidates.labels == int(class_id))
                class_hists[str(class_id)] = add_histogram(
                    class_hists[str(class_id)],
                    candidates.scores[class_mask],
                    args.bins,
                )

    global_raw = threshold_from_hist(global_hist, args.quantile)
    global_threshold = clip_threshold(
        args.min_conf if global_raw is None else global_raw,
        args.min_conf,
        args.max_conf,
    )
    classes = {}
    for class_id in old_class_ids:
        class_key = str(class_id)
        class_hist = class_hists[class_key]
        count = int(class_hist.sum().item())
        raw = threshold_from_hist(class_hist, args.quantile)
        fallback = "none"
        if raw is None or count < args.min_pixels:
            raw = global_threshold
            fallback = "global"
        if args.shrinkage > 0 and fallback == "none":
            raw = (count * raw + args.shrinkage * global_threshold) / (
                count + args.shrinkage
            )
        classes[class_key] = {
            "candidate_count": count,
            "raw_threshold": float(raw),
            "final_threshold": clip_threshold(raw, args.min_conf, args.max_conf),
            "fallback": fallback,
        }

    artifact = {
        "schema_version": 1,
        "dataset": args.dataset,
        "task": args.task,
        "step": args.curr_step,
        "setting": args.setting,
        "loss_type": args.loss_type,
        "random_seed": args.random_seed,
        "max_batches": args.max_batches,
        "teacher_checkpoint": args.teacher_ckpt,
        "teacher_sha256": file_sha256(args.teacher_ckpt),
        "split": "train",
        "transform": {"type": "train_loader", "crop_size": args.crop_size},
        "old_class_ids": old_class_ids,
        "quantile": args.quantile,
        "min_conf": args.min_conf,
        "max_conf": args.max_conf,
        "min_pixels": args.min_pixels,
        "shrinkage": args.shrinkage,
        "bins": args.bins,
        "global_candidate_count": int(global_hist.sum().item()),
        "global_threshold": global_threshold,
        "classes": classes,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote pseudo-label threshold artifact: {output_path}")


if __name__ == "__main__":
    main()
