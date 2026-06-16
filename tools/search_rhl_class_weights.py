#!/usr/bin/env python
"""Search RHL-SE class-wise ensemble weights on a validation split.

This script is intentionally validation-driven.  It never uses the test split to
choose weights.  The selected class-wise weights can then be passed to
tools/eval_rhl_ensemble.py for a single held-out test evaluation.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets import init_dataloader
from metrics import StreamSegMetrics
from utils.parser import Config
from utils.tasks import get_tasks

# 复用主评估脚本里的形状处理和模型加载函数，避免搜索脚本和正式 test
# 评估脚本在 BCE sigmoid、logit layout、类别数等细节上出现不一致。
from eval_rhl_ensemble import (
    load_model_cpu,
    logits_to_chw,
    logits_to_prob,
    normalize_weights,
    summarize_split_metrics,
    to_jsonable,
)


def get_args():
    parser = argparse.ArgumentParser(
        description="Search class-wise RHL-SE weights on val split."
    )
    parser.add_argument("--ckpts", nargs="+", required=True, help="step1 final.pth files")
    parser.add_argument("--data_root", default="/root/2TStorage/lyc/SegACIL/data_root/VOC2012")
    parser.add_argument("--dataset", default="voc", choices=["voc", "ade", "cityscapes_domain"])
    parser.add_argument("--task", default="15-5")
    parser.add_argument("--setting", default="sequential", choices=["sequential", "disjoint", "overlap"])
    parser.add_argument("--curr_step", type=int, default=1)
    parser.add_argument("--model", default="deeplabv3_resnet101")
    parser.add_argument("--loss_type", default="bce_loss", choices=["bce_loss", "ce_loss", "focal_loss"])
    parser.add_argument("--crop_size", type=int, default=513)
    parser.add_argument("--val_batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ensemble_mode", default="prob", choices=["prob", "logit"])
    parser.add_argument("--objective", default="all_miou",
                        choices=["all_miou", "new_miou", "old_new_mean"],
                        help="Validation objective used to select the best candidate.")
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="Optional global weights used as one candidate.")
    parser.add_argument("--keep_models_on_gpu", action="store_true")
    parser.add_argument("--save_json", required=True)
    parser.add_argument("--save_class_weights_json", required=True)
    parser.add_argument("--max_batches", type=int, default=-1,
                        help="Debug option: stop after this many val batches.")
    return parser.parse_args()


def build_opts(args):
    opts = Config(
        data_root=args.data_root,
        dataset=args.dataset,
        model=args.model,
        loss_type=args.loss_type,
        crop_val=True,
        crop_size=args.crop_size,
        val_batch_size=args.val_batch_size,
        task=args.task,
        curr_step=args.curr_step,
        setting=args.setting,
        gpu_id=[0] if torch.cuda.is_available() and args.device != "cpu" else None,
        local_rank=0,
    )
    opts.num_classes = [
        len(get_tasks(opts.dataset, opts.task, step))
        for step in range(opts.curr_step + 1)
    ]
    opts.target_cls = [
        get_tasks(opts.dataset, opts.task, step)
        for step in range(opts.curr_step + 1)
    ]
    opts.num_classes = [1, opts.num_classes[0] - 1] + opts.num_classes[1:]
    return opts


def one_hot_weight(num_models, best_idx, strength):
    if num_models == 1:
        return [1.0]
    off = (1.0 - strength) / (num_models - 1)
    weights = [off for _ in range(num_models)]
    weights[best_idx] = strength
    return weights


def normalize_rows(weights):
    arr = np.asarray(weights, dtype=np.float64)
    if (arr < 0).any():
        raise ValueError("candidate class weights must be non-negative")
    sums = arr.sum(axis=1, keepdims=True)
    if (sums <= 0).any():
        raise ValueError("candidate class weights must sum to >0")
    return arr / sums


def candidate_key(results, objective, first_cls):
    if objective == "all_miou":
        return results["Mean IoU"]
    if objective == "new_miou":
        return results[f"{first_cls} to {len(results['Class IoU']) - 1} mIoU"]
    return 0.5 * (
        results[f"0 to {first_cls - 1} mIoU"]
        + results[f"{first_cls} to {len(results['Class IoU']) - 1} mIoU"]
    )


def make_candidate(name, matrix):
    return {"name": name, "weights": normalize_rows(matrix)}


def build_candidates(member_results, first_cls, num_classes, num_models, global_weights):
    # 先统计每个成员在 val split 上的逐类 IoU，再据此构造候选权重矩阵。
    # 注意这里仅使用 val 结果，不读取 test 指标，避免把 test 调参写成方法收益。
    member_class_iou = np.zeros((num_models, num_classes), dtype=np.float64)
    for model_idx, result in enumerate(member_results):
        for cls, value in result["Class IoU"].items():
            member_class_iou[model_idx, int(cls)] = float(value)

    best_member = member_class_iou.argmax(axis=0)
    candidates = []

    equal = np.tile(np.ones(num_models) / num_models, (num_classes, 1))
    candidates.append(make_candidate("equal_all_classes", equal))

    if global_weights is not None:
        candidates.append(
            make_candidate(
                "global_user_weights",
                np.tile(np.asarray(global_weights, dtype=np.float64), (num_classes, 1)),
            )
        )

    if num_models == 3:
        global_244 = np.tile(np.asarray([0.2, 0.4, 0.4]), (num_classes, 1))
        candidates.append(make_candidate("global_0.2_0.4_0.4", global_244))

        old_options = {
            "old_seed1_0.60": [0.60, 0.20, 0.20],
            "old_seed1_0.70": [0.70, 0.15, 0.15],
            "old_seed1_0.80": [0.80, 0.10, 0.10],
        }
        new_options = {
            "new_0.1_0.45_0.45": [0.10, 0.45, 0.45],
            "new_0.2_0.4_0.4": [0.20, 0.40, 0.40],
            "new_seed3_mild": [0.10, 0.30, 0.60],
            "new_seed2_mild": [0.10, 0.60, 0.30],
        }
        for old_name, old_w in old_options.items():
            for new_name, new_w in new_options.items():
                mat = equal.copy()
                mat[:first_cls] = old_w
                mat[first_cls:] = new_w
                candidates.append(make_candidate(f"oldnew_{old_name}_{new_name}", mat))

    for strength in [0.55, 0.65, 0.75]:
        # classwise_valbest_all_s*: 每个类别都偏向 val 上该类最强的成员。
        # strength 越大，越接近“按类别选择单成员”；越小，越接近等权集成。
        mat = []
        for cls in range(num_classes):
            mat.append(one_hot_weight(num_models, int(best_member[cls]), strength))
        candidates.append(make_candidate(f"classwise_valbest_all_s{strength:.2f}", mat))

    for strength in [0.55, 0.65, 0.75]:
        # classwise_valbest_new_s*_oldstable: 旧类保持相对稳定，只让新类按 val
        # 逐类选成员。用于检查 RHL-SE 的互补是否主要集中在增量新类。
        mat = equal.copy()
        if num_models == 3:
            mat[:first_cls] = [0.70, 0.15, 0.15]
        for cls in range(first_cls, num_classes):
            mat[cls] = one_hot_weight(num_models, int(best_member[cls]), strength)
        candidates.append(make_candidate(f"classwise_valbest_new_s{strength:.2f}_oldstable", mat))

    return candidates, best_member.tolist(), member_class_iou


def evaluate_members(models, loader, opts, device, args):
    n_classes = sum(opts.num_classes)
    member_metrics = [StreamSegMetrics(opts.num_classes, dataset=opts.dataset) for _ in models]
    for metric in member_metrics:
        metric.reset()

    with torch.no_grad():
        for batch_idx, (images, labels, _) in enumerate(tqdm(loader, desc="member-val")):
            if args.max_batches >= 0 and batch_idx >= args.max_batches:
                break
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            labels_device = labels.to(device, dtype=torch.long, non_blocking=True)
            labels_np = labels.numpy()

            for idx, model in enumerate(models):
                if not args.keep_models_on_gpu:
                    model = model.to(device).eval()
                logits = model(images)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                probs = logits_to_prob(logits, labels_device.shape[-2:], opts.loss_type, n_classes)
                preds = probs.detach().max(dim=1)[1].cpu().numpy()
                member_metrics[idx].update(labels_np, preds)
                if not args.keep_models_on_gpu:
                    model.cpu()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

    results = []
    for metric in member_metrics:
        result = metric.get_results()
        summarize_split_metrics(result, opts)
        results.append(result)
    return results


def evaluate_candidates(models, loader, opts, device, args, candidates):
    n_classes = sum(opts.num_classes)
    metrics = [StreamSegMetrics(opts.num_classes, dataset=opts.dataset) for _ in candidates]
    for metric in metrics:
        metric.reset()

    with torch.no_grad():
        for batch_idx, (images, labels, _) in enumerate(tqdm(loader, desc="candidate-val")):
            if args.max_batches >= 0 and batch_idx >= args.max_batches:
                break
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            labels_device = labels.to(device, dtype=torch.long, non_blocking=True)
            labels_np = labels.numpy()
            scores_by_member = []

            for model in models:
                if not args.keep_models_on_gpu:
                    model = model.to(device).eval()
                logits = model(images)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                if args.ensemble_mode == "prob":
                    scores = logits_to_prob(logits, labels_device.shape[-2:], opts.loss_type, n_classes)
                else:
                    scores = logits_to_chw(logits, labels_device.shape[-2:], n_classes)
                scores_by_member.append(scores)
                if not args.keep_models_on_gpu:
                    model.cpu()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

            for candidate, metric in zip(candidates, metrics):
                weight_matrix = torch.as_tensor(
                    candidate["weights"],
                    device=device,
                    dtype=scores_by_member[0].dtype,
                )
                score_sum = None
                for model_idx, scores in enumerate(scores_by_member):
                    weighted = scores * weight_matrix[:, model_idx].view(1, -1, 1, 1)
                    score_sum = weighted if score_sum is None else score_sum + weighted
                preds = score_sum.detach().max(dim=1)[1].cpu().numpy()
                metric.update(labels_np, preds)

    results = []
    for candidate, metric in zip(candidates, metrics):
        result = metric.get_results()
        summarize_split_metrics(result, opts)
        results.append({"name": candidate["name"], "results": result})
    return results


def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    opts = build_opts(args)
    _, val_loader, _ = init_dataloader(opts)
    first_cls = len(get_tasks(opts.dataset, opts.task, 0))
    n_classes = sum(opts.num_classes)

    models = [load_model_cpu(path) for path in args.ckpts]
    if args.keep_models_on_gpu:
        models = [model.to(device).eval() for model in models]
    global_weights = normalize_weights(args.weights, len(models)) if args.weights is not None else None

    member_results = evaluate_members(models, val_loader, opts, device, args)
    candidates, best_member, member_class_iou = build_candidates(
        member_results,
        first_cls,
        n_classes,
        len(models),
        global_weights,
    )
    candidate_results = evaluate_candidates(models, val_loader, opts, device, args, candidates)

    scored = []
    for candidate, result in zip(candidates, candidate_results):
        score = candidate_key(result["results"], args.objective, first_cls)
        scored.append((score, candidate, result["results"]))
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_candidate, best_results = scored[0]

    save_dir = os.path.dirname(args.save_class_weights_json)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    with open(args.save_class_weights_json, "w") as f:
        json.dump(to_jsonable(best_candidate["weights"]), f, indent=4)

    summary = {
        "Objective": args.objective,
        "Best Candidate": best_candidate["name"],
        "Best Score": best_score,
        "Best Results": best_results,
        "Best Class Weights Path": args.save_class_weights_json,
        "Best Member Per Class": {str(i): int(v) for i, v in enumerate(best_member)},
        "Member Class IoU": member_class_iou,
        "Member Results": [
            {"Checkpoint": path, "Results": result}
            for path, result in zip(args.ckpts, member_results)
        ],
        "Candidates": [
            {
                "Name": result["name"],
                "Score": candidate_key(result["results"], args.objective, first_cls),
                "Results": result["results"],
            }
            for result in candidate_results
        ],
    }

    save_dir = os.path.dirname(args.save_json)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    with open(args.save_json, "w") as f:
        json.dump(to_jsonable(summary), f, indent=4)

    print(f"Best candidate: {best_candidate['name']}")
    print(f"Best score ({args.objective}): {best_score:.6f}")
    print(f"Saved search summary to {args.save_json}")
    print(f"Saved class weights to {args.save_class_weights_json}")


if __name__ == "__main__":
    main()
