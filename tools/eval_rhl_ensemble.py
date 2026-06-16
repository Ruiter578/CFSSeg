#!/usr/bin/env python
"""Evaluate an ensemble of AIR/RHL checkpoints.

This script is intentionally inference-only.  It does not retrain or refit the
analytic head; each checkpoint should already contain a finished step1 AIR model.
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets import init_dataloader
from metrics import StreamSegMetrics
from utils.parser import Config
from utils.tasks import get_tasks


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate RHL Subspace Ensemble checkpoints."
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
    parser.add_argument("--mode", default="test", choices=["val", "test"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save_json", default=None)
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="Optional ensemble weights aligned with --ckpts. They are normalized internally.")
    parser.add_argument("--class_weights_json", default=None,
                        help="Optional class-wise weights. Accepts a CxK list or a dict mapping class id to K weights.")
    parser.add_argument("--old_class_weights", nargs="+", type=float, default=None,
                        help="Optional weights for old classes; combined with --new_class_weights to build class-wise weights.")
    parser.add_argument("--new_class_weights", nargs="+", type=float, default=None,
                        help="Optional weights for new classes; combined with --old_class_weights to build class-wise weights.")
    parser.add_argument("--ensemble_mode", default="prob", choices=["prob", "logit"],
                        help="prob averages sigmoid/softmax scores; logit averages raw logits before argmax.")
    parser.add_argument("--gating_mode", default="none", choices=["none", "margin"],
                        help="Optional confidence gate. margin switches from one member to the ensemble on low-margin pixels.")
    parser.add_argument("--gate_base_index", type=int, default=0,
                        help="0-based checkpoint index used as the conservative base prediction for confidence gating.")
    parser.add_argument("--gate_margin_threshold", type=float, default=0.10,
                        help="Switch to ensemble prediction when the base top1-top2 margin is below this threshold.")
    parser.add_argument("--gate_require_ensemble_better", action="store_true",
                        help="Only switch when ensemble margin is higher than base margin.")
    parser.add_argument("--save_diagnostics", default=None,
                        help="Optional path for member metrics, disagreement rates, and oracle upper bound.")
    parser.add_argument("--keep_models_on_gpu", action="store_true",
                        help="Keep all ensemble members on GPU. Faster but uses more memory.")
    parser.add_argument("--max_batches", type=int, default=-1,
                        help="Debug option: stop after this many batches.")
    return parser.parse_args()


def build_opts(args):
    # 复用项目原有 dataloader / task split 逻辑，保证评估类别划分与 train.py 一致。
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


def tensor_to_chw(tensor, labels_shape, n_classes):
    if tensor.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got shape {tuple(tensor.shape)}")
    if tensor.shape[-1] == n_classes:
        tensor = tensor.permute(0, 3, 1, 2)
    elif tensor.shape[1] != n_classes:
        raise ValueError(
            f"Cannot infer class dimension for shape {tuple(tensor.shape)} "
            f"with n_classes={n_classes}"
        )
    return F.interpolate(tensor, labels_shape, mode="bilinear")


def load_model_cpu(path):
    # 默认先把 checkpoint 加载到 CPU，避免 K 个 RHL-SE 成员同时占满显存。
    checkpoint = torch.load(path, map_location="cpu")
    model = checkpoint["model_architecture"]
    model.load_state_dict(checkpoint["model_state"])
    model.cpu().eval()
    return model


def logits_to_prob(logits, labels_shape, loss_type, n_classes):
    # RHL-SE 做的是“概率平均”而不是 logits 平均：
    # BCE 路径先 sigmoid，多类 CE 路径先 softmax，再对每个像素各类别概率求均值。
    if loss_type == "bce_loss":
        probs = torch.sigmoid(logits)
    else:
        class_dim = -1 if logits.shape[-1] == n_classes else 1
        probs = torch.softmax(logits, dim=class_dim)

    return tensor_to_chw(probs, labels_shape, n_classes)


def logits_to_chw(logits, labels_shape, n_classes):
    # AIR 输出是 [B, H, W, C]，普通分割模型通常是 [B, C, H, W]。
    # logit ensemble 在 raw logits 空间融合，避免 BCE sigmoid 先压缩 margin。
    return tensor_to_chw(logits, labels_shape, n_classes)


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def normalize_weights(weights, num_models):
    if weights is None:
        return [1.0 / num_models] * num_models
    if len(weights) != num_models:
        raise ValueError(
            f"--weights expects {num_models} values to match --ckpts, got {len(weights)}"
        )
    if any(weight < 0 for weight in weights):
        raise ValueError("--weights must be non-negative")
    total = sum(weights)
    if total <= 0:
        raise ValueError("--weights sum must be greater than 0")
    return [weight / total for weight in weights]


def normalize_class_weights(
    class_weights_json,
    old_class_weights,
    new_class_weights,
    global_weights,
    num_classes,
    num_models,
    first_cls,
):
    if class_weights_json is None and old_class_weights is None and new_class_weights is None:
        return None

    raw = None
    if class_weights_json is not None:
        with open(class_weights_json) as f:
            raw = json.load(f)

    class_weights = np.tile(np.asarray(global_weights, dtype=np.float64), (num_classes, 1))
    if old_class_weights is not None:
        if len(old_class_weights) != num_models:
            raise ValueError(
                f"--old_class_weights expects {num_models} values, got {len(old_class_weights)}"
            )
        class_weights[:first_cls] = np.asarray(old_class_weights, dtype=np.float64)
    if new_class_weights is not None:
        if len(new_class_weights) != num_models:
            raise ValueError(
                f"--new_class_weights expects {num_models} values, got {len(new_class_weights)}"
            )
        class_weights[first_cls:] = np.asarray(new_class_weights, dtype=np.float64)

    if isinstance(raw, list):
        arr = np.asarray(raw, dtype=np.float64)
        if arr.shape != (num_classes, num_models):
            raise ValueError(
                f"class_weights_json list must have shape ({num_classes}, {num_models}), "
                f"got {arr.shape}"
            )
        class_weights = arr
    elif isinstance(raw, dict):
        for key, values in raw.items():
            cls = int(key)
            if cls < 0 or cls >= num_classes:
                raise ValueError(f"class id {cls} is out of range [0, {num_classes - 1}]")
            if len(values) != num_models:
                raise ValueError(
                    f"class {cls} expects {num_models} weights, got {len(values)}"
                )
            class_weights[cls] = np.asarray(values, dtype=np.float64)
    elif raw is not None:
        raise ValueError("class_weights_json must be either a CxK list or a dict")

    if (class_weights < 0).any():
        raise ValueError("class-wise weights must be non-negative")
    sums = class_weights.sum(axis=1, keepdims=True)
    if (sums <= 0).any():
        bad = np.where((sums <= 0).reshape(-1))[0].tolist()
        raise ValueError(f"class-wise weights must sum to >0 for every class, bad classes: {bad}")
    return class_weights / sums


def validate_gating_args(args, num_models):
    if args.gating_mode == "none":
        return
    if args.gate_base_index < 0 or args.gate_base_index >= num_models:
        raise ValueError(
            f"--gate_base_index must be in [0, {num_models - 1}], got {args.gate_base_index}"
        )
    if args.gate_margin_threshold < 0:
        raise ValueError("--gate_margin_threshold must be non-negative")


def top2_margin(scores):
    top2 = scores.topk(k=2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def apply_margin_gate(member_scores, ensemble_scores, base_index, threshold, require_ensemble_better):
    # 置信度门控：以某个成员作为保守 base；当 base 的 top1-top2 margin 很低时，
    # 才切换到集成分数。这样试图保留 seed1 的旧类稳定性，同时利用 RHL-SE 在不确定
    # 像素上的互补信息。
    base_scores = member_scores[base_index]
    base_margin = top2_margin(base_scores)
    ensemble_margin = top2_margin(ensemble_scores)
    switch_mask = base_margin < threshold
    if require_ensemble_better:
        switch_mask = switch_mask & (ensemble_margin > base_margin)
    return torch.where(switch_mask[:, None, :, :], ensemble_scores, base_scores), {
        "base_index": base_index,
        "margin_threshold": threshold,
        "require_ensemble_better": require_ensemble_better,
        "switched_pixels": int(switch_mask.sum().item()),
        "total_pixels": int(switch_mask.numel()),
    }


def summarize_split_metrics(results, opts):
    first_cls = len(get_tasks(opts.dataset, opts.task, 0))
    class_iou = list(results["Class IoU"].values())
    class_acc = list(results["Class Acc"].values())
    results[f"0 to {first_cls - 1} mIoU"] = np.mean(class_iou[:first_cls])
    results[f"{first_cls} to {len(class_iou) - 1} mIoU"] = np.mean(class_iou[first_cls:])
    results[f"0 to {first_cls - 1} mAcc"] = np.mean(class_acc[:first_cls])
    results[f"{first_cls} to {len(class_iou) - 1} mAcc"] = np.mean(class_acc[first_cls:])
    return first_cls, class_iou, class_acc


def new_disagreement_stats(num_classes):
    return {
        "pair_total": 0,
        "pair_disagree": 0,
        "pair_old_total": 0,
        "pair_old_disagree": 0,
        "pair_new_total": 0,
        "pair_new_disagree": 0,
        "class_total": [0 for _ in range(num_classes)],
        "class_any_disagree": [0 for _ in range(num_classes)],
    }


def update_disagreement_stats(stats, member_preds, labels, first_cls, num_classes):
    # 诊断只统计有监督像素，忽略 VOC 的 255 ignore 区域。
    valid = labels != 255
    old_mask = valid & (labels < first_cls)
    new_mask = valid & (labels >= first_cls)

    pred_stack = np.stack(member_preds, axis=0)
    num_members = pred_stack.shape[0]
    for i in range(num_members):
        for j in range(i + 1, num_members):
            disagree = pred_stack[i] != pred_stack[j]
            stats["pair_total"] += int(valid.sum())
            stats["pair_disagree"] += int((disagree & valid).sum())
            stats["pair_old_total"] += int(old_mask.sum())
            stats["pair_old_disagree"] += int((disagree & old_mask).sum())
            stats["pair_new_total"] += int(new_mask.sum())
            stats["pair_new_disagree"] += int((disagree & new_mask).sum())

    any_disagree = np.any(pred_stack != pred_stack[0:1], axis=0)
    for cls in range(num_classes):
        cls_mask = valid & (labels == cls)
        stats["class_total"][cls] += int(cls_mask.sum())
        stats["class_any_disagree"][cls] += int((any_disagree & cls_mask).sum())


def finalize_disagreement_stats(stats):
    def ratio(num, den):
        return float(num / den) if den else 0.0

    return {
        "pairwise_disagreement": ratio(stats["pair_disagree"], stats["pair_total"]),
        "pairwise_old_disagreement": ratio(stats["pair_old_disagree"], stats["pair_old_total"]),
        "pairwise_new_disagreement": ratio(stats["pair_new_disagree"], stats["pair_new_total"]),
        "per_class_any_disagreement": {
            str(cls): ratio(disagree, total)
            for cls, (disagree, total) in enumerate(
                zip(stats["class_any_disagree"], stats["class_total"])
            )
        },
    }


def oracle_predictions(member_preds, ensemble_preds, labels):
    pred_stack = np.stack(member_preds, axis=0)
    valid = labels != 255
    correct_by_any = np.any(pred_stack == labels[None, ...], axis=0) & valid
    oracle = ensemble_preds.copy()
    oracle[correct_by_any] = labels[correct_by_any]
    return oracle


def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    opts = build_opts(args)
    _, val_loader, test_loader = init_dataloader(opts)
    loader = test_loader if args.mode == "test" else val_loader
    metrics = StreamSegMetrics(opts.num_classes, dataset=opts.dataset)
    n_classes = sum(opts.num_classes)
    first_cls = len(get_tasks(opts.dataset, opts.task, 0))

    models = [load_model_cpu(path) for path in args.ckpts]
    weights = normalize_weights(args.weights, len(models))
    class_weights = normalize_class_weights(
        args.class_weights_json,
        args.old_class_weights,
        args.new_class_weights,
        weights,
        n_classes,
        len(models),
        first_cls,
    )
    validate_gating_args(args, len(models))
    run_diagnostics = args.save_diagnostics is not None
    member_metrics = [
        StreamSegMetrics(opts.num_classes, dataset=opts.dataset)
        for _ in models
    ] if run_diagnostics else []
    oracle_metrics = StreamSegMetrics(opts.num_classes, dataset=opts.dataset) if run_diagnostics else None
    disagreement_stats = new_disagreement_stats(n_classes) if run_diagnostics else None
    gating_stats = {"switched_pixels": 0, "total_pixels": 0}

    print(f"Ensemble mode: {args.ensemble_mode}")
    print("Ensemble weights:")
    for path, weight in zip(args.ckpts, weights):
        print(f"  {weight:.6f}  {path}")
    if class_weights is not None:
        print("Using class-wise weights")
        if args.class_weights_json is not None:
            print(f"  class_weights_json={args.class_weights_json}")
        if args.old_class_weights is not None:
            print(f"  old_class_weights={args.old_class_weights}")
        if args.new_class_weights is not None:
            print(f"  new_class_weights={args.new_class_weights}")
    if args.gating_mode != "none":
        print(
            "Gating: "
            f"mode={args.gating_mode}, base={args.gate_base_index}, "
            f"threshold={args.gate_margin_threshold}, "
            f"require_ensemble_better={args.gate_require_ensemble_better}"
        )
    if args.keep_models_on_gpu:
        # 显存足够时可一次性常驻 GPU，速度更快；默认不用，优先保证不 OOM。
        models = [model.to(device).eval() for model in models]

    metrics.reset()
    for metric in member_metrics:
        metric.reset()
    if oracle_metrics is not None:
        oracle_metrics.reset()

    with torch.no_grad():
        for batch_idx, (images, labels, _) in enumerate(tqdm(loader)):
            if args.max_batches >= 0 and batch_idx >= args.max_batches:
                break

            images = images.to(device, dtype=torch.float32, non_blocking=True)
            labels_device = labels.to(device, dtype=torch.long, non_blocking=True)
            weighted_score_sum = None
            member_preds = []
            member_scores = []

            for model_idx, (model, weight) in enumerate(zip(models, weights)):
                if not args.keep_models_on_gpu:
                    # 低显存模式：每次只把一个成员放上 GPU，算完立即搬回 CPU。
                    model = model.to(device).eval()
                logits = model(images)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                logits_chw = logits_to_chw(logits, labels_device.shape[-2:], n_classes)
                probs_chw = None
                if args.ensemble_mode == "prob" or run_diagnostics:
                    probs_chw = logits_to_prob(
                        logits,
                        labels_device.shape[-2:],
                        opts.loss_type,
                        n_classes,
                    )
                scores = probs_chw if args.ensemble_mode == "prob" else logits_chw

                # 加权融合：默认用全局模型权重；若传入 class_weights_json，则每个类别有
                # 自己的 K 个成员权重，可表达“旧类信 seed1，新类信 seed2/3”等策略。
                if class_weights is None:
                    score_weight = weight
                else:
                    score_weight = torch.as_tensor(
                        class_weights[:, model_idx],
                        device=scores.device,
                        dtype=scores.dtype,
                    ).view(1, -1, 1, 1)
                weighted_scores = scores * score_weight
                weighted_score_sum = (
                    weighted_scores
                    if weighted_score_sum is None
                    else weighted_score_sum + weighted_scores
                )

                if run_diagnostics:
                    member_pred = probs_chw.detach().max(dim=1)[1].cpu().numpy()
                    member_preds.append(member_pred)
                    member_metrics[len(member_preds) - 1].update(labels.numpy(), member_pred)
                if args.gating_mode != "none":
                    member_scores.append(scores)

                if not args.keep_models_on_gpu:
                    model.cpu()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

            # 像素级概率平均/加权平均：这是 RHL-SE 最终形成集成预测的地方。
            ensemble_scores = weighted_score_sum
            gate_info = None
            if args.gating_mode == "margin":
                ensemble_scores, gate_info = apply_margin_gate(
                    member_scores,
                    ensemble_scores,
                    args.gate_base_index,
                    args.gate_margin_threshold,
                    args.gate_require_ensemble_better,
                )
                gating_stats["switched_pixels"] += gate_info["switched_pixels"]
                gating_stats["total_pixels"] += gate_info["total_pixels"]
            preds = ensemble_scores.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.numpy()
            metrics.update(targets, preds)

            if run_diagnostics:
                update_disagreement_stats(
                    disagreement_stats,
                    member_preds,
                    targets,
                    first_cls,
                    n_classes,
                )
                oracle_metrics.update(
                    targets,
                    oracle_predictions(member_preds, preds, targets),
                )

    results = metrics.get_results()
    first_cls, class_iou, class_acc = summarize_split_metrics(results, opts)

    # 项目原生 metrics.to_str() 假设 results 里的非 Class IoU/Acc 字段全是 float。
    # Ensemble Members 是 checkpoint 路径列表，如果提前放进 results，会触发
    # TypeError: must be real number, not list。这里先打印纯数值结果，再补充集成元信息。
    print(metrics.to_str(results))
    print(f"...from 0 to {first_cls - 1} : ensemble/test_before_mIoU : {results[f'0 to {first_cls - 1} mIoU']:.6f}")
    print(f"...from {first_cls} to {len(class_iou) - 1} ensemble/test_after_mIoU : {results[f'{first_cls} to {len(class_iou) - 1} mIoU']:.6f}")

    results["Ensemble Members"] = args.ckpts
    results["Ensemble Size"] = len(args.ckpts)
    results["Ensemble Weights"] = weights
    results["Ensemble Mode"] = args.ensemble_mode
    if class_weights is not None:
        results["Class Weights"] = class_weights
    results["Gating Mode"] = args.gating_mode
    if args.gating_mode != "none":
        results["Gate Base Index"] = args.gate_base_index
        results["Gate Margin Threshold"] = args.gate_margin_threshold
        results["Gate Require Ensemble Better"] = args.gate_require_ensemble_better
        results["Gate Switched Pixels"] = gating_stats["switched_pixels"]
        results["Gate Total Pixels"] = gating_stats["total_pixels"]
        results["Gate Switch Ratio"] = (
            gating_stats["switched_pixels"] / gating_stats["total_pixels"]
            if gating_stats["total_pixels"]
            else 0.0
        )

    save_json = args.save_json
    if save_json is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_json = os.path.join("logs", "rhl_ensemble", f"ensemble_results_{stamp}.json")
    save_dir = os.path.dirname(save_json)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    with open(save_json, "w") as f:
        json.dump(to_jsonable(results), f, indent=4)
    print(f"Saved ensemble results to {save_json}")

    if run_diagnostics:
        member_results = []
        for path, metric in zip(args.ckpts, member_metrics):
            member_result = metric.get_results()
            summarize_split_metrics(member_result, opts)
            member_result["Checkpoint"] = path
            member_results.append(member_result)

        oracle_results = oracle_metrics.get_results()
        summarize_split_metrics(oracle_results, opts)
        diagnostics = {
            "Ensemble Mode": args.ensemble_mode,
            "Ensemble Members": args.ckpts,
            "Ensemble Weights": weights,
            "Class Weights": class_weights,
            "Member Results": member_results,
            "Oracle Results": oracle_results,
            "Disagreement": finalize_disagreement_stats(disagreement_stats),
        }
        diag_dir = os.path.dirname(args.save_diagnostics)
        if diag_dir:
            os.makedirs(diag_dir, exist_ok=True)
        with open(args.save_diagnostics, "w") as f:
            json.dump(to_jsonable(diagnostics), f, indent=4)
        print(f"Saved diagnostics to {args.save_diagnostics}")


if __name__ == "__main__":
    main()
