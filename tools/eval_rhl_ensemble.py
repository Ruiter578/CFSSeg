#!/usr/bin/env python
"""Evaluate an ensemble of AIR/RHL checkpoints by averaging pixel probabilities.

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

    # AIR 输出是 [B, H, W, C]，普通分割模型通常是 [B, C, H, W]。
    # 同时兼容两种形状，便于以后评估 RHL-SE 和普通对照模型。
    if probs.ndim != 4:
        raise ValueError(f"Expected 4D logits/probabilities, got shape {tuple(probs.shape)}")
    if probs.shape[-1] == n_classes:
        probs = probs.permute(0, 3, 1, 2)
    elif probs.shape[1] != n_classes:
        raise ValueError(
            f"Cannot infer class dimension for shape {tuple(probs.shape)} "
            f"with n_classes={n_classes}"
        )

    return F.interpolate(probs, labels_shape, mode="bilinear")


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


def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    opts = build_opts(args)
    _, val_loader, test_loader = init_dataloader(opts)
    loader = test_loader if args.mode == "test" else val_loader
    metrics = StreamSegMetrics(opts.num_classes, dataset=opts.dataset)
    n_classes = sum(opts.num_classes)

    models = [load_model_cpu(path) for path in args.ckpts]
    if args.keep_models_on_gpu:
        # 显存足够时可一次性常驻 GPU，速度更快；默认不用，优先保证不 OOM。
        models = [model.to(device).eval() for model in models]

    metrics.reset()
    with torch.no_grad():
        for batch_idx, (images, labels, _) in enumerate(tqdm(loader)):
            if args.max_batches >= 0 and batch_idx >= args.max_batches:
                break

            images = images.to(device, dtype=torch.float32, non_blocking=True)
            labels_device = labels.to(device, dtype=torch.long, non_blocking=True)
            prob_sum = None

            for model in models:
                if not args.keep_models_on_gpu:
                    # 低显存模式：每次只把一个成员放上 GPU，算完立即搬回 CPU。
                    model = model.to(device).eval()
                logits = model(images)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                probs = logits_to_prob(
                    logits,
                    labels_device.shape[-2:],
                    opts.loss_type,
                    n_classes,
                )
                prob_sum = probs if prob_sum is None else prob_sum + probs
                if not args.keep_models_on_gpu:
                    model.cpu()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

            # 像素级概率平均：这是 RHL-SE 最终形成集成预测的地方。
            prob_ens = prob_sum / len(models)
            preds = prob_ens.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.numpy()
            metrics.update(targets, preds)

    results = metrics.get_results()
    first_cls = len(get_tasks(opts.dataset, opts.task, 0))
    class_iou = list(results["Class IoU"].values())
    class_acc = list(results["Class Acc"].values())
    results[f"0 to {first_cls - 1} mIoU"] = np.mean(class_iou[:first_cls])
    results[f"{first_cls} to {len(class_iou) - 1} mIoU"] = np.mean(class_iou[first_cls:])
    results[f"0 to {first_cls - 1} mAcc"] = np.mean(class_acc[:first_cls])
    results[f"{first_cls} to {len(class_iou) - 1} mAcc"] = np.mean(class_acc[first_cls:])
    results["Ensemble Members"] = args.ckpts
    results["Ensemble Size"] = len(args.ckpts)

    print(metrics.to_str(results))
    print(f"...from 0 to {first_cls - 1} : ensemble/test_before_mIoU : {results[f'0 to {first_cls - 1} mIoU']:.6f}")
    print(f"...from {first_cls} to {len(class_iou) - 1} ensemble/test_after_mIoU : {results[f'{first_cls} to {len(class_iou) - 1} mIoU']:.6f}")

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


if __name__ == "__main__":
    main()
