#!/usr/bin/env python
"""Summarize an adaptive pseudo-label run directory."""

import argparse
import json
from pathlib import Path


def read_json(path):
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def latest_result(output_dir):
    results = list(output_dir.glob("test_results_*.json"))
    return max(results, key=lambda path: path.stat().st_mtime_ns) if results else None


def latest_manifest(output_dir):
    manifests = list(output_dir.glob("run_manifest*.json"))
    return max(manifests, key=lambda path: path.stat().st_mtime_ns) if manifests else None


def metric_line(results):
    if not results:
        return "- 测试结果：暂未生成\n"
    mean_iou = results.get("Mean IoU")
    overall_acc = results.get("Overall Acc")
    lines = []
    if mean_iou is not None:
        lines.append(f"- Mean IoU: {float(mean_iou):.6f}")
    if overall_acc is not None:
        lines.append(f"- Overall Acc: {float(overall_acc):.6f}")
    for key, value in results.items():
        if key.endswith("mIoU") or key.endswith("mAcc"):
            lines.append(f"- {key}: {float(value):.6f}")
    return "\n".join(lines) + "\n"


def stats_line(stats):
    if not stats:
        return "- 伪标签统计：未启用或暂未落盘\n"
    return (
        f"- strategy: {stats.get('strategy')}\n"
        f"- weighting: {stats.get('weighting', 'none')}\n"
        f"- candidate_count: {stats.get('candidate_count')}\n"
        f"- accepted_count: {stats.get('accepted_count')}\n"
        f"- accepted_ratio: {float(stats.get('accepted_ratio', 0.0)):.6f}\n"
        f"- weight_mean: {stats.get('weight_mean')}\n"
        f"- weight_std: {stats.get('weight_std')}\n"
        f"- old_class_ids: {stats.get('old_class_ids')}\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", help="step output directory")
    parser.add_argument("--output", help="markdown output path")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        parser.error(f"output_dir does not exist or is not a directory: {output_dir}")

    manifest_path = latest_manifest(output_dir)
    manifest = read_json(manifest_path)
    stats = read_json(output_dir / "pseudo_label_stats.json")
    result_path = latest_result(output_dir)
    results = read_json(result_path)

    title = output_dir.as_posix()
    if manifest:
        title = (
            f"{manifest.get('task')} {manifest.get('setting')} "
            f"step{manifest.get('curr_step')} {manifest.get('pseudo_label_strategy')}"
        )

    report = [
        f"# Adaptive Pseudo-Label Run Summary: {title}",
        "",
        "## 路径",
        "",
        f"- output_dir: `{output_dir}`",
        f"- manifest_json: `{manifest_path}`" if manifest_path else "- manifest_json: `<none>`",
        f"- result_json: `{result_path}`" if result_path else "- result_json: `<none>`",
        "",
        "## 配置",
        "",
    ]
    if manifest:
        for key in [
            "model",
            "task",
            "setting",
            "curr_step",
            "batch_size",
            "buffer",
            "gamma",
            "random_seed",
            "use_pseudo_label",
            "pseudo_label_strategy",
            "pseudo_label_quantile",
            "pseudo_label_confidence",
            "pseudo_label_min_conf",
            "pseudo_label_max_conf",
            "pseudo_label_min_pixels",
            "pseudo_label_shrinkage",
            "pseudo_label_margin_min",
            "pseudo_label_weighting",
        ]:
            report.append(f"- {key}: `{manifest.get(key)}`")
    else:
        report.append("- run_manifest.json: `<missing>`")

    report.extend(["", "## 指标", "", metric_line(results), "## 伪标签统计", "", stats_line(stats)])
    output_text = "\n".join(report)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text + "\n", encoding="utf-8")
        print(f"wrote summary: {output_path}")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
