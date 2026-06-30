#!/usr/bin/env python
"""Summarize a pseudo-label experiment grid."""

import argparse
import csv
import json
from pathlib import Path


GRID_FIELDS = [
    "name",
    "subpath",
    "task",
    "setting",
    "strategy",
    "confidence",
    "quantile",
    "min_conf",
    "max_conf",
    "min_pixels",
    "shrinkage",
    "margin_min",
    "base_subpath",
    "skip_step0",
    "batch_size",
    "step0_batch_size",
    "buffer",
    "gamma",
    "random_seed",
    "model",
    "air_feature_source",
]

SUMMARY_FIELDS = [
    "name",
    "subpath",
    "task",
    "setting",
    "strategy",
    "confidence",
    "quantile",
    "min_conf",
    "margin_min",
    "status",
    "all_miou",
    "old_miou",
    "new_miou",
    "overall_acc",
    "mean_acc",
    "candidate_count",
    "accepted_count",
    "accepted_ratio",
    "delta_vs_off",
    "delta_vs_fixed07",
    "base_checkpoint_sha256",
    "teacher_sha256",
    "result_json",
    "stats_json",
]


def read_json(path):
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def latest_result(output_dir):
    results = list(output_dir.glob("test_results_*.json"))
    return max(results, key=lambda path: path.stat().st_mtime_ns) if results else None


def read_grid(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != GRID_FIELDS:
            raise ValueError(
                f"Unexpected grid header in {path}: {reader.fieldnames}; expected {GRID_FIELDS}"
            )
        return [
            {key: (value or "").strip() for key, value in row.items()}
            for row in reader
            if row.get("name") and not row["name"].lstrip().startswith("#")
        ]


def group_miou(results):
    old_key = None
    new_key = None
    for key in results:
        if key.startswith("0 to ") and key.endswith("mIoU"):
            old_key = key
        elif " to " in key and key.endswith("mIoU"):
            new_key = key
    return (
        results.get(old_key) if old_key else None,
        results.get(new_key) if new_key else None,
    )


def fmt(value):
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def summarize_row(row, checkpoints_root, dataset, off_baseline, fixed07_baseline):
    output_dir = (
        Path(checkpoints_root)
        / row["subpath"]
        / dataset
        / row["task"]
        / row["setting"]
        / "step1"
    )
    result_path = latest_result(output_dir) if output_dir.exists() else None
    manifest_path = output_dir / "run_manifest.json"
    stats_path = output_dir / "pseudo_label_stats.json"
    results = read_json(result_path)
    manifest = read_json(manifest_path)
    stats = read_json(stats_path)

    summary = {field: "" for field in SUMMARY_FIELDS}
    for key in [
        "name",
        "subpath",
        "task",
        "setting",
        "strategy",
        "confidence",
        "quantile",
        "min_conf",
        "margin_min",
    ]:
        summary[key] = row.get(key, "")

    if results is None:
        summary["status"] = "missing_result" if output_dir.exists() else "missing_output"
        return summary

    old_miou, new_miou = group_miou(results)
    all_miou = results.get("Mean IoU")
    summary.update(
        {
            "status": "done",
            "all_miou": all_miou,
            "old_miou": old_miou,
            "new_miou": new_miou,
            "overall_acc": results.get("Overall Acc"),
            "mean_acc": results.get("Mean Acc"),
            "result_json": str(result_path),
            "stats_json": str(stats_path) if stats is not None else "",
        }
    )
    if stats is not None:
        summary.update(
            {
                "candidate_count": stats.get("candidate_count"),
                "accepted_count": stats.get("accepted_count"),
                "accepted_ratio": stats.get("accepted_ratio"),
                "teacher_sha256": stats.get("teacher_sha256"),
            }
        )
    if manifest is not None:
        summary["base_checkpoint_sha256"] = manifest.get("base_checkpoint_sha256")
    if all_miou is not None:
        summary["delta_vs_off"] = float(all_miou) - float(off_baseline)
        summary["delta_vs_fixed07"] = float(all_miou) - float(fixed07_baseline)
    return summary


def write_csv(rows, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key)) for key in SUMMARY_FIELDS})


def write_markdown(rows, output_path, title, fixed07_baseline):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table_fields = [
        "name",
        "strategy",
        "confidence",
        "quantile",
        "status",
        "all_miou",
        "old_miou",
        "new_miou",
        "accepted_ratio",
        "delta_vs_fixed07",
        "subpath",
    ]
    lines = [
        f"# {title}",
        "",
        "| " + " | ".join(table_fields) + " |",
        "| " + " | ".join(["---"] * len(table_fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(field)) for field in table_fields) + " |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `all_miou` 等指标保持 JSON 原始小数尺度，`0.707383` 表示 70.7383%。",
            f"- `delta_vs_fixed07` 使用当前 `--fixed07-baseline`: `{fixed07_baseline}`。",
            "- `status=missing_output/missing_result` 表示实验尚未完成或目录不存在。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(rows, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", required=True, help="TSV grid produced for pseudo-label experiments")
    parser.add_argument("--dataset", default="voc")
    parser.add_argument("--checkpoints-root", default="checkpoints")
    parser.add_argument("--output-md")
    parser.add_argument("--output-csv")
    parser.add_argument("--output-json")
    parser.add_argument("--title", default="Pseudo-Label Grid Summary")
    parser.add_argument("--off-baseline", type=float, default=0.7030910671464596)
    parser.add_argument("--fixed07-baseline", type=float, default=0.7073832387362862)
    args = parser.parse_args()

    grid_rows = read_grid(args.grid)
    summary_rows = [
        summarize_row(
            row,
            args.checkpoints_root,
            args.dataset,
            args.off_baseline,
            args.fixed07_baseline,
        )
        for row in grid_rows
    ]

    if args.output_csv:
        write_csv(summary_rows, args.output_csv)
        print(f"wrote csv: {args.output_csv}")
    if args.output_md:
        write_markdown(summary_rows, args.output_md, args.title, args.fixed07_baseline)
        print(f"wrote markdown: {args.output_md}")
    if args.output_json:
        write_json(summary_rows, args.output_json)
        print(f"wrote json: {args.output_json}")
    if not args.output_csv and not args.output_md and not args.output_json:
        for row in summary_rows:
            print(json.dumps(row, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
