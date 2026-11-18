#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


DEFAULT_SOURCES = ("decoder", "decoder_stride8", "aspp", "aspp_up")
CLASS_IDS = {
    "pottedplant": "16",
    "train": "19",
    "tvmonitor": "20",
}


def latest_result(checkpoint_root, subpath):
    result_dir = checkpoint_root / subpath / "voc" / "15-5" / "sequential" / "step1"
    candidates = sorted(result_dir.glob("test_results_*.json"))
    if not candidates:
        return None
    with candidates[-1].open(encoding="utf-8") as handle:
        return json.load(handle)


def format_metric(value):
    return "-" if value is None else f"{value:.4f}"


def build_table(checkpoint_root, prefix, sources):
    header = (
        "| source | state | Mean IoU | 0-15 mIoU | 16-20 mIoU | "
        "pottedplant | tvmonitor | train |"
    )
    rows = [header, "|---|---|---:|---:|---:|---:|---:|---:|"]
    for source in sources:
        result = latest_result(checkpoint_root, f"{prefix}_{source}")
        if result is None:
            rows.append(f"| {source} | pending | - | - | - | - | - | - |")
            continue

        class_iou = result["Class IoU"]
        values = [
            result.get("Mean IoU"),
            result.get("0 to 15 mIoU"),
            result.get("16 to 20 mIoU"),
            class_iou.get(CLASS_IDS["pottedplant"]),
            class_iou.get(CLASS_IDS["tvmonitor"]),
            class_iou.get(CLASS_IDS["train"]),
        ]
        metrics = " | ".join(format_metric(value) for value in values)
        rows.append(f"| {source} | complete | {metrics} |")
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description="Summarize DeepLabV3+ AIR source experiments")
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--prefix", default="20260622_v3plus_air")
    parser.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    table = build_table(args.checkpoint_root, args.prefix, args.sources)
    print(table)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
