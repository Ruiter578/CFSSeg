#!/usr/bin/env python
"""Extract structured ACL metrics from existing CFSSeg 3D text logs."""

import argparse
import ast
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
import sys


_CODE3D_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE3D_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE3D_ROOT))

from utils.result_io import (  # noqa: E402
    build_parsed_acl_manifest,
    paper_uncertain_t,
    portable_path,
    portable_value,
    write_json,
)


TEST_RE = re.compile(
    r"===== \[Test\]: Accuracy: (?P<accuracy>[-+0-9.eE]+) \| "
    r"mIoU: (?P<miou>[-+0-9.eE]+) \| "
    r"Base mIoU: (?P<base>[-+0-9.eE]+) \| "
    r"Incre mIoU: (?P<incre>[-+0-9.eE]+)"
)
CLASS_IOU_RE = re.compile(r"Class_(?P<class_id>\d+) IoU: (?P<iou>[-+0-9.eE]+|nan)")
UNCERTAIN_RE = re.compile(
    r"Uncertain points ratio: (?P<ratio>[-+0-9.eE]+) "
    r"\(threshold: (?P<threshold>[-+0-9.eE]+)\)"
)
OPTION_RE = re.compile(r"^(?P<key>[A-Za-z0-9_]+): (?P<value>.*)$")


def _now_utc():
    return datetime.now(timezone.utc).isoformat()


def _parse_value(raw):
    raw = raw.strip()
    if raw == "None":
        return None
    if raw == "True":
        return True
    if raw == "False":
        return False
    try:
        return ast.literal_eval(raw)
    except Exception:
        return raw


def _parse_class_list(prefix, text):
    for line in text.splitlines():
        if line.startswith(prefix):
            _, raw = line.split(":", 1)
            try:
                return ast.literal_eval(raw.strip())
            except Exception:
                return None
    return None


def _parse_options(text):
    options = {}
    in_options = False
    for line in text.splitlines():
        if line.strip() == "------------ Options -------------":
            in_options = True
            options = {}
            continue
        if in_options and line.strip().startswith("-------------- End"):
            in_options = False
            continue
        if in_options:
            match = OPTION_RE.match(line.strip())
            if match:
                options[match.group("key")] = _parse_value(match.group("value"))
    return options


def _read_log_text(run_dir):
    run_dir = Path(run_dir)
    tmux_logs = sorted(run_dir.glob("tmux_launch_*.log"), key=lambda path: path.stat().st_mtime)
    if tmux_logs:
        return tmux_logs[-1].read_text(encoding="utf-8", errors="replace"), [portable_path(str(tmux_logs[-1]))]

    parts = []
    used = []
    for rel in ("base_model/log_ACL.txt", "incre_model/log_ACL.txt"):
        path = run_dir / rel
        if path.exists():
            parts.append(path.read_text(encoding="utf-8", errors="replace"))
            used.append(portable_path(str(path)))
    if not parts:
        raise FileNotFoundError("No ACL log found under %s" % run_dir)
    return "\n".join(parts), used


def _last_test_metrics(text):
    matches = list(TEST_RE.finditer(text))
    if not matches:
        raise ValueError("No final ACL test metric line found")
    match = matches[-1]
    return match, {
        "accuracy": float(match.group("accuracy")),
        "mIoU": float(match.group("miou")),
        "base_mIoU": float(match.group("base")),
        "incremental_mIoU": float(match.group("incre")),
    }


def _class_iou_before(text, end_index, num_classes):
    prefix = text[:end_index]
    entries = [
        (int(match.group("class_id")), float(match.group("iou")))
        for match in CLASS_IOU_RE.finditer(prefix)
    ]
    if num_classes == 0:
        return []
    if num_classes > 0:
        entries = entries[-num_classes:]
    entries = sorted(entries, key=lambda item: item[0])
    return [None if math.isnan(value) else value for _, value in entries]


def _uncertainty_summary(text):
    ratios = []
    thresholds = set()
    for match in UNCERTAIN_RE.finditer(text):
        ratios.append(float(match.group("ratio")))
        thresholds.add(float(match.group("threshold")))
    if not ratios:
        return None
    return {
        "count": len(ratios),
        "min": min(ratios),
        "max": max(ratios),
        "mean": sum(ratios) / len(ratios),
        "all_zero": all(value == 0 for value in ratios),
        "thresholds": sorted(thresholds),
    }


def extract(run_dir):
    run_dir = Path(run_dir)
    text, log_files = _read_log_text(run_dir)
    options = _parse_options(text)
    test_match, metrics = _last_test_metrics(text)

    base_classes = _parse_class_list("base_class", text)
    incre_classes = _parse_class_list("incre_class", text)
    test_classes = _parse_class_list("test classes", text)
    if test_classes is None and base_classes is not None and incre_classes is not None:
        test_classes = list(base_classes) + list(incre_classes)

    class_iou = _class_iou_before(text, test_match.start(), len(test_classes or []))
    dataset = str(options.get("dataset", "")).lower()
    uncertain_t = options.get("uncertain_t")
    paper_t = paper_uncertain_t(dataset)

    return {
        "schema_version": "cfsseg3d.acl_result.v1",
        "generated_at_utc": _now_utc(),
        "source": "extract_acl_results.py",
        "run_dir": portable_path(str(run_dir)),
        "log_files": log_files,
        "experiment": {
            "phase": options.get("phase"),
            "dataset": dataset or None,
            "cvfold": options.get("cvfold"),
            "tasks": options.get("tasks"),
            "uncertain_t": uncertain_t,
            "paper_uncertain_t": paper_t,
            "uses_paper_uncertain_t": (
                abs(float(uncertain_t) - paper_t) < 1e-12
                if uncertain_t is not None and paper_t is not None
                else None
            ),
            "data_path": portable_path(options.get("data_path")),
            "log_dir": portable_path(options.get("log_dir")),
        },
        "classes": {
            "test_classes": test_classes,
            "base_classes": base_classes,
            "incremental_classes": incre_classes,
        },
        "metrics": {
            **metrics,
            "class_iou": class_iou,
        },
        "uncertainty_ratios": _uncertainty_summary(text),
        "parsed_options": portable_value(options),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="ACL run directory under checkpoints_3d")
    parser.add_argument("--write", action="store_true", help="Write result_summary.json into the run directory")
    args = parser.parse_args()

    payload = extract(args.run_dir)
    if args.write:
        write_json(Path(args.run_dir) / "result_summary.json", payload)
        write_json(Path(args.run_dir) / "run_manifest.json", build_parsed_acl_manifest(payload))
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
