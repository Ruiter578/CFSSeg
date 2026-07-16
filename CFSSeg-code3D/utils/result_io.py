"""Structured result writers for CFSSeg 3D ACL runs."""

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


_PAPER_UNCERTAIN_T = {
    "s3dis": 0.0035,
    "scannet": 0.001,
}


def _now_utc():
    return datetime.now(timezone.utc).isoformat()


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _portable_path(value):
    if not isinstance(value, str):
        return value
    expanded = os.path.expanduser(os.path.expandvars(value))
    if not os.path.isabs(expanded):
        return value
    try:
        return str(Path(expanded).resolve().relative_to(_repo_root()))
    except ValueError:
        return "<external>/" + Path(expanded).name


def _portable_value(value):
    if isinstance(value, dict):
        return {str(k): _portable_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable_value(v) for v in value]
    return _portable_path(value)


def portable_path(value):
    return _portable_path(value)


def portable_value(value):
    return _portable_value(value)


def _repo_root():
    return Path(__file__).resolve().parents[2]


def _git_value(args):
    try:
        return subprocess.check_output(
            ["git", "-C", str(_repo_root()), *args],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def git_metadata():
    dirty_status = _git_value(["status", "--short"])
    return {
        "branch": _git_value(["branch", "--show-current"]),
        "commit": _git_value(["rev-parse", "HEAD"]),
        "is_dirty": None if dirty_status is None else bool(dirty_status),
    }


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".%s." % path.name,
            suffix=".tmp",
            delete=False,
        ) as f:
            tmp_path = Path(f.name)
            json.dump(_to_jsonable(payload), f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()
    return str(path)


def paper_uncertain_t(dataset):
    return _PAPER_UNCERTAIN_T.get(str(dataset).lower())


def build_acl_manifest(args, base_classes, incre_classes, total_step):
    dataset = str(args.dataset).lower()
    paper_t = paper_uncertain_t(dataset)
    uncertain_t = float(args.uncertain_t)
    return {
        "schema_version": "cfsseg3d.acl_manifest.v1",
        "generated_at_utc": _now_utc(),
        "source": "train_ACL.py",
        "cwd": _portable_path(os.getcwd()),
        "command": _portable_value(sys.argv),
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "CFSSeg3D_DATA_ROOT": _portable_path(os.environ.get("CFSSeg3D_DATA_ROOT")),
            "CFSSeg3D_OUTPUT_ROOT": _portable_path(os.environ.get("CFSSeg3D_OUTPUT_ROOT")),
        },
        "git": git_metadata(),
        "args": _portable_value(vars(args).copy()),
        "class_split": {
            "base_classes": list(base_classes),
            "incremental_classes": list(incre_classes),
            "num_base_classes": len(base_classes),
            "num_incremental_classes": len(incre_classes),
            "total_steps": int(total_step),
        },
        "paper_alignment": {
            "paper_uncertain_t": paper_t,
            "uses_paper_uncertain_t": (
                abs(uncertain_t - paper_t) < 1e-12 if paper_t is not None else None
            ),
            "paper_buffer_size": 5000,
            "paper_gamma": 1,
        },
    }


def write_acl_manifest(log_dir, args, base_classes, incre_classes, total_step):
    manifest = build_acl_manifest(args, base_classes, incre_classes, total_step)
    return write_json(Path(log_dir) / "run_manifest.json", manifest)


def build_acl_result_summary(
    args,
    step,
    test_classes,
    base_classes,
    incre_classes,
    accuracy,
    mIoU,
    iou_perclass,
    base_mIoU,
    incre_mIoU,
):
    dataset = str(args.dataset).lower()
    paper_t = paper_uncertain_t(dataset)
    return {
        "schema_version": "cfsseg3d.acl_result.v1",
        "generated_at_utc": _now_utc(),
        "source": "train_ACL.py",
        "experiment": {
            "phase": args.phase,
            "dataset": dataset,
            "cvfold": int(args.cvfold),
            "tasks": args.tasks,
            "step": int(step),
            "uncertain_t": float(args.uncertain_t),
            "paper_uncertain_t": paper_t,
            "uses_paper_uncertain_t": (
                abs(float(args.uncertain_t) - paper_t) < 1e-12 if paper_t is not None else None
            ),
            "data_path": _portable_path(args.data_path),
            "log_dir": _portable_path(args.log_dir),
        },
        "classes": {
            "test_classes": list(test_classes),
            "base_classes": list(base_classes),
            "incremental_classes": list(incre_classes),
        },
        "metrics": {
            "accuracy": float(accuracy),
            "mIoU": float(mIoU),
            "base_mIoU": float(base_mIoU) if base_mIoU is not None else None,
            "incremental_mIoU": float(incre_mIoU) if incre_mIoU is not None else None,
            "class_iou": [float(v) for v in iou_perclass],
        },
    }


def write_acl_result_summary(
    log_dir,
    args,
    step,
    test_classes,
    base_classes,
    incre_classes,
    accuracy,
    mIoU,
    iou_perclass,
    base_mIoU,
    incre_mIoU,
):
    summary = build_acl_result_summary(
        args,
        step,
        test_classes,
        base_classes,
        incre_classes,
        accuracy,
        mIoU,
        iou_perclass,
        base_mIoU,
        incre_mIoU,
    )
    return write_json(Path(log_dir) / "result_summary.json", summary)


def build_parsed_acl_manifest(payload):
    options = payload.get("parsed_options", {})
    classes = payload.get("classes", {})
    base_classes = classes.get("base_classes") or []
    incre_classes = classes.get("incremental_classes") or []
    experiment = payload.get("experiment", {})
    return {
        "schema_version": "cfsseg3d.acl_manifest.v1",
        "generated_at_utc": _now_utc(),
        "source": "extract_acl_results.py",
        "run_dir": portable_path(payload.get("run_dir")),
        "log_files": portable_value(payload.get("log_files", [])),
        "args": portable_value(options),
        "class_split": {
            "base_classes": base_classes,
            "incremental_classes": incre_classes,
            "num_base_classes": len(base_classes),
            "num_incremental_classes": len(incre_classes),
        },
        "paper_alignment": {
            "paper_uncertain_t": experiment.get("paper_uncertain_t"),
            "uses_paper_uncertain_t": experiment.get("uses_paper_uncertain_t"),
            "paper_buffer_size": 5000,
            "paper_gamma": 1,
        },
    }
