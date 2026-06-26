import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_commit(repo_root=None):
    root = repo_root or Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def current_git_dirty(repo_root=None):
    root = repo_root or Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return bool(result.stdout.strip())


def options_to_dict(opts):
    if is_dataclass(opts):
        return asdict(opts)
    if hasattr(opts, "__dict__"):
        return vars(opts)
    raise TypeError(f"Unsupported options object for run manifest: {type(opts)!r}")


def normalize_for_json(value):
    if is_dataclass(value):
        return normalize_for_json(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): normalize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [normalize_for_json(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return normalize_for_json(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def runtime_info():
    info = {
        "python": platform.python_version(),
        "pytorch": None,
        "cuda_available": None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch
    except ImportError:
        return info

    info["pytorch"] = getattr(torch, "__version__", None)
    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        info["cuda_available"] = None
    return info


def _manifest_identity(manifest):
    return {
        "args": manifest.get("args"),
        "air": manifest.get("air"),
        "command": manifest.get("command"),
        "git": manifest.get("git"),
        "resolved_paths": manifest.get("resolved_paths"),
    }


def _next_manifest_path(output_path):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    candidate = output_path / f"run_manifest_{timestamp}.json"
    suffix = 1
    while candidate.exists():
        candidate = output_path / f"run_manifest_{timestamp}_{suffix}.json"
        suffix += 1
    return candidate


def write_run_manifest(
    output_dir,
    opts,
    requested_air_feature_source,
    resolved_air_feature_source,
    base_checkpoint_path=None,
    git_commit=None,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(base_checkpoint_path) if base_checkpoint_path else None
    checkpoint_hash = file_sha256(checkpoint_path) if checkpoint_path else None
    commit = git_commit or current_git_commit()
    args = normalize_for_json(options_to_dict(opts))
    air = {
        "requested_feature_source": requested_air_feature_source,
        "resolved_feature_source": resolved_air_feature_source,
    }
    resolved_paths = {
        "output_dir": str(output_path),
        "base_checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "base_checkpoint_sha256": checkpoint_hash,
    }
    git = {
        "commit": commit,
        "dirty": current_git_dirty(),
    }
    manifest = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "command": list(sys.argv),
        "git": git,
        "runtime": runtime_info(),
        "resolved_paths": resolved_paths,
        "air": air,
        "args": args,
        # Backward-compatible flat keys for existing reports and ad-hoc readers.
        "git_commit": commit,
        "model": args.get("model"),
        "requested_air_feature_source": requested_air_feature_source,
        "resolved_air_feature_source": resolved_air_feature_source,
        "data_root": args.get("data_root"),
        "dataset": args.get("dataset"),
        "task": args.get("task"),
        "setting": args.get("setting"),
        "curr_step": args.get("curr_step"),
        "num_classes": args.get("num_classes"),
        "target_cls": args.get("target_cls"),
        "subpath": args.get("subpath"),
        "base_subpath": args.get("base_subpath"),
        "base_checkpoint_path": resolved_paths["base_checkpoint_path"],
        "base_checkpoint_sha256": resolved_paths["base_checkpoint_sha256"],
        "batch_size": args.get("batch_size"),
        "val_batch_size": args.get("val_batch_size"),
        "crop_size": args.get("crop_size"),
        "crop_val": args.get("crop_val"),
        "output_stride": args.get("output_stride"),
        "pretrained_backbone": args.get("pretrained_backbone"),
        "bn_freeze": args.get("bn_freeze"),
        "separable_conv": args.get("separable_conv"),
        "method": args.get("method"),
        "loss_type": args.get("loss_type"),
        "lr": args.get("lr"),
        "lr_policy": args.get("lr_policy"),
        "train_epoch": args.get("train_epoch"),
        "weight_decay": args.get("weight_decay"),
        "buffer": args.get("buffer"),
        "gamma": args.get("gamma"),
        "random_seed": args.get("random_seed"),
        "rhl_norm": args.get("rhl_norm"),
        "rhl_norm_eps": args.get("rhl_norm_eps"),
        "rhl_seed": args.get("rhl_seed"),
        "rhl_stats": args.get("rhl_stats"),
        "use_pseudo_label": args.get("use_pseudo_label"),
        "pseudo_label_confidence": args.get("pseudo_label_confidence"),
    }

    manifest_path = output_path / "run_manifest.json"
    if manifest_path.exists():
        try:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing_manifest = None
        if existing_manifest is not None:
            if _manifest_identity(existing_manifest) == _manifest_identity(manifest):
                return manifest_path
            manifest_path = _next_manifest_path(output_path)

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=output_path,
            prefix=f".{manifest_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temp_path.replace(manifest_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    return manifest_path


def safe_write_run_manifest(*args, **kwargs):
    try:
        return write_run_manifest(*args, **kwargs)
    except Exception as exc:
        print(f"[warning] failed to write run_manifest.json: {exc}")
        return None
