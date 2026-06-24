import hashlib
import json
import subprocess
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
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit or current_git_commit(),
        "model": opts.model,
        "requested_air_feature_source": requested_air_feature_source,
        "resolved_air_feature_source": resolved_air_feature_source,
        "dataset": opts.dataset,
        "task": opts.task,
        "setting": opts.setting,
        "curr_step": opts.curr_step,
        "subpath": opts.subpath,
        "base_subpath": opts.base_subpath,
        "base_checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "base_checkpoint_sha256": file_sha256(checkpoint_path) if checkpoint_path else None,
        "batch_size": opts.batch_size,
        "output_stride": opts.output_stride,
        "buffer": opts.buffer,
        "gamma": opts.gamma,
        "random_seed": opts.random_seed,
        "rhl_norm": opts.rhl_norm,
        "rhl_norm_eps": opts.rhl_norm_eps,
        "rhl_seed": opts.rhl_seed,
        "rhl_stats": opts.rhl_stats,
    }

    manifest_path = output_path / "run_manifest.json"
    temp_path = output_path / ".run_manifest.json.tmp"
    temp_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(manifest_path)
    return manifest_path
