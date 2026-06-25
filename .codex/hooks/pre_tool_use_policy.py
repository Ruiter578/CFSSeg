#!/usr/bin/env python3
"""Best-effort pre-command reviewer for SegACIL Codex hooks."""

from __future__ import annotations

import json
import re
import sys
from typing import Any, List


def _read_payload() -> dict:
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return {}
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        print("[segacil-hook] cannot parse hook payload: {}".format(exc), file=sys.stderr)
        return {}


def _extract_command(payload: dict) -> str:
    candidates: List[Any] = [
        payload.get("command"),
        payload.get("input", {}).get("command") if isinstance(payload.get("input"), dict) else None,
        payload.get("tool_input", {}).get("cmd") if isinstance(payload.get("tool_input"), dict) else None,
        payload.get("tool_input", {}).get("command") if isinstance(payload.get("tool_input"), dict) else None,
    ]
    for item in candidates:
        if isinstance(item, str):
            return item
    return ""


def _looks_gpu_related(command: str) -> bool:
    tokens = (
        "train.py",
        "run.sh",
        "run_rhl_norm.sh",
        "run_trs.sh",
        "CUDA_VISIBLE_DEVICES",
        "deeplabv3",
        "SegACIL",
    )
    return any(token in command for token in tokens)


def main() -> int:
    command = _extract_command(_read_payload())
    if not command:
        return 0

    warnings: List[str] = []
    dangerous_patterns = [
        r"\brm\s+-rf\s+/(?:\s|$)",
        r"\bgit\s+reset\s+--hard\b",
        r"\bgit\s+checkout\s+--\b",
        r"\bgit\s+clean\s+-[fdx]+",
    ]
    if any(re.search(pattern, command) for pattern in dangerous_patterns):
        warnings.append("dangerous destructive command detected; user approval should be explicit")

    if "run_origin.sh" in command:
        warnings.append("run_origin.sh is collision-prone; prefer run.sh with explicit SUBPATH and BASE_SUBPATH")

    if _looks_gpu_related(command):
        if "nvidia-smi" not in command and ("train.py" in command or "run.sh" in command or "run_trs.sh" in command):
            warnings.append("GPU launch should be preceded by nvidia-smi; VRAM headroom matters more than util")
        if "run_trs.sh" not in command and "CUDA_MPS_PIPE_DIRECTORY" not in command:
            warnings.append("A100 GPU command should set CUDA_MPS_PIPE_DIRECTORY=/home/linyichen/.mps_bypass")
        if "run_trs.sh" not in command and "TMPDIR" not in command:
            warnings.append("A100 GPU command should set TMPDIR=/root/2TStorage/tmp")
        if "tmux new" in command and "tee" not in command and ".log" not in command:
            warnings.append("long tmux experiments should write a persistent log path")

    if "MODEL=deeplabv3plus" in command and "BASE_SUBPATH=20260606" in command:
        warnings.append("DeepLabV3+ should not use V3 BASE_SUBPATH=20260606; use a V3+ step0 checkpoint")

    if "--curr_step 1" in command and "--base_subpath" not in command and "BASE_SUBPATH" not in command:
        warnings.append("step1 runs should make BASE_SUBPATH / --base_subpath explicit")

    for warning in warnings:
        print("[segacil-hook] warning: {}".format(warning), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
