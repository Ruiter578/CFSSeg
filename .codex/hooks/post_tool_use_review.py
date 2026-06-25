#!/usr/bin/env python3
"""Best-effort command-output reviewer for SegACIL Codex hooks."""

from __future__ import annotations

import json
import sys
from typing import Any, List


def _flatten_strings(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        texts: List[str] = []
        for child in value.values():
            texts.extend(_flatten_strings(child))
        return texts
    if isinstance(value, list):
        texts = []
        for child in value:
            texts.extend(_flatten_strings(child))
        return texts
    return []


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except Exception as exc:
        print("[segacil-hook] cannot parse hook payload: {}".format(exc), file=sys.stderr)
        return 0

    text = "\n".join(_flatten_strings(payload))
    checks = {
        "SyntaxError": "Python syntax error detected; run py_compile before continuing",
        "invalid character": "invalid character detected; check for smart quotes or invisible chars",
        "U+201C": "smart quote detected in executable text",
        "CUDA out of memory": "CUDA OOM detected; do not interpret this as a method result",
        "CUDA error: 805": "CUDA MPS issue detected; verify CUDA_MPS_PIPE_DIRECTORY bypass",
        "MPS client failed": "MPS issue detected; use /home/linyichen/.mps_bypass on A100",
        "No such file or directory": "missing path detected; verify project-relative paths",
        "Checkpoint file": "checkpoint loading issue detected; verify BASE_SUBPATH and MODEL",
        "Requested AIR feature source": "AIR feature source mismatch detected; check run_manifest and checkpoint source",
        "size mismatch": "model/checkpoint shape mismatch detected; verify MODEL and step0 checkpoint architecture",
    }
    for needle, message in checks.items():
        if needle in text:
            print("[segacil-hook] warning: {}".format(message), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
