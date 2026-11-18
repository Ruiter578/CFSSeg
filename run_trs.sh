#!/usr/bin/env bash
set -euo pipefail

# TRS keeps its default GPU; all experiment options use the unified runner.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
exec bash "$(dirname "$0")/run.sh" "$@"
