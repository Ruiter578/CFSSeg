#!/usr/bin/env bash
set -euo pipefail

# Preserve the RHL normalization recipe while sharing model/protocol plumbing.
export BUFFER="${BUFFER:-8196}"
export RHL_NORM="${RHL_NORM:-l2_sqrt}"
export RHL_STATS="${RHL_STATS:-1}"
export DEFAULT_BATCH_SIZE="${DEFAULT_BATCH_SIZE:-64}"
export GAMMA="${GAMMA:-1}"
export SUBPATH="${SUBPATH:-$(date +%Y%m%d)_rhl_${RHL_NORM}_g${GAMMA}}"
exec bash "$(dirname "$0")/run.sh" "$@"
