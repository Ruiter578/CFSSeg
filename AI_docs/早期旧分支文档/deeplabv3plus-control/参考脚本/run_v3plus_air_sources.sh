#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

BASE_SUBPATH="${BASE_SUBPATH:-20260614_v3plus_voc15-5_seq_bs32-16}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-20260622_v3plus_air}"
BATCH_SIZE="${BATCH_SIZE:-16}"

for source in decoder decoder_stride8 aspp aspp_up; do
    subpath="${EXPERIMENT_PREFIX}_${source}"
    if [[ -e "checkpoints/${subpath}" ]]; then
        echo "Refusing to overwrite existing experiment: checkpoints/${subpath}" >&2
        exit 1
    fi

    echo "[$(date '+%F %T')] Starting ${source} -> ${subpath}"
    SUBPATH="$subpath" \
    BASE_SUBPATH="$BASE_SUBPATH" \
    BATCH_SIZE="$BATCH_SIZE" \
    AIR_FEATURE_SOURCE="$source" \
    bash ./run_v3plus_air.sh
done

echo "[$(date '+%F %T')] AIR feature source sweep finished"
