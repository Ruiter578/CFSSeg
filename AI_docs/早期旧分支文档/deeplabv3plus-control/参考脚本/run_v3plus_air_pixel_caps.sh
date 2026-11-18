#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

BASE_SUBPATH="${BASE_SUBPATH:-20260614_v3plus_voc15-5_seq_bs32-16}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-20260623_v3plus_air_aspp_up}"
BATCH_SIZE="${BATCH_SIZE:-16}"

for cap in 4096 8192; do
    subpath="${EXPERIMENT_PREFIX}_cap${cap}"
    if [[ -e "checkpoints/${subpath}" ]]; then
        echo "Refusing to overwrite existing experiment: checkpoints/${subpath}" >&2
        exit 1
    fi

    echo "[$(date '+%F %T')] Starting aspp_up class_cap=${cap} -> ${subpath}"
    SUBPATH="$subpath" \
    BASE_SUBPATH="$BASE_SUBPATH" \
    BATCH_SIZE="$BATCH_SIZE" \
    AIR_FEATURE_SOURCE=aspp_up \
    AIR_PIXEL_BALANCE=class_cap \
    AIR_MAX_PIXELS_PER_CLASS="$cap" \
    bash ./run_v3plus_air.sh
done

echo "[$(date '+%F %T')] AIR pixel-cap sweep finished"
