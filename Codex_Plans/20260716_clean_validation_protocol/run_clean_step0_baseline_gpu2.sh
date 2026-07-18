#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

RUN_ID="${RUN_ID:-20260716_clean_validation_step0_decoder_b8224_g1_gpu2}"
PRIMARY_ROOT="${PRIMARY_ROOT:-$PWD}"
LOG_ROOT="${LOG_ROOT:-${PRIMARY_ROOT}/logs/${RUN_ID}}"
DATA_ROOT="${DATA_ROOT:-$PWD/data_root/VOC2012}"
HOLDOUT_LIST="${HOLDOUT_LIST:-$PWD/Codex_Plans/20260716_clean_validation_protocol/voc15_5_tuning_holdout_seed20260716.txt}"
HOLDOUT_METADATA="${HOLDOUT_METADATA:-$PWD/Codex_Plans/20260716_clean_validation_protocol/voc15_5_tuning_holdout_seed20260716.json}"
EXPECTED_HEAD="${EXPECTED_HEAD:-2b13e2f6f440a27cdcc769c2ce001aadc0bdd0d3}"

MODEL="${MODEL:-deeplabv3_resnet101}"
TRAIN_EPOCH="${TRAIN_EPOCH:-50}"
LR="${LR:-0.01}"
BACKBONE_LR="${BACKBONE_LR:-0.001}"
CLASSIFIER_LR="${CLASSIFIER_LR:-0.01}"
RANDOM_SEED="${RANDOM_SEED:-1}"
BUFFER="${BUFFER:-8224}"
GAMMA="${GAMMA:-1}"
RHL_NORM="${RHL_NORM:-none}"
RHL_SEED="${RHL_SEED:-2}"

mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/queue_master.log") 2>&1

if [[ "$(git rev-parse HEAD)" != "$EXPECTED_HEAD" ]]; then
    echo "[ERROR] Expected code base $EXPECTED_HEAD, got $(git rev-parse HEAD)." >&2
    exit 1
fi
if ! git diff --quiet || ! git diff --cached --quiet || [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
    echo "[ERROR] Refusing to launch from a modified or untracked worktree." >&2
    exit 1
fi
if [[ ! -d "$DATA_ROOT" || ! -f "$HOLDOUT_LIST" || ! -f "$HOLDOUT_METADATA" ]]; then
    echo "[ERROR] Missing data root, holdout list, or holdout metadata." >&2
    exit 1
fi
HOLDOUT_HASH="$(sha256sum "$HOLDOUT_LIST" | awk '{print $1}')"
EXPECTED_HOLDOUT_HASH="$(jq -r '.holdout_sha256 // empty' "$HOLDOUT_METADATA")"
if [[ -z "$EXPECTED_HOLDOUT_HASH" || "$EXPECTED_HOLDOUT_HASH" != "$HOLDOUT_HASH" ]]; then
    echo "[ERROR] Holdout list hash does not match protocol metadata." >&2
    exit 1
fi
prepare_output_path() {
    local subpath="$1"
    local local_path="checkpoints/${subpath}"
    local primary_root_real
    local primary_path

    primary_root_real="$(mkdir -p "$PRIMARY_ROOT" && cd "$PRIMARY_ROOT" && pwd -P)"
    if [[ "$primary_root_real" == "$(pwd -P)" ]]; then
        if [[ -e "$local_path" || -L "$local_path" ]]; then
            echo "[ERROR] Refusing to reuse existing output path: $local_path" >&2
            exit 1
        fi
        mkdir -p "$local_path"
        return
    fi
    primary_path="${primary_root_real}/checkpoints/${subpath}"

    if [[ -e "$local_path" || -L "$local_path" ]]; then
        echo "[ERROR] Refusing to reuse existing local output path: $local_path" >&2
        exit 1
    fi
    if [[ -e "$primary_path" || -L "$primary_path" ]]; then
        echo "[ERROR] Refusing to reuse existing primary output path: $primary_path" >&2
        exit 1
    fi
    mkdir -p "$primary_path"
    mkdir -p "$(dirname "$local_path")"
    ln -s "$primary_path" "$local_path"
}

run_attempt() {
    local batch_size="$1"
    local subpath="${RUN_ID}_bs${batch_size}"
    local log_path="$LOG_ROOT/${subpath}.log"

    prepare_output_path "$subpath"
    echo "[RUN] subpath=$subpath physical_gpu=2 batch_size=$batch_size"
    set +e
    env \
        CUDA_VISIBLE_DEVICES=2 \
        DATA_ROOT="$DATA_ROOT" \
        MODEL="$MODEL" \
        AIR_FEATURE_SOURCE=decoder \
        SUBPATH="$subpath" \
        START_STEP=0 \
        END_STEP=0 \
        SPECIAL_BATCH_SIZE="$batch_size" \
        DEFAULT_BATCH_SIZE=32 \
        TRAIN_EPOCH="$TRAIN_EPOCH" \
        LR="$LR" \
        BACKBONE_LR="$BACKBONE_LR" \
        CLASSIFIER_LR="$CLASSIFIER_LR" \
        RANDOM_SEED="$RANDOM_SEED" \
        BUFFER="$BUFFER" \
        GAMMA="$GAMMA" \
        RHL_NORM="$RHL_NORM" \
        RHL_SEED="$RHL_SEED" \
        TRAIN_EXCLUDE_LIST="$HOLDOUT_LIST" \
        VALIDATION_LIST="$HOLDOUT_LIST" \
        EVALUATION_MODE=val \
        bash run.sh 2>&1 | tee "$log_path"
    local pipeline_status=("${PIPESTATUS[@]}")
    set -e
    local status="${pipeline_status[0]}"
    local tee_status="${pipeline_status[1]}"
    if (( tee_status != 0 )); then
        echo "[ERROR] Failed to write attempt log: $log_path" >&2
        return "$tee_status"
    fi
    echo "[EXIT] subpath=$subpath status=$status"
    return "$status"
}

for batch_size in 64 32 16; do
    if run_attempt "$batch_size"; then
        echo "[DONE] Clean step0 baseline completed at batch_size=$batch_size."
        exit 0
    fi
    if ! grep -Eiq 'CUDA out of memory|CUDA error: out of memory' "$LOG_ROOT/${RUN_ID}_bs${batch_size}.log"; then
        echo "[ERROR] Batch-$batch_size attempt failed for a non-OOM reason; no fallback was started." >&2
        exit 1
    fi
    if [[ "$batch_size" != "16" ]]; then
        echo "[OOM] Batch-$batch_size exhausted GPU memory. Retrying with batch_size=$((batch_size / 2))."
    fi
done

echo "[ERROR] Batch sizes 64, 32, and 16 all exhausted GPU memory." >&2
exit 1
