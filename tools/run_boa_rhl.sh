#!/usr/bin/env bash
set -euo pipefail

# P1 BOA-RHL runner.
# Runs the four first-round single-model step1 experiments:
# BOA-0 gaussian/legacy, BOA-1 orthogonal/legacy,
# BOA-2 orthogonal_antithetic/legacy, BOA-3 orthogonal_antithetic/kaiming.

export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/home/linyichen/.mps_bypass}"
export TMPDIR="${TMPDIR:-/root/2TStorage/tmp}"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$TMPDIR"

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/root/2TStorage/lyc/SegACIL/data_root/VOC2012}"
MODEL="${MODEL:-deeplabv3_resnet101}"
LR="${LR:-0.01}"
LOSS_TYPE="${LOSS_TYPE:-bce_loss}"
DATASET="${DATASET:-voc}"
TASK="${TASK:-15-5}"
LR_POLICY="${LR_POLICY:-poly}"
METHOD="${METHOD:-acil}"
SETTING="${SETTING:-sequential}"
TRAIN_EPOCH="${TRAIN_EPOCH:-50}"
BUFFER="${BUFFER:-8196}"
OUTPUT_STRIDE="${OUTPUT_STRIDE:-8}"
GAMMA="${GAMMA:-1}"
RHL_NORM="${RHL_NORM:-none}"
RHL_NORM_EPS="${RHL_NORM_EPS:-1e-6}"
RHL_SEED="${RHL_SEED:-1}"
RHL_STATS="${RHL_STATS:-1}"
BASE_SUBPATH="${BASE_SUBPATH:-20260606}"
DEFAULT_BATCH_SIZE="${DEFAULT_BATCH_SIZE:-32}"
FALLBACK_BATCH_SIZE="${FALLBACK_BATCH_SIZE:-16}"
RUN_PREFIX="${RUN_PREFIX:-$(date +%Y%m%d_%H%M%S)_boa_rhl}"
CASE_FILTER="${CASE_FILTER:-all}"
LOG_DIR="${LOG_DIR:-logs/boa_rhl/${RUN_PREFIX}}"
PRETRAINED_BACKBONE="${PRETRAINED_BACKBONE:-1}"

mkdir -p "$LOG_DIR"

PRETRAINED_ARGS=()
if [[ "$PRETRAINED_BACKBONE" == "1" ]]; then
    PRETRAINED_ARGS=(--pretrained_backbone)
fi

RHL_STATS_ARGS=()
if [[ "$RHL_STATS" == "1" ]]; then
    RHL_STATS_ARGS=(--rhl_stats)
fi

should_run_case() {
    local case_name="$1"
    [[ "$CASE_FILTER" == "all" || " ${CASE_FILTER} " == *" ${case_name} "* ]]
}

run_case() {
    local case_name="$1"
    local rhl_init="$2"
    local rhl_scale_mode="$3"
    local batch_size="$4"
    local subpath="${RUN_PREFIX}_${case_name}_${rhl_init}_${rhl_scale_mode}_seed${RHL_SEED}_bs${batch_size}"
    local log_path="${LOG_DIR}/${case_name}_bs${batch_size}.log"
    local -a cmd=(
        "$PYTHON" train.py
        --data_root "$DATA_ROOT"
        --model "$MODEL"
        --lr "$LR"
        --batch_size "$batch_size"
        --loss_type "$LOSS_TYPE"
        --dataset "$DATASET"
        --task "$TASK"
        --lr_policy "$LR_POLICY"
        --curr_step 1
        --subpath "$subpath"
        --base_subpath "$BASE_SUBPATH"
        --method "$METHOD"
        --setting "$SETTING"
        "${PRETRAINED_ARGS[@]}"
        --crop_val
        --train_epoch "$TRAIN_EPOCH"
        --gamma "$GAMMA"
        --buffer "$BUFFER"
        --rhl_norm "$RHL_NORM"
        --rhl_norm_eps "$RHL_NORM_EPS"
        --rhl_seed "$RHL_SEED"
        --rhl_init "$rhl_init"
        --rhl_scale_mode "$rhl_scale_mode"
        "${RHL_STATS_ARGS[@]}"
        --output_stride "$OUTPUT_STRIDE"
    )

    echo "Running ${case_name}: init=${rhl_init}, scale=${rhl_scale_mode}, batch=${batch_size}, subpath=${subpath}"
    printf "%q " "${cmd[@]}" | tee "${log_path}.cmd"
    printf "\n" | tee -a "${log_path}.cmd"

    set +e
    "${cmd[@]}" 2>&1 | tee "$log_path"
    local status=${PIPESTATUS[0]}
    set -e
    return "$status"
}

run_case_with_oom_fallback() {
    local case_name="$1"
    local rhl_init="$2"
    local rhl_scale_mode="$3"
    local primary_log="${LOG_DIR}/${case_name}_bs${DEFAULT_BATCH_SIZE}.log"

    if run_case "$case_name" "$rhl_init" "$rhl_scale_mode" "$DEFAULT_BATCH_SIZE"; then
        return 0
    fi

    if grep -qi "out of memory\\|CUDA out of memory" "$primary_log"; then
        echo "${case_name} hit OOM at batch ${DEFAULT_BATCH_SIZE}; retrying with batch ${FALLBACK_BATCH_SIZE}."
        run_case "$case_name" "$rhl_init" "$rhl_scale_mode" "$FALLBACK_BATCH_SIZE"
        return $?
    fi

    echo "${case_name} failed for a non-OOM reason; see ${primary_log}."
    return 1
}

CASES=(
    "BOA-0 gaussian legacy"
    "BOA-1 orthogonal legacy"
    "BOA-2 orthogonal_antithetic legacy"
    "BOA-3 orthogonal_antithetic kaiming"
)

for case_spec in "${CASES[@]}"; do
    read -r case_name rhl_init rhl_scale_mode <<< "$case_spec"
    if should_run_case "$case_name"; then
        run_case_with_oom_fallback "$case_name" "$rhl_init" "$rhl_scale_mode"
    else
        echo "Skipping ${case_name} due to CASE_FILTER=${CASE_FILTER}"
    fi
done

echo "BOA-RHL runner finished. Logs: ${LOG_DIR}"
