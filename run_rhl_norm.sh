#!/usr/bin/env bash
set -euo pipefail

# Dedicated entrypoint for RHL-normalization experiments.
# Keep the original run.sh stable; future RHL-specific changes should be made
# here so baseline reproduction scripts and RHL ablations do not interfere.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# The shared /tmp/nvidia-mps directory can be owned by another user on this
# server.  Use a per-user MPS pipe/log directory by default so CUDA initialization
# does not fail with "MPS client failed to connect to the MPS control daemon".
USE_LOCAL_MPS="${USE_LOCAL_MPS:-1}"
if [[ "$USE_LOCAL_MPS" == "1" ]]; then
    export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps-${USER:-$(id -un)}}"
    export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-mps-log-${USER:-$(id -un)}}"
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    if ! pgrep -u "$(id -u)" -f "nvidia-cuda-mps-control -d" >/dev/null 2>&1; then
        nvidia-cuda-mps-control -d >/dev/null 2>&1 || true
    fi
fi

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
PRETRAINED_BACKBONE="${PRETRAINED_BACKBONE:---pretrained_backbone}"
BUFFER="${BUFFER:-8196}"
OUTPUT_STRIDE="${OUTPUT_STRIDE:-8}"

# RHL ablation controls.  The main candidate is l2_sqrt with gamma=1; set
# RHL_NORM=none to run the new-code baseline under the same script.
RHL_NORM="${RHL_NORM:-l2_sqrt}"
RHL_NORM_EPS="${RHL_NORM_EPS:-1e-6}"
RHL_STATS="${RHL_STATS:-1}"
GAMMA="${GAMMA:-1}"

# 15-5 contains only step0 and step1.  RHL experiments should start from step1
# and load an existing step0 DeepLab checkpoint through BASE_SUBPATH.
START_STEP="${START_STEP:-1}"
END_STEP="${END_STEP:-1}"
STEP_INCREMENT="${STEP_INCREMENT:-1}"

# 20260606 currently contains the named step0 checkpoint required by the
# trainer's step1 load path.  Override BASE_SUBPATH when testing another step0.
BASE_SUBPATH="${BASE_SUBPATH:-20260606}"

# Include RHL mode and gamma in the default output directory to reduce accidental
# result mixing.  Override SUBPATH explicitly for planned ablations.
SUBPATH="${SUBPATH:-$(date +%Y%m%d)_rhl_${RHL_NORM}_g${GAMMA}}"

DEFAULT_BATCH_SIZE="${DEFAULT_BATCH_SIZE:-64}"
SPECIAL_BATCH_SIZE="${SPECIAL_BATCH_SIZE:-32}"

BASE_SUBPATH_ARG=()
if [[ -n "$BASE_SUBPATH" ]]; then
    BASE_SUBPATH_ARG=(--base_subpath "$BASE_SUBPATH")
fi

RHL_STATS_ARG=()
if [[ "$RHL_STATS" == "1" ]]; then
    RHL_STATS_ARG=(--rhl_stats)
fi

echo "RHL experiment configuration:"
echo "  task=${TASK}, steps=${START_STEP}-${END_STEP}, setting=${SETTING}"
echo "  subpath=${SUBPATH}, base_subpath=${BASE_SUBPATH}"
echo "  model=${MODEL}, buffer=${BUFFER}, gamma=${GAMMA}"
echo "  rhl_norm=${RHL_NORM}, rhl_norm_eps=${RHL_NORM_EPS}, rhl_stats=${RHL_STATS}"

for ((CURR_STEP=START_STEP; CURR_STEP<=END_STEP; CURR_STEP+=STEP_INCREMENT))
do
    if [[ "$CURR_STEP" -eq 0 ]]; then
        CURR_BATCH_SIZE="$SPECIAL_BATCH_SIZE"
    else
        CURR_BATCH_SIZE="$DEFAULT_BATCH_SIZE"
    fi

    echo "Running RHL training for step ${CURR_STEP} with batch size ${CURR_BATCH_SIZE}..."
    python train.py \
        --data_root "$DATA_ROOT" \
        --model "$MODEL" \
        --lr "$LR" \
        --batch_size "$CURR_BATCH_SIZE" \
        --loss_type "$LOSS_TYPE" \
        --dataset "$DATASET" \
        --task "$TASK" \
        --lr_policy "$LR_POLICY" \
        --curr_step "$CURR_STEP" \
        --subpath "$SUBPATH" \
        "${BASE_SUBPATH_ARG[@]}" \
        --method "$METHOD" \
        --setting "$SETTING" \
        $PRETRAINED_BACKBONE \
        --crop_val \
        --train_epoch "$TRAIN_EPOCH" \
        --gamma "$GAMMA" \
        --buffer "$BUFFER" \
        --rhl_norm "$RHL_NORM" \
        --rhl_norm_eps "$RHL_NORM_EPS" \
        "${RHL_STATS_ARG[@]}" \
        --output_stride "$OUTPUT_STRIDE"
done
