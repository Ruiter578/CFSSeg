#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/home/linyichen/.mps_bypass}"
export TMPDIR="${TMPDIR:-/root/2TStorage/tmp}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/root/2TStorage/lyc/SegACIL/data_root/VOC2012}"
MODEL="${MODEL:-deeplabv3_resnet101}"
AIR_FEATURE_SOURCE="${AIR_FEATURE_SOURCE:-auto}"
LR="${LR:-0.01}"
LOSS_TYPE="${LOSS_TYPE:-bce_loss}"
DATASET="${DATASET:-voc}"
TASK="${TASK:-15-5}"
LR_POLICY="${LR_POLICY:-poly}"
SUBPATH="${SUBPATH:-$(date +%Y%m%d)_${MODEL}}"
# The default checkpoint belongs to the default V3 model. Override BASE_SUBPATH
# whenever MODEL points at a different architecture.
BASE_SUBPATH="${BASE_SUBPATH:-20260606}"
METHOD="${METHOD:-acil}"
SETTING="${SETTING:-sequential}"
TRAIN_EPOCH="${TRAIN_EPOCH:-50}"
CKPT="${CKPT:-}"
CURR_ITRS="${CURR_ITRS:-0}"
PRETRAINED_BACKBONE="${PRETRAINED_BACKBONE:---pretrained_backbone}"
BUFFER="${BUFFER:-8192}"
OUTPUT_STRIDE="${OUTPUT_STRIDE:-8}"
GAMMA="${GAMMA:-1}"
RANDOM_SEED="${RANDOM_SEED:-1}"
RHL_NORM="${RHL_NORM:-none}"
RHL_NORM_EPS="${RHL_NORM_EPS:-1e-6}"
RHL_SEED="${RHL_SEED:--1}"
RHL_STATS="${RHL_STATS:-0}"

DEFAULT_BATCH_SIZE="${DEFAULT_BATCH_SIZE:-32}"
SPECIAL_BATCH_SIZE="${SPECIAL_BATCH_SIZE:-32}"

# Loop through steps
START_STEP="${START_STEP:-1}"
END_STEP="${END_STEP:-1}"
STEP_INCREMENT="${STEP_INCREMENT:-1}"

BASE_SUBPATH_ARG=()
if [[ -n "$BASE_SUBPATH" ]]; then
    BASE_SUBPATH_ARG=(--base_subpath "$BASE_SUBPATH")
fi

RHL_STATS_ARG=()
if [[ "$RHL_STATS" == "1" ]]; then
    RHL_STATS_ARG=(--rhl_stats)
fi

echo "SegACIL experiment configuration:"
echo "  model=${MODEL}, air_feature_source=${AIR_FEATURE_SOURCE}"
echo "  task=${TASK}, setting=${SETTING}, steps=${START_STEP}-${END_STEP}"
echo "  subpath=${SUBPATH}, base_subpath=${BASE_SUBPATH}"
echo "  buffer=${BUFFER}, gamma=${GAMMA}, random_seed=${RANDOM_SEED}"
echo "  rhl_norm=${RHL_NORM}, rhl_seed=${RHL_SEED}, rhl_stats=${RHL_STATS}"
echo "  ckpt=${CKPT:-<none>}, curr_itrs=${CURR_ITRS}"

for ((CURR_STEP=START_STEP; CURR_STEP<=END_STEP; CURR_STEP+=STEP_INCREMENT))
do
    if [[ "$CURR_STEP" -eq 0 ]]; then
        CURR_BATCH_SIZE="$SPECIAL_BATCH_SIZE"
    else
        CURR_BATCH_SIZE="$DEFAULT_BATCH_SIZE"
    fi

    CKPT_ARG=()
    if [[ "$CURR_STEP" -eq 0 && -n "$CKPT" ]]; then
        CKPT_ARG=(--ckpt "$CKPT")
    fi

    CURR_ITRS_ARG=()
    if [[ "$CURR_STEP" -eq 0 ]]; then
        CURR_ITRS_ARG=(--curr_itrs "$CURR_ITRS")
    fi

    echo "Running training for step ${CURR_STEP} with batch size ${CURR_BATCH_SIZE}..."
    "$PYTHON" train.py \
        --data_root "$DATA_ROOT" \
        --model "$MODEL" \
        --air_feature_source "$AIR_FEATURE_SOURCE" \
        --lr "$LR" \
        --batch_size "$CURR_BATCH_SIZE" \
        --loss_type "$LOSS_TYPE" \
        --dataset "$DATASET" \
        --task "$TASK" \
        --lr_policy "$LR_POLICY" \
        --curr_step "$CURR_STEP" \
        --subpath "$SUBPATH" \
        "${BASE_SUBPATH_ARG[@]}" \
        "${CKPT_ARG[@]}" \
        "${CURR_ITRS_ARG[@]}" \
        --method "$METHOD" \
        --setting "$SETTING" \
        $PRETRAINED_BACKBONE \
        --crop_val \
        --train_epoch "$TRAIN_EPOCH" \
        --gamma "$GAMMA" \
        --buffer "$BUFFER" \
        --random_seed "$RANDOM_SEED" \
        --rhl_norm "$RHL_NORM" \
        --rhl_norm_eps "$RHL_NORM_EPS" \
        --rhl_seed "$RHL_SEED" \
        "${RHL_STATS_ARG[@]}" \
        --output_stride "$OUTPUT_STRIDE"
done
