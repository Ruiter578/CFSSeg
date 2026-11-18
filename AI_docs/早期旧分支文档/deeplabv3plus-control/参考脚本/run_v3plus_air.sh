#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/root/2TStorage/lyc/SegACIL/data_root/VOC2012}"
MODEL="${MODEL:-deeplabv3plus_resnet101}"
TASK="${TASK:-15-5}"
SETTING="${SETTING:-sequential}"
SUBPATH="${SUBPATH:?Set SUBPATH to a unique experiment name}"
BASE_SUBPATH="${BASE_SUBPATH:-20260614_v3plus_voc15-5_seq_bs32-16}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TRAIN_EPOCH="${TRAIN_EPOCH:-50}"
GAMMA="${GAMMA:-1}"
BUFFER="${BUFFER:-8196}"
OUTPUT_STRIDE="${OUTPUT_STRIDE:-8}"
# Validated DeepLabV3+ default. The parser keeps decoder for backward compatibility.
AIR_FEATURE_SOURCE="${AIR_FEATURE_SOURCE:-aspp_up}"
AIR_PIXEL_BALANCE="${AIR_PIXEL_BALANCE:-none}"
AIR_MAX_PIXELS_PER_CLASS="${AIR_MAX_PIXELS_PER_CLASS:-0}"

LOG_DIR="${LOG_DIR:-logs/deeplabv3plus_air}"
LOG_FILE="${LOG_DIR}/${SUBPATH}.log"
mkdir -p "$LOG_DIR"

echo "Running DeepLabV3+ AIR feature experiment"
echo "  feature source: ${AIR_FEATURE_SOURCE}"
echo "  pixel balance: ${AIR_PIXEL_BALANCE}"
echo "  max pixels per class: ${AIR_MAX_PIXELS_PER_CLASS}"
echo "  step0 checkpoint: ${BASE_SUBPATH}"
echo "  output subpath: ${SUBPATH}"
echo "  batch size: ${BATCH_SIZE}"
echo "  log: ${LOG_FILE}"

"$PYTHON" train.py \
    --data_root "$DATA_ROOT" \
    --model "$MODEL" \
    --lr 0.01 \
    --batch_size "$BATCH_SIZE" \
    --loss_type bce_loss \
    --dataset voc \
    --task "$TASK" \
    --lr_policy poly \
    --curr_step 1 \
    --subpath "$SUBPATH" \
    --base_subpath "$BASE_SUBPATH" \
    --method acil \
    --setting "$SETTING" \
    --pretrained_backbone \
    --crop_val \
    --train_epoch "$TRAIN_EPOCH" \
    --gamma "$GAMMA" \
    --buffer "$BUFFER" \
    --output_stride "$OUTPUT_STRIDE" \
    --air_feature_source "$AIR_FEATURE_SOURCE" \
    --air_pixel_balance "$AIR_PIXEL_BALANCE" \
    --air_max_pixels_per_class "$AIR_MAX_PIXELS_PER_CLASS" \
    2>&1 | tee "$LOG_FILE"
