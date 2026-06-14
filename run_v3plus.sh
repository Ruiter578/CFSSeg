#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Usage:
#   bash run_v3plus.sh
#   SPECIAL_BATCH_SIZE=16 DEFAULT_BATCH_SIZE=8 bash run_v3plus.sh
#
# SPECIAL_BATCH_SIZE controls step0. DEFAULT_BATCH_SIZE controls step1+.

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

DATA_ROOT="/TRS-SAS/linwei/SegACIL/data_root/VOC2012"
MODEL="deeplabv3plus_resnet101"
TASK="15-5"
SETTING="sequential"
SUBPATH="20260613_v3plus_voc15-5_seq_bs16"
START_STEP=0
END_STEP=1
TRAIN_EPOCH=50
GAMMA=1
BUFFER=8196
OUTPUT_STRIDE=8

SPECIAL_BATCH_SIZE="${SPECIAL_BATCH_SIZE:-16}"
DEFAULT_BATCH_SIZE="${DEFAULT_BATCH_SIZE:-8}"

LOG_DIR="logs/deeplabv3plus"
LOG_FILE="${LOG_DIR}/${SUBPATH}_step0-bs${SPECIAL_BATCH_SIZE}_step1-bs${DEFAULT_BATCH_SIZE}.log"
mkdir -p "$LOG_DIR"

echo "Running DeepLabV3+ VOC ${TASK} ${SETTING}"
echo "  step0 batch size: ${SPECIAL_BATCH_SIZE}"
echo "  step1 batch size: ${DEFAULT_BATCH_SIZE}"
echo "  checkpoint subpath: ${SUBPATH}"
echo "  log: ${LOG_FILE}"

for ((CURR_STEP=START_STEP; CURR_STEP<=END_STEP; CURR_STEP++)); do
    if [[ "$CURR_STEP" -eq 0 ]]; then
        CURR_BATCH_SIZE="$SPECIAL_BATCH_SIZE"
    else
        CURR_BATCH_SIZE="$DEFAULT_BATCH_SIZE"
    fi

    echo "Running training for step ${CURR_STEP} with batch size ${CURR_BATCH_SIZE}..."
    python train.py \
        --data_root "$DATA_ROOT" \
        --model "$MODEL" \
        --lr 0.01 \
        --batch_size "$CURR_BATCH_SIZE" \
        --loss_type bce_loss \
        --dataset voc \
        --task "$TASK" \
        --lr_policy poly \
        --curr_step "$CURR_STEP" \
        --subpath "$SUBPATH" \
        --method acil \
        --setting "$SETTING" \
        --pretrained_backbone \
        --crop_val \
        --train_epoch "$TRAIN_EPOCH" \
        --gamma "$GAMMA" \
        --buffer "$BUFFER" \
        --output_stride "$OUTPUT_STRIDE"
done 2>&1 | tee "$LOG_FILE"
