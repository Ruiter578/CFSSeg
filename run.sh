#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=2

# DATA_ROOT="/mnt/petrelfs/lirui/SegACIL/datasets/data/voc"

DATA_ROOT="${DATA_ROOT:-/root/2TStorage/lyc/SegACIL/data_root/VOC2012}"
MODEL="${MODEL:-deeplabv3_resnet101}"
LR="${LR:-0.01}"
LOSS_TYPE="${LOSS_TYPE:-bce_loss}"
DATASET="${DATASET:-voc}"
TASK="${TASK:-15-5}"
LR_POLICY="${LR_POLICY:-poly}"
SUBPATH="${SUBPATH:-$(date +%Y%m%d)}"
BASE_SUBPATH="${BASE_SUBPATH:-}"
METHOD="${METHOD:-acil}"
SETTING="${SETTING:-sequential}"
TRAIN_EPOCH="${TRAIN_EPOCH:-50}"
PRETRAINED_BACKBONE="${PRETRAINED_BACKBONE:---pretrained_backbone}"
BUFFER="${BUFFER:-8196}"
OUTPUT_STRIDE="${OUTPUT_STRIDE:-8}"
GAMMA="${GAMMA:-1}"


DEFAULT_BATCH_SIZE="${DEFAULT_BATCH_SIZE:-32}"   # Batch sizes for different steps
SPECIAL_BATCH_SIZE="${SPECIAL_BATCH_SIZE:-32}"   # Batch size for step=0


# Loop through steps
START_STEP="${START_STEP:-1}"
END_STEP="${END_STEP:-1}"
STEP_INCREMENT="${STEP_INCREMENT:-1}"

BASE_SUBPATH_ARG=()
if [[ -n "$BASE_SUBPATH" ]]; then
    BASE_SUBPATH_ARG=(--base_subpath "$BASE_SUBPATH")
fi

for ((CURR_STEP=$START_STEP; CURR_STEP<=$END_STEP; CURR_STEP+=$STEP_INCREMENT))
do
    if [ $CURR_STEP -eq 0 ]; then
        CURR_BATCH_SIZE=$SPECIAL_BATCH_SIZE
    else
        CURR_BATCH_SIZE=$DEFAULT_BATCH_SIZE
    fi

    echo "Running training for step $CURR_STEP with batch size $CURR_BATCH_SIZE..."
    python train.py \
        --data_root $DATA_ROOT \
        --model $MODEL \
        --lr $LR \
        --batch_size $CURR_BATCH_SIZE \
        --loss_type $LOSS_TYPE \
        --dataset $DATASET \
        --task $TASK \
        --lr_policy $LR_POLICY \
        --curr_step $CURR_STEP \
        --subpath $SUBPATH \
        "${BASE_SUBPATH_ARG[@]}" \
        --method $METHOD \
        --setting $SETTING \
        $PRETRAINED_BACKBONE \
        --crop_val \
        --train_epoch $TRAIN_EPOCH \
        --gamma "$GAMMA" \
        --buffer $BUFFER \
        --output_stride $OUTPUT_STRIDE
done
