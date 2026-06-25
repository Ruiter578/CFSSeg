#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=2

# DATA_ROOT="/mnt/petrelfs/lirui/SegACIL/datasets/data/voc"

DATA_ROOT="/TRS-SAS/linwei/SegACIL/data_root/VOC2012"
MODEL="deeplabv3_resnet101"
LR=0.01
LOSS_TYPE="bce_loss"
DATASET="voc"
TASK="15-5"
LR_POLICY="poly"
SUBPATH="${SUBPATH:-20260624_buffer8200_step1_32_run2_trs}"  # TRS server output directory under checkpoints/
BASE_SUBPATH="${BASE_SUBPATH:-20260621_baseline_bs16_16_trs}" # TRS step0 checkpoint used when START_STEP=1
METHOD="acil"
SETTING="sequential"
TRAIN_EPOCH=50
PRETRAINED_BACKBONE="--pretrained_backbone"
BUFFER=8200     # origin value: 8196
OUTPUT_STRIDE=8
GAMMA="${GAMMA:-1}"
RHL_NORM="${RHL_NORM:-none}"               # default: none, RHL normalization disabled
RHL_NORM_EPS="${RHL_NORM_EPS:-1e-6}"       # default: 1e-6, only used when RHL_NORM is enabled
RHL_STATS="${RHL_STATS:-0}"                # default: 0, set to 1 to print RHL feature statistics


DEFAULT_BATCH_SIZE=32   # Batch size for incremental steps, including step 1
SPECIAL_BATCH_SIZE=32   # Batch size for step=0


# Loop through steps
START_STEP=1
END_STEP=1
STEP_INCREMENT=1

BASE_SUBPATH_ARG=()
if [[ -n "$BASE_SUBPATH" ]]; then
    BASE_SUBPATH_ARG=(--base_subpath "$BASE_SUBPATH")
fi

RHL_STATS_ARG=()
if [[ "$RHL_STATS" == "1" ]]; then
    RHL_STATS_ARG=(--rhl_stats)
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
        --rhl_norm "$RHL_NORM" \
        --rhl_norm_eps "$RHL_NORM_EPS" \
        "${RHL_STATS_ARG[@]}" \
        --output_stride $OUTPUT_STRIDE
done
