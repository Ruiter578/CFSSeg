#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

# DATA_ROOT="/mnt/petrelfs/lirui/SegACIL/datasets/data/voc"

DATA_ROOT="/root/2TStorage/lyc/SegACIL/data_root/VOC2012"
MODEL="deeplabv3_resnet101"
LR=0.01
LOSS_TYPE="bce_loss"
DATASET="voc"
TASK="15-5"
LR_POLICY="poly"
SUBPATH="${SUBPATH:-$(date +%Y%m%d)}"
BASE_SUBPATH="${BASE_SUBPATH:-}"
METHOD="acil"
SETTING="sequential"
TRAIN_EPOCH=50
PRETRAINED_BACKBONE="--pretrained_backbone"
BUFFER=8196
OUTPUT_STRIDE=8
GAMMA="${GAMMA:-1}"
RHL_NORM="${RHL_NORM:-none}"
RHL_NORM_EPS="${RHL_NORM_EPS:-1e-6}"
# RHL_SEED 是方案一 RHL-SE 的独立随机种子：
# - 默认 -1 表示完全沿用原始 RHL 初始化逻辑；
# - 设为非负整数时，只改变 RandomBuffer 的随机映射，不改变全局 random_seed。
RHL_SEED="${RHL_SEED:--1}"
RHL_STATS="${RHL_STATS:-0}"


DEFAULT_BATCH_SIZE=64   # Batch sizes for different steps
SPECIAL_BATCH_SIZE=32   # Batch size for step=0


# Loop through steps
START_STEP=1
END_STEP=1
STEP_INCREMENT=1

BASE_SUBPATH_ARG=()
if [[ -n "$BASE_SUBPATH" ]]; then
    # 仅在显式指定 BASE_SUBPATH 时传入，用于 step1 读取已有 step0 权重。
    BASE_SUBPATH_ARG=(--base_subpath "$BASE_SUBPATH")
fi

RHL_STATS_ARG=()
if [[ "$RHL_STATS" == "1" ]]; then
    # 打印 RHL 输出统计信息，仅用于诊断；正式复现实验默认关闭。
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
    # --rhl_seed 是 RHL-SE 的入口参数，最终只传到 RandomBuffer 初始化。
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
        --rhl_seed "$RHL_SEED" \
        "${RHL_STATS_ARG[@]}" \
        --output_stride $OUTPUT_STRIDE
done
