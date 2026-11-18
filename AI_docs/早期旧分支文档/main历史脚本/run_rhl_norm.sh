#!/usr/bin/env bash
set -euo pipefail

# RHL 系列实验专用入口。
# 后续和 RHL 相关的消融、RHL-SE 多种子实验、gamma 调参等，优先改这个脚本；
# 尽量减少对原始 run.sh 的扰动，避免复现实验和新方法实验混在一起。

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# 服务器上的 /tmp/nvidia-mps 可能被其他用户占用或拥有。
# 这里默认使用当前用户自己的 MPS pipe/log 目录，避免 CUDA 初始化时报
# "MPS client failed to connect to the MPS control daemon"。
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

# RHL 消融控制项：
# - RHL_NORM 控制 RHL 输出特征的归一化方式；none 表示不做归一化，可作为同脚本基线。
# - RHL_SEED 是方案一 RHL-SE 的关键参数，只改变 RandomBuffer/RHL 随机映射初始化；
#   它不会改变 DataLoader shuffle、数据增强、全局 random_seed 等其他随机因素。
# - RHL_SEED=-1 表示沿用原代码逻辑，RandomBuffer 使用当前全局 RNG 状态初始化。
RHL_NORM="${RHL_NORM:-l2_sqrt}"
RHL_NORM_EPS="${RHL_NORM_EPS:-1e-6}"
RHL_SEED="${RHL_SEED:--1}"
RHL_STATS="${RHL_STATS:-1}"
GAMMA="${GAMMA:-1}"

# VOC 15-5 只有 step0 和 step1。
# 当前 RHL 实验默认从 step1 开始，并通过 BASE_SUBPATH 读取已有 step0 权重。
START_STEP="${START_STEP:-1}"
END_STEP="${END_STEP:-1}"
STEP_INCREMENT="${STEP_INCREMENT:-1}"

# 20260606 是当前已验证可用于 step1 的 step0 权重目录。
# 如果要换另一组 step0 权重，只需要在启动前覆盖 BASE_SUBPATH。
BASE_SUBPATH="${BASE_SUBPATH:-20260606}"

# 默认输出目录包含 RHL_NORM 和 gamma，降低不同实验写进同一目录的风险。
# 正式消融建议显式设置 SUBPATH，例如 20260616_rhl_se_seed1。
SUBPATH="${SUBPATH:-$(date +%Y%m%d)_rhl_${RHL_NORM}_g${GAMMA}}"

# step1 的默认 batch size；显存不足时可在命令前加 DEFAULT_BATCH_SIZE=32 覆盖。
DEFAULT_BATCH_SIZE="${DEFAULT_BATCH_SIZE:-64}"
# step0 训练显存更高，保留单独 batch size。当前脚本默认不跑 step0。
SPECIAL_BATCH_SIZE="${SPECIAL_BATCH_SIZE:-32}"

BASE_SUBPATH_ARG=()
if [[ -n "$BASE_SUBPATH" ]]; then
    # 只在 BASE_SUBPATH 非空时传入，避免 train.py 把空字符串当作有效目录。
    BASE_SUBPATH_ARG=(--base_subpath "$BASE_SUBPATH")
fi

RHL_STATS_ARG=()
if [[ "$RHL_STATS" == "1" ]]; then
    # 只在需要诊断 RHL 输出尺度时打印统计信息；正式批量实验可设 RHL_STATS=0。
    RHL_STATS_ARG=(--rhl_stats)
fi

echo "RHL experiment configuration:"
echo "  task=${TASK}, steps=${START_STEP}-${END_STEP}, setting=${SETTING}"
echo "  subpath=${SUBPATH}, base_subpath=${BASE_SUBPATH}"
echo "  model=${MODEL}, buffer=${BUFFER}, gamma=${GAMMA}"
echo "  rhl_norm=${RHL_NORM}, rhl_norm_eps=${RHL_NORM_EPS}, rhl_seed=${RHL_SEED}, rhl_stats=${RHL_STATS}"

for ((CURR_STEP=START_STEP; CURR_STEP<=END_STEP; CURR_STEP+=STEP_INCREMENT))
do
    if [[ "$CURR_STEP" -eq 0 ]]; then
        CURR_BATCH_SIZE="$SPECIAL_BATCH_SIZE"
    else
        CURR_BATCH_SIZE="$DEFAULT_BATCH_SIZE"
    fi

    echo "Running RHL training for step ${CURR_STEP} with batch size ${CURR_BATCH_SIZE}..."
    # RHL-SE：--rhl_seed 把独立 RHL_SEED 传入 train.py，
    # 最终只作用到 RandomBuffer 初始化，不改变全局 random_seed。
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
        --rhl_seed "$RHL_SEED" \
        "${RHL_STATS_ARG[@]}" \
        --output_stride "$OUTPUT_STRIDE"
done
