#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PROJECT_ROOT="/root/2TStorage/lyc/SegACIL_deeplabv3plus"
PYTHON="${PYTHON:-/home/linyichen/miniconda3/envs/segacil/bin/python}"
DATA_ROOT="${DATA_ROOT:-/root/2TStorage/lyc/SegACIL/data_root/VOC2012}"

MODEL="deeplabv3plus_resnet101"
LR="0.01"
LOSS_TYPE="bce_loss"
DATASET="voc"
TASK="15-5"
LR_POLICY="poly"
SUBPATH="20260612_v3plus_voc15-5_seq_bs8_step1bs2"
METHOD="acil"
SETTING="sequential"
PRETRAINED_BACKBONE="--pretrained_backbone"
BUFFER="8196"
OUTPUT_STRIDE="8"
GAMMA="1"

STEP0_CKPT="checkpoints/${SUBPATH}/${DATASET}/${TASK}/${SETTING}/step0/final.pth"
STEP0_COMPLETED_EPOCHS="30"
STEP0_TOTAL_EPOCHS="50"
STEP0_BATCH_SIZE="8"
STEP0_ITERS_PER_EPOCH="1054"
STEP0_REMAINING_EPOCHS=$((STEP0_TOTAL_EPOCHS - STEP0_COMPLETED_EPOCHS))
STEP0_COMPLETED_ITERS=$((STEP0_COMPLETED_EPOCHS * STEP0_ITERS_PER_EPOCH))

STEP1_BATCH_SIZE="2"
STEP1_TRAIN_EPOCH="50"

mkdir -p "${PROJECT_ROOT}/logs/deeplabv3plus"
LOG_FILE="${PROJECT_ROOT}/logs/deeplabv3plus/resume_${SUBPATH}_$(date +%Y%m%d_%H%M%S).log"

cd "${PROJECT_ROOT}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[$(date '+%F %T')] Resume log: ${LOG_FILE}"
echo "[$(date '+%F %T')] Resuming DeepLabV3+ step0 from ${STEP0_CKPT}"
echo "[$(date '+%F %T')] completed_epochs=${STEP0_COMPLETED_EPOCHS}, completed_iters=${STEP0_COMPLETED_ITERS}, remaining_epochs=${STEP0_REMAINING_EPOCHS}"
echo "[$(date '+%F %T')] step0_batch_size=${STEP0_BATCH_SIZE}, step1_batch_size=${STEP1_BATCH_SIZE}"

if [[ ! -f "${STEP0_CKPT}" ]]; then
    echo "Checkpoint not found: ${STEP0_CKPT}" >&2
    exit 1
fi

"${PYTHON}" train.py \
    --data_root "${DATA_ROOT}" \
    --model "${MODEL}" \
    --lr "${LR}" \
    --batch_size "${STEP0_BATCH_SIZE}" \
    --loss_type "${LOSS_TYPE}" \
    --dataset "${DATASET}" \
    --task "${TASK}" \
    --lr_policy "${LR_POLICY}" \
    --curr_step 0 \
    --subpath "${SUBPATH}" \
    --method "${METHOD}" \
    --setting "${SETTING}" \
    ${PRETRAINED_BACKBONE} \
    --crop_val \
    --train_epoch "${STEP0_REMAINING_EPOCHS}" \
    --curr_itrs "${STEP0_COMPLETED_ITERS}" \
    --ckpt "${STEP0_CKPT}" \
    --gamma "${GAMMA}" \
    --buffer "${BUFFER}" \
    --output_stride "${OUTPUT_STRIDE}"

echo "[$(date '+%F %T')] Step0 resume finished. Starting step1."

"${PYTHON}" train.py \
    --data_root "${DATA_ROOT}" \
    --model "${MODEL}" \
    --lr "${LR}" \
    --batch_size "${STEP1_BATCH_SIZE}" \
    --loss_type "${LOSS_TYPE}" \
    --dataset "${DATASET}" \
    --task "${TASK}" \
    --lr_policy "${LR_POLICY}" \
    --curr_step 1 \
    --subpath "${SUBPATH}" \
    --method "${METHOD}" \
    --setting "${SETTING}" \
    ${PRETRAINED_BACKBONE} \
    --crop_val \
    --train_epoch "${STEP1_TRAIN_EPOCH}" \
    --gamma "${GAMMA}" \
    --buffer "${BUFFER}" \
    --output_stride "${OUTPUT_STRIDE}"

echo "[$(date '+%F %T')] DeepLabV3+ resume run finished."
