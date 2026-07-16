#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU_ID=${GPU_ID:-2}
PYTHON_BIN=${PYTHON_BIN:-/opt/conda/envs/segacil/bin/python}

DATASET=s3dis
DATA_PATH=./datasets/S3DIS/blocks_bs1_s1/
TASKS=8-1
CVFOLD=0

NUM_POINTS=2048
PC_ATTRIBS=xyzrgbXYZ
EDGECONV_WIDTHS='[[64,64], [64, 64], [64, 64]]'
MLP_WIDTHS='[512, 256]'
K=20

EVAL_INTERVAL=3
BATCH_SIZE=32
NUM_WORKERS=16
NUM_EPOCHS=${NUM_EPOCHS:-100}
LR=0.001
WEIGHT_DECAY=0.0001
DECAY_STEP=50
DECAY_RATIO=0.5
UNCERTAIN_T=${UNCERTAIN_T:-0.0035}
START_STEP=${START_STEP:-0}
RUN_GROUP=${RUN_GROUP:-single_s3dis_s0_8_1}
TAU_LABEL=${UNCERTAIN_T//./p}
SAVE_PATH=./log_s3dis/${RUN_GROUP}/tau_${TAU_LABEL}/
BASE_MODEL_PATH=${SAVE_PATH}log_acl_s3dis_cv${CVFOLD}_tasks${TASKS}

RUN_DIR=../checkpoints_3d/s3dis/${RUN_GROUP}/tau_${TAU_LABEL}/log_acl_s3dis_cv${CVFOLD}_tasks${TASKS}
mkdir -p "$RUN_DIR"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
exec "$PYTHON_BIN" -u main.py \
  --phase ACL \
  --dataset "$DATASET" \
  --cvfold "$CVFOLD" \
  --tasks "$TASKS" \
  --data_path "$DATA_PATH" \
  --save_path "$SAVE_PATH" \
  --base_model_checkpoint_path "$BASE_MODEL_PATH" \
  --pc_npts "$NUM_POINTS" \
  --pc_attribs "$PC_ATTRIBS" \
  --pc_augm \
  --edgeconv_widths "$EDGECONV_WIDTHS" \
  --dgcnn_k "$K" \
  --dgcnn_mlp_widths "$MLP_WIDTHS" \
  --uncertain_t "$UNCERTAIN_T" \
  --start_step "$START_STEP" \
  --n_epochs "$NUM_EPOCHS" \
  --eval_interval "$EVAL_INTERVAL" \
  --batch_size "$BATCH_SIZE" \
  --n_workers "$NUM_WORKERS" \
  --base_lr "$LR" \
  --base_weight_decay "$WEIGHT_DECAY" \
  --base_decay_size "$DECAY_STEP" \
  --base_gamma "$DECAY_RATIO"
