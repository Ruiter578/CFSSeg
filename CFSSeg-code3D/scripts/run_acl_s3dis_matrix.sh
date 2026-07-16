#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU_ID=${GPU_ID:-2}
PYTHON_BIN=${PYTHON_BIN:-python}
TAU_VALUES=${TAU_VALUES:-"0.0035 0.002"}
TASKS_VALUES=${TASKS_VALUES:-"8-1"}
CVFOLDS=${CVFOLDS:-"0"}
START_STEP=${START_STEP:-0}
RUN_GROUP=${RUN_GROUP:-}

if [[ -z "$RUN_GROUP" ]]; then
  if [[ "$START_STEP" != "0" ]]; then
    echo "START_STEP=${START_STEP} requires an explicit RUN_GROUP so resume paths are stable." >&2
    exit 2
  fi
  RUN_GROUP=$(date +%Y%m%d_%H%M%S_s3dis_acl_matrix)
fi

DATASET=s3dis
DATA_PATH=./datasets/S3DIS/blocks_bs1_s1/

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

read -r -a TAU_ARRAY <<< "$TAU_VALUES"
read -r -a TASKS_ARRAY <<< "$TASKS_VALUES"
read -r -a CVFOLD_ARRAY <<< "$CVFOLDS"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

for UNCERTAIN_T in "${TAU_ARRAY[@]}"; do
  TAU_LABEL=${UNCERTAIN_T//./p}
  SAVE_PATH="./log_s3dis/${RUN_GROUP}/tau_${TAU_LABEL}/"

  for CVFOLD in "${CVFOLD_ARRAY[@]}"; do
    for TASKS in "${TASKS_ARRAY[@]}"; do
      BASE_MODEL_PATH="${SAVE_PATH}log_acl_s3dis_cv${CVFOLD}_tasks${TASKS}"
      echo "Running S3DIS ACL: run_group=${RUN_GROUP} cvfold=${CVFOLD} tasks=${TASKS} uncertain_t=${UNCERTAIN_T}"

      "$PYTHON_BIN" -u main.py \
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
    done
  done
done
