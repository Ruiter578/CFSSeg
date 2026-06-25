#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

: "${CKPT:?Set CKPT to the step0 checkpoint path to resume from}"
: "${SUBPATH:?Set SUBPATH to a fresh output directory for the resumed run}"
: "${CURR_ITRS:?Set CURR_ITRS to completed iterations before this resume}"

export START_STEP="${START_STEP:-0}"
export END_STEP="${END_STEP:-0}"
export BASE_SUBPATH="${BASE_SUBPATH:-}"
export TRAIN_EPOCH="${TRAIN_EPOCH:-10}"
export SPECIAL_BATCH_SIZE="${SPECIAL_BATCH_SIZE:-32}"

echo "Resuming step0 training from checkpoint:"
echo "  CKPT=${CKPT}"
echo "  SUBPATH=${SUBPATH}"
echo "  CURR_ITRS=${CURR_ITRS}"
echo "  TRAIN_EPOCH=${TRAIN_EPOCH}"

bash run.sh
