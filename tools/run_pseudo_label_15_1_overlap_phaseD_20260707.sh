#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHON="${PYTHON:-/home/linyichen/miniconda3/envs/segacil/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/home/linyichen/.mps_bypass}"
export TMPDIR="${TMPDIR:-/root/2TStorage/tmp}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export SEGACIL_PIN_MEMORY="${SEGACIL_PIN_MEMORY:-0}"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_PATH:-logs/pseudo_label/phaseD_15-1_overlap_${RUN_STAMP}.log}"
GPU_WAIT_USED_MIB="${GPU_WAIT_USED_MIB:-2000}"
GPU_WAIT_INTERVAL_SECONDS="${GPU_WAIT_INTERVAL_SECONDS:-300}"

TASK="15-1"
SETTING="overlap"
OFF_SUBPATH="20260707_pseudo_15-1_overlap_off_seed1_bs32_phaseD"
FIXED06_SUBPATH="20260707_pseudo_15-1_overlap_fixed0p6_seed1_bs32_phaseD_reuse_offstep0"
FIXED07_SUBPATH="20260707_pseudo_15-1_overlap_fixed0p7_seed1_bs32_phaseD_reuse_offstep0"

mkdir -p "$(dirname "$LOG_PATH")" "$CUDA_MPS_PIPE_DIRECTORY" "$TMPDIR"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "[phaseD-15-1-overlap] start $(date -u)"
echo "[phaseD-15-1-overlap] branch=$(git rev-parse --abbrev-ref HEAD) commit=$(git rev-parse --short HEAD)"
echo "[phaseD-15-1-overlap] log=${LOG_PATH}"
echo "[phaseD-15-1-overlap] waiting for gpu memory.used <= ${GPU_WAIT_USED_MIB} MiB"
nvidia-smi

while true; do
  used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
  if [[ "$used_mib" =~ ^[0-9]+$ ]] && (( used_mib <= GPU_WAIT_USED_MIB )); then
    break
  fi
  echo "[phaseD-15-1-overlap] gpu memory.used=${used_mib} MiB; sleep ${GPU_WAIT_INTERVAL_SECONDS}s"
  sleep "$GPU_WAIT_INTERVAL_SECONDS"
done

run_off() {
  local subpath="$1"
  local step5_dir="checkpoints/${subpath}/voc/${TASK}/${SETTING}/step5"
  if compgen -G "${step5_dir}/test_results_*.json" >/dev/null; then
    echo "[phaseD-15-1-overlap] skip completed off: ${step5_dir}"
    return
  fi
  if [[ -e "$step5_dir" ]]; then
    echo "[phaseD-15-1-overlap] refusing incomplete off output dir: ${step5_dir}" >&2
    exit 3
  fi
  echo "[phaseD-15-1-overlap] run off baseline subpath=${subpath}"
  SUBPATH="$subpath" \
  BASE_SUBPATH="$subpath" \
  TASK="$TASK" \
  SETTING="$SETTING" \
  START_STEP=0 END_STEP=5 \
  USE_PSEUDO_LABEL=0 \
  PSEUDO_LABEL_STRATEGY=off \
  DEFAULT_BATCH_SIZE=32 SPECIAL_BATCH_SIZE=32 \
  BUFFER=8196 GAMMA=1 RANDOM_SEED=1 \
  bash run.sh
  "$PYTHON" tools/summarize_adaptive_pseudo_label.py "$step5_dir" \
    --output "logs/pseudo_label/${subpath}_step5_summary.md"
}

run_fixed() {
  local subpath="$1"
  local confidence="$2"
  local step5_dir="checkpoints/${subpath}/voc/${TASK}/${SETTING}/step5"
  if compgen -G "${step5_dir}/test_results_*.json" >/dev/null; then
    echo "[phaseD-15-1-overlap] skip completed fixed${confidence}: ${step5_dir}"
    return
  fi
  if [[ -e "$step5_dir" ]]; then
    echo "[phaseD-15-1-overlap] refusing incomplete fixed output dir: ${step5_dir}" >&2
    exit 3
  fi
  echo "[phaseD-15-1-overlap] run fixed confidence=${confidence} subpath=${subpath}"
  SUBPATH="$subpath" \
  BASE_SUBPATH="$OFF_SUBPATH" \
  TASK="$TASK" \
  SETTING="$SETTING" \
  START_STEP=1 END_STEP=5 \
  USE_PSEUDO_LABEL=1 \
  PSEUDO_LABEL_STRATEGY=fixed \
  PSEUDO_LABEL_CONFIDENCE="$confidence" \
  PSEUDO_LABEL_QUANTILE=0.7 \
  DEFAULT_BATCH_SIZE=32 SPECIAL_BATCH_SIZE=32 \
  BUFFER=8196 GAMMA=1 RANDOM_SEED=1 \
  bash run.sh
  "$PYTHON" tools/summarize_adaptive_pseudo_label.py "$step5_dir" \
    --output "logs/pseudo_label/${subpath}_step5_summary.md"
}

run_off "$OFF_SUBPATH"
run_fixed "$FIXED06_SUBPATH" "0.6"
run_fixed "$FIXED07_SUBPATH" "0.7"

echo "[phaseD-15-1-overlap] done $(date -u)"
