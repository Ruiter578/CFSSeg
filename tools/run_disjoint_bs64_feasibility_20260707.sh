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
LOG_PATH="${LOG_PATH:-logs/pseudo_label/disjoint_bs64_feasibility_${RUN_STAMP}.log}"
GPU_WAIT_USED_MIB="${GPU_WAIT_USED_MIB:-2000}"
GPU_WAIT_INTERVAL_SECONDS="${GPU_WAIT_INTERVAL_SECONDS:-300}"

mkdir -p "$(dirname "$LOG_PATH")" "$CUDA_MPS_PIPE_DIRECTORY" "$TMPDIR"

exec > >(tee -a "$LOG_PATH") 2>&1

echo "[disjoint-bs64-feasibility] start $(date -u)"
echo "[disjoint-bs64-feasibility] branch=$(git rev-parse --abbrev-ref HEAD) commit=$(git rev-parse --short HEAD)"
echo "[disjoint-bs64-feasibility] log=${LOG_PATH}"
echo "[disjoint-bs64-feasibility] waiting for gpu memory.used <= ${GPU_WAIT_USED_MIB} MiB"
nvidia-smi

while true; do
  used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
  if [[ "$used_mib" =~ ^[0-9]+$ ]] && (( used_mib <= GPU_WAIT_USED_MIB )); then
    break
  fi
  echo "[disjoint-bs64-feasibility] gpu memory.used=${used_mib} MiB; sleep ${GPU_WAIT_INTERVAL_SECONDS}s"
  sleep "$GPU_WAIT_INTERVAL_SECONDS"
done

echo "[disjoint-bs64-feasibility] train grid with TRAIN_EPOCH=1"
TRAIN_EPOCH=1 bash tools/run_pseudo_label_grid.sh \
  --grid configs/pseudo_label_disjoint_bs64_feasibility_train.tsv \
  --mode run

echo "[disjoint-bs64-feasibility] summarize"
"$PYTHON" tools/summarize_pseudo_label_grid.py \
  --grid configs/pseudo_label_disjoint_bs64_feasibility_train.tsv \
  --output-md logs/pseudo_label/disjoint_bs64_feasibility_summary.md \
  --output-csv logs/pseudo_label/disjoint_bs64_feasibility_summary.csv \
  --output-json logs/pseudo_label/disjoint_bs64_feasibility_summary.json \
  --title "Disjoint BS64 Feasibility Summary" \
  --off-baseline 0.689438 \
  --fixed07-baseline 0.694639

echo "[disjoint-bs64-feasibility] done $(date -u)"
