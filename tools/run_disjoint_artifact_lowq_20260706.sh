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
LOG_PATH="${LOG_PATH:-logs/pseudo_label/disjoint_artifact_lowq_${RUN_STAMP}.log}"
mkdir -p "$(dirname "$LOG_PATH")" "$CUDA_MPS_PIPE_DIRECTORY" "$TMPDIR"

exec > >(tee -a "$LOG_PATH") 2>&1

echo "[disjoint-lowq] start $(date -u)"
echo "[disjoint-lowq] branch=$(git rev-parse --abbrev-ref HEAD) commit=$(git rev-parse --short HEAD)"
echo "[disjoint-lowq] log=${LOG_PATH}"
nvidia-smi

echo "[disjoint-lowq] calibration grid"
bash tools/run_pseudo_label_artifact_calibration_grid.sh \
  --grid configs/pseudo_label_disjoint_artifact_lowq_calibration.tsv \
  --mode run

echo "[disjoint-lowq] train grid"
bash tools/run_pseudo_label_grid.sh \
  --grid configs/pseudo_label_disjoint_artifact_lowq_train.tsv \
  --mode run

echo "[disjoint-lowq] summarize"
"$PYTHON" tools/summarize_pseudo_label_grid.py \
  --grid configs/pseudo_label_disjoint_artifact_lowq_train.tsv \
  --output-md logs/pseudo_label/disjoint_phaseB_artifact_lowq_summary.md \
  --output-csv logs/pseudo_label/disjoint_phaseB_artifact_lowq_summary.csv \
  --output-json logs/pseudo_label/disjoint_phaseB_artifact_lowq_summary.json \
  --title "Disjoint PhaseB Artifact Low-Q Summary" \
  --off-baseline 0.6894381562818479 \
  --fixed07-baseline 0.6946392343829362

echo "[disjoint-lowq] done $(date -u)"
