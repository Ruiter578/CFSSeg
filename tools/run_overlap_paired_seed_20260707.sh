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
LOG_PATH="${LOG_PATH:-logs/pseudo_label/overlap_paired_seed_${RUN_STAMP}.log}"
mkdir -p "$(dirname "$LOG_PATH")" "$CUDA_MPS_PIPE_DIRECTORY" "$TMPDIR"

exec > >(tee -a "$LOG_PATH") 2>&1

echo "[overlap-paired-seed] start $(date -u)"
echo "[overlap-paired-seed] branch=$(git rev-parse --abbrev-ref HEAD) commit=$(git rev-parse --short HEAD)"
echo "[overlap-paired-seed] log=${LOG_PATH}"
nvidia-smi

echo "[overlap-paired-seed] calibration grid"
bash tools/run_pseudo_label_artifact_calibration_grid.sh \
  --grid configs/pseudo_label_overlap_paired_seed_calibration.tsv \
  --mode run

echo "[overlap-paired-seed] train grid"
bash tools/run_pseudo_label_grid.sh \
  --grid configs/pseudo_label_overlap_paired_seed_train.tsv \
  --mode run

echo "[overlap-paired-seed] summarize"
"$PYTHON" tools/summarize_pseudo_label_grid.py \
  --grid configs/pseudo_label_overlap_paired_seed_train.tsv \
  --output-md logs/pseudo_label/overlap_paired_seed_summary.md \
  --output-csv logs/pseudo_label/overlap_paired_seed_summary.csv \
  --output-json logs/pseudo_label/overlap_paired_seed_summary.json \
  --title "Overlap PhaseB Paired Seed Summary" \
  --off-baseline 0.703080 \
  --fixed07-baseline 0.707731

echo "[overlap-paired-seed] done $(date -u)"
