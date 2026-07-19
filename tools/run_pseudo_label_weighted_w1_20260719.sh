#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

GRID_PATH="${GRID_PATH:-configs/pseudo_label_weighted_w1_20260719.tsv}"
BASELINES_PATH="${BASELINES_PATH:-configs/pseudo_label_weighted_w1_baselines_20260719.json}"
LOG_PATH="${LOG_PATH:-logs/pseudo_label/weighted_w1_20260719.log}"
SUMMARY_BASE="${SUMMARY_BASE:-logs/pseudo_label/weighted_w1_20260719_summary}"
SOURCE_PREFIX="${SOURCE_PREFIX:-logs/pseudo_label/weighted_w1_20260719_source}"
LOCK_PATH="${LOCK_PATH:-logs/pseudo_label/weighted_w1_20260719.lock}"
DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-0}"

export PYTHON="${PYTHON:-/home/linyichen/miniconda3/envs/segacil/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/home/linyichen/.mps_bypass}"
export TMPDIR="${TMPDIR:-/root/2TStorage/tmp}"
export SEGACIL_PIN_MEMORY=0
export RHL_NORM=none
export RHL_SEED=-1
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export ALLOW_INCOMPLETE=0
export SKIP_EXISTING="${RESUME}"

if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
  echo "DRY_RUN must be 0 or 1" >&2
  exit 2
fi
if [[ "${RESUME}" != "0" && "${RESUME}" != "1" ]]; then
  echo "RESUME must be 0 or 1" >&2
  exit 2
fi
if [[ ! -f "${GRID_PATH}" || ! -f "${BASELINES_PATH}" ]]; then
  echo "[weighted-w1] missing grid or baseline registry" >&2
  exit 2
fi

mkdir -p "$(dirname "${LOCK_PATH}")"
exec 9>"${LOCK_PATH}"
if ! flock -n 9; then
  echo "[weighted-w1] another W1 runner holds ${LOCK_PATH}" >&2
  exit 3
fi

"${PYTHON}" - "${GRID_PATH}" "${BASELINES_PATH}" "${REPO_ROOT}" "${RESUME}" <<'PY'
import sys
from pathlib import Path
from tools.summarize_pseudo_label_weighted_w1 import (
    load_and_verify_baselines,
    read_and_validate_grid,
    verify_output_paths_absent,
    verify_step0_checkpoints,
)

grid, baselines, repo_root, resume = sys.argv[1:]
rows = read_and_validate_grid(grid)
verified_baselines = load_and_verify_baselines(baselines, repo_root=Path(repo_root))
verify_step0_checkpoints(
    rows,
    repo_root=Path(repo_root),
    baselines=verified_baselines,
)
if resume != "1":
    verify_output_paths_absent(rows, repo_root=Path(repo_root))
print("[weighted-w1] grid, W0, baselines, step0 hashes, and output paths verified")
PY

available_kib="$(df -Pk /root/2TStorage | awk 'NR==2 {print $4}')"
required_kib=$((20 * 1024 * 1024))
if (( available_kib < required_kib )); then
  echo "[weighted-w1] requires at least 20 GiB free on /root/2TStorage" >&2
  exit 2
fi

echo "[weighted-w1] branch=$(git rev-parse --abbrev-ref HEAD)"
echo "[weighted-w1] commit=$(git rev-parse HEAD)"
echo "[weighted-w1] dirty=$([[ -n "$(git status --porcelain)" ]] && echo true || echo false)"
echo "[weighted-w1] resume=$([[ "${RESUME}" == "1" ]] && echo true || echo false)"
echo "[weighted-w1] free_kib=${available_kib}"
echo "[weighted-w1] locked_runtime=RHL_NORM=${RHL_NORM} RHL_SEED=${RHL_SEED} SEGACIL_PIN_MEMORY=${SEGACIL_PIN_MEMORY}"
nvidia-smi
ps -eo pid,user,%cpu,%mem,cmd --sort=-%cpu | sed -n '1,20p'
bash tools/run_pseudo_label_grid.sh --grid "${GRID_PATH}" --mode dry-run

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[weighted-w1] dry-run complete"
  exit 0
fi

mkdir -p \
  "$(dirname "${LOG_PATH}")" \
  "${CUDA_MPS_PIPE_DIRECTORY}" \
  "${TMPDIR}"
git status --short > "${SOURCE_PREFIX}_status.txt"
git diff --binary HEAD > "${SOURCE_PREFIX}_patch.diff"
while IFS= read -r -d '' untracked; do
  diff_status=0
  git diff --no-index --binary /dev/null "${untracked}" \
    >> "${SOURCE_PREFIX}_patch.diff" || diff_status=$?
  if (( diff_status > 1 )); then
    echo "[weighted-w1] failed to capture untracked source: ${untracked}" >&2
    exit "${diff_status}"
  fi
done < <(git ls-files --others --exclude-standard -z)
SOURCE_COMMIT="$(git rev-parse HEAD)"
SOURCE_DIRTY="$([[ -s "${SOURCE_PREFIX}_status.txt" ]] && echo true || echo false)"
mapfile -t BASELINE_PROVENANCE < <(
  "${PYTHON}" - "${BASELINES_PATH}" <<'PY'
import json
import sys
from pathlib import Path

registry = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for setting in ("overlap", "disjoint"):
    record = registry["settings"][setting]
    print(record["result_path"])
    print(record["result_sha256"])
PY
)
export SEGACIL_SOURCE_COMMIT="${SOURCE_COMMIT}"
export SEGACIL_SOURCE_DIRTY="${SOURCE_DIRTY}"
export SEGACIL_SOURCE_STATUS_PATH="${SOURCE_PREFIX}_status.txt"
export SEGACIL_SOURCE_PATCH_PATH="${SOURCE_PREFIX}_patch.diff"
export SEGACIL_BASELINE_REGISTRY_PATH="${BASELINES_PATH}"
export SEGACIL_BASELINE_REGISTRY_SHA256="$(
  sha256sum "${BASELINES_PATH}" | awk '{print $1}'
)"
export SEGACIL_OVERLAP_BASELINE_RESULT_PATH="${BASELINE_PROVENANCE[0]}"
export SEGACIL_OVERLAP_BASELINE_RESULT_SHA256="${BASELINE_PROVENANCE[1]}"
export SEGACIL_DISJOINT_BASELINE_RESULT_PATH="${BASELINE_PROVENANCE[2]}"
export SEGACIL_DISJOINT_BASELINE_RESULT_SHA256="${BASELINE_PROVENANCE[3]}"
{
  echo "source_commit=${SOURCE_COMMIT}"
  echo "source_dirty=${SOURCE_DIRTY}"
  echo "source_status_path=${SOURCE_PREFIX}_status.txt"
  echo "source_patch_path=${SOURCE_PREFIX}_patch.diff"
  echo "baseline_registry=${BASELINES_PATH}"
  echo "baseline_registry_sha256=${SEGACIL_BASELINE_REGISTRY_SHA256}"
  echo "rhl_norm=${RHL_NORM}"
  echo "rhl_seed=${RHL_SEED}"
  echo "pin_memory=${SEGACIL_PIN_MEMORY}"
  echo "grid=${GRID_PATH}"
} > "${SOURCE_PREFIX}_metadata.txt"

exec > >(tee -a "${LOG_PATH}") 2>&1
echo "[weighted-w1] started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
df -h /root/2TStorage
nvidia-smi

bash tools/run_pseudo_label_grid.sh --grid "${GRID_PATH}" --mode run

"${PYTHON}" tools/summarize_pseudo_label_weighted_w1.py \
  --grid "${GRID_PATH}" \
  --baselines "${BASELINES_PATH}" \
  --output-base "${SUMMARY_BASE}"

echo "[weighted-w1] completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
