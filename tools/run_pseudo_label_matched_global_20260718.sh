#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

GRID_PATH="configs/pseudo_label_matched_global_fixed_20260718.tsv"
LOG_PATH="${LOG_PATH:-logs/pseudo_label/matched_global_20260718.log}"
SUMMARY_BASE="logs/pseudo_label/matched_global_20260718_summary"
DRY_RUN="${DRY_RUN:-0}"

export PYTHON="${PYTHON:-/home/linyichen/miniconda3/envs/segacil/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/home/linyichen/.mps_bypass}"
export TMPDIR="${TMPDIR:-/root/2TStorage/tmp}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export SEGACIL_PIN_MEMORY="${SEGACIL_PIN_MEMORY:-0}"

OVERLAP_STEP0="checkpoints/20260627_pseudo_15-5_overlap_batchclass_q0p7_seed1_bs32/voc/15-5/overlap/step0/deeplabv3_resnet101_voc_15-5_step_0_overlap.pth"
OVERLAP_SHA256="6505c26b567cefbb9eb099af174961cbd4596d139b30a7f92f952ad494a9a913"
DISJOINT_STEP0="checkpoints/20260705_pseudo_15-5_disjoint_off_seed1_bs32/voc/15-5/disjoint/step0/deeplabv3_resnet101_voc_15-5_step_0_disjoint.pth"
DISJOINT_SHA256="040c69eba9be68c4f370afaf31baf3fa14a16ad50d22a1a53752e7fd3ea37962"

if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
    echo "DRY_RUN must be 0 or 1" >&2
    exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
    echo "[matched-global] refusing dirty worktree" >&2
    git status --short >&2
    exit 2
fi
if [[ ! -f "$GRID_PATH" ]]; then
    echo "[matched-global] missing grid: $GRID_PATH" >&2
    exit 2
fi

verify_checkpoint() {
    local path="$1"
    local expected_sha="$2"
    if [[ ! -f "$path" ]]; then
        echo "[matched-global] missing checkpoint: $path" >&2
        exit 2
    fi
    local actual_sha
    actual_sha="$(sha256sum "$path" | awk '{print $1}')"
    if [[ "$actual_sha" != "$expected_sha" ]]; then
        echo "[matched-global] checkpoint SHA mismatch: $path" >&2
        echo "[matched-global] expected=$expected_sha actual=$actual_sha" >&2
        exit 2
    fi
}

verify_checkpoint "$OVERLAP_STEP0" "$OVERLAP_SHA256"
verify_checkpoint "$DISJOINT_STEP0" "$DISJOINT_SHA256"

mkdir -p "$(dirname "$LOG_PATH")" "$CUDA_MPS_PIPE_DIRECTORY" "$TMPDIR"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "[matched-global] start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[matched-global] branch=$(git rev-parse --abbrev-ref HEAD)"
echo "[matched-global] commit=$(git rev-parse HEAD)"
echo "[matched-global] grid=$GRID_PATH"
echo "[matched-global] dry_run=$DRY_RUN"
echo "[matched-global] python=$PYTHON"
echo "[matched-global] log=$LOG_PATH"
nvidia-smi

mode="run"
if [[ "$DRY_RUN" == "1" ]]; then
    mode="dry-run"
fi

"$PYTHON" --version
bash tools/run_pseudo_label_grid.sh --grid "$GRID_PATH" --mode "$mode"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[matched-global] dry-run complete"
    exit 0
fi

"$PYTHON" tools/summarize_pseudo_label_grid.py \
    --grid "$GRID_PATH" \
    --output-md "${SUMMARY_BASE}.md" \
    --output-csv "${SUMMARY_BASE}.csv" \
    --output-json "${SUMMARY_BASE}.json" \
    --title "Matched-Global Fixed Pseudo-Label Screening"

echo "[matched-global] done=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
