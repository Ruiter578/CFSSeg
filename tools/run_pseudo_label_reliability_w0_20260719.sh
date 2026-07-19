#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/linyichen/miniconda3/envs/segacil/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/artifacts/pseudo_label/reliability_audit_w0_20260719}"
LOG_FILE="${LOG_FILE:-${REPO_ROOT}/logs/pseudo_label/reliability_audit_w0_20260719.log}"
DRY_RUN="${DRY_RUN:-0}"

OVERLAP_CKPT="${REPO_ROOT}/checkpoints/20260627_pseudo_15-5_overlap_batchclass_q0p7_seed1_bs32/voc/15-5/overlap/step0/deeplabv3_resnet101_voc_15-5_step_0_overlap.pth"
DISJOINT_CKPT="${REPO_ROOT}/checkpoints/20260705_pseudo_15-5_disjoint_off_seed1_bs32/voc/15-5/disjoint/step0/deeplabv3_resnet101_voc_15-5_step_0_disjoint.pth"
OVERLAP_SHA="6505c26b567cefbb9eb099af174961cbd4596d139b30a7f92f952ad494a9a913"
DISJOINT_SHA="040c69eba9be68c4f370afaf31baf3fa14a16ad50d22a1a53752e7fd3ea37962"

build_command() {
  local setting="$1"
  local checkpoint="$2"
  local expected_sha="$3"
  local threshold="$4"

  COMMAND=(
    "${PYTHON}"
    tools/audit_pseudo_label_reliability.py
    --task 15-5
    --setting "${setting}"
    --curr-step 1
    --teacher-checkpoint "${checkpoint}"
    --expected-teacher-sha256 "${expected_sha}"
    --matched-global-threshold "${threshold}"
    --max-samples 0
    --output-dir "${OUTPUT_ROOT}/${setting}"
  )
}

print_command() {
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
}

if [[ "${DRY_RUN}" == "1" ]]; then
  build_command overlap "${OVERLAP_CKPT}" "${OVERLAP_SHA}" "0.447265625"
  print_command
  build_command disjoint "${DISJOINT_CKPT}" "${DISJOINT_SHA}" "0.029296875"
  print_command
  exit 0
fi

for checkpoint in "${OVERLAP_CKPT}" "${DISJOINT_CKPT}"; do
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing teacher checkpoint: ${checkpoint}" >&2
    exit 1
  fi
done
for setting in overlap disjoint; do
  if [[ -e "${OUTPUT_ROOT}/${setting}" ]]; then
    echo "Refusing to reuse W0 output: ${OUTPUT_ROOT}/${setting}" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "${LOG_FILE}")"
exec > >(tee -a "${LOG_FILE}") 2>&1

cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/home/linyichen/.mps_bypass}"
export TMPDIR="${TMPDIR:-/root/2TStorage/tmp}"
export SEGACIL_PIN_MEMORY="${SEGACIL_PIN_MEMORY:-0}"

echo "[W0 runner] started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
df -h /root/2TStorage
nvidia-smi

build_command overlap "${OVERLAP_CKPT}" "${OVERLAP_SHA}" "0.447265625"
print_command
"${COMMAND[@]}"

build_command disjoint "${DISJOINT_CKPT}" "${DISJOINT_SHA}" "0.029296875"
print_command
"${COMMAND[@]}"

echo "[W0 runner] completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
