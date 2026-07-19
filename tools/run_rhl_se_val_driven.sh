#!/usr/bin/env bash
set -euo pipefail

# P0 RHL-SE finalization runner.
# It searches class-wise weights only on val, then evaluates the selected
# weights once on test. Do not use test metrics to choose weights.

export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/home/linyichen/.mps_bypass}"
export TMPDIR="${TMPDIR:-/root/2TStorage/tmp}"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$TMPDIR"

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/root/2TStorage/lyc/SegACIL/data_root/VOC2012}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)_rhl_se_val_driven}"
OUT_DIR="${OUT_DIR:-logs/rhl_se_val_driven/${RUN_NAME}}"
OBJECTIVE="${OBJECTIVE:-all_miou}"
ENSEMBLE_MODE="${ENSEMBLE_MODE:-prob}"
DEVICE="${DEVICE:-cuda:0}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"
MAX_BATCHES="${MAX_BATCHES:--1}"

if [[ -n "${CKPTS:-}" ]]; then
    read -r -a CKPT_LIST <<< "$CKPTS"
else
    CKPT_LIST=(
        checkpoints/20260616_rhl_se_seed1/voc/15-5/sequential/step1/final.pth
        checkpoints/20260616_rhl_se_seed2/voc/15-5/sequential/step1/final.pth
        checkpoints/20260616_rhl_se_seed3/voc/15-5/sequential/step1/final.pth
    )
fi

mkdir -p "$OUT_DIR"
VAL_JSON="${OUT_DIR}/val_search.json"
CLASS_WEIGHTS_JSON="${OUT_DIR}/class_weights.json"
TEST_JSON="${OUT_DIR}/test_results.json"
DIAG_JSON="${OUT_DIR}/test_diagnostics.json"
SUMMARY_MD="${OUT_DIR}/run_summary.md"

COMMON_ARGS=(
    --data_root "$DATA_ROOT"
    --dataset voc
    --task 15-5
    --setting sequential
    --curr_step 1
    --model deeplabv3_resnet101
    --loss_type bce_loss
    --crop_size 513
    --val_batch_size "$VAL_BATCH_SIZE"
    --device "$DEVICE"
    --ensemble_mode "$ENSEMBLE_MODE"
    --max_batches "$MAX_BATCHES"
)

SEARCH_CMD=(
    "$PYTHON" tools/search_rhl_class_weights.py
    --ckpts "${CKPT_LIST[@]}"
    "${COMMON_ARGS[@]}"
    --objective "$OBJECTIVE"
    --save_json "$VAL_JSON"
    --save_class_weights_json "$CLASS_WEIGHTS_JSON"
)

EVAL_CMD=(
    "$PYTHON" tools/eval_rhl_ensemble.py
    --ckpts "${CKPT_LIST[@]}"
    "${COMMON_ARGS[@]}"
    --mode test
    --class_weights_json "$CLASS_WEIGHTS_JSON"
    --save_json "$TEST_JSON"
    --save_diagnostics "$DIAG_JSON"
)

quote_cmd() {
    printf "%q " "$@"
    printf "\n"
}

echo "Running val-driven RHL-SE weight search..."
"${SEARCH_CMD[@]}"

echo "Running final test evaluation with val-selected class weights..."
"${EVAL_CMD[@]}"

{
    echo "# RHL-SE val-driven run summary"
    echo
    echo "- run_name: ${RUN_NAME}"
    echo "- output_dir: ${OUT_DIR}"
    echo "- objective: ${OBJECTIVE}"
    echo "- ensemble_mode: ${ENSEMBLE_MODE}"
    echo "- class_weights_json: ${CLASS_WEIGHTS_JSON}"
    echo "- val_search_json: ${VAL_JSON}"
    echo "- test_results_json: ${TEST_JSON}"
    echo "- diagnostics_json: ${DIAG_JSON}"
    echo
    echo "## Checkpoints"
    for ckpt in "${CKPT_LIST[@]}"; do
        echo "- ${ckpt}"
    done
    echo
    echo "## Commands"
    echo '```bash'
    quote_cmd "${SEARCH_CMD[@]}"
    quote_cmd "${EVAL_CMD[@]}"
    echo '```'
} > "$SUMMARY_MD"

echo "Saved run summary to ${SUMMARY_MD}"
