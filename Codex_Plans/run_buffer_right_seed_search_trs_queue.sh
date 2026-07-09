#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_ID="${RUN_ID:-20260709_buffer_right_seed_search_trs}"
EXP_PREFIX="${EXP_PREFIX:-$RUN_ID}"
LOG_ROOT="${LOG_ROOT:-logs/$RUN_ID}"
SUMMARY_DIR="${SUMMARY_DIR:-Codex_Plans/${RUN_ID}_summaries}"

PYTHON="${PYTHON:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
DATA_ROOT="${DATA_ROOT:-/TRS-SAS/linwei/SegACIL/data_root/VOC2012}"
BASE_SUBPATH="${BASE_SUBPATH:-20260621_baseline_bs16_16_trs}"
MODEL="${MODEL:-deeplabv3_resnet101}"
AIR_FEATURE_SOURCE="${AIR_FEATURE_SOURCE:-auto}"
RANDOM_SEED="${RANDOM_SEED:-1}"
GAMMA="${GAMMA:-1}"
RHL_NORM="${RHL_NORM:-none}"
RHL_STATS="${RHL_STATS:-0}"
DEFAULT_BATCH_SIZE="${DEFAULT_BATCH_SIZE:-32}"
SPECIAL_BATCH_SIZE="${SPECIAL_BATCH_SIZE:-32}"

BUFFERS_RIGHT=(8228 8232 8236 8240)
RIGHT_RHL_SEED=2
BUFFERS_STABILITY=(8208 8224)
STABILITY_RHL_SEEDS=(4 5)

mkdir -p "$LOG_ROOT" "$SUMMARY_DIR" /tmp/mps_bypass

gamma_tag() {
    local value="$1"
    value="${value//./p}"
    value="${value//-/m}"
    printf '%s' "$value"
}

seed_tag() {
    local value="$1"
    value="${value//-/m}"
    printf '%s' "$value"
}

subpath_for() {
    local phase="$1"
    local buffer="$2"
    local rhl_seed="$3"
    printf '%s_%s_b%s_g%s_rhl%s_trs' \
        "$EXP_PREFIX" "$phase" "$buffer" "$(gamma_tag "$GAMMA")" "$(seed_tag "$rhl_seed")"
}

latest_result_json() {
    local step_dir="$1"
    if [[ ! -d "$step_dir" ]]; then
        return 0
    fi
    find "$step_dir" -maxdepth 1 -name 'test_results_*.json' 2>/dev/null | sort | tail -n 1
}

run_one() {
    local phase="$1"
    local buffer="$2"
    local rhl_seed="$3"
    local subpath
    local step_dir
    local log_path
    local existing_result

    subpath="$(subpath_for "$phase" "$buffer" "$rhl_seed")"
    step_dir="checkpoints/${subpath}/voc/15-5/sequential/step1"
    log_path="${LOG_ROOT}/${subpath}.log"
    existing_result="$(latest_result_json "$step_dir")"

    if [[ -n "$existing_result" ]]; then
        echo "[SKIP] ${subpath} already has result: ${existing_result}"
        return 0
    fi

    echo "[RUN] phase=${phase} buffer=${buffer} gamma=${GAMMA} rhl_norm=${RHL_NORM} rhl_seed=${rhl_seed} subpath=${subpath}"
    env \
        PYTHON="$PYTHON" \
        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_bypass \
        TMPDIR=/tmp \
        DATA_ROOT="$DATA_ROOT" \
        MODEL="$MODEL" \
        AIR_FEATURE_SOURCE="$AIR_FEATURE_SOURCE" \
        BASE_SUBPATH="$BASE_SUBPATH" \
        SUBPATH="$subpath" \
        START_STEP=1 \
        END_STEP=1 \
        DEFAULT_BATCH_SIZE="$DEFAULT_BATCH_SIZE" \
        SPECIAL_BATCH_SIZE="$SPECIAL_BATCH_SIZE" \
        BUFFER="$buffer" \
        GAMMA="$GAMMA" \
        RANDOM_SEED="$RANDOM_SEED" \
        RHL_NORM="$RHL_NORM" \
        RHL_SEED="$rhl_seed" \
        RHL_STATS="$RHL_STATS" \
        bash run.sh 2>&1 | tee "$log_path"
    echo "[DONE] ${subpath}"
}

collect_summary() {
    local output_path="$1"
    local manifest
    local step_dir
    local result_json
    local phase
    local buffer
    local gamma
    local rhl_seed
    local rhl_norm
    local all_miou
    local old_miou
    local new_miou
    local subpath
    local base_hash

    printf 'phase\tbuffer\tgamma\trhl_norm\trhl_seed\tall_miou\told_0_15_miou\tnew_16_20_miou\tsubpath\tresult_json\tbase_checkpoint_sha256\n' > "$output_path"

    while IFS= read -r manifest; do
        step_dir="$(dirname "$manifest")"
        result_json="$(latest_result_json "$step_dir")"
        if [[ -z "$result_json" ]]; then
            continue
        fi

        subpath="$(jq -r '.subpath // .args.subpath' "$manifest")"
        phase="${subpath#${EXP_PREFIX}_}"
        phase="${phase%%_b*}"
        buffer="$(jq -r '.buffer // .args.buffer' "$manifest")"
        gamma="$(jq -r '.gamma // .args.gamma' "$manifest")"
        rhl_norm="$(jq -r '.rhl_norm // .args.rhl_norm // ""' "$manifest")"
        rhl_seed="$(jq -r '.rhl_seed // .args.rhl_seed' "$manifest")"
        base_hash="$(jq -r '.base_checkpoint_sha256 // .resolved_paths.base_checkpoint_sha256 // ""' "$manifest")"
        all_miou="$(jq -r '."Mean IoU"' "$result_json")"
        old_miou="$(jq -r '."0 to 15 mIoU"' "$result_json")"
        new_miou="$(jq -r '."16 to 20 mIoU"' "$result_json")"

        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$phase" "$buffer" "$gamma" "$rhl_norm" "$rhl_seed" "$all_miou" "$old_miou" "$new_miou" \
            "$subpath" "$result_json" "$base_hash" >> "$output_path"
    done < <(find checkpoints -path "checkpoints/${EXP_PREFIX}_*/voc/15-5/sequential/step1/run_manifest.json" | sort)
}

echo "[INFO] run_id=${RUN_ID}"
echo "[INFO] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] data_root=${DATA_ROOT}"
echo "[INFO] base_subpath=${BASE_SUBPATH}"
echo "[INFO] fixed gamma=${GAMMA} rhl_norm=${RHL_NORM} random_seed=${RANDOM_SEED}"
echo "[INFO] right extension buffers=${BUFFERS_RIGHT[*]} rhl_seed=${RIGHT_RHL_SEED}"
echo "[INFO] stability buffers=${BUFFERS_STABILITY[*]} rhl_seeds=${STABILITY_RHL_SEEDS[*]}"

if [[ "$GAMMA" != "1" && "$GAMMA" != "1.0" ]]; then
    echo "[ERROR] This clean buffer queue is intended to keep gamma fixed at 1; got GAMMA=${GAMMA}." >&2
    exit 1
fi

if [[ "$RHL_NORM" != "none" ]]; then
    echo "[ERROR] This clean buffer queue is intended to keep rhl_norm fixed at none; got RHL_NORM=${RHL_NORM}." >&2
    exit 1
fi

for buffer in "${BUFFERS_RIGHT[@]}"; do
    run_one right "$buffer" "$RIGHT_RHL_SEED"
done

for buffer in "${BUFFERS_STABILITY[@]}"; do
    for rhl_seed in "${STABILITY_RHL_SEEDS[@]}"; do
        run_one stability "$buffer" "$rhl_seed"
    done
done

FINAL_SUMMARY="${SUMMARY_DIR}/final_summary.tsv"
collect_summary "$FINAL_SUMMARY"
echo "[INFO] final summary: ${FINAL_SUMMARY}"
cat "$FINAL_SUMMARY"
echo "[DONE] Clean buffer right-extension and seed-stability queue finished."
