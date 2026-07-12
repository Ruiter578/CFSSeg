#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_ID="${RUN_ID:-20260711_buffer_final_confirmation_trs}"
LOG_ROOT="${LOG_ROOT:-logs/$RUN_ID}"
SUMMARY_DIR="${SUMMARY_DIR:-Codex_Plans/${RUN_ID}_summaries}"
EXPECTED_HEAD="${EXPECTED_HEAD:-$(git rev-parse HEAD)}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
DATA_ROOT="${DATA_ROOT:-/TRS-SAS/linwei/SegACIL/data_root/VOC2012}"
BASE_SUBPATH="${BASE_SUBPATH:-20260621_baseline_bs16_16_trs}"

mkdir -p "$LOG_ROOT" "$SUMMARY_DIR" /tmp/mps_bypass

check_code_state() {
    local current_head
    current_head="$(git rev-parse HEAD)"
    if [[ "$current_head" != "$EXPECTED_HEAD" ]]; then
        echo "[ERROR] Git HEAD changed: expected=$EXPECTED_HEAD current=$current_head" >&2
        exit 1
    fi
    if ! git diff --quiet || ! git diff --cached --quiet; then
        echo "[ERROR] Tracked working tree changed; refusing to mix code states." >&2
        exit 1
    fi
}

latest_result_json() {
    local step_dir="$1"
    [[ -d "$step_dir" ]] || return 0
    find "$step_dir" -maxdepth 1 -name 'test_results_*.json' 2>/dev/null | sort | tail -n 1
}

run_one() {
    local buffer="$1"
    local rhl_seed="$2"
    local subpath="${RUN_ID}_b${buffer}_g1_rhl${rhl_seed}_trs"
    local step_dir="checkpoints/${subpath}/voc/15-5/sequential/step1"
    local log_path="${LOG_ROOT}/${subpath}.log"
    local existing_result

    check_code_state
    existing_result="$(latest_result_json "$step_dir")"
    if [[ -n "$existing_result" ]]; then
        echo "[SKIP] $subpath already has result: $existing_result"
        return 0
    fi

    echo "[RUN] buffer=$buffer gamma=1 rhl_norm=none rhl_seed=$rhl_seed head=$EXPECTED_HEAD"
    env \
        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_bypass \
        TMPDIR=/tmp \
        DATA_ROOT="$DATA_ROOT" \
        BASE_SUBPATH="$BASE_SUBPATH" \
        SUBPATH="$subpath" \
        START_STEP=1 \
        END_STEP=1 \
        DEFAULT_BATCH_SIZE=32 \
        SPECIAL_BATCH_SIZE=32 \
        BUFFER="$buffer" \
        GAMMA=1 \
        RANDOM_SEED=1 \
        RHL_NORM=none \
        RHL_SEED="$rhl_seed" \
        RHL_STATS=0 \
        USE_PSEUDO_LABEL=0 \
        bash run.sh 2>&1 | tee "$log_path"
    echo "[DONE] $subpath"
}

collect_summary() {
    local output="$SUMMARY_DIR/final_summary.tsv"
    printf 'buffer\trhl_seed\tall_miou\told_0_15_miou\tnew_16_20_miou\tsubpath\tresult_json\n' > "$output"
    local manifest step_dir result subpath buffer seed
    while IFS= read -r manifest; do
        step_dir="$(dirname "$manifest")"
        result="$(latest_result_json "$step_dir")"
        [[ -n "$result" ]] || continue
        subpath="$(jq -r '.subpath // .args.subpath' "$manifest")"
        buffer="$(jq -r '.buffer // .args.buffer' "$manifest")"
        seed="$(jq -r '.rhl_seed // .args.rhl_seed' "$manifest")"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$buffer" "$seed" \
            "$(jq -r '."Mean IoU"' "$result")" \
            "$(jq -r '."0 to 15 mIoU"' "$result")" \
            "$(jq -r '."16 to 20 mIoU"' "$result")" \
            "$subpath" "$result" >> "$output"
    done < <(find checkpoints -path "checkpoints/${RUN_ID}_*/voc/15-5/sequential/step1/run_manifest.json" | sort)
    echo "[INFO] final summary: $output"
    cat "$output"
}

check_code_state
echo "[INFO] run_id=$RUN_ID head=$EXPECTED_HEAD gpu=$CUDA_VISIBLE_DEVICES"
run_one 8224 2
run_one 8232 4
run_one 8232 5
collect_summary
echo "[DONE] Buffer final confirmation queue finished."
