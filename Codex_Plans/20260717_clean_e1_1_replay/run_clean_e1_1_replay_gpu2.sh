#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

RUN_ID="${RUN_ID:-20260717_clean_e1_1_replay_v2_gpu2}"
PRIMARY_ROOT="${PRIMARY_ROOT:-$PWD}"
LOG_ROOT="${LOG_ROOT:-${PRIMARY_ROOT}/logs/${RUN_ID}}"
SUMMARY_DIR="${SUMMARY_DIR:-$PWD/Codex_Plans/${RUN_ID}_summaries}"
DATA_ROOT="${DATA_ROOT:-$PWD/data_root/VOC2012}"
HOLDOUT_LIST="${HOLDOUT_LIST:-$PWD/Codex_Plans/20260716_clean_validation_protocol/voc15_5_tuning_holdout_seed20260716.txt}"
BASE_SUBPATH="${BASE_SUBPATH:-20260716_clean_validation_step0_decoder_b8224_g1_gpu2_bs16}"
EXPECTED_HEAD="${EXPECTED_HEAD:-2b13e2f6f440a27cdcc769c2ce001aadc0bdd0d3}"

MODEL="${MODEL:-deeplabv3_resnet101}"
DEFAULT_BATCH_SIZE="${DEFAULT_BATCH_SIZE:-32}"
RANDOM_SEED="${RANDOM_SEED:-1}"
GAMMA="${GAMMA:-1}"
RHL_NORM="${RHL_NORM:-none}"
ANALYTIC_TAIL_EPSILON="${ANALYTIC_TAIL_EPSILON:-1e-3}"

mkdir -p "$LOG_ROOT" "$SUMMARY_DIR"
exec > >(tee -a "$LOG_ROOT/queue_master.log") 2>&1

if [[ "$(git rev-parse HEAD)" != "$EXPECTED_HEAD" ]]; then
    echo "[ERROR] Expected code base $EXPECTED_HEAD, got $(git rev-parse HEAD)." >&2
    exit 1
fi
if ! git diff --quiet || ! git diff --cached --quiet || [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
    echo "[ERROR] Refusing to replay from a modified or untracked worktree." >&2
    exit 1
fi
if [[ ! -d "$DATA_ROOT" || ! -f "$HOLDOUT_LIST" ]]; then
    echo "[ERROR] Missing data root or holdout list." >&2
    exit 1
fi
if [[ "$RHL_NORM" != "none" || "$GAMMA" != "1" ]]; then
    echo "[ERROR] Clean replay fixes gamma=1 and rhl_norm=none." >&2
    exit 1
fi

BASE_MANIFEST="${PRIMARY_ROOT}/checkpoints/${BASE_SUBPATH}/voc/15-5/sequential/step0/run_manifest.json"
if [[ ! -f "$BASE_MANIFEST" ]]; then
    echo "[ERROR] Clean step0 manifest not found: $BASE_MANIFEST" >&2
    exit 1
fi
HOLDOUT_HASH="$(sha256sum "$HOLDOUT_LIST" | awk '{print $1}')"
if [[ "$(jq -r '.validation_list_sha256' "$BASE_MANIFEST")" != "$HOLDOUT_HASH" ]]; then
    echo "[ERROR] Clean step0 and current holdout hashes do not match." >&2
    exit 1
fi

prepare_output_path() {
    local subpath="$1"
    local local_path="checkpoints/${subpath}"
    local primary_root_real
    local primary_path

    primary_root_real="$(mkdir -p "$PRIMARY_ROOT" && cd "$PRIMARY_ROOT" && pwd -P)"
    if [[ "$primary_root_real" == "$(pwd -P)" ]]; then
        if [[ -e "$local_path" || -L "$local_path" ]]; then
            echo "[ERROR] Refusing to reuse existing output path: $local_path" >&2
            exit 1
        fi
        mkdir -p "$local_path"
        return
    fi
    primary_path="${primary_root_real}/checkpoints/${subpath}"

    if [[ -e "$local_path" || -L "$local_path" ]]; then
        echo "[ERROR] Refusing to reuse existing local output path: $local_path" >&2
        exit 1
    fi
    if [[ -e "$primary_path" || -L "$primary_path" ]]; then
        echo "[ERROR] Refusing to reuse existing primary output path: $primary_path" >&2
        exit 1
    fi
    mkdir -p "$primary_path" "$(dirname "$local_path")"
    ln -s "$primary_path" "$local_path"
}

latest_val_result() {
    local step_dir="$1"
    if [[ ! -d "$step_dir" ]]; then
        return 0
    fi
    find "$step_dir" -maxdepth 1 -name 'val_results_*.json' -type f 2>/dev/null | sort | tail -n 1
}

manifest_matches_current() {
    local manifest="$1"
    local source="$2"
    local buffer="$3"
    local rhl_seed="$4"

    [[ -f "$manifest" ]] || return 1
    [[ "$(jq -r '.requested_air_feature_source' "$manifest")" == "$source" ]] || return 1
    [[ "$(jq -r '.buffer | tostring' "$manifest")" == "$buffer" ]] || return 1
    [[ "$(jq -r '.rhl_seed | tostring' "$manifest")" == "$rhl_seed" ]] || return 1
    [[ "$(jq -r '.model' "$manifest")" == "$MODEL" ]] || return 1
    [[ "$(jq -r '.batch_size | tostring' "$manifest")" == "$DEFAULT_BATCH_SIZE" ]] || return 1
    [[ "$(jq -r '.random_seed | tostring' "$manifest")" == "$RANDOM_SEED" ]] || return 1
    [[ "$(jq -r '.gamma | tostring' "$manifest")" == "$(printf '%s' "$GAMMA" | awk '{printf \"%g\", $1}')" ]] || return 1
    [[ "$(jq -r '.rhl_norm' "$manifest")" == "$RHL_NORM" ]] || return 1
    [[ "$(jq -r '.analytic_tail_epsilon | tostring' "$manifest")" == "$(printf '%s' "$ANALYTIC_TAIL_EPSILON" | awk '{printf \"%g\", $1}')" ]] || return 1
    [[ "$(jq -r '.base_subpath' "$manifest")" == "$BASE_SUBPATH" ]] || return 1
    [[ "$(jq -r '.validation_list_sha256' "$manifest")" == "$HOLDOUT_HASH" ]] || return 1
    [[ "$(jq -r '.train_exclude_list_sha256' "$manifest")" == "$HOLDOUT_HASH" ]] || return 1
}

run_one() {
    local source="$1"
    local buffer="$2"
    local rhl_seed="$3"
    local subpath="${RUN_ID}_${source}_b${buffer}_g1_rhl${rhl_seed}"
    local step_dir="checkpoints/${subpath}/voc/15-5/sequential/step1"
    local manifest="$step_dir/run_manifest.json"
    local log_path="$LOG_ROOT/${subpath}.log"
    local result

    result="$(latest_val_result "$step_dir")"
    if [[ -n "$result" ]] && manifest_matches_current "$manifest" "$source" "$buffer" "$rhl_seed"; then
        echo "[SKIP] $subpath already has validation result: $result"
        return
    fi
    prepare_output_path "$subpath"
    echo "[RUN] source=$source buffer=$buffer rhl_seed=$rhl_seed subpath=$subpath"
    set +e
    env \
        CUDA_VISIBLE_DEVICES=2 \
        DATA_ROOT="$DATA_ROOT" \
        MODEL="$MODEL" \
        AIR_FEATURE_SOURCE="$source" \
        BASE_SUBPATH="$BASE_SUBPATH" \
        SUBPATH="$subpath" \
        START_STEP=1 \
        END_STEP=1 \
        DEFAULT_BATCH_SIZE="$DEFAULT_BATCH_SIZE" \
        BUFFER="$buffer" \
        GAMMA="$GAMMA" \
        RANDOM_SEED="$RANDOM_SEED" \
        RHL_NORM="$RHL_NORM" \
        RHL_SEED="$rhl_seed" \
        ANALYTIC_TAIL_EPSILON="$ANALYTIC_TAIL_EPSILON" \
        EVALUATION_MODE=val \
        TRAIN_EXCLUDE_LIST="$HOLDOUT_LIST" \
        VALIDATION_LIST="$HOLDOUT_LIST" \
        bash run.sh 2>&1 | tee "$log_path"
    local pipeline_status=("${PIPESTATUS[@]}")
    set -e
    if (( pipeline_status[1] != 0 )); then
        echo "[ERROR] Failed to write log: $log_path" >&2
        return "${pipeline_status[1]}"
    fi
    if (( pipeline_status[0] != 0 )); then
        echo "[ERROR] Training failed: $subpath" >&2
        return "${pipeline_status[0]}"
    fi
    echo "[DONE] $subpath"
}

write_summary() {
    local output="$SUMMARY_DIR/final_validation_summary.tsv"
    printf 'feature_source\tbuffer\trhl_seed\tall_miou\told_0_15_miou\tnew_16_20_miou\tsubpath\tresult_json\n' > "$output"
    while IFS= read -r manifest; do
        local step_dir result
        step_dir="$(dirname "$manifest")"
        result="$(latest_val_result "$step_dir")"
        [[ -n "$result" ]] || continue
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$(jq -r '.requested_air_feature_source' "$manifest")" \
            "$(jq -r '.buffer' "$manifest")" \
            "$(jq -r '.rhl_seed' "$manifest")" \
            "$(jq -r '."Mean IoU"' "$result")" \
            "$(jq -r '."0 to 15 mIoU"' "$result")" \
            "$(jq -r '."16 to 20 mIoU"' "$result")" \
            "$(jq -r '.subpath' "$manifest")" \
            "$result" >> "$output"
    done < <(find -L checkpoints -path "checkpoints/${RUN_ID}_*/voc/15-5/sequential/step1/run_manifest.json" | sort)
    echo "[SUMMARY] $output"
    cat "$output"
}

# Rebuild the complete original comparison under the clean step0/holdout protocol.
for source in decoder aspp; do
    for rhl_seed in 2 4 5; do
        run_one "$source" 8224 "$rhl_seed"
    done
done
run_one aspp 8208 2
run_one aspp 8240 2

write_summary
echo "[DONE] Clean eight-candidate replay completed."
