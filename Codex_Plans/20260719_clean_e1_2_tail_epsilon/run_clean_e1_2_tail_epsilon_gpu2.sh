#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

RUN_ID="${RUN_ID:-20260719_clean_e1_2_tail_epsilon_gpu2}"
PRIMARY_ROOT="${PRIMARY_ROOT:-$PWD}"
LOG_ROOT="${LOG_ROOT:-${PRIMARY_ROOT}/logs/${RUN_ID}}"
SUMMARY_DIR="${SUMMARY_DIR:-${PRIMARY_ROOT}/Codex_Plans/${RUN_ID}_summaries}"
DATA_ROOT="${DATA_ROOT:-${PRIMARY_ROOT}/data_root/VOC2012}"
HOLDOUT_LIST="${HOLDOUT_LIST:-${PRIMARY_ROOT}/Codex_Plans/20260716_clean_validation_protocol/voc15_5_tuning_holdout_seed20260716.txt}"
BASE_SUBPATH="${BASE_SUBPATH:-20260716_clean_validation_step0_decoder_b8224_g1_gpu2_bs16}"
BASELINE_SUBPATH="${BASELINE_SUBPATH:-20260718_clean_e1_1_replay_from_main_gpu2_decoder_b8224_g1_rhl2}"
BASELINE_EPSILON="${BASELINE_EPSILON:-1e-3}"
EXPECTED_HEAD="${EXPECTED_HEAD:-}"

MODEL="${MODEL:-deeplabv3_resnet101}"
DEFAULT_BATCH_SIZE="${DEFAULT_BATCH_SIZE:-32}"
RANDOM_SEED="${RANDOM_SEED:-1}"
AIR_FEATURE_SOURCE="${AIR_FEATURE_SOURCE:-decoder}"
BUFFER="${BUFFER:-8224}"
GAMMA="${GAMMA:-1}"
RHL_NORM="${RHL_NORM:-none}"
RHL_SEED="${RHL_SEED:-2}"
EPSILONS="${EPSILONS:-0 1e-4}"
REFERENCE_CODE_COMMIT="${REFERENCE_CODE_COMMIT:-e81a5521075a574f10f2981b9081aa89bb97a9de}"

mkdir -p "$LOG_ROOT" "$SUMMARY_DIR"
exec > >(tee -a "$LOG_ROOT/queue_master.log") 2>&1

HEAD_AT_LAUNCH="$(git rev-parse HEAD)"
if [[ -n "$EXPECTED_HEAD" && "$HEAD_AT_LAUNCH" != "$EXPECTED_HEAD" ]]; then
    echo "[ERROR] Expected code base $EXPECTED_HEAD, got $HEAD_AT_LAUNCH." >&2
    exit 1
fi
if [[ -n "$(git status --short)" ]]; then
    echo "[ERROR] Refusing to launch from a modified or untracked worktree." >&2
    git status --short >&2
    exit 1
fi
if [[ ! -d "$DATA_ROOT" || ! -f "$HOLDOUT_LIST" ]]; then
    echo "[ERROR] Missing data root or holdout list." >&2
    exit 1
fi
if [[ "$AIR_FEATURE_SOURCE" != "decoder" || "$BUFFER" != "8224" || "$GAMMA" != "1" || "$RHL_NORM" != "none" ]]; then
    echo "[ERROR] E1.2 fixes decoder, buffer=8224, gamma=1, and rhl_norm=none." >&2
    exit 1
fi

if [[ -n "$REFERENCE_CODE_COMMIT" ]]; then
    code_diff="$(
        git diff --name-only "$REFERENCE_CODE_COMMIT".."$HEAD_AT_LAUNCH" -- \
            run.sh train.py trainer network datasets utils metrics tools \
            ':!Codex_Plans' ':!AI_docs' ':!tests' || true
    )"
    if [[ -n "$code_diff" ]]; then
        echo "[ERROR] Training/evaluation source changed since $REFERENCE_CODE_COMMIT:" >&2
        printf '%s\n' "$code_diff" >&2
        exit 1
    fi
fi

BASE_MANIFEST="${PRIMARY_ROOT}/checkpoints/${BASE_SUBPATH}/voc/15-5/sequential/step0/run_manifest.json"
BASE_FINAL="${PRIMARY_ROOT}/checkpoints/${BASE_SUBPATH}/voc/15-5/sequential/step0/final.pth"
if [[ ! -f "$BASE_MANIFEST" || ! -f "$BASE_FINAL" ]]; then
    echo "[ERROR] Clean step0 manifest/checkpoint not found under: ${PRIMARY_ROOT}/checkpoints/${BASE_SUBPATH}" >&2
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

normalize_float() {
    printf '%s' "$1" | awk '{printf "%g", $1}'
}

manifest_matches_current() {
    local manifest="$1"
    local epsilon="$2"

    [[ -f "$manifest" ]] || return 1
    [[ "$(jq -r '.git.commit' "$manifest")" == "$HEAD_AT_LAUNCH" ]] || return 1
    [[ "$(jq -r '.git.dirty | tostring' "$manifest")" == "false" ]] || return 1
    [[ "$(jq -r '.requested_air_feature_source' "$manifest")" == "$AIR_FEATURE_SOURCE" ]] || return 1
    [[ "$(jq -r '.buffer | tostring' "$manifest")" == "$BUFFER" ]] || return 1
    [[ "$(jq -r '.rhl_seed | tostring' "$manifest")" == "$RHL_SEED" ]] || return 1
    [[ "$(jq -r '.model' "$manifest")" == "$MODEL" ]] || return 1
    [[ "$(jq -r '.batch_size | tostring' "$manifest")" == "$DEFAULT_BATCH_SIZE" ]] || return 1
    [[ "$(jq -r '.random_seed | tostring' "$manifest")" == "$RANDOM_SEED" ]] || return 1
    [[ "$(jq -r '.gamma | tostring' "$manifest")" == "$(normalize_float "$GAMMA")" ]] || return 1
    [[ "$(jq -r '.rhl_norm' "$manifest")" == "$RHL_NORM" ]] || return 1
    [[ "$(jq -r '.analytic_tail_epsilon | tostring' "$manifest")" == "$(normalize_float "$epsilon")" ]] || return 1
    [[ "$(jq -r '.base_subpath' "$manifest")" == "$BASE_SUBPATH" ]] || return 1
    [[ "$(jq -r '.validation_list_sha256' "$manifest")" == "$HOLDOUT_HASH" ]] || return 1
    [[ "$(jq -r '.train_exclude_list_sha256' "$manifest")" == "$HOLDOUT_HASH" ]] || return 1
}

assert_clean_worktree_for_manifest() {
    local status
    status="$(git status --short)"
    if [[ -n "$status" ]]; then
        echo "[ERROR] Worktree became visible-dirty before manifest write." >&2
        printf '%s\n' "$status" >&2
        exit 1
    fi
}

epsilon_slug() {
    case "$1" in
        0) printf 'eps0' ;;
        1e-4|0.0001) printf 'eps1em4' ;;
        1e-3|0.001) printf 'eps1em3' ;;
        *) printf 'eps%s' "$1" | tr '.+-' 'dpm' ;;
    esac
}

run_one() {
    local epsilon="$1"
    local subpath="${RUN_ID}_decoder_b${BUFFER}_g1_rhl${RHL_SEED}_$(epsilon_slug "$epsilon")"
    local step_dir="checkpoints/${subpath}/voc/15-5/sequential/step1"
    local manifest="$step_dir/run_manifest.json"
    local log_path="$LOG_ROOT/${subpath}.log"
    local result

    result="$(latest_val_result "$step_dir")"
    if [[ -n "$result" ]] && manifest_matches_current "$manifest" "$epsilon"; then
        echo "[SKIP] $subpath already has clean validation result: $result"
        return
    fi
    prepare_output_path "$subpath"
    assert_clean_worktree_for_manifest
    echo "[RUN] epsilon=$epsilon subpath=$subpath"
    set +e
    env \
        CUDA_VISIBLE_DEVICES=2 \
        DATA_ROOT="$DATA_ROOT" \
        MODEL="$MODEL" \
        AIR_FEATURE_SOURCE="$AIR_FEATURE_SOURCE" \
        BASE_SUBPATH="$BASE_SUBPATH" \
        SUBPATH="$subpath" \
        START_STEP=1 \
        END_STEP=1 \
        DEFAULT_BATCH_SIZE="$DEFAULT_BATCH_SIZE" \
        BUFFER="$BUFFER" \
        GAMMA="$GAMMA" \
        RANDOM_SEED="$RANDOM_SEED" \
        RHL_NORM="$RHL_NORM" \
        RHL_SEED="$RHL_SEED" \
        ANALYTIC_TAIL_EPSILON="$epsilon" \
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
    result="$(latest_val_result "$step_dir")"
    if [[ -z "$result" ]] || ! manifest_matches_current "$manifest" "$epsilon"; then
        echo "[ERROR] Completed run did not produce a clean matching manifest/result: $subpath" >&2
        return 1
    fi
    echo "[DONE] $subpath result=$result"
}

write_row() {
    local label="$1"
    local epsilon="$2"
    local manifest="$3"
    local result="$4"
    local output="$5"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$label" \
        "$epsilon" \
        "$(jq -r '.requested_air_feature_source' "$manifest")" \
        "$(jq -r '.buffer' "$manifest")" \
        "$(jq -r '.rhl_seed' "$manifest")" \
        "$(jq -r '."Mean IoU"' "$result")" \
        "$(jq -r '."0 to 15 mIoU"' "$result")" \
        "$(jq -r '."16 to 20 mIoU"' "$result")" \
        "$(jq -r '.git.dirty' "$manifest")" \
        "$(jq -r '.git.commit' "$manifest")" \
        "$result" >> "$output"
}

write_summary() {
    local output="$SUMMARY_DIR/final_validation_summary.tsv"
    printf 'label\tanalytic_tail_epsilon\tfeature_source\tbuffer\trhl_seed\tall_miou\told_0_15_miou\tnew_16_20_miou\tgit_dirty\tgit_commit\tresult_json\n' > "$output"

    local baseline_manifest="checkpoints/${BASELINE_SUBPATH}/voc/15-5/sequential/step1/run_manifest.json"
    local baseline_result
    baseline_result="$(latest_val_result "checkpoints/${BASELINE_SUBPATH}/voc/15-5/sequential/step1")"
    if [[ -f "$baseline_manifest" && -n "$baseline_result" ]]; then
        write_row baseline "$BASELINE_EPSILON" "$baseline_manifest" "$baseline_result" "$output"
    else
        echo "[WARN] Baseline result not found: $BASELINE_SUBPATH" >&2
    fi

    while IFS= read -r manifest; do
        local step_dir result epsilon
        step_dir="$(dirname "$manifest")"
        result="$(latest_val_result "$step_dir")"
        [[ -n "$result" ]] || continue
        epsilon="$(jq -r '.analytic_tail_epsilon' "$manifest")"
        write_row current "$epsilon" "$manifest" "$result" "$output"
    done < <(find -L checkpoints -path "checkpoints/${RUN_ID}_*/voc/15-5/sequential/step1/run_manifest.json" | sort)

    echo "[SUMMARY] $output"
    cat "$output"
}

echo "[INFO] clean E1.2 tail epsilon head=$HEAD_AT_LAUNCH run_id=$RUN_ID primary_root=$PRIMARY_ROOT"
echo "[INFO] epsilons=$EPSILONS baseline=${BASELINE_SUBPATH}@${BASELINE_EPSILON}"

for epsilon in $EPSILONS; do
    run_one "$epsilon"
done

write_summary
echo "[DONE] Clean E1.2 tail epsilon queue completed."
