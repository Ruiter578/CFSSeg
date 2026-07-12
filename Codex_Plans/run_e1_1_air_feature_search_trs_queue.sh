#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_ID="${RUN_ID:-20260712_e1_1_air_feature_search_trs}"
LOG_ROOT="${LOG_ROOT:-logs/$RUN_ID}"
SUMMARY_DIR="${SUMMARY_DIR:-Codex_Plans/${RUN_ID}_summaries}"
EXPECTED_HEAD="${EXPECTED_HEAD:-$(git rev-parse HEAD)}"
EXPECTED_BASE_SHA256="fb48c926cde03a35c1daf7ec6b9fe95340e932ddf5c4c2226a9c432f87fa244e"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
DATA_ROOT="${DATA_ROOT:-/TRS-SAS/linwei/SegACIL/data_root/VOC2012}"
BASE_SUBPATH="${BASE_SUBPATH:-20260621_baseline_bs16_16_trs}"
BASE_CHECKPOINT="checkpoints/${BASE_SUBPATH}/voc/15-5/sequential/step0/deeplabv3_resnet101_voc_15-5_step_0_sequential.pth"

FEATURE_SOURCES=(decoder aspp)
RHL_SEEDS=(2 4 5)

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

check_base_checkpoint() {
    local actual_sha256
    if [[ ! -f "$BASE_CHECKPOINT" ]]; then
        echo "[ERROR] Missing base checkpoint: $BASE_CHECKPOINT" >&2
        exit 1
    fi
    actual_sha256="$(sha256sum "$BASE_CHECKPOINT" | awk '{print $1}')"
    if [[ "$actual_sha256" != "$EXPECTED_BASE_SHA256" ]]; then
        echo "[ERROR] Base checkpoint hash mismatch: expected=$EXPECTED_BASE_SHA256 actual=$actual_sha256" >&2
        exit 1
    fi
}

latest_val_json() {
    local step_dir="$1"
    [[ -d "$step_dir" ]] || return 0
    find "$step_dir" -maxdepth 1 -name 'val_results_*.json' 2>/dev/null | sort | tail -n 1
}

run_one() {
    local source="$1"
    local seed="$2"
    local subpath="${RUN_ID}_${source}_b8224_g1_rhl${seed}_trs"
    local step_dir="checkpoints/${subpath}/voc/15-5/sequential/step1"
    local log_path="${LOG_ROOT}/${subpath}.log"
    local existing_result

    check_code_state
    check_base_checkpoint
    existing_result="$(latest_val_json "$step_dir")"
    if [[ -n "$existing_result" ]]; then
        echo "[SKIP] $subpath already has validation result: $existing_result"
        return 0
    fi

    echo "[RUN] source=$source rhl_seed=$seed buffer=8224 gamma=1 evaluation=val head=$EXPECTED_HEAD"
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
        AIR_FEATURE_SOURCE="$source" \
        BUFFER=8224 \
        GAMMA=1 \
        RANDOM_SEED=1 \
        RHL_NORM=none \
        RHL_SEED="$seed" \
        RHL_STATS=0 \
        ANALYTIC_TAIL_EPSILON=1e-3 \
        EVALUATION_MODE=val \
        bash run.sh 2>&1 | tee "$log_path"
    echo "[DONE] $subpath"
}

collect_summary() {
    local output="$SUMMARY_DIR/final_validation_summary.tsv"
    local manifest step_dir result source seed subpath
    printf 'feature_source\trhl_seed\tall_miou\told_0_15_miou\tnew_16_20_miou\tsubpath\tval_json\tgit_commit\tbase_checkpoint_sha256\n' > "$output"

    while IFS= read -r manifest; do
        step_dir="$(dirname "$manifest")"
        result="$(latest_val_json "$step_dir")"
        [[ -n "$result" ]] || continue
        source="$(jq -r '.resolved_air_feature_source // .air.resolved_feature_source' "$manifest")"
        seed="$(jq -r '.rhl_seed // .args.rhl_seed' "$manifest")"
        subpath="$(jq -r '.subpath // .args.subpath' "$manifest")"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$source" "$seed" \
            "$(jq -r '."Mean IoU"' "$result")" \
            "$(jq -r '."0 to 15 mIoU"' "$result")" \
            "$(jq -r '."16 to 20 mIoU"' "$result")" \
            "$subpath" "$result" \
            "$(jq -r '.git.commit // .git_commit' "$manifest")" \
            "$(jq -r '.base_checkpoint_sha256 // .resolved_paths.base_checkpoint_sha256' "$manifest")" \
            >> "$output"
    done < <(find checkpoints -path "checkpoints/${RUN_ID}_*/voc/15-5/sequential/step1/run_manifest.json" | sort)

    echo "[INFO] final validation summary: $output"
    cat "$output"
}

check_code_state
check_base_checkpoint
echo "[INFO] run_id=$RUN_ID head=$EXPECTED_HEAD gpu=$CUDA_VISIBLE_DEVICES"
echo "[INFO] matrix: sources=${FEATURE_SOURCES[*]} rhl_seeds=${RHL_SEEDS[*]}"

for seed in "${RHL_SEEDS[@]}"; do
    for source in "${FEATURE_SOURCES[@]}"; do
        run_one "$source" "$seed"
    done
done

collect_summary
echo "[DONE] E1.1 AIR feature-source validation queue finished."
