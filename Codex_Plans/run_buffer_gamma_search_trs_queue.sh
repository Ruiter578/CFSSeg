#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_ID="${RUN_ID:-20260705_buffer_gamma_search_trs}"
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
RHL_NORM="${RHL_NORM:-none}"
RHL_STATS="${RHL_STATS:-0}"
DEFAULT_BATCH_SIZE="${DEFAULT_BATCH_SIZE:-32}"
SPECIAL_BATCH_SIZE="${SPECIAL_BATCH_SIZE:-32}"
TOP_K="${TOP_K:-2}"

# 2026-07-06 protocol adjustment: keep gamma fixed at 1.
# The active search axis is buffer; Phase B varies only rhl_seed.
BUFFERS_PHASE_A=(8192 8196 8200 8204 8208 8212 8216 8220 8224)
GAMMAS_PHASE_A=(1)
RHL_SEEDS_PHASE_B=(1 2 3)

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

gamma_is_active() {
    local value="$1"
    local active
    for active in "${GAMMAS_PHASE_A[@]}"; do
        if [[ "$value" == "$active" || "$value" == "${active}.0" ]]; then
            return 0
        fi
    done
    return 1
}

subpath_for() {
    local phase="$1"
    local buffer="$2"
    local gamma="$3"
    local rhl_seed="$4"
    printf '%s_%s_b%s_g%s_rhl%s_trs' \
        "$EXP_PREFIX" "$phase" "$buffer" "$(gamma_tag "$gamma")" "$(seed_tag "$rhl_seed")"
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
    local gamma="$3"
    local rhl_seed="$4"
    local subpath
    local step_dir
    local log_path
    local existing_result

    subpath="$(subpath_for "$phase" "$buffer" "$gamma" "$rhl_seed")"
    step_dir="checkpoints/${subpath}/voc/15-5/sequential/step1"
    log_path="${LOG_ROOT}/${subpath}.log"
    existing_result="$(latest_result_json "$step_dir")"

    if [[ -n "$existing_result" ]]; then
        echo "[SKIP] ${subpath} already has result: ${existing_result}"
        return 0
    fi

    echo "[RUN] phase=${phase} buffer=${buffer} gamma=${gamma} rhl_seed=${rhl_seed} subpath=${subpath}"
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
        GAMMA="$gamma" \
        RANDOM_SEED="$RANDOM_SEED" \
        RHL_NORM="$RHL_NORM" \
        RHL_SEED="$rhl_seed" \
        RHL_STATS="$RHL_STATS" \
        bash run.sh 2>&1 | tee "$log_path"
    echo "[DONE] ${subpath}"
}

collect_summary() {
    local output_path="$1"
    shift
    local phases=("$@")
    local manifest
    local step_dir
    local result_json
    local phase
    local buffer
    local gamma
    local rhl_seed
    local all_miou
    local old_miou
    local new_miou
    local subpath
    local base_hash

    printf 'buffer\tgamma\trhl_seed\tall_miou\told_0_15_miou\tnew_16_20_miou\tsubpath\tresult_json\tbase_checkpoint_sha256\n' > "$output_path"

    for phase in "${phases[@]}"; do
        while IFS= read -r manifest; do
            step_dir="$(dirname "$manifest")"
            result_json="$(latest_result_json "$step_dir")"
            if [[ -z "$result_json" ]]; then
                continue
            fi

            buffer="$(jq -r '.buffer // .args.buffer' "$manifest")"
            gamma="$(jq -r '.gamma // .args.gamma' "$manifest")"
            if ! gamma_is_active "$gamma"; then
                continue
            fi
            rhl_seed="$(jq -r '.rhl_seed // .args.rhl_seed' "$manifest")"
            subpath="$(jq -r '.subpath // .args.subpath' "$manifest")"
            base_hash="$(jq -r '.base_checkpoint_sha256 // .resolved_paths.base_checkpoint_sha256 // ""' "$manifest")"
            all_miou="$(jq -r '."Mean IoU"' "$result_json")"
            old_miou="$(jq -r '."0 to 15 mIoU"' "$result_json")"
            new_miou="$(jq -r '."16 to 20 mIoU"' "$result_json")"

            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$buffer" "$gamma" "$rhl_seed" "$all_miou" "$old_miou" "$new_miou" \
                "$subpath" "$result_json" "$base_hash" >> "$output_path"
        done < <(find checkpoints -path "checkpoints/${EXP_PREFIX}_${phase}_*/voc/15-5/sequential/step1/run_manifest.json" | sort)
    done
}

write_mean_by_combo() {
    local input_path="$1"
    local output_path="$2"

    {
        printf 'buffer\tgamma\tcount\tmean_all_miou\tmean_old_0_15_miou\tmean_new_16_20_miou\n'
        tail -n +2 "$input_path" | awk -F '\t' '
            {
                key=$1 FS $2
                count[key] += 1
                all[key] += $4
                old[key] += $5
                newc[key] += $6
            }
            END {
                for (key in count) {
                    split(key, parts, FS)
                    printf "%s\t%s\t%d\t%.10f\t%.10f\t%.10f\n", parts[1], parts[2], count[key], all[key]/count[key], old[key]/count[key], newc[key]/count[key]
                }
            }
        ' | sort -t $'\t' -k4,4nr
    } > "$output_path"
}

echo "[INFO] run_id=${RUN_ID}"
echo "[INFO] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] data_root=${DATA_ROOT}"
echo "[INFO] base_subpath=${BASE_SUBPATH}"
echo "[INFO] phase A buffers=${BUFFERS_PHASE_A[*]} gammas=${GAMMAS_PHASE_A[*]}"

for buffer in "${BUFFERS_PHASE_A[@]}"; do
    for gamma in "${GAMMAS_PHASE_A[@]}"; do
        run_one phaseA "$buffer" "$gamma" -1
    done
done

PHASE_A_SUMMARY="${SUMMARY_DIR}/phaseA_summary.tsv"
collect_summary "$PHASE_A_SUMMARY" phaseA
echo "[INFO] phase A summary: ${PHASE_A_SUMMARY}"
column -t -s $'\t' "$PHASE_A_SUMMARY" || cat "$PHASE_A_SUMMARY"

mapfile -t TOP_COMBOS < <(
    tail -n +2 "$PHASE_A_SUMMARY" \
        | sort -t $'\t' -k4,4nr \
        | awk -F '\t' -v limit="$TOP_K" '!seen[$1 FS $2]++ {print $1 "\t" $2; n += 1; if (n >= limit) exit}'
)

if [[ "${#TOP_COMBOS[@]}" -eq 0 ]]; then
    echo "[ERROR] No completed phase A results found; cannot start phase B." >&2
    exit 1
fi

echo "[INFO] phase B top combos:"
printf '%s\n' "${TOP_COMBOS[@]}"

for combo in "${TOP_COMBOS[@]}"; do
    IFS=$'\t' read -r buffer gamma <<< "$combo"
    for rhl_seed in "${RHL_SEEDS_PHASE_B[@]}"; do
        run_one phaseB "$buffer" "$gamma" "$rhl_seed"
    done
done

FINAL_SUMMARY="${SUMMARY_DIR}/final_summary.tsv"
FINAL_MEAN="${SUMMARY_DIR}/final_mean_by_combo.tsv"
collect_summary "$FINAL_SUMMARY" phaseA phaseB
write_mean_by_combo "$FINAL_SUMMARY" "$FINAL_MEAN"

echo "[INFO] final summary: ${FINAL_SUMMARY}"
column -t -s $'\t' "$FINAL_SUMMARY" || cat "$FINAL_SUMMARY"
echo "[INFO] final mean by combo: ${FINAL_MEAN}"
column -t -s $'\t' "$FINAL_MEAN" || cat "$FINAL_MEAN"
echo "[DONE] Buffer-gamma search queue finished."
