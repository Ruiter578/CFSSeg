#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

MODE="dry-run"
GRID_PATH=""

usage() {
    cat <<'USAGE'
Usage:
  bash tools/run_pseudo_label_artifact_calibration_grid.sh --grid <path.tsv> [--mode dry-run|run]

The TSV must contain this header:
name artifact_path task setting teacher_ckpt quantile min_conf max_conf min_pixels
shrinkage max_batches batch_size random_seed

Environment:
  PYTHON, DATA_ROOT, DATASET, CURR_STEP, LOSS_TYPE, CROP_SIZE, GPU_ID, VAL_BATCH_SIZE
  CUDA_VISIBLE_DEVICES, CUDA_MPS_PIPE_DIRECTORY, TMPDIR
  SKIP_EXISTING=1 skips valid existing artifact JSON files.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --grid)
            if [[ $# -lt 2 ]]; then
                echo "--grid requires a value" >&2
                usage >&2
                exit 2
            fi
            GRID_PATH="$2"
            shift 2
            ;;
        --mode)
            if [[ $# -lt 2 ]]; then
                echo "--mode requires a value" >&2
                usage >&2
                exit 2
            fi
            MODE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$GRID_PATH" ]]; then
    echo "--grid is required" >&2
    usage >&2
    exit 2
fi
if [[ "$MODE" != "dry-run" && "$MODE" != "run" ]]; then
    echo "--mode must be dry-run or run" >&2
    exit 2
fi
if [[ ! -f "$GRID_PATH" ]]; then
    echo "grid file does not exist: $GRID_PATH" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/home/linyichen/.mps_bypass}"
export TMPDIR="${TMPDIR:-/root/2TStorage/tmp}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data_root/VOC2012}"
DATASET="${DATASET:-voc}"
CURR_STEP="${CURR_STEP:-1}"
LOSS_TYPE="${LOSS_TYPE:-bce_loss}"
CROP_SIZE="${CROP_SIZE:-513}"
GPU_ID="${GPU_ID:-0}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$TMPDIR"

json_is_valid() {
    "$PYTHON" -m json.tool "$1" >/dev/null 2>&1
}

print_command() {
    printf '%q ' \
        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        DATA_ROOT="$DATA_ROOT" \
        "$PYTHON" \
        tools/calibrate_pseudo_label_thresholds.py \
        --data_root "$DATA_ROOT" \
        --dataset "$DATASET" \
        --task "$task" \
        --setting "$setting" \
        --curr_step "$CURR_STEP" \
        --teacher_ckpt "$teacher_ckpt" \
        --output "$artifact_path" \
        --loss_type "$LOSS_TYPE" \
        --batch_size "$batch_size" \
        --val_batch_size "$VAL_BATCH_SIZE" \
        --crop_size "$CROP_SIZE" \
        --gpu_id "$GPU_ID" \
        --random_seed "$random_seed" \
        --quantile "$quantile" \
        --min_conf "$min_conf" \
        --max_conf "$max_conf" \
        --min_pixels "$min_pixels" \
        --shrinkage "$shrinkage" \
        --max_batches "$max_batches"
    printf '\n'
}

run_row() {
    if [[ -e "$artifact_path" ]]; then
        if ! json_is_valid "$artifact_path"; then
            echo "[artifact-grid] artifact exists but is not valid JSON for row '${name}': ${artifact_path}" >&2
            exit 3
        fi
        if [[ "$SKIP_EXISTING" == "1" ]]; then
            echo "[artifact-grid] skip existing valid artifact '${name}': ${artifact_path}"
            return
        fi
    fi

    echo "[artifact-grid] row=${name} task=${task} setting=${setting} quantile=${quantile} output=${artifact_path}"
    if [[ "$MODE" == "dry-run" ]]; then
        print_command
        return
    fi

    mkdir -p "$(dirname "$artifact_path")"
    "$PYTHON" tools/calibrate_pseudo_label_thresholds.py \
        --data_root "$DATA_ROOT" \
        --dataset "$DATASET" \
        --task "$task" \
        --setting "$setting" \
        --curr_step "$CURR_STEP" \
        --teacher_ckpt "$teacher_ckpt" \
        --output "$artifact_path" \
        --loss_type "$LOSS_TYPE" \
        --batch_size "$batch_size" \
        --val_batch_size "$VAL_BATCH_SIZE" \
        --crop_size "$CROP_SIZE" \
        --gpu_id "$GPU_ID" \
        --random_seed "$random_seed" \
        --quantile "$quantile" \
        --min_conf "$min_conf" \
        --max_conf "$max_conf" \
        --min_pixels "$min_pixels" \
        --shrinkage "$shrinkage" \
        --max_batches "$max_batches"
}

declare -A seen_artifacts=()
line_no=0
while IFS=$'\t' read -r \
    name artifact_path task setting teacher_ckpt quantile min_conf max_conf min_pixels \
    shrinkage max_batches batch_size random_seed extra \
    || [[ -n "${name:-}${artifact_path:-}${task:-}${setting:-}${teacher_ckpt:-}${quantile:-}${min_conf:-}${max_conf:-}${min_pixels:-}${shrinkage:-}${max_batches:-}${batch_size:-}${random_seed:-}${extra:-}" ]]
do
    line_no=$((line_no + 1))
    random_seed="${random_seed%$'\r'}"
    extra="${extra%$'\r'}"
    if [[ "$line_no" -eq 1 ]]; then
        if [[ -n "${extra:-}" \
            || "$name" != "name" \
            || "$artifact_path" != "artifact_path" \
            || "$task" != "task" \
            || "$setting" != "setting" \
            || "$teacher_ckpt" != "teacher_ckpt" \
            || "$quantile" != "quantile" \
            || "$min_conf" != "min_conf" \
            || "$max_conf" != "max_conf" \
            || "$min_pixels" != "min_pixels" \
            || "$shrinkage" != "shrinkage" \
            || "$max_batches" != "max_batches" \
            || "$batch_size" != "batch_size" \
            || "$random_seed" != "random_seed" ]]; then
            echo "grid header is not the expected artifact calibration grid header" >&2
            exit 2
        fi
        continue
    fi
    if [[ -n "${extra:-}" ]]; then
        echo "grid line ${line_no} has too many columns" >&2
        exit 2
    fi
    if [[ -z "${name:-}" || "$name" == \#* ]]; then
        continue
    fi
    for required in name artifact_path task setting teacher_ckpt quantile min_conf max_conf min_pixels shrinkage max_batches batch_size random_seed; do
        if [[ -z "${!required:-}" ]]; then
            echo "grid line ${line_no} missing required field: ${required}" >&2
            exit 2
        fi
    done
    if [[ ! -f "$teacher_ckpt" ]]; then
        echo "grid line ${line_no} teacher checkpoint does not exist: ${teacher_ckpt}" >&2
        exit 2
    fi
    if [[ -n "${seen_artifacts[$artifact_path]:-}" ]]; then
        echo "grid line ${line_no} reuses artifact_path '${artifact_path}' from line ${seen_artifacts[$artifact_path]}" >&2
        exit 2
    fi
    seen_artifacts["$artifact_path"]="$line_no"
    run_row
done < "$GRID_PATH"
