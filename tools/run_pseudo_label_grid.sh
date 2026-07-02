#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

MODE="dry-run"
GRID_PATH=""

usage() {
    cat <<'USAGE'
Usage:
  bash tools/run_pseudo_label_grid.sh --grid <path.tsv> [--mode dry-run|run]

The TSV must contain this header:
name subpath task setting strategy confidence quantile min_conf max_conf min_pixels
shrinkage margin_min base_subpath skip_step0 batch_size step0_batch_size buffer
gamma random_seed model air_feature_source

Phase B grids may append these optional columns:
threshold_artifact threshold_max_batches

Environment:
  PYTHON, DATA_ROOT, CUDA_VISIBLE_DEVICES, CUDA_MPS_PIPE_DIRECTORY, TMPDIR
  SKIP_EXISTING=1 skips completed rows with test_results_*.json.
  ALLOW_INCOMPLETE=1 allows reusing an existing incomplete output dir.
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
SKIP_EXISTING="${SKIP_EXISTING:-1}"
ALLOW_INCOMPLETE="${ALLOW_INCOMPLETE:-0}"

mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$TMPDIR"

print_command() {
    printf '%q ' \
        PYTHON="$PYTHON" \
        DATA_ROOT="$DATA_ROOT" \
        MODEL="$model" \
        AIR_FEATURE_SOURCE="$air_feature_source" \
        TASK="$task" \
        SETTING="$setting" \
        SUBPATH="$subpath" \
        BASE_SUBPATH="$base_subpath" \
        SKIP_STEP0="$skip_step0" \
        PSEUDO_LABEL_STRATEGY="$strategy" \
        PSEUDO_LABEL_CONFIDENCE="$confidence" \
        PSEUDO_LABEL_QUANTILE="$quantile" \
        PSEUDO_LABEL_MIN_CONF="$min_conf" \
        PSEUDO_LABEL_MAX_CONF="$max_conf" \
        PSEUDO_LABEL_MIN_PIXELS="$min_pixels" \
        PSEUDO_LABEL_SHRINKAGE="$shrinkage" \
        PSEUDO_LABEL_MARGIN_MIN="$margin_min" \
        PSEUDO_LABEL_THRESHOLD_ARTIFACT="$threshold_artifact" \
        PSEUDO_LABEL_THRESHOLD_MAX_BATCHES="$threshold_max_batches" \
        BATCH_SIZE="$batch_size" \
        STEP0_BATCH_SIZE="$step0_batch_size" \
        BUFFER="$buffer" \
        GAMMA="$gamma" \
        RANDOM_SEED="$random_seed" \
        bash tools/run_adaptive_pseudo_label.sh
    printf '\n'
}

run_row() {
    local output_dir="checkpoints/${subpath}/voc/${task}/${setting}/step1"
    if compgen -G "${output_dir}/test_results_*.json" >/dev/null; then
        if [[ "$SKIP_EXISTING" == "1" ]]; then
            echo "[grid] skip completed row '${name}': ${output_dir}"
            return
        fi
    elif [[ -e "$output_dir" && "$ALLOW_INCOMPLETE" != "1" ]]; then
        echo "[grid] refusing to reuse incomplete output dir for row '${name}': ${output_dir}" >&2
        echo "[grid] set ALLOW_INCOMPLETE=1 only if you intentionally want to continue/reuse it" >&2
        exit 3
    fi

    echo "[grid] row=${name} subpath=${subpath} strategy=${strategy} confidence=${confidence} quantile=${quantile}"
    if [[ -n "${threshold_artifact:-}" ]]; then
        echo "[grid] threshold_artifact=${threshold_artifact} threshold_max_batches=${threshold_max_batches:-<none>}"
    fi
    if [[ "$MODE" == "dry-run" ]]; then
        print_command
        return
    fi

    PYTHON="$PYTHON" \
    DATA_ROOT="$DATA_ROOT" \
    MODEL="$model" \
    AIR_FEATURE_SOURCE="$air_feature_source" \
    TASK="$task" \
    SETTING="$setting" \
    SUBPATH="$subpath" \
    BASE_SUBPATH="$base_subpath" \
    SKIP_STEP0="$skip_step0" \
    PSEUDO_LABEL_STRATEGY="$strategy" \
    PSEUDO_LABEL_CONFIDENCE="$confidence" \
    PSEUDO_LABEL_QUANTILE="$quantile" \
    PSEUDO_LABEL_MIN_CONF="$min_conf" \
    PSEUDO_LABEL_MAX_CONF="$max_conf" \
    PSEUDO_LABEL_MIN_PIXELS="$min_pixels" \
    PSEUDO_LABEL_SHRINKAGE="$shrinkage" \
    PSEUDO_LABEL_MARGIN_MIN="$margin_min" \
    PSEUDO_LABEL_THRESHOLD_ARTIFACT="$threshold_artifact" \
    PSEUDO_LABEL_THRESHOLD_MAX_BATCHES="$threshold_max_batches" \
    BATCH_SIZE="$batch_size" \
    STEP0_BATCH_SIZE="$step0_batch_size" \
    BUFFER="$buffer" \
    GAMMA="$gamma" \
    RANDOM_SEED="$random_seed" \
    bash tools/run_adaptive_pseudo_label.sh
}

declare -A seen_subpaths=()
line_no=0
while IFS=$'\t' read -r \
    name subpath task setting strategy confidence quantile min_conf max_conf min_pixels \
    shrinkage margin_min base_subpath skip_step0 batch_size step0_batch_size buffer \
    gamma random_seed model air_feature_source threshold_artifact threshold_max_batches extra \
    || [[ -n "${name:-}${subpath:-}${task:-}${setting:-}${strategy:-}${confidence:-}${quantile:-}${min_conf:-}${max_conf:-}${min_pixels:-}${shrinkage:-}${margin_min:-}${base_subpath:-}${skip_step0:-}${batch_size:-}${step0_batch_size:-}${buffer:-}${gamma:-}${random_seed:-}${model:-}${air_feature_source:-}${threshold_artifact:-}${threshold_max_batches:-}${extra:-}" ]]
do
    line_no=$((line_no + 1))
    air_feature_source="${air_feature_source%$'\r'}"
    threshold_artifact="${threshold_artifact%$'\r'}"
    threshold_max_batches="${threshold_max_batches%$'\r'}"
    extra="${extra%$'\r'}"
    if [[ "$line_no" -eq 1 ]]; then
        if [[ -n "${extra:-}" \
            || "$name" != "name" \
            || "$subpath" != "subpath" \
            || "$task" != "task" \
            || "$setting" != "setting" \
            || "$strategy" != "strategy" \
            || "$confidence" != "confidence" \
            || "$quantile" != "quantile" \
            || "$min_conf" != "min_conf" \
            || "$max_conf" != "max_conf" \
            || "$min_pixels" != "min_pixels" \
            || "$shrinkage" != "shrinkage" \
            || "$margin_min" != "margin_min" \
            || "$base_subpath" != "base_subpath" \
            || "$skip_step0" != "skip_step0" \
            || "$batch_size" != "batch_size" \
            || "$step0_batch_size" != "step0_batch_size" \
            || "$buffer" != "buffer" \
            || "$gamma" != "gamma" \
            || "$random_seed" != "random_seed" \
            || "$model" != "model" \
            || "$air_feature_source" != "air_feature_source" ]]; then
            echo "grid header is not the expected pseudo-label grid header" >&2
            exit 2
        fi
        if [[ -n "${threshold_artifact:-}${threshold_max_batches:-}" ]]; then
            if [[ "$threshold_artifact" != "threshold_artifact" \
                || "$threshold_max_batches" != "threshold_max_batches" ]]; then
                echo "grid optional header must be: threshold_artifact threshold_max_batches" >&2
                exit 2
            fi
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
    for required in name subpath task setting strategy confidence quantile min_conf max_conf min_pixels shrinkage margin_min base_subpath skip_step0 batch_size step0_batch_size buffer gamma random_seed model air_feature_source; do
        if [[ -z "${!required:-}" ]]; then
            echo "grid line ${line_no} missing required field: ${required}" >&2
            exit 2
        fi
    done
    if [[ "$strategy" == "artifact_class" && -z "${threshold_artifact:-}" ]]; then
        echo "grid line ${line_no} strategy=artifact_class requires threshold_artifact" >&2
        exit 2
    fi
    if [[ -n "${seen_subpaths[$subpath]:-}" ]]; then
        echo "grid line ${line_no} reuses subpath '${subpath}' from line ${seen_subpaths[$subpath]}" >&2
        exit 2
    fi
    seen_subpaths["$subpath"]="$line_no"
    run_row
done < "$GRID_PATH"
