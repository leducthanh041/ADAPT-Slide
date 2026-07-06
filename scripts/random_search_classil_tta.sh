#!/bin/bash
#
# Two-phase random search for MergeSlide-TTA v1 Class-IL TCP.
#
# Phase 1: tune TCP task-routing hyperparameters, select by routing_acc.
# Phase 2: fix the best routing config, tune class-level hyperparameters,
#          select by bACC.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mmlab_students/storageStudents/nguyenvd/Thanhld/WSI/MergeSlide_TTA_v1}"
USER_NAME="${USER:-thanhld}"
PROJECT_NAME="$(basename "$PROJECT_ROOT")"
export MERGESLIDE_LOCAL_ROOT="${MERGESLIDE_LOCAL_ROOT:-/docker/data/$USER_NAME/$PROJECT_NAME}"
LOG_DIR="${LOG_DIR:-logs/random_search_classil_tta_runs}"

if [ -z "${PYTHON_BIN:-}" ]; then
    DEFAULT_PYTHON="/mmlab_students/storageStudents/nguyenvd/anaconda3/envs/mergePre/bin/python3.10"
    if [ -x "$DEFAULT_PYTHON" ]; then
        PYTHON_BIN="$DEFAULT_PYTHON"
    else
        PYTHON_BIN="python"
    fi
fi

SETTING="${SETTING:-ind}"
SEED="${SEED:-42}"
N_TRIALS_ROUTING="${N_TRIALS_ROUTING:-30}"
N_TRIALS_CLASS="${N_TRIALS_CLASS:-30}"
TOP_K="${TOP_K:-10}"
ENTRYPOINT_WRAPPER="${ENTRYPOINT_WRAPPER:-tools/run_classil_with_pt_features.py}"
TUNE_ENTRYPOINT="${TUNE_ENTRYPOINT:-tools/random_search_classil_tta.py}"
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-0}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
SINGLE_GPU="${SINGLE_GPU:-1}"
TIMEOUT_SEC="${TIMEOUT_SEC:-0}"
RESET_MANIFEST="${RESET_MANIFEST:-0}"
QUIET_LOSS="${QUIET_LOSS:-0}"

cd "$PROJECT_ROOT"

mkdir -p "$MERGESLIDE_LOCAL_ROOT/logs" \
         "$MERGESLIDE_LOCAL_ROOT/checkpoints" \
         "$MERGESLIDE_LOCAL_ROOT/checkpoints_ood" \
         "$MERGESLIDE_LOCAL_ROOT/sqlite" \
         "$MERGESLIDE_LOCAL_ROOT/tmp"

for name in logs checkpoints checkpoints_ood; do
    repo_path="$PROJECT_ROOT/$name"
    local_path="$MERGESLIDE_LOCAL_ROOT/$name"
    if [ -L "$repo_path" ]; then
        :
    elif [ -e "$repo_path" ]; then
        echo "[WARN] $repo_path is not a symlink; hot writes should use $local_path"
    else
        ln -s "$local_path" "$repo_path"
    fi
done

export TMPDIR="${TMPDIR:-$MERGESLIDE_LOCAL_ROOT/tmp}"
export SQLITE_TMPDIR="${SQLITE_TMPDIR:-$MERGESLIDE_LOCAL_ROOT/sqlite}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
mkdir -p "$LOG_DIR"

case "$SETTING" in
    ind)
        BASE_CONFIG="${BASE_CONFIG:-configs/default_eval_num_workers0.yaml}"
        MERGE_DIR="${MERGE_DIR:-./checkpoints/merged}"
        FINETUNED_DIR="${FINETUNED_DIR:-./checkpoints/finetuned}"
        OUTPUT_ROOT="${OUTPUT_ROOT:-logs/random_search_classil_tta/ind}"
        ;;
    ood)
        BASE_CONFIG="${BASE_CONFIG:-configs/default_ood_eval_num_workers0.yaml}"
        MERGE_DIR="${MERGE_DIR:-./checkpoints_ood/merged}"
        FINETUNED_DIR="${FINETUNED_DIR:-./checkpoints_ood/finetuned}"
        OUTPUT_ROOT="${OUTPUT_ROOT:-logs/random_search_classil_tta/ood}"
        ;;
    *)
        echo "[ERROR] SETTING='$SETTING' is invalid. Use SETTING=ind or SETTING=ood." >&2
        exit 1
        ;;
esac

LOSS_FLAG=()
if [ "$QUIET_LOSS" = "1" ]; then
    LOSS_FLAG=(--quiet_loss)
fi

echo "[INFO] start at $(date)"
echo "[INFO] project_root=$PROJECT_ROOT"
echo "[INFO] python=$PYTHON_BIN"
echo "[INFO] setting=$SETTING"
echo "[INFO] base_config=$BASE_CONFIG"
echo "[INFO] merge_dir=$MERGE_DIR"
echo "[INFO] finetuned_dir=$FINETUNED_DIR"
echo "[INFO] output_root=$OUTPUT_ROOT"
echo "[INFO] n_trials_routing=$N_TRIALS_ROUTING n_trials_class=$N_TRIALS_CLASS top_k=$TOP_K"
echo "[INFO] gpu_a=$GPU_A gpu_b=$GPU_B workers_per_gpu=$WORKERS_PER_GPU single_gpu=$SINGLE_GPU"
echo "[INFO] timeout_sec=$TIMEOUT_SEC"

check_log_not_held() {
    local log_path="$1"
    local resolved_log
    resolved_log="$(readlink -f "$log_path" 2>/dev/null || true)"
    [ -z "$resolved_log" ] && return 0
    local fd target pid cmdline
    for fd in /proc/[0-9]*/fd/1 /proc/[0-9]*/fd/2; do
        [ -e "$fd" ] || continue
        target="$(readlink -f "$fd" 2>/dev/null || true)"
        [ "$target" = "$resolved_log" ] || continue
        pid="${fd#/proc/}"
        pid="${pid%%/*}"
        [ "$pid" = "$$" ] && continue
        cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
        case "$cmdline" in torch_shm_manager*) continue ;; esac
        echo "[ERROR] $log_path is already held by PID $pid cmd=$cmdline" >&2
        return 1
    done
}

prepare_manifest() {
    local phase="$1"
    local n_trials="$2"
    local phase_root="$3"
    local manifest_path="$phase_root/manifest_${SETTING}_${phase}_${n_trials}_${SEED}.json"

    mkdir -p "$phase_root"
    if [ "$RESET_MANIFEST" = "1" ] && [ -f "$manifest_path" ]; then
        mv "$manifest_path" "$manifest_path.bak.$(date +%Y%m%d_%H%M%S)"
    fi

    if [ -f "$manifest_path" ]; then
        echo "[INFO][$phase] reuse manifest -> $manifest_path"
    else
        echo "[INFO][$phase] preparing manifest -> $manifest_path"
        "$PYTHON_BIN" -u "$TUNE_ENTRYPOINT" \
            --phase "$phase" \
            --setting "$SETTING" \
            --n_trials "$n_trials" \
            --seed "$SEED" \
            --base_config "$BASE_CONFIG" \
            --merge_dir "$MERGE_DIR" \
            --finetuned_dir "$FINETUNED_DIR" \
            --output_dir "$phase_root" \
            --project_root "$PROJECT_ROOT" \
            --python_bin "$PYTHON_BIN" \
            --entrypoint_wrapper "$ENTRYPOINT_WRAPPER" \
            --manifest_path "$manifest_path" \
            --top_k "$TOP_K" \
            --prepare_manifest
    fi
    PHASE_MANIFEST_PATH="$manifest_path"
}

run_worker() {
    local phase="$1"
    local n_trials="$2"
    local phase_root="$3"
    local manifest_path="$4"
    local fixed_params_json="$5"
    local gpu_id="$6"
    local trial_start="$7"
    local trial_end="$8"
    local worker_tag="$9"
    local worker_output="$phase_root/$worker_tag"
    local worker_log_dir="$LOG_DIR/$SETTING/$phase/$worker_tag"
    local worker_log="$worker_log_dir/random_search.log"
    local worker_err="$worker_log_dir/random_search.err"

    mkdir -p "$worker_output" "$worker_log_dir"
    check_log_not_held "$worker_log"
    check_log_not_held "$worker_err"

    (
        export CUDA_VISIBLE_DEVICES="$gpu_id"
        echo "[INFO] start at $(date)"
        echo "[INFO] phase=$phase setting=$SETTING gpu=$gpu_id trial_start=$trial_start trial_end=$trial_end"
        echo "[INFO] fixed_params_json=$fixed_params_json"
        "$PYTHON_BIN" -u "$TUNE_ENTRYPOINT" \
            --phase "$phase" \
            --setting "$SETTING" \
            --n_trials "$n_trials" \
            --seed "$SEED" \
            --base_config "$BASE_CONFIG" \
            --merge_dir "$MERGE_DIR" \
            --finetuned_dir "$FINETUNED_DIR" \
            --output_dir "$worker_output" \
            --project_root "$PROJECT_ROOT" \
            --python_bin "$PYTHON_BIN" \
            --entrypoint_wrapper "$ENTRYPOINT_WRAPPER" \
            --manifest_path "$manifest_path" \
            --fixed_params_json "$fixed_params_json" \
            --trial_start "$trial_start" \
            --trial_end "$trial_end" \
            --timeout_sec "$TIMEOUT_SEC" \
            --top_k "$TOP_K" \
            "${LOSS_FLAG[@]}"
    ) > "$worker_log" 2> "$worker_err" &
    RUN_WORKER_PID=$!
}

run_phase() {
    local phase="$1"
    local n_trials="$2"
    local phase_root="$3"
    local fixed_params_json="${4:-}"

    echo "[INFO] ===== phase=$phase n_trials=$n_trials fixed_params=$fixed_params_json ====="
    prepare_manifest "$phase" "$n_trials" "$phase_root"
    local manifest_path="$PHASE_MANIFEST_PATH"

    local -a gpu_list
    if [ "$SINGLE_GPU" = "1" ] || { [ "$GPU_A" = "$GPU_B" ] && [ "$WORKERS_PER_GPU" -eq 1 ]; }; then
        gpu_list=("$GPU_A")
        echo "[INFO][$phase] single GPU mode -> GPU=$GPU_A"
    else
        gpu_list=("$GPU_A" "$GPU_B")
    fi

    local num_gpus="${#gpu_list[@]}"
    local total_workers=$(( num_gpus * WORKERS_PER_GPU ))
    if [ "$total_workers" -le 0 ]; then
        echo "[ERROR] total_workers must be > 0" >&2
        exit 1
    fi

    local base_trials=$(( n_trials / total_workers ))
    local remainder=$(( n_trials % total_workers ))
    local -a pids=()
    local worker_idx

    for ((worker_idx=0; worker_idx<total_workers; worker_idx++)); do
        local gpu_idx=$(( worker_idx % num_gpus ))
        local gpu_id="${gpu_list[$gpu_idx]}"
        local extra=0
        if [ "$worker_idx" -lt "$remainder" ]; then
            extra=1
        fi
        local count=$(( base_trials + extra ))
        if [ "$count" -le 0 ]; then
            echo "[INFO][$phase] skip worker_idx=$worker_idx gpu=$gpu_id (no trials)"
            continue
        fi
        local start
        if [ "$worker_idx" -lt "$remainder" ]; then
            start=$(( worker_idx * (base_trials + 1) ))
        else
            start=$(( remainder * (base_trials + 1) + (worker_idx - remainder) * base_trials ))
        fi
        local end=$(( start + count ))
        local worker_tag="gpu${gpu_id}_w${worker_idx}"
        echo "[INFO][$phase] launch worker_idx=$worker_idx gpu=$gpu_id trial_start=$start trial_end=$end tag=$worker_tag"
        run_worker "$phase" "$n_trials" "$phase_root" "$manifest_path" "$fixed_params_json" "$gpu_id" "$start" "$end" "$worker_tag"
        pids+=("$RUN_WORKER_PID")
    done

    local pid
    for pid in "${pids[@]}"; do
        wait "$pid"
    done

    echo "[INFO][$phase] building combined summary..."
    "$PYTHON_BIN" -u "$TUNE_ENTRYPOINT" \
        --phase "$phase" \
        --setting "$SETTING" \
        --n_trials "$n_trials" \
        --seed "$SEED" \
        --base_config "$BASE_CONFIG" \
        --merge_dir "$MERGE_DIR" \
        --finetuned_dir "$FINETUNED_DIR" \
        --output_dir "$phase_root" \
        --project_root "$PROJECT_ROOT" \
        --python_bin "$PYTHON_BIN" \
        --entrypoint_wrapper "$ENTRYPOINT_WRAPPER" \
        --fixed_params_json "$fixed_params_json" \
        --top_k "$TOP_K" \
        --summarize_only
}

PHASE1_ROOT="$OUTPUT_ROOT/phase1_routing"
PHASE2_ROOT="$OUTPUT_ROOT/phase2_class"

run_phase "routing" "$N_TRIALS_ROUTING" "$PHASE1_ROOT" ""

PHASE1_BEST_CONFIG="$PHASE1_ROOT/$SETTING/best_config.json"
if [ ! -f "$PHASE1_BEST_CONFIG" ]; then
    echo "[ERROR] Phase 1 best config not found: $PHASE1_BEST_CONFIG" >&2
    exit 1
fi

run_phase "class" "$N_TRIALS_CLASS" "$PHASE2_ROOT" "$PHASE1_BEST_CONFIG"

echo "[INFO] finished at $(date)"
echo "[INFO] phase1_best_config=$PHASE1_BEST_CONFIG"
echo "[INFO] phase2_best_config=$PHASE2_ROOT/$SETTING/best_config.json"
