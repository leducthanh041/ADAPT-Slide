#!/bin/bash
#
# CLASS-IL TTA evaluation runner.
# Keeps hot writes under /docker via repo-local logs/checkpoints symlinks and
# uses *_num_workers0 configs to avoid DataLoader multiprocessing.

#SBATCH --job-name=test_classIL_tta
#SBATCH --output=logs/test_classIL_tta_%j.out
#SBATCH --error=logs/test_classIL_tta_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=72:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mmlab_students/storageStudents/nguyenvd/Thanhld/WSI/MergeSlide_TTA_v1}"
USER_NAME="${USER:-thanhld}"
PROJECT_NAME="$(basename "$PROJECT_ROOT")"
export MERGESLIDE_LOCAL_ROOT="${MERGESLIDE_LOCAL_ROOT:-/docker/data/$USER_NAME/$PROJECT_NAME}"
SETTING="${SETTING:-ind}"
ORDER="${ORDER:-forward}"
MODE="${MODE:-${TTA_MODE:-tcp}}"
LOG_DIR="${LOG_DIR:-}"
if [ -n "$LOG_DIR" ] && [[ "$LOG_DIR" != /* && "$LOG_DIR" != logs && "$LOG_DIR" != logs/* ]]; then
    LOG_DIR="logs/$LOG_DIR"
fi
if [ -z "$LOG_DIR" ]; then
    LOG_DIR="logs/classil_tta/${SETTING}_${ORDER}_${MODE}"
fi
case "$MODE" in
    tcp|naive|all) ;;
    *) echo "[ERROR] Unsupported MODE=$MODE (expected tcp|naive|all)" >&2; exit 1 ;;
esac
case "${SETTING}_${ORDER}" in
    ood_forward)
        CONFIG_FORWARD="${CONFIG_FORWARD:-configs/default_ood_eval_num_workers0.yaml}"
        SAVE_DIR_FORWARD="${SAVE_DIR_FORWARD:-./checkpoints_ood/finetuned}"
        MERGE_MODEL_PATH_FORWARD="${MERGE_MODEL_PATH_FORWARD:-./checkpoints_ood/merged}"
        ;;
    ind_forward)
        CONFIG_FORWARD="${CONFIG_FORWARD:-configs/default_eval_num_workers0.yaml}"
        SAVE_DIR_FORWARD="${SAVE_DIR_FORWARD:-./checkpoints/finetuned}"
        MERGE_MODEL_PATH_FORWARD="${MERGE_MODEL_PATH_FORWARD:-./checkpoints/merged}"
        ;;
    ind_reverse)
        CONFIG_FORWARD="${CONFIG_FORWARD:-configs/default_reverse_eval_num_workers0.yaml}"
        SAVE_DIR_FORWARD="${SAVE_DIR_FORWARD:-./checkpoints/finetuned_reverse}"
        MERGE_MODEL_PATH_FORWARD="${MERGE_MODEL_PATH_FORWARD:-./checkpoints/merged_reverse}"
        ;;
    ood_reverse)
        echo "[ERROR] OOD reverse is not configured. Use SETTING=ind ORDER=reverse." >&2
        exit 1
        ;;
    *)
        echo "[ERROR] Unsupported SETTING/ORDER: SETTING=$SETTING ORDER=$ORDER (expected ind|ood with forward, or ind with reverse)" >&2
        exit 1
        ;;
esac

CLASSIL_ENTRYPOINT="${CLASSIL_ENTRYPOINT:-tools/run_classil_with_pt_features.py}"
TTA_ENTRYPOINT="${TTA_ENTRYPOINT:-test_classIL_tta.py}"
TTA_VARIANTS="${TTA_VARIANTS:-$MODE}"

# ---------------------------------------------------------------------------
TTA_M="${TTA_M:-8}"                         # sub-bags/slide
TTA_K_SUB="${TTA_K_SUB:-300}"               # patches/sub-bag
TTA_TOP_RATIO="${TTA_TOP_RATIO:-0.5}"       # confident sub-bag ratio
TTA_ALPHA="${TTA_ALPHA:-0.5}"               # task loss weight
TTA_BETA="${TTA_BETA:-1.0}"                 # L2 anchor weight
TTA_LR="${TTA_LR:-1e-4}"                    # LN optimizer lr
TTA_N_STEPS="${TTA_N_STEPS:-1}"             # adapt steps/slide
TTA_PARAM_SCOPE="${TTA_PARAM_SCOPE:-ln_only}"  # ln_only | full
TTA_ENTROPY_THRESHOLD="${TTA_ENTROPY_THRESHOLD:-0.4}"  # WSI-level filter
TTA_GAMMA="${TTA_GAMMA:-0.5}"                  # JSD task-agreement weight
TTA_SELECT_MODE="${TTA_SELECT_MODE:-intersection}"  # union | intersection
TTA_USE_TASK_DIVERSITY="${TTA_USE_TASK_DIVERSITY:-0}" # 1 reproduces old bug
TTA_NO_TASK_AGREEMENT="${TTA_NO_TASK_AGREEMENT:-0}"   # 1 disables JSD agreement

TTA_EPISODIC="${TTA_EPISODIC:-0}"
TTA_VERBOSE_LOSS="${TTA_VERBOSE_LOSS:-1}"
TTA_DIAG_DIR="${TTA_DIAG_DIR:-}"
TTA_RESULT_CSV="${TTA_RESULT_CSV:-$LOG_DIR/tta_v1_tcp_routing_results.csv}"

# ---------------------------------------------------------------------------
# Python binary  ging ht test_classIL.sh
# ---------------------------------------------------------------------------
if [ -z "${PYTHON_BIN:-}" ]; then
    DEFAULT_PYTHON="/mmlab_students/storageStudents/nguyenvd/anaconda3/envs/mergePre/bin/python3.10"
    if [ -x "$DEFAULT_PYTHON" ]; then
        PYTHON_BIN="$DEFAULT_PYTHON"
    else
        PYTHON_BIN="python"
    fi
fi

cd "$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Directory + symlink setup  ging ht test_classIL.sh
# ---------------------------------------------------------------------------
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

mkdir -p "$LOG_DIR"
export TMPDIR="${TMPDIR:-$MERGESLIDE_LOCAL_ROOT/tmp}"
export SQLITE_TMPDIR="${SQLITE_TMPDIR:-$MERGESLIDE_LOCAL_ROOT/sqlite}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

# ---------------------------------------------------------------------------
# Logging info
# ---------------------------------------------------------------------------
EPISODIC_LABEL="continual"
EPISODIC_FLAG=""
if [ "${TTA_EPISODIC}" = "1" ]; then
    EPISODIC_LABEL="episodic"
    EPISODIC_FLAG="--episodic"
fi

echo "[INFO] start at $(date)"
echo "[INFO] project_root=$PROJECT_ROOT"
echo "[INFO] python=$PYTHON_BIN"
echo "[INFO] local_hot_root=$MERGESLIDE_LOCAL_ROOT"
echo "[INFO] tta_entrypoint=$TTA_ENTRYPOINT"
echo "[INFO] setting=$SETTING"
echo "[INFO] order=$ORDER"
echo "[INFO] mode=$MODE"
echo "[INFO] log_dir=$LOG_DIR"
echo "[INFO] config=$CONFIG_FORWARD"
echo "[INFO] save_dir=$SAVE_DIR_FORWARD"
echo "[INFO] merge_model_path=$MERGE_MODEL_PATH_FORWARD"
echo "[INFO] tta_result_csv=$TTA_RESULT_CSV"
echo "[INFO] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[INFO] tta_variants=$TTA_VARIANTS"
echo "[INFO] TTA M=$TTA_M | K_sub=$TTA_K_SUB | top_ratio=$TTA_TOP_RATIO | alpha=$TTA_ALPHA | beta=$TTA_BETA | lr=$TTA_LR | n_steps=$TTA_N_STEPS | param_scope=$TTA_PARAM_SCOPE | entropy_threshold=$TTA_ENTROPY_THRESHOLD | reset=$EPISODIC_LABEL | verbose_loss=$TTA_VERBOSE_LOSS"
echo "[INFO] bugfix_ablation gamma=$TTA_GAMMA | select_mode=$TTA_SELECT_MODE | use_task_diversity=$TTA_USE_TASK_DIVERSITY | no_task_agreement=$TTA_NO_TASK_AGREEMENT"

entrypoint_path="$TTA_ENTRYPOINT"
if [[ "$entrypoint_path" != /* ]]; then
    entrypoint_path="$PROJECT_ROOT/$entrypoint_path"
fi

supports_arg() {
    local arg_name="$1"
    grep -q -- "$arg_name" "$entrypoint_path"
}

# ---------------------------------------------------------------------------
# check_log_not_held  ging ht test_classIL.sh
# ---------------------------------------------------------------------------
check_log_not_held() {
    local log_path="$1"
    local resolved_log
    resolved_log="$(readlink -f "$log_path" 2>/dev/null || true)"
    if [ -z "$resolved_log" ]; then return 0; fi
    local fd target pid state cmdline
    for fd in /proc/[0-9]*/fd/1 /proc/[0-9]*/fd/2; do
        [ -e "$fd" ] || continue
        target="$(readlink -f "$fd" 2>/dev/null || true)"
        [ "$target" = "$resolved_log" ] || continue
        pid="${fd#/proc/}"; pid="${pid%%/*}"
        [ "$pid" = "$$" ] && continue
        state="$(awk '/^State:/ {print $2}' "/proc/$pid/status" 2>/dev/null || true)"
        cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
        case "$cmdline" in torch_shm_manager*) continue ;; esac
        echo "[ERROR] $log_path is already held by PID $pid state=$state cmd=$cmdline" >&2
        echo "[ERROR] Refusing to reuse this log." >&2
        return 1
    done
}

# ---------------------------------------------------------------------------
# run_to_logs  ging ht test_classIL.sh
# ---------------------------------------------------------------------------
run_to_logs() {
    local result_log="$1"
    local error_log="$2"
    shift 2
    echo "[INFO] running: $*"
    echo "[INFO] result_log=$result_log"
    echo "[INFO] error_log=$error_log"
    check_log_not_held "$result_log"
    check_log_not_held "$error_log"
    { echo "[INFO] start at $(date)"; echo "[INFO] command=$*"; } > "$result_log"
    { echo "[INFO] start at $(date)"; echo "[INFO] command=$*"; } > "$error_log"
    "$@" >> "$result_log" 2>> "$error_log"
}

variant_enabled() {
    local variant="$1"
    if [ "$TTA_VARIANTS" = "all" ]; then
        return 0
    fi
    case ",$TTA_VARIANTS," in
        *,"$variant",*) return 0 ;;
        *) return 1 ;;
    esac
}

# ---------------------------------------------------------------------------
# TTA base args  ti s dng cho tt c 4 variants
# ---------------------------------------------------------------------------
TTA_ARGS=(
    --entrypoint        "$TTA_ENTRYPOINT"
    --M                 "$TTA_M"
    --K_sub             "$TTA_K_SUB"
    --top_ratio         "$TTA_TOP_RATIO"
    --alpha             "$TTA_ALPHA"
    --beta              "$TTA_BETA"
    --lr                "$TTA_LR"
    --n_steps           "$TTA_N_STEPS"
    --tta_param_scope   "$TTA_PARAM_SCOPE"
    --entropy_threshold "$TTA_ENTROPY_THRESHOLD"
    --gamma             "$TTA_GAMMA"
    --select_mode       "$TTA_SELECT_MODE"
)
if [ -n "$EPISODIC_FLAG" ]; then
    TTA_ARGS+=("$EPISODIC_FLAG")
fi
if [ "$TTA_VERBOSE_LOSS" = "1" ]; then
    TTA_ARGS+=(--verbose_loss)
fi
if [ "$TTA_USE_TASK_DIVERSITY" = "1" ]; then
    TTA_ARGS+=(--use_task_diversity)
fi
if [ "$TTA_NO_TASK_AGREEMENT" = "1" ]; then
    TTA_ARGS+=(--no_task_agreement)
fi
if [ -n "$TTA_DIAG_DIR" ]; then
    if supports_arg "--diag_dir"; then
        TTA_ARGS+=(--diag_dir "$TTA_DIAG_DIR")
    else
        echo "[WARN] $TTA_ENTRYPOINT does not support --diag_dir; skipping TTA_DIAG_DIR=$TTA_DIAG_DIR" >&2
    fi
fi
if [ -n "$TTA_RESULT_CSV" ]; then
    if supports_arg "--result_csv"; then
        TTA_ARGS+=(--result_csv "$TTA_RESULT_CSV")
    else
        echo "[WARN] $TTA_ENTRYPOINT does not support --result_csv; routing CSV will not be saved" >&2
    fi
fi

# ---------------------------------------------------------------------------
# Run selected Class-IL TTA mode on selected protocol.
# Protocols:
#   SETTING=ind ORDER=forward  -> B->R->N->E->T->C
#   SETTING=ood ORDER=forward  -> cross-site/OOD forward
#   SETTING=ind ORDER=reverse  -> C->T->E->N->R->B
# Modes:
#   MODE=tcp | MODE=naive | MODE=all
# ---------------------------------------------------------------------------

if variant_enabled tcp; then
    run_to_logs \
        "$LOG_DIR/result_tta_tcp.log" \
        "$LOG_DIR/error_tta_tcp.log" \
        "$PYTHON_BIN" -u "$CLASSIL_ENTRYPOINT" \
            --config           "$CONFIG_FORWARD" \
            --save_dir         "$SAVE_DIR_FORWARD" \
            --merge_model_path "$MERGE_MODEL_PATH_FORWARD" \
            --mode tcp \
            "${TTA_ARGS[@]}"
fi

if variant_enabled naive; then
   run_to_logs \
       "$LOG_DIR/result_tta_naive.log" \
       "$LOG_DIR/error_tta_naive.log" \
       "$PYTHON_BIN" -u "$CLASSIL_ENTRYPOINT" \
           --config           "$CONFIG_FORWARD" \
           --save_dir         "$SAVE_DIR_FORWARD" \
           --merge_model_path "$MERGE_MODEL_PATH_FORWARD" \
           --mode naive \
           "${TTA_ARGS[@]}"
fi

echo "[INFO] finished at $(date)"
