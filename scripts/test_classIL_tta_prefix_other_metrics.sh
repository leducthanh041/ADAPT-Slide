#!/bin/bash
#
# CLASS-IL prefix-merge TTA mACC/FGT/BWT runner.
#
# This matches test_classIL_task_prompt_other_metrics.py's prefix protocol:
#   task_0.pt -> merged_task_1.pth -> ... -> merged_final.pth
# but runs online MergeSlide-TTA during inference at every prefix.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mmlab_students/storageStudents/nguyenvd/Thanhld/WSI/MergeSlide_TTA_v1}"
USER_NAME="${USER:-thanhld}"
PROJECT_NAME="$(basename "$PROJECT_ROOT")"
export MERGESLIDE_LOCAL_ROOT="${MERGESLIDE_LOCAL_ROOT:-/docker/data/$USER_NAME/$PROJECT_NAME}"

SETTING="${SETTING:-ind}"
ORDER="${ORDER:-forward}"
MODE="${MODE:-tcp}"
USE_BEST_CONFIG="${USE_BEST_CONFIG:-1}"
FOLD_START="${FOLD_START:-0}"
FOLD_END="${FOLD_END:-}"

case "$MODE" in
    tcp|naive|all) ;;
    *) echo "[ERROR] Unsupported MODE=$MODE (expected tcp|naive|all)" >&2; exit 1 ;;
esac

case "${SETTING}_${ORDER}" in
    ood_forward)
        CONFIG="${CONFIG:-configs/default_ood_eval_num_workers0.yaml}"
        SAVE_DIR="${SAVE_DIR:-./checkpoints_ood/finetuned}"
        MERGE_MODEL_PATH="${MERGE_MODEL_PATH:-./checkpoints_ood/merged}"
        SETTING_LABEL="ood"
        ;;
    ind_forward)
        CONFIG="${CONFIG:-configs/default_eval_num_workers0.yaml}"
        SAVE_DIR="${SAVE_DIR:-./checkpoints/finetuned}"
        MERGE_MODEL_PATH="${MERGE_MODEL_PATH:-./checkpoints/merged}"
        SETTING_LABEL="ind"
        ;;
    ind_reverse)
        CONFIG="${CONFIG:-configs/default_reverse_eval_num_workers0.yaml}"
        SAVE_DIR="${SAVE_DIR:-./checkpoints/finetuned_reverse}"
        MERGE_MODEL_PATH="${MERGE_MODEL_PATH:-./checkpoints/merged_reverse}"
        SETTING_LABEL="ind_reverse"
        ;;
    ood_reverse)
        echo "[ERROR] OOD reverse is not configured. Use SETTING=ind ORDER=reverse for reverse." >&2
        exit 1
        ;;
    *)
        echo "[ERROR] Unsupported SETTING/ORDER: SETTING=$SETTING ORDER=$ORDER" >&2
        exit 1
        ;;
esac

if [ -z "${PYTHON_BIN:-}" ]; then
    DEFAULT_PYTHON="/mmlab_students/storageStudents/nguyenvd/anaconda3/envs/mergePre/bin/python3.10"
    if [ -x "$DEFAULT_PYTHON" ]; then
        PYTHON_BIN="$DEFAULT_PYTHON"
    else
        PYTHON_BIN="python"
    fi
fi

cd "$PROJECT_ROOT"

TTA_IND_BEST_CONFIG="${TTA_IND_BEST_CONFIG:-configs/ind/best_config.json}"
TTA_OOD_BEST_CONFIG="${TTA_OOD_BEST_CONFIG:-configs/ood/best_config.json}"

if [ "$USE_BEST_CONFIG" = "1" ]; then
    ACTIVE_BEST_CONFIG=""
    ACTIVE_BEST_LABEL=""
    if [ "$SETTING" = "ind" ]; then
        ACTIVE_BEST_CONFIG="$TTA_IND_BEST_CONFIG"
        ACTIVE_BEST_LABEL="IND"
    elif [ "$SETTING" = "ood" ]; then
        ACTIVE_BEST_CONFIG="$TTA_OOD_BEST_CONFIG"
        ACTIVE_BEST_LABEL="OOD"
    fi

    if [ -n "$ACTIVE_BEST_CONFIG" ]; then
        if [ ! -f "$ACTIVE_BEST_CONFIG" ]; then
            echo "[ERROR] $ACTIVE_BEST_LABEL best TTA config not found: $ACTIVE_BEST_CONFIG" >&2
            echo "[ERROR] Set USE_BEST_CONFIG=0 to run with script defaults/env overrides." >&2
            exit 1
        fi
        eval "$("$PYTHON_BIN" - "$ACTIVE_BEST_CONFIG" <<'PY'
import json
import os
import shlex
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

mapping = {
    "M": "TTA_M",
    "K_sub": "TTA_K_SUB",
    "top_ratio": "TTA_TOP_RATIO",
    "alpha": "TTA_ALPHA",
    "beta": "TTA_BETA",
    "lr": "TTA_LR",
    "n_steps": "TTA_N_STEPS",
    "tta_param_scope": "TTA_PARAM_SCOPE",
    "entropy_threshold": "TTA_ENTROPY_THRESHOLD",
    "gamma": "TTA_GAMMA",
    "select_mode": "TTA_SELECT_MODE",
    "ema_alpha": "TTA_EMA_ALPHA",
    "ema_alpha_prompt": "TTA_EMA_ALPHA_PROMPT",
    "delta_margin": "TTA_DELTA_MARGIN",
    "tp_anchor_beta": "TTA_TP_ANCHOR_BETA",
    "gamma_margin": "TTA_GAMMA_MARGIN",
}
for key, env_name in mapping.items():
    if key in cfg and env_name not in os.environ:
        print(f"export {env_name}={shlex.quote(str(cfg[key]))}")
PY
)"
        echo "[INFO] loaded $ACTIVE_BEST_LABEL best TTA config: $ACTIVE_BEST_CONFIG"
    fi
fi

TTA_M="${TTA_M:-8}"
TTA_K_SUB="${TTA_K_SUB:-300}"
TTA_TOP_RATIO="${TTA_TOP_RATIO:-0.5}"
TTA_ALPHA="${TTA_ALPHA:-0.5}"
TTA_BETA="${TTA_BETA:-1.0}"
TTA_LR="${TTA_LR:-1e-4}"
TTA_N_STEPS="${TTA_N_STEPS:-5}"
TTA_PARAM_SCOPE="${TTA_PARAM_SCOPE:-ln_only}"
TTA_ENTROPY_THRESHOLD="${TTA_ENTROPY_THRESHOLD:-0.4}"
TTA_GAMMA="${TTA_GAMMA:-0.5}"
TTA_SELECT_MODE="${TTA_SELECT_MODE:-intersection}"
TTA_USE_TASK_DIVERSITY="${TTA_USE_TASK_DIVERSITY:-0}"
TTA_NO_TASK_AGREEMENT="${TTA_NO_TASK_AGREEMENT:-0}"
TTA_NO_TEACHER="${TTA_NO_TEACHER:-0}"
TTA_EMA_ALPHA="${TTA_EMA_ALPHA:-0.999}"
TTA_NO_ADAPT_PROMPTS="${TTA_NO_ADAPT_PROMPTS:-0}"
TTA_EMA_ALPHA_PROMPT="${TTA_EMA_ALPHA_PROMPT:-0.999}"
TTA_DELTA_MARGIN="${TTA_DELTA_MARGIN:-0.10}"
TTA_TP_ANCHOR_BETA="${TTA_TP_ANCHOR_BETA:-0.3}"
TTA_GAMMA_MARGIN="${TTA_GAMMA_MARGIN:-0.0}"
TTA_NO_RESET_PROMPT_PER_TASK="${TTA_NO_RESET_PROMPT_PER_TASK:-0}"
TTA_VERBOSE_LOSS="${TTA_VERBOSE_LOSS:-0}"

CLASSIL_WRAPPER="${CLASSIL_WRAPPER:-tools/run_classil_with_pt_features.py}"
ENTRYPOINT="${ENTRYPOINT:-test_classIL_tta_prefix_other_metrics.py}"

LOG_ROOT="${LOG_ROOT:-${LOG_DIR:-logs/prefix_tta_metrics/$SETTING_LABEL}}"
if [[ "$LOG_ROOT" != /* && "$LOG_ROOT" != logs && "$LOG_ROOT" != logs/* ]]; then
    LOG_ROOT="logs/$LOG_ROOT"
fi

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

if [ "${TTA_EPISODIC:-0}" = "1" ]; then
    echo "[WARN] TTA_EPISODIC=1 is ignored. This runner uses continual adaptation." >&2
fi

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
        return 1
    done
}

run_one_mode() {
    local run_mode="$1"
    local mode_log_dir="$LOG_ROOT/$run_mode"
    local output_dir="${OUTPUT_DIR:-$mode_log_dir/outputs}"
    local result_log="$mode_log_dir/result.log"
    local error_log="$mode_log_dir/error.log"

    mkdir -p "$mode_log_dir" "$output_dir"
    check_log_not_held "$result_log"
    check_log_not_held "$error_log"

    local args=(
        "$PYTHON_BIN" -u "$CLASSIL_WRAPPER"
        --entrypoint "$ENTRYPOINT"
        --config "$CONFIG"
        --save_dir "$SAVE_DIR"
        --merge_model_path "$MERGE_MODEL_PATH"
        --output_dir "$output_dir"
        --mode "$run_mode"
        --fold_start "$FOLD_START"
        --M "$TTA_M"
        --K_sub "$TTA_K_SUB"
        --top_ratio "$TTA_TOP_RATIO"
        --alpha "$TTA_ALPHA"
        --beta "$TTA_BETA"
        --lr "$TTA_LR"
        --n_steps "$TTA_N_STEPS"
        --tta_param_scope "$TTA_PARAM_SCOPE"
        --entropy_threshold "$TTA_ENTROPY_THRESHOLD"
        --gamma "$TTA_GAMMA"
        --select_mode "$TTA_SELECT_MODE"
        --ema_alpha "$TTA_EMA_ALPHA"
        --ema_alpha_prompt "$TTA_EMA_ALPHA_PROMPT"
        --delta_margin "$TTA_DELTA_MARGIN"
        --tp_anchor_beta "$TTA_TP_ANCHOR_BETA"
        --gamma_margin "$TTA_GAMMA_MARGIN"
    )
    if [ -n "$FOLD_END" ]; then
        args+=(--fold_end "$FOLD_END")
    fi
    if [ "$TTA_USE_TASK_DIVERSITY" = "1" ]; then
        args+=(--use_task_diversity)
    fi
    if [ "$TTA_NO_TASK_AGREEMENT" = "1" ]; then
        args+=(--no_task_agreement)
    fi
    if [ "$TTA_NO_TEACHER" = "1" ] || [ "$run_mode" = "naive" ]; then
        args+=(--no_teacher)
    fi
    if [ "$TTA_NO_ADAPT_PROMPTS" = "1" ]; then
        args+=(--no_adapt_prompts)
    fi
    if [ "$TTA_NO_RESET_PROMPT_PER_TASK" = "1" ]; then
        args+=(--no_reset_prompt_per_task)
    fi
    if [ "$TTA_VERBOSE_LOSS" = "1" ]; then
        args+=(--verbose_loss)
    fi

    {
        echo "[INFO] start at $(date)"
        echo "[INFO] project_root=$PROJECT_ROOT"
        echo "[INFO] setting=$SETTING order=$ORDER setting_label=$SETTING_LABEL mode=$run_mode"
        echo "[INFO] config=$CONFIG"
        echo "[INFO] save_dir=$SAVE_DIR"
        echo "[INFO] merge_model_path=$MERGE_MODEL_PATH"
        echo "[INFO] output_dir=$output_dir"
        echo "[INFO] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
        echo "[INFO] TTA M=$TTA_M K_sub=$TTA_K_SUB top_ratio=$TTA_TOP_RATIO alpha=$TTA_ALPHA beta=$TTA_BETA lr=$TTA_LR n_steps=$TTA_N_STEPS param_scope=$TTA_PARAM_SCOPE entropy_threshold=$TTA_ENTROPY_THRESHOLD"
        echo "[INFO] command=${args[*]}"
        "${args[@]}"
        echo "[INFO] finished at $(date)"
    } > >(tee "$result_log") 2> >(tee "$error_log" >&2)
}

echo "[INFO] start at $(date)"
echo "[INFO] setting=$SETTING order=$ORDER mode=$MODE use_best_config=$USE_BEST_CONFIG"
echo "[INFO] log_root=$LOG_ROOT"

if [ "$MODE" = "all" ]; then
    run_one_mode tcp
    run_one_mode naive
else
    run_one_mode "$MODE"
fi

echo "[INFO] all done at $(date)"
