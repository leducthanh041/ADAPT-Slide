#!/bin/bash
#
# Run the v2 task-routing bugfix diagnosis for Class-IL TCP:
#   Step 1: sweep n_steps with the fixed default
#   Step 2: compare fixed/default vs agreement-off vs v1 bug vs selection-only
#
# Defaults mirror the command in the experiment note. Override with env vars:
#   SETTING=ind ORDER=forward CONFIG_FORWARD=configs/default_eval_num_workers0.yaml
#   N_STEPS_SWEEP="1 2 3 5 8" BEST_N_STEPS=3
#   RUN_SWEEP=1 RUN_ABLATION=1 bash scripts/test_classIL_tta_v2_bugfix_ablation.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mmlab_students/storageStudents/nguyenvd/Thanhld/WSI/MergeSlide_TTA_v1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SETTING="${SETTING:-ind}"
ORDER="${ORDER:-forward}"
MODE="${MODE:-tcp}"
RUN_SWEEP="${RUN_SWEEP:-1}"
RUN_ABLATION="${RUN_ABLATION:-1}"
N_STEPS_SWEEP="${N_STEPS_SWEEP:-1 2 3 5 8}"
BEST_N_STEPS="${BEST_N_STEPS:-3}"
BASE_LOG_DIR="${BASE_LOG_DIR:-logs/fixed_v2}"

TTA_M="${TTA_M:-8}"
TTA_K_SUB="${TTA_K_SUB:-300}"
TTA_TOP_RATIO="${TTA_TOP_RATIO:-0.5}"
TTA_ALPHA="${TTA_ALPHA:-0.5}"
TTA_BETA="${TTA_BETA:-1.0}"
TTA_LR="${TTA_LR:-1e-4}"
TTA_PARAM_SCOPE="${TTA_PARAM_SCOPE:-ln_only}"
TTA_ENTROPY_THRESHOLD="${TTA_ENTROPY_THRESHOLD:-0.4}"
TTA_GAMMA="${TTA_GAMMA:-0.5}"
TTA_VERBOSE_LOSS="${TTA_VERBOSE_LOSS:-1}"

cd "$PROJECT_ROOT"
mkdir -p "$BASE_LOG_DIR"

run_one() {
    local tag="$1"
    local n_steps="$2"
    local select_mode="$3"
    local use_task_diversity="$4"
    local no_task_agreement="$5"

    local log_dir="$BASE_LOG_DIR/$tag"
    local result_csv="$BASE_LOG_DIR/${tag}.csv"

    echo ""
    echo "==================================================================="
    echo "[RUN] tag=$tag n_steps=$n_steps select_mode=$select_mode use_task_diversity=$use_task_diversity no_task_agreement=$no_task_agreement"
    echo "==================================================================="

    SETTING="$SETTING" \
    ORDER="$ORDER" \
    MODE="$MODE" \
    LOG_DIR="$log_dir" \
    TTA_RESULT_CSV="$result_csv" \
    TTA_M="$TTA_M" \
    TTA_K_SUB="$TTA_K_SUB" \
    TTA_TOP_RATIO="$TTA_TOP_RATIO" \
    TTA_ALPHA="$TTA_ALPHA" \
    TTA_BETA="$TTA_BETA" \
    TTA_LR="$TTA_LR" \
    TTA_N_STEPS="$n_steps" \
    TTA_PARAM_SCOPE="$TTA_PARAM_SCOPE" \
    TTA_ENTROPY_THRESHOLD="$TTA_ENTROPY_THRESHOLD" \
    TTA_GAMMA="$TTA_GAMMA" \
    TTA_SELECT_MODE="$select_mode" \
    TTA_USE_TASK_DIVERSITY="$use_task_diversity" \
    TTA_NO_TASK_AGREEMENT="$no_task_agreement" \
    TTA_VERBOSE_LOSS="$TTA_VERBOSE_LOSS" \
        bash "$SCRIPT_DIR/test_classIL_tta.sh"
}

if [ "$RUN_SWEEP" = "1" ]; then
    for n in $N_STEPS_SWEEP; do
        run_one "nsteps_${n}" "$n" "intersection" "0" "0"
    done
fi

if [ "$RUN_ABLATION" = "1" ]; then
    # (a) Baseline fixed/proposed: no task diversity, task agreement on.
    run_one "ablation_a_fixed_intersection_agreement_n${BEST_N_STEPS}" \
        "$BEST_N_STEPS" "intersection" "0" "0"

    # (b) Disable agreement: tests whether removing task diversity alone is enough.
    run_one "ablation_b_no_agreement_n${BEST_N_STEPS}" \
        "$BEST_N_STEPS" "intersection" "0" "1"

    # (c) Reproduce v1 bug: union selection + SHOT-style task diversity.
    run_one "ablation_c_reproduce_v1_bug_union_taskdiv_n${BEST_N_STEPS}" \
        "$BEST_N_STEPS" "union" "1" "0"

    # (d) Selection-only isolation: intersection but keep old task-diversity bug.
    run_one "ablation_d_intersection_with_taskdiv_n${BEST_N_STEPS}" \
        "$BEST_N_STEPS" "intersection" "1" "0"
fi

echo ""
echo "[INFO] Done. Logs and CSV files are under: $BASE_LOG_DIR"
