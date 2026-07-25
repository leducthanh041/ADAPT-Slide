# test_classIL_tta.py
"""
Class-IL TTA evaluation -- mirrors test_classIL_task_prompt.py.

Modes (same as original):
  tcp   (default): Task-to-Class Prompt-Aligned inference + TTA
  naive          : Direct class inference + TTA

TTA hyperparams:
  --M                  : sub-bags per slide (TTA batch size), default=8
  --K_sub              : patches per sub-bag, default=300
  --top_ratio          : confident sub-bag keep ratio, default=0.5
  --alpha              : task loss weight, default=0.5
  --beta               : L2 anchor weight, default=1.0
  --lr                 : LN optimizer learning rate, default=1e-4
  --n_steps            : adapt steps per slide, default=1
  --episodic           : deprecated/ignored; adaptation is always continual
  --entropy_threshold  : only TTA when entropy >= threshold, default=0.4
                         Set 0.0 to TTA all slides.

Usage:
    python test_classIL_tta.py \\
        --save_dir ./checkpoints/finetuned \\
        --merge_model_path ./checkpoints/merged \\
        --mode tcp
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.metrics import (
    balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from tqdm import tqdm
from transformers import AutoModel

from adapt_slide.constants import K_PATCHES, NUM_TASKS
from adapt_slide.datasets import Sequential_Generic_MIL_Dataset
from adapt_slide.metrics import pad_numpy_arrays
from adapt_slide.prompts_zeroshot import (
    brca_prompts, rcc_prompts, nsclc_prompts,
    esca_prompts, tgct_prompts, cesc_prompts,
)
from adapt_slide.utils import get_eval_metrics, seed_torch
from adapt_slide.tta_adapter import MergeSlide_TTA, load_task_weights

PROJECT_ROOT = Path(__file__).resolve().parent
HOT_DIR_NAMES = {"checkpoints", "logs", "sqlite"}

_PROMPT_FN_MAP = {
    "BRCA":  brca_prompts,
    "RCC":   rcc_prompts,
    "NSCLC": nsclc_prompts,
    "ESCA":  esca_prompts,
    "TGCT":  tgct_prompts,
    "CESC":  cesc_prompts,
}


# ---------------------------------------------------------------------------
# Path helpers -- identical to test_classIL_task_prompt.py
# ---------------------------------------------------------------------------

def get_local_hot_root() -> Path:
    user = os.environ.get("USER") or "thanhld"
    default_root = Path("/docker/data") / user / PROJECT_ROOT.name
    return Path(os.environ.get("MERGESLIDE_LOCAL_ROOT", default_root)).expanduser()


def ensure_local_hot_storage() -> Path:
    local_root = get_local_hot_root()
    local_root.mkdir(parents=True, exist_ok=True)
    for name in HOT_DIR_NAMES:
        (local_root / name).mkdir(parents=True, exist_ok=True)
    (local_root / "tmp").mkdir(parents=True, exist_ok=True)
    for name in ("logs", "checkpoints"):
        repo_path  = PROJECT_ROOT / name
        local_path = local_root / name
        if not repo_path.exists() and not repo_path.is_symlink():
            repo_path.symlink_to(local_path, target_is_directory=True)
    os.environ.setdefault("TMPDIR",                str(local_root / "tmp"))
    os.environ.setdefault("SQLITE_TMPDIR",         str(local_root / "sqlite"))
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    return local_root


def resolve_hot_path(path: str, local_root: Path) -> Path:
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        parts = raw.parts
        if parts and parts[0] in HOT_DIR_NAMES:
            return local_root.joinpath(*parts)
        return raw
    try:
        relative = raw.relative_to(PROJECT_ROOT)
    except ValueError:
        return raw
    parts = relative.parts
    if parts and parts[0] in HOT_DIR_NAMES:
        return local_root.joinpath(*parts)
    return raw


# ---------------------------------------------------------------------------
# Build all_class_embeddings (naive mode only)
# ---------------------------------------------------------------------------

def build_class_embeddings(device, task_names: list) -> torch.Tensor:
    """[768, C_total] -- identical logic to test_classIL_task_prompt.py."""
    print("Building all_class_embeddings for naive mode ...")
    titan = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
    titan = titan.to(device)
    _, templates = brca_prompts()
    all_prompts = []
    for name in task_names:
        class_prompts, _ = _PROMPT_FN_MAP[name]()
        all_prompts.extend(class_prompts)
    with torch.autocast("cuda", torch.float16), torch.inference_mode():
        classifier = titan.zero_shot_classifier(
            all_prompts, templates, device=str(device)
        )
    del titan
    torch.cuda.empty_cache()
    return classifier.to(device)


# ---------------------------------------------------------------------------
# TTA inference loop for 1 task
# ---------------------------------------------------------------------------

def eval_task_tta(
    test_loader,
    task_id:              int,
    tta_model:            MergeSlide_TTA,
    task_to_global_class: dict,
    device,
    mode:                 str  = "tcp",
    num_classes_per_task: list = None,
    verbose_loss:         bool = False,
) -> tuple:
    """
    TTA inference for 1 task. Each slide: adapt_and_predict().

    TCP mode : pred_class is LOCAL index (0..C_task-1), same as original eval_task_tcp.
    Naive mode: pred_class is COLUMN index of all_class_embeddings (= global index 0..12),
                same as original eval_task_naive. targets also converted to global space.
    """
    preds_all           = []
    probs_all           = []
    targets_all         = []
    convert_preds_all   = []
    convert_targets_all = []
    times               = []
    loss_logs           = []
    routing_correct     = 0

    # Naive mode: build column_to_global mapping (mirrors original eval_task_naive)
    if mode == "naive":
        column_to_global = np.array([
            task_to_global_class[t][local]
            for t in range(len(task_to_global_class))
            for local in sorted(task_to_global_class[t].keys())
        ])
        total_classes = len(column_to_global)
    else:
        column_to_global = None
        total_classes    = None

    for features, coords, label in tqdm(test_loader, leave=False):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        features = features.to(device)
        coords   = coords.long().to(device)

        idx = torch.randperm(features.shape[0])[:K_PATCHES]
        features, coords = features[idx], coords[idx]

        pred_class, probs, pred_task, adapt_log = tta_model.adapt_and_predict(
            features, coords
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t0)

        if verbose_loss:
            loss_logs.append(adapt_log)

        probs_np = probs.numpy()  # [1, C_task] for tcp | [1, C_total] for naive

        if mode == "tcp":
            # --- TCP: identical to original eval_task_tcp ---
            # pred_class is LOCAL (0..C_task-1), label is LOCAL
            routing_correct += int(pred_task == task_id)
            preds_all.append(np.array([pred_class]))
            probs_all.append(probs_np)
            targets_all.append(label.numpy())

            g_label = task_to_global_class[task_id].get(int(label), -1)
            g_pred  = task_to_global_class[task_id].get(pred_class, -1)
            convert_targets_all.append(np.array([g_label]))
            convert_preds_all.append(np.array([g_pred]))

        else:
            # --- Naive: identical to original eval_task_naive ---
            # pred_class is COLUMN index (= global class index for forward order)
            pred_global = int(column_to_global[pred_class])
            true_global = task_to_global_class[task_id][int(label)]

            # Remap probs from column order to global class order
            probs_out = np.zeros((1, total_classes), dtype=np.float32)
            for col_idx, g_idx in enumerate(column_to_global):
                probs_out[0, g_idx] = probs_np[0, col_idx]

            preds_all.append(np.array([pred_global]))
            probs_all.append(probs_out)
            targets_all.append(np.array([true_global]))

            convert_preds_all.append(np.array([pred_global]))
            convert_targets_all.append(np.array([true_global]))

    preds_arr   = np.concatenate(preds_all)
    targets_arr = np.concatenate(targets_all)
    try:
        probs_arr = np.concatenate(probs_all)
    except ValueError:
        probs_arr = pad_numpy_arrays(probs_all)

    metrics = get_eval_metrics(
        targets_arr, preds_arr, probs_arr,
        roc_kwargs={"multi_class": "ovo", "average": "macro"},
        prefix="",
    )

    if verbose_loss and loss_logs:
        adapted = [d for d in loss_logs if d.get("slide/adapted")]
        if adapted:
            mean_loss = np.mean([d.get("loss/total_with_reg", 0) for d in adapted])
            print(f"    [TTA] task={task_id} adapted={len(adapted)}/{len(loss_logs)} "
                  f"mean_loss={mean_loss:.4f}")

    return (
        metrics, preds_arr, targets_arr, probs_arr,
        np.concatenate(convert_preds_all),
        np.concatenate(convert_targets_all),
        sum(times),
        routing_correct / max(1, len(test_loader)) if mode == "tcp" else float("nan"),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")

    parser = argparse.ArgumentParser(description="Class-IL TTA evaluation")

    # Paths (same as original)
    parser.add_argument("--config",           type=str, default="configs/default.yaml")
    parser.add_argument("--save_dir",         type=str, required=True)
    parser.add_argument("--merge_model_path", type=str, required=True)

    # Mode (same as original)
    parser.add_argument("--mode", type=str, default="tcp",
                        choices=["tcp", "naive"],
                        help="tcp (default): TCP inference | naive: direct class inference")

    # TTA hyperparams
    parser.add_argument("--M",                 type=int,   default=8)
    parser.add_argument("--K_sub",             type=int,   default=300)
    parser.add_argument("--top_ratio",         type=float, default=0.5)
    parser.add_argument("--alpha",             type=float, default=0.5)
    parser.add_argument("--beta",              type=float, default=1.0)
    parser.add_argument("--lr",                type=float, default=1e-4)
    parser.add_argument("--n_steps",           type=int,   default=1)
    parser.add_argument("--tta_param_scope",    type=str,   default="ln_only",
                        choices=["ln_only", "full"],
                        help="Backbone parameter scope for TTA.")
    parser.add_argument("--entropy_threshold", type=float, default=0.4,
                        help="Only TTA when slide entropy >= threshold. "
                             "Set 0.0 to TTA all slides.")
    parser.add_argument("--episodic",          action="store_true",
                        help="[Deprecated/Ignored] MergeSlide-TTA now always "
                             "uses continual adaptation without per-slide reset.")
    # --- PATCH: bugfix / ablation flags -------------------------------
    parser.add_argument("--use_task_diversity", action="store_true",
                        help="[ABLATION ONLY] Re-enable v1's buggy SHOT-style "
                             "diversity on task-routing logits. Default OFF "
                             "(the fixed, correct behavior). Only pass this "
                             "flag to reproduce/compare against the old bug.")
    parser.add_argument("--no_task_agreement",  action="store_true",
                        help="Disable the new JSD task-agreement (CoTTA-style) "
                             "term. Default: agreement term is ON.")
    parser.add_argument("--gamma",             type=float, default=0.5,
                        help="Weight of the JSD task-agreement term.")
    parser.add_argument("--select_mode",       type=str,   default="intersection",
                        choices=["union", "intersection"],
                        help="Confident sub-bag selection: v1 used 'union'. "
                             "'intersection' (default) is stricter (EATA-style).")
    # --------------------------------------------------------------------
    # PATCH v2.5 -- prompt embedding-space adaptation (ported from
    # tta_engine_v3.py, teacher EMA + gated task-prompt update + margin loss)
    # --------------------------------------------------------------------
    parser.add_argument("--no_teacher",        action="store_true",
                        help="Disable mean-teacher; route/infer with the "
                             "backbone being adapted directly (v1/v2 behavior).")
    parser.add_argument("--ema_alpha",         type=float, default=0.999,
                        help="Teacher EMA momentum.")
    parser.add_argument("--no_adapt_prompts",  action="store_true",
                        help="Disable task-prompt EMA update; task_prompts "
                             "stay frozen (v1/v2 behavior).")
    parser.add_argument("--ema_alpha_prompt",  type=float, default=0.999,
                        help="Task-prompt EMA momentum.")
    parser.add_argument("--delta_margin",      type=float, default=0.10,
                        help="Confidence-gap gate for task-prompt update "
                             "(top1-top2 softmax score over task_prompts).")
    parser.add_argument("--tp_anchor_beta",    type=float, default=0.3,
                        help="Anchor pull-back toward source task_prompts "
                             "in [0,1]. 0 = original tta_engine_v3.py "
                             "behavior (no anchor, unbounded drift). "
                             "1 = prompts never move. Default 0.3.")
    parser.add_argument("--gamma_margin",      type=float, default=0.0,
                        help="Weight of task_margin_loss (0.0 = off).")
    parser.add_argument("--no_reset_prompt_per_task", action="store_true",
                        help="Do NOT reset task_prompts to source between "
                             "tasks. Default: reset per task (bounds "
                             "cross-task drift). Pass this flag only for "
                             "ablation / order-dependence stress-testing.")
    # --------------------------------------------------------------------
    parser.add_argument("--verbose_loss",      action="store_true")
    parser.add_argument(
        "--result_csv",
        type=str,
        default="",
        help="Optional CSV path to save per-fold/per-task TTA metrics, including TCP routing_acc.",
    )
    parser.add_argument(
        "--efficiency_json",
        type=str,
        default="",
        help="Optional JSON path to save updated params, TTA steps, throughput, and peak VRAM.",
    )

    args = parser.parse_args()
    if args.episodic:
        print("[WARN] --episodic is ignored; running continual adaptation without reset.")
    args.episodic = False
    if args.mode == "naive":
        # Naive Class-IL does not use TCP/task prompt routing. Keep the
        # teacher branch disabled by default even when users call this
        # entrypoint directly without the bash runner.
        args.no_teacher = True

    local_hot_root        = ensure_local_hot_storage()
    args.save_dir         = str(resolve_hot_path(args.save_dir,         local_hot_root))
    args.merge_model_path = str(resolve_hot_path(args.merge_model_path, local_hot_root))
    if args.result_csv:
        args.result_csv = str(resolve_hot_path(args.result_csv, local_hot_root))
    if args.efficiency_json:
        args.efficiency_json = str(resolve_hot_path(args.efficiency_json, local_hot_root))
    else:
        args.efficiency_json = str(
            Path(args.merge_model_path) / f"efficiency_classil_tta_{args.mode}.json"
        )

    cfg    = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(device, cfg.training.seed)

    num_tasks   = cfg.training.num_tasks
    seq_dataset = Sequential_Generic_MIL_Dataset(cfg)

    # Load embeddings by mode (same logic as original)
    task_prompts = torch.load(PROJECT_ROOT / "task_prompts.pt").to(device)
    if getattr(cfg.dataset, "order", "forward") == "reverse":
        task_prompts = task_prompts.flip(0)

    if args.mode == "naive":
        all_class_embeddings = build_class_embeddings(device, seq_dataset.task_names)
    else:
        all_class_embeddings = None

    print("Loading TITAN base model ...")
    base_model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
    base_model = base_model.to(device)

    overall_accs         = []
    overall_baccs        = []
    overall_macro_f1s    = []
    overall_weighted_f1s = []
    overall_recalls      = []
    overall_precisions   = []
    overall_aucs         = []
    overall_times        = []
    all_acc_per_task     = []
    overall_routing_acc  = []
    overall_routing_acc_per_task = []
    all_results          = []
    efficiency_params    = None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    eval_wall_start = time.perf_counter()

    for fold_id in tqdm(range(cfg.training.num_folds), desc="Folds"):
        fold = f"fold_{fold_id}"

        merge_path = Path(args.merge_model_path) / fold / "merged_final.pth"
        print(f"\nLoading: {merge_path}")
        base_model.vision_encoder.load_state_dict(
            torch.load(str(merge_path), map_location="cpu")
        )

        task_model_paths = [
            str(Path(args.save_dir) / fold / f"task_{t}.pt")
            for t in range(num_tasks)
        ]
        task_weights = load_task_weights(task_model_paths, device)

        tta_model = MergeSlide_TTA(
            backbone             = base_model.vision_encoder,
            task_prompts         = task_prompts,
            task_weights         = task_weights,
            num_classes          = seq_dataset.num_classes,
            device               = device,
            mode                 = args.mode,
            all_class_embeddings = all_class_embeddings,
            param_scope          = args.tta_param_scope,
            M                    = args.M,
            K_sub                = args.K_sub,
            top_ratio            = args.top_ratio,
            alpha                = args.alpha,
            beta                 = args.beta,
            lr                   = args.lr,
            n_steps              = args.n_steps,
            episodic             = args.episodic,
            entropy_threshold    = args.entropy_threshold,
            use_task_diversity   = args.use_task_diversity,
            use_task_agreement   = (not args.no_task_agreement),
            gamma                = args.gamma,
            select_mode          = args.select_mode,
            use_teacher          = (not args.no_teacher),
            ema_alpha            = args.ema_alpha,
            adapt_task_prompts   = (not args.no_adapt_prompts),
            ema_alpha_prompt     = args.ema_alpha_prompt,
            delta_margin         = args.delta_margin,
            tp_anchor_beta       = args.tp_anchor_beta,
            gamma_margin         = args.gamma_margin,
        )
        if efficiency_params is None:
            efficiency_params = {
                "updated_object": f"{args.tta_param_scope} backbone parameters",
                "updated_params": int(tta_model.updated_params),
                "total_params": int(tta_model.total_params),
                "update_ratio": float(tta_model.update_ratio),
                "ln_layers": int(tta_model.num_ln_layers),
            }

        all_baccs     = []
        all_accs      = []
        all_aucs      = []
        all_preds_g   = []
        all_targets_g = []
        acc_per_task  = {}
        routing_acc_per_task = {}
        fold_time     = 0.0
        num_total     = 0.0

        for task_id in range(num_tasks):
            _, _, test_loader = seq_dataset.get_data_loaders(fold_id, task_id)

            result = eval_task_tta(
                test_loader          = test_loader,
                task_id              = task_id,
                tta_model            = tta_model,
                task_to_global_class = seq_dataset.task_to_global_class,
                device               = device,
                mode                 = args.mode,
                num_classes_per_task = seq_dataset.num_classes,
                verbose_loss         = args.verbose_loss,
            )
            results, preds_all, targets_all, probs_all, \
                conv_preds, conv_targets, task_time, task_routing_acc = result

            num_total += len(test_loader)
            fold_time += task_time / len(test_loader)

            acc_per_task[task_id] = results["/acc"]
            routing_acc_per_task[task_id] = task_routing_acc
            task_bacc = balanced_accuracy_score(targets_all, preds_all)
            task_acc = sum(preds_all == targets_all) / len(test_loader)
            all_baccs.append(task_bacc)
            all_accs.append(task_acc)
            all_preds_g.append(conv_preds)
            all_targets_g.append(conv_targets)

            print(
                f"  [Fold {fold_id}] Task {task_id} ({seq_dataset.task_names[task_id]}) "
                f"ACC={task_acc*100:.4f}% BAcc={task_bacc*100:.4f}% "
                f"routing_acc={task_routing_acc*100:.4f}% "
                f"prompt_updates={tta_model.n_prompt_updates.get(task_id, 0)}"
            )
            all_results.append({
                "fold": fold_id,
                "task_id": task_id,
                "task_name": seq_dataset.task_names[task_id],
                "mode": args.mode,
                "bacc": task_bacc,
                "acc": task_acc,
                "n_samples": len(test_loader),
                "elapsed_s": task_time,
                "routing_acc": task_routing_acc,
            })

            if len(probs_all.shape) == 3:
                probs_all = probs_all.squeeze(1)

            if args.mode == "tcp":
                # Local class indices
                for i in range(seq_dataset.num_classes[task_id]):
                    all_aucs.append(
                        roc_auc_score((targets_all == i).astype(int), probs_all[:, i])
                    )
            else:
                # Global class indices (targets_all already in global space)
                global_idxs = sorted(seq_dataset.task_to_global_class[task_id].values())
                for g_idx in global_idxs:
                    all_aucs.append(
                        roc_auc_score(
                            (targets_all == g_idx).astype(int),
                            probs_all[:, g_idx]
                        )
                    )

        n_adapted = tta_model.n_adapted
        n_skipped = tta_model.n_skipped
        n_total   = n_adapted + n_skipped
        print(f"[Fold {fold_id}] adapted={n_adapted}/{n_total} "
              f"({100*n_adapted/max(n_total,1):.1f}%) | "
              f"entropy_threshold={args.entropy_threshold}")
        tta_model.hard_reset()

        all_preds_g   = np.concatenate(all_preds_g)
        all_targets_g = np.concatenate(all_targets_g)

        overall_accs.append(np.mean(all_accs))
        overall_baccs.append(np.mean(all_baccs))
        overall_macro_f1s.append(
            f1_score(all_targets_g, all_preds_g, average="macro"))
        overall_weighted_f1s.append(
            f1_score(all_targets_g, all_preds_g, average="weighted"))
        overall_recalls.append(
            recall_score(all_targets_g, all_preds_g, average=None))
        overall_precisions.append(
            precision_score(all_targets_g, all_preds_g, average=None))
        overall_aucs.append(np.array(all_aucs))
        overall_times.append(fold_time / num_tasks)
        all_acc_per_task.append(acc_per_task)
        valid_routing = [v for v in routing_acc_per_task.values() if not np.isnan(v)]
        overall_routing_acc.append(np.mean(valid_routing) if valid_routing else float("nan"))
        overall_routing_acc_per_task.append(routing_acc_per_task)

        print(f"[Fold {fold_id}] Acc={np.mean(all_accs)*100:.4f}% "
              f"BAcc={np.mean(all_baccs)*100:.4f}%")

    reset_label = "episodic" if args.episodic else "continual"
    print(
        f"\n===== Class-IL TTA ({args.mode.upper()}, {reset_label}, "
        f"{args.tta_param_scope}, M={args.M}) ====="
    )
    print(f"Accuracy:       {np.mean(overall_accs)*100:.4f}%"
          f" ({np.std(overall_accs)*100:.4f}%)")
    print(f"Balanced Acc:   {np.mean(overall_baccs)*100:.4f}%"
          f" ({np.std(overall_baccs)*100:.4f}%)")
    print(f"Macro F1:       {np.mean(overall_macro_f1s)*100:.4f}%"
          f" ({np.std(overall_macro_f1s)*100:.4f}%)")
    print(f"Weighted F1:    {np.mean(overall_weighted_f1s)*100:.4f}%"
          f" ({np.std(overall_weighted_f1s)*100:.4f}%)")
    print(f"Inference time: {np.mean(overall_times):.3f}s"
          f" ({np.std(overall_times):.3f}s)")

    print("\nAcc per task:")
    accs = {t: [] for t in range(num_tasks)}
    for fold_acc in all_acc_per_task:
        for t in range(num_tasks):
            accs[t].append(fold_acc[t])
    for t in range(num_tasks):
        print(f"  Task {t}: {np.mean(accs[t])*100:.4f}%"
              f" ({np.std(accs[t])*100:.4f}%)")

    if args.mode == "tcp":
        print("\nRouting Accuracy per task (pred_task == true_task):")
        routing_by_task = {t: [] for t in range(num_tasks)}
        for fold_r in overall_routing_acc_per_task:
            for t in range(num_tasks):
                v = fold_r.get(t, float("nan"))
                if not np.isnan(v):
                    routing_by_task[t].append(v)
        for t in range(num_tasks):
            vals = routing_by_task[t]
            if vals:
                print(f"  Routing Task {t}: {np.mean(vals)*100:.4f}%"
                      f" ({np.std(vals)*100:.4f}%)")
            else:
                print(f"  Routing Task {t}: N/A")
        overall_routing_valid = [v for v in overall_routing_acc if not np.isnan(v)]
        if overall_routing_valid:
            print(f"Routing Accuracy (mean): {np.mean(overall_routing_valid)*100:.4f}%"
                  f" ({np.std(overall_routing_valid)*100:.4f}%)")

    if args.result_csv:
        result_csv_path = Path(args.result_csv)
        result_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with result_csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n[INFO] Saved result CSV: {result_csv_path}")

    total_elapsed_s = float(time.perf_counter() - eval_wall_start)
    total_slide_count = int(sum(row["n_samples"] for row in all_results))
    timed_elapsed_s = float(sum(row["elapsed_s"] for row in all_results))
    time_per_slide_s = timed_elapsed_s / max(total_slide_count, 1)
    peak_vram_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        if device.type == "cuda" else 0.0
    )
    efficiency = {
        "method": "MergeSlide-TTA",
        "eval_setting": "class_il",
        "mode": args.mode,
        "param_scope": args.tta_param_scope,
        "tta_steps": int(args.n_steps),
        "patches_per_wsi": int(K_PATCHES),
        "subbags": int(args.M),
        "patches_per_subbag": int(args.K_sub),
        "num_slides": total_slide_count,
        "timing_scope": "per-slide online TTA update plus final prediction; checkpoint/model setup excluded",
        "timing_cuda_synchronized": device.type == "cuda",
        "adapt_merge_elapsed_s": None,
        "inference_only_elapsed_s": None,
        "online_adapt_inference_elapsed_s": timed_elapsed_s,
        "end_to_end_elapsed_s": timed_elapsed_s,
        "timed_elapsed_s": timed_elapsed_s,
        "wall_elapsed_s": total_elapsed_s,
        "inference_only_time_per_slide_s": None,
        "end_to_end_time_per_slide_s": time_per_slide_s,
        "time_per_slide_s": time_per_slide_s,
        "end_to_end_throughput_slides_per_s": total_slide_count / max(timed_elapsed_s, 1e-12),
        "throughput_slides_per_s": total_slide_count / max(timed_elapsed_s, 1e-12),
        "peak_vram_eval_mb": peak_vram_mb,
        "peak_vram_adapt_mb": peak_vram_mb,
        "backprop": True,
        "source_free": True,
        "label_free": True,
        **(efficiency_params or {}),
    }
    efficiency_path = Path(args.efficiency_json)
    efficiency_path.parent.mkdir(parents=True, exist_ok=True)
    efficiency_path.write_text(json.dumps(efficiency, indent=2), encoding="utf-8")
    print(f"[EFFICIENCY] {json.dumps(efficiency, sort_keys=True)}")
    print(f"[INFO] Saved efficiency JSON: {efficiency_path}")
