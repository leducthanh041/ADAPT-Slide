# merge.py
"""
C.OPCM continual model merging for ADAPT-Slide.

For each fold, sequentially merge task checkpoints into one vision encoder and
save both intermediate and final checkpoints for downstream evaluation.

Usage:
    python merge.py --config configs/default.yaml
    python merge.py --config configs/default.yaml --fold_start 0 --fold_end 1
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from tqdm import tqdm
from transformers import AutoModel
from omegaconf import OmegaConf

from adapt_slide.checkpoint_mirror import save_checkpoint_with_mirror
from adapt_slide.utils import (
    get_task_vector_norm,
    get_task_vector_state_dict,
    is_leaf_module,
    svd,
)


# ---------------------------------------------------------------------------
# C.OPCM core functions
# ---------------------------------------------------------------------------

def merge_linear_weights(
    merged_W: Tensor,
    pretrained_W: Tensor,
    task_W: Tensor,
    previous_lambda_t: float,
    lambda_t: float,
    accelerator: str = "cpu",
) -> Tensor:
    """
    Merge Linear layer weights with C.OPCM.

    The task update is projected through the SVD basis of the current merged
    task vector, with diagonal components removed before accumulation.
    """
    original_device = merged_W.device
    merged_W    = merged_W.to(accelerator)
    pretrained_W = pretrained_W.to(accelerator)
    task_W      = task_W.to(accelerator)

    previous_merged_tv = merged_W - pretrained_W
    task_tv            = task_W - pretrained_W

    u, s, v = svd(previous_merged_tv)
    projected_task_tv = u.T @ task_tv @ v
    projected_task_tv.diag().fill_(0)
    cleaned_task_tv = u @ projected_task_tv @ v.T

    new_merged_W = (
        pretrained_W
        + (previous_lambda_t * previous_merged_tv + cleaned_task_tv) / lambda_t
    )
    return new_merged_W.to(original_device)


def merge_other_parameters(
    merged_W: Tensor,
    pretrained_W: Tensor,
    task_W: Tensor,
    previous_lambda_t: float,
    lambda_t: float,
    accelerator: str = "cpu",
) -> Tensor:
    """
    Merge non-Linear-weight parameters with direct weighted accumulation.
    """
    original_device = merged_W.device
    merged_W    = merged_W.to(accelerator)
    pretrained_W = pretrained_W.to(accelerator)
    task_W      = task_W.to(accelerator)

    previous_merged_tv = merged_W - pretrained_W
    task_tv            = task_W - pretrained_W

    new_merged_W = (
        pretrained_W
        + (previous_lambda_t * previous_merged_tv + task_tv) / lambda_t
    )
    return new_merged_W.to(original_device)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def extract_backbone_weights(ckpt_path: str) -> dict:
    """
    Load a checkpoint and extract backbone weights.

    Args:
        ckpt_path: Path to a .pt checkpoint.

    Returns:
        State dict containing backbone keys with the 'backbone.' prefix removed.
    """
    state = torch.load(ckpt_path, map_location="cpu")
    backbone_keys = list(state.keys())[:-2]
    return {
        k.split("backbone.")[-1]: state[k].detach()
        for k in backbone_keys
    }


def merge_one_task(
    base_model: nn.Module,
    merged_weight: dict,
    task_weight: dict,
    previous_lambda_t: float,
    lambda_t: float,
    task_idx: int,
) -> dict:
    """
    Merge one task checkpoint into the current vision encoder weights.

    Args:
        base_model: Frozen TITAN base model used to access pretrained weights.
        merged_weight: Current merged-model state dict.
        task_weight: Task-specific state dict to merge.
        previous_lambda_t: Previous merge coefficient.
        lambda_t: Current merge coefficient.
        task_idx: Current task index for progress logging.

    Returns:
        Updated merged_weight state dict.
    """
    vision_encoder = base_model.vision_encoder

    for module_name, module in tqdm(
        list(vision_encoder.named_modules()),
        desc=f"Merging task {task_idx}",
        leave=False,
    ):
        if not is_leaf_module(module):
            continue

        pretrained_module = vision_encoder.get_submodule(module_name)

        if isinstance(module, nn.Linear):
            # Linear weights use SVD projection.
            merged_weight[f"{module_name}.weight"] = merge_linear_weights(
                merged_W    = merged_weight[f"{module_name}.weight"],
                pretrained_W = pretrained_module.weight.detach(),
                task_W      = task_weight[f"{module_name}.weight"],
                previous_lambda_t=previous_lambda_t,
                lambda_t=lambda_t,
            )
            # Linear biases are merged directly.
            if module.bias is not None:
                merged_weight[f"{module_name}.bias"] = merge_other_parameters(
                    merged_W    = merged_weight[f"{module_name}.bias"],
                    pretrained_W = pretrained_module.bias.detach(),
                    task_W      = task_weight[f"{module_name}.bias"],
                    previous_lambda_t=previous_lambda_t,
                    lambda_t=lambda_t,
                )
        else:
            # Other parameters, including LayerNorm, are merged directly.
            for param_name, _ in module.named_parameters():
                key = f"{module_name}.{param_name}"
                merged_weight[key] = merge_other_parameters(
                    merged_W    = merged_weight[key],
                    pretrained_W = pretrained_module.get_parameter(param_name).detach(),
                    task_W      = task_weight[key],
                    previous_lambda_t=previous_lambda_t,
                    lambda_t=lambda_t,
                )

    return merged_weight


def normalize_merged_weight(
    base_model: nn.Module,
    merged_weight: dict,
    avg_task_vector_norm: float,
) -> dict:
    """
    Rescale the merged task vector to avoid magnitude drift.
    """
    vision_encoder = base_model.vision_encoder
    task_vector_norm = get_task_vector_norm(
        merged_weight,
        {k: v.detach() for k, v in vision_encoder.state_dict().items()},
    )

    for param_name, param in vision_encoder.named_parameters():
        base_W = param.detach()
        task_vector = merged_weight[param_name] - base_W
        merged_weight[param_name] = base_W + task_vector * (
            avg_task_vector_norm / task_vector_norm
        )

    return merged_weight, task_vector_norm


# ---------------------------------------------------------------------------
# Main merging pipeline
# ---------------------------------------------------------------------------

def run_merging_for_fold(
    fold_id: int,
    base_model: nn.Module,
    src_dir: str,
    dst_dir: str,
    num_tasks: int,
) -> None:
    """
    Run C.OPCM merging for one fold.

    Args:
        fold_id: Fold index.
        base_model: Loaded frozen TITAN base model.
        src_dir: Directory containing per-task fine-tuned checkpoints.
        dst_dir: Directory for merged checkpoints.
        num_tasks: Number of tasks to merge.
    """
    fold_name   = f"fold_{fold_id}"
    output_dir  = Path(dst_dir) / fold_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-task checkpoint paths.
    task_ckpt_paths = [
        str(Path(src_dir) / fold_name / f"task_{t}.pt")
        for t in range(num_tasks)
    ]

    base_weight = {
        k: v.detach()
        for k, v in base_model.vision_encoder.state_dict().items()
    }

    # Initialize from task 0 before sequential merging starts.
    merged_weight        = extract_backbone_weights(task_ckpt_paths[0])
    previous_lambda_t    = 1.0
    avg_task_vector_norm = get_task_vector_norm(merged_weight, base_weight)
    all_task_vector_norms = [avg_task_vector_norm]

    print(f"\n[Fold {fold_id}] Task 0 norm: {avg_task_vector_norm:.4f}")

    # Sequentially merge the remaining tasks.
    for model_idx, task_ckpt in enumerate(task_ckpt_paths[1:], start=1):
        task_weight = extract_backbone_weights(task_ckpt)

        all_task_vector_norms.append(get_task_vector_norm(task_weight, base_weight))
        avg_task_vector_norm = float(np.mean(all_task_vector_norms))

        lambda_t = 1.0

        merged_weight = merge_one_task(
            base_model=base_model,
            merged_weight=merged_weight,
            task_weight=task_weight,
            previous_lambda_t=previous_lambda_t,
            lambda_t=lambda_t,
            task_idx=model_idx,
        )

        # Rescale lambda according to task-vector norm.
        merged_weight, task_vector_norm = normalize_merged_weight(
            base_model, merged_weight, avg_task_vector_norm
        )
        lambda_t           = lambda_t * (task_vector_norm / avg_task_vector_norm)
        previous_lambda_t  = lambda_t

        # Save intermediate checkpoints for continual metrics.
        intermediate_path = output_dir / f"merged_task_{model_idx}.pth"
        primary_path, mirror_path = save_checkpoint_with_mirror(merged_weight, intermediate_path)
        if mirror_path is not None:
            print(f"[Fold {fold_id}] Task {model_idx} merged → {primary_path} | Mirror: {mirror_path}")
        else:
            print(f"[Fold {fold_id}] Task {model_idx} merged → {primary_path}")

    # Save the final checkpoint for Class-IL evaluation.
    final_path = output_dir / f"merged_final.pth"
    primary_path, mirror_path = save_checkpoint_with_mirror(merged_weight, final_path)
    if mirror_path is not None:
        print(f"[Fold {fold_id}] Final checkpoint → {primary_path} | Mirror: {mirror_path}")
    else:
        print(f"[Fold {fold_id}] Final checkpoint → {primary_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="C.OPCM Continual Model Merging")
    parser.add_argument("--config",     type=str, default="configs/default.yaml")
    parser.add_argument("--fold_start", type=int, default=None,
                        help="Override fold start")
    parser.add_argument("--fold_end",   type=int, default=None,
                        help="Override fold end")
    parser.add_argument("--finetuned_checkpoints", type=str, default=None,
                        help="Override cfg.paths.finetuned_checkpoints")
    parser.add_argument("--merged_checkpoints", type=str, default=None,
                        help="Override cfg.paths.merged_checkpoints")
    args = parser.parse_args()

    cfg        = OmegaConf.load(args.config)
    fold_start = args.fold_start if args.fold_start is not None else 0
    fold_end   = args.fold_end   if args.fold_end   is not None else cfg.training.num_folds
    num_tasks  = cfg.training.num_tasks

    src_dir = args.finetuned_checkpoints or cfg.paths.finetuned_checkpoints
    dst_dir = args.merged_checkpoints or cfg.paths.merged_checkpoints

    print(f"[INFO] finetuned checkpoints: {src_dir}")
    print(f"[INFO] merged checkpoints: {dst_dir}")

    print("Loading TITAN base model ...")
    base_model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
    base_model.eval()

    for fold_id in range(fold_start, fold_end):
        run_merging_for_fold(
            fold_id=fold_id,
            base_model=base_model,
            src_dir=src_dir,
            dst_dir=dst_dir,
            num_tasks=num_tasks,
        )

    print("\nMerging complete.")
