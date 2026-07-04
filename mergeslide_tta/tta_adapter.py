"""
tta_adapter.py - MergeSlide-TTA core module.

Pipeline per slide:
  1. Quick no-grad forward -> compute entropy (WSI-level filter)
  2. If entropy < threshold (IND slide) -> skip TTA, return directly
  3. If entropy >= threshold (OOD slide) -> create M sub-bags -> forward
     -> compute dual-level loss -> update selected backbone params -> re-infer

Batch size = M = 8 sub-bags per slide, each sub-bag has K_sub = 300 patches.

Modes:
  tcp   -- TCP routing: t_hat from task_prompts -> class logits from task MLP
  naive -- use all_class_embeddings [768, C_total] directly

Param scopes:
  ln_only -- update only LayerNorm weight/bias in the backbone
  full    -- update all backbone parameters
"""

from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mergeslide_tta.constants import TITAN_PS_ARG
from mergeslide_tta.tta_losses import (
    dual_level_tta_loss,
    l2_anchor_loss,
    select_confident_subbags,
    select_confident_subbags_intersection,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_ln_params(model: nn.Module) -> Tuple[List[torch.Tensor], List[str]]:
    """Collect weight + bias of all nn.LayerNorm in model."""
    params, names = [], []
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                params.append(module.weight)
                names.append(f"{name}.weight")
            if module.bias is not None:
                params.append(module.bias)
                names.append(f"{name}.bias")
    return params, names


def collect_adaptation_params(
    model: nn.Module,
    param_scope: str = "ln_only",
) -> Tuple[List[torch.Tensor], List[str]]:
    """Collect trainable parameters according to the chosen adaptation scope."""
    if param_scope == "ln_only":
        return collect_ln_params(model)
    if param_scope == "full":
        params, names = [], []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params.append(param)
                names.append(name)
        return params, names
    raise ValueError(f"Unsupported param_scope: {param_scope}")


def configure_backbone_for_tta(
    backbone: nn.Module,
    param_scope: str = "ln_only",
) -> nn.Module:
    """Configure which backbone params are trainable during TTA."""
    backbone.train()
    backbone.requires_grad_(False)
    if param_scope == "ln_only":
        for module in backbone.modules():
            if isinstance(module, nn.LayerNorm):
                module.requires_grad_(True)
    elif param_scope == "full":
        backbone.requires_grad_(True)
    else:
        raise ValueError(f"Unsupported param_scope: {param_scope}")
    return backbone


# ---------------------------------------------------------------------------
# MergeSlide-TTA
# ---------------------------------------------------------------------------

class MergeSlide_TTA(nn.Module):
    """
    Test-Time Adaptation wrapper for MergeSlide.

    Args:
        backbone             : model.backbone (TITAN vision_encoder after merging)
        task_prompts         : [T, 768] task-level prompt embeddings (frozen)
        task_weights         : list of dict{'weight', 'bias'} MLP weights per task
        num_classes          : list of int, classes per task
        device               : torch.device
        mode                 : 'tcp' or 'naive' -- mirrors --mode in original eval
        all_class_embeddings : [768, C_total] required when mode='naive'
        M                    : sub-bags per slide = TTA batch size (default 8)
        K_sub                : patches per sub-bag (default 300)
        top_ratio            : confident sub-bag keep ratio (default 0.5)
        alpha                : task-level loss weight (default 0.5)
        beta                 : L2 anchor regularizer weight (default 1.0)
        lr                   : Adam learning rate (default 1e-4)
        n_steps              : adapt steps per slide (default 1)
        episodic             : reset LN params after each slide (default False = continual)
        entropy_threshold    : only TTA when entropy >= threshold (default 0.4)
    """

    def __init__(
        self,
        backbone:             nn.Module,
        task_prompts:         torch.Tensor,
        task_weights:         List[Dict],
        num_classes:          List[int],
        device:               torch.device,
        mode:                 str                     = "tcp",
        all_class_embeddings: Optional[torch.Tensor] = None,
        fixed_task_id:        Optional[int]           = None,
        param_scope:          str  = "ln_only",
        M:                    int   = 8,
        K_sub:                int   = 300,
        top_ratio:            float = 0.5,
        alpha:                float = 0.5,
        beta:                 float = 1.0,
        lr:                   float = 1e-4,
        n_steps:              int   = 1,
        episodic:             bool  = False,
        entropy_threshold:    float = 0.4,
        use_task_diversity:   bool  = False,  # PATCH: was implicitly True (the bug). Keep False.
        use_task_agreement:   bool  = True,   # PATCH: new CoTTA-style consistency term
        gamma:                float = 0.5,    # PATCH: weight of agreement term
        select_mode:          str   = "intersection",  # PATCH: "union" (v1 behavior) or "intersection"
    ):
        super().__init__()

        assert mode in ("tcp", "naive", "task_il"), \
            f"mode must be 'tcp', 'naive', or 'task_il', got: {mode}"
        if mode == "naive":
            assert all_class_embeddings is not None, \
                "all_class_embeddings required when mode='naive'"
        if mode == "task_il":
            assert fixed_task_id is not None, \
                "fixed_task_id required when mode='task_il'"
        if param_scope not in ("ln_only", "full"):
            raise ValueError(f"param_scope must be 'ln_only' or 'full', got: {param_scope}")

        self.param_scope          = param_scope
        self.backbone             = configure_backbone_for_tta(backbone, param_scope)
        self.device               = device
        self.mode                 = mode
        self.task_prompts         = task_prompts.to(device)
        self.task_weights         = task_weights
        self.num_classes          = num_classes
        self.all_class_embeddings = (
            all_class_embeddings.detach().clone().to(device)
            if all_class_embeddings is not None else None
        )
        self.fixed_task_id        = fixed_task_id
        self.M                    = M
        self.K_sub                = K_sub
        self.top_ratio            = top_ratio
        self.alpha                = alpha
        self.beta                 = beta
        self.n_steps              = n_steps
        self.episodic             = episodic
        self.entropy_threshold    = entropy_threshold
        self.use_task_diversity   = use_task_diversity
        self.use_task_agreement   = use_task_agreement
        self.gamma                = gamma
        self.select_mode          = select_mode
        assert select_mode in ("union", "intersection"), \
            f"select_mode must be 'union' or 'intersection', got: {select_mode}"
        self.ps                   = torch.tensor(TITAN_PS_ARG).int().to(device)

        self.n_adapted = 0
        self.n_skipped = 0

        adapt_params, self.adapt_names = collect_adaptation_params(
            self.backbone, self.param_scope
        )
        self.adapt_params_anchor: List[torch.Tensor] = [
            p.detach().clone() for p in adapt_params
        ]
        self.ln_params_anchor = self.adapt_params_anchor  # backward-compatible alias

        self.optimizer = torch.optim.Adam(adapt_params, lr=lr)

        self._init_backbone = deepcopy(self.backbone.state_dict())
        self._init_optim    = deepcopy(self.optimizer.state_dict())

        mode_info = (f"mode={mode}" if mode != "task_il"
                     else f"mode=task_il(task={fixed_task_id})")
        num_ln      = len([m for m in self.backbone.modules()
                           if isinstance(m, nn.LayerNorm)])
        n_trainable = sum(p.numel() for p in adapt_params)
        n_total     = sum(p.numel() for p in self.backbone.parameters())
        self.num_ln_layers = num_ln
        self.updated_params = n_trainable
        self.total_params = n_total
        self.update_ratio = n_trainable / max(n_total, 1)
        print(
            f"[MergeSlide-TTA] {mode_info} | LN layers={num_ln} | "
            f"param_scope={param_scope} | trainable_params={n_trainable:,}/{n_total:,} | "
            f"M={M} sub-bags | K_sub={K_sub} | "
            f"top_ratio={top_ratio} | alpha={alpha} | beta={beta} | "
            f"lr={lr} | n_steps={n_steps} | episodic={episodic} | "
            f"entropy_threshold={entropy_threshold} | "
            f"select_mode={select_mode} | use_task_diversity={use_task_diversity} | "
            f"use_task_agreement={use_task_agreement} | gamma={gamma}"
        )

    # -----------------------------------------------------------------------
    # Sub-bag creation
    # -----------------------------------------------------------------------

    def _make_subbags(
        self, features: torch.Tensor, coords: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        K    = features.shape[0]
        k_sub = min(self.K_sub, K)
        feat_list, coord_list = [], []
        for _ in range(self.M):
            idx = torch.randperm(K, device=features.device)[:k_sub]
            feat_list.append(features[idx])
            coord_list.append(coords[idx])
        return feat_list, coord_list

    # -----------------------------------------------------------------------
    # Forward sub-bags
    # -----------------------------------------------------------------------

    def _forward_subbags(
        self,
        feat_list:  List[torch.Tensor],
        coord_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """Returns [M, 768] keeping computation graph alive (grad through LN)."""
        embeds = [self.backbone(f, c, self.ps)
                  for f, c in zip(feat_list, coord_list)]
        return torch.cat(embeds, dim=0)

    # -----------------------------------------------------------------------
    # Class logits by mode
    # -----------------------------------------------------------------------

    def _class_logits_tcp(
        self, embeds: torch.Tensor, task_id: int
    ) -> torch.Tensor:
        """[N, C_task] using frozen MLP weights of task_id."""
        w = self.task_weights[task_id]["weight"].detach()
        b = self.task_weights[task_id]["bias"].detach()
        return F.linear(embeds.float(), w, b)

    def _class_logits_naive(self, embeds: torch.Tensor) -> torch.Tensor:
        """[N, C_total] using all_class_embeddings."""
        return embeds.float() @ self.all_class_embeddings.detach()

    # -----------------------------------------------------------------------
    # 1 adaptation step
    # -----------------------------------------------------------------------

    @torch.enable_grad()
    def _adapt_step(
        self, features: torch.Tensor, coords: torch.Tensor,
    ) -> dict:
        feat_list, coord_list = self._make_subbags(features, coords)
        embeds      = self._forward_subbags(feat_list, coord_list)   # [M, 768]
        task_logits = embeds.float() @ self.task_prompts.T.detach()  # [M, T]

        if self.mode == "tcp":
            with torch.no_grad():
                mean_z = embeds.detach().float().mean(dim=0, keepdim=True)
                t_hat  = int(
                    (mean_z @ self.task_prompts.T.detach()).argmax(dim=1)
                )
            class_logits = self._class_logits_tcp(embeds, t_hat)
        elif self.mode == "task_il":
            # Task identity known: use fixed task, no routing
            t_hat        = self.fixed_task_id
            class_logits = self._class_logits_tcp(embeds, t_hat)
        else:
            t_hat        = -1
            class_logits = self._class_logits_naive(embeds)

        if self.select_mode == "intersection":
            sel_idx = select_confident_subbags_intersection(
                class_logits.detach(), task_logits.detach(), self.top_ratio
            )
        else:  # "union" -- v1 behavior, kept for ablation comparison
            _, idx_class = select_confident_subbags(class_logits.detach(), self.top_ratio)
            _, idx_task  = select_confident_subbags(task_logits.detach(),  self.top_ratio)
            sel_idx      = torch.unique(torch.cat([idx_class, idx_task]))

        # Loss mode:
        #   tcp     : class entropy + class diversity + alpha * (task entropy + agreement)
        #   naive   : class entropy only (alpha=0, no diversity over 13 classes)
        #   task_il : class entropy + class diversity only (alpha=0, task routing irrelevant)
        if self.mode == "naive":
            effective_alpha = 0.0
            use_class_div   = False
            use_task_div    = False
            use_task_agree  = False
        elif self.mode == "task_il":
            effective_alpha = 0.0    # no task loss, task is already known
            use_class_div   = True   # diversity over C_task=2~3 classes is meaningful
            use_task_div    = False
            use_task_agree  = False
        else:  # tcp
            effective_alpha = self.alpha
            use_class_div   = True
            use_task_div    = self.use_task_diversity   # default False (bug fixed)
            use_task_agree  = self.use_task_agreement    # default True (PATCH)

        loss, log = dual_level_tta_loss(
            class_logits[sel_idx], task_logits[sel_idx],
            effective_alpha,
            use_class_diversity=use_class_div,
            use_task_diversity=use_task_div,
            use_task_agreement=use_task_agree,
            gamma=self.gamma,
        )

        if self.beta > 0:
            adapt_params = [p for p in self.backbone.parameters() if p.requires_grad]
            reg          = l2_anchor_loss(adapt_params, self.adapt_params_anchor)
            loss      = loss + self.beta * reg
            log["loss/l2_reg"] = reg.item()

        log["loss/total_with_reg"] = loss.item()
        log["adapt/t_hat"]         = t_hat
        log["adapt/n_selected"]    = sel_idx.numel()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return log

    # -----------------------------------------------------------------------
    # Quick inference (no grad)
    # -----------------------------------------------------------------------

    def _quick_inference(
        self, features: torch.Tensor, coords: torch.Tensor,
    ) -> Tuple[int, torch.Tensor, int, float]:
        """
        No-grad forward with full K patches.
        Returns (pred_class, probs[1,C], pred_task, entropy_value).
        """
        self.backbone.eval()
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            z           = self.backbone(features, coords, self.ps)
            task_logits = z.float() @ self.task_prompts.T
            pred_task   = int(task_logits.argmax(dim=1))

            if self.mode == "tcp":
                class_logits = self._class_logits_tcp(z.float(), pred_task)
            elif self.mode == "task_il":
                # Use fixed task -- override pred_task with known identity
                pred_task    = self.fixed_task_id
                class_logits = self._class_logits_tcp(z.float(), pred_task)
            else:
                class_logits = self._class_logits_naive(z.float())

            probs      = F.softmax(class_logits.float(), dim=1)
            pred_class = int(class_logits.argmax(dim=1))
            entropy    = -(probs * probs.clamp(min=1e-8).log()).sum().item()

        return pred_class, probs.cpu(), pred_task, entropy

    # -----------------------------------------------------------------------
    # Public: adapt + predict
    # -----------------------------------------------------------------------

    def adapt_and_predict(
        self, features: torch.Tensor, coords: torch.Tensor,
    ) -> Tuple[int, torch.Tensor, int, dict]:
        """
        WSI-level filter + TTA + inference for 1 slide.

        Flow:
          1. Quick forward -> entropy
          2. entropy < threshold (IND) -> skip TTA, return directly
          3. entropy >= threshold (OOD) -> TTA -> re-infer

        Returns:
            pred_class : int
            probs      : [1, C] softmax probs (CPU)
            pred_task  : int
            adapt_log  : dict
        """
        pred_class, probs, pred_task, entropy = self._quick_inference(
            features, coords
        )
        adapt_log = {"slide/entropy": entropy, "slide/adapted": False}

        if entropy < self.entropy_threshold:
            self.n_skipped += 1
            self.backbone.train()
            return pred_class, probs, pred_task, adapt_log

        self.n_adapted += 1

        if self.episodic:
            self._reset()

        self.backbone.train()

        for _ in range(self.n_steps):
            adapt_log = self._adapt_step(features, coords)

        pred_class, probs, pred_task, _ = self._quick_inference(features, coords)
        adapt_log["slide/entropy"] = entropy
        adapt_log["slide/adapted"] = True

        self.backbone.train()
        return pred_class, probs, pred_task, adapt_log

    # -----------------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------------

    def _reset(self):
        self.backbone.load_state_dict(self._init_backbone, strict=True)
        self.optimizer.load_state_dict(self._init_optim)

    def hard_reset(self):
        """Call after each fold to restore params and reset counters."""
        self._reset()
        self.n_adapted = 0
        self.n_skipped = 0


# ---------------------------------------------------------------------------
# Helper: load task MLP weights
# ---------------------------------------------------------------------------

def load_task_weights(
    task_model_paths: List[str],
    device:           torch.device,
) -> List[Dict]:
    """
    Load MLP weight + bias for each task from task_{t}.pt checkpoint.
    Takes the last 'weight' and 'bias' keys (MLP linear head).
    """
    task_weights = []
    for path in task_model_paths:
        state = torch.load(path, map_location=device)
        keys  = list(state.keys())
        w_key = next(k for k in reversed(keys) if "weight" in k)
        b_key = next(k for k in reversed(keys) if "bias"   in k)
        task_weights.append({
            "weight": state[w_key].to(device),
            "bias":   state[b_key].to(device),
        })
    return task_weights