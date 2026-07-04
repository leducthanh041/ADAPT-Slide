"""
tta_losses.py - Loss functions for MergeSlide-TTA. (PATCHED)

Sources:
  entropy_loss      : TENT (Wang et al., ICLR 2021)
  diversity_loss    : SHOT (Liang et al., TPAMI 2021)
  task_agreement_loss (JSD) : CoTTA-style consistency, replaces SHOT-diversity
                              for task ROUTING (multi-view of the SAME slide)
  select_confident  : TPT  (Shu et al., NeurIPS 2022)
  l2_anchor_loss    : EATA (Niu et al., ICML 2022) simplified

--------------------------------------------------------------------------
BUG FIXED (v1 -> v2):
  Previously `dual_level_tta_loss` applied SHOT's diversity term to
  task_logits. SHOT's diversity assumption (mean prediction over a batch
  SHOULD be close to uniform) is valid for a batch of DIFFERENT samples
  with different true labels. It is INVALID for task-routing logits here,
  because all M sub-bags in a batch come from sub-sampling the SAME slide
  -> their true task label is IDENTICAL. Rewarding the mean routing
  prediction for being close to uniform actively teaches the M sub-bags to
  DISAGREE with each other on which task the slide belongs to -- this is
  the opposite of what TCP routing needs (near one-hot agreement).

  Fix: task-level diversity is replaced by an agreement objective. Default
  config disables SHOT-style task diversity entirely (use_task_diversity is
  no longer used, kept as forced-False path for backward compatibility) and
  optionally adds a Jensen-Shannon-Divergence agreement loss across the M
  sub-bag routing distributions (minimize disagreement, CoTTA-style
  consistency regularization instead of SHOT-style diversity).
--------------------------------------------------------------------------
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple


def entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    """Mean Shannon entropy over batch. logits: [N, C]"""
    probs = F.softmax(logits, dim=1).clamp(min=1e-8)
    return -(probs * probs.log()).sum(dim=1).mean()


def diversity_loss(logits: torch.Tensor) -> torch.Tensor:
    """Marginal entropy over batch (SHOT diversity term). logits: [N, C]
    Only valid when rows correspond to DIFFERENT true labels (e.g. class-level,
    multiple sub-bags may genuinely contain different tissue/class content).
    Do NOT use this for task-routing logits of sub-bags from the same slide.
    """
    mean_probs = F.softmax(logits, dim=1).mean(dim=0).clamp(min=1e-8)
    return -(mean_probs * mean_probs.log()).sum()


def task_agreement_loss(task_logits: torch.Tensor) -> torch.Tensor:
    """
    Jensen-Shannon-Divergence based agreement loss across M sub-bags.

    All M sub-bags come from the SAME slide -> should agree on task routing.
    Minimizing this loss pulls the M routing distributions together
    (CoTTA-style consistency), which is the correct inductive bias here --
    the opposite of SHOT's diversity term.

    task_logits: [M, T]
    Returns a scalar in [0, log 2] (JSD upper bound), 0 = perfect agreement.
    """
    probs = F.softmax(task_logits, dim=1).clamp(min=1e-8)   # [M, T]
    mean_probs = probs.mean(dim=0, keepdim=True).clamp(min=1e-8)  # [1, T]
    # mean KL(p_i || mean_p) over the M sub-bags = JSD-like agreement measure
    kl = (probs * (probs.log() - mean_probs.log())).sum(dim=1)  # [M]
    return kl.mean()


def dual_level_tta_loss(
    class_logits:        torch.Tensor,
    task_logits:         torch.Tensor,
    alpha:                float = 0.5,
    use_class_diversity:  bool  = True,
    use_task_diversity:   bool  = False,   # PATCH: default OFF (was the bug)
    use_task_agreement:   bool  = True,    # PATCH: new, CoTTA-style
    gamma:                float = 0.5,     # PATCH: weight for agreement term
) -> Tuple[torch.Tensor, dict]:
    """
    Combined class-level and task-level objective. (PATCHED SIGNATURE)

    L_class = H_class_ent [- H_class_div]           (SHOT, class-level only)
    L_task  = H_task_ent  [- H_task_div]             (kept for ablation, OFF by default)
              [+ gamma * JSD_agreement]              (PATCH: replaces task diversity)

    total = L_class + alpha * L_task

    Args:
        class_logits        : [N, C_task] (tcp) or [N, C_total] (naive)
        task_logits          : [N, T]
        alpha                : task loss weight (set 0.0 for naive / task_il)
        use_class_diversity  : SHOT diversity on class logits (valid, keep True for tcp)
        use_task_diversity   : SHOT diversity on task logits (INVALID for this setting,
                                default False -- this was the v1 bug; only expose for
                                ablation comparison, do not enable in production runs)
        use_task_agreement   : JSD consistency across sub-bags' task routing (PATCH)
        gamma                : weight of the agreement term relative to l_task_ent
    """
    l_class_ent = entropy_loss(class_logits)
    l_class_div = diversity_loss(class_logits) if use_class_diversity else torch.tensor(0.0, device=class_logits.device)
    l_class     = l_class_ent - l_class_div

    l_task_ent  = entropy_loss(task_logits)
    l_task_div  = diversity_loss(task_logits) if use_task_diversity else torch.tensor(0.0, device=task_logits.device)
    l_task_agree = task_agreement_loss(task_logits) if use_task_agreement else torch.tensor(0.0, device=task_logits.device)

    l_task = l_task_ent - l_task_div + gamma * l_task_agree

    total = l_class + alpha * l_task

    log = {
        "loss/class_ent":    l_class_ent.item(),
        "loss/class_div":    l_class_div.item() if use_class_diversity else 0.0,
        "loss/task_ent":     l_task_ent.item(),
        "loss/task_div":     l_task_div.item() if use_task_diversity else 0.0,
        "loss/task_agree":   l_task_agree.item() if use_task_agreement else 0.0,
        "loss/total":        total.item(),
    }
    return total, log


def select_confident_subbags(
    logits:    torch.Tensor,
    top_ratio: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select top-(top_ratio * N) sub-bags with lowest entropy (most confident).
    TPT-style confidence selection.

    Args:
        logits    : [N, C]
        top_ratio : fraction to keep

    Returns:
        selected_logits : [K, C]
        selected_idx    : [K]
    """
    with torch.no_grad():
        ent = -(F.softmax(logits, dim=1).clamp(min=1e-8) *
                F.log_softmax(logits, dim=1)).sum(dim=1)
    k   = max(1, int(ent.size(0) * top_ratio))
    idx = torch.argsort(ent)[:k]
    return logits[idx], idx


def select_confident_subbags_intersection(
    class_logits: torch.Tensor,
    task_logits:  torch.Tensor,
    top_ratio:    float = 0.5,
) -> torch.Tensor:
    """
    PATCH: EATA-style stricter selection -- keep only sub-bags that are
    confident on BOTH class and task logits (intersection), instead of
    the union used in v1 (which lets a task-confident-but-wrong sub-bag
    leak into the gradient via the class branch, and vice versa).

    Falls back to the union's top-1 if intersection is empty, so at least
    one sub-bag is always kept.

    Returns:
        selected_idx : [K] (K may be < top_ratio*N)
    """
    _, idx_class = select_confident_subbags(class_logits, top_ratio)
    _, idx_task  = select_confident_subbags(task_logits,  top_ratio)
    set_class = set(idx_class.tolist())
    set_task  = set(idx_task.tolist())
    inter = sorted(set_class & set_task)
    if len(inter) == 0:
        # fallback: single most-confident sub-bag by class entropy
        inter = [int(idx_class[0].item())]
    return torch.tensor(inter, dtype=torch.long, device=class_logits.device)


def l2_anchor_loss(
    params:   List[torch.Tensor],
    params_0: List[torch.Tensor],
) -> torch.Tensor:
    """
    L2 regularization toward initial parameter values (EATA simplified).

    params   : current LN params (requires_grad=True)
    params_0 : initial LN params (detached)
    """
    return sum(((p - p0) ** 2).sum() for p, p0 in zip(params, params_0))