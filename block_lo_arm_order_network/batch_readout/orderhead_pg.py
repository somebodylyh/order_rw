"""Stage-3 policy-gradient helpers for the internal OrderHead (L0DynamicGBeta).

Batch-level: one sampled order per step -> one logp. Groups only reduce baseline
variance (see spec). No CDL anywhere in this module.
"""
from __future__ import annotations

from analyses.v3_group_credit import per_sample_loss, group_rewards


def gbeta_scores_with_grad(gbeta_model, B_det):
    """Grad-enabled gβ forward on ONE detached B, returning model-frame scores.

    Model-agnostic across the two deployed readouts:
      - NodewiseReadout (single-head L1H7 path): ``model(B)`` -> scores (1, 64),
        B_det is (1, 64, 64), built like ``FrozenBetaHook.step``.
      - L0DynamicGBeta (multi-head path): ``model(B, apply_head_dropout=False)``
        -> (scores, aux), B_det is (1, 8, 65, 65).

    Returns the model-frame score vector (64,) with grad on gβ params.
    """
    import inspect
    dev = next(gbeta_model.parameters()).device
    B_det = B_det.to(dev)                         # align B to the gβ's device
    if "apply_head_dropout" in inspect.signature(gbeta_model.forward).parameters:
        out = gbeta_model(B_det, apply_head_dropout=False)  # L0DynamicGBeta
    else:
        out = gbeta_model(B_det)                  # NodewiseReadout (no such kwarg)
    scores = out[0] if isinstance(out, tuple) else out  # (1, 64)
    return scores[0]


def batch_advantage(token_losses, groups, ema, adv_clip):
    """Per-sample CE -> per-group reward + EMA baseline -> scalar advantage.

    Returns (A_batch scalar detached+clamped, ell_i (Bs,)). Lower loss -> higher
    advantage. With a single batch-level order there is one action, so group
    rewards are aggregated to a scalar (baseline variance reduction only).
    """
    ell_i = per_sample_loss(token_losses.detach())      # (Bs,)
    ell_g = group_rewards(ell_i, groups)                # (G,)
    baseline = ema.update(ell_g)                        # (G,) detached pre-update
    adv_g = baseline - ell_g                            # (G,)
    A = adv_g.mean().detach().clamp(-adv_clip, adv_clip)
    return A, ell_i


def grpo_advantage(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """GRPO-style within-group normalized advantage (detached).

    Args:
        rewards: (K,) tensor of scalar rewards. Higher = better (use -NLL).
        eps: small constant to prevent division by zero when std ≈ 0.

    Returns:
        (K,) detached tensor. mean ≈ 0, std ≈ 1 for non-constant rewards.
        Zero-variance rewards produce all-zeros (no signal).

    Advantage sign convention:
        A_k > 0 → reward_k is above the group mean → increase P(sigma_k).
        L_GRPO = -mean_k(stopgrad(A_k) * log P(sigma_k)).
    """
    import torch as _torch
    r = rewards.float()
    mean_r = r.mean()
    std_r = r.std()
    if std_r < eps:
        return _torch.zeros_like(r)
    adv = (r - mean_r) / (std_r + eps)
    return adv.detach()


__all__ = ["gbeta_scores_with_grad", "batch_advantage", "grpo_advantage"]
