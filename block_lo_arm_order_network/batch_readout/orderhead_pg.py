"""Stage-3 policy-gradient helpers for the internal OrderHead (L0DynamicGBeta).

Batch-level: one sampled order per step -> one logp. Groups only reduce baseline
variance (see spec). No CDL anywhere in this module.
"""
from __future__ import annotations

from analyses.v3_group_credit import per_sample_loss, group_rewards


def gbeta_scores_with_grad(gbeta_model, B_det):
    """Grad-enabled L0DynamicGBeta forward on ONE batch-mean B.

    B_det: (1, H=8, 65, 65) detached tensor built identically to the frozen
    provider's B. Returns the model-frame score vector (64,) with grad on gβ
    params (head dropout OFF so it matches the deployed argsort path).
    """
    scores, _aux = gbeta_model(B_det, apply_head_dropout=False)  # (1, 64)
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


__all__ = ["gbeta_scores_with_grad", "batch_advantage"]
