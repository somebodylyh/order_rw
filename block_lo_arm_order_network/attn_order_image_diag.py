"""Per-refresh diagnostics for the IMAGE alternating arm (orientation='original', no
source_start, no Manhattan distance). Mirrors attn_order_distill.refresh_diagnostics
but reports image structural metrics (locality / B-edge-following) instead of L2R tau.
"""
from __future__ import annotations
import numpy as np
import torch

from attn_order_teacher import rollout_order
from train_attn_order_mlp import locality_stats, top4_follow_and_edge, diversity
import attn_order_mlp_policy as P


def image_refresh_diagnostics(B, mlp, *, tau=0.5, top_k=4, seed=0, K=128, grid=8, device="cpu"):
    """Snapshot teacher + distilled-beta order structure on the current B (attention-only).

    Returns student rollout metrics (p_le1/p_le2/top4_follow/B_edge_ratio/entropy/unique)
    plus the matched teacher metrics (teacher_p_le1/teacher_top4_follow), so a refresh row
    shows whether the loop is sharpening attention-derived structure.
    """
    B = np.ascontiguousarray(np.asarray(B, dtype=np.float64))

    teach = np.stack([rollout_order(B, tau_T=tau, seed=int(seed) + s, mode="C-D+L", standardize=True)
                      for s in range(K)])
    t_loc = locality_stats(teach, grid=grid)
    t_t4, t_er = top4_follow_and_edge(teach, B)

    orders, ent = P.sample_orders_batched_mlp(
        B, K, mlp.to(device), "original", base_seed=int(seed), device=torch.device(device),
        tau=tau, top_k=top_k, return_entropy=True,
    )
    o = orders.cpu().numpy()
    s_loc = locality_stats(o, grid=grid)
    s_t4, s_er = top4_follow_and_edge(o, B)
    return dict(
        p_le1=round(s_loc["p_le1"], 4), p_le2=round(s_loc["p_le2"], 4),
        mean_manh=round(s_loc["mean_manh"], 4), top4_follow=round(s_t4, 4),
        B_edge_ratio=round(s_er, 4), rollout_entropy=round(float(ent), 4),
        rollout_unique=int(diversity(o)),
        teacher_p_le1=round(t_loc["p_le1"], 4), teacher_top4_follow=round(t_t4, 4),
        teacher_B_edge_ratio=round(t_er, 4),
    )
