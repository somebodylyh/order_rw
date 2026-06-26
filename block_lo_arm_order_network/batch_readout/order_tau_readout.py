"""Per-(layer,head) order-τ readout from model-frame strict65 B graphs.

Label-free: input B is model-frame; τ_vs_l2r is a posthoc comparison of the
rolled-out σ_model against np.arange, valid only because the trajectory logger
uses identity probe_orders (model-frame == physical L2R reference). No inv_perm
is used anywhere here.
"""
from __future__ import annotations

import numpy as np

from none_separated_block_graph import (
    rollout_by_method,
    discovery_metrics,
    classify_gate_status,
)

DEFAULT_METHODS = ("C-D+L", "L")


def per_head_tau(B65: np.ndarray, method: str) -> dict:
    order = rollout_by_method(np.asarray(B65, dtype=np.float64), method)
    m = discovery_metrics(order)
    return {
        "tau_vs_l2r": float(m["tau_vs_l2r"]),
        "phys0_rank": int(m["phys0_rank"]),
        "prefix8_overlap": int(m["prefix8_overlap"]),
        "first_is_phys0": bool(m["first_is_phys0"]),
    }


def layer_head_tau_table(B_lhn: np.ndarray, methods=DEFAULT_METHODS) -> dict:
    B_lhn = np.asarray(B_lhn, dtype=np.float64)
    L, H = B_lhn.shape[:2]
    Mn = len(methods)
    tau = np.full((L, H, Mn), np.nan, dtype=np.float64)
    rank = np.full((L, H, Mn), -1, dtype=np.int64)
    pre8 = np.zeros((L, H, Mn), dtype=np.int64)
    for li in range(L):
        for hi in range(H):
            for mi, method in enumerate(methods):
                info = per_head_tau(B_lhn[li, hi], method)
                tau[li, hi, mi] = info["tau_vs_l2r"]
                rank[li, hi, mi] = info["phys0_rank"]
                pre8[li, hi, mi] = info["prefix8_overlap"]
    return {"tau": tau, "phys0_rank": rank, "prefix8": pre8, "methods": list(methods)}


def derived_views(table: dict, strong_tau: float = 0.7) -> dict:
    tau = table["tau"]  # (L,H,M)
    L, H, Mn = tau.shape
    abs_tau = np.abs(tau)
    max_signed = tau.max(axis=1)            # (L,M)
    best_signed = tau.argmax(axis=1)        # (L,M)
    max_abs = abs_tau.max(axis=1)           # (L,M)
    best_abs = abs_tau.argmax(axis=1)       # (L,M)
    strong = (abs_tau >= strong_tau).sum(axis=1)  # (L,M)
    glob = np.array([np.unravel_index(abs_tau[:, :, m].argmax(), (L, H))
                     for m in range(Mn)])   # (M,2) -> (layer,head)
    return {
        "best_head_per_layer": {"signed": best_signed, "abs": best_abs},
        "best_head_global": glob,
        "max_abs_tau_per_layer": max_abs,
        "max_signed_tau_per_layer": max_signed,
        "strong_pass_count_per_layer": strong,
    }
