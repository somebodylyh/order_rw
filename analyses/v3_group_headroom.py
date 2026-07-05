"""Group-level oracle headroom: does a SINGLE shared reveal order have room to
beat physical L2R when it must serve m samples at once?

P7 established per-sample oracle headroom = +0.19 nat (hill-climbed order beats
true L2R), but the good orders are sample-specific (cross-sample transfer -0.13).
The V3 main arm uses m=16 GROUP orders (one σ per 16 samples), so its reachable
room is bounded by the best SHARED order — which can be far below the per-sample
oracle. This module measures headroom[m] on a frozen backbone BEFORE spending the
m=16 co-adapt sweep. If headroom[16] ≈ 0 the group arm is structurally hopeless
and we pivot (per-sample m=1, or document the group ceiling).

Frame-correct: all NLLs via order_nll(σ_PHYSICAL_block, clean_perm). Baseline =
true physical L2R = arange(N). Reuses the p6 hill-climb swap move, group-mean obj.
"""

from __future__ import annotations

import sys
import pathlib

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
BLOCK_ROOT = ROOT / "block_lo_arm_order_network"
for path in (ROOT, BLOCK_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyses.p5_utility_controller import order_nll, N  # noqa: E402


@torch.no_grad()
def group_order_nll(model, group_rows, sigma_phys, clean_perm, dev):
    """Mean teacher-forced NLL of the group's rows under ONE shared physical-block
    order ``sigma_phys``."""
    nlls = [
        order_nll(model, group_rows[i:i + 1], sigma_phys, clean_perm, dev)
        for i in range(group_rows.shape[0])
    ]
    return float(np.mean(nlls))


@torch.no_grad()
def group_hill_climb(model, group_rows, clean_perm, dev, n_steps=300, seed=0):
    """Random-swap hill climb over ONE shared physical-block order minimising the
    group-mean NLL. Starts from physical L2R (arange).

    Returns ``(best_order, best_group_nll, l2r_group_nll)``.
    """
    rng = np.random.default_rng(seed)
    best = np.arange(N, dtype=np.int64)
    l2r_nll = group_order_nll(model, group_rows, best, clean_perm, dev)
    best_nll = l2r_nll
    for _ in range(n_steps):
        cand = best.copy()
        i, j = rng.integers(0, N, size=2)
        cand[i], cand[j] = cand[j], cand[i]
        nll = group_order_nll(model, group_rows, cand, clean_perm, dev)
        if nll < best_nll:
            best, best_nll = cand, nll
    return best, float(best_nll), float(l2r_nll)


@torch.no_grad()
def group_headroom_sweep(model, held, clean_perm, dev, *, m_list=(1, 4, 16, 64),
                         n_steps=300, seed=0):
    """For each group size m, partition ``held`` (M, T) into contiguous groups of
    m, hill-climb one shared order per group, and aggregate the L2R headroom.

    Returns ``{m: {headroom, group_oracle_val, l2r_val, n_groups}}`` where
    ``headroom = l2r_val - group_oracle_val`` (mean over groups). Positive
    headroom at m means a single shared order can still beat L2R for m samples.
    """
    M = held.shape[0]
    out = {}
    for m in m_list:
        if M % m != 0:
            raise ValueError(f"held size {M} not divisible by m={m}")
        n_groups = M // m
        oracle_vals, l2r_vals = [], []
        for g in range(n_groups):
            rows = held[g * m:(g + 1) * m]
            _best, best_nll, l2r_nll = group_hill_climb(
                model, rows, clean_perm, dev, n_steps=n_steps, seed=seed + g
            )
            oracle_vals.append(best_nll)
            l2r_vals.append(l2r_nll)
        oracle_val = float(np.mean(oracle_vals))
        l2r_val = float(np.mean(l2r_vals))
        out[m] = {
            "headroom": l2r_val - oracle_val,
            "group_oracle_val": oracle_val,
            "l2r_val": l2r_val,
            "n_groups": int(n_groups),
        }
    return out


__all__ = ["group_order_nll", "group_hill_climb", "group_headroom_sweep"]
