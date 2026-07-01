"""Group-level mechanism probe with false-positive guards. Fixed held-out grouping.
delta_probe_group = E_g[ nll(group order) - nll(L2R) ]. Guards: real-B vs
shuffle-B vs zero-B, tau_to_l2r, tau_consensus. All orders MODEL-frame -> physical
remap via wrap.inv_perm before order_nll."""

from __future__ import annotations

import sys
import pathlib
from typing import List

import numpy as np
import torch
from scipy.stats import kendalltau

ROOT = pathlib.Path(__file__).resolve().parents[1]
BLOCK_ROOT = ROOT / "block_lo_arm_order_network"
for path in (ROOT, BLOCK_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyses.v3_group_credit import group_ids_for
from analyses.p5_utility_controller import order_nll, N
from batch_readout.hook_order_provider import random_probe_token_orders

L2R = np.arange(N, dtype=np.int64)


def _delta_for_B_variant(wrap, held_idx, groups, dev, variant, seed):
    """Mean over groups of [nll(group order from variant-B) - nll(L2R)].

    Returns ``(delta, orders_model)`` where ``orders_model`` is (G, N) int64.
    """
    probe = random_probe_token_orders(held_idx.shape[0], int(seed), 0, dev)
    A = wrap.extract_B(held_idx, probe).detach()                # (M, N+1, N+1) real B

    if variant == "shuffle":
        perm = torch.randperm(A.shape[0], device=dev)
        A = A[perm]                                              # sample-shuffled B
    elif variant == "zero":
        A = torch.zeros_like(A)

    deltas: List[float] = []
    orders_model: List[np.ndarray] = []
    for g in groups:
        gi = torch.as_tensor(g, dtype=torch.long, device=dev)
        Bg = A.index_select(0, gi)                               # this group's B
        scores_g = wrap.order_head.scores(Bg, per_sample=False)[0]  # (N,) group consensus
        sig_model = torch.argsort(scores_g.detach(), descending=True,
                                  stable=True).cpu().numpy().astype(np.int64)
        orders_model.append(sig_model)
        sig_phys = wrap.inv_perm[sig_model]                      # model -> physical
        row = torch.stack([held_idx[i] for i in g])
        nll_group = order_nll(wrap.backbone, row, sig_phys, wrap.clean_perm, dev)
        nll_l2r = order_nll(wrap.backbone, row, L2R, wrap.clean_perm, dev)
        deltas.append(nll_group - nll_l2r)

    return float(np.mean(deltas)), np.stack(orders_model)


def group_probe(wrap, held_idx, *, m=16, seed=0, device="cpu"):
    """Compute group-level mechanism metrics on a fixed held-out set.

    Parameters
    ----------
    wrap : AOGPTWithOrderHead
        Wrapper holding backbone + OrderHead + clean_perm.
    held_idx : torch.Tensor
        Fixed held-out chunks, shape ``(M, T)``.
    m : int
        Group size (default 16).
    seed : int
        RNG seed for probe orders.
    device : str
        Torch device.

    Returns
    -------
    dict with keys:
        delta_probe_group, tau_to_l2r, tau_consensus,
        delta_real, delta_shuffle, delta_zero, real_beats_controls
    """
    dev = torch.device(device)
    groups = group_ids_for(held_idx.shape[0], m)

    d_real, orders_real = _delta_for_B_variant(wrap, held_idx, groups, dev, "real", seed)
    d_shuf, _ = _delta_for_B_variant(wrap, held_idx, groups, dev, "shuffle", seed + 1)
    d_zero, _ = _delta_for_B_variant(wrap, held_idx, groups, dev, "zero", seed + 2)

    # τ(σ_g, L2R) — how close each group order is to L2R
    taus_l2r = [kendalltau(o, L2R).correlation for o in orders_real]

    # τ_consensus — pairwise τ between group orders (are groups producing DIFFERENT orders?)
    G = len(orders_real)
    taus_cons = []
    for i in range(G):
        for j in range(i + 1, G):
            tc = kendalltau(orders_real[i], orders_real[j]).correlation
            if tc is not None and not np.isnan(tc):
                taus_cons.append(tc)

    return {
        "delta_probe_group": d_real,
        "tau_to_l2r": float(np.nanmean(taus_l2r)) if taus_l2r else float("nan"),
        "tau_consensus": float(np.nanmean(taus_cons)) if taus_cons else float("nan"),
        "delta_real": d_real,
        "delta_shuffle": d_shuf,
        "delta_zero": d_zero,
        "real_beats_controls": bool(d_real < min(d_shuf, d_zero)),
        "groups": int(G),
        "seed": int(seed),
    }


__all__ = ["group_probe"]
