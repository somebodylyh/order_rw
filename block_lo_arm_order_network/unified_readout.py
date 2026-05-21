#!/usr/bin/env python3
"""Unified continuous readout family. One scoring function instantiates
readiness-Graph-RW, Bcov-proximity, distance-only, and random by params alone.

score_t(v) = beta_sup*Csup(v) - beta_dep*Dfut(v) + rho*r(v)
             + gamma_B*localB(v) - gamma_d*dist(v)
Each term is min-max normalized over the current candidate set so weights are
commensurable across graphs of different scale. fallback_mix blends the final
policy with uniform. Orders are PHYSICAL frame (remap to model frame downstream).
"""
import numpy as np

PARAM_KEYS = ["beta_sup", "beta_dep", "rho", "gamma_B", "gamma_d", "tau", "fallback_mix"]
PARAM_BOUNDS = {
    "beta_sup": (0.0, 2.0), "beta_dep": (0.0, 2.0), "rho": (0.0, 2.0),
    "gamma_B": (0.0, 2.0), "gamma_d": (0.0, 2.0), "tau": (0.03, 0.5),
    "fallback_mix": (0.0, 1.0),
}


def _coords(N, topology, grid):
    if topology == "grid2d":
        idx = np.arange(N)
        return np.stack([idx // grid, idx % grid], 1).astype(float)
    return np.arange(N).reshape(N, 1).astype(float)


def _minmax(x):
    x = np.asarray(x, float)
    r = x.ptp()
    return (x - x.min()) / (r + 1e-12) if r > 0 else np.zeros_like(x)


def sample_order(B, source, coords, w, rng, has_grid):
    """One order (length N) in physical frame from params w (dict)."""
    N = B.shape[0]
    U = list(range(N)); S = []; last = -1; order = []
    for _ in range(N):
        Uarr = np.array(U)
        Csup = _minmax(B[S].sum(0)[Uarr]) if S else np.zeros(len(U))
        Dfut = _minmax(B[Uarr].sum(0)[Uarr]) if len(U) > 1 else np.zeros(len(U))
        r = _minmax(source[Uarr])
        localB = _minmax(B[last, Uarr]) if last >= 0 else np.zeros(len(U))
        if has_grid and last >= 0:
            d = np.abs(coords[Uarr] - coords[last]).sum(1)
            dist = _minmax(d)
        else:
            dist = np.zeros(len(U))
        score = (w["beta_sup"]*Csup - w["beta_dep"]*Dfut + w["rho"]*r
                 + w["gamma_B"]*localB - w["gamma_d"]*dist)
        score = score - score.max()
        p = np.exp(score / max(w["tau"], 1e-3)); p = p / p.sum()
        p = (1.0 - w["fallback_mix"]) * p + w["fallback_mix"] * (1.0 / len(U))
        p = p / p.sum()
        ci = int(rng.choice(len(U), p=p)); v = U[ci]
        order.append(v); S.append(v); last = v; U.pop(ci)
    return np.array(order, dtype=np.int64)


def sample_orders_batch(B, source, coords, w, k, seed, has_grid):
    rng = np.random.default_rng(seed)
    return np.stack([sample_order(B, source, coords, w, rng, has_grid) for _ in range(k)])


def clip_params(w):
    return {k: float(np.clip(w[k], *PARAM_BOUNDS[k])) for k in PARAM_KEYS}


def make_coords(N, topology, grid):
    return _coords(N, topology, grid), (topology == "grid2d")
