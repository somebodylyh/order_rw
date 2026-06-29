#!/usr/bin/env python3
r"""Calibrate pos_tau for the position-only entropy-matched attribution control.

Find the pos_tau whose position-only sampler avg_step_entropy matches source_start's
reported value (0.0465), and verify it lands at a comparable forwardness (tau_vs_l2r ~0.99).
This is the §6 control isolating "attention-B contribution" from "L2R/position prior".

source_start reference (from cont_MLP_CDL_source_start/report.json orientation diag):
    avg_step_entropy = 0.0465,  tau_vs_l2r = 0.9924,  unique = 35/128
"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kendalltau

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))
from attn_order_mlp_policy import sample_orders_batched_position  # noqa: E402

N = 64
K = 128
TOP_K = 4
TARGET_ENT = 0.0465       # source_start avg_step_entropy
REF_TAU = 0.9924          # source_start tau_vs_l2r
REF_UNIQUE = 35
raster = np.arange(N)


def measure(pos_tau, seed=20000):
    orders, ent = sample_orders_batched_position(
        N, K, seed, torch.device("cpu"), pos_tau=pos_tau, top_k=TOP_K, return_entropy=True
    )
    o = orders.cpu().numpy()
    taus = np.array([kendalltau(row, raster).correlation for row in o])
    uniq = len({tuple(r.tolist()) for r in o})
    start_frac = np.bincount(o[:, 0], minlength=N).max() / K
    return dict(pos_tau=pos_tau, ent=float(ent), tau=float(np.nanmean(taus)),
                abs_tau=float(np.nanmean(np.abs(taus))), unique=uniq, start0_frac=float(start_frac))


def main():
    grid = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40, 0.50]
    rows = [measure(t) for t in grid]
    print(f"{'pos_tau':>8} {'avg_step_ent':>12} {'tau_vs_l2r':>11} {'abs_tau':>8} {'unique':>7} {'start@0':>8}")
    for r in rows:
        print(f"{r['pos_tau']:8.3f} {r['ent']:12.4f} {r['tau']:11.4f} {r['abs_tau']:8.4f} "
              f"{r['unique']:7d} {r['start0_frac']:8.3f}")
    best = min(rows, key=lambda r: abs(r["ent"] - TARGET_ENT))
    print(f"\nsource_start ref:   avg_step_ent={TARGET_ENT}  tau_vs_l2r={REF_TAU}  unique={REF_UNIQUE}/128")
    print(f"closest pos_tau   = {best['pos_tau']:.3f}  -> avg_step_ent={best['ent']:.4f} "
          f"(target {TARGET_ENT}), tau_vs_l2r={best['tau']:.4f}, unique={best['unique']}/128")
    # quick local refine around best
    lo, hi = best["pos_tau"] * 0.7, best["pos_tau"] * 1.4
    fine = [measure(t) for t in np.linspace(lo, hi, 13)]
    bf = min(fine, key=lambda r: abs(r["ent"] - TARGET_ENT))
    print(f"refined pos_tau   = {bf['pos_tau']:.4f}  -> avg_step_ent={bf['ent']:.4f}, "
          f"tau_vs_l2r={bf['tau']:.4f}, abs_tau={bf['abs_tau']:.4f}, unique={bf['unique']}/128, "
          f"start@0={bf['start0_frac']:.3f}")


if __name__ == "__main__":
    main()
