#!/usr/bin/env python3
r"""Fine calibration of pos_tau for the position-only entropy-matched attribution control.

Stricter than calibrate_position_only.py: instead of matching the cached K=128 reference
(avg_step_entropy=0.0465), this RECOMPUTES source_start's avg_step_entropy at K=512 over
multiple seeds (removing the single-seed/K=128 estimator noise), then fine-sweeps pos_tau
(K=512, same seeds) to match that target within +/-0.001. Writes the chosen pos_tau and
both samplers' diagnostics to calibration_position_only.json for the launcher to consume.

The ONLY difference between the two samplers is the score source:
    source_start : attention-B + distilled MLP (+ readiness anchor at t=0)
    position-only: logits[v] = -v / pos_tau   (pure positional prior, no B, no MLP)
so a clean entropy + forwardness match isolates "attention-B contribution" from "L2R prior".

Run (CPU, ~1-2 min):
    python block_lo_arm_order_network/scripts/calibrate_position_only_fine.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kendalltau

_HERE = Path(__file__).resolve().parent.parent          # block_lo_arm_order_network/
_REPO = _HERE.parent
sys.path.insert(0, str(_HERE))

from directed_graph_policy import build_directed_graph                       # noqa: E402
import attn_order_mlp_policy as P                                            # noqa: E402

# --- fixed substrate / policy config (must mirror source_start exactly) ---
V3_A = _HERE / "probe_results/clean_method_graph_rw_v3_from20k/A_global_eval.npy"
MLP_PT = _REPO / "probe_results/attention_order_mlp/phase1_text_mlp_ckpt20k.pt"
OUT_JSON = _HERE / "scripts/calibration_position_only.json"

K = 512                       # samples per seed (vs 128 for the cached reference)
SEEDS = [20000, 20001, 20002]
TOP_K = 4
SS_TAU, SS_SRC_RHO = 0.5, 0.3   # source_start sampler config
TOL = 0.001                   # acceptance band around the recomputed target
DEVICE = torch.device("cpu")


def _forwardness(orders_np):
    """Mean Kendall tau vs raster (L2R), plus start-node mode and uniqueness."""
    N = orders_np.shape[1]
    raster = np.arange(N)
    taus = np.array([kendalltau(o, raster).correlation for o in orders_np])
    sc = np.bincount(orders_np[:, 0], minlength=N)
    return dict(
        tau_vs_l2r=float(np.nanmean(taus)),
        abs_tau=float(np.nanmean(np.abs(taus))),
        start_mode=int(sc.argmax()),
        start_mode_frac=float(sc.max()) / orders_np.shape[0],
        unique=int(len({tuple(r.tolist()) for r in orders_np})),
    )


def source_start_target(B):
    """Recompute source_start avg_step_entropy at K=512 over SEEDS; also report forwardness."""
    ents, all_orders = [], []
    mlp = P.load_order_mlp(MLP_PT, DEVICE)
    for s in SEEDS:
        orders, ent = P.sample_orders_batched_mlp(
            B, K, mlp, "source_start", base_seed=s, device=DEVICE,
            tau=SS_TAU, top_k=TOP_K, src_rho=SS_SRC_RHO, return_entropy=True,
        )
        ents.append(ent)
        all_orders.append(orders.cpu().numpy())
    o = np.concatenate(all_orders, axis=0)
    diag = _forwardness(o)
    diag["avg_step_entropy"] = float(np.mean(ents))
    diag["entropy_per_seed"] = [round(e, 5) for e in ents]
    return diag


def position_entropy(N, pos_tau):
    """avg_step_entropy + forwardness for position-only at pos_tau, K=512 over SEEDS."""
    ents, all_orders = [], []
    for s in SEEDS:
        orders, ent = P.sample_orders_batched_position(
            N, K, s, DEVICE, pos_tau=pos_tau, top_k=TOP_K, return_entropy=True,
        )
        ents.append(ent)
        all_orders.append(orders.cpu().numpy())
    o = np.concatenate(all_orders, axis=0)
    diag = _forwardness(o)
    diag["avg_step_entropy"] = float(np.mean(ents))
    diag["pos_tau"] = float(pos_tau)
    return diag


def main():
    A = np.load(V3_A).astype(np.float32)
    np.fill_diagonal(A, 0.0)
    B = build_directed_graph(A)
    N = B.shape[0]

    ss = source_start_target(B)
    target = ss["avg_step_entropy"]
    print(f"source_start (K={K}x{len(SEEDS)} seeds): avg_step_entropy={target:.5f} "
          f"(per-seed {ss['entropy_per_seed']}), tau_vs_l2r={ss['tau_vs_l2r']:.4f}, "
          f"start_mode={ss['start_mode']} (frac {ss['start_mode_frac']:.3f}), unique={ss['unique']}/{K*len(SEEDS)}")

    # coarse grid bracketing (K=128 calib showed 0.20->0.040, 0.21->0.049 for target ~0.047)
    coarse = np.round(np.arange(0.180, 0.241, 0.005), 4)
    print(f"\n{'pos_tau':>8} {'avg_step_ent':>12} {'d_target':>9} {'tau_vs_l2r':>11} {'unique':>8} {'start@0':>8}")
    rows = []
    for t in coarse:
        d = position_entropy(N, float(t))
        rows.append(d)
        print(f"{t:8.4f} {d['avg_step_entropy']:12.5f} {d['avg_step_entropy']-target:+9.5f} "
              f"{d['tau_vs_l2r']:11.4f} {d['unique']:8d} {d['start_mode_frac']:8.3f}")

    # bracket the target and bisection-refine to TOL
    below = [r for r in rows if r["avg_step_entropy"] <= target]
    above = [r for r in rows if r["avg_step_entropy"] >= target]
    lo = max(below, key=lambda r: r["avg_step_entropy"]) if below else min(rows, key=lambda r: r["avg_step_entropy"])
    hi = min(above, key=lambda r: r["avg_step_entropy"]) if above else max(rows, key=lambda r: r["avg_step_entropy"])
    a, b = lo["pos_tau"], hi["pos_tau"]
    best = min(rows, key=lambda r: abs(r["avg_step_entropy"] - target))
    for _ in range(25):
        if abs(best["avg_step_entropy"] - target) <= TOL or (b - a) < 1e-4:
            break
        mid = 0.5 * (a + b)
        dm = position_entropy(N, mid)
        if abs(dm["avg_step_entropy"] - target) < abs(best["avg_step_entropy"] - target):
            best = dm
        if dm["avg_step_entropy"] < target:
            a = mid
        else:
            b = mid

    ok = abs(best["avg_step_entropy"] - target) <= TOL
    print(f"\n=> chosen pos_tau = {best['pos_tau']:.5f}  avg_step_ent={best['avg_step_entropy']:.5f} "
          f"(target {target:.5f}, |d|={abs(best['avg_step_entropy']-target):.5f}, tol {TOL}) "
          f"-> {'WITHIN TOL' if ok else 'OUT OF TOL'}")
    print(f"   forwardness: tau_vs_l2r={best['tau_vs_l2r']:.4f} (ss {ss['tau_vs_l2r']:.4f}), "
          f"start_mode={best['start_mode']} (ss {ss['start_mode']}), "
          f"start@0={best['start_mode_frac']:.3f}, unique={best['unique']}/{K*len(SEEDS)}")

    json.dump(
        dict(pos_tau=round(best["pos_tau"], 5), within_tol=bool(ok), tol=TOL, K=K, seeds=SEEDS,
             top_k=TOP_K, source_start=ss, position_only=best,
             note="logits[v]=-v/pos_tau; entropy-matched to recomputed source_start K=512 target."),
        open(OUT_JSON, "w"), indent=2)
    print(f"\nwrote {OUT_JSON}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
