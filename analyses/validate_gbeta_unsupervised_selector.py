"""Validate g_β-based unsupervised head selector (edge-direction consistency).

For each head, compute:
  1. S_forward(B^h, g_β(B^h)) = Σ |B_ij| · 1[r_σ(j) > r_σ(i)] / Σ |B_ij|
     → edge-direction consistency: do B edges point "forward" in g_β's order?
     → FULLY unsupervised: no L2R, no CDL teacher, no ground-truth order
  2. |g_β τ| = |τ(g_β(B^h), L2R)|
     → oracle: how L2R-aligned is g_β's readout of this head?
     → uses L2R label (for validation only, NOT for selection)

Then report:
  - ρ(S_forward, |g_β τ|) across all 32 heads
  - Top-1 agreement: argmax S_forward == argmax |g_β τ|?
  - Top-3 overlap

Usage:
  python analyses/validate_gbeta_unsupervised_selector.py [--device cuda:0]
"""
from __future__ import annotations

import argparse, os, sys, time
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "block_lo_arm_order_network"))

import torch
from batch_readout.model import NodewiseReadout
from per_head_order_scan import extract_per_head_and_heavy_A
from neural_readout.extract_b import _load_model_and_chunks

N_BLOCKS = 64
L2R = np.arange(N_BLOCKS)


# ── Metrics ──────────────────────────────────────────────────────────────────
def kendall_tau(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n = len(a); conc = disc = 0
    for i in range(n):
        for j in range(i+1, n):
            da, db = a[i]-a[j], b[i]-b[j]
            if da==0 and db==0: continue
            if da*db>0: conc+=1
            elif da*db<0: disc+=1
    d=conc+disc; return (conc-disc)/d if d>0 else 0.0


def _A_to_B(A):
    B = np.asarray(A.T, dtype=np.float64).copy(); np.fill_diagonal(B, 0.0); return B


def edge_direction_consistency(B, sigma):
    """S_forward: fraction of |B| mass on edges that point "forward" in order σ.

    S_forward = Σ |B_ij| · 1[rank_σ(j) > rank_σ(i)] / Σ |B_ij|

    sigma: (N,) physical order (token indices in reveal order)
    rank_σ[token_idx] = position in reveal sequence (0 = earliest)

    High S_forward → B edges predominantly point from earlier to later tokens
    in g_β's output order — the graph structure is consistent with the readout order.
    """
    B = np.asarray(B, dtype=np.float64)
    N = B.shape[0]
    # Build rank map: rank[physical_token] = position in sigma
    rank = np.zeros(N, dtype=np.int32)
    for pos, token in enumerate(sigma):
        rank[token] = pos

    total_mass = 0.0
    forward_mass = 0.0
    for i in range(N):
        for j in range(N):
            w = abs(B[i, j])
            if w < 1e-15:
                continue
            total_mass += w
            if rank[j] > rank[i]:  # j comes AFTER i in sigma → forward edge
                forward_mass += w

    return float(forward_mass / total_mass) if total_mass > 1e-15 else 0.5


# ── g_β loader ───────────────────────────────────────────────────────────────
def load_g_beta(path, device="cpu"):
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = state["config"]
    if cfg["model_name"] == "nodewise":
        model = NodewiseReadout(N=64, d_model=64, n_layers=2, n_heads=4)
    else:
        raise ValueError(f"unknown model {cfg['model_name']!r}")
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def gbeta_predict_batch(model, B_batch, device, batch_size=64):
    """B_batch: (M, 64, 64) → orders (M, 64), scores (M, 64).

    Convention: sigma = argsort(-z) = descending score → earliest first.
    """
    model.eval()
    all_orders = []
    all_scores = []
    for start in range(0, len(B_batch), batch_size):
        batch = torch.from_numpy(B_batch[start:start+batch_size]).float().to(device)
        z = model(batch)
        orders = torch.argsort(-z, dim=-1).cpu().numpy()
        all_orders.append(orders)
        all_scores.append(z.cpu().numpy())
    return np.concatenate(all_orders), np.concatenate(all_scores)


# ── Main validation ──────────────────────────────────────────────────────────
def validate(checkpoint_path, gbeta_path, device="cuda:0", M=200, seed=42):
    print(f"Loading checkpoint: {checkpoint_path}")
    t0 = time.time()
    model, chunks, clean_perm, dev, ci = _load_model_and_chunks(
        ckpt_path=checkpoint_path, M=M, seed=seed, device=device, split="train")

    A_lh, _ = extract_per_head_and_heavy_A(
        model, chunks, clean_perm, dev, seed=seed, fwd_batch=64, none_mode="b0")
    n_samples, L, H, N, _ = A_lh.shape
    print(f"Extracted {n_samples} graphs, {L}x{H}={L*H} heads in {time.time()-t0:.1f}s")
    del model
    torch.cuda.empty_cache()

    # Load g_β
    print(f"Loading g_β: {gbeta_path}")
    gbeta = load_g_beta(gbeta_path, device)

    # For each head: compute g_β orders on B_mean, then S_forward and |g_β τ|
    results = []
    for l in range(L):
        for h in range(H):
            # Per-sample B matrices for this head
            B_samples = np.array([_A_to_B(A_lh[i, l, h]) for i in range(n_samples)])
            B_mean = B_samples.mean(axis=0)

            # g_β readout on batch-mean B (single input → need batch dim)
            B_input = B_mean[np.newaxis, :, :]  # (1, 64, 64)
            orders, scores = gbeta_predict_batch(gbeta, B_input, device)
            sigma = orders[0]  # (64,)

            # Oracle: |τ(g_β(B), L2R)|
            tau_gbeta = kendall_tau(sigma, L2R)
            abs_tau = abs(tau_gbeta)

            # Unsupervised: S_forward (edge-direction consistency)
            s_forward = edge_direction_consistency(B_mean, sigma)

            # Also compute on per-sample basis for robustness
            s_forward_per_sample = []
            tau_per_sample = []
            for i in range(min(n_samples, 50)):  # 50 samples enough
                B_i = B_samples[i]
                sigma_i, _ = gbeta_predict_batch(gbeta, B_i[np.newaxis], device)
                sigma_i = sigma_i[0]
                s_forward_per_sample.append(edge_direction_consistency(B_i, sigma_i))
                tau_per_sample.append(abs(kendall_tau(sigma_i, L2R)))

            results.append({
                "head": f"L{l}H{h}", "l": l, "h": h,
                "abs_tau_gbeta": abs_tau,
                "tau_gbeta": tau_gbeta,
                "s_forward": s_forward,
                "s_forward_mean50": float(np.mean(s_forward_per_sample)),
                "s_forward_std50": float(np.std(s_forward_per_sample)),
                "abs_tau_mean50": float(np.mean(tau_per_sample)),
                "abs_tau_std50": float(np.std(tau_per_sample)),
            })

    del gbeta
    torch.cuda.empty_cache()

    # ── Rankings ──────────────────────────────────────────────────────────
    oracle_rank = sorted(results, key=lambda x: -x["abs_tau_gbeta"])
    sfwd_rank = sorted(results, key=lambda x: -x["s_forward"])
    sfwd50_rank = sorted(results, key=lambda x: -x["s_forward_mean50"])

    # ── Print table ───────────────────────────────────────────────────────
    print(f"\n{'Head':>8s}  {'|g_β τ|':>10s}  {'S_fwd(B_mean)':>14s}  {'S_fwd(μ±σ 50)':>18s}  {'|τ|(μ±σ 50)':>16s}")
    print("-" * 85)
    for r in sorted(results, key=lambda x: -x["abs_tau_gbeta"]):
        marker = " ★" if r["abs_tau_gbeta"] > 0.3 else ""
        print(f"  {r['head']}  {r['abs_tau_gbeta']:>10.4f}  {r['s_forward']:>14.4f}  "
              f"{r['s_forward_mean50']:>8.4f}±{r['s_forward_std50']:.4f}  "
              f"{r['abs_tau_mean50']:>8.4f}±{r['abs_tau_std50']:.4f}{marker}")

    # ── Top-k concordance ────────────────────────────────────────────────
    def topk(ranked, k=3):
        return set(r["head"] for r in ranked[:k])

    oracle_top1 = oracle_rank[0]["head"]
    oracle_top3 = topk(oracle_rank, 3)
    sfwd_top1 = sfwd_rank[0]["head"]
    sfwd_top3 = topk(sfwd_rank, 3)
    sfwd50_top1 = sfwd50_rank[0]["head"]
    sfwd50_top3 = topk(sfwd50_rank, 3)

    print(f"\nOracle top-1 (|g_β τ|): {oracle_top1}  τ={oracle_rank[0]['abs_tau_gbeta']:.4f}")
    print(f"S_fwd  top-1:            {sfwd_top1}  sf={sfwd_rank[0]['s_forward']:.4f}  "
          f"{'✓' if sfwd_top1==oracle_top1 else '✗'}")
    print(f"S_fwd50 top-1:          {sfwd50_top1}  sf50={sfwd50_rank[0]['s_forward_mean50']:.4f}  "
          f"{'✓' if sfwd50_top1==oracle_top1 else '✗'}")

    print(f"\nOracle top-3: {oracle_top3}")
    print(f"S_fwd  top-3: {sfwd_top3}  overlap={len(oracle_top3 & sfwd_top3)}/3")
    print(f"S_fwd50 top-3: {sfwd50_top3}  overlap={len(oracle_top3 & sfwd50_top3)}/3")

    # ── Spearman ρ ───────────────────────────────────────────────────────
    from scipy.stats import spearmanr
    oracle_arr = np.array([r["abs_tau_gbeta"] for r in results])
    sfwd_arr = np.array([r["s_forward"] for r in results])
    sfwd50_arr = np.array([r["s_forward_mean50"] for r in results])
    tau50_arr = np.array([r["abs_tau_mean50"] for r in results])

    rho_sf, p_sf = spearmanr(oracle_arr, sfwd_arr)
    rho_sf50, p_sf50 = spearmanr(oracle_arr, sfwd50_arr)
    # Sanity: per-sample S_fwd vs per-sample |τ|
    rho_cross, p_cross = spearmanr(sfwd50_arr, tau50_arr)

    print(f"\nSpearman ρ:")
    print(f"  ρ(S_fwd(B_mean), |g_β τ|)        = {rho_sf:+.4f}  p={p_sf:.4f}")
    print(f"  ρ(S_fwd(μ 50), |g_β τ|)          = {rho_sf50:+.4f}  p={p_sf50:.4f}")
    print(f"  ρ(S_fwd(μ 50), |τ|(μ 50))       = {rho_cross:+.4f}  p={p_cross:.4f}  [per-sample sanity]")

    return {
        "checkpoint": checkpoint_path,
        "g_beta": gbeta_path,
        "M": M,
        "oracle_top1": oracle_top1,
        "sfwd_top1": sfwd_top1,
        "sfwd_top1_correct": sfwd_top1 == oracle_top1,
        "sfwd50_top1": sfwd50_top1,
        "sfwd50_top1_correct": sfwd50_top1 == oracle_top1,
        "oracle_top3": sorted(oracle_top3),
        "sfwd_top3": sorted(sfwd_top3),
        "sfwd_top3_overlap": len(oracle_top3 & sfwd_top3),
        "sfwd50_top3": sorted(sfwd50_top3),
        "sfwd50_top3_overlap": len(oracle_top3 & sfwd50_top3),
        "rho_sf": rho_sf, "p_sf": p_sf,
        "rho_sf50": rho_sf50, "p_sf50": p_sf50,
        "rho_cross": rho_cross, "p_cross": p_cross,
        "heads": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ckpt", default=None,
                        help="model checkpoint (default: seed2 @ 10k random baseline)")
    parser.add_argument("--g-beta", default=None,
                        help="g_β checkpoint path")
    parser.add_argument("--M", type=int, default=200,
                        help="number of batch windows for attention extraction")
    parser.add_argument("--seed", type=int, default=42,
                        help="extraction seed")
    args = parser.parse_args()

    device = args.device

    # Default paths
    ckpt_path = args.ckpt or str(REPO /
        "block_lo_arm_order_network/probe_results/"
        "random_baseline_continuous_jun08_seed2/ckpt_step10000.pt")

    gbeta_path = args.g_beta or str(REPO /
        "block_lo_arm_order_network/batch_readout/logs/"
        "phase33_gbeta_seed2_from10k_l0h2/"
        "random_baseline_continuous_jun08_seed2/full/g_beta_best.pt")

    if not os.path.exists(ckpt_path):
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        return 1
    if not os.path.exists(gbeta_path):
        print(f"ERROR: g_β not found: {gbeta_path}")
        return 1

    result = validate(ckpt_path, gbeta_path, device=device, M=args.M, seed=args.seed)

    # Save
    out_dir = REPO / "analyses" / "attention_diagnostic_20260609"
    out_dir.mkdir(exist_ok=True)
    import json
    out_path = out_dir / "gbeta_unsupervised_selector.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else str(x))
    print(f"\nSaved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
