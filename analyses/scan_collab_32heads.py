"""Scan all 32 heads from collaborator's all_head_attn_dataset with CDL teacher.

Dataset: /home/admin/ych/nanogpt-learned-order/data/attn/all_head_attn_dataset/
Model: 4L×8H, d=384, Random-order training, block-permuted wikitext-103
      313 records, attention extracted from step 10k→15k.

For each of the 32 heads:
  1. Aggregate B (64×64 inter-block attention) across all records
  2. Run CDL greedy rollout (C-D+L, C-D, -D only, L only)
  3. Compute tau_vs_L2R, row-concentration, entropy
"""

import sys, os
import numpy as np
import torch
from scipy.stats import kendalltau

# Add project root for CDL imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'block_lo_arm_order_network'))
from attn_order_teacher import rollout_order

DATA_DIR = "/home/admin/ych/nanogpt-learned-order/data/attn/all_head_attn_dataset"
N_BLOCKS = 64
N_LAYERS = 4
N_HEADS = 8
N_SHARDS = 5
L2R = np.arange(N_BLOCKS, dtype=np.int64)


def load_all_data():
    """Load all 5 shards, return stacked attention [n_records, layers, heads, 64, 64]."""
    pieces = []
    for i in range(N_SHARDS):
        p = f"{DATA_DIR}/all_head_attn_shard_{i:05d}.pt"
        d = torch.load(p, map_location='cpu', weights_only=False)
        pieces.append(d['attention'].numpy().astype(np.float64))  # [R, L, H, 64, 64]
    attn = np.concatenate(pieces, axis=0)  # [313, 4, 8, 64, 64]
    print(f"Loaded {attn.shape[0]} records, {attn.shape[1]} layers × {attn.shape[2]} heads")
    return attn


def compute_row_conc(B):
    """Row concentration = mean negentropy of row distributions."""
    B = np.asarray(B, dtype=np.float64)
    eps = 1e-12
    row_sums = B.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, eps, row_sums)
    P = B / row_sums
    H = -np.sum(P * np.log(P + eps), axis=1)  # per-row entropy
    H_max = np.log(N_BLOCKS - 1)  # exclude diagonal
    conc = 1.0 - np.mean(H) / H_max
    return float(conc)


def scan_all_heads(attn):
    """For each head, aggregate B, run CDL, compute metrics."""
    n_records, n_layers, n_heads = attn.shape[:3]
    l2r = L2R

    results = []
    print(f"\n{'Layer':>5s} {'H':>4s}  {'tau(C-D+L)':>11s}  {'tau(C-D)':>9s}  {'tau(-D)':>9s}  {'tau(L)':>9s}  {'row_conc':>9s}  {'B_mean':>8s}  {'B_max':>8s}")
    print("-" * 90)

    for L in range(n_layers):
        for H in range(n_heads):
            # Aggregate B across all records: mean attention
            B_all = attn[:, L, H, :, :]  # [R, 64, 64]

            # Zero diagonal for each record
            for r in range(B_all.shape[0]):
                np.fill_diagonal(B_all[r], 0.0)

            # Mean B across records
            B_mean = B_all.mean(axis=0)  # [64, 64]

            # Also compute B via column-normalize? No, use raw B.
            # CDL works on B[u,v] = edge u→v
            # The dataset attention is row-normalized (softmax output)
            # In our pipeline, B = A.T where A is row-mean of block attention
            # Here attention is already row-normalized per token
            # Use as-is: B[u,v] = mean attention from block u to block v

            B = np.asarray(B_mean, dtype=np.float64).copy()
            np.fill_diagonal(B, 0.0)

            # Row concentration
            rc = compute_row_conc(B)

            # CDL with different modes
            tau_cdl = None
            tau_cd = None
            tau_d = None
            tau_l = None

            try:
                order_cdl = rollout_order(B, tau_T=1.0, seed=42, mode="C-D+L", greedy=True)
                tau_cdl = kendalltau(order_cdl, l2r).correlation
            except Exception as e:
                pass

            try:
                order_cd = rollout_order(B, tau_T=1.0, seed=42, mode="C-D", greedy=True)
                tau_cd = kendalltau(order_cd, l2r).correlation
            except Exception:
                pass

            try:
                order_d = rollout_order(B, tau_T=1.0, seed=42, mode="-D", greedy=True)
                tau_d = kendalltau(order_d, l2r).correlation
            except Exception:
                pass

            try:
                order_l = rollout_order(B, tau_T=1.0, seed=42, mode="L", greedy=True)
                tau_l = kendalltau(order_l, l2r).correlation
            except Exception:
                pass

            b_mean = float(B.mean())
            b_max = float(B.max())

            print(f"  L{L}   H{H}  {tau_cdl:>+11.4f}  {tau_cd:>+9.4f}  {tau_d:>+9.4f}  {tau_l:>+9.4f}  {rc:>9.4f}  {b_mean:>8.5f}  {b_max:>8.5f}")

            results.append({
                'layer': L, 'head': H,
                'tau_cdl': tau_cdl, 'tau_cd': tau_cd, 'tau_d': tau_d, 'tau_l': tau_l,
                'row_conc': rc, 'B_mean': b_mean, 'B_max': b_max,
            })

    return results


def summarize(results):
    """Print top/bottom heads and distribution summary."""
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    # Sort by |tau|
    sorted_by_tau = sorted(results, key=lambda r: abs(r['tau_cdl'] or 0), reverse=True)

    print("\nTop-8 heads by |tau(C-D+L)|:")
    print(f"  {'Rank':>4s}  {'Head':>8s}  {'tau(C-D+L)':>11s}  {'tau(C-D)':>9s}  {'tau(-D)':>9s}  {'tau(L)':>9s}  {'row_conc':>9s}")
    print("  " + "-" * 70)
    for i, r in enumerate(sorted_by_tau[:8]):
        print(f"  {i+1:>4d}  L{r['layer']}H{r['head']:1d}   {r['tau_cdl']:>+11.4f}  {r['tau_cd']:>+9.4f}  {r['tau_d']:>+9.4f}  {r['tau_l']:>+9.4f}  {r['row_conc']:>9.4f}")

    # Sort by row_conc
    sorted_by_rc = sorted(results, key=lambda r: r['row_conc'], reverse=True)
    print("\nTop-8 heads by row_conc:")
    for i, r in enumerate(sorted_by_rc[:8]):
        print(f"  {i+1:>4d}  L{r['layer']}H{r['head']:1d}  row_conc={r['row_conc']:.4f}  tau(C-D+L)={r['tau_cdl']:>+.4f}")

    # Distribution
    taus = [r['tau_cdl'] for r in results if r['tau_cdl'] is not None]
    if taus:
        print(f"\nτ(C-D+L) distribution: min={min(taus):+.4f}  max={max(taus):+.4f}  "
              f"mean={np.mean(taus):+.4f}  std={np.std(taus):.4f}")
        n_pos = sum(1 for t in taus if t > 0.3)
        n_neg = sum(1 for t in taus if t < -0.3)
        n_near_zero = sum(1 for t in taus if abs(t) <= 0.3)
        print(f"  |τ|>0.3: pos={n_pos}  neg={n_neg}  |τ|≤0.3={n_near_zero}")

    rcs = [r['row_conc'] for r in results]
    print(f"\nrow_conc distribution: min={min(rcs):.4f}  max={max(rcs):.4f}  "
          f"mean={np.mean(rcs):.4f}  std={np.std(rcs):.4f}")

    # Correlation between row_conc and |tau|
    abs_taus = [abs(r['tau_cdl'] or 0) for r in results]
    rc_tau_corr = np.corrcoef(rcs, abs_taus)[0, 1]
    print(f"\nCorr(row_conc, |tau|) = {rc_tau_corr:.4f}")


if __name__ == "__main__":
    attn = load_all_data()
    results = scan_all_heads(attn)
    summarize(results)
