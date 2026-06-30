"""Validate unsupervised head selector on seed1 @ 10k.

For each head, compute supervised τ_vs_L2R (oracle) vs unsupervised signals:
  1. row-concentration: negentropy of B row means (signal = how focused each row is)
  2. S_forward (split-half): τ between CDL orders from two independent batch halves
  3. S_drift: τ between CDL orders from adjacent batch subsets (stability)
  4. B variance: mean variance across batch-mean B entries (higher = more structure)

Then report: if we had no L2R label, which head would each unsupervised signal pick?
"""
import sys, os, time
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AOGPT_DIR = os.path.join(_SCRIPT_DIR, "..", "block_lo_arm_order_network")
sys.path.insert(0, _AOGPT_DIR)

from attn_order_teacher import rollout_order
from per_head_order_scan import extract_per_head_and_heavy_A
from neural_readout.extract_b import _load_model_and_chunks

N_BLOCKS = 64
L2R = np.arange(N_BLOCKS)
TAU_T = 1.0


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


def row_concentration(B):
    """Per-row negentropy mean: higher = rows have clear peaks (more structure)."""
    B = np.asarray(B, dtype=np.float64)
    N = B.shape[0]
    concs = []
    for i in range(N):
        row = B[i].copy()
        row = np.abs(row)  # use magnitude
        row_sum = row.sum()
        if row_sum < 1e-12:
            concs.append(0.0)
        else:
            p = row / row_sum
            p = p[p > 1e-12]
            H = -np.sum(p * np.log(p))
            H_max = np.log(N - 1)  # max entropy (uniform over N-1 targets, diag=0)
            concs.append(1.0 - H / max(H_max, 1e-12))  # 0=uniform, 1=one-hot
    return float(np.mean(concs))


def B_variance(B_batch):
    """Mean variance across batch of B matrices — higher = more structured signal."""
    return float(np.var(B_batch, axis=0).mean())


def split_half_tau(B_batch_list, seed=42):
    """Split M batch B matrices in half, CDL each half, τ between orders."""
    M = len(B_batch_list)
    if M < 20:
        return 0.0
    mid = M // 2
    rng = np.random.default_rng(seed)
    idx = rng.permutation(M)
    half1 = idx[:mid]; half2 = idx[mid:2*mid]

    B1 = np.mean([B_batch_list[i] for i in half1], axis=0)
    B2 = np.mean([B_batch_list[i] for i in half2], axis=0)

    o1 = rollout_order(B1, tau_T=TAU_T, seed=seed, mode="C-D+L", standardize=True)
    o2 = rollout_order(B2, tau_T=TAU_T, seed=seed+1, mode="C-D+L", standardize=True)
    return kendall_tau(o1, o2)


CKPT = ("/home/admin/lyuyuhuan/order_lyu/block_lo_arm_order_network/"
        "probe_results/random_baseline_continuous_jun09_seed1/ckpt_step10000.pt")

t0 = time.time()
print(f"Loading {CKPT}")
model, chunks, clean_perm, dev, ci = _load_model_and_chunks(
    ckpt_path=CKPT, M=200, seed=42, device="cuda:0", split="train")

A_lh, _ = extract_per_head_and_heavy_A(
    model, chunks, clean_perm, dev, seed=42, fwd_batch=64, none_mode="b0")
n, L, H, N, _ = A_lh.shape
print(f"Extracted {n} graphs, {L}x{H} heads in {time.time()-t0:.1f}s\n")

# Compute all signals per head
results = []
for l in range(L):
    for h in range(H):
        # Batch-mean B
        B_mean = _A_to_B(A_lh[:, l, h].mean(axis=0))

        # Oracle: τ_vs_L2R
        o_cdl = rollout_order(B_mean, tau_T=TAU_T, seed=42, mode="C-D+L", standardize=True)
        tau_oracle = kendall_tau(o_cdl, L2R)

        # Unsupervised signals
        row_conc = row_concentration(B_mean)
        b_var = B_variance(A_lh[:, l, h])
        s_forward = split_half_tau([_A_to_B(A_lh[i, l, h]) for i in range(n)], seed=42)
        s_drift = split_half_tau([_A_to_B(A_lh[i, l, h]) for i in range(n)], seed=43)

        results.append({
            "head": f"L{l}H{h}", "l": l, "h": h,
            "tau_oracle": tau_oracle,
            "row_conc": row_conc,
            "b_var": b_var,
            "s_forward": s_forward,
            "s_drift": s_drift,
        })

# ── Ranking comparison ──────────────────────────────────────────
oracle_rank = sorted(results, key=lambda x: -x["tau_oracle"])
row_conc_rank = sorted(results, key=lambda x: -x["row_conc"])
s_forward_rank = sorted(results, key=lambda x: -x["s_forward"])
b_var_rank = sorted(results, key=lambda x: -x["b_var"])

print(f"{'Head':>8s}  {'τ(L2R)':>8s}  {'row-conc':>10s}  {'S_forward':>10s}  {'B_var':>10s}  {'S_drift':>10s}")
print("-" * 70)
for r in sorted(results, key=lambda x: -x["tau_oracle"]):
    marker = " ★" if r["tau_oracle"] > 0.3 else ""
    print(f"  {r['head']}  {r['tau_oracle']:+.4f}    {r['row_conc']:.4f}       {r['s_forward']:.4f}       {r['b_var']:.6f}     {r['s_drift']:.4f}{marker}")

# ── Top-3 concordance ──────────────────────────────────────────
def top3(ranked):
    return set(r["head"] for r in ranked[:3])

oracle_top3 = top3(oracle_rank)
print(f"\nOracle top-3 (τ_vs_L2R):  {oracle_top3}")
print(f"Row-conc top-3:           {top3(row_conc_rank)}  overlap={len(oracle_top3 & top3(row_conc_rank))}/3")
print(f"S_forward top-3:          {top3(s_forward_rank)}  overlap={len(oracle_top3 & top3(s_forward_rank))}/3")
print(f"B_var top-3:              {top3(b_var_rank)}  overlap={len(oracle_top3 & top3(b_var_rank))}/3")

# ── Spearman ρ between oracle and each unsupervised signal ──────
from scipy.stats import spearmanr
oracle_taus = np.array([r["tau_oracle"] for r in results])
for sig_name in ["row_conc", "s_forward", "s_drift", "b_var"]:
    sig = np.array([r[sig_name] for r in results])
    rho, p = spearmanr(oracle_taus, sig)
    print(f"  ρ(τ_oracle, {sig_name:12s}) = {rho:+.4f}  (p={p:.4f})")

# ── Key question: does S_forward pick L1H1 over L0H7? ─────────
print(f"\n=== Critical test: L1H1 vs L0H7 ===")
for r in results:
    if r["head"] in ("L1H1", "L0H7"):
        print(f"  {r['head']}: τ_oracle={r['tau_oracle']:+.4f}  "
              f"row_conc={r['row_conc']:.4f}  S_forward={r['s_forward']:.4f}  B_var={r['b_var']:.6f}")

print("\nDone.")
