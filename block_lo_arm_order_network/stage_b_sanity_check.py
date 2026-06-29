"""
Stage B: Offline sanity checks for MLP residual policy.

Verifies 6 checks against v2 30k checkpoint:
  1. Logprob match (gamma=0 → KL≈0)
  2. Tau match (gamma=0 → tau≈1.0)
  3. Entropy match (gamma=0 → H matches)
  4. L2R logprob match (gamma=0 → same as RW)
  5. Gamma control (gamma=1.0 + random weights → tau < 0.95)
  6. Direction test (inject L2R-favoring residual → logprob(L2R) increases)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

# ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from directed_graph_policy import (
    build_directed_graph,
    sample_order,
    sample_orders,
    _policy_step_entropy_replay,
)
from order_diagnostics import _kendall_tau
from mlp_residual_policy import (
    MLPResidualPolicy,
    compute_step_features,
    sample_order_with_mlp,
    compute_logprob_for_order_mlp,
    _softmax_torch,
    compute_source,
    progressive_rw_step,
    _softmax,
)


def run_sanity_checks(ckpt_path, n_samples=2000, n_replay=100, seed=42):
    """Run all 6 offline sanity checks."""
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # Reconstruct B from A_global in same directory
    ckpt_dir = os.path.dirname(ckpt_path)
    a_path = os.path.join(ckpt_dir, "A_global_step30000.npy")
    if not os.path.exists(a_path):
        a_path = os.path.join(ckpt_dir, "A_global_eval.npy")
    print(f"Loading A_global from: {a_path}")
    A_global = np.load(a_path)
    np.fill_diagonal(A_global, 0.0)
    B = build_directed_graph(A_global)

    rw_params = ckpt.get("rw_params", {
        "tau_start": 0.1, "tau_step": 0.1, "alpha_dep": 0.5, "alpha_pr": 0.85,
        "beta_sup": 1.0, "beta_fut": 0.5, "beta_src": 0.2, "beta_loc": 0.5,
        "top_k": 4,
    })
    N = B.shape[0]
    L2R = np.arange(N, dtype=np.int64)
    device = "cpu"
    all_pass = True

    print(f"\nN={N}, rw_params={json.dumps(rw_params, indent=2)}")

    # ── Create MLP with gamma=0 ──
    mlp_zero = MLPResidualPolicy(gamma=0.0).to(device)
    mlp_zero.eval()
    print(f"\nMLP gamma: {mlp_zero.gamma.item():.6f}")

    # ── Sample matched orders ──
    print(f"\nSampling {n_samples} matched orders (same seed) ...")
    t0 = time.time()
    rw_orders = np.zeros((n_samples, N), dtype=np.int64)
    mlp_orders = np.zeros((n_samples, N), dtype=np.int64)
    rw_logprobs = np.zeros(n_samples, dtype=np.float64)
    mlp_logprobs = np.zeros(n_samples, dtype=np.float64)

    for k in range(n_samples):
        s = seed * 10000 + k
        rw_orders[k], rw_logprobs[k] = sample_order(B, 'progressive_rw', rw_params, seed=s)
        mlp_orders[k], mlp_logprobs[k], _, _ = sample_order_with_mlp(
            B, rw_params, seed=s, mlp=mlp_zero, mlp_device=device,
        )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({n_samples/elapsed:.0f} samples/s)")

    # ── Check 1: Logprob match ──
    logprob_diff = rw_logprobs - mlp_logprobs
    mean_diff = float(logprob_diff.mean())
    max_abs_diff = float(np.max(np.abs(logprob_diff)))
    print(f"\n--- Check 1: Logprob match ---")
    print(f"  Mean diff (KL approx): {mean_diff:.6f}  (want ~0)")
    print(f"  Max |diff|:            {max_abs_diff:.6f}  (want < 0.01)")
    c1 = abs(mean_diff) < 0.01
    print(f"  {'PASS' if c1 else 'FAIL'}")

    # ── Check 2: Tau match ──
    taus = np.array([_kendall_tau(rw_orders[i], mlp_orders[i]) for i in range(n_samples)])
    mean_tau = float(taus.mean())
    min_tau = float(taus.min())
    print(f"\n--- Check 2: Tau match ---")
    print(f"  Mean tau(RW, MLP) same seed: {mean_tau:.4f}  (want > 0.99)")
    print(f"  Min tau:                     {min_tau:.4f}")
    c2 = mean_tau > 0.99
    print(f"  {'PASS' if c2 else 'FAIL'}")

    # ── Check 3: Entropy match ──
    print(f"\n--- Check 3: Entropy match ---")
    rw_H_all = np.zeros((n_replay, N), dtype=np.float64)
    for i in range(n_replay):
        s = seed * 10000 + i
        rw_H_all[i] = _policy_step_entropy_replay(B, 'progressive_rw', rw_params, rw_orders[i], s)

    rw_entropy_mean = float(rw_H_all.mean())
    rw_entropy_early = float(rw_H_all[:, :21].mean())
    rw_entropy_mid = float(rw_H_all[:, 21:42].mean())
    rw_entropy_late = float(rw_H_all[:, 42:].mean())

    # Replay MLP orders using same RW entropy function (gamma=0 gives same distribution)
    mlp_H_all = np.zeros((n_replay, N), dtype=np.float64)
    for i in range(n_replay):
        s = seed * 10000 + i
        mlp_H_all[i] = _policy_step_entropy_replay(B, 'progressive_rw', rw_params, mlp_orders[i], s)

    mlp_entropy_mean = float(mlp_H_all.mean())
    mlp_entropy_early = float(mlp_H_all[:, :21].mean())
    mlp_entropy_mid = float(mlp_H_all[:, 21:42].mean())
    mlp_entropy_late = float(mlp_H_all[:, 42:].mean())

    print(f"  RW entropy  (total/early/mid/late): {rw_entropy_mean:.4f} / {rw_entropy_early:.4f} / {rw_entropy_mid:.4f} / {rw_entropy_late:.4f}")
    print(f"  MLP entropy (total/early/mid/late): {mlp_entropy_mean:.4f} / {mlp_entropy_early:.4f} / {mlp_entropy_mid:.4f} / {mlp_entropy_late:.4f}")
    c3 = abs(rw_entropy_mean - mlp_entropy_mean) < 0.01
    print(f"  {'PASS' if c3 else 'FAIL'}")

    # ── Check 4: L2R logprob ──
    print(f"\n--- Check 4: L2R logprob match ---")
    l2r_lp_rw = compute_logprob_for_order_mlp(B, rw_params, L2R, mlp=None, mlp_device=device)
    l2r_lp_mlp = compute_logprob_for_order_mlp(B, rw_params, L2R, mlp=mlp_zero, mlp_device=device)
    print(f"  logprob(L2R | RW):  {l2r_lp_rw:.4f}")
    print(f"  logprob(L2R | MLP): {l2r_lp_mlp:.4f}")
    c4 = abs(l2r_lp_rw - l2r_lp_mlp) < 0.01
    print(f"  {'PASS' if c4 else 'FAIL'}")

    # ── Also compute random order logprob for reference ──
    rand_order = np.random.default_rng(12345).permutation(N)
    rand_lp_rw = compute_logprob_for_order_mlp(B, rw_params, rand_order, mlp=None, mlp_device=device)
    print(f"  logprob(random | RW): {rand_lp_rw:.4f}  (reference)")

    # ── Check 5: Gamma control ──
    print(f"\n--- Check 5: Gamma control (gamma=1.0, random weights) ---")
    mlp_random = MLPResidualPolicy(gamma=1.0).to(device)
    mlp_random.eval()
    # don't zero the last layer — use random init
    with torch.no_grad():
        for name, param in mlp_random.net.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.uniform_(param, -0.1, 0.1)

    n_ctrl = min(500, n_samples)
    taus_random = np.zeros(n_ctrl, dtype=np.float64)
    for k in range(n_ctrl):
        s = seed * 10000 + k
        rw_ord, _ = sample_order(B, 'progressive_rw', rw_params, seed=s)
        mlp_ord, _, _, _ = sample_order_with_mlp(B, rw_params, seed=s, mlp=mlp_random, mlp_device=device)
        taus_random[k] = _kendall_tau(rw_ord, mlp_ord)

    mean_tau_rand = float(taus_random.mean())
    print(f"  Mean tau(RW, MLP_random) same seed: {mean_tau_rand:.4f}  (want < 0.95)")
    c5 = mean_tau_rand < 0.95
    print(f"  {'PASS' if c5 else 'FAIL'}")

    # ── Check 6: Direction test ──
    print(f"\n--- Check 6: Direction test ---")
    # Inject a fake residual that at each step gives bonus to the L2R-next candidate.
    # This is debug-only: verifies that boosting L2R-favoring scores increases logprob(L2R).
    # We do this by setting mlp to produce a fixed residual vector favoring small-block indices.

    class DirectionTestResidual:
        """Fake residual: gives +bonus to candidates with smallest physical block index."""
        def __init__(self, bonus=1.0):
            self.bonus = bonus
        def __call__(self, U_phys):
            # U_phys: physical block indices of candidates
            # return residual proportional to -phys_idx (lower idx = higher residual)
            r = np.zeros(len(U_phys), dtype=np.float64)
            # rank by physical index: lowest gets bonus, highest gets -bonus
            ranks = np.argsort(np.argsort(U_phys).astype(np.float64))
            if len(U_phys) > 1:
                r = self.bonus * (1.0 - 2.0 * ranks / (len(U_phys) - 1))
            return r

    dir_residual = DirectionTestResidual(bonus=1.0)

    # Compute logprob(L2R) with and without direction residual
    # We modify the replay to add direction_residual to the RW score at each step
    def logprob_with_direction_residual(B, params, order, residual_fn):
        B_np = np.asarray(B, dtype=np.float64)
        order_np = np.asarray(order, dtype=np.int64)
        N = B_np.shape[0]

        tau_start = float(params.get('tau_start', 0.1))
        tau_step = float(params.get('tau_step', 0.1))
        alpha_dep = float(params.get('alpha_dep', 0.5))
        top_k = int(params.get('top_k', 0) or 0)

        source, out_deg, in_deg = compute_source(B_np, alpha_dep)
        betas = {
            'sup': float(params.get('beta_sup', 1.0)),
            'fut': float(params.get('beta_fut', 0.5)),
            'src': float(params.get('beta_src', 0.2)),
            'loc': float(params.get('beta_loc', 0.5)),
        }

        rng0 = np.random.default_rng(0)
        p0_scores = source.copy() + residual_fn(np.arange(N, dtype=np.int64))
        p0 = _softmax(p0_scores, tau_start, rng0, top_k=top_k)
        idx0 = int(order_np[0])
        lp = float(np.log(max(p0[idx0], 1e-300)))

        S = np.array([idx0], dtype=np.int64)
        U = np.setdiff1d(np.arange(N, dtype=np.int64), S)
        last = idx0
        rng_step = np.random.default_rng(1)

        for t in range(1, N):
            chosen = int(order_np[t])
            if chosen not in U:
                break
            _, rw_score = progressive_rw_step(B_np, S, U, last, betas, tau_step, source, rng_step, top_k=top_k)
            combined = rw_score + residual_fn(U)
            p_t = _softmax(combined, tau_step, rng_step, top_k=top_k)
            chosen_idx = int(np.where(U == chosen)[0][0])
            lp += float(np.log(max(p_t[chosen_idx], 1e-300)))
            S = np.append(S, chosen)
            U = U[U != chosen]
            last = chosen
        return lp

    l2r_lp_base = compute_logprob_for_order_mlp(B, rw_params, L2R, mlp=None, mlp_device=device)
    l2r_lp_boost = logprob_with_direction_residual(B, rw_params, L2R, dir_residual)

    # Also test: random order should get less boost
    rand_lp_base = compute_logprob_for_order_mlp(B, rw_params, rand_order, mlp=None, mlp_device=device)
    rand_lp_boost = logprob_with_direction_residual(B, rw_params, rand_order, dir_residual)

    print(f"  logprob(L2R) base:      {l2r_lp_base:.4f}")
    print(f"  logprob(L2R) boosted:   {l2r_lp_boost:.4f}  (want > base)")
    print(f"  logprob(random) base:   {rand_lp_base:.4f}")
    print(f"  logprob(random) boosted:{rand_lp_boost:.4f}")
    l2r_delta = l2r_lp_boost - l2r_lp_base
    print(f"  L2R delta:              {l2r_delta:+.4f}  (want > 0)")
    c6 = l2r_delta > 0
    print(f"  {'PASS' if c6 else 'FAIL'}")

    # ── Summary ──
    all_pass = all([c1, c2, c3, c4, c5, c6])
    print(f"\n{'='*50}")
    print(f"  ALL CHECKS: {'PASS' if all_pass else 'FAIL'}")
    print(f"  Check 1 (logprob match):  {'PASS' if c1 else 'FAIL'}")
    print(f"  Check 2 (tau match):      {'PASS' if c2 else 'FAIL'}")
    print(f"  Check 3 (entropy match):  {'PASS' if c3 else 'FAIL'}")
    print(f"  Check 4 (L2R logprob):    {'PASS' if c4 else 'FAIL'}")
    print(f"  Check 5 (gamma control):  {'PASS' if c5 else 'FAIL'}")
    print(f"  Check 6 (direction test): {'PASS' if c6 else 'FAIL'}")
    print(f"{'='*50}")

    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Stage B sanity checks")
    parser.add_argument("--ckpt", type=str,
                        default="probe_results/clean_method_graph_rw_a09_from20k_v2/ckpt_step30000.pt")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ok = run_sanity_checks(args.ckpt, n_samples=args.n_samples, seed=args.seed)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
