"""Sweep tau_step to find the sweet spot between greedy (τ≈0.91) and noise (τ≈0)."""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from directed_graph_policy import build_directed_graph, sample_orders

A_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_results", "A_train_n64_10k.npy")

def sweep(B, tau_values, K=2000, seed_base=42):
    results = []
    for tau in tau_values:
        params = {
            'tau_start': tau,
            'tau_step': tau,
            'alpha_dep': 0.5, 'alpha_pr': 0.85,
            'beta_sup': 1.0, 'beta_fut': 0.5,
            'beta_src': 0.2, 'beta_loc': 0.5,
        }
        t0 = time.time()
        out = sample_orders(B, 'progressive_rw', params, K=K, seed_base=seed_base)
        elapsed = time.time() - t0
        r = {
            'tau': tau,
            'tau_vs_l2r_mean': out['tau_vs_l2r_mean'],
            'tau_vs_l2r_std': out['tau_vs_l2r_std'],
            'pairwise_tau_mean': out['pairwise_tau_mean'],
            'first_node_entropy': out['first_node_entropy'],
            'policy_step_entropy_mean': out['policy_step_entropy']['mean'],
            'policy_step_entropy_early': out['policy_step_entropy']['early'],
            'policy_step_entropy_mid': out['policy_step_entropy']['mid'],
            'policy_step_entropy_late': out['policy_step_entropy']['late'],
            'mean_directed_score': out['mean_directed_score'],
            'mean_progressive_support': out['mean_progressive_support'],
            'logprob_mean': out['logprob_mean'],
            'logprob_std': out['logprob_std'],
            'elapsed_s': elapsed,
        }
        results.append(r)
        print(f"tau={tau:.3f} | tau_vs_l2r={r['tau_vs_l2r_mean']:.4f}±{r['tau_vs_l2r_std']:.4f} | "
              f"pairwise_tau={r['pairwise_tau_mean']:.4f} | H_mean={r['policy_step_entropy_mean']:.3f} | "
              f"H_early={r['policy_step_entropy_early']:.3f} | H_late={r['policy_step_entropy_late']:.3f} | "
              f"dir_score={r['mean_directed_score']:.4f} | {elapsed:.0f}s")
    return results

def main():
    print("Loading A matrix...")
    A_all = np.load(A_PATH)
    A_global = A_all.mean(axis=0).astype(np.float32)
    np.fill_diagonal(A_global, 0.0)
    B = build_directed_graph(A_global)
    print(f"A_global: {A_global.shape}, B: {B.shape}, mean={B.mean():.6f}\n")

    # Dense sweep: 0.01 to 2.0
    tau_values = [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0]
    print(f"Sweeping tau_step over {len(tau_values)} values (K=2000 each)...\n")

    results = sweep(B, tau_values, K=2000)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_results", "tau_sweep.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Summary table
    print(f"\n{'tau':>8s}  {'τ_vs_L2R':>10s}  {'pair_τ':>8s}  {'H_mean':>7s}  {'H_early':>7s}  {'H_late':>7s}  {'dir_score':>10s}")
    print("-" * 76)
    for r in results:
        print(f"{r['tau']:8.3f}  {r['tau_vs_l2r_mean']:10.4f}  {r['pairwise_tau_mean']:8.4f}  "
              f"{r['policy_step_entropy_mean']:7.4f}  {r['policy_step_entropy_early']:7.4f}  "
              f"{r['policy_step_entropy_late']:7.4f}  {r['mean_directed_score']:10.4f}")

if __name__ == "__main__":
    main()
