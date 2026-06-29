"""Verify CDL tau vs training step on clean_base_random_perm checkpoints.

Hypothesis: early checkpoints (step 0-5k) have near-uniform A[v,0] →
CDL follows row_mean gradient → high tau_L2R. Later checkpoints (>10k)
develop content-specific A[v,0] structure → CDL follows content, not L2R →
tau drops.
"""
import sys, os, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'block_lo_arm_order_network'))

from training_utils import load_train_chunks, SEQ_LEN, N, BLOCK_LEN
from train_clean_aogpt import (
    extract_A_matrices, build_model, clean_model_args,
    phys_to_model_idx_clean, CleanPermutation, N as _N,
)
from clean_training_protocol import build_clean_block_permutation
from directed_graph_policy import build_directed_graph
from attn_order_teacher import rollout_order
from train_attn_order_mlp import kendall_tau_vs_raster
from scipy.stats import kendalltau


def simulate_greedy_cdl(B):
    """Greedy C-D+L rollout, returns trajectory."""
    N_ = B.shape[0]
    S, U, last = [], list(range(N_)), None
    traj = []
    for t in range(N_):
        if len(U) == 1:
            v = U[0]
        else:
            S_arr = np.array(S, dtype=np.int64)
            U_arr = np.array(U, dtype=np.int64)
            C = B[np.ix_(S_arr, U_arr)].mean(axis=0) if len(S) > 0 else np.zeros(len(U))
            M = B[np.ix_(U_arr, U_arr)]
            D = M.sum(axis=0) / (len(U) - 1) if len(U) > 1 else np.zeros(len(U))
            L = B[last, U_arr] if last is not None else np.zeros(len(U))
            q = C - D + L
            v = int(U_arr[np.argmax(q)])
        traj.append(v)
        S.append(v)
        U.remove(v)
        last = v
    return np.array(traj, dtype=np.int64)


def compute_metrics(A, K=128, seed=42):
    """Compute all metrics for a given A matrix."""
    B = build_directed_graph(A)  # B = A.T with zero diag

    # Greedy CDL
    traj_greedy = simulate_greedy_cdl(B)
    tau_greedy, _ = kendalltau(traj_greedy, np.arange(N))

    # Stochastic CDL (matched to refresh_diagnostics)
    orders = np.stack([
        rollout_order(B, tau_T=0.5, seed=seed + s, mode="C-D+L", standardize=True)
        for s in range(K)
    ])
    tau_stoch = float(kendall_tau_vs_raster(orders))

    # A[v,0] statistics
    a_v0 = A[1:, 0]  # attention from blocks 1..N-1 to block 0
    row_means = A.mean(axis=1)
    rm_range = row_means.max() - row_means.min()
    rm_corr = float(np.corrcoef(row_means, np.arange(N))[0, 1])

    return {
        "tau_greedy": float(tau_greedy),
        "tau_stoch": float(tau_stoch),
        "a_v0_mean": float(a_v0.mean()),
        "a_v0_std": float(a_v0.std()),
        "a_v0_min": float(a_v0.min()),
        "a_v0_max": float(a_v0.max()),
        "rm_range": float(rm_range),
        "rm_corr": float(rm_corr),
        "a_v0_to_rm_ratio": float(a_v0.std() / max(rm_range, 1e-12)),
        "traj_greedy_first8": traj_greedy[:8].tolist(),
        "B_fwd_bwd_ratio": float(
            np.triu(B, k=1)[np.triu(B, k=1) != 0].mean() /
            max(np.tril(B, k=-1)[np.tril(B, k=-1) != 0].mean(), 1e-12)
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=str,
                        default="/home/admin/lyuyuhuan/order_lyu/block_lo_arm_order_network/probe_results/clean_base_random_perm")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--steps", type=str, default="0,1000,5000,10000,20000,30000,40000,50000,60000")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n-chunks", type=int, default=8,
                        help="number of eval chunks for attention extraction")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading wikitext-103 data...")
    idx_phys = load_train_chunks(n_chunks=None)
    print(f"idx_phys: {idx_phys.shape}")

    # Load first checkpoint to get protocol + model args
    ckpt0 = torch.load(os.path.join(args.ckpt_dir, "ckpt_step0.pt"),
                       map_location="cpu", weights_only=False)
    protocol = ckpt0["clean_protocol"]
    model_args_dict = ckpt0["model_args"]

    clean_perm = CleanPermutation(
        block_perm_phys_to_model=torch.tensor(protocol["block_perm_phys_to_model"], dtype=torch.long),
        inv_perm_model_to_phys=torch.tensor(protocol["inv_perm_model_to_phys"], dtype=torch.long),
    )
    # Verify
    inv_perm = clean_perm.inv_perm_model_to_phys.cpu().numpy()
    print(f"inv_perm[:8]: {inv_perm[:8].tolist()}")

    # Convert data to model coords
    print("Converting data to model coords...")
    idx_model = phys_to_model_idx_clean(idx_phys, clean_perm)
    eval_indices = np.asarray(protocol["eval_indices"], dtype=np.int64)
    idx_eval = idx_model[eval_indices]
    print(f"Eval chunks: {len(idx_eval)}")

    # Use a fixed subset of eval chunks for consistency across steps
    rng = np.random.RandomState(args.seed)
    extract_chunks = idx_eval[rng.choice(len(idx_eval), size=min(args.n_chunks, len(idx_eval)), replace=False)]

    # model_args_dict is already dict from checkpoint, just add block_size
    model_args_dict['block_size'] = SEQ_LEN

    steps = [int(s) for s in args.steps.split(",")]
    results = []

    for step in steps:
        ckpt_path = os.path.join(args.ckpt_dir, f"ckpt_step{step}.pt")
        if not os.path.exists(ckpt_path):
            print(f"SKIP step {step}: checkpoint not found")
            results.append({"step": step, "error": "ckpt not found"})
            continue

        print(f"\n=== Step {step} ===")
        t0 = time.time()

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = build_model(model_args_dict, device, compile_model=False)
        state_dict = ckpt.get("model") or ckpt.get("model_state_dict")
        # Handle compiled model prefixes
        clean_sd = {}
        for k, v in state_dict.items():
            k = k.replace("_orig_mod.", "")
            clean_sd[k] = v
        model.load_state_dict(clean_sd)
        model.to(device)
        model.eval()

        # Extract A
        A_all = extract_A_matrices(model, extract_chunks, clean_perm, device, n_chunks=args.n_chunks)
        A_global = A_all.mean(axis=0).astype(np.float64)
        np.fill_diagonal(A_global, 0.0)

        metrics = compute_metrics(A_global)
        metrics["step"] = step
        metrics["extract_time_s"] = round(time.time() - t0, 1)
        results.append(metrics)

        print(f"  tau_greedy={metrics['tau_greedy']:.4f}  tau_stoch={metrics['tau_stoch']:.4f}")
        print(f"  a_v0_std={metrics['a_v0_std']:.6f}  rm_range={metrics['rm_range']:.6f}  "
              f"ratio={metrics['a_v0_to_rm_ratio']:.2f}")
        print(f"  rm_corr={metrics['rm_corr']:.4f}  traj[:8]={metrics['traj_greedy_first8']}")
        print(f"  time: {metrics['extract_time_s']:.1f}s")

        del model, ckpt
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Print summary table
    print("\n" + "=" * 100)
    print(f"{'Step':>6}  {'tau_greedy':>10}  {'tau_stoch':>10}  {'a_v0_std':>10}  "
          f"{'rm_range':>10}  {'ratio':>8}  {'rm_corr':>8}  {'traj_first8'}")
    print("-" * 100)
    for r in results:
        if "error" in r:
            print(f"{r['step']:>6}  ERROR: {r['error']}")
        else:
            print(f"{r['step']:>6}  {r['tau_greedy']:>10.4f}  {r['tau_stoch']:>10.4f}  "
                  f"{r['a_v0_std']:>10.6f}  {r['rm_range']:>10.6f}  "
                  f"{r['a_v0_to_rm_ratio']:>8.2f}  {r['rm_corr']:>8.4f}  "
                  f"{r['traj_greedy_first8']}")

    # Save
    output_path = args.output or os.path.join(args.ckpt_dir, "cdl_evolution.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
