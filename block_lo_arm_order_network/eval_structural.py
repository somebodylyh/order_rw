"""Structural evaluation for attention-consistent reranker.

Metrics:
  W_edge(order)   = Σ_t A[order_{t+1}, order_t]
  W_prefix(order) = Σ_t mean_{p in prefix_t} A[order_t, p]
  τ vs DP(A)
  Weight ratio vs DP(A) optimal

Baselines: random, old_ON, A-greedy, DP.
NLL is NOT used — purely structural evaluation.
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RerankerConfig
from reranker import (
    load_old_on,
    OldONPrior,
    masks_to_revealed_bool,
    sequential_generate,
    compute_kendall_tau,
    compute_swap_metrics,
    check_order_valid,
)
from dp_solver import solve_dp_full, solve_dp_batch


def get_or_compute_dp(A_batch: np.ndarray, cache_path: str = None) -> list:
    """Get DP results, computing and caching if needed.

    Args:
        A_batch: (B, N, N)
        cache_path: path to .npy cache file (uses A hash if None)

    Returns:
        list of dicts with optimal_path, max_weight
    """
    B = A_batch.shape[0]
    if cache_path is None:
        import hashlib
        h = hashlib.md5(A_batch.tobytes()[:1024]).hexdigest()[:8]
        cache_path = f"probe_results/reranker/dp_cache_{h}_{B}.npy"

    if os.path.exists(cache_path):
        print(f"Loading cached DP from {cache_path}")
        cached = np.load(cache_path, allow_pickle=True)
        return cached.tolist()

    print(f"Computing DP for {B} seqs (this may take a few minutes)...")
    t0 = time.time()
    results = solve_dp_batch(A_batch, num_blocks=A_batch.shape[1], show_progress=True)
    print(f"DP took {time.time() - t0:.1f}s")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, np.array(results, dtype=object), allow_pickle=True)
    print(f"Cached DP to {cache_path}")
    return results


# ── Structural scores ──────────────────────────────────────────────────────────

def compute_path_weight(A: np.ndarray, order: np.ndarray, mode: str = "edge") -> float:
    """Total structural path score for a full order.

    Args:
        A: (N, N) attention matrix, A[j,i] = j's attention to i
        order: (N,) block order
        mode: "edge" → Σ A[order_{t+1}, order_t]
              "prefix_mean" → Σ mean_{p in prefix_t} A[order_t, p]

    Returns:
        scalar path weight
    """
    N = len(order)
    total = 0.0
    for t in range(N - 1):
        if mode == "edge":
            total += A[order[t + 1], order[t]]
        elif mode == "prefix_mean":
            prefix = order[:t + 1]
            nxt = order[t + 1]
            total += A[nxt, prefix].mean()
        else:
            raise ValueError(f"Unknown mode: {mode}")
    return total


def compute_prefix_score(
    A: np.ndarray,
    prefix: np.ndarray,
    candidate: int,
    mode: str = "edge",
) -> float:
    """Structural score for one candidate given prefix.

    Args:
        A: (N, N) attention matrix
        prefix: (t,) already-placed blocks
        candidate: scalar block index
        mode: "edge" → A[candidate, prefix[-1]]
              "prefix_mean" → mean(A[candidate, prefix])
              "prefix_max" → max(A[candidate, prefix])
    """
    if mode == "edge":
        return float(A[candidate, prefix[-1]])
    elif mode == "prefix_mean":
        return float(A[candidate, prefix].mean())
    elif mode == "prefix_max":
        return float(A[candidate, prefix].max())
    else:
        raise ValueError(f"Unknown mode: {mode}")


def build_structural_label(
    A: np.ndarray,
    prefix: np.ndarray,
    candidates: np.ndarray,
    mode: str = "edge",
    tau: float = 0.3,
) -> np.ndarray:
    """Per-step soft label from attention structure.

    struct_score_i = compute_prefix_score(A, prefix, i, mode)
    q_i = softmax(struct_score_i / tau)

    Returns:
        q: (K,) soft label distribution
    """
    scores = np.array([compute_prefix_score(A, prefix, int(c), mode) for c in candidates])
    scores = scores / tau
    scores = scores - scores.max()  # stable softmax
    q = np.exp(scores)
    q = q / q.sum()
    return q


# ── Greedy A-reranker (Stage 1 baseline, no training) ──────────────────────────

@torch.no_grad()
def generate_order_A_greedy(A: torch.Tensor, mode: str = "edge") -> torch.Tensor:
    """Greedy order by structural score: at each step pick argmax struct_score(candidate | prefix).

    Args:
        A: (B, N, N)
        mode: "edge" (A[last,i]) or "prefix_mean"

    Returns:
        orders: (B, N)
    """
    B, N, _ = A.shape
    device = A.device
    A_np = A.cpu().numpy()

    orders = torch.zeros(B, N, dtype=torch.long, device=device)
    for b in range(B):
        visited = set()
        prefix = []
        for t in range(N):
            unvisited = [i for i in range(N) if i not in visited]
            if t == 0:
                # First node: pick argmax incoming attention from others
                # (best target for the first edge A[next, start])
                scores = [A_np[b, :, i].mean() for i in unvisited]
            else:
                prefix_arr = np.array(prefix)
                scores = [compute_prefix_score(A_np[b], prefix_arr, i, mode) for i in unvisited]
            best = unvisited[int(np.argmax(scores))]
            orders[b, t] = best
            visited.add(best)
            prefix.append(best)
    return orders


# ── Full structural evaluation ─────────────────────────────────────────────────

def structural_eval(
    A_batch: torch.Tensor,
    old_on_prior: OldONPrior = None,
    num_random_mc: int = 20,
    modes: list = ["edge"],
    seed: int = 42,
):
    """Compare random, old_ON, A-greedy, DP on structural metrics.

    Args:
        A_batch: (B, N, N)
        old_on_prior: optional OldONPrior for old_ON orders
        num_random_mc: MC samples for random baseline
        modes: structural modes to evaluate

    Returns:
        results dict with per-mode metrics
    """
    B, N, _ = A_batch.shape
    device = A_batch.device
    A_np = A_batch.cpu().numpy()

    rng = np.random.RandomState(seed)
    all_results = {}

    # Pre-compute DP (cached)
    dp_results = get_or_compute_dp(A_np)
    dp_orders = np.array([r["optimal_path"] for r in dp_results])
    dp_weights = np.array([r["max_weight"] for r in dp_results])

    for mode in modes:
        mode_results = defaultdict(list)

        for b in tqdm(range(B), desc=f"Structural eval [{mode}]"):
            A = A_np[b]
            dp_order = dp_orders[b]
            dp_weight = dp_weights[b]

            # A-greedy
            greedy_order = generate_order_A_greedy(
                A_batch[b:b+1], mode=mode
            )[0].cpu().numpy()
            greedy_weight = compute_path_weight(A, greedy_order, mode=mode)

            # Old ON (if available)
            if old_on_prior is not None:
                old_on_order = old_on_prior.get_full_order(A_batch[b:b+1])[0].cpu().numpy()
                old_on_weight = compute_path_weight(A, old_on_order, mode=mode)
                old_on_tau = compute_kendall_tau(
                    torch.from_numpy(old_on_order), torch.from_numpy(dp_order)
                )
                old_on_swaps = compute_swap_metrics(
                    torch.from_numpy(old_on_order), torch.from_numpy(dp_order)
                )

            # Random MC
            rand_weights = []
            rand_taus = []
            for _ in range(num_random_mc):
                rand_order = rng.permutation(N)
                rand_weights.append(compute_path_weight(A, rand_order, mode=mode))
                rand_taus.append(compute_kendall_tau(
                    torch.from_numpy(rand_order), torch.from_numpy(dp_order)
                ))
            rand_weight_mean = np.mean(rand_weights)
            rand_tau_mean = np.mean(rand_taus)

            # Collect
            mode_results["dp_weight"].append(dp_weight)
            mode_results["greedy_weight"].append(greedy_weight)
            mode_results["greedy_weight_ratio"].append(greedy_weight / dp_weight if dp_weight > 0 else 0)
            mode_results["greedy_tau_vs_dp"].append(compute_kendall_tau(
                torch.from_numpy(greedy_order), torch.from_numpy(dp_order)
            ))
            mode_results["rand_weight"].append(rand_weight_mean)
            mode_results["rand_weight_ratio"].append(rand_weight_mean / dp_weight if dp_weight > 0 else 0)
            mode_results["rand_tau_vs_dp"].append(rand_tau_mean)

            if old_on_prior is not None:
                mode_results["old_on_weight"].append(old_on_weight)
                mode_results["old_on_weight_ratio"].append(old_on_weight / dp_weight if dp_weight > 0 else 0)
                mode_results["old_on_tau_vs_dp"].append(old_on_tau)

        # Aggregate
        summary = {}
        for k, v in mode_results.items():
            arr = np.array(v)
            summary[k] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }

        print(f"\n=== Structural Results [{mode}] ({B} seqs) ===")
        print(f"  DP optimal weight:          {summary['dp_weight']['mean']:.4f}")
        print(f"  A-greedy weight:            {summary['greedy_weight']['mean']:.4f}  "
              f"(ratio={summary['greedy_weight_ratio']['mean']:.3f})")
        print(f"  A-greedy τ vs DP:          {summary['greedy_tau_vs_dp']['mean']:.3f}")
        if old_on_prior is not None:
            print(f"  Old ON weight:              {summary['old_on_weight']['mean']:.4f}  "
                  f"(ratio={summary['old_on_weight_ratio']['mean']:.3f})")
            print(f"  Old ON τ vs DP:            {summary['old_on_tau_vs_dp']['mean']:.3f}")
        print(f"  Random weight:              {summary['rand_weight']['mean']:.4f}  "
              f"(ratio={summary['rand_weight_ratio']['mean']:.3f})")
        print(f"  Random τ vs DP:            {summary['rand_tau_vs_dp']['mean']:.3f}")

        all_results[mode] = summary

    return all_results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data-path", type=str,
                        default="probe_results/A_train_10k.npy")
    parser.add_argument("--old-on-ckpt", type=str,
                        default="probe_results/crossattn_on_best.pt")
    parser.add_argument("--num-seqs", type=int, default=100)
    parser.add_argument("--num-random-mc", type=int, default=50)
    parser.add_argument("--modes", type=str, default="edge,prefix_mean",
                        help="comma-separated structural modes")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device

    print("Loading A matrices...")
    A_all = np.load(args.data_path, mmap_mode="r")
    A_batch = torch.from_numpy(A_all[:args.num_seqs].copy()).float().to(device)

    # Load old ON for comparison
    old_on_prior = None
    if os.path.exists(args.old_on_ckpt):
        print("Loading Old ON...")
        old_on = load_old_on(args.old_on_ckpt, device)
        old_on_prior = OldONPrior(old_on, num_blocks=16)

    modes = [m.strip() for m in args.modes.split(",")]

    results = structural_eval(
        A_batch, old_on_prior=old_on_prior,
        num_random_mc=args.num_random_mc,
        modes=modes, seed=args.seed,
    )

    # Save
    out_dir = "probe_results/reranker"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "structural_eval_results.pt")
    torch.save(results, out_path)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
