"""
GPU-accelerated attention-only curriculum order generation.

Replaces the CPU for-loop in nn_paths_n64.py with batched GPU operations.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

_current = os.path.dirname(os.path.abspath(__file__))
if _current not in sys.path:
    sys.path.insert(0, _current)

from order_diagnostics import evaluate_order_diagnostics, write_summary_tsv

METHODS = ("old_nn", "mixed")


def _symmetrize_and_zero_diag(W: torch.Tensor) -> torch.Tensor:
    W = 0.5 * (W + W.transpose(-2, -1))
    W.diagonal(dim1=-2, dim2=-1).zero_()
    return W


def batched_old_nn_greedy(W: torch.Tensor, n_candidates: int = 2) -> torch.Tensor:
    """W: (B, N, N) symmetrized proximity with zero diag, on GPU."""
    B, N, _ = W.shape
    device = W.device

    row_sum = W.sum(dim=2)
    k = min(n_candidates, N)
    # Match CPU old_nn_greedy: np.argsort(W.sum(axis=1))[:k] picks LOWEST row-sum
    start_candidates = torch.argsort(row_sum, dim=1)[:, :k]  # (B, K)

    best_orders = torch.zeros(B, N, dtype=torch.int64, device=device)
    best_weights = torch.full((B,), -float('inf'), device=device)

    for kidx in range(k):
        start = start_candidates[:, kidx]  # (B,)
        visited = torch.zeros(B, N, dtype=torch.bool, device=device)
        orders = torch.zeros(B, N, dtype=torch.int64, device=device)
        cur = start

        for t in range(N):
            batch_idx = torch.arange(B, device=device)
            visited[batch_idx, cur] = True
            orders[:, t] = cur

            if t == N - 1:
                break

            W_cur = W[batch_idx, cur, :]  # (B, N)
            W_cur = W_cur.masked_fill(visited, -float('inf'))
            cur = W_cur.argmax(dim=1)  # (B,)

        # path weight
        pw = torch.zeros(B, device=device)
        for i in range(N - 1):
            pw += W[batch_idx, orders[:, i], orders[:, i + 1]]
        better = pw > best_weights
        best_weights[better] = pw[better]
        best_orders[better] = orders[better]

    return best_orders


def generate_orders_gpu(
    A_all: np.ndarray,
    method: str,
    A_global: np.ndarray = None,
    lambda_mix: float = 0.5,
    batch_size: int = 4096,
) -> np.ndarray:
    """Process all samples on GPU in batches."""
    n_seqs, n = A_all.shape[0], A_all.shape[1]
    device = torch.device('cuda')

    A_global_t = None
    if A_global is not None:
        A_global_t = torch.from_numpy(A_global).float().to(device)
        A_global_t = _symmetrize_and_zero_diag(A_global_t)

    orders_all = np.zeros((n_seqs, n), dtype=np.int16)
    t0 = time.time()

    for start in range(0, n_seqs, batch_size):
        end = min(start + batch_size, n_seqs)
        chunk = torch.from_numpy(A_all[start:end]).float().to(device)
        W_chunk = _symmetrize_and_zero_diag(chunk)  # (B, N, N)

        if method == "old_nn":
            W_final = W_chunk
        elif method == "mixed":
            lam = lambda_mix
            W_final = lam * W_chunk + (1.0 - lam) * A_global_t.unsqueeze(0)
        else:
            raise ValueError(f"GPU script only supports old_nn/mixed for now, got {method}")

        orders_t = batched_old_nn_greedy(W_final)
        orders_all[start:end] = orders_t.cpu().numpy().astype(np.int16)

        elapsed = time.time() - t0
        done = end
        rate = done / max(elapsed, 1e-9)
        eta = (n_seqs - done) / max(rate, 1e-9)
        print(f"  {done}/{n_seqs} | {rate:.1f} seq/s | ETA {eta:.0f}s",
              flush=True)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({elapsed / n_seqs * 1000:.1f} ms/seq)")
    return orders_all


def _path_weight_np(A: np.ndarray, order: np.ndarray) -> float:
    W = 0.5 * (A + A.T)
    np.fill_diagonal(W, 0.0)
    if len(order) < 2:
        return 0.0
    return float(sum(W[order[i], order[i + 1]] for i in range(len(order) - 1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-matrices", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--lambda-mix", type=float, default=0.5)
    parser.add_argument("--global-a-path", default=None)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--train-fraction", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--log-every", type=int, default=2000)
    args = parser.parse_args()

    print(f"Loading A matrices from {args.a_matrices}...")
    A_all = np.load(args.a_matrices)
    if A_all.ndim != 3 or A_all.shape[1] != A_all.shape[2]:
        raise ValueError("--a-matrices must have shape (num_seqs, N, N)")
    if args.max_samples is not None:
        A_all = A_all[:args.max_samples]
    n_seqs, n = A_all.shape[0], A_all.shape[1]
    print(f"  {n_seqs} sequences, {n}x{n}, method={args.method}")

    # Compute A_global (CPU numpy, then to GPU)
    A_global = None
    if args.method != "old_nn":
        if args.global_a_path:
            A_global = np.load(args.global_a_path)
            if A_global.ndim == 3:
                A_global = A_global.mean(axis=0)
        elif args.train_size is not None:
            A_global = A_all[:args.train_size].mean(axis=0)
        elif args.train_fraction is not None:
            train_n = max(1, int(round(n_seqs * args.train_fraction)))
            A_global = A_all[:train_n].mean(axis=0)
        else:
            A_global = A_all.mean(axis=0)
        A_global = A_global.astype(np.float32)
        print(f"  A_global: mean over {'--global-a-path' if args.global_a_path else 'input'}")

    t0 = time.time()
    orders = generate_orders_gpu(
        A_all, args.method,
        A_global=A_global,
        lambda_mix=args.lambda_mix,
        batch_size=args.batch_size,
    )
    orders_i64 = orders.astype(np.int64)

    # Path weights on CPU (fast enough)
    weights = np.array(
        [_path_weight_np(A_all[i], orders_i64[i]) for i in range(n_seqs)],
        dtype=np.float32,
    )

    # old_nn orders for comparison
    old_orders = None
    if args.method != "old_nn":
        print("Generating old_nn reference orders on GPU...")
        old_orders = generate_orders_gpu(
            A_all, "old_nn",
            batch_size=args.batch_size,
        ).astype(np.int64)

    elapsed = time.time() - t0
    print(f"  Total in {elapsed:.1f}s ({elapsed / n_seqs * 1000:.1f} ms/seq)")

    summary = evaluate_order_diagnostics(
        orders_i64,
        old_orders=None if old_orders is None else old_orders,
    )
    summary.update({
        "method": args.method,
        "lambda_mix": float(args.lambda_mix),
        "mean_path_weight": float(weights.mean()),
        "a_matrices": args.a_matrices,
        "global_a_path": args.global_a_path or "",
    })

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_kwargs = {"paths": orders, "orders": orders, "weights": weights}
    if A_global is not None:
        save_kwargs["A_global"] = A_global
    if old_orders is not None:
        save_kwargs["old_orders"] = old_orders
    np.savez(args.output, **save_kwargs)

    summary_out = args.summary_output or (os.path.splitext(args.output)[0] + ".tsv")
    write_summary_tsv(summary_out, [summary])

    print("\nDiagnostics:")
    for key in sorted(summary):
        print(f"  {key}: {summary[key]}")
    print(f"\nSaved: {args.output}")
    print(f"Saved: {summary_out}")


if __name__ == "__main__":
    main()
