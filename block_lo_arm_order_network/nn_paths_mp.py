"""Multiprocessing nn_paths — splits work across CPU cores. Fast for N≤128."""
import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np

_current = os.path.dirname(os.path.abspath(__file__))
if _current not in sys.path:
    sys.path.insert(0, _current)

from attention_curriculum_orders import generate_order_from_attention, old_nn_greedy
from order_diagnostics import evaluate_order_diagnostics, write_summary_tsv


def _path_weight_np(A: np.ndarray, order: np.ndarray) -> float:
    W = 0.5 * (A + A.T)
    np.fill_diagonal(W, 0.0)
    if len(order) < 2:
        return 0.0
    return float(sum(W[order[i], order[i + 1]] for i in range(len(order) - 1)))


def _worker_generate(args):
    """Worker function: generate orders for a chunk."""
    idx, A_chunk, method, A_global, lambda_mix = args
    n_seqs = A_chunk.shape[0]
    orders = np.zeros((n_seqs, A_chunk.shape[1]), dtype=np.int16)
    weights = np.zeros(n_seqs, dtype=np.float32)
    for s in range(n_seqs):
        order = generate_order_from_attention(
            A_chunk[s], method=method, A_global=A_global, lambda_mix=lambda_mix,
        )
        orders[s] = order.astype(np.int16)
        weights[s] = _path_weight_np(A_chunk[s], order.astype(np.int64))
    return idx, orders, weights


def _worker_old_nn(args):
    """Worker function: generate old_nn orders for a chunk."""
    idx, A_chunk = args
    n_seqs = A_chunk.shape[0]
    orders = np.zeros((n_seqs, A_chunk.shape[1]), dtype=np.int16)
    for s in range(n_seqs):
        orders[s] = old_nn_greedy(A_chunk[s]).astype(np.int16)
    return idx, orders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-matrices", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--method", choices=("old_nn", "mixed"), default="old_nn")
    parser.add_argument("--lambda-mix", type=float, default=0.5)
    parser.add_argument("--global-a-path", default=None)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--train-fraction", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    print(f"Loading A matrices from {args.a_matrices}...")
    A_all = np.load(args.a_matrices)
    if A_all.ndim != 3 or A_all.shape[1] != A_all.shape[2]:
        raise ValueError("--a-matrices must have shape (num_seqs, N, N)")
    if args.max_samples is not None:
        A_all = A_all[:args.max_samples]
    n_seqs, n = A_all.shape[0], A_all.shape[1]
    print(f"  {n_seqs} sequences, {n}x{n}, method={args.method}")

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
        print(f"  A_global: {'--global-a-path' if args.global_a_path else 'input'}")

    n_workers = args.workers if args.workers > 0 else min(mp.cpu_count(), 16)
    chunk_size = max(1, n_seqs // n_workers)
    chunks = []
    for i, start in enumerate(range(0, n_seqs, chunk_size)):
        end = min(start + chunk_size, n_seqs)
        chunks.append((i, A_all[start:end], args.method, A_global, args.lambda_mix))

    print(f"  {n_workers} workers, ~{chunk_size} samples each")

    t0 = time.time()
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(_worker_generate, chunks)

    orders = np.zeros((n_seqs, n), dtype=np.int16)
    weights = np.zeros(n_seqs, dtype=np.float32)
    for idx, o, w in results:
        orders[idx * chunk_size: idx * chunk_size + len(o)] = o
        weights[idx * chunk_size: idx * chunk_size + len(w)] = w

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({elapsed / n_seqs * 1000:.1f} ms/seq)")

    orders_i64 = orders.astype(np.int64)

    old_orders = None
    if args.method != "old_nn":
        print("Generating old_nn reference orders...")
        t1 = time.time()
        old_chunks = [(i, A_all[start:start + chunk_size])
                      for i, start in enumerate(range(0, n_seqs, chunk_size))]
        with mp.Pool(processes=n_workers) as pool:
            old_results = pool.map(_worker_old_nn, old_chunks)
        old_orders = np.zeros((n_seqs, n), dtype=np.int16)
        for idx, o in old_results:
            old_orders[idx * chunk_size: idx * chunk_size + len(o)] = o
        print(f"  old_nn done in {time.time() - t1:.1f}s")

    summary = evaluate_order_diagnostics(
        orders_i64,
        old_orders=None if old_orders is None else old_orders.astype(np.int64),
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
