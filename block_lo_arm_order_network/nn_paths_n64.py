"""
Generate attention-only curriculum orders from pre-extracted A matrices.

No AO-GPT forward pass is needed. Works for N=16, N=32, N=64, or any N.

Examples:
    python nn_paths_n64.py \
      --a-matrices probe_results/A_train_n64_10k.npy \
      --method set_aware \
      --train-size 8000 \
      --output probe_results/set_aware_paths_n64.npz

For validation/test A files, pass a train-only global A:
    python nn_paths_n64.py \
      --a-matrices probe_results/A_val_n64.npy \
      --method mixed \
      --global-a-path probe_results/A_global_train_n64.npy
"""

import argparse
import os
import sys
import time

import numpy as np

_current = os.path.dirname(os.path.abspath(__file__))
if _current not in sys.path:
    sys.path.insert(0, _current)

from attention_curriculum_orders import generate_order_from_attention, old_nn_greedy
from order_diagnostics import evaluate_order_diagnostics, write_summary_tsv


METHODS = ("old_nn", "global", "mixed", "set_aware", "consensus")


def _path_weight(A: np.ndarray, order: np.ndarray) -> float:
    W = 0.5 * (A + A.T)
    np.fill_diagonal(W, 0.0)
    if len(order) < 2:
        return 0.0
    return float(sum(W[order[i], order[i + 1]] for i in range(len(order) - 1)))


def _load_train_mask(path: str, n: int) -> np.ndarray:
    raw = np.load(path)
    arr = np.asarray(raw)
    if arr.dtype == bool:
        if arr.shape != (n,):
            raise ValueError(f"boolean train mask must have shape ({n},)")
        return arr
    mask = np.zeros(n, dtype=bool)
    idx = arr.astype(np.int64).reshape(-1)
    if np.any(idx < 0) or np.any(idx >= n):
        raise ValueError("train index mask contains out-of-range values")
    mask[idx] = True
    return mask


def _compute_global_A(A_all: np.ndarray, args) -> np.ndarray:
    if args.global_a_path:
        A_global = np.load(args.global_a_path)
        if A_global.ndim == 3:
            A_global = A_global.mean(axis=0)
        if A_global.shape != A_all.shape[1:]:
            raise ValueError(
                f"A_global shape {A_global.shape} does not match A shape {A_all.shape[1:]}"
            )
        return A_global.astype(np.float32)

    n = A_all.shape[0]
    if args.train_mask:
        train_mask = _load_train_mask(args.train_mask, n)
    elif args.train_size is not None:
        if args.train_size <= 0 or args.train_size > n:
            raise ValueError("--train-size must be in [1, n_samples]")
        train_mask = np.zeros(n, dtype=bool)
        train_mask[:args.train_size] = True
    elif args.train_fraction is not None:
        if not 0.0 < args.train_fraction <= 1.0:
            raise ValueError("--train-fraction must be in (0, 1]")
        train_n = max(1, int(round(n * args.train_fraction)))
        train_mask = np.zeros(n, dtype=bool)
        train_mask[:train_n] = True
    else:
        # Backward-compatible path for files that are already train-only.
        train_mask = np.ones(n, dtype=bool)

    if not train_mask.any():
        raise ValueError("train split is empty; cannot compute A_global")
    return A_all[train_mask].mean(axis=0).astype(np.float32)


def _generate_orders(A_all: np.ndarray, args, A_global: np.ndarray = None) -> np.ndarray:
    n_seqs, n = A_all.shape[0], A_all.shape[1]
    orders = np.zeros((n_seqs, n), dtype=np.int16 if n < 32768 else np.int64)
    t0 = time.time()
    for s in range(n_seqs):
        orders[s] = generate_order_from_attention(
            A_all[s],
            method=args.method,
            A_global=A_global,
            lambda_mix=args.lambda_mix,
        ).astype(orders.dtype)
        if (s + 1) % args.log_every == 0:
            elapsed = time.time() - t0
            rate = (s + 1) / max(elapsed, 1e-9)
            eta = (n_seqs - s - 1) / max(rate, 1e-9)
            print(f"  {s + 1}/{n_seqs} | {rate:.1f} seqs/s | ETA {eta:.0f}s")
    return orders


def _summary_path(output: str, explicit_summary: str = None) -> str:
    if explicit_summary:
        return explicit_summary
    root, _ = os.path.splitext(output)
    return root + "_summary.tsv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-matrices", default="probe_results/A_train_n64_10k.npy")
    parser.add_argument("--a-matrices-ckpt2", default=None,
                        help="Optional same-sample A matrices from another checkpoint")
    parser.add_argument("--output", default="probe_results/nn_paths_n64.npz")
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--method", choices=METHODS, default="old_nn")
    parser.add_argument("--lambda-mix", type=float, default=0.5)
    parser.add_argument("--global-a-path", default=None,
                        help="Train-only A_global .npy, or stack whose mean is train-only A_global")
    parser.add_argument("--train-mask", default=None,
                        help="Boolean mask or integer indices for train rows in --a-matrices")
    parser.add_argument("--train-size", type=int, default=None,
                        help="Use first K rows of --a-matrices to compute train-only A_global")
    parser.add_argument("--train-fraction", type=float, default=None,
                        help="Use first fraction of rows to compute train-only A_global")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=500)
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
        A_global = _compute_global_A(A_all, args)
        print("  A_global: train-only source "
              f"({'--global-a-path' if args.global_a_path else 'input train split'})")

    t0 = time.time()
    orders = _generate_orders(A_all, args, A_global)
    weights = np.asarray(
        [_path_weight(A_all[i], orders[i].astype(np.int64)) for i in range(n_seqs)],
        dtype=np.float32,
    )

    old_orders = None
    if args.method != "old_nn":
        old_orders = np.zeros_like(orders)
        for i in range(n_seqs):
            old_orders[i] = old_nn_greedy(A_all[i]).astype(old_orders.dtype)

    checkpoint_orders = None
    if args.a_matrices_ckpt2:
        A_ckpt2 = np.load(args.a_matrices_ckpt2)
        if args.max_samples is not None:
            A_ckpt2 = A_ckpt2[:args.max_samples]
        if A_ckpt2.shape != A_all.shape:
            raise ValueError("--a-matrices-ckpt2 must match --a-matrices shape after slicing")
        checkpoint_orders = _generate_orders(A_ckpt2, args, A_global)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({elapsed / n_seqs * 1000:.1f} ms/seq)")

    summary = evaluate_order_diagnostics(
        orders.astype(np.int64),
        old_orders=None if old_orders is None else old_orders.astype(np.int64),
        checkpoint_orders=None if checkpoint_orders is None else checkpoint_orders.astype(np.int64),
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
        save_kwargs["A_global"] = A_global.astype(np.float32)
    if old_orders is not None:
        save_kwargs["old_orders"] = old_orders
    if checkpoint_orders is not None:
        save_kwargs["checkpoint_orders"] = checkpoint_orders
    np.savez(args.output, **save_kwargs)

    summary_out = _summary_path(args.output, args.summary_output)
    write_summary_tsv(summary_out, [summary])

    print("\nDiagnostics:")
    for key in sorted(summary):
        print(f"  {key}: {summary[key]}")
    print(f"\nSaved orders: {args.output}")
    print(f"Saved summary: {summary_out}")


if __name__ == "__main__":
    main()
