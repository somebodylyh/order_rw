#!/usr/bin/env python3
"""Frozen-checkpoint order-sensitivity diagnostic.

Evaluates a single frozen AO-GPT checkpoint under multiple block-order policies
and reports token-averaged CE + Kendall tau vs reference orders (L2R, RW-base).

Policies:
  1. ori_l2r
  2. random (N seeds averaged separately)
  3. RW top_k=4, epsilon=0        (base RW)
  4. RW epsilon=0.15
  5. RW epsilon=0.20
  6. RW top_k=8
  7. model_order / ON order (if available)
"""

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from directed_graph_policy import build_directed_graph, sample_order
from eval_fixed_order_protocol import (
    EvalBlockOrder,
    block_orders_to_token_orders,
    evaluate_order_matrices,
    load_aogpt_any,
    make_eval_indices,
    original_l2r_block_order,
    physical_orders_to_model_orders,
)
from order_diagnostics import _kendall_tau
from train_aogpt_graph_rw import (
    A_PATH_DEFAULT,
    AO_GPT_CKPT,
    BLOCK_LEN,
    N,
    load_train_chunks,
    phys_to_model_idx,
)


def _kendall_tau_batch(orders_a, orders_b):
    """Mean Kendall tau between two (B, N) order matrices."""
    taus = []
    for i in range(orders_a.shape[0]):
        taus.append(_kendall_tau(orders_a[i].numpy(), orders_b[i].numpy()))
    return float(np.mean(taus)), float(np.std(taus))


def build_rw_orders(num_eval, graph, params, block_perm, base_seed=42):
    matrices = {}
    for label, rw_params in params.items():
        rows = []
        for seq_idx in range(num_eval):
            seed = base_seed * 10000 + seq_idx
            order, _ = sample_order(graph, "progressive_rw", rw_params, seed=seed)
            rows.append(order)
        phys = torch.tensor(np.stack(rows), dtype=torch.long)
        matrices[label] = physical_orders_to_model_orders(phys, block_perm)
    return matrices


def main():
    parser = argparse.ArgumentParser(
        description="Frozen-checkpoint order-sensitivity diagnostic"
    )
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--permutation-ckpt", required=True)
    parser.add_argument("--a-path", default=A_PATH_DEFAULT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", default="order_sensitivity")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--max-eval-seqs", type=int, default=200)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--random-seeds", type=int, default=10)
    parser.add_argument("--tau-start", type=float, default=0.1)
    parser.add_argument("--tau-step", type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.ckpt}", flush=True)
    model, block_perm, inv_perm = load_aogpt_any(
        args.ckpt, args.permutation_ckpt, device
    )

    if not torch.equal(inv_perm[block_perm], torch.arange(block_perm.numel())):
        raise RuntimeError("invalid checkpoint permutation")

    print("Loading eval chunks...", flush=True)
    idx_phys = load_train_chunks(n_chunks=None)
    eval_indices, full_val_size = make_eval_indices(
        idx_phys.size(0), args.seed, args.val_fraction, args.max_eval_seqs
    )
    idx_model = phys_to_model_idx(idx_phys[eval_indices], block_perm).to(device)

    print(f"Eval chunks: {len(eval_indices)} / {full_val_size}", flush=True)

    # Build graph
    A_all = np.load(args.a_path)
    if A_all.ndim == 3:
        A_global = A_all.mean(axis=0).astype(np.float32)
    elif A_all.ndim == 2:
        A_global = A_all.astype(np.float32)
    else:
        raise ValueError(f"Unexpected A matrix shape: {A_all.shape}")
    np.fill_diagonal(A_global, 0.0)
    graph = build_directed_graph(A_global)

    base_rw_params = {
        "tau_start": args.tau_start,
        "tau_step": args.tau_step,
        "alpha_dep": 0.5,
        "alpha_pr": 0.85,
        "beta_sup": 1.0,
        "beta_fut": 0.5,
        "beta_src": 0.2,
        "beta_loc": 0.5,
    }

    # ── Build all order sets ────────────────────────────────────────────
    num_eval = len(eval_indices)

    # L2R (physical ascending)
    ori = original_l2r_block_order(block_perm)
    l2r_order = ori.model.unsqueeze(0).expand(num_eval, -1)  # (B, N)

    # Random orders
    random_orders = {}
    rng = np.random.RandomState(args.seed)
    for s in range(args.random_seeds):
        seed = rng.randint(0, 2**31)
        rows = []
        for seq_idx in range(num_eval):
            rows.append(
                np.random.default_rng(seed * 10000 + seq_idx).permutation(N)
            )
        phys = torch.tensor(np.stack(rows), dtype=torch.long)
        random_orders[f"seed_{s}"] = physical_orders_to_model_orders(phys, block_perm)

    # RW variants
    rw_configs = OrderedDict({
        "rw_topk4_eps0": dict(base_rw_params, top_k=4, epsilon_uniform=0.0),
        "rw_eps015": dict(base_rw_params, top_k=4, epsilon_uniform=0.15),
        "rw_eps020": dict(base_rw_params, top_k=4, epsilon_uniform=0.20),
        "rw_topk8": dict(base_rw_params, top_k=8, epsilon_uniform=0.0),
    })
    rw_orders = build_rw_orders(
        num_eval, graph, rw_configs, block_perm,
    )

    # ── Evaluate all orders ─────────────────────────────────────────────
    def eval_single(orders_batch):
        """orders_batch: (B, N) tensor of model-block orders. Evaluate as one batch."""
        return evaluate_order_matrices(
            model, idx_model, [orders_batch], device, args.eval_batch_size
        )

    def eval_multi_seed(orders_dict):
        """Average CE over seeds."""
        ce_values = []
        for label, orders_batch in orders_dict.items():
            metrics = eval_single(orders_batch)
            ce_values.append(metrics["loss_token_avg"])
        return float(np.mean(ce_values)), float(np.std(ce_values))

    results = OrderedDict()
    tau_results = OrderedDict()

    # 1. ori_l2r
    print("Evaluating ori_l2r...", flush=True)
    m = eval_single(l2r_order)
    results["ori_l2r"] = m["loss_token_avg"]
    # tau(l2r, l2r) = 1.0, tau(l2r, rw_base) computed below

    # 2. random (avg over seeds)
    print(f"Evaluating random ({args.random_seeds} seeds)...", flush=True)
    rand_mean, rand_std = eval_multi_seed(random_orders)
    results["random"] = rand_mean
    results["random_std"] = rand_std

    # 3-6. RW variants
    for label, orders_batch in rw_orders.items():
        print(f"Evaluating {label}...", flush=True)
        m = eval_single(orders_batch)
        results[label] = m["loss_token_avg"]

    # ── Compute Kendall tau ─────────────────────────────────────────────
    print("Computing Kendall tau...", flush=True)

    # Reference orders: L2R and RW base (topk4_eps0)
    l2r_np = l2r_order.cpu()
    rw_base_np = rw_orders["rw_topk4_eps0"].cpu()

    # Tau vs L2R
    tau_vs_l2r = {}
    tau_vs_l2r["ori_l2r"] = (1.0, 0.0)

    for s in range(args.random_seeds):
        t, _ = _kendall_tau_batch(random_orders[f"seed_{s}"], l2r_np)
        tau_vs_l2r.setdefault("random", []).append(t)

    for label, orders_batch in rw_orders.items():
        t_mean, t_std = _kendall_tau_batch(orders_batch.cpu(), l2r_np)
        tau_vs_l2r[label] = (t_mean, t_std)

    # Tau vs RW base
    tau_vs_rw = {}
    tau_vs_rw["rw_topk4_eps0"] = (1.0, 0.0)

    for s in range(args.random_seeds):
        t, _ = _kendall_tau_batch(random_orders[f"seed_{s}"], rw_base_np)
        tau_vs_rw.setdefault("random", []).append(t)

    for label, orders_batch in rw_orders.items():
        if label == "rw_topk4_eps0":
            continue
        t_mean, t_std = _kendall_tau_batch(orders_batch.cpu(), rw_base_np)
        tau_vs_rw[label] = (t_mean, t_std)

    # Tau vs L2R for RW base (already computed above)
    tau_vs_rw["ori_l2r"] = tau_vs_l2r["rw_topk4_eps0"]

    # Aggregate random tau
    tau_vs_l2r["random"] = (float(np.mean(tau_vs_l2r["random"])),
                            float(np.std(tau_vs_l2r["random"])))
    tau_vs_rw["random"] = (float(np.mean(tau_vs_rw["random"])),
                           float(np.std(tau_vs_rw["random"])))

    # ── Report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("Order Sensitivity Diagnostic Results")
    print("=" * 72)
    print(f"{'Order':<20s} {'NLL':>8s}  {'τ(L2R)':>12s}  {'τ(RW)':>12s}")
    print("-" * 56)

    for label in results:
        nll = results[label]
        nll_str = f"{nll:.4f}"
        if isinstance(nll, float):
            pass
        else:
            nll_str = f"{nll:.4f}"

        t_l2r = tau_vs_l2r.get(label, (float("nan"), 0.0))
        t_rw = tau_vs_rw.get(label, (float("nan"), 0.0))
        t_l2r_str = f"{t_l2r[0]:.4f}±{t_l2r[1]:.4f}" if not np.isnan(t_l2r[0]) else "N/A"
        t_rw_str = f"{t_rw[0]:.4f}±{t_rw[1]:.4f}" if not np.isnan(t_rw[0]) else "N/A"

        print(f"{label:<20s} {nll_str:>8s}  {t_l2r_str:>12s}  {t_rw_str:>12s}")

    # Sort by NLL
    sorted_by_nll = sorted(
        [(k, v) for k, v in results.items() if isinstance(v, float)],
        key=lambda x: x[1],
    )
    print(f"\nBest NLL: {sorted_by_nll[0][0]} = {sorted_by_nll[0][1]:.4f}")
    print(f"Worst NLL: {sorted_by_nll[-1][0]} = {sorted_by_nll[-1][1]:.4f}")
    ce_spread = sorted_by_nll[-1][1] - sorted_by_nll[0][1]
    print(f"CE spread: {ce_spread:.4f}")
    print(f"ori_l2r vs random delta: {results['ori_l2r'] - results['random']:+.4f}")

    # ── Save ────────────────────────────────────────────────────────────
    payload = {
        "metadata": {
            "label": args.label,
            "ckpt": os.path.abspath(args.ckpt),
            "permutation_ckpt": os.path.abspath(args.permutation_ckpt),
            "a_path": os.path.abspath(args.a_path),
            "num_eval_chunks": int(len(eval_indices)),
            "max_eval_seqs": args.max_eval_seqs,
            "random_seeds": args.random_seeds,
            "val_fraction": args.val_fraction,
        },
        "results": {
            "nll": {k: float(v) if isinstance(v, float) else v for k, v in results.items()},
            "tau_vs_l2r": {
                k: {"mean": float(v[0]), "std": float(v[1])}
                for k, v in tau_vs_l2r.items()
            },
            "tau_vs_rw": {
                k: {"mean": float(v[0]), "std": float(v[1])}
                for k, v in tau_vs_rw.items()
            },
        },
    }

    json_path = output_dir / f"{args.label}.json"
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Also save compact TSV
    tsv_path = output_dir / f"{args.label}.tsv"
    with tsv_path.open("w") as f:
        f.write("order\tNLL\ttau_vs_L2R_mean\ttau_vs_L2R_std\ttau_vs_RW_mean\ttau_vs_RW_std\n")
        for label in results:
            nll = results[label]
            t_l2r = tau_vs_l2r.get(label, (float("nan"), float("nan")))
            t_rw = tau_vs_rw.get(label, (float("nan"), float("nan")))
            f.write(f"{label}\t{nll:.6f}\t{t_l2r[0]:.6f}\t{t_l2r[1]:.6f}\t{t_rw[0]:.6f}\t{t_rw[1]:.6f}\n")
    print(f"Saved: {tsv_path}")


if __name__ == "__main__":
    main()
