"""Diagnostics for generated orders.

This file is allowed to compare generated orders with L2R references. Order
generation code must not import this module.
"""

from collections import Counter
from typing import Dict, Optional

import numpy as np


def _kendall_tau(order_a: np.ndarray, order_b: np.ndarray) -> float:
    a = np.asarray(order_a, dtype=np.int64)
    b = np.asarray(order_b, dtype=np.int64)
    if len(a) != len(b):
        raise ValueError("orders must have the same length")
    pos_a = {int(v): i for i, v in enumerate(a)}
    pos_b = {int(v): i for i, v in enumerate(b)}
    concordant = 0
    discordant = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            x = int(a[i])
            y = int(a[j])
            if (pos_a[x] < pos_a[y]) == (pos_b[x] < pos_b[y]):
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return float((concordant - discordant) / total) if total else 0.0


def _is_valid_permutation(order: np.ndarray) -> bool:
    arr = np.asarray(order, dtype=np.int64)
    return sorted(arr.tolist()) == list(range(len(arr)))


def evaluate_order_diagnostics(
    orders: np.ndarray,
    *,
    old_orders: Optional[np.ndarray] = None,
    checkpoint_orders: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Evaluate generated orders. L2R appears only as a reference here."""
    arr = np.asarray(orders, dtype=np.int64)
    single = arr.ndim == 1
    if single:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError("orders must be a 1D or 2D array")

    n_orders, n = arr.shape
    l2r = np.arange(n, dtype=np.int64)
    valid = np.asarray([_is_valid_permutation(row) for row in arr], dtype=bool)
    tau_vs_l2r = np.asarray([_kendall_tau(row, l2r) for row in arr], dtype=np.float64)

    first_nodes = arr[:, 0]
    first_counts = Counter(first_nodes.tolist())
    probs = np.asarray(list(first_counts.values()), dtype=np.float64) / max(n_orders, 1)
    first_entropy = float(-(probs * np.log(probs + 1e-12)).sum())

    summary = {
        "n_orders": float(n_orders),
        "n_blocks": float(n),
        "valid_fraction": float(valid.mean()),
        "is_valid_permutation": bool(valid.all()) if single else float(valid.mean()),
        "tau_vs_l2r": float(tau_vs_l2r.mean()),
        "tau_vs_l2r_std": float(tau_vs_l2r.std()),
        "first_node_entropy": first_entropy,
    }

    for rank, (node, count) in enumerate(first_counts.most_common(5), start=1):
        summary[f"first_node_top{rank}"] = float(node)
        summary[f"first_node_top{rank}_freq"] = float(count / n_orders)

    if n_orders > 1:
        pair_taus = []
        for i in range(n_orders):
            for j in range(i + 1, n_orders):
                pair_taus.append(_kendall_tau(arr[i], arr[j]))
        summary["batch_order_agreement_tau"] = float(np.mean(pair_taus)) if pair_taus else 0.0
    else:
        summary["batch_order_agreement_tau"] = 0.0

    if old_orders is not None:
        old = np.asarray(old_orders, dtype=np.int64)
        if old.ndim == 1:
            old = old[None, :]
        if old.shape != arr.shape:
            raise ValueError("old_orders must match orders shape")
        overlaps = [
            len(set(arr[i, :8].tolist()) & set(old[i, :8].tolist())) / min(8, n)
            for i in range(n_orders)
        ]
        summary["top8_overlap_against_old"] = float(np.mean(overlaps))

    if checkpoint_orders is not None:
        other = np.asarray(checkpoint_orders, dtype=np.int64)
        if other.ndim == 1:
            other = other[None, :]
        if other.shape != arr.shape:
            raise ValueError("checkpoint_orders must match orders shape")
        ckpt_taus = [_kendall_tau(arr[i], other[i]) for i in range(n_orders)]
        summary["same_sample_tau_across_ckpt"] = float(np.mean(ckpt_taus))

    return summary


def write_summary_tsv(path: str, rows) -> None:
    rows = list(rows)
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(keys) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(key, "")) for key in keys) + "\n")
