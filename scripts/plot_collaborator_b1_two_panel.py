#!/usr/bin/env python3
"""Two-panel B1 attention map for collaborator ckpt: before/after remap."""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = ROOT / "block_lo_arm_order_network"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(PKG))

from scan_collaborator_ckpt_b1 import _clean_perm_from_ckpt, _load_model, _physical_chunks_to_model  # noqa: E402
from per_head_order_scan import (  # noqa: E402
    _batch_mean_B,
    _orders_from_graphs,
    extract_per_head_and_heavy_A,
    head_order_metrics,
)
from training_utils import load_train_chunks  # noqa: E402


def _head_B_and_metrics(ckpt, model, dev, chunks_phys, *, none_mode, head, seed, batch_size, fwd_batch):
    clean_perm = _clean_perm_from_ckpt(ckpt, "model_to_phys")
    chunks_model = _physical_chunks_to_model(chunks_phys, clean_perm)
    A_lh, _ = extract_per_head_and_heavy_A(
        model,
        chunks_model,
        clean_perm,
        dev,
        seed=seed,
        n_top=4,
        fwd_batch=fwd_batch,
        none_mode=none_mode,
        head=head,
    )
    l, h = head
    A = A_lh[:, l, h]
    B_mean = A.mean(axis=0).T.astype(np.float32)
    np.fill_diagonal(B_mean, 0.0)
    M = len(chunks_phys) // batch_size
    sigmas = _orders_from_graphs(_batch_mean_B(A, M, batch_size), alpha_dep=0.5)
    return B_mean, head_order_metrics(sigmas)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="/home/admin/ych/nanogpt-learned-order/ckpt/ckpt.pt")
    p.add_argument("--out", default="reports/collaborator_ckpt_b1_scan_20260616/figures/collaborator_b1_L0H3_same_forward_axes_relabel.png")
    p.add_argument("--device", default="cpu")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--head", type=int, default=3)
    p.add_argument("--M", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--fwd-batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    ckpt, model, dev = _load_model(args.ckpt, args.device)
    chunks_phys = load_train_chunks(n_chunks=args.M * args.batch_size)
    head = (args.layer, args.head)

    B_raw, m_raw = _head_B_and_metrics(
        ckpt, model, dev, chunks_phys,
        none_mode="predictor", head=head, seed=args.seed,
        batch_size=args.batch_size, fwd_batch=args.fwd_batch,
    )
    B_phys, m_phys = _head_B_and_metrics(
        ckpt, model, dev, chunks_phys,
        none_mode="b1", head=head, seed=args.seed,
        batch_size=args.batch_size, fwd_batch=args.fwd_batch,
    )

    np.save(out.with_suffix(".raw_predictor.npy"), B_raw)
    np.save(out.with_suffix(".physical_remap.npy"), B_phys)

    vmax = max(float(B_raw.max()), float(B_phys.max())) * 0.95
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    panels = [
        (axes[0], B_raw, "Model-fed reveal frame", m_raw),
        (axes[1], B_phys, "Same attention, axes relabeled to original blocks", m_phys),
    ]
    im = None
    for ax, B, title, metrics in panels:
        im = ax.imshow(B, cmap="viridis", aspect="equal", vmin=0.0, vmax=vmax, interpolation="nearest")
        ax.set_title(f"{title}\nL{args.layer}H{args.head} tau={metrics['tau_vs_l2r']:.3f}", fontsize=11)
        ax.set_xlabel("source/key block")
        ax.set_ylabel("target/query block")
        ax.tick_params(labelsize=7)
    fig.colorbar(im, ax=axes.tolist(), shrink=0.86, label="B = A^T mean weight")
    fig.suptitle(
        "B1 map from the same shuffled-order forward pass; remap only relabels axes",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(out, dpi=190)
    plt.close(fig)
    print(f"saved {out}")
    print(f"before_remap tau={m_raw['tau_vs_l2r']:.6f} pair={m_raw['mean_pairwise_tau']:.6f}")
    print(f"after_remap tau={m_phys['tau_vs_l2r']:.6f} pair={m_phys['mean_pairwise_tau']:.6f}")


if __name__ == "__main__":
    main()
