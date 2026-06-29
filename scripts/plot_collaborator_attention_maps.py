#!/usr/bin/env python3
"""Plot collaborator checkpoint attention/B maps under predictor and remapped frames."""

from __future__ import annotations

import argparse
import json
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

from scan_collaborator_ckpt_b1 import (  # noqa: E402
    _clean_perm_from_ckpt,
    _load_model,
    _physical_chunks_to_model,
)
from per_head_order_scan import (  # noqa: E402
    _batch_mean_B,
    _orders_from_graphs,
    extract_per_head_and_heavy_A,
    head_order_metrics,
)
from training_utils import load_train_chunks  # noqa: E402


def _extract_head_B(ckpt, model, dev, chunks_phys, *, orientation, none_mode, head, seed, batch_size, fwd_batch):
    clean_perm = _clean_perm_from_ckpt(ckpt, orientation)
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
    B_batch = _batch_mean_B(A, M, batch_size)
    sig = _orders_from_graphs(B_batch, alpha_dep=0.5)
    metrics = head_order_metrics(sig)
    return B_mean, metrics


def _plot_matrix(ax, B, title, subtitle, vmax):
    im = ax.imshow(B, cmap="viridis", aspect="equal", vmin=0.0, vmax=vmax, interpolation="nearest")
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    ax.set_xlabel("key/source block")
    ax.set_ylabel("query/target block")
    ax.tick_params(labelsize=7)
    return im


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="/home/admin/ych/nanogpt-learned-order/ckpt/ckpt.pt")
    p.add_argument("--out-dir", default="reports/collaborator_ckpt_b1_scan_20260616/figures")
    p.add_argument("--device", default="cpu")
    p.add_argument("--M", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--fwd-batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt, model, dev = _load_model(args.ckpt, args.device)
    chunks_phys = load_train_chunks(n_chunks=args.M * args.batch_size)

    panels = [
        {
            "name": "predictor_L1H7",
            "title": "B1/predictor raw",
            "subtitle_prefix": "L1H7, no physical remap",
            "orientation": "model_to_phys",
            "none_mode": "predictor",
            "head": (1, 7),
        },
        {
            "name": "b1_phys_correct_L0H3",
            "title": "B1 physical remap",
            "subtitle_prefix": "L0H3, correct ckpt convention",
            "orientation": "model_to_phys",
            "none_mode": "b1",
            "head": (0, 3),
        },
        {
            "name": "b0_correct_L0H3",
            "title": "B0 legacy remap",
            "subtitle_prefix": "L0H3, correct ckpt convention",
            "orientation": "model_to_phys",
            "none_mode": "b0",
            "head": (0, 3),
        },
        {
            "name": "b1_phys_wrong_L0H3",
            "title": "B1 remap, wrong convention",
            "subtitle_prefix": "L0H3, clean-protocol orientation",
            "orientation": "phys_to_model",
            "none_mode": "b1",
            "head": (0, 3),
        },
    ]

    mats = []
    summary = []
    for spec in panels:
        print(
            f"[plot] {spec['name']} none_mode={spec['none_mode']} "
            f"orientation={spec['orientation']} head={spec['head']}",
            flush=True,
        )
        B, metrics = _extract_head_B(
            ckpt,
            model,
            dev,
            chunks_phys,
            orientation=spec["orientation"],
            none_mode=spec["none_mode"],
            head=spec["head"],
            seed=args.seed,
            batch_size=args.batch_size,
            fwd_batch=args.fwd_batch,
        )
        mats.append((spec, B, metrics))
        np.save(out_dir / f"{spec['name']}.npy", B)
        summary.append(
            {
                **{k: spec[k] for k in ("name", "title", "orientation", "none_mode")},
                "head": list(spec["head"]),
                "tau_vs_l2r": metrics["tau_vs_l2r"],
                "mean_pairwise_tau": metrics["mean_pairwise_tau"],
                "first_step_entropy": metrics["first_step_entropy"],
                "npy": str(out_dir / f"{spec['name']}.npy"),
            }
        )

    vmax = max(float(B.max()) for _, B, _ in mats) * 0.95
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    axes = axes.ravel()
    im = None
    for ax, (spec, B, metrics) in zip(axes, mats):
        subtitle = (
            f"{spec['subtitle_prefix']} | tau={metrics['tau_vs_l2r']:.3f}, "
            f"pair={metrics['mean_pairwise_tau']:.3f}"
        )
        im = _plot_matrix(ax, B, spec["title"], subtitle, vmax)
    if im is not None:
        fig.colorbar(im, ax=axes.tolist(), shrink=0.78, label="B = A^T mean weight")
    fig.suptitle(
        "Collaborator ckpt attention-derived B maps (M=20, batch=8)",
        fontsize=14,
        fontweight="bold",
    )
    png = out_dir / "collaborator_ckpt_attention_maps_M20.png"
    fig.savefig(png, dpi=180)
    plt.close(fig)

    with open(out_dir / "collaborator_ckpt_attention_maps_M20_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved {png}")
    print(f"saved {out_dir / 'collaborator_ckpt_attention_maps_M20_summary.json'}")


if __name__ == "__main__":
    main()
