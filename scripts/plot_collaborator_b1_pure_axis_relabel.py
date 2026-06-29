#!/usr/bin/env python3
"""Pure axis relabel for B1 model-frame attention map.

Left: average B in model-frame coordinates pi1.
Right: the exact same averaged B matrix, row/column permuted to physical pi0.

Important: this script does NOT re-aggregate or re-fold [None] after remap.
The [None] sink is kept wherever model block 0 maps under pi1 -> pi0.
"""

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
from scipy.stats import kendalltau

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = ROOT / "block_lo_arm_order_network"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(PKG))

from clean_training_protocol import expand_model_blocks_to_token_order  # noqa: E402
from scan_collaborator_ckpt_b1 import _clean_perm_from_ckpt, _load_model, _physical_chunks_to_model  # noqa: E402
from neural_readout.teacher_labels import generate_teacher_label  # noqa: E402
from training_utils import BLOCK_LEN, N, SEQ_LEN, load_train_chunks  # noqa: E402


def _attn_to_A_block_b1_model_frame(attn, reveal_tokens):
    """B1 predictor frame + remap to model-block labels, with [None] folded to model block 0."""
    attn = np.asarray(attn, dtype=np.float64)
    a = attn.reshape(SEQ_LEN + 1, SEQ_LEN + 1)[:-1, :-1]
    reveal_tokens = np.asarray(reveal_tokens, dtype=np.int64)
    labels = np.empty(SEQ_LEN, dtype=np.int64)
    labels[0] = 0
    labels[1:] = reveal_tokens[:-1] // BLOCK_LEN
    counts = np.bincount(labels, minlength=N).astype(np.float64)
    S = np.zeros((N, SEQ_LEN), dtype=np.float64)
    S[labels, np.arange(SEQ_LEN)] = 1.0
    S = S / counts[:, None]
    A = np.einsum("bt,tu,cu->bc", S, a, S, optimize=True).astype(np.float32)
    np.fill_diagonal(A, 0.0)
    return A


def _tau(B):
    sigma, _, _ = generate_teacher_label(B, alpha_dep=0.5)
    t, _ = kendalltau(sigma, np.arange(B.shape[0]))
    return float(t), sigma


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="/home/admin/ych/nanogpt-learned-order/ckpt/ckpt.pt")
    p.add_argument("--out", default="reports/collaborator_ckpt_b1_scan_20260616/figures/collaborator_b1_L0H3_pure_axis_relabel.png")
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
    clean_perm = _clean_perm_from_ckpt(ckpt, "model_to_phys")
    model_to_phys = clean_perm.inv_perm_model_to_phys.cpu().numpy()
    phys_to_model = clean_perm.block_perm_phys_to_model.cpu().numpy()

    total = args.M * args.batch_size
    chunks_phys = load_train_chunks(n_chunks=total)
    chunks_model = _physical_chunks_to_model(chunks_phys, clean_perm)

    A_model_all = []
    model.eval()
    with torch.no_grad():
        token_orders = torch.empty((total, SEQ_LEN), dtype=torch.long)
        for i in range(total):
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(args.seed) + int(i))
            rand_blocks = torch.randperm(N, generator=gen, device="cpu")
            token_orders[i] = expand_model_blocks_to_token_order(rand_blocks.unsqueeze(0), BLOCK_LEN)[0]

        for start in range(0, total, args.fwd_batch):
            stop = min(start + args.fwd_batch, total)
            tokens = chunks_model[start:stop].to(dev)
            orders = token_orders[start:stop].to(dev)
            _, _, attn_list = model.forward_fn(tokens, orders, return_attentions=True)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            attn_batch = torch.stack(attn_list).cpu().numpy()
            for bi in range(stop - start):
                i = start + bi
                attn = attn_batch[args.layer, bi, args.head]
                reveal_tokens = token_orders[i].numpy()
                A_model_all.append(_attn_to_A_block_b1_model_frame(attn, reveal_tokens))
            print(f"  [pure relabel] {stop}/{total}", flush=True)

    B_model = np.mean(A_model_all, axis=0).T.astype(np.float32)
    np.fill_diagonal(B_model, 0.0)

    # Pure relabel: physical-axis matrix is just the same model-axis matrix
    # ordered by phys_to_model. No re-folding of [None] to physical block 0.
    B_phys_relabel = B_model[np.ix_(phys_to_model, phys_to_model)]

    tau_model, sigma_model = _tau(B_model)
    tau_phys, sigma_phys = _tau(B_phys_relabel)

    np.save(out.with_suffix(".model_frame.npy"), B_model)
    np.save(out.with_suffix(".physical_axis_relabel.npy"), B_phys_relabel)

    sink_phys = int(model_to_phys[0])
    vmax = max(float(B_model.max()), float(B_phys_relabel.max())) * 0.95
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    panels = [
        (axes[0], B_model, f"pi1/model-frame axes\nL{args.layer}H{args.head}, tau={tau_model:.3f}; [None]->model block 0"),
        (axes[1], B_phys_relabel, f"same values, pi0/physical axes\nL{args.layer}H{args.head}, tau={tau_phys:.3f}; sink at physical block {sink_phys}"),
    ]
    im = None
    for ax, B, title in panels:
        im = ax.imshow(B, cmap="viridis", aspect="equal", vmin=0.0, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("source/key block")
        ax.set_ylabel("target/query block")
        ax.tick_params(labelsize=7)
    fig.colorbar(im, ax=axes.tolist(), shrink=0.86, label="B = A^T mean weight")
    fig.suptitle(
        "B1 average map: right panel is a pure row/column relabel of the left panel",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(out, dpi=190)
    plt.close(fig)

    print(f"saved {out}")
    print(f"model_to_phys[0]={sink_phys}")
    print(f"model_frame_tau={tau_model:.6f} sigma_first16={sigma_model[:16].tolist()}")
    print(f"physical_axis_relabel_tau={tau_phys:.6f} sigma_first16={sigma_phys[:16].tolist()}")


if __name__ == "__main__":
    main()
