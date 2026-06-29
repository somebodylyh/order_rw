#!/usr/bin/env python3
"""B1 map for one fixed model-fed order, shown before/after axis relabeling."""

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

from clean_training_protocol import expand_model_blocks_to_token_order  # noqa: E402
from scan_collaborator_ckpt_b1 import _clean_perm_from_ckpt, _load_model, _physical_chunks_to_model  # noqa: E402
from per_head_order_scan import _attn_to_A_block_b1_vec  # noqa: E402
from training_utils import BLOCK_LEN, N, load_train_chunks  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="/home/admin/ych/nanogpt-learned-order/ckpt/ckpt.pt")
    p.add_argument("--out", default="reports/collaborator_ckpt_b1_scan_20260616/figures/collaborator_b1_L0H3_fixed_order_axis_relabel.png")
    p.add_argument("--device", default="cpu")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--head", type=int, default=3)
    p.add_argument("--n-chunks", type=int, default=64)
    p.add_argument("--fwd-batch", type=int, default=8)
    p.add_argument("--order-seed", type=int, default=0)
    args = p.parse_args()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    ckpt, model, dev = _load_model(args.ckpt, args.device)
    clean_perm = _clean_perm_from_ckpt(ckpt, "model_to_phys")
    inv_perm = clean_perm.inv_perm_model_to_phys.cpu().numpy()

    chunks_phys = load_train_chunks(n_chunks=args.n_chunks)
    chunks_model = _physical_chunks_to_model(chunks_phys, clean_perm)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.order_seed)
    model_order_blocks = torch.randperm(N, generator=gen, device="cpu")
    token_order_1 = expand_model_blocks_to_token_order(model_order_blocks.view(1, -1), BLOCK_LEN)
    token_order = token_order_1.expand(args.n_chunks, -1).contiguous()
    reveal_tokens = token_order_1[0].numpy()
    phys_by_reveal = inv_perm[model_order_blocks.numpy()]

    Bs_phys = []
    model.eval()
    with torch.no_grad():
        for start in range(0, args.n_chunks, args.fwd_batch):
            stop = min(start + args.fwd_batch, args.n_chunks)
            tokens = chunks_model[start:stop].to(dev)
            orders = token_order[start:stop].to(dev)
            _, _, attn_list = model.forward_fn(tokens, orders, return_attentions=True)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            attn_batch = torch.stack(attn_list).cpu().numpy()  # (L,B,H,257,257)
            for bi in range(stop - start):
                attn = attn_batch[args.layer, bi, args.head]
                A_phys = _attn_to_A_block_b1_vec(attn, reveal_tokens, inv_perm)
                B_phys = A_phys.T.astype(np.float32)
                np.fill_diagonal(B_phys, 0.0)
                Bs_phys.append(B_phys)
            print(f"  [extract fixed-order] {stop}/{args.n_chunks}", flush=True)

    B_phys_mean = np.mean(Bs_phys, axis=0).astype(np.float32)
    np.fill_diagonal(B_phys_mean, 0.0)
    B_model_order = B_phys_mean[np.ix_(phys_by_reveal, phys_by_reveal)]

    np.save(out.with_suffix(".model_order.npy"), B_model_order)
    np.save(out.with_suffix(".physical_order.npy"), B_phys_mean)

    vmax = max(float(B_model_order.max()), float(B_phys_mean.max())) * 0.95
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    panels = [
        (axes[0], B_model_order, "Axes in the shuffled order fed to the model"),
        (axes[1], B_phys_mean, "Same B matrix, axes relabeled to original order"),
    ]
    im = None
    for ax, B, title in panels:
        im = ax.imshow(B, cmap="viridis", aspect="equal", vmin=0.0, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("source/key block")
        ax.set_ylabel("target/query block")
        ax.tick_params(labelsize=7)
    fig.colorbar(im, ax=axes.tolist(), shrink=0.86, label="B = A^T mean weight")
    fig.suptitle(
        f"B1 L{args.layer}H{args.head}: fixed shuffled-order forward; right panel is only axis relabeling",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(out, dpi=190)
    plt.close(fig)

    print(f"saved {out}")
    print("The two panels contain the same values up to row/column permutation.")
    print("model_order_first16", model_order_blocks[:16].tolist())
    print("physical_labels_for_model_order_first16", phys_by_reveal[:16].tolist())


if __name__ == "__main__":
    main()
