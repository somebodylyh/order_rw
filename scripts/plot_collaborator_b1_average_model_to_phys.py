#!/usr/bin/env python3
"""Average B1 maps over random probe orders: model-frame then physical-frame."""

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
from per_head_order_scan import _attn_to_A_block_b1_vec  # noqa: E402
from training_utils import BLOCK_LEN, N, SEQ_LEN, load_train_chunks  # noqa: E402


def _resolve_perm_orientation(ckpt, requested: str) -> str:
    if requested != "auto":
        return requested
    convention = (ckpt.get("data_permutation") or {}).get("convention")
    if convention == "clean_phys_to_model":
        return "phys_to_model"
    return "model_to_phys"


def _attn_to_A_block_b1_model_frame(attn, reveal_tokens):
    """B1 predictor frame + remap to model-block labels, with [None] folded to block 0."""
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


def _tau_of_B(B):
    sigma, _, _ = generate_teacher_label(B, alpha_dep=0.5)
    tau, _ = kendalltau(sigma, np.arange(B.shape[0]))
    return float(tau), sigma


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="/home/admin/ych/nanogpt-learned-order/ckpt/ckpt.pt")
    p.add_argument("--out", default="reports/collaborator_ckpt_b1_scan_20260616/figures/collaborator_b1_L0H3_avg_model_to_physical_nomask.png")
    p.add_argument("--device", default="cpu")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--head", type=int, default=3)
    p.add_argument("--M", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--fwd-batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--perm-orientation", choices=["auto", "phys_to_model", "model_to_phys"], default="auto")
    args = p.parse_args()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    ckpt, model, dev = _load_model(args.ckpt, args.device)
    perm_orientation = _resolve_perm_orientation(ckpt, args.perm_orientation)
    clean_perm = _clean_perm_from_ckpt(ckpt, perm_orientation)
    inv_perm = clean_perm.inv_perm_model_to_phys.cpu().numpy()

    total = args.M * args.batch_size
    chunks_phys = load_train_chunks(n_chunks=total)
    chunks_model = _physical_chunks_to_model(chunks_phys, clean_perm)

    token_orders = torch.empty((total, SEQ_LEN), dtype=torch.long)
    for i in range(total):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(args.seed) + int(i))
        rand_blocks = torch.randperm(N, generator=gen, device="cpu")
        token_orders[i] = expand_model_blocks_to_token_order(rand_blocks.unsqueeze(0), BLOCK_LEN)[0]

    A_model_all = []
    A_phys_all = []
    model.eval()
    with torch.no_grad():
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
                A_phys_all.append(_attn_to_A_block_b1_vec(attn, reveal_tokens, inv_perm))
            print(f"  [extract avg-map] {stop}/{total}", flush=True)

    B_model = np.mean(A_model_all, axis=0).T.astype(np.float32)
    B_phys = np.mean(A_phys_all, axis=0).T.astype(np.float32)
    np.fill_diagonal(B_model, 0.0)
    np.fill_diagonal(B_phys, 0.0)

    tau_model, sigma_model = _tau_of_B(B_model)
    tau_phys, sigma_phys = _tau_of_B(B_phys)

    np.save(out.with_suffix(".model_frame.npy"), B_model)
    np.save(out.with_suffix(".physical_frame.npy"), B_phys)

    vmax = max(float(B_model.max()), float(B_phys.max())) * 0.95
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    panels = [
        (axes[0], B_model, f"Average B1 map in model frame (pi1)\nL{args.layer}H{args.head}, tau={tau_model:.3f}"),
        (axes[1], B_phys, f"Same probes remapped to original frame (pi0)\nL{args.layer}H{args.head}, tau={tau_phys:.3f}"),
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
        "B1 average over random probe orders; [None] is kept/folded to block 0",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(out, dpi=190)
    plt.close(fig)

    print(f"saved {out}")
    print(f"perm_orientation={perm_orientation}")
    print(f"model_frame_tau={tau_model:.6f} sigma_first16={sigma_model[:16].tolist()}")
    print(f"physical_frame_tau={tau_phys:.6f} sigma_first16={sigma_phys[:16].tolist()}")


if __name__ == "__main__":
    main()
