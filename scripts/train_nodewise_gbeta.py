#!/usr/bin/env python3
"""Reproducible single-head NodewiseReadout gβ producer (Stage-1, canonical).

Fixes the previously-inline recipe that produced
``reports/uniform_label_free_v1/nodewise_K1000.pt`` into a standalone script:

  1. (optional) label-free head selection via uniform_label_free_v1 Stage A
     (split-half + reveal-invariance); default uses the selected head L1H7.
  2. extract per-sample single-head strict65 B65 from the 10k ckpt.
  3. bootstrap K groups of m samples → batch-mean B65 + CDL teacher (soft
     pairwise Y from the CDL rollout order).
  4. train NodewiseReadout(N=64, d_model=64, n_layers=2, n_heads=4) with
     pairwise BCE, select by val pairwise_accuracy.
  5. save ``g_beta_best.pt`` in FrozenBetaHook/HookOrderProvider format
     ({"model": state_dict, "config": {...}}) + a provenance sidecar.

The dataset-building and training math are factored into pure functions so they
are unit-testable without a checkpoint; the full pipeline is exercised by a
``@slow`` reproduce test against the real 10k ckpt.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "block_lo_arm_order_network"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from batch_readout.label_free_cdl_teacher import (  # noqa: E402
    cdl_rollout_with_standardized_margin, order_to_rank, soft_pairwise_teacher,
)
from batch_readout.model import NodewiseReadout  # noqa: E402
from batch_readout.soft_pairwise import (  # noqa: E402
    soft_pairwise_bce_loss, pairwise_accuracy,
)

DEFAULT_SOURCE_CKPT = str(
    ROOT / "block_lo_arm_order_network/probe_results/"
    "overnight_20260625_random_baseline/ckpt_step10000.pt"
)
DEFAULT_HEAD = (1, 7)
NODEWISE_CONFIG = {"model_name": "nodewise", "N": 64,
                   "d_model": 64, "n_layers": 2, "n_heads": 4}


# ── pure, unit-testable core ────────────────────────────────────────────────

def bootstrap_cdl_dataset(B65_per_sample, K, m, seed=0, mode="C-D+L"):
    """Bootstrap K groups of m samples → batch-mean B65 → CDL teacher.

    Args:
        B65_per_sample: (total, 65, 65) per-sample single-head strict65 graphs.
        K: number of bootstrap groups.
        m: samples per group (drawn with replacement).
        seed: RNG seed.

    Returns:
        B_content: (K, 64, 64) float32 — batch-mean B with None node stripped
                   (NodewiseReadout input).
        Y_pair:    (K, 64, 64) float32 — CDL soft pairwise teacher
                   (Y[i,j] = 1 iff block i revealed before j; diag 0.5).
    """
    B = np.asarray(B65_per_sample, dtype=np.float64)
    if B.ndim != 3 or B.shape[1:] != (65, 65):
        raise ValueError(f"B65_per_sample must be (total, 65, 65), got {B.shape}")
    total = B.shape[0]
    rng = np.random.default_rng(seed)

    B_content = np.zeros((K, 64, 64), dtype=np.float32)
    Y_pair = np.zeros((K, 64, 64), dtype=np.float32)
    for k in range(K):
        idx = rng.integers(0, total, size=m)          # bootstrap (with replacement)
        Bmean = B[idx].mean(axis=0)                    # (65, 65)
        np.fill_diagonal(Bmean, 0.0)
        cdl = cdl_rollout_with_standardized_margin(Bmean, mode=mode)
        rank = order_to_rank(cdl["content_order"])     # (64,)
        Y_pair[k] = soft_pairwise_teacher(rank[None, :], np.array([1.0]))
        B_content[k] = Bmean[1:, 1:].astype(np.float32)
    return B_content, Y_pair


def train_readout_on_dataset(B_content, Y_pair, *, epochs=40, lr=3e-4,
                             batch_size=64, seed=0, device="cpu", val_frac=0.1):
    """Train a NodewiseReadout on (B_content, Y_pair) with pairwise BCE.

    Selects the epoch with the best validation ``pairwise_accuracy``.
    Returns (model, best_val_acc, history).
    """
    torch.manual_seed(seed)
    dev = torch.device(device)
    B = torch.from_numpy(np.asarray(B_content, np.float32)).to(dev)
    Y = torch.from_numpy(np.asarray(Y_pair, np.float32)).to(dev)
    n = B.shape[0]
    n_val = max(1, int(n * val_frac))
    perm = np.random.default_rng(seed).permutation(n)
    val_idx = torch.as_tensor(perm[:n_val], dtype=torch.long, device=dev)
    tr_idx = torch.as_tensor(perm[n_val:], dtype=torch.long, device=dev)

    model = NodewiseReadout(N=64, d_model=64, n_layers=2, n_heads=4).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    best = {"acc": -1.0, "state": None}
    history = []
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        model.train()
        order = tr_idx[torch.as_tensor(rng.permutation(len(tr_idx)), device=dev)]
        losses = []
        for s in range(0, len(order), batch_size):
            bi = order[s:s + batch_size]
            scores = model(B[bi])
            loss = soft_pairwise_bce_loss(scores, Y[bi])
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            val_acc = pairwise_accuracy(model(B[val_idx]), Y[val_idx])
        history.append({"epoch": ep, "train_loss": float(np.mean(losses)),
                        "val_pairwise_acc": float(val_acc)})
        if val_acc > best["acc"]:
            best = {"acc": float(val_acc),
                    "state": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}}
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, best["acc"], history


def save_gbeta(model, out_path):
    """Write a NodewiseReadout in FrozenBetaHook/HookOrderProvider format."""
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                "config": dict(NODEWISE_CONFIG)}, out_path)
    return str(out_path)


# ── extraction (needs the model; covered by the @slow reproduce test) ────────

def extract_single_head_B65(model, chunks, total, layer, head, seed, dev, n_reveal=1):
    """Per-sample single-head strict65 B65 (total, 65, 65), averaged over
    n_reveal random probe orders (matches the deployed extraction convention)."""
    from analyses.uniform_label_free_v1 import _random_probe_orders, _A_model_vec
    from none_separated_block_graph import build_none_separated_B

    B_sum = np.zeros((total, 65, 65), dtype=np.float64)
    fwd = 32
    for ri in range(n_reveal):
        for start in range(0, total, fwd):
            end = min(start + fwd, total)
            bs = end - start
            probe = _random_probe_orders(bs, seed * 1000 + ri * 100 + start, dev)
            model.eval()
            with torch.no_grad():
                _, _, attn_list = model.forward_fn(
                    chunks[start:end].to(dev), probe, return_attentions=True)
            attn = attn_list[layer][:, head].cpu().numpy()
            probe_np = probe.cpu().numpy()
            for bi in range(bs):
                A65 = _A_model_vec(attn[bi], probe_np[bi])
                B_sum[start + bi] += build_none_separated_B(A65)
            del attn_list
    return (B_sum / max(1, n_reveal)).astype(np.float32)


def train_nodewise_gbeta(ckpt_path=DEFAULT_SOURCE_CKPT, *, out_dir, head=DEFAULT_HEAD,
                         M=2000, n_reveal=1, K=1000, m=8, epochs=40, lr=3e-4,
                         batch_size=64, seed=123, device="cpu", select_head=False):
    """Full single-head NodewiseReadout gβ producer. Returns g_beta_best.pt path."""
    from neural_readout.extract_b import _load_model_and_chunks

    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    model, chunks, clean_perm, dev, _ci = _load_model_and_chunks(
        ckpt_path, M, seed, device, "train")

    if select_head:
        from analyses.uniform_label_free_v1 import stage_a_uniform_discovery
        sel = stage_a_uniform_discovery(ckpt_path, M=min(M, 200), seed=seed,
                                        device=device)
        head = tuple(sel["selected_head"]) if isinstance(sel, dict) and \
            "selected_head" in sel else head

    layer, h = int(head[0]), int(head[1])
    B65 = extract_single_head_B65(model, chunks, M, layer, h, seed, dev, n_reveal)
    B_content, Y_pair = bootstrap_cdl_dataset(B65, K=K, m=m, seed=seed)
    readout, val_acc, history = train_readout_on_dataset(
        B_content, Y_pair, epochs=epochs, lr=lr, batch_size=batch_size,
        seed=seed, device=device)

    ckpt_out = save_gbeta(readout, out / "g_beta_best.pt")
    (out / "gbeta_provenance.json").write_text(json.dumps({
        "producer": "scripts/train_nodewise_gbeta.py",
        "source_ckpt": str(ckpt_path), "head": [layer, h],
        "M": M, "n_reveal": n_reveal, "K": K, "m": m,
        "epochs": epochs, "lr": lr, "seed": seed,
        "best_val_pairwise_acc": float(val_acc),
    }, indent=2))
    return ckpt_out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default=DEFAULT_SOURCE_CKPT)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--head", type=int, nargs=2, default=list(DEFAULT_HEAD))
    p.add_argument("--select-head", action="store_true")
    p.add_argument("--M", type=int, default=2000)
    p.add_argument("--n-reveal", type=int, default=1)
    p.add_argument("--K", type=int, default=1000)
    p.add_argument("--m", type=int, default=8)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    path = train_nodewise_gbeta(
        a.ckpt, out_dir=a.out_dir, head=tuple(a.head), M=a.M, n_reveal=a.n_reveal,
        K=a.K, m=a.m, epochs=a.epochs, lr=a.lr, batch_size=a.batch_size,
        seed=a.seed, device=a.device, select_head=a.select_head)
    print(f"saved → {path}")


if __name__ == "__main__":
    main()
