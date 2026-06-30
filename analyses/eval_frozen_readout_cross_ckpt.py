#!/usr/bin/env python3
"""Cross-checkpoint transfer test for a FROZEN FlattenReadout (g_β).

Loads a trained label-free readout and, WITHOUT retraining, evaluates its
posthoc physical-order τ on B extracted from multiple checkpoints of the SAME
training lineage. Answers: does g_β (trained at one step) still read physical
order from the same model at other training steps?

g_β stays frozen. Only the substrate (model checkpoint → attention → B) changes.
"""
from __future__ import annotations

import argparse, pathlib, sys
import numpy as np
import torch
from scipy.stats import kendalltau

ROOT = pathlib.Path(__file__).resolve().parent.parent
PKG = ROOT / "block_lo_arm_order_network"
sys.path.insert(0, str(PKG))

from neural_readout.extract_b import _load_model_and_chunks
from none_separated_block_graph import build_none_separated_B
from per_head_order_scan import (
    _attn_to_A_block_loss_aligned_with_none_model_vec as _A_model_vec,
)
from uniform_label_free_v1 import _random_probe_orders  # same probe distribution
from batch_readout.model import FlattenReadout


def extract_B_bm(ckpt_path, cluster_heads, M, batch_size, seed, device):
    """Cluster-mean, batch-mean B (M,64,64) + inv_perm for this ckpt."""
    total = M * batch_size
    dev = torch.device(device)
    model, chunks, clean_perm, dev_actual, _ = _load_model_and_chunks(
        ckpt_path, total, seed, str(dev), "train"
    )
    inv_perm = clean_perm.inv_perm_model_to_phys.cpu().numpy()

    B_content = np.zeros((total, 64, 64), dtype=np.float32)
    fwd_batch = 32
    for start in range(0, total, fwd_batch):
        end = min(start + fwd_batch, total)
        bs = end - start
        probe = _random_probe_orders(bs, seed * 100 + start, dev_actual)
        model.eval()
        with torch.no_grad():
            _, _, attn_list = model.forward_fn(
                chunks[start:end].to(dev_actual), probe, return_attentions=True
            )
            if dev_actual.type == "cuda":
                torch.cuda.synchronize(dev_actual)
        probe_np = probe.cpu().numpy()
        for bi in range(bs):
            B_sum = np.zeros((65, 65), dtype=np.float64)
            for layer, head in cluster_heads:
                attn = attn_list[layer][bi, head].cpu().numpy()
                A65 = _A_model_vec(attn, probe_np[bi])
                B65 = build_none_separated_B(A65)
                B_sum += B65
            B_mean = (B_sum / len(cluster_heads)).astype(np.float32)
            B_content[start + bi] = B_mean[1:, 1:]
        del attn_list
    B_bm = B_content.reshape(M, batch_size, 64, 64).mean(axis=1).astype(np.float32)
    return B_bm, inv_perm


def eval_frozen(readout, B_bm, inv_perm, device):
    dev = torch.device(device)
    M = B_bm.shape[0]
    taus = []
    with torch.no_grad():
        for m in range(M):
            scores = readout(torch.from_numpy(B_bm[m:m+1]).float().to(dev))
            sigma_model = scores[0].argsort(descending=True).cpu().numpy()
            sigma_phys = inv_perm[sigma_model]
            tau, _ = kendalltau(sigma_phys, np.arange(64))
            taus.append(tau)
    return float(np.mean(taus)), float(np.std(taus))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--readout", default="reports/uniform_label_free_v1/label_free_readout.pt")
    p.add_argument("--head", type=int, nargs=2, default=[1, 7], metavar=("L", "H"))
    p.add_argument("--ckpt-dir", default="block_lo_arm_order_network/probe_results/overnight_20260625_random_baseline")
    p.add_argument("--steps", default="5000,10000,20000,30000,40000,50000,60000")
    p.add_argument("--M", type=int, default=200)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    d = torch.load(args.readout, map_location="cpu", weights_only=False)
    cfg = d["config"]
    readout = FlattenReadout(N=cfg["N"], hidden=tuple(cfg["hidden"]))
    readout.load_state_dict(d["model"])
    readout.to(args.device).eval()
    cluster_heads = [tuple(args.head)]
    print(f"frozen readout={args.readout}  head=L{args.head[0]}H{args.head[1]}  M={args.M} bs={args.batch}")
    print(f"{'step':>8} | {'tau_phys(mean)':>14} | {'std':>6}")
    print("-" * 36)
    for s in args.steps.split(","):
        ckpt = f"{args.ckpt_dir}/ckpt_step{s}.pt"
        if not pathlib.Path(ckpt).exists():
            print(f"{s:>8} | MISSING"); continue
        B_bm, inv_perm = extract_B_bm(ckpt, cluster_heads, args.M, args.batch, args.seed, args.device)
        tau_m, tau_s = eval_frozen(readout, B_bm, inv_perm, args.device)
        print(f"{s:>8} | {tau_m:>14.4f} | {tau_s:>6.3f}", flush=True)


if __name__ == "__main__":
    main()
