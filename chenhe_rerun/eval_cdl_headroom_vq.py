"""Headroom gate: does zero-prior CDL(attention) block order lower own-order CE
vs random on VQ image tokens? Deployment-faithful (batch-mean single σ broadcast).

Reference-only: raster (ascending patch order). raster is a POST-HOC oracle here —
reported for context, never used to build the CDL order.
"""
import os
import sys
import json
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gbeta_cdl_pretrain import (
    load_chenhe_backbone, single_head_B_from_forward, _sample_data_windows, _make_probe,
)
from orderhead_v3.none_separated_block_graph import build_none_separated_B, rollout_from_none
from orderhead_v3.constants import N

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(REPO, "probe_results_image/vq64_fixed_random_l8h8e512/ckpt_step30000.pt")
VAL_BIN = os.path.join(REPO, "nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch2x2/val.bin")
GBETA_PROV = os.path.join(REPO, "out/rerun_vq/gbeta_vq_bm16/gbeta_provenance.json")


@torch.no_grad()
def _ce(model, idx, block_orders, device):
    tok = model._expand_block_orders_to_token_orders(block_orders.to(device))
    out = model.forward_fn(idx.to(device), tok, return_logits=True)
    return float(out[1].item())


@torch.no_grad()
def _batch_mean_B65(model, idx, layer, head, device, probes=4, seed=0):
    Bsum = None
    for k in range(probes):
        probe = _make_probe(model, idx.shape[0], seed * 1000 + k, device)
        Bk = single_head_B_from_forward(model, idx, layer, head, probe, device)  # (bs,64,64)
        Bsum = Bk if Bsum is None else Bsum + Bk
    Bmean = (Bsum / probes).mean(0).numpy()          # (64,64) batch-mean, None-stripped
    B65 = np.zeros((65, 65), np.float32)
    B65[1:, 1:] = Bmean
    return B65


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--rand-reps", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    prov = json.load(open(GBETA_PROV))
    layer, head = int(prov["sel_layer"]), int(prov["sel_head"])
    model, _ = load_chenhe_backbone(CKPT, args.device)

    windows = _sample_data_windows(VAL_BIN, args.n, args.seed)
    n_batches = args.n // args.batch_size
    ce_cdl, ce_rand, ce_raster = [], [], []
    g = torch.Generator().manual_seed(args.seed)

    for b in range(n_batches):
        idx = windows[b * args.batch_size:(b + 1) * args.batch_size]
        bs = idx.shape[0]

        B65 = _batch_mean_B65(model, idx, layer, head, args.device, seed=args.seed + b)
        sigma_cdl = torch.from_numpy(rollout_from_none(B65, mode="C-D+L")).long()  # (64,)
        ce_cdl.append(_ce(model, idx, sigma_cdl.unsqueeze(0).expand(bs, -1), args.device))

        reps = []
        for r in range(args.rand_reps):
            rp = torch.randperm(N, generator=g).long()
            reps.append(_ce(model, idx, rp.unsqueeze(0).expand(bs, -1), args.device))
        ce_rand.append(float(np.mean(reps)))

        raster = torch.arange(N).long()
        ce_raster.append(_ce(model, idx, raster.unsqueeze(0).expand(bs, -1), args.device))

        print(f"[batch {b+1}/{n_batches}] cdl={ce_cdl[-1]:.4f} rand={ce_rand[-1]:.4f} "
              f"raster={ce_raster[-1]:.4f} delta={ce_cdl[-1]-ce_rand[-1]:+.4f}", flush=True)

    mc, mr, mras = float(np.mean(ce_cdl)), float(np.mean(ce_rand)), float(np.mean(ce_raster))
    print("\n=== HEADROOM GATE (VQ image, zero-prior CDL-attention) ===")
    print(f"ce_cdl    = {mc:.4f}")
    print(f"ce_random = {mr:.4f}")
    print(f"ce_raster = {mras:.4f}  (post-hoc reference only)")
    print(f"delta (cdl-random) = {mc-mr:+.4f}   "
          f"{'PASS (cdl<random)' if mc < mr else 'FAIL/NULL (no headroom)'}")


if __name__ == "__main__":
    main()
