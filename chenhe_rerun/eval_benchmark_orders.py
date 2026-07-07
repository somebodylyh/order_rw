"""Aligned benchmark for AMOR text arms.

- own-order (aligned to deployment): taken from each ckpt's recorded best_val_loss
  (authentic loss under the arm's actual reveal order + permute_data layout).
- origin-L2R (anti-gaming reference): natural text order on raw contiguous windows,
  model(idx, mode='AR'). A neutral order the model never optimized for — checks that
  the own-order advantage is not pure order-gaming. Same harness for every arm.

Consistent held-out windows (val.bin, fixed seed) across all arms.
"""
import os
import sys
import math
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gbeta_cdl_pretrain import load_chenhe_backbone, _sample_data_windows


@torch.no_grad()
def origin_l2r(ckpt, val_bin, n, batch_size, device, seed=0):
    model, _ = load_chenhe_backbone(ckpt, device)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    own = float(ck.get("best_val_loss", float("nan")))
    windows = _sample_data_windows(val_bin, n, seed)
    n_batches = n // batch_size
    losses = []
    for b in range(n_batches):
        idx = windows[b * batch_size:(b + 1) * batch_size].to(device)   # raw natural-text order
        losses.append(float(model(idx, mode='AR', return_logits=True)[1].item()))
    ol2r = float(np.mean(losses))
    return {"own": own, "origin_l2r": ol2r,
            "ppl_own": math.exp(own), "ppl_ol2r": math.exp(ol2r),
            "gap": ol2r - own}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--val-bin", default="data/wikitext103/val.bin")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print(f"{'arm':<42}{'own':>8}{'origL2R':>9}{'gap':>7}   {'PPL_own':>9}{'PPL_oL2R':>10}", flush=True)
    for c in args.ckpts:
        name = os.path.basename(os.path.dirname(c))
        r = origin_l2r(c, args.val_bin, args.n, args.batch_size, args.device)
        print(f"{name:<42}{r['own']:>8.4f}{r['origin_l2r']:>9.4f}{r['gap']:>7.2f}   "
              f"{r['ppl_own']:>9.1f}{r['ppl_ol2r']:>10.1f}", flush=True)


if __name__ == "__main__":
    main()
