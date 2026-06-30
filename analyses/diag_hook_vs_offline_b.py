#!/usr/bin/env python3
"""Decisive check: does the ACTUAL in-loop HookOrderProvider reproduce the
offline τ=1.0 of the frozen readout at step10k?

Offline (uniform_label_free_v1) builds B via
  build_none_separated_B(_attn_to_A_block_loss_aligned_with_none_model_vec(attn, probe))[1:,1:]
in MODEL frame, inv_perm applied only posthoc.

HookOrderProvider(none_mode='b1') builds B via _attn_to_A_block_b1_vec(attn, probe, inv_perm)
+ transpose. If these disagree, the hook feeds g_β an out-of-distribution matrix.

We run the real provider and measure τ(sigma_phys, arange) for several none_modes.
"""
from __future__ import annotations
import pathlib, sys
import numpy as np
import torch
from scipy.stats import kendalltau

ROOT = pathlib.Path(__file__).resolve().parent.parent
PKG = ROOT / "block_lo_arm_order_network"
sys.path.insert(0, str(PKG))

from neural_readout.extract_b import _load_model_and_chunks
from batch_readout.hook_order_provider import HookOrderProvider

CKPT = "block_lo_arm_order_network/probe_results/overnight_20260625_random_baseline/ckpt_step10000.pt"
READOUT = "reports/uniform_label_free_v1/label_free_readout.pt"
HEAD = (1, 7)
SEED = 123
DEV = "cuda:0"


def main():
    total = 8  # one batch
    model, chunks, clean_perm, dev, _ = _load_model_and_chunks(CKPT, total, SEED, DEV, "train")
    inv_perm = clean_perm.inv_perm_model_to_phys.cpu().numpy()
    idx_batch = chunks[:total].to(dev)

    print(f"ckpt=step10000  head=L{HEAD[0]}H{HEAD[1]}  readout={READOUT}")
    print(f"{'none_mode':>14} | {'tau_phys':>9} | note")
    print("-" * 50)
    # b1 is what the run actually used. Also try 'model' and 'loss_aligned' for contrast.
    for nm in ["b1", "model", "loss_aligned", "strict65_model"]:
        prov = HookOrderProvider(
            g_beta_ckpt=READOUT, head=HEAD, clean_perm=clean_perm,
            refresh_every=1, mode="argsort", tau=1.0, seed=SEED,
            device=DEV, none_mode=nm, reverse=False,
        )
        sigma = prov.physical_order(model, idx_batch, global_step=0).cpu().numpy()
        # For model/content the trainer remaps model->phys afterwards; emulate that.
        if nm in ("model", "content", "strict65_model"):
            sigma_phys = inv_perm[sigma]
            note = "model-frame → posthoc inv_perm"
        else:
            sigma_phys = sigma
            note = "b1/loss_aligned → already physical"
        tau, _ = kendalltau(sigma_phys, np.arange(64))
        print(f"{nm:>14} | {tau:>9.4f} | {note}", flush=True)


if __name__ == "__main__":
    main()
