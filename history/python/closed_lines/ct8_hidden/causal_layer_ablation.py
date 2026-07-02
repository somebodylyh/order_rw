"""Activation-level causal probe for the order-signal carrier.

Tests whether a later-layer carrier head's order-tau causally depends on an
upstream layer's attention output. Mean-ablates the upstream layer's attention
contribution (removes its per-sample write to the residual stream) and re-reads
the carrier head's block-order tau. A downstream layer is ablated as a negative
control (the carrier, being upstream of it, must be unaffected).

This is the activation-level test the weight-composition (Pillar 2) result
could not settle: weight composition into the carrier was weak/non-rising, but
an activation-level handoff could still exist.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))

from train_clean_aogpt import build_model  # noqa: E402
from training_utils import load_train_chunks, SEQ_LEN  # noqa: E402
from attention_trajectory import extract_all_layer_B  # noqa: E402
from batch_readout.order_tau_readout import per_head_tau  # noqa: E402


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ck["model_args"], device, compile_model=False)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def _mean_ablate_hook(mod, inp, out):
    """Replace the attention module output with its batch-mean (per-sample write
    removed). Handles the (y, attn_probs) tuple returned under return_attn."""
    if isinstance(out, tuple):
        y, att = out
        return (y.mean(0, keepdim=True).expand_as(y), att)
    return out.mean(0, keepdim=True).expand_as(out)


def carrier_tau(model, tokens, orders, layer, head, device, ablate_layer=None):
    handles = []
    if ablate_layer is not None:
        handles.append(
            model.transformer.h[ablate_layer].attn.register_forward_hook(_mean_ablate_hook)
        )
    try:
        with torch.no_grad():
            _, _, attn_list = model.forward_fn(
                tokens.to(device), orders.to(device), return_attentions=True
            )
    finally:
        for h in handles:
            h.remove()
    B_all = extract_all_layer_B([a.cpu().numpy() for a in attn_list], orders.cpu().numpy())
    B_carrier = B_all[layer].mean(axis=0)[head]  # batch-mean over probes (65,65)
    tau = float(per_head_tau(B_carrier, "C-D+L")["tau_vs_l2r"])
    return tau, B_carrier


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", required=True,
                    help="seed:ckpt:layer:head  (carrier to probe)")
    ap.add_argument("--n-probes", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    dev = torch.device(args.device)
    chunks = load_train_chunks(n_chunks=args.n_probes)  # (S,256) physical chunks
    tokens = chunks[: args.n_probes]
    orders = torch.tile(torch.arange(SEQ_LEN, dtype=torch.long), (args.n_probes, 1))  # identity

    for case in args.cases:
        seed, ckpt, layer, head = case.split(":")
        layer, head = int(layer), int(head)
        model = load_model(ckpt, dev)

        def Bcorr(a, b):  # 1.0 == carrier B unchanged
            return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])

        base, B0 = carrier_tau(model, tokens, orders, layer, head, dev, ablate_layer=None)
        print(f"\nseed{seed} carrier L{layer}H{head}:")
        print(f"  baseline tau            = {base:+.3f}")
        if layer >= 1:
            up_t, up_B = carrier_tau(model, tokens, orders, layer, head, dev, ablate_layer=layer - 1)
            print(f"  ablate L{layer-1} (upstream)   tau = {up_t:+.3f}  (Δτ={up_t-base:+.3f}) "
                  f"| carrier B corr={Bcorr(B0, up_B):.4f}  <- causal test")
        down = layer + 1
        if down < model.config.n_layer:
            dn_t, dn_B = carrier_tau(model, tokens, orders, layer, head, dev, ablate_layer=down)
            print(f"  ablate L{down} (downstream) tau = {dn_t:+.3f}  (Δτ={dn_t-base:+.3f}) "
                  f"| carrier B corr={Bcorr(B0, dn_B):.4f}  <- neg control")


if __name__ == "__main__":
    main()
