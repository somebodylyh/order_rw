#!/usr/bin/env python3
"""Run B1/predictor per-head order-signal scan on a collaborator AO-GPT ckpt.

This is a read-only adapter: it does not train and does not modify the ckpt.
It loads the collaborator checkpoint format (`config` + `data_permutation`) and
feeds Wikitext chunks through the same per-head B1/predictor extraction used by
`block_lo_arm_order_network/per_head_order_scan.py`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = ROOT / "block_lo_arm_order_network"
sys.path.insert(0, str(PKG))

from clean_training_protocol import CleanPermutation, expand_model_blocks_to_token_order  # noqa: E402
from per_head_order_scan import (  # noqa: E402
    _batch_mean_B,
    _mean_tau_vs,
    _orders_from_graphs,
    extract_per_head_and_heavy_A,
    head_order_metrics,
)
from training_utils import AOGPT, AOGPTConfig, BLOCK_LEN, N, SEQ_LEN, load_train_chunks  # noqa: E402


def _expand_block_perm_to_tokens(block_perm: torch.Tensor) -> torch.Tensor:
    return expand_model_blocks_to_token_order(block_perm.view(1, -1), BLOCK_LEN)[0]


def _load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_args = dict(ckpt["model_args"])
    model_args["block_size"] = SEQ_LEN
    sig = list(AOGPTConfig.__init__.__code__.co_varnames)
    valid = {k: v for k, v in model_args.items() if k in sig}
    model = AOGPT(AOGPTConfig(**valid))
    model.crop_block_size(SEQ_LEN)
    sd = dict(ckpt["model"])
    clean_sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(clean_sd)
    dev = torch.device(device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    model.to(dev).eval()
    return ckpt, model, dev


def _clean_perm_from_ckpt(ckpt, perm_orientation: str) -> CleanPermutation:
    dp = ckpt["data_permutation"]
    bp = torch.tensor(dp["block_perm"], dtype=torch.long)
    ip = torch.tensor(dp["inverse_block_perm"], dtype=torch.long)
    if perm_orientation == "phys_to_model":
        block_perm_phys_to_model = bp
        inv_perm_model_to_phys = ip
    elif perm_orientation == "model_to_phys":
        block_perm_phys_to_model = ip
        inv_perm_model_to_phys = bp
    else:
        raise ValueError(f"unknown orientation {perm_orientation!r}")
    return CleanPermutation(
        block_perm_phys_to_model=block_perm_phys_to_model,
        inv_perm_model_to_phys=inv_perm_model_to_phys,
    )


def _physical_chunks_to_model(chunks_phys: torch.Tensor, clean_perm: CleanPermutation) -> torch.Tensor:
    # clean_perm.block_perm_phys_to_model[physical_block] = model_block.
    # For model-frame input position m, take the physical block p where block_perm[p] == m.
    inv = clean_perm.inv_perm_model_to_phys
    token_gather = _expand_block_perm_to_tokens(inv)
    return chunks_phys[:, token_gather]


def scan(args):
    ckpt, model, dev = _load_model(args.ckpt, args.device)
    clean_perm = _clean_perm_from_ckpt(ckpt, args.perm_orientation)

    total = int(args.M) * int(args.batch_size)
    chunks_phys = load_train_chunks(n_chunks=total)
    chunks_model = _physical_chunks_to_model(chunks_phys, clean_perm)

    A_lh, A_heavy = extract_per_head_and_heavy_A(
        model,
        chunks_model,
        clean_perm,
        dev,
        seed=args.seed,
        n_top=args.n_top,
        fwd_batch=args.fwd_batch,
        none_mode=args.none_mode,
    )
    Ln, Hn = A_lh.shape[1], A_lh.shape[2]

    sigma_heavy = _orders_from_graphs(_batch_mean_B(A_heavy, args.M, args.batch_size), args.alpha_dep)
    heavy = {
        "tau_vs_l2r": _mean_tau_vs(sigma_heavy, np.arange(N)),
        "mean_pairwise_tau": head_order_metrics(sigma_heavy)["mean_pairwise_tau"],
        "first_step_entropy": head_order_metrics(sigma_heavy)["first_step_entropy"],
    }

    per_head = []
    for l in range(Ln):
        for h in range(Hn):
            B = _batch_mean_B(A_lh[:, l, h], args.M, args.batch_size)
            sig = _orders_from_graphs(B, args.alpha_dep)
            m = head_order_metrics(sig, sigmas_heavy=sigma_heavy)
            per_head.append({"layer": l, "head": h, **m})
    per_head.sort(key=lambda d: abs(d["tau_vs_l2r"]), reverse=True)

    result = {
        "config": {
            "ckpt": args.ckpt,
            "iter_num": ckpt.get("iter_num"),
            "best_val_loss": ckpt.get("best_val_loss"),
            "model_args": ckpt.get("model_args"),
            "ckpt_config_subset": {
                k: ckpt.get("config", {}).get(k)
                for k in [
                    "dataset",
                    "data_record_mode",
                    "permute_data",
                    "permute_seed",
                    "permute_mode",
                    "aogpt_train_mode",
                    "max_iters",
                ]
            },
            "M": args.M,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "alpha_dep": args.alpha_dep,
            "none_mode": args.none_mode,
            "perm_orientation": args.perm_orientation,
            "input_chunks": "Wikitext-103 train chunks from local HF cache; physical chunks permuted to model frame",
            "L": Ln,
            "H": Hn,
        },
        "data_permutation": ckpt.get("data_permutation"),
        "heavy_baseline": heavy,
        "per_head_layer_sorted_by_abs_tau_vs_l2r": per_head,
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    top = per_head[0]
    n09 = sum(abs(x["tau_vs_l2r"]) > 0.9 for x in per_head)
    n07 = sum(abs(x["tau_vs_l2r"]) > 0.7 for x in per_head)
    print(f"saved {args.out}")
    print(
        f"top_abs=L{top['layer']}H{top['head']} tau={top['tau_vs_l2r']:.6f} "
        f"pair={top['mean_pairwise_tau']:.6f}; |tau|>0.9={n09}/{Ln*Hn}; "
        f"|tau|>0.7={n07}/{Ln*Hn}; heavy_tau={heavy['tau_vs_l2r']:.6f}"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="/home/admin/ych/nanogpt-learned-order/ckpt/ckpt.pt")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--M", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha-dep", type=float, default=0.5)
    p.add_argument("--none-mode", choices=["predictor", "b1", "b0", "old", "loss_aligned"], default="predictor")
    p.add_argument("--perm-orientation", choices=["phys_to_model", "model_to_phys"], default="phys_to_model")
    p.add_argument("--n-top", type=int, default=4)
    p.add_argument("--fwd-batch", type=int, default=8)
    args = p.parse_args()
    scan(args)


if __name__ == "__main__":
    main()
