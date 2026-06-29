#!/usr/bin/env python3
"""Reproduce the original checkpoint Random val loss."""

from __future__ import annotations

import argparse
from pathlib import Path

from debug_eval_mismatch_common import (
    DEFAULT_CKPT,
    DEFAULT_OUT_DIR,
    NANOGPT_ROOT,
    autocast_context,
    eval_batches,
    get_permutation_state,
    load_aogpt_from_ckpt,
    load_memmap_batch,
    resolve_device,
    set_seeds,
    write_json,
)


def make_memmap_batches(data_path, batch_size, block_size, num_batches, seed, device, perm_state):
    import torch

    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    for _ in range(num_batches):
        yield load_memmap_batch(
            str(data_path),
            batch_size=batch_size,
            block_size=block_size,
            rng=rng,
            device=device,
            perm_state=perm_state,
        )


def run_eval(args):
    device = resolve_device(args.device)
    set_seeds(args.seed)
    model, ckpt = load_aogpt_from_ckpt(args.ckpt, device)
    perm_state = get_permutation_state(ckpt, device=device)
    cfg = ckpt.get("config", {})
    block_size = int(ckpt["model_args"]["block_size"])
    batch_size = int(args.batch_size or cfg.get("batch_size", 64))
    data_dir = Path(args.data_dir or (NANOGPT_ROOT / "data" / cfg.get("dataset", "wikitext103")))
    val_path = data_dir / "val.bin"
    ctx = autocast_context(device, args.dtype)

    small_batches = make_memmap_batches(
        val_path, batch_size, block_size, args.small_iters, args.seed, device, perm_state
    )
    small = eval_batches(model, small_batches, mode="Random", ctx=ctx, return_counts=True)

    full_batches = make_memmap_batches(
        val_path, batch_size, block_size, args.full_iters, args.seed, device, perm_state
    )
    full = eval_batches(model, full_batches, mode="Random", ctx=ctx, return_counts=True)

    # A negative control: same sampled val chunks, but without the checkpoint input permutation.
    raw_batches = make_memmap_batches(
        val_path, batch_size, block_size, args.small_iters, args.seed, device, None
    )
    raw_small = eval_batches(model, raw_batches, mode="Random", ctx=ctx, return_counts=True)

    payload = {
        "checkpoint": str(args.ckpt),
        "checkpoint_iter_num": int(ckpt.get("iter_num", -1)),
        "checkpoint_best_val_loss": float(ckpt.get("best_val_loss", float("nan"))),
        "data_source": str(val_path),
        "input_frame": "model/permuted by checkpoint fixed_token_perm",
        "eval_objective": "AO-GPT MDM Random block-order reveal loss",
        "loss_normalization": "mean CE over batch * 256 predicted reveal tokens, then mean over eval batches",
        "batch_size": batch_size,
        "block_size": block_size,
        "small_iters": int(args.small_iters),
        "full_iters": int(args.full_iters),
        "reproduced_val_loss_small": small["loss"],
        "reproduced_val_loss_full_or_approx": full["loss"],
        "raw_unpermuted_negative_control_small": raw_small["loss"],
        "small_predicted_tokens": int(small["predicted_tokens"]),
        "full_predicted_tokens": int(full["predicted_tokens"]),
        "diff_small_vs_ckpt_best": small["loss"] - float(ckpt.get("best_val_loss", 0.0)),
        "diff_full_vs_ckpt_best": full["loss"] - float(ckpt.get("best_val_loss", 0.0)),
        "key_parameters": {
            "aogpt_train_mode": cfg.get("aogpt_train_mode"),
            "main_eval_mode": cfg.get("main_eval_mode"),
            "permute_data": cfg.get("permute_data"),
            "permute_seed": cfg.get("permute_seed"),
            "permute_mode": cfg.get("permute_mode"),
            "block_order_block_len": cfg.get("block_order_block_len"),
            "order_impl": ckpt["model_args"].get("order_impl"),
            "dtype": args.dtype,
        },
    }

    out_path = Path(args.output_dir) / "reproduce_ckpt_val_loss.json"
    write_json(out_path, payload)
    print(f"wrote {out_path}")
    print(f"checkpoint_best_val_loss={payload['checkpoint_best_val_loss']:.6f}")
    print(f"reproduced_val_loss_small={payload['reproduced_val_loss_small']:.6f}")
    print(f"reproduced_val_loss_full_or_approx={payload['reproduced_val_loss_full_or_approx']:.6f}")
    print(f"raw_unpermuted_negative_control_small={payload['raw_unpermuted_negative_control_small']:.6f}")
    return payload


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--small-iters", type=int, default=10)
    parser.add_argument("--full-iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


if __name__ == "__main__":
    run_eval(parse_args())
