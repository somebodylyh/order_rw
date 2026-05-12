"""Frozen-ckpt evaluation of validation patch MSE under 5 reveal-order modes.

Usage:
    python -u image_order/eval_image_orders.py \
        --ckpt <ckpt.pt> \
        --b-path <B_global.npy> \
        --output-dir <dir> \
        [--val-images 1000] [--batch-size 64] [--n-seed-rounds 3] \
        [--seed-base 42] [--device cuda]
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Resolve repo root so imports work from any cwd
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(_REPO_ROOT, "block_lo_arm_order_network"))

from data_image_patches import CIFAR10Patches
from model_image_aogpt import ImageAOGPT, ImageAOGPTConfig
from graph_rw_image import (
    IMAGE_RW_PARAMS_DEFAULT,
    sample_image_orders_batch,
)

# ---------------------------------------------------------------------------
# Order-mode definitions
# ---------------------------------------------------------------------------

MODES = [
    "random",
    "raster",
    "rw_top4_eps0",
    "rw_eps015",
    "rw_topk8",
]

RW_PARAMS = {
    "rw_top4_eps0": {**IMAGE_RW_PARAMS_DEFAULT, "top_k": 4, "epsilon_uniform": 0.0},
    "rw_eps015":    {**IMAGE_RW_PARAMS_DEFAULT, "top_k": 0, "epsilon_uniform": 0.15},
    "rw_topk8":     {**IMAGE_RW_PARAMS_DEFAULT, "top_k": 8, "epsilon_uniform": 0.0},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filter_config_keys(raw: dict) -> dict:
    """Drop unknown keys so ImageAOGPTConfig(**raw) doesn't crash."""
    import dataclasses
    known = {f.name for f in dataclasses.fields(ImageAOGPTConfig)}
    return {k: v for k, v in raw.items() if k in known}


def build_orders(mode: str, B_np: np.ndarray, actual_bs: int,
                 seed_base: int, seed_round: int, batch_idx: int,
                 device: torch.device) -> torch.LongTensor:
    """Return (actual_bs, 64) long tensor of orders for one mini-batch."""
    if mode == "random":
        orders = torch.stack([torch.randperm(64, device=device) for _ in range(actual_bs)])
    elif mode == "raster":
        orders = torch.arange(64, device=device).unsqueeze(0).expand(actual_bs, -1).contiguous()
    else:
        effective_seed_base = seed_base * 100 + seed_round
        orders = sample_image_orders_batch(
            B_np, RW_PARAMS[mode],
            batch_size=actual_bs,
            seed_base=effective_seed_base,
            step=batch_idx,
            device=device,
        )
    return orders


def eval_mode(mode: str, patches_all: torch.Tensor, B_np: np.ndarray,
              model: torch.nn.Module, batch_size: int, n_seed_rounds: int,
              seed_base: int, device: torch.device):
    """Evaluate one mode; returns (mean_mse, std_mse, per_round_means)."""
    N_total = patches_all.shape[0]
    per_round_means = []

    with torch.no_grad():
        for seed_round in range(n_seed_rounds):
            loss_sum = 0.0
            count = 0
            n_batches = math.ceil(N_total / batch_size)

            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, N_total)
                batch = patches_all[start:end]          # (actual_bs, 64, 48)
                actual_bs = batch.shape[0]

                # For random mode, use seed to make it deterministic per round
                if mode == "random":
                    g = torch.Generator(device=device)
                    g.manual_seed(seed_base * 100 + seed_round * 10000 + batch_idx)
                    orders = torch.stack([
                        torch.randperm(64, device=device, generator=g)
                        for _ in range(actual_bs)
                    ])
                else:
                    orders = build_orders(
                        mode, B_np, actual_bs,
                        seed_base, seed_round, batch_idx, device
                    )

                _, loss = model(batch, mode=None, orders=orders)
                loss_sum += loss.item() * actual_bs
                count += actual_bs

            round_mean = loss_sum / count
            per_round_means.append(round_mean)

    mean_mse = float(np.mean(per_round_means))
    std_mse = float(np.std(per_round_means, ddof=0)) if n_seed_rounds > 1 else 0.0
    return mean_mse, std_mse, per_round_means


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Eval frozen ImageAOGPT ckpt under 5 order modes")
    p.add_argument("--ckpt",         required=True,  help="path to ckpt .pt")
    p.add_argument("--b-path",       required=True,  help="path to B_global.npy (64x64)")
    p.add_argument("--output-dir",   required=True,  help="directory for eval_summary.tsv/.json")
    p.add_argument("--val-images",   type=int, default=1000)
    p.add_argument("--batch-size",   type=int, default=64)
    p.add_argument("--n-seed-rounds",type=int, default=3)
    p.add_argument("--seed-base",    type=int, default=42)
    p.add_argument("--device",       type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # ---- Load checkpoint ----
    print(f"Loading ckpt: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    raw_cfg = ckpt.get("config", {})
    cfg_kwargs = _filter_config_keys(raw_cfg)
    cfg = ImageAOGPTConfig(**cfg_kwargs)
    model = ImageAOGPT(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Model loaded. Config: {cfg}")

    # ---- Load B matrix ----
    print(f"Loading B: {args.b_path}")
    B_np = np.load(args.b_path)
    assert B_np.shape == (64, 64), f"B shape must be (64,64), got {B_np.shape}"

    # ---- Load val patches ----
    print(f"Loading CIFAR-10 test patches (first {args.val_images})...")
    ds = CIFAR10Patches("test")
    patches_all = ds.patches[: args.val_images].to(device)   # (val_images, 64, 48)
    print(f"Val patches shape: {patches_all.shape}")

    # ---- Evaluate ----
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}

    print(f"\n{'Mode':<16}  {'mean_mse':>10}  {'std_mse':>10}  {'per_round'}")
    print("-" * 70)

    for mode in MODES:
        mean_mse, std_mse, per_round = eval_mode(
            mode, patches_all, B_np, model,
            batch_size=args.batch_size,
            n_seed_rounds=args.n_seed_rounds,
            seed_base=args.seed_base,
            device=device,
        )
        results[mode] = {
            "mean_mse": mean_mse,
            "std_mse": std_mse,
            "per_round_means": per_round,
            "n_seed_rounds": args.n_seed_rounds,
            "val_images": args.val_images,
        }
        rounds_str = "  ".join(f"{v:.6f}" for v in per_round)
        print(f"{mode:<16}  {mean_mse:>10.6f}  {std_mse:>10.6f}  [{rounds_str}]")

    # ---- Sanity check ----
    random_mse = results["random"]["mean_mse"]
    raster_mse = results["raster"]["mean_mse"]
    print(f"\nSanity: random={random_mse:.4f} (expected ~0.060), "
          f"raster={raster_mse:.4f} (expected ~0.052)")
    if abs(random_mse - 0.060) > 0.020:
        print("WARNING: random MSE deviates >0.02 from expected 0.060 — "
              "check ckpt loading!")

    # ---- Write TSV ----
    tsv_path = os.path.join(args.output_dir, "eval_summary.tsv")
    header = "mode\tmean_mse\tstd_mse\tn_seed_rounds\tval_images\tnotes"
    lines = [header]
    for mode in MODES:
        r = results[mode]
        notes = ""
        if mode == "random":
            notes = "per-sample randperm; seeded per round+batch"
        elif mode == "raster":
            notes = "arange(64); deterministic"
        else:
            p = RW_PARAMS[mode]
            notes = f"top_k={p['top_k']} eps={p['epsilon_uniform']}"
        lines.append(
            f"{mode}\t{r['mean_mse']:.6f}\t{r['std_mse']:.6f}\t"
            f"{r['n_seed_rounds']}\t{r['val_images']}\t{notes}"
        )
    tsv_content = "\n".join(lines) + "\n"
    with open(tsv_path, "w") as f:
        f.write(tsv_content)
    print(f"\nTSV written: {tsv_path}")
    print(tsv_content)

    # ---- Write JSON ----
    json_path = os.path.join(args.output_dir, "eval_summary.json")
    summary = {
        "ckpt": args.ckpt,
        "b_path": args.b_path,
        "args": vars(args),
        "results": results,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON written: {json_path}")


if __name__ == "__main__":
    main()
