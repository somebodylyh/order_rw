#!/usr/bin/env python3
"""Phase-B continuous ImageNet32 Round-2 continuation.

This is the continuous-patch counterpart of the E3-control-small VQ Round-2:
use an attention-derived 8x8 graph B, train online orders, and evaluate fixed
orders. It is intentionally minimal for cross-graph validation.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_float32_matmul_precision("high")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "image_order"))
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))

from data_imagenet32_patches import ImageNet32Patches
from directed_graph_policy import build_directed_graph
from graph_rw_image import IMAGE_RW_PARAMS_DEFAULT, sample_image_orders_batch
from model_image_aogpt import ImageAOGPT, ImageAOGPTConfig
from readout_order_diagnostic import hilbert_order
from train_imagelarge_round2 import make_shuffled_Bcov_control, sample_coverage_batched_torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", choices=["random", "Bcov_balanced", "distance_only_coverage", "shuffled_Bcov_balanced"], required=True)
    p.add_argument("--baseline-ckpt", type=Path, required=True)
    p.add_argument("--a-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--min-lr", type=float, default=3e-6)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-iters", type=int, default=100)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--max-eval-batches", type=int, default=16)
    p.add_argument("--val-images", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--alpha", type=float, default=0.9)
    p.add_argument("--alpha-warmup", type=int, default=1000)
    p.add_argument("--compile", action="store_true")
    return p.parse_args()


def get_lr(step, warmup_iters, max_steps, lr, min_lr):
    if step < warmup_iters:
        return lr * step / max(warmup_iters, 1)
    if step >= max_steps:
        return min_lr
    ratio = (step - warmup_iters) / max(max_steps - warmup_iters, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (lr - min_lr)


def get_alpha(step, alpha, warmup):
    if warmup <= 0:
        return float(alpha)
    return float(min(alpha, alpha * step / max(warmup, 1)))


def load_model(path: Path, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("config", {})
    valid = set(inspect.signature(ImageAOGPTConfig.__init__).parameters.keys()) - {"self"}
    cfg = ImageAOGPTConfig(**{k: v for k, v in cfg_dict.items() if k in valid})
    model = ImageAOGPT(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    return model, ckpt


def sample_policy_orders(policy, B_np, batch_size, step, seed, device):
    if policy == "random":
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed * 1000003 + step))
        return torch.stack([torch.randperm(64, device=device, generator=gen) for _ in range(batch_size)])
    if policy == "Bcov_balanced":
        return sample_coverage_batched_torch(B_np, batch_size, step, seed, device, gamma_B=1.0, gamma_d=1.0)
    if policy == "distance_only_coverage":
        return sample_coverage_batched_torch(B_np, batch_size, step, seed, device, gamma_B=0.01, gamma_d=1.0)
    if policy == "shuffled_Bcov_balanced":
        B_shuf = make_shuffled_Bcov_control(B_np, seed + 424242)
        return sample_coverage_batched_torch(B_shuf, batch_size, step, seed, device, gamma_B=1.0, gamma_d=1.0)
    raise ValueError(policy)


@torch.no_grad()
def evaluate(model, val_patches, B_np, batch_size, max_batches, step, seed, device):
    model.eval()
    results = {k: [] for k in ["random", "raster", "hilbert", "Bcov_balanced", "distance_only_coverage", "shuffled_Bcov_balanced", "graph_rw_top4"]}
    n_batches = min(max_batches, math.ceil(val_patches.shape[0] / batch_size))
    raster = torch.arange(64, device=device).view(1, -1)
    hil = torch.from_numpy(hilbert_order().astype(np.int64)).long().to(device).view(1, -1)
    B_shuf = make_shuffled_Bcov_control(B_np, seed + 424242)
    rw_params = {**IMAGE_RW_PARAMS_DEFAULT, "top_k": 4, "epsilon_uniform": 0.0}
    for bi in range(n_batches):
        batch = val_patches[bi * batch_size : min((bi + 1) * batch_size, val_patches.shape[0])]
        bs = batch.shape[0]
        rand = torch.stack([torch.randperm(64, device=device) for _ in range(bs)])
        orders = {
            "random": rand,
            "raster": raster.expand(bs, -1),
            "hilbert": hil.expand(bs, -1),
            "Bcov_balanced": sample_coverage_batched_torch(B_np, bs, step + bi, seed + 17, device, gamma_B=1.0, gamma_d=1.0),
            "distance_only_coverage": sample_coverage_batched_torch(B_np, bs, step + bi, seed + 23, device, gamma_B=0.01, gamma_d=1.0),
            "shuffled_Bcov_balanced": sample_coverage_batched_torch(B_shuf, bs, step + bi, seed + 31, device, gamma_B=1.0, gamma_d=1.0),
            "graph_rw_top4": sample_image_orders_batch(B_np, rw_params, bs, seed_base=seed + 9999 + step, step=bi, device=device),
        }
        for name, ords in orders.items():
            _, loss = model(batch, mode=None, orders=ords.contiguous())
            results[name].append(float(loss.item()))
    model.train()
    return {k: float(np.mean(v)) for k, v in results.items()}


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(args.output_dir / "train_log.txt", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    log("=== train_imagenet32_continuous_round2.py ===")
    log(f"args: {vars(args)}")
    with open(args.output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = args.device
    is_cuda = device.startswith("cuda")

    A = np.load(args.a_path).astype(np.float64)
    B_np = build_directed_graph(A).astype(np.float32)
    log(f"Loaded A={args.a_path}, B shape={B_np.shape}, policy={args.policy}")

    model, ckpt = load_model(args.baseline_ckpt, device)
    if args.compile:
        model = torch.compile(model, mode="reduce-overhead")
    model.train()
    log(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    log("Loading ImageNet32 train/val patches...")
    train_ds = ImageNet32Patches("train")
    val_ds = ImageNet32Patches("val")
    val_patches = (val_ds.patches[: args.val_images] if val_ds.patches is not None else val_ds.patches_for_indices(np.arange(args.val_images))).to(device)
    log(f"train N={len(train_ds)} lazy={train_ds.lazy}; val={tuple(val_patches.shape)}")

    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.lr,
        betas=(args.beta1, args.beta2),
        device_type="cuda" if is_cuda else "cpu",
    )

    eval_tsv = args.output_dir / "eval_curve.tsv"
    header = "step\ttrain_loss\ttrain_random\ttrain_policy\talpha\tval_random\tval_raster\tval_hilbert\tval_Bcov_balanced\tval_distance_only_coverage\tval_shuffled_Bcov_balanced\tval_graph_rw_top4\tlr\n"
    write_header = not eval_tsv.exists()

    best_metric = float("inf")
    best_state = None
    best_step = -1
    losses = []
    t0 = time.time()

    def do_eval(step, train_loss, train_random, train_policy, alpha, lr):
        nonlocal write_header, best_metric, best_state, best_step
        vals = evaluate(model, val_patches, B_np, args.batch_size, args.max_eval_batches, step, args.seed, device)
        cross = float(np.mean(list(vals.values())))
        if cross < best_metric:
            best_metric = cross
            best_step = step
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        log(
            f"[eval] step={step} train={train_loss:.5f} alpha={alpha:.3f} "
            f"rand={vals['random']:.5f} raster={vals['raster']:.5f} hil={vals['hilbert']:.5f} "
            f"bcov={vals['Bcov_balanced']:.5f} dist={vals['distance_only_coverage']:.5f} "
            f"shuf={vals['shuffled_Bcov_balanced']:.5f} rw={vals['graph_rw_top4']:.5f} "
            f"cross={cross:.5f} elapsed={time.time()-t0:.1f}s"
        )
        with open(eval_tsv, "w" if write_header else "a", encoding="utf-8") as f:
            if write_header:
                f.write(header)
            f.write(
                f"{step}\t{train_loss:.8f}\t{train_random:.8f}\t{train_policy:.8f}\t{alpha:.6f}\t"
                f"{vals['random']:.8f}\t{vals['raster']:.8f}\t{vals['hilbert']:.8f}\t"
                f"{vals['Bcov_balanced']:.8f}\t{vals['distance_only_coverage']:.8f}\t"
                f"{vals['shuffled_Bcov_balanced']:.8f}\t{vals['graph_rw_top4']:.8f}\t{lr:.8e}\n"
            )
        write_header = False
        return vals

    do_eval(0, float("nan"), float("nan"), float("nan"), get_alpha(0, args.alpha, args.alpha_warmup), args.lr)

    for step in range(args.max_steps):
        lr = get_lr(step, args.warmup_iters, args.max_steps, args.lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        alpha = get_alpha(step, args.alpha, args.alpha_warmup)

        idx = rng.integers(0, len(train_ds), size=args.batch_size)
        batch = train_ds.patches_for_indices(idx, device=device) if train_ds.lazy else train_ds.patches[idx].to(device)
        rand_orders = torch.stack([torch.randperm(64, device=device) for _ in range(batch.shape[0])])
        pol_orders = sample_policy_orders(args.policy, B_np, batch.shape[0], step, args.seed, device)

        _, loss_random = model(batch, mode=None, orders=rand_orders)
        _, loss_policy = model(batch, mode=None, orders=pol_orders)
        loss = (1.0 - alpha) * loss_random + alpha * loss_policy

        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        losses.append(float(loss.item()))
        if (step + 1) % args.log_interval == 0:
            log(f"step={step+1} loss={loss.item():.5f} rnd={loss_random.item():.5f} pol={loss_policy.item():.5f} alpha={alpha:.3f} lr={lr:.2e}")
        if (step + 1) % args.eval_interval == 0 or step == args.max_steps - 1:
            recent = float(np.mean(losses[-args.log_interval:]))
            do_eval(step + 1, recent, float(loss_random.item()), float(loss_policy.item()), alpha, lr)

    ckpt_path = args.output_dir / "ckpt_final.pt"
    torch.save(
        {
            "model_state_dict": best_state if best_state is not None else {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "config": ckpt.get("config", {}),
            "args": vars(args),
            "best_cross": best_metric,
            "best_step": best_step,
            "source_ckpt": str(args.baseline_ckpt),
            "source_a_path": str(args.a_path),
        },
        ckpt_path,
    )
    log(f"Saved {ckpt_path}; best_cross={best_metric:.6f} at step={best_step}")
    log_f.close()


if __name__ == "__main__":
    main()
