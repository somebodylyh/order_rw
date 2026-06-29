"""E1: random-patch-order baseline on ImageNet32 (bilinear from ImageNet64).

Minimal clone of train_image_random.py — only data loading differs:
  - Uses ImageNet32Patches instead of CIFAR10Patches.
  - val split key is "val" (CIFAR's clone used "test").
  - val_images default is 1000 (same as CIFAR).

Everything else (model = ImageAOGPT default l4h8e256, optimizer, schedule, eval)
matches the toy E0 recipe so the only changed axis vs E0 is *dataset only*.

Usage (smoke):
    python -u image_order/train_image_random_imagenet32.py \
        --output-dir probe_results_image/baseline_imagenet32/smoke500 \
        --max-steps 500 --batch-size 64 --eval-interval 100 --device cuda

Usage (full 10k, matches cifar_patch_baseline10k.py):
    python -u image_order/train_image_random_imagenet32.py \
        --output-dir probe_results_image/baseline_imagenet32/baseline10k \
        --max-steps 10000 --batch-size 64 --lr 3e-4 --min-lr 3e-5 \
        --warmup-iters 200 --eval-interval 500 --max-eval-batches 16 \
        --val-images 1000 --seed 42 --device cuda
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "image_order"))

from data_imagenet32_patches import ImageNet32Patches
from model_image_aogpt import ImageAOGPT, ImageAOGPTConfig


def parse_args():
    parser = argparse.ArgumentParser(description="E1 baseline on ImageNet32")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-iters", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--max-eval-batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--val-images", type=int, default=1000)
    parser.add_argument("--max-train-images", type=int, default=None,
                        help="Cap training images loaded in memory (smoke).")
    return parser.parse_args()


def get_lr(step, warmup_iters, max_steps, lr, min_lr):
    if step < warmup_iters:
        return lr * step / max(warmup_iters, 1)
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_iters) / max(max_steps - warmup_iters, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


@torch.no_grad()
def evaluate(model, val_patches, batch_size, max_eval_batches, device):
    model.eval()
    V = val_patches.shape[0]
    random_losses, raster_losses = [], []
    n_batches = 0
    for start in range(0, V, batch_size):
        if n_batches >= max_eval_batches:
            break
        end = min(start + batch_size, V)
        batch = val_patches[start:end]
        _, loss_rand = model(batch, mode="Random")
        _, loss_ar = model(batch, mode="AR")
        random_losses.append(loss_rand.item())
        raster_losses.append(loss_ar.item())
        n_batches += 1
    model.train()
    return float(np.mean(random_losses)), float(np.mean(raster_losses))


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "train_log.txt"
    log_file = open(log_path, "a")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    log("=== train_image_random_imagenet32.py (E1) ===")
    log(f"args: {vars(args)}")

    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = args.device
    is_cuda = device.startswith("cuda")

    log("Loading ImageNet32 train split ...")
    train_ds = ImageNet32Patches("train", max_images=args.max_train_images)
    log(f"  train: N={len(train_ds)}  lazy={train_ds.lazy}")
    log("Loading ImageNet32 val split ...")
    val_ds = ImageNet32Patches("val")
    log(f"  val:   N={len(val_ds)}  lazy={val_ds.lazy}")

    # val keeps the materialized path (50k images < threshold)
    if val_ds.patches is not None:
        val_patches = val_ds.patches[: args.val_images].to(device)
    else:
        val_patches = val_ds.patches_for_indices(
            np.arange(min(args.val_images, len(val_ds))), device=device
        )
    log(f"val_patches: {tuple(val_patches.shape)}")

    cfg = ImageAOGPTConfig()
    model = ImageAOGPT(cfg)
    model.to(device)
    model.train()
    log(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.lr,
        betas=(args.beta1, args.beta2),
        device_type="cuda" if is_cuda else "cpu",
    )

    eval_tsv = output_dir / "eval_curve.tsv"
    write_header = not eval_tsv.exists()

    train_losses = []
    best_val_random = float("inf")
    best_step = -1
    best_state = None

    t0 = time.time()
    running_losses = []

    def do_eval(step, current_train_loss, current_lr):
        nonlocal best_val_random, best_step, best_state, write_header
        val_rand, val_ar = evaluate(
            model, val_patches, args.batch_size, args.max_eval_batches, device
        )
        elapsed = time.time() - t0
        log(
            f"[eval] step={step:5d} | train_loss={current_train_loss:.4f} | "
            f"val_random={val_rand:.4f} | val_raster={val_ar:.4f} | "
            f"lr={current_lr:.2e} | elapsed={elapsed:.1f}s"
        )
        mode = "w" if write_header else "a"
        with open(eval_tsv, mode) as f:
            if write_header:
                f.write("step\ttrain_loss\tval_random\tval_raster\tlr\n")
            f.write(f"{step}\t{current_train_loss:.6f}\t{val_rand:.6f}\t{val_ar:.6f}\t{current_lr:.6e}\n")
        write_header = False
        if val_rand < best_val_random:
            best_val_random = val_rand
            best_step = step
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            log(f"  ** new best val_random={best_val_random:.4f} at step {best_step}")
        return val_rand

    step0_val_rand = do_eval(step=0, current_train_loss=float("nan"), current_lr=args.lr)

    for step in range(args.max_steps):
        current_lr = get_lr(step, args.warmup_iters, args.max_steps, args.lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        idx = rng.integers(0, len(train_ds), size=args.batch_size)
        if train_ds.lazy:
            batch = train_ds.patches_for_indices(idx, device=device)
        else:
            batch = train_ds.patches[idx].to(device)

        _, loss = model(batch, mode="Random")
        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        loss_val = loss.item()
        train_losses.append(loss_val)
        running_losses.append(loss_val)

        if (step + 1) % args.log_interval == 0:
            elapsed = time.time() - t0
            recent_mean = float(np.mean(running_losses[-args.log_interval:]))
            log(
                f"step={step+1:5d} | loss={loss_val:.4f} | mean(last {args.log_interval})={recent_mean:.4f} | "
                f"lr={current_lr:.2e} | elapsed={elapsed:.1f}s"
            )

        is_last_step = (step == args.max_steps - 1)
        if ((step + 1) % args.eval_interval == 0) or is_last_step:
            do_eval(
                step=step + 1,
                current_train_loss=float(np.mean(running_losses[-args.log_interval:])),
                current_lr=current_lr,
            )

    final_ckpt_path = output_dir / f"ckpt_step{args.max_steps}.pt"
    torch.save({
        "model_state_dict": best_state if best_state is not None else model.state_dict(),
        "config": dict(
            n_patches=64, patch_dim=48, n_embd=256, n_layer=4, n_head=8,
            cond_dim=128, dropout=0.0, bias=True,
        ),
        "args": vars(args),
        "train_losses": train_losses,
        "best_val_random": best_val_random,
        "best_step": best_step,
    }, final_ckpt_path)
    log(f"Saved checkpoint: {final_ckpt_path}")

    improvement = (step0_val_rand - best_val_random) / max(step0_val_rand, 1e-9)
    log(f"\n=== Smoke acceptance check ===")
    log(f"step-0 val_random : {step0_val_rand:.4f}")
    log(f"best val_random   : {best_val_random:.4f} (at step {best_step})")
    log(f"improvement       : {improvement*100:.1f}%")
    if improvement >= 0.25:
        log("PASSED: val_random improved >= 25% from step-0.")
    else:
        log("WARNING: val_random improved < 25% from step-0 — DONE_WITH_CONCERNS.")

    log("\nFiles written:")
    for p in sorted(output_dir.iterdir()):
        log(f"  {p}")
    log_file.close()


if __name__ == "__main__":
    main()
