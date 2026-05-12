"""Random-patch-order baseline training for ImageAOGPT.

Usage (smoke test, 500 steps):
    python -u image_order/train_image_random.py \
        --output-dir probe_results_image/baseline/smoke500 \
        --max-steps 500 --batch-size 64 --eval-interval 100 --device cuda
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

# Make sure repo root is importable
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "image_order"))

from data_image_patches import CIFAR10Patches
from model_image_aogpt import ImageAOGPT, ImageAOGPTConfig


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Random-patch-order baseline training for ImageAOGPT")
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
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--val-images", type=int, default=1000)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# LR schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def get_lr(step: int, warmup_iters: int, max_steps: int, lr: float, min_lr: float) -> float:
    if step < warmup_iters:
        return lr * step / max(warmup_iters, 1)
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_iters) / max(max_steps - warmup_iters, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, val_patches: torch.Tensor, batch_size: int, max_eval_batches: int, device: str):
    """Evaluate val_random_loss and val_raster_loss on val_patches.

    val_patches: (V, N, patch_dim) on device.
    Returns (val_random_loss, val_raster_loss).
    """
    model.eval()
    V = val_patches.shape[0]

    random_losses = []
    raster_losses = []
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "train_log.txt"
    log_file = open(log_path, "a")

    def log(msg: str):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"=== train_image_random.py ===")
    log(f"args: {vars(args)}")

    # Save config for human inspection
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = args.device
    is_cuda = device.startswith("cuda")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    log("Loading CIFAR-10 train split ...")
    train_ds = CIFAR10Patches("train")
    log("Loading CIFAR-10 test split ...")
    test_ds = CIFAR10Patches("test")

    # Pre-move all train patches to CPU as a plain tensor
    train_patches = train_ds.patches  # (50000, 64, 48) float32, already a tensor

    # Val patches: first val_images from test split, move to device once
    val_patches = test_ds.patches[: args.val_images].to(device)  # (1000, 64, 48)

    log(f"train_patches: {tuple(train_patches.shape)}, val_patches: {tuple(val_patches.shape)}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    cfg = ImageAOGPTConfig()
    model = ImageAOGPT(cfg)
    model.to(device)
    model.train()

    param_count = sum(p.numel() for p in model.parameters())
    log(f"Total parameters: {param_count:,}")

    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.lr,
        betas=(args.beta1, args.beta2),
        device_type="cuda" if is_cuda else "cpu",
    )

    # ------------------------------------------------------------------
    # TSV header
    # ------------------------------------------------------------------
    eval_tsv = output_dir / "eval_curve.tsv"
    write_header = not eval_tsv.exists()

    # ------------------------------------------------------------------
    # Training state
    # ------------------------------------------------------------------
    train_losses = []
    best_val_random = float("inf")
    best_step = -1
    best_state = None

    t0 = time.time()
    running_losses = []

    # ------------------------------------------------------------------
    # Helper: eval + log + TSV append
    # ------------------------------------------------------------------
    def do_eval(step: int, current_train_loss: float, current_lr: float):
        nonlocal best_val_random, best_step, best_state
        val_rand, val_ar = evaluate(
            model, val_patches, args.batch_size, args.max_eval_batches, device
        )
        elapsed = time.time() - t0
        log(
            f"[eval] step={step:5d} | train_loss={current_train_loss:.4f} | "
            f"val_random={val_rand:.4f} | val_raster={val_ar:.4f} | "
            f"lr={current_lr:.2e} | elapsed={elapsed:.1f}s"
        )

        # TSV append
        mode = "w" if write_header else "a"
        with open(eval_tsv, mode) as f:
            if write_header:
                f.write("step\ttrain_loss\tval_random\tval_raster\tlr\n")
            f.write(f"{step}\t{current_train_loss:.6f}\t{val_rand:.6f}\t{val_ar:.6f}\t{current_lr:.6e}\n")

        # Best-ckpt tracking (clone to CPU to avoid GPU memory bloat)
        if val_rand < best_val_random:
            best_val_random = val_rand
            best_step = step
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            log(f"  ** new best val_random={best_val_random:.4f} at step {best_step}")

        return val_rand

    # ------------------------------------------------------------------
    # Step-0 eval
    # ------------------------------------------------------------------
    step0_val_rand = do_eval(step=0, current_train_loss=float("nan"), current_lr=args.lr)
    # After header written, always append
    write_header = False

    # ------------------------------------------------------------------
    # Train loop
    # ------------------------------------------------------------------
    for step in range(args.max_steps):
        # LR schedule
        current_lr = get_lr(step, args.warmup_iters, args.max_steps, args.lr, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        # Sample batch
        idx = rng.integers(0, len(train_ds), size=args.batch_size)
        batch = train_patches[idx].to(device)

        # Forward
        _, loss = model(batch, mode="Random")

        # Backward + grad clip + step
        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        loss_val = loss.item()
        train_losses.append(loss_val)
        running_losses.append(loss_val)

        # Log interval
        if (step + 1) % args.log_interval == 0:
            elapsed = time.time() - t0
            recent_mean = float(np.mean(running_losses[-args.log_interval:]))
            log(
                f"step={step+1:5d} | loss={loss_val:.4f} | mean(last {args.log_interval})={recent_mean:.4f} | "
                f"lr={current_lr:.2e} | elapsed={elapsed:.1f}s"
            )

        # Eval interval (skip step 0 which is already done; also eval at last step)
        is_last_step = (step == args.max_steps - 1)
        if ((step + 1) % args.eval_interval == 0) or is_last_step:
            do_eval(
                step=step + 1,
                current_train_loss=float(np.mean(running_losses[-args.log_interval:])),
                current_lr=current_lr,
            )

    # ------------------------------------------------------------------
    # Final checkpoint save
    # ------------------------------------------------------------------
    final_ckpt_path = output_dir / f"ckpt_step{args.max_steps}.pt"
    save_dict = {
        "model_state_dict": best_state if best_state is not None else model.state_dict(),
        "config": dict(
            n_patches=64,
            patch_dim=48,
            n_embd=256,
            n_layer=4,
            n_head=8,
            cond_dim=128,
            dropout=0.0,
            bias=True,
        ),
        "args": vars(args),
        "train_losses": train_losses,
        "best_val_random": best_val_random,
        "best_step": best_step,
    }
    torch.save(save_dict, final_ckpt_path)
    log(f"Saved checkpoint: {final_ckpt_path}")

    # ------------------------------------------------------------------
    # Smoke acceptance check
    # ------------------------------------------------------------------
    final_val_rand = best_val_random
    improvement = (step0_val_rand - final_val_rand) / max(step0_val_rand, 1e-9)
    log(f"\n=== Smoke acceptance check ===")
    log(f"step-0 val_random : {step0_val_rand:.4f}")
    log(f"best val_random   : {final_val_rand:.4f} (at step {best_step})")
    log(f"improvement       : {improvement*100:.1f}%")
    if improvement >= 0.25:
        log("PASSED: val_random improved >= 25% from step-0.")
    else:
        log("WARNING: val_random improved < 25% from step-0 — DONE_WITH_CONCERNS.")

    log(f"\nFiles written:")
    for p in sorted(output_dir.iterdir()):
        log(f"  {p}")

    log_file.close()


if __name__ == "__main__":
    main()
