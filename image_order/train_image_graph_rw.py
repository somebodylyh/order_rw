"""Alpha-mixed Graph-RW continuation training for ImageAOGPT.

Usage (smoke test, 200 steps):
    python -u image_order/train_image_graph_rw.py \
        --baseline-ckpt probe_results_image/baseline/baseline10k/ckpt_step10000.pt \
        --b-path probe_results_image/attention/baseline10k/B_global.npy \
        --output-dir probe_results_image/graph_rw/smoke200_top4 \
        --max-steps 200 --batch-size 64 --eval-interval 100 \
        --alpha 0.9 --rw-top-k 4 --rw-epsilon 0.0
"""

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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "image_order"))

from data_image_patches import CIFAR10Patches
from model_image_aogpt import ImageAOGPT, ImageAOGPTConfig
from graph_rw_image import IMAGE_RW_PARAMS_DEFAULT, IMAGE_RW_PARAMS_V2, IMAGE_RW_PARAMS_V3, sample_image_orders_batch


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Alpha-mixed Graph-RW continuation training for ImageAOGPT"
    )
    parser.add_argument("--baseline-ckpt", type=str, required=True)
    parser.add_argument("--b-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--min-lr", type=float, default=3e-6)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--max-eval-batches", type=int, default=16)
    parser.add_argument("--val-images", type=int, default=1000)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=42)
    # alpha schedule
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--alpha-warmup", type=int, default=0)
    # RW policy config
    parser.add_argument("--rw-top-k", type=int, default=4)
    parser.add_argument("--rw-epsilon", type=float, default=0.0)
    parser.add_argument("--rw-tau-start", type=float, default=0.10)
    parser.add_argument("--rw-tau-step", type=float, default=0.10)
    parser.add_argument("--rw-policy", type=str, default="progressive_rw",
                       choices=["progressive_rw", "progressive_rw_v2", "progressive_rw_v3"])
    parser.add_argument("--rw-lam", type=float, default=1.0,
                       help="λ trade-off for dependency penalty (v2/v3)")
    parser.add_argument("--rw-rho", type=float, default=0.2,
                       help="ρ global readiness prior strength (v3 only)")
    parser.add_argument("--save-steps", type=str, default="",
                       help="comma-separated step numbers to save intermediate ckpts, e.g. '5000,10000'")
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
# Alpha schedule
# ---------------------------------------------------------------------------

def get_alpha(step: int, alpha: float, alpha_warmup: int) -> float:
    if alpha_warmup <= 0:
        return alpha
    return min(alpha, alpha * step / max(alpha_warmup, 1))


# ---------------------------------------------------------------------------
# Frozen 5-order eval
# ---------------------------------------------------------------------------

EVAL_SEED_BASE = 9999

FROZEN_EVAL_CONFIGS = {
    "rw_top4_eps0": {**IMAGE_RW_PARAMS_DEFAULT, "top_k": 4, "epsilon_uniform": 0.0},
    "rw_eps015":    {**IMAGE_RW_PARAMS_DEFAULT, "top_k": 0, "epsilon_uniform": 0.15},
    "rw_topk8":     {**IMAGE_RW_PARAMS_DEFAULT, "top_k": 8, "epsilon_uniform": 0.0},
}


@torch.no_grad()
def evaluate_5orders(
    model,
    val_patches: torch.Tensor,
    B_global: np.ndarray,
    batch_size: int,
    max_eval_batches: int,
    step: int,
    device: str,
) -> dict:
    """Run frozen 5-order eval; return dict with keys matching TSV columns."""
    model.eval()
    V = val_patches.shape[0]
    n_batches = min(max_eval_batches, math.ceil(V / batch_size))

    results = {name: [] for name in ["random", "raster", "rw_top4_eps0", "rw_eps015", "rw_topk8"]}

    for bi in range(n_batches):
        start = bi * batch_size
        end = min(start + batch_size, V)
        batch = val_patches[start:end]
        B_actual = batch.shape[0]

        # random
        rand_orders = torch.stack([torch.randperm(64, device=device) for _ in range(B_actual)])
        _, loss = model(batch, mode=None, orders=rand_orders)
        results["random"].append(loss.item())

        # raster
        raster_orders = torch.arange(64, device=device).unsqueeze(0).expand(B_actual, -1)
        _, loss = model(batch, mode=None, orders=raster_orders)
        results["raster"].append(loss.item())

        # three RW configs
        for name, params in FROZEN_EVAL_CONFIGS.items():
            seed = EVAL_SEED_BASE * 1000 + step * 10 + bi
            rw_orders = sample_image_orders_batch(
                B_global, params, B_actual, seed_base=seed, step=0, device=device
            )
            _, loss = model(batch, mode=None, orders=rw_orders)
            results[name].append(loss.item())

    model.train()
    return {k: float(np.mean(v)) for k, v in results.items()}


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

    log("=== train_image_graph_rw.py ===")
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
    # Load B
    # ------------------------------------------------------------------
    B_global = np.load(args.b_path)
    assert B_global.shape == (64, 64), f"B shape must be (64, 64), got {B_global.shape}"
    log(f"Loaded B_global from {args.b_path}, shape={B_global.shape}")

    # ------------------------------------------------------------------
    # Build RW params
    # ------------------------------------------------------------------
    if args.rw_policy == "progressive_rw_v3":
        rw_defaults = IMAGE_RW_PARAMS_V3
    elif args.rw_policy == "progressive_rw_v2":
        rw_defaults = IMAGE_RW_PARAMS_V2
    else:
        rw_defaults = IMAGE_RW_PARAMS_DEFAULT
    rw_params = rw_defaults.copy()
    rw_params["top_k"] = args.rw_top_k
    rw_params["epsilon_uniform"] = args.rw_epsilon
    rw_params["tau_start"] = args.rw_tau_start
    rw_params["tau_step"] = args.rw_tau_step
    if args.rw_policy in ("progressive_rw_v2",):
        rw_params["lam"] = args.rw_lam
    elif args.rw_policy == "progressive_rw_v3":
        rw_params["lam"] = args.rw_lam
        rw_params["rho"] = args.rw_rho
    log(f"RW policy={args.rw_policy}, params: {rw_params}")

    # ------------------------------------------------------------------
    # Load baseline checkpoint
    # ------------------------------------------------------------------
    log(f"Loading baseline ckpt from {args.baseline_ckpt} ...")
    ckpt = torch.load(args.baseline_ckpt, map_location=device, weights_only=False)
    saved_config = ckpt.get("config", {})
    log(f"Saved config: {saved_config}")

    # Filter unknown keys to avoid dataclass __init__ errors
    valid_fields = set(inspect.signature(ImageAOGPTConfig.__init__).parameters.keys()) - {"self"}
    filtered_config = {k: v for k, v in saved_config.items() if k in valid_fields}
    cfg = ImageAOGPTConfig(**filtered_config)
    model = ImageAOGPT(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model = torch.compile(model, mode="reduce-overhead")
    model.train()

    param_count = sum(p.numel() for p in model.parameters())
    log(f"Total parameters: {param_count:,}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    log("Loading CIFAR-10 train split ...")
    train_ds = CIFAR10Patches("train")
    log("Loading CIFAR-10 test split ...")
    test_ds = CIFAR10Patches("test")

    train_patches = train_ds.patches  # (50000, 64, 48) float32, CPU tensor
    val_patches = test_ds.patches[: args.val_images].to(device)  # (val_images, 64, 48)
    log(f"train_patches: {tuple(train_patches.shape)}, val_patches: {tuple(val_patches.shape)}")

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.lr,
        betas=(args.beta1, args.beta2),
        device_type="cuda" if is_cuda else "cpu",
    )

    # ------------------------------------------------------------------
    # TSV setup
    # ------------------------------------------------------------------
    eval_tsv = output_dir / "eval_curve.tsv"
    tsv_header = "step\ttrain_loss\ttrain_loss_random\ttrain_loss_rw\talpha\tval_random\tval_raster\tval_rw_top4_eps0\tval_rw_eps015\tval_rw_topk8\tlr\n"
    write_header = not eval_tsv.exists()

    # ------------------------------------------------------------------
    # Training state
    # ------------------------------------------------------------------
    train_losses = []
    train_losses_random = []
    train_losses_rw = []
    best_val_rw_top4_eps0 = float("inf")
    best_step = -1
    best_state = None

    t0 = time.time()
    running_losses = []
    running_losses_random = []
    running_losses_rw = []

    # ------------------------------------------------------------------
    # Helper: eval + log + TSV append
    # ------------------------------------------------------------------
    def do_eval(step: int, current_train_loss: float,
                current_train_loss_random: float, current_train_loss_rw: float,
                current_alpha: float, current_lr: float):
        nonlocal best_val_rw_top4_eps0, best_step, best_state, write_header

        val_results = evaluate_5orders(
            model, val_patches, B_global,
            args.batch_size, args.max_eval_batches, step, device
        )
        elapsed = time.time() - t0

        log(
            f"[eval] step={step:5d} | "
            f"train_loss={current_train_loss:.4f} | "
            f"train_rnd={current_train_loss_random:.4f} | "
            f"train_rw={current_train_loss_rw:.4f} | "
            f"alpha={current_alpha:.3f} | "
            f"val_random={val_results['random']:.4f} | "
            f"val_raster={val_results['raster']:.4f} | "
            f"val_rw_top4_eps0={val_results['rw_top4_eps0']:.4f} | "
            f"val_rw_eps015={val_results['rw_eps015']:.4f} | "
            f"val_rw_topk8={val_results['rw_topk8']:.4f} | "
            f"lr={current_lr:.2e} | elapsed={elapsed:.1f}s"
        )

        # TSV append
        with open(eval_tsv, "w" if write_header else "a") as f:
            if write_header:
                f.write(tsv_header)
            f.write(
                f"{step}\t{current_train_loss:.6f}\t"
                f"{current_train_loss_random:.6f}\t{current_train_loss_rw:.6f}\t"
                f"{current_alpha:.6f}\t"
                f"{val_results['random']:.6f}\t{val_results['raster']:.6f}\t"
                f"{val_results['rw_top4_eps0']:.6f}\t{val_results['rw_eps015']:.6f}\t"
                f"{val_results['rw_topk8']:.6f}\t{current_lr:.6e}\n"
            )
        write_header = False

        # Best-ckpt tracking
        if val_results["rw_top4_eps0"] < best_val_rw_top4_eps0:
            best_val_rw_top4_eps0 = val_results["rw_top4_eps0"]
            best_step = step
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            log(f"  ** new best val_rw_top4_eps0={best_val_rw_top4_eps0:.4f} at step {best_step}")

        return val_results

    # ------------------------------------------------------------------
    # Step-0 eval
    # ------------------------------------------------------------------
    step0_alpha = get_alpha(0, args.alpha, args.alpha_warmup)
    step0_results = do_eval(
        step=0,
        current_train_loss=float("nan"),
        current_train_loss_random=float("nan"),
        current_train_loss_rw=float("nan"),
        current_alpha=step0_alpha,
        current_lr=args.lr,
    )

    # Smoke sanity check
    step0_val_random = step0_results["random"]
    if abs(step0_val_random - 0.060) > 0.020:
        log(
            f"ERROR: step-0 val_random={step0_val_random:.4f} differs from "
            f"baseline10k (~0.060) by more than 0.02 — ckpt loading likely wrong. HALTING."
        )
        log_file.close()
        raise SystemExit(1)
    log(f"Smoke check passed: step-0 val_random={step0_val_random:.4f}")

    def save_dict_fn():
        return {
            "model_state_dict": best_state if best_state is not None else {k: v.cpu().clone() for k, v in model.state_dict().items()},
            "config": saved_config,
            "args": vars(args),
            "train_losses": train_losses,
            "train_losses_random": train_losses_random,
            "train_losses_rw": train_losses_rw,
            "best_val_rw_top4_eps0": best_val_rw_top4_eps0,
            "best_step": best_step,
            "rw_params": rw_params,
        }

    save_steps = set()
    if args.save_steps:
        save_steps = set(int(x.strip()) for x in args.save_steps.split(",") if x.strip())

    # ------------------------------------------------------------------
    # Train loop
    # ------------------------------------------------------------------
    for step in range(args.max_steps):
        # LR schedule
        current_lr = get_lr(step, args.warmup_iters, args.max_steps, args.lr, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        # Alpha schedule
        current_alpha = get_alpha(step, args.alpha, args.alpha_warmup)

        # Sample batch
        idx = rng.integers(0, len(train_ds), size=args.batch_size)
        batch = train_patches[idx].to(device)

        # Build orders
        rand_orders = torch.stack([torch.randperm(64, device=device) for _ in range(args.batch_size)])
        rw_orders = sample_image_orders_batch(
            B_global, rw_params, args.batch_size,
            seed_base=args.seed, step=step, device=device, policy=args.rw_policy,
        )

        # Forward
        _, loss_random = model(batch, mode=None, orders=rand_orders)
        _, loss_rw = model(batch, mode=None, orders=rw_orders)
        loss = (1.0 - current_alpha) * loss_random + current_alpha * loss_rw

        # Backward + grad clip + step
        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        loss_val = loss.item()
        loss_random_val = loss_random.item()
        loss_rw_val = loss_rw.item()

        train_losses.append(loss_val)
        train_losses_random.append(loss_random_val)
        train_losses_rw.append(loss_rw_val)
        running_losses.append(loss_val)
        running_losses_random.append(loss_random_val)
        running_losses_rw.append(loss_rw_val)

        # Log interval
        if (step + 1) % args.log_interval == 0:
            elapsed = time.time() - t0
            recent_loss = float(np.mean(running_losses[-args.log_interval:]))
            recent_rnd = float(np.mean(running_losses_random[-args.log_interval:]))
            recent_rw = float(np.mean(running_losses_rw[-args.log_interval:]))
            log(
                f"step={step+1:5d} | loss={loss_val:.4f} | "
                f"mean(last {args.log_interval})={recent_loss:.4f} | "
                f"rnd={recent_rnd:.4f} | rw={recent_rw:.4f} | "
                f"alpha={current_alpha:.3f} | lr={current_lr:.2e} | elapsed={elapsed:.1f}s"
            )

        # Eval interval and last step
        is_last_step = (step == args.max_steps - 1)
        if ((step + 1) % args.eval_interval == 0) or is_last_step:
            recent_loss = float(np.mean(running_losses[-args.log_interval:]))
            recent_rnd = float(np.mean(running_losses_random[-args.log_interval:]))
            recent_rw = float(np.mean(running_losses_rw[-args.log_interval:]))
            do_eval(
                step=step + 1,
                current_train_loss=recent_loss,
                current_train_loss_random=recent_rnd,
                current_train_loss_rw=recent_rw,
                current_alpha=current_alpha,
                current_lr=current_lr,
            )

        # Intermediate save
        if (step + 1) in save_steps:
            ckpt_path = output_dir / f"ckpt_step{step+1}.pt"
            torch.save(save_dict_fn(), ckpt_path)
            log(f"Saved intermediate ckpt: {ckpt_path}")

    # ------------------------------------------------------------------
    # Final checkpoint
    # ------------------------------------------------------------------
    final_ckpt_path = output_dir / f"ckpt_step{args.max_steps}.pt"
    save_dict = save_dict_fn()
    torch.save(save_dict, final_ckpt_path)
    log(f"Saved checkpoint: {final_ckpt_path}")

    log(f"\n=== Summary ===")
    log(f"best val_rw_top4_eps0 : {best_val_rw_top4_eps0:.4f} at step {best_step}")
    log(f"\nFiles written:")
    for p in sorted(output_dir.iterdir()):
        log(f"  {p}")

    log_file.close()


if __name__ == "__main__":
    main()
