#!/usr/bin/env python3
"""From-0 fixed-order image AOGPT trainer (raster-only baseline for alt comparison)."""
from __future__ import annotations
import argparse, json, math, pickle, sys, time
from pathlib import Path
import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "block_lo_arm_order_network"))
sys.path.insert(0, str(_REPO / "nanogpt-learned-order"))
sys.path.insert(0, str(_REPO / "scripts"))

from AOGPT import AOGPTConfig, AOGPT
from directed_graph_policy import build_directed_graph
from train_vq64_round2 import _forward_with_block_orders, evaluate_7orders, get_lr

N_BLOCKS = 64
GRID = 8
DEFAULT_MODEL_ARGS = dict(vocab_size=8192, n_layer=4, n_head=8,
                          n_embd=256, dropout=0.0, bias=False, order_impl="block")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-train", required=True)
    p.add_argument("--data-val", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--order", choices=["raster", "random"], default="raster")
    p.add_argument("--max-steps", type=int, default=40000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--warmup-iters", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.99)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-interval", type=int, default=1000)
    p.add_argument("--max-eval-batches", type=int, default=16)
    p.add_argument("--save-steps", type=str, default="10000,20000,30000,40000")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--block-len", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    with open(args.meta, "rb") as f: meta = pickle.load(f)
    tokens_per_image = int(meta["tokens_per_image"])
    block_len = args.block_len if args.block_len is not None else int(meta.get("block_order_block_len", 1))
    block_size = N_BLOCKS * block_len
    assert tokens_per_image == block_size
    train_mm = np.memmap(args.data_train, dtype=np.uint16, mode="r")
    val_mm = np.memmap(args.data_val, dtype=np.uint16, mode="r")
    n_train = len(train_mm) // block_size
    n_val = min(2000, len(val_mm) // block_size)
    val_tokens = torch.from_numpy(
        np.asarray(val_mm[:n_val * block_size], dtype=np.int64).reshape(n_val, block_size))

    model_args = dict(DEFAULT_MODEL_ARGS, block_size=block_size, block_order_block_len=block_len)
    model = AOGPT(AOGPTConfig(**model_args)).to(device)
    optimizer = model.configure_optimizers(args.weight_decay, args.lr,
                                           (args.beta1, args.beta2), args.device.split(":")[0])
    model.train()

    json.dump(vars(args), open(out / "config.json", "w"), indent=2)
    tsv = out / "eval_curve.tsv"
    header = ("step\ttrain_loss\tlr\tval_random\tval_raster\tval_hilbert\t"
              "val_Bcov_balanced\tval_distance_only_coverage\t"
              "val_rw_top4_eps0\tval_rw_eps015\tval_rw_topk8\n")
    if not tsv.exists(): tsv.write_text(header)
    log_f = open(out / "train_log.txt", "a")
    def log(m): print(m, flush=True); log_f.write(m + "\n"); log_f.flush()
    save_steps = {int(x) for x in args.save_steps.split(",") if x}

    # Pre-build raster order
    fixed_order = torch.arange(N_BLOCKS, device=device)
    raster_order_64 = fixed_order.clone()

    # Dummy B for evaluate_7orders (not used by val_random/val_raster but required by graph_rw arms)
    B_dummy = np.eye(N_BLOCKS, dtype=np.float64)

    log(f"[start] fixed_order={args.order} max_steps={args.max_steps} "
        f"eff_batch={args.batch_size*args.grad_accum} block_len={block_len}")

    def get_batch():
        idxs = rng.integers(0, n_train, size=args.batch_size)
        toks = np.stack([np.asarray(train_mm[i*block_size:(i+1)*block_size], dtype=np.int64) for i in idxs])
        return torch.from_numpy(toks).to(device, non_blocking=True)

    @torch.no_grad()
    def run_eval(step, train_loss, lr):
        cols = {f"val_{k}": v for k, v in evaluate_7orders(
            model, val_tokens, B_dummy, raster_order_64, device,
            args.batch_size, args.max_eval_batches, step,
            fixed_token_perm=None, inv_block_perm=None).items()}
        with open(tsv, "a") as f:
            f.write(f"{step}\t{train_loss:.4f}\t{lr:.2e}\t"
                    f"{cols['val_random']:.4f}\t{cols['val_raster']:.4f}\t{cols['val_hilbert']:.4f}\t"
                    f"{cols['val_Bcov_balanced']:.4f}\t{cols['val_distance_only_coverage']:.4f}\t"
                    f"{cols['val_rw_top4_eps0']:.4f}\t{cols['val_rw_eps015']:.4f}\t"
                    f"{cols['val_rw_topk8']:.4f}\n")
        log(f"[eval] step={step} train={train_loss:.4f} "
            f"rnd={cols['val_random']:.4f} ras={cols['val_raster']:.4f}")

    t0 = time.time(); running = []
    for step in range(args.max_steps + 1):
        lr_now = get_lr(step, args)
        for g in optimizer.param_groups: g["lr"] = lr_now

        if step > 0:
            optimizer.zero_grad(set_to_none=True)
            for micro in range(args.grad_accum):
                x = get_batch()
                if args.order == "raster":
                    orders = fixed_order.unsqueeze(0).expand(args.batch_size, -1)
                else:
                    orders = torch.stack([torch.randperm(N_BLOCKS, device=device)
                                          for _ in range(args.batch_size)])
                loss = _forward_with_block_orders(model, x, orders,
                                                  fixed_token_perm=None, inv_block_perm=None)
                (loss / args.grad_accum).backward()
                running.append(float(loss.item()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        if step % args.eval_interval == 0:
            run_eval(step, float(np.mean(running[-100:])) if running else float("nan"), lr_now)

        if step > 0 and step in save_steps:
            torch.save({"model": model.state_dict(), "model_args": model_args, "step": step,
                        "config": vars(args)}, out / f"ckpt_step{step}.pt")
            log(f"[save] ckpt_step{step}.pt")

    log(f"[done] step {args.max_steps}"); log_f.close()


if __name__ == "__main__":
    main()
