#!/usr/bin/env python3
"""Single-arm from-0 image MLP-refresh alternating trainer (self-bootstrapping feasibility).
See docs/superpowers/specs/2026-05-24-image-mlp-refresh-alternating-design.md."""
from __future__ import annotations
import argparse, json, math, pickle, sys, time
from pathlib import Path
import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "block_lo_arm_order_network"))
sys.path.insert(0, str(_REPO / "nanogpt-learned-order"))
sys.path.insert(0, str(_REPO / "image_order"))
sys.path.insert(0, str(_REPO / "scripts"))

from AOGPT import AOGPTConfig, AOGPT
from directed_graph_policy import build_directed_graph
from extract_image_attention_e2 import extract_a_global
from attn_order_distill import distill_order_mlp
from attn_order_image_diag import image_refresh_diagnostics
import attn_order_mlp_policy as P
# reuse round2 helpers without modifying it:
from train_vq64_round2 import _forward_with_block_orders, evaluate_7orders, get_lr

N_BLOCKS = 64
GRID = 8  # 8×8 block grid (both for 1-token and 4-token block_len)
DEFAULT_MODEL_ARGS = dict(vocab_size=8192, n_layer=8, n_head=8,
                          n_embd=512, dropout=0.0, bias=False, order_impl="block")


def build_or_load_model_for_extraction(ckpt_path, device):
    """Test/utility: load an existing AOGPT ckpt (used only by extraction tests)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ma = ckpt["model_args"]
    # Use the ckpt's own block_size/block_order_block_len + shared defaults for the rest
    model_args = dict(DEFAULT_MODEL_ARGS,
                      block_size=ma["block_size"],
                      block_order_block_len=ma.get("block_order_block_len", 1))
    model = AOGPT(AOGPTConfig(**model_args))
    sd = ckpt["model"]
    if all(k.startswith("_orig_mod.") for k in sd):
        sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model, model_args


def extract_B_from_model(model, data_tokens, tokens_per_image, n_images, m_passes, device, seed):
    """Extract A_global from the CURRENT model (un-permuted physical coords) and return B=A^T.

    Reuses extract_image_attention_e2.extract_a_global verbatim; seeds torch so the random
    AO orders (hence the snapshot) are reproducible.

    When block_len > 1, token-level attention (T×T) is aggregated to block-level (64×64)
    by mean-pooling within each block before building the directed graph.
    """
    was_training = model.training
    model.eval()
    torch.manual_seed(int(seed))
    A = extract_a_global(model, data_tokens, tokens_per_image, n_images, m_passes, device)
    if was_training:
        model.train()
    # Aggregate token-level → block-level when block_len > 1
    T = A.shape[0]
    blk_len = T // N_BLOCKS
    if blk_len > 1:
        A = A.reshape(N_BLOCKS, blk_len, N_BLOCKS, blk_len).mean(axis=(1, 3))
    B = build_directed_graph(np.ascontiguousarray(A.astype(np.float64)))
    return B


def get_alpha_alt(step, *, warmup_start, ramp, alpha_max):
    """0 until warmup_start; linear ramp to alpha_max over `ramp` steps; then flat."""
    if step < warmup_start:
        return 0.0
    if ramp <= 0:
        return alpha_max
    frac = (step - warmup_start) / float(ramp)
    return float(min(alpha_max, alpha_max * max(0.0, frac)))


@torch.no_grad()
def evaluate_with_mlp(model, val_tokens, B, *, mlp, device, batch_size, max_eval_batches,
                      step, tau=0.5, top_k=4):
    """8 fixed-order NLLs (reused) + val_mlp_order. mlp=None -> val_mlp_order == val_random."""
    raster_order = torch.arange(N_BLOCKS, device=device)
    cols = {f"val_{k}": v for k, v in evaluate_7orders(
        model, val_tokens, B, raster_order, device, batch_size, max_eval_batches, step,
        fixed_token_perm=None, inv_block_perm=None).items()}

    if mlp is None:
        cols["val_mlp_order"] = cols["val_random"]
        return cols

    model.eval()
    V = val_tokens.shape[0]
    n_batches = min(max_eval_batches, math.ceil(V / batch_size))
    losses = []
    for bi in range(n_batches):
        s, e = bi * batch_size, min((bi + 1) * batch_size, V)
        x = val_tokens[s:e].to(device)
        mlp_orders = P.sample_orders_batched_mlp(
            B, x.shape[0], mlp, "original", base_seed=7_000_000 + step + bi,
            device=torch.device(device), tau=tau, top_k=top_k)
        loss = _forward_with_block_orders(model, x, mlp_orders,
                                          fixed_token_perm=None, inv_block_perm=None)
        losses.append(float(loss.item()))
    cols["val_mlp_order"] = float(np.mean(losses))
    return cols


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-train", required=True)
    p.add_argument("--data-val", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-steps", type=int, default=30000)
    p.add_argument("--warmup-start", type=int, default=3000, help="alpha=0 until this step (random warmup)")
    p.add_argument("--alpha-ramp", type=int, default=10000)
    p.add_argument("--alpha-max", type=float, default=0.9)
    p.add_argument("--refresh-interval", type=int, default=3000)
    p.add_argument("--first-refresh", type=int, default=3000, help="step of the first refresh (== warmup end)")
    p.add_argument("--extract-n-images", type=int, default=500)
    p.add_argument("--extract-m-passes", type=int, default=3)
    p.add_argument("--mlp-tau", type=float, default=0.5)
    p.add_argument("--mlp-top-k", type=int, default=4)
    p.add_argument("--mlp-refresh-mode", choices=["finetune", "scratch"], default="finetune")
    p.add_argument("--distill-n-orders", type=int, default=200)
    p.add_argument("--distill-epochs", type=int, default=60)
    p.add_argument("--distill-tau-t", type=float, default=0.5)
    p.add_argument("--distill-tau-train", type=float, default=0.5)
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
    p.add_argument("--save-steps", type=str, default="3000,15000,30000")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--block-len", type=int, default=None, help="tokens per block (1=8x8 single-token, 4=2x2 patch); inferred from meta if omitted")
    p.add_argument("--resume-from", default=None, help="path to ckpt_step{N}.pt to resume from")
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
    assert tokens_per_image == block_size, f"tokens_per_image={tokens_per_image} != block_size={block_size}"
    train_mm = np.memmap(args.data_train, dtype=np.uint16, mode="r")
    val_mm = np.memmap(args.data_val, dtype=np.uint16, mode="r")
    n_train = len(train_mm) // block_size
    n_val = min(2000, len(val_mm) // block_size)
    val_tokens = torch.from_numpy(
        np.asarray(val_mm[:n_val * block_size], dtype=np.int64).reshape(n_val, block_size))

    start_step = 0
    rw_mlp = None
    B = np.zeros((N_BLOCKS, N_BLOCKS), dtype=np.float64)

    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        model_args = ckpt["model_args"]
        block_len = model_args.get("block_order_block_len", 1)
        block_size = model_args["block_size"]
        model = AOGPT(AOGPTConfig(**model_args)).to(device)
        sd = ckpt["model"]
        if all(k.startswith("_orig_mod.") for k in sd):
            sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        optimizer = model.configure_optimizers(args.weight_decay, args.lr,
                                               (args.beta1, args.beta2), args.device.split(":")[0])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        # Try to load latest beta + B
        beta_path = out / f"beta_step{start_step}.pt"
        if beta_path.exists():
            from train_attn_order_mlp import OrderMLP
            rw_mlp = OrderMLP()
            rw_mlp.load_state_dict(torch.load(beta_path, map_location=device, weights_only=False))
            rw_mlp.eval()
        a_path = out / f"A_global_step{start_step}.npy"
        if a_path.exists():
            B = np.load(a_path).T.copy()   # saved as B^T
            np.fill_diagonal(B, 0.0)
        model.train()
    else:
        model_args = dict(DEFAULT_MODEL_ARGS, block_size=block_size, block_order_block_len=block_len)
        model = AOGPT(AOGPTConfig(**model_args)).to(device)
        optimizer = model.configure_optimizers(args.weight_decay, args.lr,
                                               (args.beta1, args.beta2), args.device.split(":")[0])
        model.train()

    next_refresh = args.first_refresh if not args.resume_from else (
        start_step + args.refresh_interval - (start_step % args.refresh_interval))

    json.dump({**vars(args), "start_step": start_step}, open(out / "config.json", "w"), indent=2)
    tsv = out / "eval_curve.tsv"
    header = ("step\ttrain_loss\talpha\tlr\tval_random\tval_raster\tval_hilbert\t"
              "val_Bcov_balanced\tval_distance_only_coverage\tval_rw_top4_eps0\t"
              "val_rw_eps015\tval_rw_topk8\tval_mlp_order\n")
    if not tsv.exists(): tsv.write_text(header)
    log_f = open(out / "train_log.txt", "a")
    def log(m): print(m, flush=True); log_f.write(m + "\n"); log_f.flush()
    save_steps = {int(x) for x in args.save_steps.split(",") if x}

    tag = "RESUME" if args.resume_from else "FROM-0"
    log(f"[start] {tag} alternating max_steps={args.max_steps} start_step={start_step} "
        f"warmup_start={args.warmup_start} alpha_ramp={args.alpha_ramp} alpha_max={args.alpha_max} "
        f"refresh_interval={args.refresh_interval} eff_batch={args.batch_size*args.grad_accum}")

    def get_batch():
        idxs = rng.integers(0, n_train, size=args.batch_size)
        toks = np.stack([np.asarray(train_mm[i*block_size:(i+1)*block_size], dtype=np.int64) for i in idxs])
        return torch.from_numpy(toks).to(device, non_blocking=True)

    def run_eval(step, train_loss, lr, alpha):
        cols = evaluate_with_mlp(model, val_tokens, B, mlp=rw_mlp, device=device,
                                 batch_size=args.batch_size, max_eval_batches=args.max_eval_batches,
                                 step=step, tau=args.mlp_tau, top_k=args.mlp_top_k)
        with open(tsv, "a") as f:
            f.write(f"{step}\t{train_loss:.4f}\t{alpha:.4f}\t{lr:.2e}\t"
                    f"{cols['val_random']:.4f}\t{cols['val_raster']:.4f}\t{cols['val_hilbert']:.4f}\t"
                    f"{cols['val_Bcov_balanced']:.4f}\t{cols['val_distance_only_coverage']:.4f}\t"
                    f"{cols['val_rw_top4_eps0']:.4f}\t{cols['val_rw_eps015']:.4f}\t"
                    f"{cols['val_rw_topk8']:.4f}\t{cols['val_mlp_order']:.4f}\n")
        log(f"[eval] step={step} train={train_loss:.4f} a={alpha:.3f} "
            f"rnd={cols['val_random']:.4f} ras={cols['val_raster']:.4f} mlp={cols['val_mlp_order']:.4f}")

    t0 = time.time(); running = []
    for step in range(start_step, args.max_steps + 1):
        lr_now = get_lr(step, args)
        for g in optimizer.param_groups: g["lr"] = lr_now
        alpha = get_alpha_alt(step, warmup_start=args.warmup_start, ramp=args.alpha_ramp,
                              alpha_max=args.alpha_max)

        if step > start_step:
            optimizer.zero_grad(set_to_none=True)
            for micro in range(args.grad_accum):
                x = get_batch()
                random_orders = torch.stack([torch.randperm(N_BLOCKS, device=device)
                                             for _ in range(args.batch_size)])
                use_mlp = (rw_mlp is not None) and (alpha > 0.0)
                if not use_mlp:
                    block_orders = random_orders
                else:
                    mlp_orders = P.sample_orders_batched_mlp(
                        B, args.batch_size, rw_mlp, "original",
                        base_seed=args.seed * 100000000 + step * 1000 + micro,
                        device=device, tau=args.mlp_tau, top_k=args.mlp_top_k)
                    crng = torch.Generator(device=device)
                    crng.manual_seed(args.seed * 7 + step * 1000 + micro)
                    pick = torch.rand(args.batch_size, generator=crng, device=device) < alpha
                    block_orders = torch.where(pick.unsqueeze(1), mlp_orders, random_orders)
                loss = _forward_with_block_orders(model, x, block_orders,
                                                  fixed_token_perm=None, inv_block_perm=None)
                (loss / args.grad_accum).backward()
                running.append(float(loss.item()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        if step % args.eval_interval == 0:
            run_eval(step, float(np.mean(running[-100:])) if running else float("nan"), lr_now, alpha)

        # ---- refresh + distill (mirror train_clean_aogpt.py:1008-1049) ----
        if step > start_step and step >= next_refresh and step <= args.max_steps:
            B = extract_B_from_model(model, train_mm, tokens_per_image, args.extract_n_images,
                                     args.extract_m_passes, args.device, seed=args.seed + step)
            np.save(out / f"A_global_step{step}.npy", B.T)   # A = B^T
            init = rw_mlp if (args.mlp_refresh_mode == "finetune" and rw_mlp is not None) else None
            seed_r = args.seed * 100000 + step
            rw_mlp, ddiag = distill_order_mlp(
                B, mlp=init, n_orders=args.distill_n_orders, tau_T=args.distill_tau_t,
                tau_train=args.distill_tau_train, epochs=args.distill_epochs,
                seed=seed_r, device=str(device))
            rdiag = image_refresh_diagnostics(B, rw_mlp, tau=args.mlp_tau, top_k=args.mlp_top_k,
                                              seed=seed_r + 1, device=str(device))
            torch.save(rw_mlp.state_dict(), out / f"beta_step{step}.pt")
            with (out / "refresh_diagnostics.jsonl").open("a") as f:
                f.write(json.dumps(dict(step=int(step), refresh_mode=args.mlp_refresh_mode,
                                        **ddiag, **rdiag)) + "\n")
            log(f"[Refresh+Distill @ {step}] val_kl={ddiag['val_kl']} top1={ddiag['top1']} "
                f"top4={ddiag['top4']} | p_le1={rdiag['p_le1']} top4_follow={rdiag['top4_follow']} "
                f"B_edge={rdiag['B_edge_ratio']} ent={rdiag['rollout_entropy']}")
            next_refresh = step + args.refresh_interval

        if step > start_step and step in save_steps:
            torch.save({"model": model.state_dict(), "model_args": model_args, "step": step,
                        "config": vars(args)}, out / f"ckpt_step{step}.pt")
            log(f"[save] ckpt_step{step}.pt")

    log(f"[done] step {args.max_steps}; eval_curve={tsv}"); log_f.close()


if __name__ == "__main__":
    main()
