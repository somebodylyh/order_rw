#!/usr/bin/env python3
"""Clean AO-GPT baseline/method training with fixed data order and eval protocol."""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import torch.nn.functional as F

torch.set_float32_matmul_precision("high")

from clean_training_protocol import (
    CleanPermutation,
    batch_indices_for_step,
    build_clean_block_permutation,
    build_fixed_split_and_shuffle,
    build_phys_to_model_token_gather,
    expand_model_blocks_to_token_order,
    load_token_stream,
    model_blocks_to_physical_blocks,
    physical_blocks_to_model_blocks,
    physical_blocks_to_model_token_order,
    phys_to_model_idx_clean,
    sample_stream_batch,
    sha256_int_array,
    train_cursor_for_next_step,
    verify_clean_coordinate_round_trip,
)
from directed_graph_policy import build_directed_graph, sample_order, sample_orders_batched_torch
from attn_order_mlp_policy import load_order_mlp, sample_orders_batched_mlp, sample_orders_batched_position
from training_utils import (
    AOGPT,
    AOGPTConfig,
    A_PATH_DEFAULT,
    BLOCK_LEN,
    N,
    SEQ_LEN,
    load_train_chunks,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "probe_results" / "clean_base_random_perm"
DEFAULT_SAVE_STEPS = "0,1000,5000,10000,20000,30000,40000,50000"
WANDB_EVAL_KEYS = [
    "val_train_objective",
    "val_ori_l2r_block",
    "val_ar_l2r",
    "val_model_order",
    "val_unstructured_order",
    "val_rw_order",
    "val_beta_order",
    "val_direct_order",
    "val_cdl_order",
]


def _metric_loss(metrics, key):
    value = metrics.get(key, {}).get("loss_token_avg", float("nan"))
    return float(value)


def wandb_train_payload(step, train_loss, alpha, lr):
    return {
        "step": int(step),
        "train_loss": float(train_loss),
        "alpha": float(alpha),
        "lr": float(lr),
    }


def wandb_eval_payload(step, metrics, train_loss, alpha, lr):
    payload = wandb_train_payload(step, train_loss, alpha, lr)
    for key in WANDB_EVAL_KEYS:
        payload[key] = _metric_loss(metrics, key)
    return payload


def maybe_init_wandb(args, output_dir, clean_perm=None, log_fn=print):
    if not getattr(args, "wandb_log", False):
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "--wandb-log requested, but the wandb package is not installed. "
            "Install wandb or rerun without --wandb-log."
        ) from exc

    config = dict(vars(args))
    config["output_dir"] = str(output_dir)
    if clean_perm is not None:
        config["block_perm_first16"] = clean_perm.block_perm_phys_to_model[:16].tolist()
        config["inv_perm_first16"] = clean_perm.inv_perm_model_to_phys[:16].tolist()

    kwargs = {
        "project": args.wandb_project,
        "name": args.wandb_run_name or Path(args.output_dir).name,
        "tags": args.wandb_tags or None,
        "config": config,
        "dir": str(output_dir),
    }
    if args.wandb_mode:
        kwargs["mode"] = args.wandb_mode

    run = wandb.init(**kwargs)
    log_fn(
        f"[wandb] logging enabled project={args.wandb_project} "
        f"name={kwargs['name']} mode={args.wandb_mode or 'default'}"
    )
    return run


def parse_step_list(text):
    if not text:
        return set()
    return {int(x) for x in str(text).split(",") if str(x).strip()}


def get_lr(global_step, args):
    if global_step < args.warmup_iters:
        return args.lr * (global_step + 1) / max(args.warmup_iters, 1)
    if global_step >= args.lr_decay_steps:
        return args.min_lr
    ratio = (global_step - args.warmup_iters) / max(args.lr_decay_steps - args.warmup_iters, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return args.min_lr + coeff * (args.lr - args.min_lr)


@torch.no_grad()
def extract_A_matrices(model, idx_chunks, clean_perm, device, n_chunks=None, seed=None):
    """Extract NxN attention matrices from current model on given chunks.

    Args:
        seed: Optional int. When provided, the per-sample random block-reveal
            permutation is drawn from a CPU torch.Generator seeded with
            ``int(seed) + i`` (where ``i`` is the chunk index), making B
            extraction bit-for-bit reproducible across runs. When ``None``
            (default), falls back to the legacy un-seeded
            ``torch.randperm(N, device='cpu')`` call, preserving exact
            backward-compatible behavior for existing callers.
    """
    if n_chunks is None:
        n_chunks = len(idx_chunks)
    n_chunks = min(n_chunks, len(idx_chunks))

    inv_perm = clean_perm.inv_perm_model_to_phys.cpu().numpy()

    A_all = np.zeros((n_chunks, N, N), dtype=np.float32)
    model.eval()

    for i in range(n_chunks):
        tokens = idx_chunks[i:i+1].to(device)  # (1, 256) model-coordinate tokens

        if seed is not None:
            gen = torch.Generator(device='cpu')
            gen.manual_seed(int(seed) + int(i))
            rand_blocks = torch.randperm(N, generator=gen, device='cpu')
        else:
            rand_blocks = torch.randperm(N, device='cpu')
        token_order = expand_model_blocks_to_token_order(
            rand_blocks.unsqueeze(0), BLOCK_LEN
        ).to(device)

        _, _, attn_list = model.forward_fn(tokens, token_order, return_attentions=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        attn_stack = torch.stack(attn_list).squeeze(1).cpu().numpy()  # (L, H, 257, 257)

        L, H = attn_stack.shape[:2]

        head_vars = np.zeros(H)
        for h in range(H):
            content = attn_stack[:, h, 1:, 1:]
            mask = ~np.eye(SEQ_LEN, dtype=bool)
            offdiag = content[:, mask].reshape(L, SEQ_LEN, SEQ_LEN - 1)
            head_vars[h] = float(np.var(offdiag))
        top_heads = np.argsort(head_vars)[-4:]
        avg_attn = attn_stack[:, top_heads, :, :].mean(axis=(0, 1))  # (257, 257)

        # Map from reveal space to physical space
        reveal_tokens = token_order[0].cpu().numpy()  # (256,) model-coordinate token positions
        model_blocks = reveal_tokens // BLOCK_LEN  # (256,) model block indices
        phys_blocks = inv_perm[model_blocks]  # (256,) physical block indices
        phys_tokens = phys_blocks * BLOCK_LEN + (reveal_tokens % BLOCK_LEN)  # (256,)

        attn_content = avg_attn[1:, 1:]  # (256, 256)
        # Vectorized physical-frame remap: equivalent to the legacy
        #   for rq in range(SEQ_LEN):
        #       for rk in range(SEQ_LEN):
        #           attn_phys[phys_tokens[rq], phys_tokens[rk]] += attn_content[rq, rk]
        # but ~7-10x faster. Bit-identity is pinned by
        # tests/test_extract_A_remap_vectorized.py across random/identity
        # permutations, both dtypes, and an explicit collision case.
        attn_phys = np.zeros((SEQ_LEN, SEQ_LEN), dtype=np.float32)
        np.add.at(attn_phys, (phys_tokens[:, None], phys_tokens[None, :]), attn_content)

        # Aggregate to NxN
        A = np.zeros((N, N), dtype=np.float32)
        for bi in range(N):
            i_s, i_e = bi * BLOCK_LEN, (bi + 1) * BLOCK_LEN
            for bj in range(N):
                j_s, j_e = bj * BLOCK_LEN, (bj + 1) * BLOCK_LEN
                A[bi, bj] = attn_phys[i_s:i_e, j_s:j_e].mean()

        # [None] signal
        none_attn = avg_attn[1:, 0]
        none_block = np.array([
            none_attn[b * BLOCK_LEN:(b + 1) * BLOCK_LEN].mean() for b in range(N)
        ])
        A += none_block[np.newaxis, :] * 0.1
        np.fill_diagonal(A, 0.0)

        A_all[i] = A

    model.train()
    return A_all


def refresh_rw_graph(model, idx_chunks, clean_perm, device, A_global_old, n_chunks, ema_beta=0.9):
    """Extract A from current model, update A_global with EMA, rebuild graph B."""
    t0 = time.time()
    A_new = extract_A_matrices(model, idx_chunks, clean_perm, device, n_chunks=n_chunks)
    A_new_global = A_new.mean(axis=0).astype(np.float32)
    np.fill_diagonal(A_new_global, 0.0)

    if A_global_old is not None:
        A_global = ema_beta * A_global_old + (1.0 - ema_beta) * A_new_global
    else:
        A_global = A_new_global

    B = build_directed_graph(A_global)
    elapsed = time.time() - t0
    return B, A_global, elapsed, A_new.shape[0]


def alpha_for_step(global_step, start_step, args):
    if args.run_kind not in {
        "graph_rw", "graph_rw_bag", "frozen_beta", "cdl_teacher",
        "direct_policy",
    }:
        return 0.0
    offset = int(getattr(args, "alpha_warmup_start", 0) or 0)
    local_step = max(0, int(global_step) - int(start_step) - offset)
    warmup = max(int(args.alpha_warmup_steps), 1)
    frac = min(1.0, local_step / warmup)
    return float(args.alpha_start + frac * (args.alpha_target - args.alpha_start))


def clean_model_args(args):
    return {
        "n_layer": int(args.n_layer),
        "n_head": int(args.n_head),
        "n_embd": int(args.n_embd),
        "block_size": SEQ_LEN,
        "bias": bool(args.bias),
        "vocab_size": int(args.vocab_size),
        "dropout": float(args.dropout),
        "block_order_block_len": BLOCK_LEN,
        "order_impl": "block",
    }


def build_model(model_args, device, compile_model=True):
    sig = list(AOGPTConfig.__init__.__code__.co_varnames)
    valid = {k: v for k, v in dict(model_args).items() if k in sig}
    model = AOGPT(AOGPTConfig(**valid))
    model.crop_block_size(SEQ_LEN)
    model.to(device)
    if compile_model:
        model = torch.compile(model, mode="reduce-overhead")
    return model


def clean_state_dict(state_dict):
    state_dict = dict(state_dict)
    for key in list(state_dict.keys()):
        clean = key.replace("_orig_mod.", "")
        if clean != key:
            state_dict[clean] = state_dict.pop(key)
    return state_dict


def sample_random_physical_orders(batch_size, seed, global_step, micro_step, device):
    rows = []
    for b in range(batch_size):
        rng = np.random.default_rng(int(seed) * 100000000 + int(global_step) * 1000 + int(micro_step) * 100 + b)
        rows.append(rng.permutation(N))
    return torch.tensor(np.stack(rows), dtype=torch.long, device=device)


def sample_token_orders_granular(batch_size, seed, global_step, micro_step,
                                  device, granularity):
    """Generate model-space token orders at a given shuffle granularity.

    granularity in {32, 64, 128}: number of independently shuffled groups.
    Each group is a contiguous chunk of group_size = SEQ_LEN // granularity tokens
    (8, 4, or 2 tokens respectively). Groups are permuted randomly per sample,
    but tokens within each group stay in contiguous left-to-right order.

    Returns a (batch_size, SEQ_LEN) long tensor of model-coordinate token indices,
    suitable for direct use as `token_orders` in model.forward_fn.
    """
    if granularity not in {32, 64, 128}:
        raise ValueError(f"granularity must be in {{32, 64, 128}}, got {granularity}")
    group_size = SEQ_LEN // granularity
    rows = []
    for b in range(batch_size):
        rng = np.random.default_rng(
            int(seed) * 100_000_000 + int(global_step) * 1000
            + int(micro_step) * 100 + b
        )
        perm_groups = rng.permutation(granularity)
        tokens = np.concatenate([
            np.arange(g * group_size, (g + 1) * group_size)
            for g in perm_groups
        ])
        rows.append(tokens)
    return torch.tensor(np.stack(rows), dtype=torch.long, device=device)


def should_sample_rw(alpha, rw_policy, rw_mlp):
    """Sample rw orders only when they will actually be used: alpha>0, and (for mlp_cdl) beta exists."""
    if alpha <= 0.0:
        return False
    if rw_policy == "mlp_cdl" and rw_mlp is None:
        return False
    return True


def sample_rw_physical_orders(batch_size, B, policy, params, seed, global_step, micro_step, device, bag_idx=0, mlp=None):
    if policy == "position_only":
        base_seed = (
            int(seed) * 100000000
            + int(global_step) * 10000
            + int(micro_step) * 1000
            + int(bag_idx) * 100
        )
        return sample_orders_batched_position(
            N, batch_size, base_seed, device,
            pos_tau=float(params["pos_tau"]), top_k=int(params.get("top_k", 4) or 0),
        )
    if policy == "mlp_cdl":
        base_seed = (
            int(seed) * 100000000
            + int(global_step) * 10000
            + int(micro_step) * 1000
            + int(bag_idx) * 100
        )
        return sample_orders_batched_mlp(
            B, batch_size, mlp, params["orientation"], base_seed, device,
            tau=float(params["mlp_tau"]), tau_start=float(params.get("tau_start", 0.1)),
            top_k=int(params.get("top_k", 4) or 0), src_rho=float(params.get("src_rho", 0.0)),
            alpha_dep=float(params.get("alpha_dep", 0.5)),
        )
    if policy == "progressive_rw_v3" and device.type == "cuda":
        base_seed = (
            int(seed) * 100000000
            + int(global_step) * 10000
            + int(micro_step) * 1000
            + int(bag_idx) * 100
        )
        return sample_orders_batched_torch(B, batch_size, policy, params, base_seed, device)

    rows = []
    for b in range(batch_size):
        order_seed = (
            int(seed) * 100000000
            + int(global_step) * 10000
            + int(micro_step) * 1000
            + int(bag_idx) * 100
            + b
        )
        order, _ = sample_order(B, policy, params, seed=order_seed)
        rows.append(order)
    return torch.tensor(np.stack(rows), dtype=torch.long, device=device)


def compute_token_ce(model, idx_batch, token_orders, device):
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        logits, model_loss = model.forward_fn(idx_batch, token_orders)
        targets = idx_batch.gather(1, token_orders)
        shift_logits = logits[:, :-1, :].contiguous()
        token_losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        ).view(idx_batch.size(0), idx_batch.size(1))
    return token_losses, model_loss


def order_loss(model, idx_batch, physical_orders, clean_perm, device):
    token_orders = physical_blocks_to_model_token_order(physical_orders, clean_perm, BLOCK_LEN).to(device)
    _, loss = model.forward_fn(idx_batch, token_orders)
    return loss


@torch.no_grad()
def evaluate_orders(model, idx_eval_model, clean_perm, B, rw_policy, rw_params, args, alpha,
                    rw_mlp=None, beta_provider=None, cdl_provider=None,
                    direct_provider=None, eval_step=0):
    model.eval()
    device = next(model.parameters()).device

    def describe(model_order):
        model_order = model_order.detach().cpu().long()
        physical_order = model_blocks_to_physical_blocks(model_order, clean_perm).cpu().long()
        ranges = []
        for phys in physical_order[:8].tolist():
            ranges.append(f"phys[{phys * BLOCK_LEN}:{(phys + 1) * BLOCK_LEN}]")
        return {
            "model_order_first16": model_order[:16].tolist(),
            "physical_order_first16": physical_order[:16].tolist(),
            "token_ranges_first8": ranges,
        }

    def eval_model_orders(order_matrices):
        total_ce = 0.0
        total_tokens = 0
        total_step_ce = 0.0
        total_steps = 0
        for orders_cpu in order_matrices:
            if orders_cpu.dim() == 1:
                orders_cpu = orders_cpu.unsqueeze(0).expand(idx_eval_model.size(0), -1)
            for start in range(0, idx_eval_model.size(0), args.eval_batch_size):
                stop = min(start + args.eval_batch_size, idx_eval_model.size(0))
                idx_batch = idx_eval_model[start:stop].to(device)
                orders = orders_cpu[start:stop].to(device)
                token_orders = expand_model_blocks_to_token_order(orders, BLOCK_LEN).to(device)
                token_losses, _ = compute_token_ce(model, idx_batch, token_orders, device)
                total_ce += float(token_losses.float().sum().item())
                total_tokens += int(token_losses.numel())
                block_losses = token_losses.float().view(idx_batch.size(0), N, BLOCK_LEN).mean(dim=-1)
                total_step_ce += float(block_losses.sum().item())
                total_steps += int(block_losses.numel())
        return {
            "loss_token_avg": total_ce / total_tokens,
            "loss_step_avg": total_step_ce / total_steps,
            "num_predicted_tokens": total_tokens,
            "num_order_steps": total_steps,
        }

    physical_l2r = torch.arange(N, dtype=torch.long)
    ori_model = physical_blocks_to_model_blocks(physical_l2r, clean_perm)
    model_ascending = torch.arange(N, dtype=torch.long)

    unstructured_model_orders = []
    rw_model_orders = []
    n_eval = idx_eval_model.size(0)
    for seed in args.eval_order_seeds:
        random_rows = [np.random.default_rng(int(seed) * 10000 + seq_idx).permutation(N)
                       for seq_idx in range(n_eval)]
        unstructured_phys = torch.tensor(np.stack(random_rows), dtype=torch.long)
        if rw_policy == "position_only":
            rw_phys = sample_orders_batched_position(
                N, n_eval, int(seed) * 10000, device,
                pos_tau=float(rw_params["pos_tau"]), top_k=int(rw_params.get("top_k", 4) or 0),
            ).cpu().long()
        elif rw_policy == "mlp_cdl":
            if rw_mlp is None:
                # alternating warmup: beta not yet born — fall back to random (same as unstructured)
                rw_phys = unstructured_phys
            else:
                rw_phys = sample_orders_batched_mlp(
                    B, n_eval, rw_mlp, rw_params["orientation"], int(seed) * 10000, device,
                    tau=float(rw_params["mlp_tau"]), tau_start=float(rw_params.get("tau_start", 0.1)),
                    top_k=int(rw_params.get("top_k", 4) or 0), src_rho=float(rw_params.get("src_rho", 0.0)),
                    alpha_dep=float(rw_params.get("alpha_dep", 0.5)),
                ).cpu().long()
        else:
            rw_rows = [sample_order(B, rw_policy, rw_params, seed=int(seed) * 10000 + seq_idx)[0]
                       for seq_idx in range(n_eval)]
            rw_phys = torch.tensor(np.stack(rw_rows), dtype=torch.long)
        unstructured_model_orders.append(physical_blocks_to_model_blocks(unstructured_phys, clean_perm))
        rw_model_orders.append(physical_blocks_to_model_blocks(rw_phys, clean_perm))

    # --- frozen_beta: compute val_beta_order from the current model state ---
    beta_model_orders = []
    if beta_provider is not None and args.run_kind == "frozen_beta":
        eval_batch = idx_eval_model[: min(args.eval_batch_size, n_eval)].to(device)
        cached_sigma = getattr(beta_provider, "_sigma", None)
        cached_last_refresh = getattr(beta_provider, "_last_refresh", None)
        if hasattr(beta_provider, "_sigma"):
            beta_provider._sigma = None
        beta_sigma = beta_provider.physical_order(model, eval_batch, int(eval_step)).cpu()
        if hasattr(beta_provider, "_sigma"):
            beta_provider._sigma = cached_sigma
            beta_provider._last_refresh = cached_last_refresh
        if beta_provider.none_mode in ("model", "content"):
            beta_model = beta_sigma  # already model frame
        else:
            beta_model = physical_blocks_to_model_blocks(beta_sigma, clean_perm)
        for _ in args.eval_order_seeds:
            beta_model_orders.append(beta_model.unsqueeze(0).expand(n_eval, -1))

    # --- cdl_teacher: compute val_cdl_order from the current model state ---
    cdl_model_orders = []
    if cdl_provider is not None and args.run_kind == "cdl_teacher":
        eval_batch_cdl = idx_eval_model[: min(args.eval_batch_size, n_eval)].to(device)
        cdl_phys = cdl_provider.physical_order(model, eval_batch_cdl, 0)  # (N,) physical-frame
        cdl_model = physical_blocks_to_model_blocks(cdl_phys, clean_perm)
        for _ in args.eval_order_seeds:
            cdl_model_orders.append(cdl_model.unsqueeze(0).expand(n_eval, -1))

    # --- direct_policy: force a fresh model-frame strict65 score at eval ---
    direct_model_orders = []
    if direct_provider is not None and args.run_kind == "direct_policy":
        eval_batch_direct = idx_eval_model[: min(args.eval_batch_size, n_eval)].to(device)
        cached_sigma = direct_provider._sigma
        cached_last_refresh = direct_provider._last_refresh
        direct_provider._sigma = None
        direct_model = direct_provider.physical_order(
            model, eval_batch_direct, int(eval_step)
        ).cpu()
        direct_provider._sigma = cached_sigma
        direct_provider._last_refresh = cached_last_refresh
        for _ in args.eval_order_seeds:
            direct_model_orders.append(
                direct_model.unsqueeze(0).expand(n_eval, -1)
            )

    modes = {
        "val_ori_l2r_block": ([ori_model], ori_model, None),
        "val_ar_l2r": ([ori_model], ori_model, None),
        "val_model_order": ([model_ascending], model_ascending, None),
        "val_unstructured_order": (
            unstructured_model_orders,
            unstructured_model_orders[0][0],
            list(args.eval_order_seeds),
        ),
        "val_rw_order": (
            rw_model_orders,
            rw_model_orders[0][0],
            list(args.eval_order_seeds),
        ),
    }
    if beta_model_orders:
        modes["val_beta_order"] = (
            beta_model_orders,
            beta_model_orders[0][0],
            list(args.eval_order_seeds),
        )
    if cdl_model_orders:
        modes["val_cdl_order"] = (
            cdl_model_orders,
            cdl_model_orders[0][0],
            list(args.eval_order_seeds),
        )
    if direct_model_orders:
        modes["val_direct_order"] = (
            direct_model_orders,
            direct_model_orders[0][0],
            list(args.eval_order_seeds),
        )

    results = {}
    for name, (orders, desc_order, seeds) in modes.items():
        metrics = eval_model_orders(orders)
        results[name] = {**metrics, **describe(desc_order), "eval_seeds": seeds}

    random_loss = results["val_unstructured_order"]["loss_token_avg"]
    rw_loss = results["val_rw_order"]["loss_token_avg"]
    if args.run_kind in {"graph_rw", "graph_rw_bag"}:
        train_objective = (1.0 - alpha) * random_loss + alpha * rw_loss
    elif args.run_kind == "frozen_beta" and "val_beta_order" in results:
        beta_loss = results["val_beta_order"]["loss_token_avg"]
        train_objective = (1.0 - alpha) * random_loss + alpha * beta_loss
    elif args.run_kind == "cdl_teacher" and "val_cdl_order" in results:
        cdl_loss = results["val_cdl_order"]["loss_token_avg"]
        train_objective = (1.0 - alpha) * random_loss + alpha * cdl_loss
    elif args.run_kind == "direct_policy" and "val_direct_order" in results:
        direct_loss = results["val_direct_order"]["loss_token_avg"]
        train_objective = (1.0 - alpha) * random_loss + alpha * direct_loss
    elif args.run_kind == "l2r":
        train_objective = results["val_ori_l2r_block"]["loss_token_avg"]
    elif args.run_kind == "shuffled_l2r":
        train_objective = results["val_model_order"]["loss_token_avg"]
    else:
        train_objective = random_loss
    results["val_train_objective"] = {
        "loss_token_avg": float(train_objective),
        "loss_step_avg": float(train_objective),
        "num_predicted_tokens": results["val_unstructured_order"]["num_predicted_tokens"],
        "num_order_steps": results["val_unstructured_order"]["num_order_steps"],
        "eval_seeds": list(args.eval_order_seeds),
    }

    model.train()
    return results


def load_or_create_protocol(args, output_dir, ckpt=None):
    if ckpt is not None and "clean_protocol" in ckpt:
        protocol = ckpt["clean_protocol"]
        clean_perm = CleanPermutation(
            block_perm_phys_to_model=torch.tensor(protocol["block_perm_phys_to_model"], dtype=torch.long),
            inv_perm_model_to_phys=torch.tensor(protocol["inv_perm_model_to_phys"], dtype=torch.long),
        )
        return clean_perm, {
            "train_indices": np.asarray(protocol["train_indices"], dtype=np.int64),
            "val_indices": np.asarray(protocol["val_indices"], dtype=np.int64),
            "train_shuffle_order": np.asarray(protocol["train_shuffle_order"], dtype=np.int64),
            "eval_indices": np.asarray(protocol["eval_indices"], dtype=np.int64),
        }

    clean_perm = build_clean_block_permutation(N, args.permute_seed)
    return clean_perm, None


def save_protocol_files(output_dir, args, clean_perm, split):
    np.save(output_dir / "block_perm.npy", clean_perm.block_perm_phys_to_model.numpy())
    np.save(output_dir / "inv_perm.npy", clean_perm.inv_perm_model_to_phys.numpy())
    np.save(output_dir / "train_indices.npy", split["train_indices"])
    np.save(output_dir / "val_indices.npy", split["val_indices"])
    np.save(output_dir / "train_shuffle_order.npy", split["train_shuffle_order"])
    np.save(output_dir / "eval_indices.npy", split["eval_indices"])
    with (output_dir / "eval_order_seeds.json").open("w") as f:
        json.dump({"eval_order_seeds": list(args.eval_order_seeds)}, f, indent=2)

    ok = verify_clean_coordinate_round_trip(clean_perm, BLOCK_LEN)
    check = {
        "ok": bool(ok),
        "convention": {
            "block_perm": "block_perm[physical_block] = model_block",
            "inv_perm": "inv_perm[model_block] = physical_block",
            "physical_to_model": "block_perm[physical_order]",
            "model_to_physical": "inv_perm[model_order]",
        },
        "block_perm_first16": clean_perm.block_perm_phys_to_model[:16].tolist(),
        "inv_perm_first16": clean_perm.inv_perm_model_to_phys[:16].tolist(),
    }
    with (output_dir / "coordinate_check.json").open("w") as f:
        json.dump(check, f, indent=2)
    if not ok:
        raise RuntimeError("clean coordinate round-trip failed")


def checkpoint_payload(model, optimizer, args, clean_perm, split, global_step, train_losses, metrics, rw_policy, rw_params):
    cursor = train_cursor_for_next_step(
        split["train_shuffle_order"], global_step, args.batch_size, args.grad_accum
    )
    protocol = {
        "train_indices": split["train_indices"].tolist(),
        "val_indices": split["val_indices"].tolist(),
        "train_shuffle_order": split["train_shuffle_order"].tolist(),
        "eval_indices": split["eval_indices"].tolist(),
        "block_perm_phys_to_model": clean_perm.block_perm_phys_to_model.tolist(),
        "inv_perm_model_to_phys": clean_perm.inv_perm_model_to_phys.tolist(),
        "train_indices_sha256": sha256_int_array(split["train_indices"]),
        "val_indices_sha256": sha256_int_array(split["val_indices"]),
        "train_shuffle_order_sha256": sha256_int_array(split["train_shuffle_order"]),
        "eval_indices_sha256": sha256_int_array(split["eval_indices"]),
    }
    model_args = clean_model_args(args)
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": model_args,
        "iter_num": int(global_step),
        "global_step": int(global_step),
        "train_cursor": int(cursor),
        "args": vars(args),
        "train_losses": train_losses,
        "last_eval": metrics,
        "rw_policy": rw_policy,
        "rw_params": rw_params,
        "clean_protocol": protocol,
        "data_permutation": {
            "convention": "clean_phys_to_model",
            "block_perm": clean_perm.block_perm_phys_to_model.tolist(),
            "inverse_block_perm": clean_perm.inv_perm_model_to_phys.tolist(),
            "block_perm_phys_to_model": clean_perm.block_perm_phys_to_model.tolist(),
            "inv_perm_model_to_phys": clean_perm.inv_perm_model_to_phys.tolist(),
        },
    }


def direct_policy_audit_metadata(args):
    if getattr(args, "run_kind", None) != "direct_policy":
        return None
    return {
        "policy": args.direct_policy,
        "matrix_convention": "B[source,target]",
        "frame": "model-frame strict65",
        "layer_heads": "L0 all-head",
        "probe_aggregation": (
            f"mean over {int(args.batch_mean_probes)} probes"
        ),
        "head_fusion": "mean over heads",
        "diagonal_handling": (
            "exclude self; strict65 diagonal expected zero"
        ),
        "order_direction": "larger score earlier",
        "lambda_dep": float(args.direct_policy_lambda_dep),
        "refresh_every": int(args.direct_policy_refresh),
        "sequential_cdl_equivalent": False,
    }


def write_config(output_dir, args, split, clean_perm, rw_policy, rw_params):
    # Honest config: only emit Graph-RW fields if Graph-RW is actually active.
    # Writing rw_policy / rw_params for non-Graph-RW runs is misleading even
    # when another order provider uses the shared alpha schedule.
    graph_rw_active = args.run_kind in {"graph_rw", "graph_rw_bag"}
    alpha_active = args.run_kind in {
        "graph_rw", "graph_rw_bag", "frozen_beta", "cdl_teacher",
        "direct_policy",
    }
    payload = {
        "args": vars(args),
        "model_args": clean_model_args(args),
        "seq_len": SEQ_LEN,
        "num_blocks": N,
        "block_len": BLOCK_LEN,
        "coordinate_convention": {
            "block_perm": "block_perm[physical_block] = model_block",
            "inv_perm": "inv_perm[model_block] = physical_block",
        },
        "data_protocol": {
            "train_count": int(len(split["train_indices"])),
            "val_count": int(len(split["val_indices"])),
            "eval_count": int(len(split["eval_indices"])),
            "train_indices_sha256": sha256_int_array(split["train_indices"]),
            "val_indices_sha256": sha256_int_array(split["val_indices"]),
            "train_shuffle_order_sha256": sha256_int_array(split["train_shuffle_order"]),
            "eval_indices_sha256": sha256_int_array(split["eval_indices"]),
        },
        "graph_rw_active": graph_rw_active,
        "actual_alpha_schedule": (
            {
                "alpha_start": float(args.alpha_start),
                "alpha_target": float(args.alpha_target),
                "alpha_warmup_steps": int(args.alpha_warmup_steps),
            }
            if alpha_active
            else {"alpha_constant": 0.0, "reason": f"run_kind={args.run_kind} short-circuits alpha_for_step to 0"}
        ),
        "rw_policy": rw_policy if graph_rw_active else None,
        "rw_params": rw_params if graph_rw_active else None,
        "direct_policy_audit": direct_policy_audit_metadata(args),
        "block_perm_first16": clean_perm.block_perm_phys_to_model[:16].tolist(),
        "inv_perm_first16": clean_perm.inv_perm_model_to_phys[:16].tolist(),
    }
    with (output_dir / "config.json").open("w") as f:
        json.dump(payload, f, indent=2)


def parse_args(default_run_kind="baseline"):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-kind", choices=["baseline", "random_continuation", "graph_rw", "graph_rw_bag", "l2r", "shuffled_l2r", "frozen_beta", "cdl_teacher", "direct_policy"],
                   default=default_run_kind)
    p.add_argument("--resume-ckpt", default="")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--permute-seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--max-eval-seqs", type=int, default=200)
    p.add_argument("--data-source", choices=["chunks", "continuous"], default="chunks",
                   help="chunks = fixed wikitext arrow chunks (Graph-RW protocol, ~18ep reuse → "
                        "overfit); continuous = collaborator-style memmap random-window stream "
                        "(no reuse; L2R/baseline validation toward ~3.34). continuous excludes "
                        "Graph-RW refresh paths and requires --refresh-interval 0.")
    p.add_argument("--shuffle-granularity", type=int, default=64, choices=[32, 64, 128],
                   help="Shuffle granularity for baseline/random_continuation runs: number of "
                        "independently shuffled token groups. 32 = groups of 8 tokens, "
                        "64 = groups of 4 (standard), 128 = groups of 2 tokens. "
                        "Model always uses 64-block internal structure.")
    p.add_argument("--train-bin", default="/home/admin/ych/nanogpt-learned-order/data/wikitext103/train.bin",
                   help="continuous: flat uint16 token .bin for the training stream")
    p.add_argument("--val-bin", default="/home/admin/ych/nanogpt-learned-order/data/wikitext103/val.bin",
                   help="continuous: flat uint16 token .bin for the eval stream")
    p.add_argument("--stream-eval-windows", type=int, default=2000,
                   help="continuous: number of fixed seeded val windows used for eval "
                        "(replaces the 200 fixed chunks; larger → lower eval noise)")
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--eval-interval", type=int, default=1000)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=50000)
    p.add_argument("--save-steps", default=DEFAULT_SAVE_STEPS)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-4)
    p.add_argument("--lr-decay-steps", type=int, default=50000)
    p.add_argument("--warmup-iters", type=int, default=0)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.99)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--compile-model", dest="compile_model", action="store_true", default=True,
                   help="compile AO-GPT with torch.compile(mode='reduce-overhead') (default)")
    p.add_argument("--no-compile-model", dest="compile_model", action="store_false",
                   help="disable torch.compile for determinism/ablation runs")
    p.add_argument("--vocab-size", type=int, default=50304)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--n-embd", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--bias", action="store_true")
    p.add_argument("--a-path", default=A_PATH_DEFAULT)
    p.add_argument("--refresh-interval", type=int, default=0,
                   help="Extract A from current model and rebuild graph every N steps (0 = no refresh)")
    p.add_argument("--refresh-n-chunks", type=int, default=1000,
                   help="Number of chunks for A extraction during refresh")
    p.add_argument("--refresh-data-source", choices=["eval", "random_train"], default="eval",
                   help="Data source for A extraction: 'eval' = fixed eval set, 'random_train' = random train samples")
    p.add_argument("--refresh-ema-beta", type=float, default=0.9,
                   help="EMA beta for A_global update during refresh")
    p.add_argument("--eval-order-seeds", type=int, nargs="+", default=[42, 123, 456])
    p.add_argument("--alpha-start", type=float, default=0.0)
    p.add_argument("--alpha-target", type=float, default=0.9)
    p.add_argument("--alpha-warmup-steps", type=int, default=10000)
    p.add_argument("--alpha-warmup-start", type=int, default=0,
                   help="hold alpha at alpha_start for this many steps (relative to start_step) "
                        "BEFORE the linear ramp begins. From-0 alternating: random-order warmup "
                        "before the MLP-order curriculum (e.g. 5000).")
    p.add_argument("--alpha-ramp-from-resume", action="store_true",
                   help="ramp alpha over alpha_warmup_steps starting at the resume point (skip the "
                        "absolute-step resume-alpha compensation). Reproduces the v3 continuation schedule "
                        "(alpha 0->0.9 over the 10k continuation), required for a clean v3 comparison.")
    p.add_argument("--tau-start", type=float, default=0.1)
    p.add_argument("--tau-step", type=float, default=0.1)
    p.add_argument("--rw-top-k", type=int, default=4)
    p.add_argument("--rw-order-bag-k", type=int, default=1)
    p.add_argument("--rw-policy", type=str, default="progressive_rw",
                   choices=["progressive_rw", "progressive_rw_v2", "progressive_rw_v3", "mlp_cdl", "position_only"])
    p.add_argument("--pos-tau", type=float, default=0.1,
                   help="positional-prior temperature for rw-policy=position_only "
                        "(logits[v]=-v/pos_tau); calibrate to match source_start avg_step_entropy. "
                        "Phase-2 §6 entropy-matched position-only attribution control.")
    p.add_argument("--rw-lam", type=float, default=0.75,
                   help="λ trade-off for dependency penalty (v2/v3)")
    p.add_argument("--rw-rho", type=float, default=0.2,
                   help="ρ global readiness prior strength (v3 only)")
    p.add_argument("--epsilon-uniform", type=float, default=0.0,
                   help="Epsilon-uniform exploration mixing (0=disabled, 0.15=recommended)")
    # --- distilled Attn-Order MLP policy (rw-policy=mlp_cdl) ---
    p.add_argument("--mlp-path", type=str, default=None,
                   help="Phase-1 distilled OrderMLP state_dict (.pt) for rw-policy=mlp_cdl")
    p.add_argument("--mlp-orientation", type=str, default="original",
                   choices=["original", "reversed", "source_start"],
                   help="orientation control for the MLP order policy (Phase-2 §1b)")
    p.add_argument("--mlp-tau", type=float, default=0.5,
                   help="per-step sampling temperature for the (z-scored) MLP policy")
    p.add_argument("--mlp-src-rho", type=float, default=0.3,
                   help="readiness direction-prior strength for orientation=source_start")
    p.add_argument("--mlp-graph", type=str, default=None,
                   help="fixed A_global .npy used as the MLP substrate B (must match the "
                        "distillation graph; e.g. v3's ckpt20000 A_global_eval.npy)")
    p.add_argument("--mlp-alternating", action="store_true",
                   help="alternating mlp_cdl: at each --refresh-interval re-extract B from current "
                        "theta and re-distill/finetune beta on it, then sample with it. Requires "
                        "--refresh-interval>0; B/beta are born at the first refresh (no --mlp-graph).")
    p.add_argument("--mlp-refresh-mode", choices=["finetune", "scratch"], default="finetune",
                   help="alternating: warm-start beta from the previous beta (finetune) or re-init (scratch).")
    p.add_argument("--mlp-distill-n-orders", type=int, default=200,
                   help="alternating: teacher/random rollouts used to build the distillation dataset "
                        "for beta at each refresh")
    p.add_argument("--mlp-distill-tau-t", type=float, default=0.5,
                   help="alternating: C-D+L teacher softmax temperature for the distillation target")
    p.add_argument("--mlp-distill-tau-train", type=float, default=0.5,
                   help="alternating: student (beta) softmax temperature during KL distillation")
    p.add_argument("--mlp-distill-epochs", type=int, default=60,
                   help="alternating: KL-distillation epochs to (re)train beta at each refresh")
    p.add_argument("--mlp-distill-lr", type=float, default=1e-3,
                   help="alternating: Adam learning rate for the per-refresh beta distillation")
    p.add_argument("--mlp-distill-batch-states", type=int, default=256,
                   help="alternating: states per minibatch in the per-refresh beta distillation")
    # --- frozen-g_β order hook (run-kind=frozen_beta; Phase 2 §5) ---
    p.add_argument("--frozen-beta-ckpt", type=str, default=None,
                   help="run-kind=frozen_beta: path to the pretrained g_β checkpoint "
                        "(g_beta_best.pt). The in-loop order is g_β(B1(selected head)).")
    p.add_argument("--frozen-beta-head", type=int, nargs=2, default=[0, 0], metavar=("LAYER", "HEAD"),
                   help="(layer, head) the g_β was pretrained on; must match the dataset head (default L0H0).")
    p.add_argument("--frozen-beta-mode", choices=["argsort", "sample"], default="argsort",
                   help="g_β hook order mode: argsort (greedy) or sample (PL, temperature --frozen-beta-tau).")
    p.add_argument("--frozen-beta-tau", type=float, default=1.0,
                   help="temperature for --frozen-beta-mode=sample.")
    p.add_argument("--frozen-beta-refresh", type=int, default=1,
                   help="recompute the g_β order every N steps (K-step refresh; 1 = every step). "
                        "Bounds the probe-forward overhead to ~1/N.")
    p.add_argument("--frozen-beta-none-mode", choices=["b1", "predictor", "model", "content", "loss_aligned"], default="b1",
                   help="block-aggregation mode for the in-loop probe extraction; must match g_β training (B1=65-node).")
    p.add_argument("--frozen-beta-rev", action="store_true",
                   help="reverse the frozen g_beta emitted block order, matching audition rows marked rev.")
    # --- head-gated g_beta order hook (extends frozen_beta with multi-head gate) ---
    p.add_argument("--gbeta-input-mode", choices=["single_head", "layer_heads", "all_layers"], default="single_head",
                   help="g_beta input mode: single_head (existing), layer_heads (head-gated from one layer), "
                        "or all_layers (head-gated from ALL layers, H=L*H).")
    p.add_argument("--gbeta-layer", type=int, default=0,
                   help="which layer to extract heads from (for --gbeta-input-mode layer_heads).")
    p.add_argument("--gbeta-topk", type=int, default=2,
                   help="top-k for head gate (for --gbeta-input-mode layer_heads).")
    p.add_argument("--gbeta-frozen", type=lambda x: x.lower() != "false", default=True,
                   help="freeze g_beta weights during training (default True).")
    p.add_argument("--batch-mean-probes", type=int, default=1,
                   help="number of probe forward passes to average B over "
                        "(>1 activates model-frame FrozenGBetaModelFrameBlockProvider).")
    # --- direct model-frame strict65 policies ---
    p.add_argument(
        "--direct-policy",
        choices=["initial_cdl_one_shot", "source_mass", "readiness"],
        default="initial_cdl_one_shot",
        help="run-kind=direct_policy: non-learned L0 all-head strict65 score.",
    )
    p.add_argument(
        "--direct-policy-lambda-dep",
        type=float,
        default=1.0,
        help="dependency coefficient for --direct-policy readiness.",
    )
    p.add_argument(
        "--direct-policy-refresh",
        type=int,
        default=10,
        help="recompute the direct model-frame order every N optimizer steps.",
    )
    # --- Weights & Biases logging (formal runs) ---
    p.add_argument("--wandb-log", action="store_true",
                   help="log train loss and eval metrics to Weights & Biases.")
    p.add_argument("--wandb-project", default="order-lyu",
                   help="W&B project name when --wandb-log is enabled.")
    p.add_argument("--wandb-run-name", default=None,
                   help="Optional W&B run name; defaults to output directory name.")
    p.add_argument("--wandb-tags", nargs="*", default=[],
                   help="Optional W&B tags.")
    p.add_argument("--wandb-mode", default=None, choices=["online", "offline", "disabled"],
                   help="Optional W&B mode override.")
    # --- direct CDL teacher order hook (run-kind=cdl_teacher) ---
    p.add_argument("--cdl-teacher-head", type=int, nargs=2, default=[0, 7], metavar=("LAYER", "HEAD"),
                   help="run-kind=cdl_teacher: (layer, head) to extract attention from for C-D+L teacher.")
    p.add_argument("--cdl-teacher-tau", type=float, default=1.0,
                   help="C-D+L teacher softmax temperature.")
    p.add_argument("--cdl-teacher-refresh", type=int, default=10,
                   help="recompute the CDL order every N steps.")
    p.add_argument("--cdl-teacher-none-mode", type=str, default="b1",
                   choices=["b1", "predictor", "model", "content", "loss_aligned"],
                   help="none_mode for CDL teacher attention extraction (B1=65-node).")
    p.add_argument("--cdl-teacher-rev", action="store_true",
                   help="reverse the CDL teacher order (flip physical block order).")
    # --- per-head CDL signal tracking (diagnostic, any run-kind) ---
    p.add_argument("--track-all-heads", action="store_true",
                   help="at each tracking interval, scan every (layer, head) with the same lightweight probe batches.")
    p.add_argument("--track-head", type=int, nargs=2, default=None, metavar=("LAYER", "HEAD"),
                   help="at each eval, extract this head's B and log CDL tau / pairwise / uniqueness.")
    p.add_argument("--track-head-m", type=int, default=20,
                   help="number of probe batches for --track-head diagnostic (default 20).")
    p.add_argument("--track-head-interval", type=int, default=20,
                   help="run head-signal diagnostic every N steps (default 20).")
    p.add_argument("--track-head-none-mode", choices=["b1", "predictor", "model", "content", "loss_aligned"], default="b1",
                   help="attention-to-block aggregation: b1=predictor frame+physical remap (65-node), predictor=old predictor frame no remap, loss_aligned=AR target queries to source keys.")
    p.add_argument("--track-head-maps", action="store_true",
                   help="dump raw per-sample all-layer/all-head block attention maps at a fixed interval; no CDL/tau/top-head metrics.")
    p.add_argument("--track-head-map-interval", type=int, default=200,
                   help="dump raw head maps every N optimizer steps when --track-head-maps is enabled (default 200).")
    p.add_argument("--track-head-map-samples", type=int, default=4,
                   help="number of fixed eval probe samples to dump per raw head-map snapshot (default 4).")
    p.add_argument("--track-head-map-dtype", choices=["float32", "float16"], default="float32",
                   help="dtype used for raw head-map snapshots (default float32).")
    # --- attention trajectory logging (L0 all-head B maps at eval time) ---
    p.add_argument("--attn-trajectory", action="store_true",
                   help="log L0 all-head model-frame strict65 B maps at each eval step "
                        "(fixed samples, summary metrics, W&B logging).")
    p.add_argument("--attn-trajectory-samples", type=int, default=8,
                   help="number of fixed eval samples for attention trajectory extraction (default 8).")
    p.add_argument("--attn-trajectory-heatmap-interval", type=int, default=5000,
                   help="save heatmap PNGs every N steps (default 5000).")
    p.add_argument("--attn-composition-pairs", default="all",
                   choices=["all", "adjacent"],
                   help="layer pairs for candidate composition: all i<j, or adjacent only.")
    return p.parse_args()


def main(default_run_kind="baseline"):
    args = parse_args(default_run_kind=default_run_kind)
    if args.run_kind == "graph_rw_bag" and args.rw_order_bag_k < 1:
        raise ValueError("--rw-order-bag-k must be >= 1")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.txt"

    def log(message):
        print(message, flush=True)
        with log_path.open("a") as f:
            f.write(message + "\n")

    log(f"Run kind: {args.run_kind}")
    _graph_rw_active = args.run_kind in {"graph_rw", "graph_rw_bag"}
    _alpha_active = args.run_kind in {
        "graph_rw", "graph_rw_bag", "frozen_beta", "cdl_teacher",
        "direct_policy",
    }
    log(f"graph_rw_active: {_graph_rw_active}  alpha_active: {_alpha_active}")
    if _alpha_active:
        log(
            f"alpha_schedule: start={args.alpha_start} target={args.alpha_target} "
            f"warmup_steps={args.alpha_warmup_steps}"
        )
    if _graph_rw_active:
        log(
            f"rw_policy={args.rw_policy} top_k={args.rw_top_k} "
            f"epsilon={getattr(args, 'rw_epsilon_uniform', None)} "
            f"tau_start={args.tau_start} lam={getattr(args, 'rw_lam', None)} "
            f"rho={getattr(args, 'rw_rho', None)}"
        )
    if not _alpha_active:
        log(
            f"alpha_schedule: constant 0.0 (run_kind={args.run_kind} short-circuits "
            f"alpha_for_step). Any rw_* CLI flags are ignored at training time."
        )
    if args.run_kind in {"baseline", "random_continuation"}:
        log(f"shuffle_granularity: {args.shuffle_granularity} "
            f"(groups of {SEQ_LEN // args.shuffle_granularity} tokens)")
    log(f"Output dir: {output_dir}")
    log(f"Device: {device}")

    ckpt = None
    if args.resume_ckpt:
        ckpt = torch.load(os.path.expanduser(args.resume_ckpt), map_location=device, weights_only=False)
        log(f"Loaded resume checkpoint: {args.resume_ckpt}")

    clean_perm, loaded_split = load_or_create_protocol(args, output_dir, ckpt)
    wandb_run = maybe_init_wandb(args, output_dir, clean_perm=clean_perm, log_fn=log)

    continuous = (args.data_source == "continuous")
    if continuous:
        # frozen_beta is allowed: its g_β hook extracts B per-batch from the model (no fixed
        # chunk pool needed) and reads no idx_train. The Graph-RW refresh/random_train paths
        # DO need the fixed pool, so graph_rw* stay excluded.
        if args.run_kind not in {
            "l2r", "shuffled_l2r", "baseline", "random_continuation",
            "frozen_beta", "cdl_teacher", "direct_policy",
        }:
            raise SystemExit(
                "--data-source continuous only supports --run-kind in "
                "{l2r,shuffled_l2r,baseline,random_continuation,frozen_beta,"
                "cdl_teacher,direct_policy} (the Graph-RW refresh/random_train "
                "paths need the fixed chunk pool); got "
                f"{args.run_kind!r}.")
        if args.refresh_interval > 0:
            raise SystemExit("--data-source continuous does not support --refresh-interval > 0 "
                             "(no fixed eval/train chunk pool to extract A from).")

    # Token-level gather: physical-space token tensor [:, g_gather] == model-space tensor.
    g_gather = build_phys_to_model_token_gather(clean_perm, BLOCK_LEN)

    if continuous:
        log(f"[data-source=continuous] memmap streams train={args.train_bin} val={args.val_bin}")
        stream_train = load_token_stream(args.train_bin)
        stream_val = load_token_stream(args.val_bin)
        # Fixed, seeded eval windows from the val stream -> model coordinates.
        eval_phys = sample_stream_batch(
            stream_val, args.stream_eval_windows, SEQ_LEN,
            seed=args.permute_seed, step=-1, micro=0,
        )
        idx_eval_model = eval_phys[:, g_gather].contiguous()
        idx_train = None  # streamed per-step; no fixed train pool
        # Harmless dummy split so checkpoint/protocol-file plumbing stays intact.
        # train_* are 1-length (not empty) because train_cursor_for_next_step rejects empties.
        split = {
            "train_indices": np.zeros(1, dtype=np.int64),
            "val_indices": np.zeros(0, dtype=np.int64),
            "train_shuffle_order": np.zeros(1, dtype=np.int64),
            "eval_indices": np.arange(idx_eval_model.size(0), dtype=np.int64),
        }
        save_protocol_files(output_dir, args, clean_perm, split)
        log(f"[continuous] train stream tokens={len(stream_train)} "
            f"eval windows={idx_eval_model.size(0)} (block={SEQ_LEN})")
    else:
        stream_train = None
        log("Loading train-arrow chunks...")
        idx_phys = load_train_chunks(n_chunks=None)
        if loaded_split is None:
            fixed = build_fixed_split_and_shuffle(
                total_chunks=idx_phys.size(0),
                seed=args.seed,
                val_fraction=args.val_fraction,
                max_eval_seqs=args.max_eval_seqs,
            )
            split = {
                "train_indices": fixed.train_indices,
                "val_indices": fixed.val_indices,
                "train_shuffle_order": fixed.train_shuffle_order,
                "eval_indices": fixed.eval_indices,
            }
        else:
            split = loaded_split
        save_protocol_files(output_dir, args, clean_perm, split)

        log(f"Train chunks: {len(split['train_indices'])}, val chunks: {len(split['val_indices'])}, eval chunks: {len(split['eval_indices'])}")
        log("Converting chunks to model coordinates...")
        idx_model = phys_to_model_idx_clean(idx_phys, clean_perm)
        idx_eval_model = idx_model[split["eval_indices"]]
        idx_train = idx_model[split["train_indices"]]

    if ckpt is None:
        model_args = clean_model_args(args)
        model = build_model(model_args, device, compile_model=bool(args.compile_model))
        start_step = 0
        train_losses = []
    else:
        model_args = ckpt.get("model_args") or clean_model_args(args)
        model = build_model(model_args, device, compile_model=False)
        state_dict = ckpt.get("model") or ckpt.get("model_state_dict")
        if state_dict is None:
            raise KeyError("resume checkpoint lacks model/model_state_dict")
        model.load_state_dict(clean_state_dict(state_dict))
        if args.compile_model:
            model = torch.compile(model, mode="reduce-overhead")
        start_step = int(ckpt.get("global_step", ckpt.get("iter_num", 0)))
        train_losses = list(ckpt.get("train_losses", []))

    # Fix alpha on resume: don't restart warmup from 0.
    # alpha_for_step measures local_step = global_step - start_step,
    # so a resume makes local_step jump back to 0.  Compensate by
    # advancing alpha_start to whatever alpha should be at start_step.
    # --alpha-ramp-from-resume skips this so the warmup ramps from the resume point
    # (the v3 continuation schedule: alpha 0->0.9 over the 10k continuation).
    if start_step > 0 and not args.alpha_ramp_from_resume:
        current_alpha = alpha_for_step(start_step, 0, args)
        args.alpha_start = current_alpha

    # Initialize Graph-RW policy from current model's attention
    rw_policy = args.rw_policy
    rw_params = {
        "tau_start": args.tau_start,
        "tau_step": args.tau_step,
        "alpha_dep": 0.5,
        "alpha_pr": 0.85,
    }
    if rw_policy in ("progressive_rw_v2", "progressive_rw_v3"):
        rw_params["lam"] = args.rw_lam
    if rw_policy == "progressive_rw_v3":
        rw_params["rho"] = args.rw_rho
    if rw_policy == "progressive_rw":
        rw_params.update({
            "beta_sup": 1.0, "beta_fut": 0.5,
            "beta_src": 0.2, "beta_loc": 0.5,
        })
    if args.rw_top_k > 0:
        rw_params["top_k"] = int(args.rw_top_k)
    if args.epsilon_uniform > 0.0:
        rw_params["epsilon_uniform"] = float(args.epsilon_uniform)

    rw_mlp = None
    if rw_policy == "mlp_cdl":
        rw_params.update({
            "orientation": args.mlp_orientation,
            "mlp_tau": float(args.mlp_tau),
            "src_rho": float(args.mlp_src_rho) if args.mlp_orientation == "source_start" else 0.0,
            "mlp_path": str(args.mlp_path) if args.mlp_path else None,
        })
        if args.mlp_alternating:
            if args.refresh_interval <= 0:
                raise SystemExit("--mlp-alternating requires --refresh-interval > 0")
            if args.mlp_graph:
                raise SystemExit("--mlp-alternating must NOT take --mlp-graph (B is born from refresh, "
                                 "no future-B leakage)")
            if args.run_kind not in {"graph_rw", "graph_rw_bag"}:
                raise SystemExit("--mlp-alternating requires --run-kind graph_rw (or graph_rw_bag); "
                                 f"got {args.run_kind!r} — the refresh loop and alpha curriculum are "
                                 "inactive otherwise, so B/beta would never be born (silent no-op)")
            rw_mlp = load_order_mlp(args.mlp_path, device) if args.mlp_path else None  # optional seed
            log(f"[mlp_cdl ALTERNATING] mode={args.mlp_refresh_mode}; beta "
                f"{'seeded from --mlp-path' if args.mlp_path else 'born at first refresh'}; "
                f"orientation={args.mlp_orientation} tau={args.mlp_tau} src_rho={rw_params['src_rho']}")
        else:
            if not args.mlp_path:
                raise SystemExit("--rw-policy mlp_cdl (fixed) requires --mlp-path")
            rw_mlp = load_order_mlp(args.mlp_path, device)
            log(f"Loaded distilled MLP policy: {args.mlp_path} orientation={args.mlp_orientation} "
                f"tau={args.mlp_tau} top_k={args.rw_top_k} src_rho={rw_params['src_rho']}")
    if rw_policy == "position_only":
        rw_params["pos_tau"] = float(args.pos_tau)
        if args.refresh_interval > 0:
            raise SystemExit("position_only uses no graph B; do not set --refresh-interval > 0")
        log(f"[position_only] pos_tau={args.pos_tau} top_k={args.rw_top_k}; no graph B / MLP "
            f"(positional prior only) — entropy-matched attribution control for source_start")

    # idx_eval_model / idx_train were set in the data-source branch above.
    if rw_policy == "mlp_cdl":
        if args.mlp_alternating:
            if start_step > 0:
                # Resume: re-extract B from the current (resumed) model so the MLP has a real
                # substrate immediately, instead of a zero placeholder that wastes steps.
                if args.refresh_data_source == "random_train":
                    extract_n = min(args.refresh_n_chunks, len(idx_train))
                    rng = np.random.RandomState(start_step + args.seed)
                    extract_chunks = idx_train[rng.choice(len(idx_train), size=extract_n, replace=False)]
                    log(f"[mlp_cdl ALTERNATING resume] Extracting B from current model on "
                        f"{extract_n} random train chunks...")
                else:
                    extract_n = min(args.refresh_n_chunks, len(idx_eval_model))
                    extract_chunks = idx_eval_model
                    log(f"[mlp_cdl ALTERNATING resume] Extracting B from current model on "
                        f"{extract_n} eval chunks...")
                B, A_global, elapsed, _ = refresh_rw_graph(
                    model, extract_chunks, clean_perm, device, None,
                    n_chunks=extract_n, ema_beta=0.0,
                )
                log(f"[mlp_cdl ALTERNATING resume] B extracted in {elapsed:.1f}s; "
                    f"MLP-guided sampling active immediately.")
            else:
                A_global = np.zeros((N, N), dtype=np.float32)
                B = A_global
                log("[mlp_cdl ALTERNATING] placeholder zero-B; first refresh extracts B from theta + "
                    "distills beta. Use --refresh-ema-beta 0.0 so B is the fresh extraction.")
        else:
            if args.refresh_interval > 0:
                raise SystemExit("fixed mlp_cdl requires a fixed B; do not set --refresh-interval > 0")
            if not args.mlp_graph:
                raise SystemExit("--rw-policy mlp_cdl requires --mlp-graph (the fixed A_global substrate)")
            A_global = np.load(args.mlp_graph).astype(np.float32)
            np.fill_diagonal(A_global, 0.0)
            B = build_directed_graph(A_global)
            log(f"[mlp_cdl] loaded FIXED B from {args.mlp_graph} (shape {tuple(A_global.shape)}); no refresh.")
    elif rw_policy == "position_only":
        A_global = np.zeros((N, N), dtype=np.float32)
        B = A_global  # placeholder; the position sampler ignores B entirely
        log("[position_only] no graph substrate; zero-A placeholder B (sampler is positional only).")
    elif args.run_kind in {"graph_rw", "graph_rw_bag"}:
        if args.refresh_data_source == "random_train":
            extract_n = min(args.refresh_n_chunks, len(idx_train))
            rng = np.random.RandomState(start_step + args.seed)
            extract_chunks = idx_train[rng.choice(len(idx_train), size=extract_n, replace=False)]
            log(f"Extracting A matrices from current model on {extract_n} random train chunks...")
        else:
            extract_n = min(args.refresh_n_chunks, len(idx_eval_model))
            extract_chunks = idx_eval_model
            log(f"Extracting A matrices from current model on {extract_n} eval chunks...")
        B, A_global, extract_elapsed, _ = refresh_rw_graph(
            model, extract_chunks, clean_perm, device, None,
            n_chunks=extract_n, ema_beta=0.0,
        )
        log(f"Extracted A_global in {extract_elapsed:.1f}s")
    else:
        A_all = np.load(args.a_path)
        A_global = A_all.mean(axis=0).astype(np.float32)
        np.fill_diagonal(A_global, 0.0)
        B = build_directed_graph(A_global)

    np.save(output_dir / "A_global_eval.npy", A_global)

    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.lr,
        betas=(args.beta1, args.beta2),
        device_type="cuda" if device.type == "cuda" else "cpu",
    )
    if ckpt is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        log("Resumed optimizer state.")

    write_config(output_dir, args, split, clean_perm, rw_policy, rw_params)
    save_steps = parse_step_list(args.save_steps)
    eval_curve_path = output_dir / "eval_curve.tsv"
    if not eval_curve_path.exists() or start_step == 0:
        with eval_curve_path.open("w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow([
                "step",
                "alpha",
                "train_loss",
                "val_train_objective",
                "val_ori_l2r_block",
                "val_ar_l2r",
                "val_model_order",
                "val_unstructured_order",
                "val_rw_order",
                "val_beta_order",
                "val_direct_order",
                "val_cdl_order",
                "lr",
            ])

    model.train()
    t0 = time.time()
    last_log_time = t0
    last_metrics = {}
    next_refresh_step = start_step + args.refresh_interval if args.refresh_interval > 0 else None

    attn_traj_logger = None  # initialised below if --attn-trajectory

    def run_eval_and_save(global_step, avg_loss, lr, alpha):
        nonlocal last_metrics
        log(f"[Eval @ {global_step}] alpha={alpha:.4f}")
        metrics = evaluate_orders(
            model, idx_eval_model, clean_perm, B, rw_policy, rw_params, args, alpha,
            rw_mlp=rw_mlp, beta_provider=beta_provider,
            cdl_provider=cdl_provider, direct_provider=direct_provider,
            eval_step=global_step,
        )
        last_metrics = metrics
        with eval_curve_path.open("a", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            row = [
                global_step,
                f"{alpha:.6f}",
                f"{avg_loss:.6f}",
                f"{metrics['val_train_objective']['loss_token_avg']:.6f}",
                f"{metrics['val_ori_l2r_block']['loss_token_avg']:.6f}",
                f"{metrics['val_ar_l2r']['loss_token_avg']:.6f}",
                f"{metrics['val_model_order']['loss_token_avg']:.6f}",
                f"{metrics['val_unstructured_order']['loss_token_avg']:.6f}",
                f"{metrics['val_rw_order']['loss_token_avg']:.6f}",
                f"{metrics.get('val_beta_order', {}).get('loss_token_avg', float('nan')):.6f}",
                f"{metrics.get('val_direct_order', {}).get('loss_token_avg', float('nan')):.6f}",
                f"{metrics.get('val_cdl_order', {}).get('loss_token_avg', float('nan')):.6f}",
                f"{lr:.8e}",
            ]
            writer.writerow(row)
        beta_str = f"beta={metrics['val_beta_order']['loss_token_avg']:.4f} | " if "val_beta_order" in metrics else ""
        direct_str = f"direct={metrics['val_direct_order']['loss_token_avg']:.4f} | " if "val_direct_order" in metrics else ""
        cdl_str = f"cdl={metrics['val_cdl_order']['loss_token_avg']:.4f} | " if "val_cdl_order" in metrics else ""
        log(
            f"[Eval @ {global_step}] train_obj={metrics['val_train_objective']['loss_token_avg']:.4f} | "
            f"ori_l2r={metrics['val_ori_l2r_block']['loss_token_avg']:.4f} | "
            f"model_order={metrics['val_model_order']['loss_token_avg']:.4f} | "
            f"unstructured={metrics['val_unstructured_order']['loss_token_avg']:.4f} | "
            f"{beta_str}"
            f"{direct_str}"
            f"{cdl_str}"
            f"rw={metrics['val_rw_order']['loss_token_avg']:.4f}"
        )
        if wandb_run is not None:
            wandb_run.log(
                wandb_eval_payload(global_step, metrics, avg_loss, alpha, lr),
                step=int(global_step),
            )

        # ── attention trajectory snapshot (L0 all-head B maps) ──
        if attn_traj_logger is not None:
            try:
                attn_traj_logger.log_snapshot(
                    model, idx_eval_model, global_step,
                    clean_perm, device,
                )
            except Exception as exc:
                log(f"[attn_trajectory] WARNING: snapshot @ step {global_step} failed: {exc}")

        # ── per-head CDL signal tracking (diagnostic) ──
        if args.track_head is not None:
            _track_head_signal(model, global_step)
        if args.track_all_heads:
            _track_all_heads_signal(model, global_step)

        return metrics

    # ── head signal tracker ─────────────────────────────────────────────
    head_signal_path = None
    if args.track_head is not None:
        head_signal_path = output_dir / "head_signal.tsv"
        head_signal_path.write_text(
            "step\tnone_mode\ttau_vs_l2r\tmean_pairwise_tau\tunique_sigma\trow_conc_mean\telapsed_s\n"
        )
    head_signal_all_path = None
    if args.track_all_heads:
        head_signal_all_path = output_dir / "head_signal_all.tsv"
        head_signal_all_path.write_text(
            "step\tnone_mode\tlayer\thead\ttau_vs_l2r\tmean_pairwise_tau\tunique_sigma\trow_conc_mean\telapsed_s\n"
        )
    head_maps_raw_dir = None
    if args.track_head_maps:
        if int(args.track_head_map_interval) < 1:
            raise ValueError("--track-head-map-interval must be >= 1")
        if int(args.track_head_map_samples) < 1:
            raise ValueError("--track-head-map-samples must be >= 1")
        head_maps_raw_dir = output_dir / "head_maps_raw"
        head_maps_raw_dir.mkdir(parents=True, exist_ok=True)
        (head_maps_raw_dir / "config.json").write_text(json.dumps({
            "format": "npz",
            "array": "head_maps",
            "shape": ["sample", "layer", "head", "block_query", "block_key"],
            "none_mode": args.track_head_none_mode,
            "interval": int(args.track_head_map_interval),
            "samples": int(args.track_head_map_samples),
            "dtype": args.track_head_map_dtype,
            "probe_policy": "fixed eval windows and fixed probe orders across snapshots",
            "metrics": "none",
        }, indent=2) + "\n")

    @torch.no_grad()
    def _track_head_signal(model, global_step):
        from neural_readout.teacher_labels import generate_teacher_label
        from per_head_order_scan import (
            _attn_to_A_block_b1_vec,
            _attn_to_A_block_loss_aligned_content_vec,
            _attn_to_A_block_predictor_vec,
            _attn_to_A_block_model_vec,
            _attn_to_A_block_content_vec,
            _batch_mean_B,
        )
        from batch_readout.diversity_batch import teacher_diversity_stats
        from scipy.stats import kendalltau
        import time as _time

        t0 = _time.time()
        device = next(model.parameters()).device
        inv_perm = clean_perm.inv_perm_model_to_phys.cpu().numpy()
        M_track_req = int(args.track_head_m)
        l, h = int(args.track_head[0]), int(args.track_head[1])
        none_mode = str(args.track_head_none_mode)
        track_is_model = (none_mode in ("model", "content"))
        PROBE_BATCH = 32  # grouping factor for batch-mean B
        FWD_BATCH = 8     # small forward batch to avoid OOM (attention tensors are large)
        model.eval()

        # Use eval windows for probe
        n_avail = len(idx_eval_model)
        n_probe = min(M_track_req * PROBE_BATCH, n_avail)
        M_track = max(1, n_probe // PROBE_BATCH)
        n_probe = M_track * PROBE_BATCH
        rng = np.random.default_rng(int(args.seed) * 10000 + global_step)
        probe_idx = rng.choice(n_avail, size=n_probe, replace=False)
        from batch_readout.hook_order_provider import random_probe_token_orders

        # Forward in small batches to avoid OOM
        A_chunks = []
        for start in range(0, n_probe, FWD_BATCH):
            stop = min(start + FWD_BATCH, n_probe)
            probe_chunks = idx_eval_model[probe_idx[start:stop]].to(device)
            probe_orders = random_probe_token_orders(
                probe_chunks.shape[0], args.seed, global_step + start, device)
            _, _, attn_list = model.forward_fn(probe_chunks, probe_orders, return_attentions=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            attn_stack = torch.stack(attn_list).cpu().numpy()  # (L, B, H, T+1, T+1)
            for bi in range(attn_stack.shape[1]):
                if none_mode == "predictor":
                    A_chunks.append(_attn_to_A_block_predictor_vec(
                        attn_stack[l, bi, h], probe_orders[bi].cpu().numpy(), inv_perm))
                elif none_mode == "b1":
                    A_chunks.append(_attn_to_A_block_b1_vec(
                        attn_stack[l, bi, h], probe_orders[bi].cpu().numpy(), inv_perm))
                elif none_mode == "model":
                    A_chunks.append(_attn_to_A_block_model_vec(
                        attn_stack[l, bi, h], probe_orders[bi].cpu().numpy(), inv_perm))
                elif none_mode == "content":
                    A_chunks.append(_attn_to_A_block_content_vec(
                        attn_stack[l, bi, h], probe_orders[bi].cpu().numpy(), inv_perm))
                elif none_mode == "loss_aligned":
                    A_chunks.append(_attn_to_A_block_loss_aligned_content_vec(
                        attn_stack[l, bi, h], probe_orders[bi].cpu().numpy(), inv_perm))
                else:
                    raise ValueError(f"unknown --track-head-none-mode={none_mode!r}")
        A_track = np.stack(A_chunks, axis=0)  # (n_probe, N, N)
        # Batch-mean B
        B_track = _batch_mean_B(A_track, M_track, PROBE_BATCH)  # (M, N, N)

        # CDL per graph
        sigmas = np.empty((M_track, N), dtype=np.int64)
        for m in range(M_track):
            sigmas[m], _, _ = generate_teacher_label(B_track[m], alpha_dep=0.5)

        l2r = np.arange(N)
        taus = []
        for s in sigmas:
            if track_is_model:
                s_phys = inv_perm[s]                         # model→physical remap for tau
            else:
                s_phys = s
            t, _ = kendalltau(s_phys, l2r)
            if not np.isnan(t):
                taus.append(t)
        tau = float(np.mean(taus)) if taus else float("nan")
        div = teacher_diversity_stats(sigmas)
        unique = len(set(tuple(s.tolist()) for s in sigmas))
        # Proper row-negentropy: normalize each row → entropy → log(N) - H
        row_negs = []
        for m in range(M_track):
            for i in range(N):
                row = np.abs(B_track[m, i]) + 1e-12
                p = row / row.sum()
                H = -np.sum(p * np.log(p))
                row_negs.append(np.log(N) - H)
        row_conc = float(np.mean(row_negs))
        elapsed = _time.time() - t0

        with head_signal_path.open("a", newline="") as f:
            f.write(f"{global_step}\t{none_mode}\t{tau:.6f}\t{div['mean_pairwise_tau']:.6f}\t{unique}\t{row_conc:.4f}\t{elapsed:.1f}\n")
        log(f"[HeadSignal L{l}H{h} {none_mode}] τ={tau:+.4f} pw={div['mean_pairwise_tau']:.4f} unique={unique}/{M_track} row_conc={row_conc:.2f} ({elapsed:.1f}s)")

        model.train()

    @torch.no_grad()
    def _track_all_heads_signal(model, global_step):
        from neural_readout.teacher_labels import generate_teacher_label
        from per_head_order_scan import (
            _attn_to_A_block_b1_vec,
            _attn_to_A_block_loss_aligned_content_vec,
            _attn_to_A_block_predictor_vec,
            _attn_to_A_block_model_vec,
            _attn_to_A_block_content_vec,
        )
        from batch_readout.diversity_batch import teacher_diversity_stats
        from batch_readout.eval_metrics import kendall_tau_batch
        import time as _time

        t0 = _time.time()
        device = next(model.parameters()).device
        inv_perm = clean_perm.inv_perm_model_to_phys.cpu().numpy()
        M_track_req = int(args.track_head_m)
        none_mode = str(args.track_head_none_mode)
        track_is_model = (none_mode in ("model", "content"))
        PROBE_BATCH = 32
        FWD_BATCH = 8
        model.eval()

        n_avail = len(idx_eval_model)
        n_probe = min(M_track_req * PROBE_BATCH, n_avail)
        M_track = max(1, n_probe // PROBE_BATCH)
        n_probe = M_track * PROBE_BATCH
        rng = np.random.default_rng(int(args.seed) * 10000 + global_step)
        probe_idx = rng.choice(n_avail, size=n_probe, replace=False)
        from batch_readout.hook_order_provider import random_probe_token_orders

        A_lh = None
        L = H = None
        di = np.arange(N)
        for start in range(0, n_probe, FWD_BATCH):
            stop = min(start + FWD_BATCH, n_probe)
            probe_chunks = idx_eval_model[probe_idx[start:stop]].to(device)
            probe_orders = random_probe_token_orders(
                probe_chunks.shape[0], args.seed, global_step + start, device)
            _, _, attn_list = model.forward_fn(probe_chunks, probe_orders, return_attentions=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            attn_stack = torch.stack(attn_list).cpu().numpy()  # (L, B, H, T+1, T+1)
            if A_lh is None:
                L, H = attn_stack.shape[0], attn_stack.shape[2]
                A_lh = np.zeros((L, H, M_track, N, N), dtype=np.float64)
            for bi in range(attn_stack.shape[1]):
                global_i = start + bi
                m = global_i // PROBE_BATCH
                if m >= M_track:
                    continue
                sample = np.transpose(attn_stack[:, bi], (0, 1, 2, 3))  # (L,H,T+1,T+1)
                if none_mode == "predictor":
                    A_blocks = _attn_to_A_block_predictor_vec(
                        sample, probe_orders[bi].cpu().numpy(), inv_perm)
                elif none_mode == "b1":
                    A_blocks = _attn_to_A_block_b1_vec(
                        sample, probe_orders[bi].cpu().numpy(), inv_perm)
                elif none_mode == "model":
                    A_blocks = _attn_to_A_block_model_vec(
                        sample, probe_orders[bi].cpu().numpy(), inv_perm)
                elif none_mode == "content":
                    A_blocks = _attn_to_A_block_content_vec(
                        sample, probe_orders[bi].cpu().numpy(), inv_perm)
                elif none_mode == "loss_aligned":
                    A_blocks = _attn_to_A_block_loss_aligned_content_vec(
                        sample, probe_orders[bi].cpu().numpy(), inv_perm)
                else:
                    raise ValueError(f"unknown --track-head-none-mode={none_mode!r}")
                B_blocks = np.swapaxes(A_blocks, -1, -2)
                B_blocks[..., di, di] = 0.0
                A_lh[:, :, m] += B_blocks

        Bb_lh = (A_lh / PROBE_BATCH).astype(np.float32)
        Bb_lh[..., di, di] = 0.0
        l2r = np.tile(np.arange(N), (M_track, 1))
        rows = []
        for ell in range(L):
            for h in range(H):
                sigmas = np.stack([
                    generate_teacher_label(Bb_lh[ell, h, m], alpha_dep=0.5)[0]
                    for m in range(M_track)
                ])
                div = teacher_diversity_stats(sigmas)
                unique = len(set(tuple(s.tolist()) for s in sigmas))
                row = np.abs(Bb_lh[ell, h]) + 1e-12
                p = row / row.sum(axis=-1, keepdims=True)
                entropy = -np.sum(p * np.log(p), axis=-1)
                row_conc = float(np.mean(np.log(N) - entropy))
                tau = float(kendall_tau_batch(
                    inv_perm[sigmas] if track_is_model else sigmas, l2r))
                rows.append((ell, h, tau, div["mean_pairwise_tau"], unique, row_conc))

        elapsed = _time.time() - t0
        with head_signal_all_path.open("a", newline="") as f:
            for ell, h, tau, pw, unique, row_conc in rows:
                f.write(
                    f"{global_step}\t{none_mode}\t{ell}\t{h}\t{tau:.6f}\t"
                    f"{pw:.6f}\t{unique}\t{row_conc:.4f}\t{elapsed:.1f}\n"
                )
        top = max(rows, key=lambda r: abs(r[2]))
        log(
            f"[HeadSignalAll {none_mode}] top=L{top[0]}H{top[1]} "
            f"τ={top[2]:+.4f} pw={top[3]:.4f} unique={top[4]}/{M_track} "
            f"row_conc={top[5]:.2f} ({elapsed:.1f}s)"
        )

        model.train()

    @torch.no_grad()
    def _track_head_maps_raw(model, global_step):
        from per_head_order_scan import (
            _attn_to_A_block_b1_vec,
            _attn_to_A_block_loss_aligned_content_vec,
            _attn_to_A_block_predictor_vec,
            _attn_to_A_block_model_vec,
            _attn_to_A_block_content_vec,
        )
        from batch_readout.hook_order_provider import random_probe_token_orders
        import time as _time

        assert head_maps_raw_dir is not None
        t0 = _time.time()
        device = next(model.parameters()).device
        inv_perm = clean_perm.inv_perm_model_to_phys.cpu().numpy()
        none_mode = str(args.track_head_none_mode)
        n_avail = len(idx_eval_model)
        n_samples = min(int(args.track_head_map_samples), n_avail)
        if n_samples <= 0:
            log(f"[HeadMapsRaw {none_mode}] skipped: no eval windows available")
            return

        # Keep probes fixed across snapshots so step-to-step differences are model changes.
        rng = np.random.default_rng(int(args.seed) * 10000 + 4242)
        probe_idx = rng.choice(n_avail, size=n_samples, replace=False)
        probe_chunks = idx_eval_model[probe_idx].to(device)
        probe_orders = random_probe_token_orders(n_samples, args.seed, 0, device)

        was_training = model.training
        model.eval()
        _, _, attn_list = model.forward_fn(probe_chunks, probe_orders, return_attentions=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        attn_stack = torch.stack(attn_list).cpu().numpy()  # (L, B, H, T+1, T+1)
        probe_orders_np = probe_orders.cpu().numpy().astype(np.int64)

        maps = []
        for bi in range(attn_stack.shape[1]):
            sample = np.transpose(attn_stack[:, bi], (0, 1, 2, 3))  # (L,H,T+1,T+1)
            if none_mode == "predictor":
                head_maps = _attn_to_A_block_predictor_vec(sample, probe_orders_np[bi], inv_perm)
            elif none_mode == "b1":
                head_maps = _attn_to_A_block_b1_vec(sample, probe_orders_np[bi], inv_perm)
            elif none_mode == "model":
                head_maps = _attn_to_A_block_model_vec(sample, probe_orders_np[bi], inv_perm)
            elif none_mode == "content":
                head_maps = _attn_to_A_block_content_vec(sample, probe_orders_np[bi], inv_perm)
            elif none_mode == "loss_aligned":
                head_maps = _attn_to_A_block_loss_aligned_content_vec(sample, probe_orders_np[bi], inv_perm)
            else:
                raise ValueError(f"unknown --track-head-none-mode={none_mode!r}")
            maps.append(head_maps)

        dtype = np.float16 if args.track_head_map_dtype == "float16" else np.float32
        head_maps = np.stack(maps, axis=0).astype(dtype, copy=False)
        meta = {
            "step": int(global_step),
            "none_mode": none_mode,
            "shape": list(head_maps.shape),
            "axis": ["sample", "layer", "head", "block_query", "block_key"],
            "dtype": str(head_maps.dtype),
            "probe_idx_kind": "indices_into_idx_eval_model",
            "probe_seed": int(args.seed) * 10000 + 4242,
            "probe_order_step": 0,
            "metrics": "none",
        }
        path = head_maps_raw_dir / f"head_maps_step{int(global_step):06d}.npz"
        np.savez_compressed(
            path,
            head_maps=head_maps,
            probe_idx=probe_idx.astype(np.int64),
            probe_orders=probe_orders_np,
            meta_json=np.array(json.dumps(meta)),
        )
        elapsed = _time.time() - t0
        log(f"[HeadMapsRaw {none_mode}] saved {path} shape={tuple(head_maps.shape)} ({elapsed:.1f}s)")
        if was_training:
            model.train()

    def save_ckpt(global_step, metrics):
        path = output_dir / f"ckpt_step{global_step}.pt"
        payload = checkpoint_payload(
            model, optimizer, args, clean_perm, split, global_step,
            train_losses, metrics, rw_policy, rw_params
        )
        torch.save(payload, path)
        log(f"Saved checkpoint: {path}")

    beta_provider = None
    cdl_provider = None
    direct_provider = None

    # ── attention trajectory logger (L0 all-head B maps at eval time) ──
    if args.attn_trajectory:
        from attention_trajectory import AttentionTrajectoryLogger, phys_perm_from_clean_perm  # noqa: E402 (lazy import to avoid circular deps)
        phys_perm_arr = phys_perm_from_clean_perm(clean_perm)
        traj_output = output_dir / "attention_trajectory"
        attn_extra_meta = {
            "order_policy": args.run_kind,
            "start_ckpt": str(args.resume_ckpt) if args.resume_ckpt else "none",
            "start_step": int(start_step),
            "max_steps": int(args.max_steps),
            "seed": int(args.seed),
            "eval_indices_hash": sha256_int_array(split["eval_indices"]),
            "data_source": str(args.data_source),
            "device": str(args.device),
            "wandb_run_name": args.wandb_run_name if args.wandb_run_name else str(output_dir.name),
        }
        n_layer = model.config.n_layer
        if args.attn_composition_pairs == "adjacent":
            comp_pairs = [(i, i + 1) for i in range(n_layer - 1)]
        else:
            comp_pairs = [(i, j) for i in range(n_layer) for j in range(i + 1, n_layer)]
        attn_traj_logger = AttentionTrajectoryLogger(
            output_root=traj_output,
            run_name=output_dir.name,
            n_attention_samples=args.attn_trajectory_samples,
            wandb_run=wandb_run,
            phys_perm=phys_perm_arr,
            heatmap_interval=args.attn_trajectory_heatmap_interval,
            heatmap_png_interval=args.attn_trajectory_heatmap_interval,
            seed=args.seed,
            extra_metadata=attn_extra_meta,
            composition_pairs=comp_pairs,
        )
        attn_traj_logger.set_log_fn(log)
        log(f"[attn_trajectory] enabled — {args.attn_trajectory_samples} fixed samples, "
            f"heatmap every {args.attn_trajectory_heatmap_interval} steps, "
            f"output={traj_output}")

    if start_step == 0 and 0 in save_steps:
        lr0 = get_lr(0, args)
        alpha0 = alpha_for_step(0, start_step, args)
        metrics0 = run_eval_and_save(0, float("nan"), lr0, alpha0)
        save_ckpt(0, metrics0)
    if args.run_kind == "frozen_beta":
        if not args.frozen_beta_ckpt:
            raise ValueError("run-kind=frozen_beta requires --frozen-beta-ckpt")
        if args.gbeta_input_mode == "all_layers":
            # All-layer head-gated: extract ALL heads from ALL layers, gate across L*H heads
            from batch_readout.hook_order_provider import AllLayerHeadGatedHookOrderProvider
            beta_provider = AllLayerHeadGatedHookOrderProvider(
                g_beta_ckpt=args.frozen_beta_ckpt,
                clean_perm=clean_perm, refresh_every=args.frozen_beta_refresh,
                topk=args.gbeta_topk,
                mode=args.frozen_beta_mode, tau=args.frozen_beta_tau,
                seed=args.seed, device=str(device), none_mode=args.frozen_beta_none_mode,
                reverse=args.frozen_beta_rev,
            )
            log(f"[frozen_beta all-layers] g_β={args.frozen_beta_ckpt} "
                f"all_layers topk={args.gbeta_topk} "
                f"none_mode={args.frozen_beta_none_mode} "
                f"mode={args.frozen_beta_mode} rev={args.frozen_beta_rev} "
                f"refresh_every={args.frozen_beta_refresh}")
        elif args.gbeta_input_mode == "layer_heads":
            # Head-gated: extract ALL heads from one layer, apply learned gate
            from batch_readout.hook_order_provider import HeadGatedHookOrderProvider
            beta_provider = HeadGatedHookOrderProvider(
                g_beta_ckpt=args.frozen_beta_ckpt, layer=args.gbeta_layer,
                clean_perm=clean_perm, refresh_every=args.frozen_beta_refresh,
                topk=args.gbeta_topk,
                mode=args.frozen_beta_mode, tau=args.frozen_beta_tau,
                seed=args.seed, device=str(device), none_mode=args.frozen_beta_none_mode,
                reverse=args.frozen_beta_rev,
            )
            log(f"[frozen_beta head-gated] g_β={args.frozen_beta_ckpt} "
                f"layer=L{args.gbeta_layer} topk={args.gbeta_topk} "
                f"none_mode={args.frozen_beta_none_mode} "
                f"mode={args.frozen_beta_mode} rev={args.frozen_beta_rev} "
                f"refresh_every={args.frozen_beta_refresh}")
        elif args.batch_mean_probes > 1:
            # Model-frame batch-mean path: strict65 extraction, mean B over N
            # probe forwards, multi-head g_beta — fully label-free (L0-only).
            from batch_readout.frozen_gbeta_hook import FrozenGBetaModelFrameBlockProvider
            beta_provider = FrozenGBetaModelFrameBlockProvider(
                g_beta_ckpt=args.frozen_beta_ckpt,
                batch_mean_probes=args.batch_mean_probes,
                refresh_every=args.frozen_beta_refresh,
                seed=args.seed, device=str(device),
            )
            if args.frozen_beta_none_mode not in ("model", "content"):
                log(f"[frozen_beta model-frame] WARNING: none_mode={args.frozen_beta_none_mode} "
                    f"→ forcing to 'model' for batch_mean_probes path")
            args.frozen_beta_none_mode = "model"
            log(f"[frozen_beta model-frame batch_mean] g_β={args.frozen_beta_ckpt} "
                f"batch_mean_probes={args.batch_mean_probes} "
                f"none_mode={args.frozen_beta_none_mode} "
                f"mode={args.frozen_beta_mode} rev={args.frozen_beta_rev} "
                f"refresh_every={args.frozen_beta_refresh} (label-free, strict65)")
        else:
            # Single-head: existing behaviour
            from batch_readout.hook_order_provider import HookOrderProvider
            beta_provider = HookOrderProvider(
                g_beta_ckpt=args.frozen_beta_ckpt, head=tuple(args.frozen_beta_head),
                clean_perm=clean_perm, refresh_every=args.frozen_beta_refresh,
                mode=args.frozen_beta_mode, tau=args.frozen_beta_tau,
                seed=args.seed, device=str(device), none_mode=args.frozen_beta_none_mode,
                reverse=args.frozen_beta_rev,
            )
            log(f"[frozen_beta] g_β={args.frozen_beta_ckpt} head=L{args.frozen_beta_head[0]}H{args.frozen_beta_head[1]} "
                f"none_mode={args.frozen_beta_none_mode} "
                f"mode={args.frozen_beta_mode} rev={args.frozen_beta_rev} "
                f"refresh_every={args.frozen_beta_refresh}")
    if args.run_kind == "cdl_teacher":
        from batch_readout.cdl_order_provider import CdlOrderProvider
        cdl_provider = CdlOrderProvider(
            head=tuple(args.cdl_teacher_head), clean_perm=clean_perm,
            refresh_every=args.cdl_teacher_refresh, tau_T=args.cdl_teacher_tau,
            mode="C-D+L", seed=args.seed, device=str(device),
            none_mode=args.cdl_teacher_none_mode,
            reverse=getattr(args, 'cdl_teacher_rev', False),
        )
        log(f"[cdl_teacher] C-D+L teacher head=L{args.cdl_teacher_head[0]}H{args.cdl_teacher_head[1]} "
            f"none_mode={args.cdl_teacher_none_mode} rev={getattr(args, 'cdl_teacher_rev', False)} "
            f"refresh_every={args.cdl_teacher_refresh} tau_T={args.cdl_teacher_tau}")
    if args.run_kind == "direct_policy":
        from batch_readout.direct_order_provider import (
            DirectModelFrameOrderProvider,
        )
        direct_provider = DirectModelFrameOrderProvider(
            policy=args.direct_policy,
            lambda_dep=args.direct_policy_lambda_dep,
            batch_mean_probes=args.batch_mean_probes,
            refresh_every=args.direct_policy_refresh,
            seed=args.seed,
            device=str(device),
        )
        audit = direct_policy_audit_metadata(args)
        log(
            "[direct_policy] "
            f"policy={audit['policy']} | "
            f"matrix={audit['matrix_convention']} | "
            f"frame={audit['frame']} | "
            f"layer_heads={audit['layer_heads']} | "
            f"probe_aggregation={audit['probe_aggregation']} | "
            f"head_fusion={audit['head_fusion']} | "
            f"diagonal={audit['diagonal_handling']} | "
            f"order={audit['order_direction']} | "
            f"lambda_dep={audit['lambda_dep']} | "
            f"refresh_every={audit['refresh_every']}"
        )

    if args.track_head_maps and start_step % int(args.track_head_map_interval) == 0:
        _track_head_maps_raw(model, start_step)

    for global_step in range(start_step, args.max_steps):
        alpha = alpha_for_step(global_step, start_step, args)
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(args.grad_accum):
            if continuous:
                idx_batch = sample_stream_batch(
                    stream_train, args.batch_size, SEQ_LEN, args.seed, global_step, micro_step
                )[:, g_gather].to(device)
            else:
                batch_indices = batch_indices_for_step(
                    split["train_shuffle_order"], global_step, micro_step, args.batch_size, args.grad_accum
                )
                idx_batch = idx_model[batch_indices].to(device)

            if (args.run_kind in {"baseline", "random_continuation"}
                    and args.shuffle_granularity != 64):
                token_orders = sample_token_orders_granular(
                    args.batch_size, args.seed, global_step, micro_step,
                    device, args.shuffle_granularity
                )
                _, loss = model.forward_fn(idx_batch, token_orders)
            else:
                random_phys = sample_random_physical_orders(
                    args.batch_size, args.seed, global_step, micro_step, device
                )

                if args.run_kind in {"baseline", "random_continuation"}:
                    loss = order_loss(model, idx_batch, random_phys, clean_perm, device)
                elif args.run_kind == "frozen_beta":
                    if alpha > 0.0:
                        sigma = beta_provider.physical_order(model, idx_batch, global_step).to(device)
                        # model mode: sigma is in model frame; remap to physical for mixing
                        if args.frozen_beta_none_mode in ("model", "content"):
                            sigma = model_blocks_to_physical_blocks(sigma, clean_perm)
                        phys = sigma.unsqueeze(0).expand(args.batch_size, -1)
                        # Per-sample alpha mixing (same pattern as graph_rw)
                        choose_rng = torch.Generator(device=device)
                        choose_rng.manual_seed(args.seed * 100000000 + global_step * 1000 + micro_step)
                        use_beta = torch.rand(args.batch_size, generator=choose_rng, device=device) < alpha
                        mixed = torch.where(use_beta.unsqueeze(1), phys, random_phys)
                        loss = order_loss(model, idx_batch, mixed, clean_perm, device)
                    else:
                        loss = order_loss(model, idx_batch, random_phys, clean_perm, device)
                elif args.run_kind == "cdl_teacher":
                    if alpha > 0.0:
                        sigma_phys = cdl_provider.physical_order(model, idx_batch, global_step).to(device)
                        phys = sigma_phys.unsqueeze(0).expand(args.batch_size, -1)
                        # Per-sample alpha mixing
                        choose_rng = torch.Generator(device=device)
                        choose_rng.manual_seed(args.seed * 100000000 + global_step * 1000 + micro_step)
                        use_cdl = torch.rand(args.batch_size, generator=choose_rng, device=device) < alpha
                        mixed = torch.where(use_cdl.unsqueeze(1), phys, random_phys)
                        loss = order_loss(model, idx_batch, mixed, clean_perm, device)
                    else:
                        loss = order_loss(model, idx_batch, random_phys, clean_perm, device)
                elif args.run_kind == "direct_policy":
                    if alpha > 0.0:
                        sigma_model = direct_provider.physical_order(
                            model, idx_batch, global_step
                        ).to(device)
                        sigma_phys = model_blocks_to_physical_blocks(
                            sigma_model, clean_perm
                        )
                        phys = sigma_phys.unsqueeze(0).expand(
                            args.batch_size, -1
                        )
                        choose_rng = torch.Generator(device=device)
                        choose_rng.manual_seed(
                            args.seed * 100000000
                            + global_step * 1000
                            + micro_step
                        )
                        use_direct = (
                            torch.rand(
                                args.batch_size,
                                generator=choose_rng,
                                device=device,
                            )
                            < alpha
                        )
                        mixed = torch.where(
                            use_direct.unsqueeze(1),
                            phys,
                            random_phys,
                        )
                        loss = order_loss(
                            model, idx_batch, mixed, clean_perm, device
                        )
                    else:
                        loss = order_loss(
                            model, idx_batch, random_phys, clean_perm, device
                        )
                elif args.run_kind == "l2r":
                    l2r = torch.arange(N, dtype=torch.long, device=device).unsqueeze(0).expand(args.batch_size, -1)
                    loss = order_loss(model, idx_batch, l2r, clean_perm, device)
                elif args.run_kind == "shuffled_l2r":
                    # Control: AR along the data's shuffled layout (model_ascending), NOT the
                    # recovered original order. fixed_phys maps model-ascending back to physical
                    # so order_loss reproduces exactly the `val_model_order` eval traversal.
                    # Same clean_perm/data as random; only the (fixed, wrong-adjacency) order differs.
                    fixed_phys = model_blocks_to_physical_blocks(
                        torch.arange(N, dtype=torch.long, device=device), clean_perm
                    ).unsqueeze(0).expand(args.batch_size, -1)
                    loss = order_loss(model, idx_batch, fixed_phys, clean_perm, device)
                elif args.run_kind == "graph_rw":
                    if should_sample_rw(alpha, rw_policy, rw_mlp):
                        rw_phys = sample_rw_physical_orders(
                            args.batch_size, B, rw_policy, rw_params, args.seed, global_step, micro_step,
                            device, mlp=rw_mlp,
                        )
                        choose_rng = torch.Generator(device=device)
                        choose_rng.manual_seed(args.seed * 100000000 + global_step * 1000 + micro_step)
                        use_rw = torch.rand(args.batch_size, generator=choose_rng, device=device) < alpha
                        mixed = torch.where(use_rw.unsqueeze(1), rw_phys, random_phys)
                        loss = order_loss(model, idx_batch, mixed, clean_perm, device)
                    else:
                        loss = order_loss(model, idx_batch, random_phys, clean_perm, device)
                else:
                    random_loss = order_loss(model, idx_batch, random_phys, clean_perm, device)
                    weighted = (1.0 - alpha) * random_loss
                    if alpha > 0.0:
                        rw_total = 0.0
                        for bag_idx in range(args.rw_order_bag_k):
                            rw_phys = sample_rw_physical_orders(
                                args.batch_size, B, rw_policy, rw_params, args.seed,
                                global_step, micro_step, device, bag_idx=bag_idx, mlp=rw_mlp,
                            )
                            rw_total = rw_total + order_loss(model, idx_batch, rw_phys, clean_perm, device)
                        weighted = weighted + alpha * rw_total / float(args.rw_order_bag_k)
                    loss = weighted

            total_loss += float(loss.item())
            (loss / args.grad_accum).backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        lr = get_lr(global_step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()

        avg_loss = total_loss / args.grad_accum
        train_losses.append(avg_loss)
        next_step = global_step + 1

        # ── per-head CDL signal tracking (lightweight, every N steps) ──
        if args.track_head is not None and next_step % args.track_head_interval == 0:
            _track_head_signal(model, next_step)
        if args.track_all_heads and next_step % args.track_head_interval == 0:
            _track_all_heads_signal(model, next_step)
        if args.track_head_maps and next_step % int(args.track_head_map_interval) == 0:
            _track_head_maps_raw(model, next_step)

        if global_step % args.log_interval == 0:
            now = time.time()
            elapsed = now - t0
            elapsed_since_log = now - last_log_time
            s_per_step = elapsed_since_log / args.log_interval
            steps_left = args.max_steps - global_step
            eta = s_per_step * steps_left
            eta_str = f"{eta/60:.0f}m" if eta < 3600 else f"{eta/3600:.1f}h"
            log(
                f"step {global_step:5d}->{next_step:5d}/{args.max_steps} | "
                f"loss={avg_loss:.4f} | alpha={alpha:.3f} | lr={lr:.2e} | "
                f"{s_per_step:.2f}s/step | ETA {eta_str} | total {elapsed:.0f}s"
            )
            if wandb_run is not None:
                wandb_run.log(
                    wandb_train_payload(next_step, avg_loss, alpha, lr),
                    step=int(next_step),
                )
            last_log_time = now

        if next_step % args.eval_interval == 0 or next_step in save_steps or next_step == args.max_steps:
            metrics = run_eval_and_save(next_step, avg_loss, lr, alpha_for_step(next_step, start_step, args))
        else:
            metrics = last_metrics

        if (next_refresh_step is not None
                and next_step >= next_refresh_step
                and args.run_kind in {"graph_rw", "graph_rw_bag"}):
            if args.refresh_data_source == "random_train":
                extract_n = min(args.refresh_n_chunks, len(idx_train))
                rng = np.random.RandomState(next_step + args.seed)
                refresh_chunks = idx_train[rng.choice(len(idx_train), size=extract_n, replace=False)]
                log(f"[Refresh @ {next_step}] Extracting A from current model on {extract_n} random train chunks...")
            else:
                extract_n = min(args.refresh_n_chunks, len(idx_eval_model))
                refresh_chunks = idx_eval_model
                log(f"[Refresh @ {next_step}] Extracting A from current model on {extract_n} chunks...")
            B, A_global, elapsed, n_extracted = refresh_rw_graph(
                model, refresh_chunks, clean_perm, device, A_global,
                n_chunks=extract_n, ema_beta=args.refresh_ema_beta,
            )
            log(f"[Refresh @ {next_step}] Done in {elapsed:.1f}s ({n_extracted} chunks), A_global saved")
            np.save(output_dir / f"A_global_step{next_step}.npy", A_global)
            np.save(output_dir / "A_global_eval.npy", A_global)
            if rw_policy == "mlp_cdl" and args.mlp_alternating:
                from attn_order_distill import distill_order_mlp, refresh_diagnostics
                seed_r = int(args.seed) * 100000 + int(next_step)
                init = rw_mlp if (args.mlp_refresh_mode == "finetune" and rw_mlp is not None) else None
                rw_mlp, ddiag = distill_order_mlp(
                    B, mlp=init, n_orders=args.mlp_distill_n_orders, tau_T=args.mlp_distill_tau_t,
                    tau_train=args.mlp_distill_tau_train, epochs=args.mlp_distill_epochs,
                    lr=args.mlp_distill_lr, batch_states=args.mlp_distill_batch_states,
                    seed=seed_r, device=device,
                )
                rdiag = refresh_diagnostics(
                    B, rw_mlp, tau=float(args.mlp_tau), top_k=int(args.rw_top_k),
                    src_rho=float(rw_params["src_rho"]), seed=seed_r + 1, device=device,
                )
                torch.save(rw_mlp.state_dict(), output_dir / f"beta_step{next_step}.pt")
                rec = dict(step=int(next_step), refresh_mode=args.mlp_refresh_mode, **ddiag, **rdiag)
                with (output_dir / "refresh_diagnostics.jsonl").open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                log(f"[Refresh+Distill @ {next_step}] beta={args.mlp_refresh_mode} "
                    f"val_kl={ddiag['val_kl']} top1={ddiag['top1']} top4={ddiag['top4']} | "
                    f"rollout tau_vs_l2r={rdiag['rollout_tau_vs_l2r']} ent={rdiag['rollout_entropy']} "
                    f"uniq={rdiag['rollout_unique']} src={rdiag['src_node']} | teacher_tau={rdiag['teacher_tau_vs_l2r']}")
            next_refresh_step = next_step + args.refresh_interval

        if next_step in save_steps or next_step == args.max_steps:
            save_ckpt(next_step, metrics)

    if wandb_run is not None:
        wandb_run.finish()
    if attn_traj_logger is not None:
        attn_traj_logger.flush_summary_log()
        log(f"[attn_trajectory] trajectory summary written to {attn_traj_logger.output_root}")
    log(f"Done. eval_curve={eval_curve_path}")


if __name__ == "__main__":
    main()
