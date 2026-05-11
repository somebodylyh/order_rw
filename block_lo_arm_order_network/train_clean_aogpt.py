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

from clean_training_protocol import (
    CleanPermutation,
    batch_indices_for_step,
    build_clean_block_permutation,
    build_fixed_split_and_shuffle,
    expand_model_blocks_to_token_order,
    model_blocks_to_physical_blocks,
    physical_blocks_to_model_blocks,
    physical_blocks_to_model_token_order,
    phys_to_model_idx_clean,
    sha256_int_array,
    train_cursor_for_next_step,
    verify_clean_coordinate_round_trip,
)
from directed_graph_policy import build_directed_graph, sample_order
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
def extract_A_matrices(model, idx_chunks, clean_perm, device, n_chunks=None):
    """Extract NxN attention matrices from current model on given chunks."""
    if n_chunks is None:
        n_chunks = len(idx_chunks)
    n_chunks = min(n_chunks, len(idx_chunks))

    inv_perm = clean_perm.inv_perm_model_to_phys.cpu().numpy()

    A_all = np.zeros((n_chunks, N, N), dtype=np.float32)
    model.eval()

    for i in range(n_chunks):
        tokens = idx_chunks[i:i+1].to(device)  # (1, 256) model-coordinate tokens

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
        attn_phys = np.zeros((SEQ_LEN, SEQ_LEN), dtype=np.float32)
        for rq in range(SEQ_LEN):
            pq = phys_tokens[rq]
            for rk in range(SEQ_LEN):
                attn_phys[pq, phys_tokens[rk]] += attn_content[rq, rk]

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
    if args.run_kind not in {"graph_rw", "graph_rw_bag"}:
        return 0.0
    local_step = max(0, int(global_step) - int(start_step))
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


def build_model(model_args, device):
    sig = list(AOGPTConfig.__init__.__code__.co_varnames)
    valid = {k: v for k, v in dict(model_args).items() if k in sig}
    model = AOGPT(AOGPTConfig(**valid))
    model.crop_block_size(SEQ_LEN)
    model.to(device)
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


def sample_rw_physical_orders(batch_size, B, policy, params, seed, global_step, micro_step, device, bag_idx=0):
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
def evaluate_orders(model, idx_eval_model, clean_perm, B, rw_policy, rw_params, args, alpha):
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
    for seed in args.eval_order_seeds:
        random_rows = []
        rw_rows = []
        for seq_idx in range(idx_eval_model.size(0)):
            random_rows.append(np.random.default_rng(int(seed) * 10000 + seq_idx).permutation(N))
            rw_order, _ = sample_order(B, rw_policy, rw_params, seed=int(seed) * 10000 + seq_idx)
            rw_rows.append(rw_order)
        unstructured_phys = torch.tensor(np.stack(random_rows), dtype=torch.long)
        rw_phys = torch.tensor(np.stack(rw_rows), dtype=torch.long)
        unstructured_model_orders.append(physical_blocks_to_model_blocks(unstructured_phys, clean_perm))
        rw_model_orders.append(physical_blocks_to_model_blocks(rw_phys, clean_perm))

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

    results = {}
    for name, (orders, desc_order, seeds) in modes.items():
        metrics = eval_model_orders(orders)
        results[name] = {**metrics, **describe(desc_order), "eval_seeds": seeds}

    random_loss = results["val_unstructured_order"]["loss_token_avg"]
    rw_loss = results["val_rw_order"]["loss_token_avg"]
    if args.run_kind in {"graph_rw", "graph_rw_bag"}:
        train_objective = (1.0 - alpha) * random_loss + alpha * rw_loss
    elif args.run_kind == "l2r":
        train_objective = results["val_ori_l2r_block"]["loss_token_avg"]
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


def write_config(output_dir, args, split, clean_perm, rw_policy, rw_params):
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
        "rw_policy": rw_policy,
        "rw_params": rw_params,
        "block_perm_first16": clean_perm.block_perm_phys_to_model[:16].tolist(),
        "inv_perm_first16": clean_perm.inv_perm_model_to_phys[:16].tolist(),
    }
    with (output_dir / "config.json").open("w") as f:
        json.dump(payload, f, indent=2)


def parse_args(default_run_kind="baseline"):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-kind", choices=["baseline", "random_continuation", "graph_rw", "graph_rw_bag", "l2r"],
                   default=default_run_kind)
    p.add_argument("--resume-ckpt", default="")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--permute-seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--max-eval-seqs", type=int, default=200)
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
    p.add_argument("--tau-start", type=float, default=0.1)
    p.add_argument("--tau-step", type=float, default=0.1)
    p.add_argument("--rw-top-k", type=int, default=4)
    p.add_argument("--rw-order-bag-k", type=int, default=1)
    p.add_argument("--epsilon-uniform", type=float, default=0.0,
                   help="Epsilon-uniform exploration mixing (0=disabled, 0.15=recommended)")
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
    log(f"Output dir: {output_dir}")
    log(f"Device: {device}")

    ckpt = None
    if args.resume_ckpt:
        ckpt = torch.load(os.path.expanduser(args.resume_ckpt), map_location=device, weights_only=False)
        log(f"Loaded resume checkpoint: {args.resume_ckpt}")

    clean_perm, loaded_split = load_or_create_protocol(args, output_dir, ckpt)
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

    if ckpt is None:
        model_args = clean_model_args(args)
        model = build_model(model_args, device)
        start_step = 0
        train_losses = []
    else:
        model_args = ckpt.get("model_args") or clean_model_args(args)
        model = build_model(model_args, device)
        state_dict = ckpt.get("model") or ckpt.get("model_state_dict")
        if state_dict is None:
            raise KeyError("resume checkpoint lacks model/model_state_dict")
        model.load_state_dict(clean_state_dict(state_dict))
        start_step = int(ckpt.get("global_step", ckpt.get("iter_num", 0)))
        train_losses = list(ckpt.get("train_losses", []))

    # Initialize Graph-RW policy from current model's attention
    rw_policy = "progressive_rw"
    rw_params = {
        "tau_start": args.tau_start,
        "tau_step": args.tau_step,
        "alpha_dep": 0.5,
        "alpha_pr": 0.85,
        "beta_sup": 1.0,
        "beta_fut": 0.5,
        "beta_src": 0.2,
        "beta_loc": 0.5,
    }
    if args.rw_top_k > 0:
        rw_params["top_k"] = int(args.rw_top_k)
    if args.epsilon_uniform > 0.0:
        rw_params["epsilon_uniform"] = float(args.epsilon_uniform)

    idx_eval_model = idx_model[split["eval_indices"]]
    if args.run_kind in {"graph_rw", "graph_rw_bag"}:
        idx_train = idx_model[split["train_indices"]]
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
                "lr",
            ])

    model.train()
    t0 = time.time()
    last_metrics = {}
    next_refresh_step = start_step + args.refresh_interval if args.refresh_interval > 0 else None

    def run_eval_and_save(global_step, avg_loss, lr, alpha):
        nonlocal last_metrics
        log(f"[Eval @ {global_step}] alpha={alpha:.4f}")
        metrics = evaluate_orders(
            model, idx_eval_model, clean_perm, B, rw_policy, rw_params, args, alpha
        )
        last_metrics = metrics
        with eval_curve_path.open("a", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow([
                global_step,
                f"{alpha:.6f}",
                f"{avg_loss:.6f}",
                f"{metrics['val_train_objective']['loss_token_avg']:.6f}",
                f"{metrics['val_ori_l2r_block']['loss_token_avg']:.6f}",
                f"{metrics['val_ar_l2r']['loss_token_avg']:.6f}",
                f"{metrics['val_model_order']['loss_token_avg']:.6f}",
                f"{metrics['val_unstructured_order']['loss_token_avg']:.6f}",
                f"{metrics['val_rw_order']['loss_token_avg']:.6f}",
                f"{lr:.8e}",
            ])
        log(
            f"[Eval @ {global_step}] train_obj={metrics['val_train_objective']['loss_token_avg']:.4f} | "
            f"ori_l2r={metrics['val_ori_l2r_block']['loss_token_avg']:.4f} | "
            f"model_order={metrics['val_model_order']['loss_token_avg']:.4f} | "
            f"unstructured={metrics['val_unstructured_order']['loss_token_avg']:.4f} | "
            f"rw={metrics['val_rw_order']['loss_token_avg']:.4f}"
        )
        return metrics

    def save_ckpt(global_step, metrics):
        path = output_dir / f"ckpt_step{global_step}.pt"
        payload = checkpoint_payload(
            model, optimizer, args, clean_perm, split, global_step,
            train_losses, metrics, rw_policy, rw_params
        )
        torch.save(payload, path)
        log(f"Saved checkpoint: {path}")

    if start_step == 0 and 0 in save_steps:
        lr0 = get_lr(0, args)
        alpha0 = alpha_for_step(0, start_step, args)
        metrics0 = run_eval_and_save(0, float("nan"), lr0, alpha0)
        save_ckpt(0, metrics0)

    for global_step in range(start_step, args.max_steps):
        alpha = alpha_for_step(global_step, start_step, args)
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(args.grad_accum):
            batch_indices = batch_indices_for_step(
                split["train_shuffle_order"], global_step, micro_step, args.batch_size, args.grad_accum
            )
            idx_batch = idx_model[batch_indices].to(device)

            random_phys = sample_random_physical_orders(
                args.batch_size, args.seed, global_step, micro_step, device
            )

            if args.run_kind in {"baseline", "random_continuation"}:
                loss = order_loss(model, idx_batch, random_phys, clean_perm, device)
            elif args.run_kind == "l2r":
                l2r = torch.arange(N, dtype=torch.long, device=device).unsqueeze(0).expand(args.batch_size, -1)
                loss = order_loss(model, idx_batch, l2r, clean_perm, device)
            elif args.run_kind == "graph_rw":
                rw_phys = sample_rw_physical_orders(
                    args.batch_size, B, rw_policy, rw_params, args.seed, global_step, micro_step, device
                )
                choose_rng = torch.Generator(device=device)
                choose_rng.manual_seed(args.seed * 100000000 + global_step * 1000 + micro_step)
                use_rw = torch.rand(args.batch_size, generator=choose_rng, device=device) < alpha
                mixed = torch.where(use_rw.unsqueeze(1), rw_phys, random_phys)
                loss = order_loss(model, idx_batch, mixed, clean_perm, device)
            else:
                random_loss = order_loss(model, idx_batch, random_phys, clean_perm, device)
                weighted = (1.0 - alpha) * random_loss
                if alpha > 0.0:
                    rw_total = 0.0
                    for bag_idx in range(args.rw_order_bag_k):
                        rw_phys = sample_rw_physical_orders(
                            args.batch_size, B, rw_policy, rw_params, args.seed,
                            global_step, micro_step, device, bag_idx=bag_idx
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

        if global_step % args.log_interval == 0:
            elapsed = time.time() - t0
            log(
                f"step {global_step:5d}->{next_step:5d}/{args.max_steps} | "
                f"loss={avg_loss:.4f} | alpha={alpha:.3f} | lr={lr:.2e} | {elapsed:.0f}s"
            )

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
            next_refresh_step = next_step + args.refresh_interval

        if next_step in save_steps or next_step == args.max_steps:
            save_ckpt(next_step, metrics)

    log(f"Done. eval_curve={eval_curve_path}")


if __name__ == "__main__":
    main()
