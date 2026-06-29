"""
Phase 2: Train AO-GPT with Graph-RW stochastic order policy (NO Order Network).

At each training step, samples block orders from pi_RW(sigma|B) and trains
AO-GPT under those orders.  An alpha schedule mixes in unstructured (random)
orders as a regularizer.  Every R=1500 steps, re-extracts attention from the
current model (EMA refresh) and updates B.

Usage:
    python -u train_aogpt_graph_rw.py --max-steps 3000 --output-dir probe_results/graph_rw_pilot/seed_42
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", "AO-GPT-MDM"))

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import GPT2TokenizerFast

from model_AOGPT_AdaLN6_NoRep_cond_128_trunc_qknorm import AOGPT, AOGPTConfig
from directed_graph_policy import (
    build_directed_graph,
    compute_source,
    sample_order,
    sample_orders,
    policy_step_entropy,
)
from extract_real_attention import (
    extract_attention_for_sequence,
    select_structural_heads,
    aggregate_to_block_attention,
)
from order_diagnostics import _kendall_tau

# ── Paths ────────────────────────────────────────────────────────────────────────
AO_GPT_CKPT = os.path.expanduser(
    "~/ych/nanogpt-learned-order/out/base/permute/seq256/block64/"
    "out-wikitext103-seq256-random-b64-permute-block-50000-iters/ckpt.pt"
)
WIKITEXT_DIR = os.path.expanduser(
    "~/.cache/huggingface/datasets/wikitext/wikitext-103-raw-v1/0.0.0/"
    "b08601e04326c79dfdd32d625aee71d232d685c3"
)
TOKENIZER_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--gpt2/snapshots/"
    "607a30d783dfa663caf39e06633721c8d4cfcd7e"
)
A_PATH_DEFAULT = os.path.join(_SCRIPT_DIR, "probe_results", "A_train_n64_10k.npy")
OUTPUT_DIR_DEFAULT = os.path.join(_SCRIPT_DIR, "probe_results", "graph_rw_pilot", "seed_42")

# ── Constants ─────────────────────────────────────────────────────────────────────
SEQ_LEN = 256
N = 64              # number of blocks
BLOCK_LEN = 4       # tokens per block (256/64 = 4)


# ═══════════════════════════════════════════════════════════════════════════════════
# Coordinate Mapping
#
# From checkpoint: block_perm[model_idx] = phys_idx, inv_perm[phys_idx] = model_idx.
#   phys_block_to_model_block(phys) = inv_perm[phys]     # phys → model
#   model_block_to_phys_block(model) = block_perm[model]  # model → phys
#
# The AOGPT was trained with blocks permuted according to block_perm, so the model's
# internal token order DOES NOT match the original text L2R order.  These helpers
# translate between "physical" block indices (original text order, block 0 = tokens
# 0..3 of the raw text) and "model" block indices (where those tokens are stored in
# the model's coordinate space).
# ═══════════════════════════════════════════════════════════════════════════════════

def phys_block_to_model_block(phys_block, inv_perm):
    """Return the model-block index that holds the given physical block.

    inv_perm[phys] = model, so phys_block_to_model_block[p] = inv_perm[p].
    """
    return inv_perm[phys_block]

def model_block_to_phys_block(model_block, block_perm):
    """Return the physical-block index stored at the given model block.

    block_perm[model] = phys, so model_block_to_phys_block[m] = block_perm[m].
    """
    return block_perm[model_block]

def phys_to_model_idx(idx_phys, inv_perm):
    """Scatter token sequences from physical (L2R text) order into model-coordinate order.

    Physical token at position s moves to model position inv_perm[s//blk]*blk + s%blk.
    This is the inverse of model_to_phys_idx.
    """
    B_val, T = idx_phys.shape
    blk_size = T // len(inv_perm)
    device = idx_phys.device
    idx_model = torch.zeros_like(idx_phys)
    for s in range(T):
        phys_block = s // blk_size
        model_block = inv_perm[phys_block].item()
        model_pos = model_block * blk_size + (s % blk_size)
        idx_model[:, model_pos] = idx_phys[:, s]
    return idx_model

def model_to_phys_idx(idx_model, block_perm):
    """Scatter token sequences from model-coordinate order back to physical order.

    Model token at position s moves to physical position block_perm[s//blk]*blk + s%blk.
    This is the inverse of phys_to_model_idx.
    """
    B_val, T = idx_model.shape
    blk_size = T // len(block_perm)
    device = idx_model.device
    idx_phys = torch.zeros_like(idx_model)
    for s in range(T):
        model_block = s // blk_size
        phys_block = block_perm[model_block].item()
        phys_pos = phys_block * blk_size + (s % blk_size)
        idx_phys[:, phys_pos] = idx_model[:, s]
    return idx_phys

def verify_coordinate_round_trip(block_perm, inv_perm, blk_size=4, N=64):
    """Verify that phys<->model coordinate round-trips hold."""
    T = N * blk_size
    idx_phys = torch.tensor([[i for i in range(T)]], dtype=torch.long)
    idx_model = phys_to_model_idx(idx_phys, inv_perm)
    idx_phys_back = model_to_phys_idx(idx_model, block_perm)
    ok_forward = torch.equal(idx_phys, idx_phys_back)

    # Check block-level helpers
    ok_block = True
    for phys in range(N):
        model = phys_block_to_model_block(phys, inv_perm)
        back = model_block_to_phys_block(model, block_perm)
        if back != phys:
            ok_block = False
            break
    for model in range(N):
        phys = model_block_to_phys_block(model, block_perm)
        back = phys_block_to_model_block(phys, inv_perm)
        if back != model:
            ok_block = False
            break

    if not (ok_forward and ok_block):
        raise RuntimeError(
            f"Coordinate round-trip failed: token={ok_forward}, block={ok_block}"
        )

def phys_n64_block_order_to_model_token_order(block_order, inv_perm):
    """Convert physical N64 block order -> model-coordinate token order.

    Args:
        block_order: (B, 64) physical block indices at each reveal step.
            block_order[:, t] = physical block to reveal at step t.
        inv_perm: (64,) inv_perm[phys] = model (i.e. phys_block_to_model_block).
    Returns:
        token_order: (B, 256) token-level order for AO-GPT forward_fn.
            tokens token_order[:, t*4:(t+1)*4] are the 4 tokens of
            physical block block_order[:, t], addressed in model coordinates.
    """
    B_bs, N_blocks = block_order.shape  # B_bs, 64
    T = N_blocks * BLOCK_LEN  # 256
    device = block_order.device
    ip = inv_perm.to(device)

    token_order = torch.zeros(B_bs, T, dtype=torch.long, device=device)
    for t in range(N_blocks):
        phys_blk = block_order[:, t]        # (B,) physical block index
        model_blk = ip[phys_blk]             # (B,) model block where it lives
        for k in range(BLOCK_LEN):
            token_order[:, t * BLOCK_LEN + k] = model_blk * BLOCK_LEN + k
    return token_order


# ═══════════════════════════════════════════════════════════════════════════════════
# Data Loading (verbatim from train_aogpt_with_on.py)
# ═══════════════════════════════════════════════════════════════════════════════════

def load_train_chunks(n_chunks=None):
    """Load token chunks from wikitext-103 TRAIN set, replicating extract_train_A.py logic.

    Identical chunking logic: long texts produce sliding windows (stride=128),
    short texts accumulate in a buffer and get concatenated into 256-token chunks.
    Deterministic: produces the same chunks as extract_train_A.py for the same n_chunks.

    If n_chunks is None, loads ALL available chunks.
    """
    from tqdm import tqdm
    tok = GPT2TokenizerFast.from_pretrained(TOKENIZER_DIR, local_files_only=True)

    chunks = []
    buffer_ids = []
    total_texts = 0
    _unlimited = (n_chunks is None)

    for shard in ["wikitext-train-00000-of-00002.arrow", "wikitext-train-00001-of-00002.arrow"]:
        ds = Dataset.from_file(os.path.join(WIKITEXT_DIR, shard))
        for ex in tqdm(ds, desc=f"Chunking {shard}", unit=" texts"):
            total_texts += 1
            ids = tok.encode(ex["text"])
            if len(ids) < SEQ_LEN:
                buffer_ids.extend(ids)
                while len(buffer_ids) >= SEQ_LEN:
                    chunks.append(torch.tensor(buffer_ids[:SEQ_LEN], dtype=torch.long))
                    buffer_ids = buffer_ids[SEQ_LEN:]
                    if not _unlimited and len(chunks) >= n_chunks:
                        break
            else:
                for start in range(0, len(ids) - SEQ_LEN + 1, SEQ_LEN // 2):
                    chunks.append(torch.tensor(ids[start:start + SEQ_LEN], dtype=torch.long))
                    if not _unlimited and len(chunks) >= n_chunks:
                        break
            if not _unlimited and len(chunks) >= n_chunks:
                break
        if not _unlimited and len(chunks) >= n_chunks:
            break
    print(f"Processed {total_texts} texts, got {len(chunks)} chunks", flush=True)
    return torch.stack(chunks)  # (n_chunks, 256)


# ═══════════════════════════════════════════════════════════════════════════════════
# Model Loading (verbatim from train_aogpt_with_on.py)
# ═══════════════════════════════════════════════════════════════════════════════════

def load_aogpt(ckpt_path, device, freeze=False):
    """Load AO-GPT checkpoint. Returns (model, block_perm, inv_perm)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sig = list(AOGPTConfig.__init__.__code__.co_varnames)
    valid = {k: v for k, v in dict(ckpt["model_args"]).items() if k in sig}
    model = AOGPT(AOGPTConfig(**valid))
    sd = ckpt["model"]
    for k in list(sd.keys()):
        clean = k.replace("_orig_mod.", "")
        if clean != k:
            sd[clean] = sd.pop(k)
    model.load_state_dict(sd)
    model.crop_block_size(SEQ_LEN)
    model.to(device)
    if freeze:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
    dp = ckpt["data_permutation"]
    convention = dp.get("convention", "")
    if convention == "clean_phys_to_model":
        # clean_base stores block_perm[phys]=model, inverse[model]=phys
        # but the code expects block_perm[model]=phys, inv_perm[phys]=model
        inv_perm = torch.tensor(dp["block_perm"], dtype=torch.long)
        block_perm = torch.tensor(dp["inverse_block_perm"], dtype=torch.long)
    else:
        block_perm = torch.tensor(dp["block_perm"], dtype=torch.long)
        inv_perm = torch.tensor(dp["inverse_block_perm"], dtype=torch.long)
    return model, block_perm, inv_perm


# ═══════════════════════════════════════════════════════════════════════════════════
# pi_RW Batch Order Sampling
# ═══════════════════════════════════════════════════════════════════════════════════

def sample_rw_batch_orders(B, policy, params, batch_size, device, seed_base, step):
    """Sample a batch of orders from pi_RW.

    Each sample in the batch gets a unique seed for reproducibility.
    Returns: (batch_size, N) LongTensor of block permutations.
    """
    N_blocks = B.shape[0]
    orders = np.zeros((batch_size, N_blocks), dtype=np.int64)
    for b in range(batch_size):
        seed = seed_base * 100000 + step * batch_size + b
        order, _ = sample_order(B, policy, params, seed)
        orders[b] = order
    return torch.from_numpy(orders).long().to(device)


def compute_aogpt_order_loss(aogpt, idx_b, block_order_b, inv_perm, device):
    """Compute AO-GPT loss for one sample under one physical N64 block order."""
    token_order = phys_n64_block_order_to_model_token_order(
        block_order_b, inv_perm
    )  # (1, 256)
    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
        _, loss = aogpt.forward_fn(idx_b, token_order)
    return loss


# ═══════════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_aogpt_rw(model, idx_val, B, policy, params, block_perm, inv_perm, device,
                       eval_order_seeds, max_eval_seqs):
    """Evaluate 4 order modes.

    - val_rw_order:  pi_RW stochastic (M seeds averaged)
    - val_model_order: model-coordinate ascending (model block order 0..63)
    - val_unstructured_order:  random block permutations (M seeds averaged)
    - val_ori_l2r:  original physical L2R block order [0, 1, ..., 63]

    Returns dict with keys matching those names + val_ar (alias for val_model_order).
    """
    model.eval()
    M = len(eval_order_seeds)
    N_blocks = B.shape[0]

    rw_losses = []
    model_order_losses = []
    rand_losses = []
    ori_l2r_losses = []

    n_eval = min(len(idx_val), max_eval_seqs)

    for seq_idx in range(n_eval):
        idx = idx_val[seq_idx:seq_idx + 1].to(device)

        # val_rw_order (M seeds)
        rw_sum = 0.0
        for m in range(M):
            order_np, _ = sample_order(B, policy, params,
                                       seed=eval_order_seeds[m] * 10000 + seq_idx)
            block_order = torch.from_numpy(order_np).unsqueeze(0).long().to(device)
            token_order = phys_n64_block_order_to_model_token_order(block_order, inv_perm)
            _, loss = model.forward_fn(idx, token_order)
            rw_sum += loss.item()
        rw_losses.append(rw_sum / M)

        # val_model_order: model-coordinate ascending (model's native AR mode = model blocks 0..63)
        _, model_order_loss = model(idx, mode='AR')
        model_order_losses.append(model_order_loss.item())

        # val_unstructured_order (M random permutations)
        rand_sum = 0.0
        rng = np.random.default_rng(seq_idx)
        for m in range(M):
            rand_order_np = rng.permutation(N_blocks)
            block_order = torch.from_numpy(rand_order_np).unsqueeze(0).long().to(device)
            token_order = phys_n64_block_order_to_model_token_order(block_order, inv_perm)
            _, rand_loss = model.forward_fn(idx, token_order)
            rand_sum += rand_loss.item()
        rand_losses.append(rand_sum / M)

        # val_ori_l2r: physical block order [0, 1, 2, ..., 63] (original text order)
        ori_l2r_block = torch.arange(N_blocks, device=device).unsqueeze(0)
        token_order_ori_l2r = phys_n64_block_order_to_model_token_order(ori_l2r_block, inv_perm)
        _, ori_l2r_loss = model.forward_fn(idx, token_order_ori_l2r)
        ori_l2r_losses.append(ori_l2r_loss.item())

    model.train()
    return {
        'val_rw_order': float(np.mean(rw_losses)),
        'val_model_order': float(np.mean(model_order_losses)),
        'val_unstructured_order': float(np.mean(rand_losses)),
        'val_ori_l2r': float(np.mean(ori_l2r_losses)),
    }


# ═══════════════════════════════════════════════════════════════════════════════════
# Attention Extraction for Refresh
# ═══════════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def refresh_extract_attention(model, refresh_indices, tokens_phys, device):
    """Extract A_curr from current model on fixed REFRESH_SUBSET.

    Uses extract_attention_for_sequence for each sequence in the subset,
    then averages the resulting A matrices.

    Args:
        model: current AOGPT model
        refresh_indices: (2000,) int64 array of chunk indices
        tokens_phys: (n_chunks, 256) token tensor in physical coordinates
        device: torch device
    Returns:
        A_curr: (64, 64) float32 averaged attention matrix
    """
    A_sum = np.zeros((N, N), dtype=np.float64)
    n_refresh = len(refresh_indices)

    for idx in refresh_indices:
        # Extract tokens for this chunk, reshape to (N, BLOCK_LEN)
        block_seqs = tokens_phys[idx].reshape(N, BLOCK_LEN).long().to(device)
        A_seq, _ = extract_attention_for_sequence(
            model, block_seqs, device,
            model_source="ours",
            num_blocks=N,
            M=3,  # 3 random-order passes
            debug=False,
        )
        A_sum += A_seq.astype(np.float64)

    A_curr = (A_sum / n_refresh).astype(np.float32)
    np.fill_diagonal(A_curr, 0.0)
    return A_curr


# ═══════════════════════════════════════════════════════════════════════════════════
# Refresh Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════════

def compute_refresh_diagnostics(B, A_global_old, A_global_new, policy, params, round_num,
                                prev_orders=None):
    """Compute all refresh-boundary diagnostics."""
    # Drift
    drift = (np.linalg.norm(A_global_new - A_global_old, 'fro') /
             max(np.linalg.norm(A_global_old, 'fro'), 1e-10))

    # Top-3 overlap
    N_blocks = A_global_new.shape[0]
    overlap_sum = 0.0
    for i in range(N_blocks):
        old_top3 = set(np.argsort(-A_global_old[i])[:3])
        new_top3 = set(np.argsort(-A_global_new[i])[:3])
        overlap_sum += len(old_top3 & new_top3) / 3.0
    top3_overlap = float(overlap_sum / N_blocks)

    # pi_RW diagnostics via sample_orders
    result = sample_orders(B, policy, params, K=5000, seed_base=42)

    # Round-to-round tau (use same seeds)
    r2r_tau = None
    if prev_orders is not None:
        current_orders = np.zeros((5000, N_blocks), dtype=np.int64)
        for k in range(5000):
            seed = 42 * 10000 + k
            current_orders[k], _ = sample_order(B, policy, params, seed)
        taus = [_kendall_tau(prev_orders[k], current_orders[k]) for k in range(5000)]
        r2r_tau = float(np.mean(taus))

    pse = result['policy_step_entropy']

    return {
        'round': round_num,
        'A_global_drift': float(drift),
        'A_global_top3_overlap': top3_overlap,
        'pi_RW_first_node_entropy': result['first_node_entropy'],
        'pi_RW_pairwise_tau': result['pairwise_tau_mean'],
        'pi_RW_tau_vs_l2r': result['tau_vs_l2r_mean'],
        'pi_RW_tau_round_r_vs_r_1': r2r_tau,
        'policy_step_entropy_mean': pse['mean'],
        'policy_step_entropy_early': pse['early'],
        'policy_step_entropy_mid': pse['mid'],
        'policy_step_entropy_late': pse['late'],
        'mean_directed_score': result['mean_directed_score'],
        'mean_progressive_support': result['mean_progressive_support'],
    }


# ═══════════════════════════════════════════════════════════════════════════════════
# LR Schedule (adapted from train_aogpt_with_on.py; max_iters -> max_steps)
# ═══════════════════════════════════════════════════════════════════════════════════

def get_lr(iter_num, args):
    """Cosine decay from lr to min_lr over max_steps."""
    if iter_num < args.warmup_iters:
        return args.lr * (iter_num + 1) / max(args.warmup_iters, 1)
    if iter_num > args.max_steps:
        return args.min_lr
    decay_ratio = (iter_num - args.warmup_iters) / (args.max_steps - args.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return args.min_lr + coeff * (args.lr - args.min_lr)


# ═══════════════════════════════════════════════════════════════════════════════════
# Argparse
# ═══════════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Train AO-GPT with Graph-RW order policy (no ON)")
    p.add_argument("--a-path", default=A_PATH_DEFAULT)
    p.add_argument("--ckpt-path", default=AO_GPT_CKPT,
                   help="AO-GPT checkpoint to start from")
    p.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT)
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)  # effective batch = 16
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.99)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-iters", type=int, default=0)
    p.add_argument("--val-fraction", type=float, default=0.05)  # smaller val set
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--max-eval-seqs", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--alpha-target", type=float, default=0.9)
    p.add_argument("--alpha-warmup", type=int, default=10000)
    p.add_argument("--tau-start", type=float, default=0.10)
    p.add_argument("--tau-step", type=float, default=0.10)
    p.add_argument("--rw-policy", type=str, default="progressive_rw",
                   choices=["progressive_rw", "progressive_rw_v2", "progressive_rw_v3"],
                   help="Graph-RW policy variant")
    p.add_argument("--rw-lam", type=float, default=0.75,
                   help="λ trade-off for dependency penalty (v2/v3)")
    p.add_argument("--rw-rho", type=float, default=0.2,
                   help="ρ global readiness prior strength (v3 only)")
    p.add_argument("--rw-top-k", type=int, default=4,
                   help="Restrict RW sampling to top-k candidates per step; <=0 disables")
    p.add_argument("--mixing-mode", choices=["hard", "loss_bag"], default="hard",
                   help="hard: per-sample order choice; loss_bag: weighted random/RW losses")
    p.add_argument("--rw-order-bag-k", type=int, default=1,
                   help="Number of RW orders per sample in loss_bag mode")
    p.add_argument("--refresh-every", type=int, default=1500)
    p.add_argument("--refresh-subset-size", type=int, default=2000)
    p.add_argument("--ema-beta", type=float, default=0.9)
    p.add_argument("--no-refresh", action="store_true",
                   help="Disable refresh loop (fixed-B ablation)")
    # eval_order_seeds
    p.add_argument("--eval-order-seeds", type=int, nargs="+", default=[42, 123, 456])
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    if args.rw_order_bag_k < 1:
        raise ValueError("--rw-order-bag-k must be >= 1")
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "train_log.txt"

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log(f"Device: {device}")
    log(f"Max steps: {args.max_steps}, batch_size: {args.batch_size}, "
        f"grad_accum: {args.grad_accum}, effective: {args.batch_size * args.grad_accum}")
    log(f"LR: {args.lr:.0e} -> {args.min_lr:.0e}, warmup: {args.warmup_iters}")
    log(f"Alpha: 0 -> {args.alpha_target} over {args.alpha_warmup} steps")
    log(f"Tau: start={args.tau_start}, step={args.tau_step}")
    log(f"RW top-k: {args.rw_top_k if args.rw_top_k > 0 else 'disabled'}")
    log(f"Mixing mode: {args.mixing_mode}, rw_order_bag_k: {args.rw_order_bag_k}")
    log(f"Refresh every: {args.refresh_every} steps, EMA beta: {args.ema_beta}")
    log(f"Refresh subset size: {args.refresh_subset_size}")
    log(f"No-refresh: {args.no_refresh}")
    log(f"Output dir: {output_dir}")

    # ── Load A matrices and compute A_global ───────────────────────────────────────
    log(f"\nLoading A matrices from {args.a_path}...")
    A_all = np.load(args.a_path)  # (num_seqs, 64, 64)
    num_seqs = A_all.shape[0]
    log(f"A matrices: {A_all.shape}, dtype: {A_all.dtype}, range: [{A_all.min():.6f}, {A_all.max():.6f}]")

    A_global = A_all.mean(axis=0).astype(np.float32)  # (64, 64)
    np.fill_diagonal(A_global, 0.0)
    log(f"A_global shape: {A_global.shape}, mean: {A_global.mean():.6f}, max: {A_global.max():.6f}")

    B = build_directed_graph(A_global)  # (64, 64) float64
    log(f"B shape: {B.shape}, mean: {B.mean():.6f}")

    # ── Load AO-GPT (unfrozen) ─────────────────────────────────────────────────────
    log("\nLoading AO-GPT (unfrozen)...")
    aogpt, block_perm, inv_perm = load_aogpt(args.ckpt_path, device, freeze=False)
    n_params = sum(p.numel() for p in aogpt.parameters())
    log(f"AO-GPT loaded. Params: {n_params:,}")

    # ── Load data (ALL wikitext-103 train chunks, not limited to num_seqs) ─────────
    log(f"\nLoading ALL wikitext-103 train chunks (A matrices: {num_seqs})...")
    idx_phys = load_train_chunks(n_chunks=None)  # load everything
    total_chunks = idx_phys.shape[0]
    log(f"Token seqs (phys coords): {idx_phys.shape}")

    # Convert to model coordinates (single batch call — O(T) not O(B*T))
    log(f"Converting {total_chunks} chunks to model coords...")
    idx_all = phys_to_model_idx(idx_phys, inv_perm)
    log(f"Converted to model coords: {idx_all.shape}")

    # ── Train/val split ────────────────────────────────────────────────────────────
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(total_chunks)
    val_size = max(1, int(round(total_chunks * args.val_fraction)))
    val_indices = set(int(x) for x in perm[:val_size])
    train_indices = np.array(sorted(set(perm[val_size:].tolist())), dtype=np.int64)
    log(f"Train seqs: {len(train_indices)}, Val seqs: {len(val_indices)}")

    # ── Select fixed REFRESH_SUBSET ────────────────────────────────────────────────
    refresh_rng = np.random.default_rng(args.seed)
    refresh_subset = refresh_rng.choice(
        len(train_indices), size=min(args.refresh_subset_size, len(train_indices)),
        replace=False
    )
    refresh_indices = train_indices[refresh_subset]  # chunk indices into idx_phys
    log(f"Refresh subset: {len(refresh_indices)} indices")

    refresh_subset_path = output_dir / "refresh_subset_indices.npy"
    np.save(refresh_subset_path, refresh_indices)
    log(f"Saved refresh subset indices to {refresh_subset_path}")

    # ── Save eval_order_seeds ──────────────────────────────────────────────────────
    eval_order_seeds_path = output_dir / "eval_order_seeds.json"
    with open(eval_order_seeds_path, "w") as f:
        json.dump({"eval_order_seeds": args.eval_order_seeds}, f)
    log(f"Saved eval order seeds to {eval_order_seeds_path}")

    # ── Define pi_RW params ────────────────────────────────────────────────────────
    policy = args.rw_policy
    if policy == "progressive_rw_v2":
        params = {
            'tau_start': args.tau_start,
            'tau_step': args.tau_step,
            'alpha_dep': 0.5,
            'alpha_pr': 0.85,
            'lam': args.rw_lam,
        }
    elif policy == "progressive_rw_v3":
        params = {
            'tau_start': args.tau_start,
            'tau_step': args.tau_step,
            'alpha_dep': 0.5,
            'alpha_pr': 0.85,
            'lam': args.rw_lam,
            'rho': args.rw_rho,
        }
    else:
        params = {
            'tau_start': args.tau_start,
            'tau_step': args.tau_step,
            'alpha_dep': 0.5,
            'alpha_pr': 0.85,
            'beta_sup': 1.0,
            'beta_fut': 0.5,
            'beta_src': 0.2,
            'beta_loc': 0.5,
        }
    if args.rw_top_k > 0:
        params['top_k'] = int(args.rw_top_k)
    if getattr(args, 'epsilon_uniform', 0.0) > 0.0:
        params['epsilon_uniform'] = float(args.epsilon_uniform)
    log(f"\npi_RW policy: {policy}")
    log(f"pi_RW params: {json.dumps(params)}")

    # ── Move data to device once ───────────────────────────────────────────────────
    idx_all_dev = idx_all.to(device)
    # idx_phys stays on CPU for refresh_extract_attention (iterates per-chunk onto GPU)

    # ── Optimizer ──────────────────────────────────────────────────────────────────
    optimizer = aogpt.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.lr,
        betas=(args.beta1, args.beta2),
        device_type="cuda" if device.type == "cuda" else "cpu",
    )

    # ── Save A_global_round0 ───────────────────────────────────────────────────────
    np.save(output_dir / "A_global_round0.npy", A_global)
    log(f"\nSaved A_global_round0.npy")

    # ── Pre-training diagnostics (round 0) ─────────────────────────────────────────
    log("\n=== Pre-training diagnostics (round 0) ===")
    diag_round_0 = compute_refresh_diagnostics(B, A_global, A_global, policy, params, 0,
                                                prev_orders=None)
    round_0_dir = output_dir / "round_0"
    round_0_dir.mkdir(parents=True, exist_ok=True)
    with open(round_0_dir / "diag.json", "w") as f:
        json.dump(diag_round_0, f, indent=2)
    log(f"Round 0 diagnostics: {json.dumps(diag_round_0, indent=2)}")

    # ── Training ───────────────────────────────────────────────────────────────────
    # Coordinate round-trip verification (catches checkpoint / mapping mismatches early)
    verify_coordinate_round_trip(block_perm, inv_perm, blk_size=BLOCK_LEN, N=N)
    log("Coordinate round-trip verification: PASSED")

    # Open eval_curve.tsv
    eval_curve_path = output_dir / "eval_curve.tsv"
    with open(eval_curve_path, "w") as f_tsv:
        f_tsv.write("step\talpha\ttrain_loss\tval_rw_order\tval_model_order\tval_unstructured_order\tval_ori_l2r\tlr\n")

    aogpt.train()
    effective_batch = args.batch_size * args.grad_accum
    train_losses = []
    best_val_rw_order = float("inf")
    best_state = None
    best_step = 0
    t0 = time.time()

    # Track round number for refresh
    round_num = 0
    prev_orders_5k = None  # stored after each refresh for r2r tau

    # Store initial pi_RW orders for round-to-round comparison
    initial_result = sample_orders(B, policy, params, K=5000, seed_base=42)
    prev_orders_5k = initial_result['orders'].copy()

    for step in range(args.max_steps):
        # 1. Alpha schedule: linear 0 -> alpha_target over warmup, constant thereafter
        alpha = min(args.alpha_target, (step / args.alpha_warmup) * args.alpha_target)

        # 2. Gradient accumulation loop
        total_loss = 0.0

        for accum_step in range(args.grad_accum):
            # Get micro-batch tokens
            micro_indices = np.random.choice(train_indices, size=args.batch_size, replace=True)
            micro_idx = idx_all_dev[micro_indices]  # (batch_size, 256)

            # 3. Sample random orders for batch
            rand_orders = torch.zeros(args.batch_size, N, dtype=torch.long, device=device)
            for b in range(args.batch_size):
                rand_orders[b] = torch.randperm(N, device=device)

            if args.mixing_mode == "hard":
                # 4a. Per-sample hard alpha mixing: choose RW or random order.
                rw_orders = sample_rw_batch_orders(
                    B, policy, params, args.batch_size, device,
                    args.seed, step * args.grad_accum + accum_step
                )
                use_rw = torch.rand(args.batch_size, device=device) < alpha
                use_rw_expanded = use_rw.unsqueeze(1).expand(-1, N)
                mixed_block_orders = torch.where(use_rw_expanded, rw_orders, rand_orders)

                for b in range(args.batch_size):
                    idx_b = micro_idx[b:b + 1]
                    loss = compute_aogpt_order_loss(
                        aogpt, idx_b, mixed_block_orders[b:b + 1], inv_perm, device
                    )
                    total_loss += loss.item()
                    (loss / effective_batch).backward()

            else:
                # 4b. Low-variance loss mixing:
                #     L = (1-alpha) * L_random + alpha * mean_k L_RW,k
                rw_order_bags = []
                if alpha > 0.0:
                    for bag_idx in range(args.rw_order_bag_k):
                        order_step = (
                            (step * args.grad_accum + accum_step)
                            * args.rw_order_bag_k
                            + bag_idx
                        )
                        rw_order_bags.append(
                            sample_rw_batch_orders(
                                B, policy, params, args.batch_size, device,
                                args.seed, order_step
                            )
                        )

                for b in range(args.batch_size):
                    idx_b = micro_idx[b:b + 1]

                    rand_loss = compute_aogpt_order_loss(
                        aogpt, idx_b, rand_orders[b:b + 1], inv_perm, device
                    )
                    weighted_loss_value = (1.0 - alpha) * rand_loss.item()
                    ((1.0 - alpha) * rand_loss / effective_batch).backward()

                    if alpha > 0.0:
                        rw_item_sum = 0.0
                        rw_grad_weight = alpha / float(args.rw_order_bag_k)
                        for rw_orders in rw_order_bags:
                            rw_loss = compute_aogpt_order_loss(
                                aogpt, idx_b, rw_orders[b:b + 1], inv_perm, device
                            )
                            rw_item_sum += rw_loss.item()
                            (rw_grad_weight * rw_loss / effective_batch).backward()
                        weighted_loss_value += alpha * rw_item_sum / float(args.rw_order_bag_k)

                    total_loss += weighted_loss_value

        # 7. Gradient clipping and step
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(aogpt.parameters(), args.grad_clip)

        lr = get_lr(step, args)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()
        optimizer.zero_grad()

        avg_loss = total_loss / effective_batch
        train_losses.append(avg_loss)

        # 8. Logging
        if step % args.log_interval == 0 or step == 0:
            elapsed = time.time() - t0
            running_avg = float(np.mean(train_losses[-args.log_interval:])) if step >= args.log_interval else avg_loss
            log(f"Step {step:5d}/{args.max_steps} | "
                f"loss={avg_loss:.4f} | running_avg={running_avg:.4f} | "
                f"alpha={alpha:.3f} | lr={lr:.2e} | "
                f"{elapsed:.0f}s")

        # 9. Periodic evaluation
        if step % args.eval_interval == 0 or step == args.max_steps - 1:
            log(f"  [Eval @ step {step}] ...")
            eval_t0 = time.time()
            metrics = evaluate_aogpt_rw(
                aogpt, idx_all_dev[list(val_indices)], B, policy, params,
                block_perm, inv_perm, device, args.eval_order_seeds, args.max_eval_seqs,
            )
            eval_elapsed = time.time() - eval_t0
            log(f"  [Eval @{step:5d}] val_rw_order={metrics['val_rw_order']:.4f} | "
                f"val_model_order={metrics['val_model_order']:.4f} | "
                f"val_unstructured={metrics['val_unstructured_order']:.4f} | "
                f"val_ori_l2r={metrics['val_ori_l2r']:.4f} | "
                f"({eval_elapsed:.0f}s)")

            # Write to eval_curve.tsv
            with open(eval_curve_path, "a") as f_tsv:
                f_tsv.write(f"{step}\t{alpha:.4f}\t{avg_loss:.6f}\t"
                            f"{metrics['val_rw_order']:.6f}\t"
                            f"{metrics['val_model_order']:.6f}\t"
                            f"{metrics['val_unstructured_order']:.6f}\t"
                            f"{metrics['val_ori_l2r']:.6f}\t"
                            f"{lr:.8e}\n")

            # Track best
            if metrics['val_rw_order'] < best_val_rw_order:
                best_val_rw_order = metrics['val_rw_order']
                best_step = step
                best_state = {k: v.cpu().clone() for k, v in aogpt.state_dict().items()}
                log(f"  [*] New best val_rw_order at step {step}")

        # 10. Refresh (extract attention from model, EMA update B)
        if (not args.no_refresh and step > 0 and step % args.refresh_every == 0):
            round_num += 1
            log(f"\n{'=' * 60}")
            log(f"=== Refresh round {round_num} @ step {step} ===")
            log(f"{'=' * 60}")

            # Extract attention from current model
            log(f"Extracting attention from {len(refresh_indices)} sequences...")
            extraction_t0 = time.time()
            A_curr = refresh_extract_attention(
                aogpt, refresh_indices, idx_phys, device
            )
            extraction_elapsed = time.time() - extraction_t0
            log(f"Extraction done in {extraction_elapsed:.0f}s")
            log(f"A_curr mean: {A_curr.mean():.6f}, max: {A_curr.max():.6f}, "
                f"min: {A_curr.min():.6f}")

            # EMA update: A_global = beta * A_global + (1 - beta) * A_curr
            A_global_old = A_global.copy()
            A_global = args.ema_beta * A_global_old + (1.0 - args.ema_beta) * A_curr
            A_global = A_global.astype(np.float32)
            np.fill_diagonal(A_global, 0.0)
            log(f"A_global_new mean: {A_global.mean():.6f}, max: {A_global.max():.6f}")

            # Rebuild B
            B = build_directed_graph(A_global)
            log(f"B_new mean: {B.mean():.6f}")

            # Compute diagnostics
            diag = compute_refresh_diagnostics(
                B, A_global_old, A_global, policy, params, round_num,
                prev_orders=prev_orders_5k
            )

            # Store current orders for next round's r2r comparison
            result_5k = sample_orders(B, policy, params, K=5000, seed_base=42)
            prev_orders_5k = result_5k['orders'].copy()

            # Save round diagnostics
            round_dir = output_dir / f"round_{round_num}"
            round_dir.mkdir(parents=True, exist_ok=True)
            with open(round_dir / "diag.json", "w") as f:
                json.dump(diag, f, indent=2)
            log(f"Round {round_num} diagnostics: {json.dumps(diag, indent=2)}")

            # Save A_global snapshot
            a_global_path = output_dir / f"A_global_round{round_num}.npy"
            np.save(a_global_path, A_global)
            log(f"Saved {a_global_path}")

    # ── End of training ────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    log(f"\nTraining done. Total time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    log(f"Best val_rw_order: {best_val_rw_order:.4f} at step {best_step}")

    # Save final checkpoint
    ckpt_path = output_dir / f"ckpt_step{args.max_steps}.pt"
    torch.save(
        {
            "model_state_dict": best_state if best_state else aogpt.state_dict(),
            "args": vars(args),
            "train_losses": train_losses,
            "best_val_rw_order": best_val_rw_order,
            "best_step": best_step,
            "policy": policy,
            "params": params,
            "A_global": A_global,
            "B": B,
        },
        ckpt_path,
    )
    log(f"Saved checkpoint: {ckpt_path}")

    # Final eval curve
    log(f"\nEval curve saved to: {eval_curve_path}")
    log(f"Training log saved to: {log_path}")


if __name__ == "__main__":
    main()
