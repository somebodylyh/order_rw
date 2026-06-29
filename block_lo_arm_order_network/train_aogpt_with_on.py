"""
Train AO-GPT with frozen ON-generated shared batch order.

Freeze the GRPO-trained Order Network. Each training step:
  1. Sample a batch of sequences
  2. Train AO-GPT on the whole batch with ON global order

Usage:
    python -u train_aogpt_with_on.py --device cuda:0 --max-iters 5000 --batch-size 8
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "AO-GPT-MDM"))

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import GPT2TokenizerFast

from config import Config
from order_network import CrossAttentionOrderNetwork, masks_to_revealed_bool
from model_AOGPT_AdaLN6_NoRep_cond_128_trunc_qknorm import AOGPT, AOGPTConfig

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

ON_CKPT = "probe_results/grpo_on_round2.pt"
A_MATRICES = "probe_results/A_train_10k.npy"
OUTPUT_DIR = "probe_results"
OUTPUT_NAME = "aogpt_on_train_10k_5k.pt"

SEQ_LEN = 256
N16 = 16
SUB_BLOCKS = 4
BLOCK_LEN = 4
N64 = N16 * SUB_BLOCKS  # 64


# ═══════════════════════════════════════════════════════════════════════════════════
# Coordinate Mapping (from train_grpo_on.py)
# ═══════════════════════════════════════════════════════════════════════════════════

def phys_n16_block_order_to_model_token_order(
    n16_order, block_perm, sub_blocks=SUB_BLOCKS, block_len=BLOCK_LEN
):
    """
    Convert physical N16 block order -> model-coordinate token order.

    n16_order:  (B, 16)  -- physical N16 block index at each reveal step
    block_perm: (64,)    -- block_perm[phys64] = model64
    Returns:    (B, 256) -- token-level order for AO-GPT forward_fn
    """
    B, N = n16_order.shape
    T = N * sub_blocks * block_len  # 256
    device = n16_order.device
    bp = block_perm.to(device)

    token_order = torch.zeros(B, T, dtype=torch.long, device=device)
    for t in range(N):
        phys_n16_blk = n16_order[:, t]  # (B,)
        for a in range(sub_blocks):
            phys_n64_blk = phys_n16_blk * sub_blocks + a
            model_n64_blk = bp[phys_n64_blk]
            for k in range(block_len):
                token_order[:, t * sub_blocks * block_len + a * block_len + k] = (
                    model_n64_blk * block_len + k
                )
    return token_order


def phys_to_model_idx(idx_phys, inv_perm):
    """Convert token sequences from physical order to model-coordinate order."""
    B, T = idx_phys.shape
    M = len(inv_perm)
    blk_size = T // M
    device = idx_phys.device
    idx_model = torch.zeros_like(idx_phys)
    for s in range(T):
        model_block = inv_perm[s // blk_size].item()
        offset = s % blk_size
        model_pos = model_block * blk_size + offset
        idx_model[:, model_pos] = idx_phys[:, s]
    return idx_model


# ═══════════════════════════════════════════════════════════════════════════════════
# Data & Model Loading
# ═══════════════════════════════════════════════════════════════════════════════════

def load_train_chunks(n_chunks):
    """Load token chunks from wikitext-103 TRAIN set, replicating extract_train_A.py logic.

    Identical chunking logic: long texts produce sliding windows (stride=128),
    short texts accumulate in a buffer and get concatenated into 256-token chunks.
    Deterministic: produces the same chunks as extract_train_A.py for the same n_chunks.
    """
    from tqdm import tqdm
    tok = GPT2TokenizerFast.from_pretrained(TOKENIZER_DIR, local_files_only=True)

    chunks = []
    buffer_ids = []
    total_texts = 0

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
                    if len(chunks) >= n_chunks:
                        break
            else:
                for start in range(0, len(ids) - SEQ_LEN + 1, SEQ_LEN // 2):
                    chunks.append(torch.tensor(ids[start:start + SEQ_LEN], dtype=torch.long))
                    if len(chunks) >= n_chunks:
                        break
            if len(chunks) >= n_chunks:
                break
        if len(chunks) >= n_chunks:
            break
    print(f"Processed {total_texts} texts, got {len(chunks)} chunks", flush=True)
    return torch.stack(chunks)  # (n_chunks, 256)


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
    block_perm = torch.tensor(ckpt["data_permutation"]["block_perm"], dtype=torch.long)
    inv_perm = torch.tensor(ckpt["data_permutation"]["inverse_block_perm"], dtype=torch.long)
    return model, block_perm, inv_perm


def load_on(on_ckpt_path, device):
    """Load frozen ON from GRPO checkpoint."""
    ckpt = torch.load(on_ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]
    d_edge = sd["edge_mlp.0.weight"].shape[0]
    d_model = sd["score_mlp.0.weight"].shape[0]
    model = CrossAttentionOrderNetwork(num_blocks=N16, d_edge=d_edge, d_model=d_model)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# ═══════════════════════════════════════════════════════════════════════════════════
# ON Greedy Order Generation
# ═══════════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def greedy_order_from_on(on_model, A):
    """
    Deterministic greedy order via sequential argmax.

    Args:
        on_model: frozen CrossAttentionOrderNetwork
        A: (B, N16, N16) attention probability matrix

    Returns:
        orders: (B, N16) LongTensor -- physical N16 block indices per step
    """
    B, N, _ = A.shape
    device = A.device

    visited_mask = torch.zeros(B, dtype=torch.long, device=device)
    last_node = torch.zeros(B, dtype=torch.long, device=device)
    order_list = []

    for step in range(N):
        logits = on_model(A, visited_mask, last_node)  # (B, N)
        visited_bool = masks_to_revealed_bool(visited_mask, N)
        logits = logits.masked_fill(visited_bool, float("-inf"))
        chosen = logits.argmax(dim=-1)  # (B,) -- greedy, no sampling

        order_list.append(chosen)
        visited_mask = visited_mask | (1 << chosen)
        last_node = chosen

    return torch.stack(order_list, dim=1)  # (B, N16)


# ═══════════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_aogpt(model, idx_all, global_token_order_eval, order_mode, eval_indices, args):
    """Compute AR loss, Random loss, and train-order loss on validation set."""
    model.eval()
    device = next(model.parameters()).device
    eval_list = sorted(eval_indices)
    if args.max_eval_seqs and len(eval_list) > args.max_eval_seqs:
        eval_list = eval_list[:args.max_eval_seqs]

    ar_losses = []
    random_losses = []
    order_losses = []

    for seq_idx in eval_list:
        idx = idx_all[seq_idx:seq_idx + 1].to(device)
        _, ar_loss = model(idx, mode='AR')
        ar_losses.append(ar_loss.item())

        rand_sum = 0.0
        for _ in range(args.num_random_evals):
            _, r_loss = model(idx, mode='Random')
            rand_sum += r_loss.item()
        random_losses.append(rand_sum / args.num_random_evals)

        if order_mode == "ar":
            # Training with AR -> train-order loss = AR loss
            order_losses.append(ar_losses[-1])
        else:
            _, on_loss = model.forward_fn(idx, global_token_order_eval)
            order_losses.append(on_loss.item())

    model.train()

    return {
        "ar_loss": float(np.mean(ar_losses)),
        "random_loss": float(np.mean(random_losses)),
        "on_loss": float(np.mean(order_losses)),
    }


# ═══════════════════════════════════════════════════════════════════════════════════
# LR Schedule
# ═══════════════════════════════════════════════════════════════════════════════════

def get_lr(iter_num, args):
    """Cosine decay from lr to min_lr over max_iters."""
    if iter_num < args.warmup_iters:
        return args.lr * (iter_num + 1) / max(args.warmup_iters, 1)
    if iter_num > args.max_iters:
        return args.min_lr
    decay_ratio = (iter_num - args.warmup_iters) / (args.max_iters - args.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return args.min_lr + coeff * (args.lr - args.min_lr)


# ═══════════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Train AO-GPT with frozen ON shared-batch orders")
    p.add_argument("--a-matrices", default=A_MATRICES)
    p.add_argument("--on-ckpt", default=ON_CKPT)
    p.add_argument("--output-dir", default=OUTPUT_DIR)
    p.add_argument("--output-name", default=OUTPUT_NAME)
    p.add_argument("--max-iters", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--min-lr", type=float, default=3e-6)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.99)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-iters", type=int, default=200)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--num-random-evals", type=int, default=4)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--max-eval-seqs", type=int, default=200,
                   help="Max validation sequences to evaluate (limits eval time)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    p.add_argument("--order-mode", default="on", choices=["on", "ar"],
                   help="'on' = ON global order, 'ar' = AR ascending (control)")
    return p.parse_args()


def main():
    args = parse_args()
    Config.seed = args.seed
    Config.set_seed()
    device = torch.device(args.device)

    print(f"Device: {device}", flush=True)
    print(f"Max iters: {args.max_iters}, batch_size: {args.batch_size}, "
          f"LR: {args.lr:.0e} -> {args.min_lr:.0e}", flush=True)

    # ── Load AO-GPT (unfrozen) ──────────────────────────────────────────────
    print("Loading AO-GPT (unfrozen)...", flush=True)
    aogpt, block_perm, inv_perm = load_aogpt(AO_GPT_CKPT, device, freeze=False)
    n_params = sum(p.numel() for p in aogpt.parameters())
    print(f"AO-GPT loaded. Params: {n_params:,}", flush=True)

    # ── Load data ───────────────────────────────────────────────────────────
    print("Loading data...", flush=True)
    A_all = np.load(args.a_matrices)  # (num_seqs, 16, 16)
    num_seqs = A_all.shape[0]
    print(f"A matrices: {args.a_matrices} -> {A_all.shape}")

    # Recreate the SAME chunks as extract_train_A.py (deterministic logic)
    idx_phys = load_train_chunks(num_seqs)
    idx_phys = idx_phys[:num_seqs]
    print(f"Token seqs: {idx_phys.shape[0]}")

    # Convert to model coordinates
    idx_all = torch.cat(
        [phys_to_model_idx(idx_phys[i:i + 1], inv_perm) for i in range(num_seqs)], dim=0
    )
    print(f"Converted to model coords: {idx_all.shape}")

    # ── Load ON (frozen) ────────────────────────────────────────────────────
    print("Loading ON (frozen)...", flush=True)
    on_model = load_on(args.on_ckpt, device)
    n_on = sum(p.numel() for p in on_model.parameters())
    ckpt_meta = torch.load(args.on_ckpt, map_location="cpu", weights_only=False)
    print(f"ON loaded: best_val_reward={ckpt_meta.get('best_val_reward','?'):.4f}, "
          f"best_epoch={ckpt_meta.get('best_epoch','?')}, params={n_on:,}", flush=True)

    # ── Train/val split ─────────────────────────────────────────────────────
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(num_seqs)
    val_size = max(1, int(round(num_seqs * args.val_fraction)))
    val_indices = set(int(x) for x in perm[:val_size])
    train_indices = np.array(sorted(set(perm[val_size:].tolist())), dtype=np.int64)
    print(f"Train seqs: {len(train_indices)}, Val seqs: {len(val_indices)}", flush=True)

    # ── Move data to GPU once ───────────────────────────────────────────────
    A_tensor = torch.as_tensor(A_all, dtype=torch.float32, device=device)
    idx_all_dev = idx_all.to(device)

    # ── Compute global ON order (or use AR for control) ──
    if args.order_mode == "on":
        print("Computing global ON order (mean A over train set)...", flush=True)
        train_A_mean = A_tensor[train_indices].mean(dim=0, keepdim=True)  # (1, 16, 16)
        global_n16_order = greedy_order_from_on(on_model, train_A_mean)  # (1, 16)
        global_token_order = phys_n16_block_order_to_model_token_order(
            global_n16_order, block_perm
        )  # (1, 256)
        print(f"Global ON order: {global_n16_order[0].tolist()}", flush=True)
    else:
        print("Using AR ascending order (control experiment)", flush=True)
        global_token_order = None  # use model.forward(mode='AR') instead

    # ── Optimizer ───────────────────────────────────────────────────────────
    optimizer = aogpt.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.lr,
        betas=(args.beta1, args.beta2),
        device_type="cuda" if device.type == "cuda" else "cpu",
    )

    # ── Training ────────────────────────────────────────────────────────────
    aogpt.train()
    train_losses = []
    best_val_ar_loss = float("inf")
    best_state = None
    best_iter = 0
    t0 = time.time()

    for iter_num in range(args.max_iters):
        # 1. Sample a batch of training sequences
        batch_idx = np.random.choice(train_indices, size=args.batch_size, replace=True)

        # 2. Train AO-GPT on each sequence
        batch_loss = 0.0
        for i in range(args.batch_size):
            idx = idx_all_dev[batch_idx[i]:batch_idx[i] + 1]  # (1, 256)
            if args.order_mode == "ar":
                _, loss = aogpt(idx, mode='AR')
            else:
                _, loss = aogpt.forward_fn(idx, global_token_order)
            batch_loss += loss.item()
            (loss / args.batch_size).backward()

        # 3. Gradient step
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(aogpt.parameters(), args.grad_clip)

        lr = get_lr(iter_num, args)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()
        optimizer.zero_grad()
        train_losses.append(batch_loss / args.batch_size)

        # 5. Logging
        if iter_num % args.log_interval == 0 or iter_num == 0:
            avg_loss = float(np.mean(train_losses[-args.log_interval:])) if iter_num > 0 else train_losses[-1]
            elapsed = time.time() - t0
            print(f"Iter {iter_num:5d}/{args.max_iters} | "
                  f"loss={train_losses[-1]:.4f} | avg={avg_loss:.4f} | "
                  f"lr={lr:.2e} | {elapsed:.0f}s", flush=True)

        # 6. Periodic evaluation
        if iter_num % args.eval_interval == 0 or iter_num == args.max_iters - 1:
            metrics = evaluate_aogpt(
                aogpt, idx_all_dev, global_token_order, args.order_mode,
                val_indices, args
            )
            mode_label = "AR" if args.order_mode == "ar" else "ON"
            print(f"  [Eval @{iter_num:5d}] AR_loss={metrics['ar_loss']:.4f} | "
                  f"Random_loss={metrics['random_loss']:.4f} | "
                  f"{mode_label}_loss={metrics['on_loss']:.4f}", flush=True)

            if metrics["ar_loss"] < best_val_ar_loss:
                best_val_ar_loss = metrics["ar_loss"]
                best_iter = iter_num
                best_state = {k: v.cpu().clone() for k, v in aogpt.state_dict().items()}
                print(f"  [*] New best AR loss at iter {iter_num}", flush=True)

    # ── Save ────────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\nTraining done. Best val AR loss: {best_val_ar_loss:.4f} at iter {best_iter}", flush=True)
    print(f"Total time: {elapsed:.0f}s ({elapsed / 60:.1f} min)", flush=True)

    output_path = Path(args.output_dir) / args.output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state if best_state else aogpt.state_dict(),
            "args": vars(args),
            "train_losses": train_losses,
            "best_val_ar_loss": best_val_ar_loss,
            "best_iter": best_iter,
        },
        output_path,
    )
    print(f"Saved: {output_path}", flush=True)


if __name__ == "__main__":
    main()
