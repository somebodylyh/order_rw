"""
Joint ON-AOGPT Training: Gumbel-Softmax + Soft Permutation (方案 A)

ON → Gumbel-Softmax sequential selection → soft permutation P (16x16)
→ expand P to token level (256x256) → soft-permute embeddings
→ AO-GPT transformer (frozen) → NLL loss
→ backprop through soft P → ON

Key trick: hard=True Gumbel-Softmax (straight-through) gives:
  - Forward: exact hard permutation → clean embeddings → standard NLL
  - Backward: softmax gradients → smoothed training signal for ON

Usage:
    python -u train_joint.py --device cuda:0 --epochs 100 --batch-size 1
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "AO-GPT-MDM"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import GPT2TokenizerFast

from config import Config
from order_network import CrossAttentionOrderNetwork, masks_to_revealed_bool
from model_AOGPT_AdaLN6_NoRep_cond_128_trunc_qknorm import AOGPT, AOGPTConfig

# ── Paths ────────────────────────────────────────────────────────────────────
CKPT_PATH = os.path.expanduser(
    "~/ych/nanogpt-learned-order/out/base/permute/seq256/block64/"
    "out-wikitext103-seq256-random-b64-permute-block-50000-iters/ckpt.pt"
)
WIKITEXT_ARROW = os.path.expanduser(
    "~/.cache/huggingface/datasets/wikitext/wikitext-103-raw-v1/0.0.0/"
    "b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-test.arrow"
)
TOKENIZER_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--gpt2/snapshots/"
    "607a30d783dfa663caf39e06633721c8d4cfcd7e"
)
ON_CKPT = "probe_results/crossattn_on_best.pt"
A_MATRICES = "probe_results/A_n16_direct_500x5.npy"
OUTPUT_DIR = "probe_results"

N = 16           # number of order blocks
BLOCK_LEN = 16   # tokens per order block (256 / 16)
SEQ_LEN = 256    # total tokens


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_sequences(min_len=SEQ_LEN, max_count=200):
    """Load token sequences from wikitext-103 test set."""
    ds = Dataset.from_file(WIKITEXT_ARROW)
    tok = GPT2TokenizerFast.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    seqs = []
    for ex in ds:
        tokens = tok.encode(ex["text"])
        if len(tokens) >= min_len:
            tokens = tokens[:min_len]
            seqs.append(tokens)
            if len(seqs) >= max_count:
                break
    return torch.tensor(seqs, dtype=torch.long)


def load_aogpt(ckpt_path, device):
    """Load frozen AO-GPT model and its block_perm / inv_perm."""
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
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    block_perm = torch.tensor(ckpt["data_permutation"]["block_perm"], dtype=torch.long)
    inv_perm = torch.tensor(ckpt["data_permutation"]["inverse_block_perm"], dtype=torch.long)
    return model, block_perm, inv_perm


def phys_to_model_idx(idx_phys, inv_perm):
    """Convert token sequences from physical order to model-coordinate order."""
    B, T = idx_phys.shape
    N64 = len(inv_perm)  # 64
    blk_size = T // N64  # 4 tokens per N64 block
    device = idx_phys.device
    idx_model = torch.zeros_like(idx_phys)
    for s in range(T):
        model_block = inv_perm[s // blk_size].item()
        offset = s % blk_size
        model_pos = model_block * blk_size + offset
        idx_model[:, model_pos] = idx_phys[:, s]
    return idx_model


# ── Token-Level Soft Permutation ──────────────────────────────────────────────

def build_token_permutation(P, block_len=BLOCK_LEN):
    """
    Expand block-level soft permutation P to token level.

    Args:
        P: (B, N, N) — P[b, t, i] = prob block i goes to new position t.
           One-hot in forward (hard=True GS), with straight-through grad_fn.
        block_len: tokens per block.

    Returns:
        P_token: (B, T, T) where T = N * block_len.
                 P_token[b, t*bl+k, i*bl+k] = P[b, t, i] for k in [0, bl).
    """
    B, Nblk, _ = P.shape
    T = Nblk * block_len
    device = P.device
    I_blk = torch.eye(block_len, device=device, dtype=P.dtype).view(1, 1, block_len, 1, block_len)
    P_exp = P.unsqueeze(2).unsqueeze(4)  # (B, N, 1, N, 1)
    return (P_exp * I_blk).reshape(B, T, T)


def permute_embeddings(emb, P_token):
    """Soft-permute embeddings: (B, T, D) → (B, T, D) via P_token @ emb."""
    return torch.bmm(P_token, emb)


# ── Core: Sequential Gumbel-Softmax Permutation ───────────────────────────────

def sequential_gumbel_permutation(on_model, A, tau, hard=True):
    """
    Run ON sequentially with Gumbel-Softmax to produce soft permutation P.

    Each step creates a fresh visited_bool from visited_mask to avoid in-place
    modification of tensors that are still referenced by the computation graph.

    Args:
        on_model: CrossAttentionOrderNetwork
        A: (B, N, N) frozen attention/NLL matrix
        tau: Gumbel-Softmax temperature
        hard: use straight-through (hard forward, soft backward)

    Returns:
        P: (B, N, N) soft permutation matrix with grad_fn attached
        choices: (B, N) integer chosen block indices at each step
    """
    B, Nblk, _ = A.shape
    device = A.device

    visited_mask = torch.zeros(B, dtype=torch.long, device=device)
    last_node = torch.zeros(B, dtype=torch.long, device=device)

    P_rows = []
    choices_list = []

    for step in range(Nblk):
        logits = on_model(A, visited_mask, last_node)  # (B, N)

        # Fresh visited_bool each step — no in-place reuse across steps
        visited_bool = masks_to_revealed_bool(visited_mask, Nblk)
        logits = logits.masked_fill(visited_bool, float("-inf"))

        # Gumbel-Softmax with straight-through
        p_t = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)  # (B, N)
        P_rows.append(p_t)

        chosen = p_t.argmax(dim=-1)  # (B,) integer
        choices_list.append(chosen)

        # Update hard state (integer ops, no grad)
        visited_mask = visited_mask | (1 << chosen)
        last_node = chosen

    P = torch.stack(P_rows, dim=1)  # (B, N, N): P[b, t, i] = prob block i at step t
    choices = torch.stack(choices_list, dim=1)  # (B, N)
    return P, choices


# ── Custom Soft Forward Through AO-GPT ────────────────────────────────────────

def soft_forward_aogpt(model, idx, P):
    """
    Forward AO-GPT with soft-permuted embeddings.

    Instead of using model.forward_fn with integer orders (non-differentiable),
    directly soft-permute token embeddings and position embeddings using P.

    Args:
        model: frozen AOGPT
        idx: (B, T) token IDs in MODEL coordinates
        P: (B, N, N) soft permutation with grad_fn (one-hot in forward)

    Returns:
        logits: (B, T+1, V)
    """
    B, T = idx.shape
    Nblk = P.shape[1]
    block_len = T // Nblk
    device = idx.device

    P_token = build_token_permutation(P, block_len)  # (B, T, T)

    # Token embeddings (soft-permuted)
    tok_emb_raw = model.transformer.wte(idx)  # (B, T, n_embd)
    tok_emb_perm = permute_embeddings(tok_emb_raw, P_token)

    # [None] token prepended
    none_emb = model.transformer.wnonee(
        torch.tensor([[0]], device=device)
    ).expand(B, -1, -1)  # (B, 1, n_embd)

    # Position embeddings
    pos = torch.arange(0, T + 1, dtype=torch.long, device=device)  # (T+1,)
    pos_emb_full = model.transformer.wpe(pos).unsqueeze(0).expand(B, -1, -1)  # (B, T+1, n_embd)
    pos_emb_prefix = pos_emb_full[:, :1, :]   # (B, 1, n_embd) — for [None]
    pos_emb_postfix = pos_emb_full[:, 1:, :]  # (B, T, n_embd)
    pos_emb_perm = permute_embeddings(pos_emb_postfix, P_token)

    # Target position embeddings (conditioning, soft-permuted)
    tgt_emb = model.transformer.wtpe(pos[:T]).unsqueeze(0).expand(B, -1, -1)  # (B, T, 128)
    tgt_emb_perm = permute_embeddings(tgt_emb, P_token)
    tgt_emb_postfix = torch.zeros(B, 1, 128, device=device)
    c = torch.cat([tgt_emb_perm, tgt_emb_postfix], dim=1)  # (B, T+1, 128)

    # Build transformer input
    x = torch.cat([none_emb, tok_emb_perm], dim=1)  # (B, T+1, n_embd)
    x = x + torch.cat([pos_emb_prefix, pos_emb_perm], dim=1)

    # Run transformer blocks
    x = model.transformer.drop(x)
    for block in model.transformer.h:
        x = block(x, c)
    x = model.transformer.final_layer(x, c)

    logits = model.lm_head(x)  # (B, T+1, V)
    return logits


def compute_gather_loss(logits, idx, P, block_len=BLOCK_LEN):
    """
    Compute NLL loss with soft-permuted targets using gather.

    Since P is one-hot in forward (hard=True GS), this is equivalent to
    standard CE. In backward, gradients flow through the soft version of P.

    Args:
        logits: (B, T+1, V)
        idx: (B, T) token IDs in model coordinates
        P: (B, N, N) soft permutation

    Returns:
        loss: scalar
    """
    B, T = idx.shape
    V = logits.shape[-1]

    shift_logits = logits[:, :-1, :].contiguous()  # (B, T, V)
    log_probs = F.log_softmax(shift_logits, dim=-1)  # (B, T, V)

    P_token = build_token_permutation(P, block_len)  # (B, T, T)

    # Gather log_probs at original token indices
    # lp_at_targets[b, j, i] = log_probs[b, j, idx[b, i]]
    idx_expanded = idx.unsqueeze(1).expand(-1, T, -1)  # (B, T, T)
    lp_at_targets = log_probs.gather(dim=2, index=idx_expanded)  # (B, T, T)

    loss = -(P_token * lp_at_targets).sum() / (B * T)
    return loss


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_on_accuracy(on_model, A, visited_masks, last_nodes, next_nodes):
    """Standard ON next-step accuracy (hard predictions)."""
    on_model.eval()
    A_dev = A.to(next_nodes.device)
    logits = on_model(
        A_dev,
        visited_masks.to(next_nodes.device),
        last_nodes.to(next_nodes.device),
    )
    pred = logits.argmax(dim=-1)
    acc = (pred == next_nodes.to(pred.device)).float().mean().item()
    return acc


@torch.no_grad()
def compute_aogpt_nll(model, idx, orders):
    """Standard AO-GPT NLL with hard integer orders (for monitoring)."""
    model.eval()
    _, loss = model.forward_fn(idx, orders)
    return loss.item()


# ── Temperature Schedule ──────────────────────────────────────────────────────

def get_temperature(epoch, tau_init=5.0, tau_min=0.5, decay_rate=0.95):
    """Exponential decay: tau = max(tau_min, tau_init * decay_rate^epoch)."""
    return max(tau_min, tau_init * (decay_rate ** epoch))


# ── Main Training Loop ────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Joint ON-AOGPT Training")
    parser.add_argument("--a-matrices", default=A_MATRICES)
    parser.add_argument("--on-ckpt", default=ON_CKPT)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--tau-init", type=float, default=5.0)
    parser.add_argument("--tau-min", type=float, default=0.5)
    parser.add_argument("--tau-decay", type=float, default=0.95)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--d-edge", type=int, default=0, help="0 = auto-detect from checkpoint")
    parser.add_argument("--d-model", type=int, default=0, help="0 = auto-detect from checkpoint")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--val-interval", type=int, default=5,
                        help="Run full validation every N epochs")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    Config.seed = args.seed
    Config.set_seed()
    device = torch.device(args.device)
    print(f"Device: {device}", flush=True)

    # ── Load AO-GPT (frozen) ──────────────────────────────────────────────
    print("Loading AO-GPT...", flush=True)
    aogpt, block_perm, inv_perm = load_aogpt(CKPT_PATH, device)
    print(f"AO-GPT loaded. Params: {sum(p.numel() for p in aogpt.parameters()):,}")

    # ── Load ON from checkpoint (auto-detect d_edge/d_model) ──────────────
    print("Loading Order Network...", flush=True)
    on_ckpt = torch.load(args.on_ckpt, map_location=device, weights_only=False)
    sd = on_ckpt["model_state_dict"]
    d_edge = sd["edge_mlp.0.weight"].shape[0]
    d_model = sd["score_mlp.0.weight"].shape[0]
    print(f"Detected d_edge={d_edge}, d_model={d_model}")
    on_model = CrossAttentionOrderNetwork(
        num_blocks=N, d_edge=d_edge, d_model=d_model
    )
    on_model.load_state_dict(sd)
    on_model.to(device)
    n_on_params = sum(p.numel() for p in on_model.parameters())
    print(f"ON loaded. Best val acc (pretrained): {on_ckpt['best_val_acc']:.4f}, "
          f"params: {n_on_params:,}")

    # ── Load data ─────────────────────────────────────────────────────────
    print("Loading data...", flush=True)
    A_all = np.load(args.a_matrices)  # (num_seqs, N, N)
    num_seqs = A_all.shape[0]

    idx_phys = load_sequences(min_len=SEQ_LEN, max_count=num_seqs)
    idx_phys = idx_phys[:num_seqs]
    print(f"Loaded {num_seqs} A matrices, {idx_phys.shape[0]} token sequences")

    # Convert token sequences to model coordinates
    idx_model_list = []
    for i in range(idx_phys.shape[0]):
        idx_model_list.append(
            phys_to_model_idx(idx_phys[i:i + 1], inv_perm)
        )
    idx_all = torch.cat(idx_model_list, dim=0)
    print(f"Token sequences converted to model coordinates: {idx_all.shape}")

    # Train/val split by sequence
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(num_seqs)
    val_size = max(1, int(round(num_seqs * args.val_fraction)))
    val_indices = set(perm[:val_size].tolist())
    train_indices = set(perm[val_size:].tolist())
    train_list = sorted(train_indices)
    val_list = sorted(val_indices)
    print(f"Train seqs: {len(train_list)}, Val seqs: {len(val_list)}")

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        on_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # ── Training ──────────────────────────────────────────────────────────
    history = []
    best_val_acc = 0.0
    best_epoch = 0
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        tau = get_temperature(epoch, args.tau_init, args.tau_min, args.tau_decay)

        # Shuffle training sequences
        epoch_train = list(train_list)
        np.random.shuffle(epoch_train)

        on_model.train()
        train_loss_sum = 0.0
        train_n_batches = 0

        for batch_start in range(0, len(epoch_train), args.batch_size):
            batch_seqs = epoch_train[batch_start:batch_start + args.batch_size]
            B = len(batch_seqs)

            A_batch = torch.as_tensor(A_all[batch_seqs], dtype=torch.float32, device=device)
            idx_batch = idx_all[batch_seqs].to(device)

            # 1. Sequential Gumbel-Softmax → soft permutation P
            P_soft, choices = sequential_gumbel_permutation(
                on_model, A_batch, tau, hard=True
            )

            # 2. Soft-permuted forward through AO-GPT
            logits = soft_forward_aogpt(aogpt, idx_batch, P_soft)

            # 3. Gather-based NLL loss
            loss = compute_gather_loss(logits, idx_batch, P_soft)

            # 4. Backprop
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(on_model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            train_n_batches += 1

        avg_train_loss = train_loss_sum / max(train_n_batches, 1)

        # ── Validation (periodic) ───────────────────────────────────────
        do_val = (epoch % args.val_interval == 0) or (epoch == 1)
        avg_val_nll = float("nan")
        if do_val:
            on_model.eval()
            val_nll_sum = 0.0
            val_count = 0

            for seq_idx in val_list:
                A_val = torch.as_tensor(
                    A_all[seq_idx:seq_idx + 1], dtype=torch.float32, device=device
                )
                idx_val = idx_all[seq_idx:seq_idx + 1].to(device)

                with torch.no_grad():
                    _, choices_val = sequential_gumbel_permutation(
                        on_model, A_val, tau=0.1, hard=True
                    )
                token_orders = block_perm_to_token_orders(choices_val, BLOCK_LEN, device)
                with torch.no_grad():
                    nll = compute_aogpt_nll(aogpt, idx_val, token_orders)
                val_nll_sum += nll
                val_count += 1

            avg_val_nll = val_nll_sum / max(val_count, 1)

        # Log
        if epoch % args.log_interval == 0 or epoch == 1:
            val_str = f"val_nll={avg_val_nll:.4f}" if do_val else "val_nll=—"
            print(
                f"Epoch {epoch:03d}/{args.epochs} | tau={tau:.3f} | "
                f"train_loss={avg_train_loss:.4f} | {val_str}",
                flush=True,
            )

        # Early stopping (on validation epochs only)
        if do_val:
            if epoch == 1 or avg_val_nll < best_val_acc:
                best_val_acc = avg_val_nll
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in on_model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

        history.append({
            "epoch": epoch,
            "tau": tau,
            "train_loss": avg_train_loss,
            "val_nll": avg_val_nll,
        })

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch} (patience={args.patience})")
            break

    # ── Save ──────────────────────────────────────────────────────────────
    print(f"\nBest val NLL: {best_val_acc:.4f} at epoch {best_epoch}")

    output_path = Path(args.output_dir) / "joint_on_best.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "args": vars(args),
            "history": history,
            "best_val_nll": best_val_acc,
            "best_epoch": best_epoch,
        },
        output_path,
    )
    print(f"Saved: {output_path}")


def block_perm_to_token_orders(choices, block_len, device):
    """Convert block-level choices (B, N) to token-level orders (B, T)."""
    B, Nblk = choices.shape
    T = Nblk * block_len
    orders = torch.zeros(B, T, dtype=torch.long, device=device)
    for t in range(Nblk):
        blk = choices[:, t]  # (B,)
        for k in range(block_len):
            orders[:, t * block_len + k] = blk * block_len + k
    return orders


if __name__ == "__main__":
    main()
