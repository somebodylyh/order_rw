"""
M-step: AO-GPT Continual Pretraining with HC Golden Orderings.

EM-style: E-step produces golden orderings via Hill Climbing, M-step fine-tunes AO-GPT.

Three-group comparison (各 5000 steps):
  1. Baseline: 100% random block permutations
  2. Golden only: 100% HC golden orderings
  3. Mixed: 50% golden + 50% random (alpha-mixing)

Core evaluation metric: val_l2r_loss (L2R ordering NLL on held-out sequences).
The goal is to see if training with better orderings improves L2R decoding.

Usage:
    # Quick test (100 seqs, 500 HC steps, 500 train steps)
    python -u m_step_train.py --hc-n-seqs 100 --hc-steps 500 --train-steps 500

    # Full run (500 seqs, 2000 HC steps, 5000 train steps)
    python -u m_step_train.py --hc-n-seqs 500 --hc-steps 2000 --train-steps 5000
"""

import argparse
import os
import sys
import time
import json
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
from model_AOGPT_AdaLN6_NoRep_cond_128_trunc_qknorm import AOGPT, AOGPTConfig

# ── Paths ──────────────────────────────────────────────────────────────────────
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

SEQ_LEN = 256
N_BLOCKS = 64
BLOCK_LEN = 4  # tokens per block


# ── Data Loading ────────────────────────────────────────────────────────────────

def load_wikitext_train(min_len=SEQ_LEN, max_count=None):
    """Load token sequences from wikitext-103 train set."""
    tok = GPT2TokenizerFast.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    tokens_all = []
    for shard in ["wikitext-train-00000-of-00002.arrow", "wikitext-train-00001-of-00002.arrow"]:
        arrow_file = os.path.join(WIKITEXT_DIR, shard)
        ds = Dataset.from_file(arrow_file)
        for ex in ds:
            raw = tok.encode(ex["text"])
            if len(raw) >= min_len:
                tokens_all.append(raw[:min_len])
                if max_count and len(tokens_all) >= max_count:
                    break
        if max_count and len(tokens_all) >= max_count:
            break
    return torch.tensor(tokens_all, dtype=torch.long)


# ── Model Loading ───────────────────────────────────────────────────────────────

def load_aogpt(ckpt_path, device, freeze=False):
    """Load AO-GPT from checkpoint. Returns (model, block_perm, inv_perm, opt_state)."""
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
    else:
        model.train()
    block_perm = torch.tensor(ckpt["data_permutation"]["block_perm"], dtype=torch.long)
    inv_perm = torch.tensor(ckpt["data_permutation"]["inverse_block_perm"], dtype=torch.long)
    opt_state = ckpt.get("optimizer", None)
    return model, block_perm, inv_perm, opt_state


# ── Coordinate Conversion ───────────────────────────────────────────────────────

def phys_to_model_idx(idx_phys, inv_perm):
    """Convert token sequences from physical order to model-coordinate order."""
    B, T = idx_phys.shape
    N64 = len(inv_perm)
    blk_size = T // N64
    device = idx_phys.device
    idx_model = torch.zeros_like(idx_phys)
    for s in range(T):
        model_block = inv_perm[s // blk_size].item()
        offset = s % blk_size
        model_pos = model_block * blk_size + offset
        idx_model[:, model_pos] = idx_phys[:, s]
    return idx_model


# ── Order Utilities ─────────────────────────────────────────────────────────────

def block_order_to_token_order(block_order, block_len=BLOCK_LEN):
    """(N,) block indices → (N*block_len,) token indices."""
    N = len(block_order)
    token_order = np.zeros(N * block_len, dtype=np.int64)
    for i, blk in enumerate(block_order):
        for k in range(block_len):
            token_order[i * block_len + k] = blk * block_len + k
    return token_order


def random_block_orders(batch_size, n_blocks=N_BLOCKS, block_len=BLOCK_LEN, device="cpu"):
    """Generate random block-level permutations for a batch."""
    orders = torch.zeros(batch_size, n_blocks * block_len, dtype=torch.long, device=device)
    for b in range(batch_size):
        perm = torch.randperm(n_blocks, device=device)
        token_order = torch.repeat_interleave(perm, block_len) * block_len
        offsets = torch.arange(block_len, device=device).repeat(n_blocks)
        orders[b] = token_order + offsets
    return orders


def l2r_orders(batch_size, n_blocks=N_BLOCKS, block_len=BLOCK_LEN, device="cpu"):
    """L2R token orders: [0, 1, 2, ..., T-1]."""
    T = n_blocks * block_len
    return torch.arange(T, device=device).unsqueeze(0).expand(batch_size, -1)


# ── AO-GPT NLL Computation ─────────────────────────────────────────────────────

@torch.no_grad()
def compute_per_seq_nll(model, idx, orders):
    """
    Per-sequence NLL. Replicates model.forward_fn internals but returns (B,) NLL.

    Unlike model.forward_fn which returns a scalar (mean) loss over all tokens,
    this computes CE per sequence.
    """
    B, T = idx.shape
    device = idx.device
    pos = torch.arange(0, T + 1, dtype=torch.long, device=device)

    batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, T)

    # Token embeddings (shuffled)
    tok_emb = model.transformer.wte(idx)
    tok_emb = tok_emb[batch_indices, orders]
    none_emb = model.transformer.wnonee(
        torch.tensor([[0]], device=device)
    ).expand(B, -1, -1)
    tok_emb = torch.cat([none_emb, tok_emb], dim=1)

    # Position embeddings
    pos_emb = model.transformer.wpe(pos).unsqueeze(0).expand(B, -1, -1)
    pos_emb_prefix = pos_emb[:, :1, :]
    pos_emb_postfix = pos_emb[:, 1:, :][batch_indices, orders]
    pos_emb_final = torch.cat([pos_emb_prefix, pos_emb_postfix], dim=1)

    # Target position embeddings
    tgt_emb = model.transformer.wtpe(pos[:T]).unsqueeze(0).expand(B, -1, -1)
    tgt_emb_prefix = tgt_emb[batch_indices, orders]
    tgt_emb_postfix = torch.zeros(B, 1, tgt_emb.shape[-1], device=device)
    c = torch.cat([tgt_emb_prefix, tgt_emb_postfix], dim=1)

    # Targets (shuffled tokens)
    targets = idx[batch_indices, orders]

    x = tok_emb + pos_emb_final
    x = model.transformer.drop(x)
    for block in model.transformer.h:
        x = block(x, c)
    x = model.transformer.final_layer(x, c)

    logits = model.lm_head(x)[:, :-1, :].contiguous()
    ce_per_token = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
        ignore_index=-1,
    ).reshape(B, T)
    return ce_per_token.mean(dim=1)


@torch.no_grad()
def compute_val_l2r_loss(model, idx_val, batch_size=8):
    """Compute val L2R NLL over a set of sequences."""
    model.eval()
    model_device = next(model.parameters()).device
    total_loss = 0.0
    n = len(idx_val)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        idx_b = idx_val[start:end].to(model_device)
        B = idx_b.shape[0]
        orders = l2r_orders(B, device=model_device)
        nll = compute_per_seq_nll(model, idx_b, orders)
        total_loss += nll.sum().item()
    model.train()
    return total_loss / n


# ── E-step: Hill Climbing ──────────────────────────────────────────────────────

@torch.no_grad()
def run_hill_climb(model, idx_all, n_steps, batch_size, device, verbose=True):
    """
    Batched random-swap hill climbing.
    Returns (best_orders, best_nlls, history).
    """
    num_seqs, T = idx_all.shape
    N = T // BLOCK_LEN

    l2r_blocks = np.arange(N, dtype=np.int64)
    l2r_tokens = block_order_to_token_order(l2r_blocks)

    current_blocks = np.tile(l2r_blocks[None, :], (num_seqs, 1))
    candidate_blocks = np.copy(current_blocks)

    # Baseline NLL
    if verbose:
        print("  Computing L2R baseline NLL...", flush=True)
    current_nlls = np.zeros(num_seqs, dtype=np.float32)
    for b_start in range(0, num_seqs, batch_size):
        b_end = min(b_start + batch_size, num_seqs)
        B = b_end - b_start
        idx_b = idx_all[b_start:b_end].to(model.transformer.wte.weight.device)
        orders_b = torch.as_tensor(
            np.tile(l2r_tokens[None, :], (B, 1)), dtype=torch.long,
            device=idx_b.device
        )
        nll_b = compute_per_seq_nll(model, idx_b, orders_b)
        current_nlls[b_start:b_end] = nll_b.cpu().numpy()

    best_blocks = np.copy(current_blocks)
    best_nlls = np.copy(current_nlls)
    if verbose:
        print(f"  Baseline L2R NLL: mean={current_nlls.mean():.4f}, "
              f"min={current_nlls.min():.4f}, max={current_nlls.max():.4f}", flush=True)

    history = []
    t_start = time.perf_counter()
    model_device = model.transformer.wte.weight.device

    for step in range(1, n_steps + 1):
        i_vals = np.random.randint(0, N, size=num_seqs)
        j_vals = np.random.randint(0, N, size=num_seqs)
        for s in range(num_seqs):
            while j_vals[s] == i_vals[s]:
                j_vals[s] = np.random.randint(0, N)

        candidate_blocks[:] = current_blocks
        for s in range(num_seqs):
            candidate_blocks[s, i_vals[s]], candidate_blocks[s, j_vals[s]] = \
                candidate_blocks[s, j_vals[s]], candidate_blocks[s, i_vals[s]].copy()

        candidate_tokens = np.zeros((num_seqs, T), dtype=np.int64)
        for s in range(num_seqs):
            candidate_tokens[s] = block_order_to_token_order(candidate_blocks[s])

        candidate_nlls = np.zeros(num_seqs, dtype=np.float32)
        for b_start in range(0, num_seqs, batch_size):
            b_end = min(b_start + batch_size, num_seqs)
            idx_b = idx_all[b_start:b_end].to(model_device)
            orders_b = torch.as_tensor(
                candidate_tokens[b_start:b_end], dtype=torch.long, device=model_device
            )
            nll_b = compute_per_seq_nll(model, idx_b, orders_b)
            candidate_nlls[b_start:b_end] = nll_b.cpu().numpy()

        improved = candidate_nlls < current_nlls
        current_blocks[improved] = candidate_blocks[improved].copy()
        current_nlls[improved] = candidate_nlls[improved]

        new_best = current_nlls < best_nlls
        best_blocks[new_best] = current_blocks[new_best].copy()
        best_nlls[new_best] = current_nlls[new_best].copy()

        if step % 200 == 0 or step == 1:
            elapsed = time.perf_counter() - t_start
            n_accepted = improved.sum()
            history.append((step, int(n_accepted), float(best_nlls.mean())))
            if verbose:
                print(f"  Step {step:5d}/{n_steps} | accepted={n_accepted:4d}/{num_seqs} "
                      f"| best_nll={best_nlls.mean():.4f} | {elapsed:.1f}s", flush=True)

    elapsed = time.perf_counter() - t_start
    if verbose:
        improvement = 1 - best_nlls.mean() / (current_nlls.mean() if current_nlls.mean() > 0 else best_nlls.mean())
        print(f"  HC done. {n_steps} steps, {elapsed:.1f}s, "
              f"best NLL mean={best_nlls.mean():.4f} ({improvement*100:.1f}% improvement)", flush=True)

    best_token_orders = np.zeros((num_seqs, T), dtype=np.int64)
    for s in range(num_seqs):
        best_token_orders[s] = block_order_to_token_order(best_blocks[s])

    return best_token_orders, best_nlls, history


# ── M-step: Continual Pretraining ───────────────────────────────────────────────

def sample_soft_orders(batch_idx, best_blocks_t, rigidity_t, tau, n_blocks, block_len, device):
    """PL-style soft sampling: rigidity-biased Gaussian perturbation of positions.

    Each position k gets noise ~ N(0, tau / rigidity[k]), then blocks are re-sorted
    by their perturbed positions. τ→0 = deterministic best, τ→∞ = random shuffle.
    """
    B = len(batch_idx)
    best = best_blocks_t[batch_idx]           # (B, N) block IDs per position
    rig = rigidity_t[batch_idx].clamp(min=1e-8)  # (B, N) avoid div by zero
    positions = torch.arange(n_blocks, dtype=torch.float32, device=device).unsqueeze(0).expand(B, -1)
    noise = torch.randn(B, n_blocks, device=device) * tau / rig
    perturbed = positions + noise
    _, sort_idx = perturbed.sort(dim=1)
    new_blocks = best.gather(1, sort_idx)     # (B, N)
    token_orders = torch.repeat_interleave(new_blocks, block_len, dim=1) * block_len
    offsets = torch.arange(block_len, device=device).repeat(n_blocks)
    return token_orders + offsets


def train_group(model, group_name, idx_train, golden_orders, val_l2r_fn,
                train_steps=5000, batch_size=16, lr=3e-5, warmup_steps=500,
                max_epochs=3, weight_decay=0.1, grad_clip=1.0,
                log_interval=100, val_interval=250, mixed_ratio=0.5,
                opt_state=None, device="cuda",
                soft_tau=None, rigidity=None, best_blocks=None):
    """
    Train one group with specified ordering strategy.

    Args:
        model: AO-GPT in train mode
        group_name: 'baseline', 'golden', 'mixed', 'soft'
        idx_train: (num_train_seqs, 256) token IDs in model coordinates
        golden_orders: (n_golden_seqs, 256) golden token orders from HC.
        train_steps: requested optimizer steps (capped by max_epochs)
        soft_tau: if set, use PL soft sampling with this temperature (for 'soft' group)
        rigidity: (n_golden_seqs, N_BLOCKS) rigidity from adjacent probe
        best_blocks: (n_golden_seqs, N_BLOCKS) best block orders
    """
    num_seqs = len(idx_train)
    n_golden = len(golden_orders)
    model_device = next(model.parameters()).device

    # Hard cap: max_epochs * (num_seqs / batch_size)
    steps_per_epoch = max(1, num_seqs // batch_size)
    max_steps = max_epochs * steps_per_epoch
    effective_steps = min(train_steps, max_steps)
    if effective_steps < train_steps:
        print(f"  train_steps={train_steps} capped to {effective_steps} "
              f"({max_epochs} epochs × {steps_per_epoch} steps/epoch)", flush=True)

    # Build optimizer, optionally loading pretraining state
    optimizer = model.configure_optimizers(weight_decay, lr, (0.9, 0.99), device)
    if opt_state is not None:
        try:
            # Map old param names (may have _orig_mod. prefix)
            state_dict = opt_state if isinstance(opt_state, dict) else opt_state
            if "state" in state_dict and "param_groups" in state_dict:
                mapped_state = {}
                for k, v in state_dict["state"].items():
                    mapped_state[k] = v
                # Try loading; fall back to fresh if param names differ
                optimizer.load_state_dict(state_dict)
                print(f"  Loaded pretraining optimizer state", flush=True)
            else:
                print(f"  opt_state has unexpected format, using fresh optimizer", flush=True)
        except Exception as e:
            print(f"  Could not load opt_state ({e}), using fresh optimizer", flush=True)

    # LR schedule: linear warmup → cosine decay
    def get_lr(step):
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        if step >= effective_steps:
            return lr * 0.1
        progress = (step - warmup_steps) / max(1, effective_steps - warmup_steps)
        return lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress)))

    golden_t = torch.as_tensor(golden_orders, dtype=torch.long, device=model_device)
    idx_t = idx_train.to(model_device)

    # Precompute soft-sampling tensors if needed
    best_blocks_t = None
    rigidity_t = None
    if group_name == "soft" and best_blocks is not None and rigidity is not None:
        best_blocks_t = torch.as_tensor(best_blocks, dtype=torch.long, device=model_device)
        rigidity_t = torch.as_tensor(rigidity, dtype=torch.float32, device=model_device)
        n_soft = best_blocks_t.shape[0]
        print(f"  Soft PL: tau={soft_tau}, n_soft={n_soft}, "
              f"rigidity range=[{rigidity_t.min():.6f}, {rigidity_t.max():.6f}]", flush=True)

    log = []
    best_val = float("inf")
    best_state = None
    loss_ema = None

    print(f"\n{'='*60}")
    print(f"Group: {group_name} | {effective_steps} steps | lr={lr} | batch_size={batch_size}")
    print(f"Warmup: {warmup_steps} steps | max_epochs={max_epochs} | golden seqs={n_golden}/{num_seqs}")
    if soft_tau is not None:
        print(f"soft_tau: {soft_tau}")
    print(f"{'='*60}", flush=True)

    for step in range(1, effective_steps + 1):
        current_lr = get_lr(step - 1)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        batch_idx = torch.randint(0, num_seqs, (batch_size,), device=model_device)

        orders = random_block_orders(batch_size, device=model_device)
        if group_name == "golden":
            has_golden = batch_idx < n_golden
            if has_golden.any():
                orders[has_golden] = golden_t[batch_idx[has_golden]]
        elif group_name == "mixed":
            use_golden = (torch.rand(batch_size, device=model_device) < mixed_ratio) & (batch_idx < n_golden)
            orders[use_golden] = golden_t[batch_idx[use_golden]]
        elif group_name == "soft":
            has_soft = batch_idx < n_golden
            if has_soft.any():
                soft_idx = batch_idx[has_soft]
                orders[has_soft] = sample_soft_orders(
                    soft_idx, best_blocks_t, rigidity_t, soft_tau,
                    N_BLOCKS, BLOCK_LEN, model_device
                )
        elif group_name == "baseline":
            pass  # already random
        else:
            raise ValueError(f"Unknown group: {group_name}")

        _, loss = model.forward_fn(idx_t[batch_idx], orders)

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        loss_val = loss.item()
        loss_ema = loss_val if loss_ema is None else 0.95 * loss_ema + 0.05 * loss_val

        if step % log_interval == 0 or step == 1:
            print(f"  Step {step:5d}/{effective_steps} | loss={loss_val:.4f} | "
                  f"ema={loss_ema:.4f} | lr={current_lr:.2e}", flush=True)

        if step % val_interval == 0 or step == 1:
            val_l2r = val_l2r_fn()
            log.append({"step": step, "train_loss": loss_ema, "val_l2r_loss": val_l2r})
            improved = "*" if val_l2r < best_val else ""
            print(f"  >>> Step {step:5d} | val_l2r_loss={val_l2r:.4f} "
                  f"(best={min(val_l2r, best_val):.4f}) {improved}", flush=True)
            if val_l2r < best_val:
                best_val = val_l2r
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    val_l2r = val_l2r_fn()
    log.append({"step": effective_steps, "train_loss": loss_ema, "val_l2r_loss": val_l2r})

    if best_state is None:
        best_val = val_l2r
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"  Final val_l2r_loss={best_val:.4f}", flush=True)
    return best_state, best_val, log


# ── Main ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="M-step: HC golden → AO-GPT continual pretraining")
    # E-step (HC)
    p.add_argument("--hc-n-seqs", type=int, default=1000,
                   help="Number of train sequences for HC (subset of train)")
    p.add_argument("--hc-steps", type=int, default=2000,
                   help="HC steps per sequence")
    p.add_argument("--hc-batch-size", type=int, default=20,
                   help="HC evaluation batch size")
    # M-step (training)
    p.add_argument("--train-n-seqs", type=int, default=5000,
                   help="Total training sequences (can be > hc-n-seqs; extra seqs use random orders)")
    p.add_argument("--train-steps", type=int, default=5000,
                   help="Training steps per group (capped by --max-epochs)")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Training batch size")
    p.add_argument("--lr", type=float, default=3e-5,
                   help="Learning rate for continual pretraining (1/10 of pretrain lr)")
    p.add_argument("--warmup-steps", type=int, default=500,
                   help="Linear LR warmup steps")
    p.add_argument("--max-epochs", type=int, default=3,
                   help="Hard cap: max epochs over training data")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--mixed-ratio", type=float, default=0.5,
                   help="Golden ratio in mixed group")
    # Eval
    p.add_argument("--val-n-seqs", type=int, default=200,
                   help="Number of held-out sequences for val_l2r")
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--val-interval", type=int, default=250)
    # Groups to run
    p.add_argument("--groups", nargs="+", default=["baseline", "golden", "mixed"],
                   choices=["baseline", "golden", "mixed", "soft"])
    # Soft PL sampling
    p.add_argument("--soft", action="store_true",
                   help="Enable soft PL sampling for 'soft' group")
    p.add_argument("--soft-tau", type=float, default=0.01,
                   help="Temperature for soft PL sampling (default 0.01)")
    p.add_argument("--rigidity-file", default="",
                   help="Path to adjacent_rigidity.npz (for soft sampling)")
    # Output
    p.add_argument("--output-dir", default="probe_results/m_step")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    # Resume
    p.add_argument("--hc-results", default="",
                   help="Path to pre-computed HC .npz (skip E-step)")
    return p.parse_args()


def main():
    args = parse_args()
    Config.seed = args.seed
    Config.set_seed()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Output: {output_dir}")

    # ── Load data ─────────────────────────────────────────────────────────
    print("\nLoading training data...", flush=True)
    total_needed = args.train_n_seqs + args.val_n_seqs
    idx_phys = load_wikitext_train(min_len=SEQ_LEN, max_count=total_needed)
    n_loaded = idx_phys.shape[0]
    print(f"Loaded {n_loaded} sequences (requested {total_needed})", flush=True)
    if n_loaded < total_needed:
        print(f"WARNING: only {n_loaded} sequences available, adjusting split")
        ratio = args.train_n_seqs / (args.train_n_seqs + args.val_n_seqs)
        args.train_n_seqs = max(args.hc_n_seqs, int(n_loaded * ratio))
        args.val_n_seqs = n_loaded - args.train_n_seqs

    # ── Load initial model (for coordinate conversion and HC) ─────────────
    print("\nLoading AO-GPT for coordinate conversion...", flush=True)
    ref_model, block_perm, inv_perm, _ = load_aogpt(ckpt_path=AO_GPT_CKPT, device=device, freeze=True)
    n_params = sum(p.numel() for p in ref_model.parameters())
    print(f"Model params: {n_params:,}")

    # Convert to model coordinates
    print("Converting to model coordinates...", flush=True)
    idx_list = [phys_to_model_idx(idx_phys[i:i + 1], inv_perm) for i in range(n_loaded)]
    idx_all = torch.cat(idx_list, dim=0)
    print(f"idx_all: {idx_all.shape}")

    # ── Split train / val ─────────────────────────────────────────────────
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n_loaded)
    train_idx = perm[:args.train_n_seqs]
    val_idx = perm[args.train_n_seqs:args.train_n_seqs + args.val_n_seqs]

    # HC runs on a subset; M-step training uses all training seqs
    hc_indices = train_idx[:args.hc_n_seqs]
    idx_train_hc = idx_all[hc_indices]     # only for HC
    idx_train = idx_all[train_idx]         # full training set for M-step
    idx_val = idx_all[val_idx]              # for val_l2r loss

    print(f"Train seqs: {len(train_idx)} (HC on first {args.hc_n_seqs}), Val seqs: {len(val_idx)}")

    # ── E-step: Hill Climbing ─────────────────────────────────────────────
    if args.hc_results:
        print(f"\nLoading pre-computed HC results: {args.hc_results}", flush=True)
        hc_data = np.load(args.hc_results, allow_pickle=True)
        golden_orders = hc_data["best_orders"]
        hc_best_nlls = hc_data["best_nlls"]
        print(f"Loaded: golden_orders {golden_orders.shape}, "
              f"NLL mean={hc_best_nlls.mean():.4f}")
    else:
        print(f"\n{'='*60}")
        print(f"E-step: Hill Climbing on {args.hc_n_seqs} train seqs × {args.hc_steps} steps")
        print(f"{'='*60}", flush=True)

        golden_orders, hc_best_nlls, hc_history = run_hill_climb(
            ref_model, idx_train_hc, args.hc_steps, args.hc_batch_size, device, verbose=True
        )

        hc_path = output_dir / "hc_golden_orders.npz"
        np.savez_compressed(
            hc_path,
            best_orders=golden_orders,
            best_nlls=hc_best_nlls,
            history=np.array(hc_history),
            train_indices=train_idx,
        )
        print(f"Saved HC results: {hc_path}")

    # Free ref_model memory
    del ref_model
    torch.cuda.empty_cache()

    # ── Soft PL sampling: load rigidity ────────────────────────────────────
    soft_rigidity = None
    soft_best_blocks = None
    if "soft" in args.groups:
        rigidity_file = args.rigidity_file
        if not rigidity_file:
            rigidity_file = output_dir / "adjacent_rigidity.npz"
        if os.path.exists(rigidity_file):
            rig_data = np.load(rigidity_file, allow_pickle=True)
            soft_rigidity = rig_data["rigidity"]          # (n_probe, N_BLOCKS)
            soft_best_blocks = rig_data["best_blocks"]     # (n_probe, N_BLOCKS)
            print(f"\nLoaded rigidity: {soft_rigidity.shape}, "
                  f"mean={soft_rigidity.mean():.6f}, best_blocks: {soft_best_blocks.shape}")
        else:
            print(f"\nWARNING: rigidity file not found: {rigidity_file}")
            print("  Run probe_adjacent.py first, then use --rigidity-file")

    # ── M-step: Three-group training ──────────────────────────────────────
    def make_val_l2r_fn(model):
        def fn():
            return compute_val_l2r_loss(model, idx_val, batch_size=8)
        return fn

    results = {}
    original_val_l2r = None  # computed once with fresh model

    for group_name in args.groups:
        print(f"\n{'='*60}")
        print(f"M-step: Group = '{group_name}'")
        print(f"{'='*60}", flush=True)

        # Fresh model for each group
        model, _, _, opt_state = load_aogpt(ckpt_path=AO_GPT_CKPT, device=device, freeze=False)
        model.train()

        # Compute original val_l2r once
        if original_val_l2r is None:
            model.eval()
            original_val_l2r = compute_val_l2r_loss(model, idx_val)
            model.train()
            print(f"Original val_l2r_loss (before any training): {original_val_l2r:.4f}")

        best_state, best_val, log = train_group(
            model=model,
            group_name=group_name,
            idx_train=idx_train,
            golden_orders=golden_orders,
            val_l2r_fn=make_val_l2r_fn(model),
            train_steps=args.train_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            warmup_steps=args.warmup_steps,
            max_epochs=args.max_epochs,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            log_interval=args.log_interval,
            val_interval=args.val_interval,
            mixed_ratio=args.mixed_ratio,
            opt_state=opt_state,
            device=device,
            soft_tau=args.soft_tau if group_name == "soft" else None,
            rigidity=soft_rigidity,
            best_blocks=soft_best_blocks,
        )

        # Save checkpoint
        ckpt_path = output_dir / f"m_step_{group_name}.pt"
        torch.save({
            "model_state_dict": best_state,
            "best_val_l2r_loss": best_val,
            "original_val_l2r_loss": original_val_l2r,
            "train_log": log,
            "group": group_name,
            "args": vars(args),
        }, ckpt_path)
        print(f"Saved: {ckpt_path}")

        results[group_name] = {
            "best_val_l2r_loss": best_val,
            "original_val_l2r_loss": original_val_l2r,
            "delta": best_val - original_val_l2r,
        }

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("M-step Results Summary")
    print(f"{'='*60}")
    print(f"{'Group':<12} {'val_l2r_loss':>13} {'Δ vs original':>14}")
    print("-" * 39)
    for group_name in args.groups:
        r = results[group_name]
        sign = "+" if r["delta"] > 0 else ""
        print(f"{group_name:<12} {r['best_val_l2r_loss']:>13.4f} {sign}{r['delta']:>13.4f}")

    summary_path = output_dir / "m_step_summary.json"
    summary_path.write_text(json.dumps({
        "original_val_l2r_loss": original_val_l2r,
        "results": results,
        "args": vars(args),
        "hc_n_seqs": args.hc_n_seqs,
        "val_n_seqs": args.val_n_seqs,
    }, indent=2))
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
