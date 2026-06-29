"""
ON-mixed AO-GPT continual training.

Per-sample alpha-mixed orders: ON-derived (or edge-greedy) vs random.
Aligns with ych's segment_guided_ratio pattern — per-sample mixing within each batch.

Usage:
    # Random baseline (alpha=0)
    python train_on_mixed.py --alpha 0.0 --device cuda:0

    # Edge-greedy mixed (no ON training needed, quick smoke test)
    python train_on_mixed.py --alpha 0.5 --order-source edge_greedy --device cuda:0

    # ON32 mixed (after ON32 is trained)
    python train_on_mixed.py --alpha 0.5 --order-source on32 --on-ckpt probe_results/on32_best.pt --device cuda:0
"""

import os, sys, time, math, argparse, json
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ── Paths ──
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "AO-GPT-MDM"))
sys.path.insert(0, os.path.expanduser("~/ych/nanogpt-learned-order"))

from AOGPT import AOGPTConfig, AOGPT
from order_utils import (
    expand_block_orders_to_token_orders,
    invert_permutation,
    block_permutation_to_token_permutation,
    build_fixed_block_permutation,
    token_losses_to_block_losses,
)
from order_network import CrossAttentionOrderNetwork

# ── Config ──
AO_GPT_CKPT_DEFAULT = os.path.expanduser(
    "~/ych/nanogpt-learned-order/out/base/permute/seq256/block64/"
    "out-wikitext103-seq256-random-b64-permute-block-50000-iters/ckpt.pt"
)
A32_PATH = "probe_results/A_train_n32_10k.npy"
TOKENS_PATH = "probe_results/A_train_n32_10k.tokens.npy"
OUT_DIR = "probe_results/on_mixed_training"

NUM_BLOCKS = 32
TOKENS_PER_BLOCK = 8  # 256 / 32
SEQ_LEN = 256


# ── Edge-greedy order source (no ON needed) ──

def sample_edge_greedy_order(A_batch: torch.Tensor) -> torch.Tensor:
    """Greedy: at each step, pick candidate with max A[candidate, last].

    A_batch: (B, 32, 32) in model coordinates.
    Returns: (B, 32) block orders.
    """
    B, N, _ = A_batch.shape
    device = A_batch.device
    orders = torch.zeros(B, N, dtype=torch.long, device=device)
    visited = torch.zeros(B, dtype=torch.long, device=device)

    for t in range(N):
        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        for b in range(B):
            for i in range(N):
                if visited[b] & (1 << i):
                    mask[b, i] = True

        if t == 0:
            # First block: max mean row attention (no last node)
            scores = A_batch.mean(dim=2)  # (B, N)
        else:
            last = orders[:, t - 1]  # (B,)
            scores = A_batch[torch.arange(B), :, last]  # (B, N) — A[cand, last]

        scores = scores.masked_fill(mask, float('-inf'))
        chosen = scores.argmax(dim=1)  # (B,)
        orders[:, t] = chosen
        for b in range(B):
            visited[b] |= (1 << chosen[b].item())

    return orders


# ── ON order source (generic, works for any num_blocks) ──

def load_on(ckpt_path, device, num_blocks=None):
    """Load frozen ON from checkpoint. Auto-detects d_edge, d_model."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]
    d_edge = sd["edge_mlp.0.weight"].shape[0]
    d_model = sd["score_mlp.0.weight"].shape[0]
    if num_blocks is None:
        # Infer from state dict: edge_mlp.0.weight has in_features = d_edge + 64 + ...
        num_blocks = NUM_BLOCKS
    on = CrossAttentionOrderNetwork(num_blocks=num_blocks, d_edge=d_edge, d_model=d_model).to(device)
    on.load_state_dict(sd)
    on.eval()
    for p in on.parameters():
        p.requires_grad = False
    return on


@torch.no_grad()
def sample_on_order(on_model, A_batch, temperature=1.0):
    """Autoregressive sample from ON (any num_blocks).

    A_batch: (B, N, N)
    Returns: (B, N) block orders.
    """
    B, N, _ = A_batch.shape
    device = A_batch.device
    visited = torch.zeros(B, dtype=torch.long, device=device)
    last = torch.zeros(B, dtype=torch.long, device=device)
    orders = torch.zeros(B, N, dtype=torch.long, device=device)

    for t in range(N):
        logits = on_model(A_batch, visited, last)
        if temperature < 1e-8:
            chosen = logits.argmax(dim=-1)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            chosen = torch.multinomial(probs, 1).squeeze(-1)
        orders[:, t] = chosen
        visited |= (1 << chosen)
        last = chosen

    return orders


# ── Data ──

class AlignedDataset(torch.utils.data.Dataset):
    """Dataset where tokens[i] and A32[i] are aligned by index."""

    def __init__(self, tokens_path, a32_path, split='train', train_ratio=0.85):
        self.tokens = np.load(tokens_path, mmap_mode='r')  # (n, 256) int32
        self.A32 = np.load(a32_path, mmap_mode='r')  # (n, 32, 32) float32
        n = len(self.A32)
        assert len(self.tokens) == n, f"Tokens {len(self.tokens)} != A32 {n}"
        split_idx = int(n * train_ratio)
        self.indices = range(split_idx) if split == 'train' else range(split_idx, n)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        tokens = torch.from_numpy(self.tokens[i].astype(np.int64))
        A = torch.from_numpy(self.A32[i].copy()).float()
        return tokens, A, i


class TokenDataset(torch.utils.data.Dataset):
    """Dataset for full-token training when orders do not require A32."""

    def __init__(self, tokens_path, split='train', train_ratio=0.85):
        self.tokens = np.load(tokens_path, mmap_mode='r')  # (n, 256) int32
        n = len(self.tokens)
        split_idx = int(n * train_ratio)
        self.indices = range(split_idx) if split == 'train' else range(split_idx, n)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        tokens = torch.from_numpy(self.tokens[i].astype(np.int64))
        return tokens, i


# ── Training ──

def get_alpha(step, alpha_target, alpha_warmup_steps):
    """Linear ramp: 0 → alpha_target over alpha_warmup_steps."""
    if alpha_target <= 0 or alpha_warmup_steps <= 0:
        return alpha_target
    return alpha_target * min(1.0, step / alpha_warmup_steps)


def get_checkpoint_metric(val_losses, save_metric):
    if save_metric not in val_losses:
        raise KeyError(f"Unknown save metric {save_metric!r}; available={sorted(val_losses)}")
    return val_losses[save_metric]


def sample_mixed_block_orders(A_batch, alpha, order_source, on_model, temperature, device,
                              precomputed_orders=None, sample_indices=None):
    """Per-sample mixed block orders.

    A_batch: (B, 32, 32) in MODEL coordinates.
    alpha: probability of using guided order (per-sample).
    Returns: (B, 32) block orders in model coordinates.
    """
    if A_batch is not None:
        B = A_batch.shape[0]
    elif sample_indices is not None:
        B = len(sample_indices)
    else:
        raise ValueError("A_batch or sample_indices is required")
    orders = []
    for b in range(B):
        if np.random.random() < alpha:
            if order_source == 'edge_greedy':
                ord_b = sample_edge_greedy_order(A_batch[b:b + 1])
            elif order_source in ('on32', 'on'):
                ord_b = sample_on_order(on_model, A_batch[b:b + 1], temperature)
            elif order_source == 'precomputed':
                if precomputed_orders is None or sample_indices is None:
                    raise ValueError("precomputed order source requires orders and sample indices")
                order_idx = int(sample_indices[b]) % len(precomputed_orders)
                ord_b = precomputed_orders[order_idx:order_idx + 1].to(device)
            else:
                ord_b = torch.randperm(NUM_BLOCKS, device=device).unsqueeze(0)
        else:
            ord_b = torch.randperm(NUM_BLOCKS, device=device).unsqueeze(0)
        orders.append(ord_b.squeeze(0))
    return torch.stack(orders, dim=0).to(device)


@torch.no_grad()
def estimate_losses(model, dataset, ctx, eval_iters, order_source, on_model,
                    alpha, temperature, device, block_perm, inv_perm, effective_block_len,
                    precomputed_orders=None):
    """Estimate losses: training mode, AR mode, random mode, ON mode."""
    model.eval()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    losses = {'train_mode': [], 'ar': [], 'random': [], 'l2r': [], 'on_order': []}
    for k, (tokens, A, sample_indices) in enumerate(dataloader):
        if k >= eval_iters:
            break
        tokens = tokens.to(device)
        A = A.to(device)
        B = tokens.shape[0]

        # Training mode loss (mixed)
        block_orders = sample_mixed_block_orders(
            A, alpha, order_source, on_model, temperature, device,
            precomputed_orders=precomputed_orders,
            sample_indices=sample_indices,
        )
        token_orders = expand_block_orders_to_token_orders(block_orders, block_len=effective_block_len)
        with ctx:
            _, loss = model(tokens, mode=None, orders=token_orders)
        losses['train_mode'].append(loss.item())

        # AR loss
        with ctx:
            _, ar_loss = model(tokens, mode='AR')
        losses['ar'].append(ar_loss.item())

        # Random loss
        with ctx:
            _, rand_loss = model(tokens, mode='Random')
        losses['random'].append(rand_loss.item())

        # L2R loss on original (pre-permute) data
        if inv_perm is not None:
            original_block = inv_perm.unsqueeze(0)
            original_token = expand_block_orders_to_token_orders(
                original_block, block_len=effective_block_len)
            original_token = original_token.expand(B, -1).to(device)
            with ctx:
                _, l2r_loss = model(tokens, mode=None, orders=original_token)
            losses['l2r'].append(l2r_loss.item())

        # ON order loss (always use guided order, no mixing)
        if order_source != 'random':
            if order_source == 'edge_greedy':
                on_block = sample_edge_greedy_order(A)
            elif order_source in ('on32', 'on'):
                on_block = sample_on_order(on_model, A, temperature)
            elif order_source == 'precomputed':
                on_block = precomputed_orders[sample_indices].to(device)
            else:
                on_block = torch.randperm(NUM_BLOCKS, device=device).unsqueeze(0)
            on_token = expand_block_orders_to_token_orders(on_block, block_len=effective_block_len)
            with ctx:
                _, on_loss = model(tokens, mode=None, orders=on_token)
            losses['on_order'].append(on_loss.item())

    model.train()
    return {k: float(np.mean(v)) if v else float('nan') for k, v in losses.items()}


@torch.no_grad()
def estimate_token_losses(model, dataloader, ctx, eval_iters, order_source, on_model,
                          alpha, temperature, device, block_perm, inv_perm,
                          effective_block_len, precomputed_orders=None):
    """Estimate losses for token-only datasets."""
    model.eval()
    losses = {'train_mode': [], 'ar': [], 'random': [], 'l2r': [], 'on_order': []}
    for k, (tokens, sample_indices) in enumerate(dataloader):
        if k >= eval_iters:
            break
        tokens = tokens.to(device)
        B = tokens.shape[0]

        block_orders = sample_mixed_block_orders(
            None, alpha, order_source, on_model, temperature, device,
            precomputed_orders=precomputed_orders,
            sample_indices=sample_indices,
        )
        token_orders = expand_block_orders_to_token_orders(block_orders, block_len=effective_block_len)
        with ctx:
            _, loss = model(tokens, mode=None, orders=token_orders)
        losses['train_mode'].append(loss.item())

        with ctx:
            _, ar_loss = model(tokens, mode='AR')
        losses['ar'].append(ar_loss.item())

        with ctx:
            _, rand_loss = model(tokens, mode='Random')
        losses['random'].append(rand_loss.item())

        if inv_perm is not None:
            original_block = inv_perm.unsqueeze(0)
            original_token = expand_block_orders_to_token_orders(
                original_block, block_len=effective_block_len)
            original_token = original_token.expand(B, -1).to(device)
            with ctx:
                _, l2r_loss = model(tokens, mode=None, orders=original_token)
            losses['l2r'].append(l2r_loss.item())

        if order_source == 'precomputed':
            fixed_block = precomputed_orders[sample_indices % len(precomputed_orders)].to(device)
        elif order_source == 'random':
            fixed_block = torch.stack([
                torch.randperm(NUM_BLOCKS, device=device)
                for _ in range(B)
            ])
        else:
            fixed_block = block_orders
        fixed_token = expand_block_orders_to_token_orders(fixed_block, block_len=effective_block_len)
        with ctx:
            _, fixed_loss = model(tokens, mode=None, orders=fixed_token)
        losses['on_order'].append(fixed_loss.item())

    model.train()
    return {k: float(np.mean(v)) if v else float('nan') for k, v in losses.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ao-gpt-ckpt", type=str, default=AO_GPT_CKPT_DEFAULT)
    parser.add_argument("--num-blocks", type=int, default=32,
                        help="Number of blocks (32 or 64); auto-computes tokens_per_block")
    parser.add_argument("--alpha", type=float, default=0.5, help="Target alpha (ON-mix ratio)")
    parser.add_argument("--alpha-warmup", type=int, default=500, help="Alpha ramp steps")
    parser.add_argument("--order-source", type=str, default="edge_greedy",
                        choices=["edge_greedy", "on32", "on", "random", "precomputed"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--on-ckpt", type=str, default="")
    parser.add_argument("--on-temperature", type=float, default=1.0,
                        help="Temperature for ON order sampling (0=greedy, 1.0=default)")
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out-dir", type=str, default=OUT_DIR)
    parser.add_argument("--a32-path", type=str, default=A32_PATH)
    parser.add_argument("--tokens-path", type=str, default=TOKENS_PATH)
    parser.add_argument("--precomputed-orders-path", type=str, default="")
    parser.add_argument("--eval-log", type=str, default="")
    parser.add_argument("--save-metric", type=str, default="train_mode",
                        choices=["train_mode", "ar", "random", "l2r", "on_order"])
    parser.add_argument("--tokens-only", action="store_true", default=False,
                        help="Train on tokens_path without aligned A32; valid for random/precomputed orders")
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    ptdtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    global NUM_BLOCKS, TOKENS_PER_BLOCK
    NUM_BLOCKS = args.num_blocks
    TOKENS_PER_BLOCK = SEQ_LEN // NUM_BLOCKS
    effective_block_len = TOKENS_PER_BLOCK

    print(f"ON-mixed training: alpha={args.alpha}, source={args.order_source}, "
          f"τ={args.temperature}, max_iters={args.max_iters}")

    # ── Load AO-GPT ──
    print(f"Loading AO-GPT from {args.ao_gpt_ckpt}...")
    ckpt = torch.load(args.ao_gpt_ckpt, map_location=device, weights_only=False)
    from AOGPT import AOGPTConfig as AOGPTCfg
    sig = list(AOGPTCfg.__init__.__code__.co_varnames)
    valid = {k: v for k, v in dict(ckpt["model_args"]).items() if k in sig}
    model = AOGPT(AOGPTCfg(**valid))
    sd = ckpt["model"]
    for k in list(sd.keys()):
        clean = k.replace("_orig_mod.", "")
        if clean != k:
            sd[clean] = sd.pop(k)
    model.load_state_dict(sd)
    model.crop_block_size(SEQ_LEN)
    model.to(device)
    block_perm = torch.tensor(ckpt["data_permutation"]["block_perm"], dtype=torch.long)
    inv_perm_raw = torch.tensor(ckpt["data_permutation"]["inverse_block_perm"], dtype=torch.long)
    ckpt_n_blocks = len(block_perm)
    # Aggregate inv_perm for L2R eval if ckpt uses finer blocks than ON
    if ckpt_n_blocks != NUM_BLOCKS:
        ratio = ckpt_n_blocks // NUM_BLOCKS
        inv_perm = torch.zeros(NUM_BLOCKS, dtype=torch.long)
        for i in range(NUM_BLOCKS):
            inv_perm[i] = inv_perm_raw[i * ratio] // ratio
    else:
        inv_perm = inv_perm_raw
    print(f"  ckpt_n_blocks={ckpt_n_blocks}, ON_N={NUM_BLOCKS}, "
          f"block_perm[:8]={block_perm[:8].tolist()}, model has {sum(p.numel() for p in model.parameters()):,} params")

    # ── Load ON ──
    on_model = None
    if args.order_source in ('on32', 'on'):
        assert args.on_ckpt, "Must provide --on-ckpt for ON order source"
        print(f"Loading ON from {args.on_ckpt}...")
        on_model = load_on(args.on_ckpt, device, num_blocks=NUM_BLOCKS)
        print(f"  ON loaded, {on_model.count_parameters():,} params")

    precomputed_orders = None
    if args.order_source == 'precomputed':
        assert args.precomputed_orders_path, "Must provide --precomputed-orders-path"
        print(f"Loading precomputed orders from {args.precomputed_orders_path}...")
        precomputed_orders_np = np.load(args.precomputed_orders_path)
        assert precomputed_orders_np.shape[1] == NUM_BLOCKS, \
            f"Orders have {precomputed_orders_np.shape[1]} blocks, expected {NUM_BLOCKS}"
        precomputed_orders = torch.from_numpy(precomputed_orders_np.astype(np.int64))
        print(f"  precomputed_orders={tuple(precomputed_orders.shape)}")

    # ── Load data ──
    needs_A = args.order_source in ('on32', 'on', 'edge_greedy')
    if args.tokens_only and not needs_A:
        print("Loading token-only data...")
        train_ds = TokenDataset(args.tokens_path, split='train')
        val_ds = TokenDataset(args.tokens_path, split='val')
    else:
        if args.tokens_only and needs_A:
            print("Note: ON/edge_greedy source requires A matrices, using aligned data")
        print("Loading aligned data...")
        assert args.a32_path, "Must provide --a32-path for A-dependent order source"
        train_ds = AlignedDataset(args.tokens_path, args.a32_path, split='train')
        val_ds = AlignedDataset(args.tokens_path, args.a32_path, split='val')
    print(f"  train={len(train_ds)}, val={len(val_ds)}")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    def cycle_loader(loader):
        while True:
            for batch in loader:
                yield batch

    train_iter = cycle_loader(train_loader)

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    best_val_loss = float('inf')

    # ── Training loop ──
    iter_num = 0
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"Training: alpha ramp 0→{args.alpha} over {args.alpha_warmup} steps")
    print(f"{'='*60}")

    while iter_num < args.max_iters:
        # Eval
        if iter_num % args.eval_interval == 0:
            cur_alpha = get_alpha(iter_num, args.alpha, args.alpha_warmup)
            if args.tokens_only and not needs_A:
                val_loader = torch.utils.data.DataLoader(
                    val_ds, batch_size=args.batch_size, shuffle=True)
                val_losses = estimate_token_losses(
                    model, val_loader, ctx, args.eval_iters,
                    args.order_source, on_model, cur_alpha,
                    args.temperature, device, block_perm, inv_perm, effective_block_len,
                    precomputed_orders=precomputed_orders)
            else:
                val_losses = estimate_losses(
                    model, val_ds, ctx, args.eval_iters,
                    args.order_source, on_model, cur_alpha,
                    args.temperature, device, block_perm, inv_perm, effective_block_len,
                    precomputed_orders=precomputed_orders)

            print(f"\nStep {iter_num} (α={cur_alpha:.3f}):")
            for k, v in val_losses.items():
                marker = " <<<" if k == 'train_mode' else ""
                print(f"  val_{k}: {v:.4f}{marker}")

            if args.eval_log:
                with open(args.eval_log, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "iter_num": iter_num,
                        "alpha": cur_alpha,
                        **{f"val_{k}": v for k, v in val_losses.items()},
                    }, sort_keys=True) + "\n")

            checkpoint_metric = get_checkpoint_metric(val_losses, args.save_metric)
            if checkpoint_metric < best_val_loss:
                best_val_loss = checkpoint_metric
                ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'save_metric': args.save_metric,
                    'alpha': cur_alpha,
                }, ckpt_path)
                print(f"  -> saved {ckpt_path}")

        # Train step
        cur_alpha = get_alpha(iter_num, args.alpha, args.alpha_warmup)

        for micro_step in range(args.grad_accum):
            batch = next(train_iter)
            if args.tokens_only:
                tokens, sample_indices = batch
                A_batch = None
            else:
                tokens, A_batch, sample_indices = batch
                A_batch = A_batch.to(device)
            tokens = tokens.to(device)

            with ctx:
                block_orders = sample_mixed_block_orders(
                    A_batch, cur_alpha, args.order_source, on_model,
                    args.temperature, device,
                    precomputed_orders=precomputed_orders,
                    sample_indices=sample_indices)
                token_orders = expand_block_orders_to_token_orders(
                    block_orders, block_len=effective_block_len)
                _, loss = model(tokens, mode=None, orders=token_orders)
                loss = loss / args.grad_accum

            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if iter_num % 100 == 0:
            elapsed = time.time() - t0
            print(f"  iter {iter_num}: loss={loss.item() * args.grad_accum:.4f}, "
                  f"α={cur_alpha:.3f}, {elapsed:.0f}s")

        iter_num += 1

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
