"""
ON-mixed AO-GPT continual training v2 — order refresh.

Instead of pre-extracted A32, uses frozen extractor to get A on-the-fly,
then caches orders (not A). Every K steps, orders are refreshed.

Key differences from v1:
- Data: full wikitext-103 (not 10k subset)
- Order cache: seq_id → block_perm (32,) int64, refreshed every K steps
- No pre-extracted A32 needed
- Frozen extractor separate from training model

Usage:
    # ON32 mixed
    python train_on_mixed_v2.py --alpha 0.5 --order-source on32 --on-ckpt probe_results/on32_soft_edge_best.pt --device cuda:0

    # Edge-greedy mixed (no ON needed)
    python train_on_mixed_v2.py --alpha 0.5 --order-source edge_greedy --device cuda:0

    # Random baseline
    python train_on_mixed_v2.py --alpha 0.0 --order-source random --device cuda:0
"""

import os, sys, time, math, argparse
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
AO_GPT_CKPT = os.path.expanduser(
    "~/ych/nanogpt-learned-order/out/base/permute/seq256/block32/"
    "out-wikitext103-seq256-random-b32-permute-block/ckpt.pt"
)
TOKENS_PATH = "probe_results/wikitext103_train_tokens.npy"
OUT_DIR = "probe_results/on_mixed_training"

NUM_BLOCKS = 32
TOKENS_PER_BLOCK = 8  # 256 / 32
SEQ_LEN = 256


# ── Frozen A-extractor ──

def load_extractor(ckpt_path, device):
    """Load frozen AO-GPT for A extraction. Returns (model, block_perm_np, inv_perm_np)."""
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
    bp = np.array(ckpt["data_permutation"]["block_perm"], dtype=np.int64)
    inv_perm = np.array(ckpt["data_permutation"]["inverse_block_perm"], dtype=np.int64)
    return model, bp, inv_perm


@torch.no_grad()
def extract_batch_A(extractor, tokens, block_perm, device):
    """Extract A matrices for a batch of tokens.

    tokens: (B, 256) on CPU.
    Returns: (B, 32, 32) tensor on device, in MODEL coordinates.
    """
    B = tokens.shape[0]
    A_list = []

    for b in range(B):
        idx = tokens[b:b + 1].to(device)  # (1, 256)

        # Random block order in physical coords → token order in model coords
        rand_order = torch.randperm(NUM_BLOCKS, device=device)
        token_order = torch.zeros(SEQ_LEN, dtype=torch.long, device=device)
        for t in range(NUM_BLOCKS):
            phys_block = rand_order[t].item()
            model_block = int(block_perm[phys_block])
            for k in range(TOKENS_PER_BLOCK):
                token_order[t * TOKENS_PER_BLOCK + k] = model_block * TOKENS_PER_BLOCK + k
        token_order = token_order.unsqueeze(0)  # (1, 256)

        _, _, attn_list = extractor.forward_fn(idx, token_order, return_attentions=True)
        attn_stack = torch.stack(attn_list).squeeze(1).cpu().numpy()  # (L, H, 257, 257)
        L, H = attn_stack.shape[:2]

        # Top-4 variance heads
        head_vars = np.zeros(H)
        for h in range(H):
            content = attn_stack[:, h, 1:, 1:]
            offdiag = content[:, ~np.eye(256, dtype=bool)].reshape(L, 256, 255)
            head_vars[h] = float(np.var(offdiag))
        top_heads = np.argsort(head_vars)[-4:]
        avg_attn = attn_stack[:, top_heads, :, :].mean(axis=(0, 1))  # (257, 257)

        reveal_to_model = token_order[0].cpu().numpy()
        model_to_reveal = np.zeros(256, dtype=np.int64)
        model_to_reveal[reveal_to_model] = np.arange(256)

        attn_content = avg_attn[1:, 1:]  # (256, 256)
        attn_model = attn_content[model_to_reveal][:, model_to_reveal]

        A = np.zeros((NUM_BLOCKS, NUM_BLOCKS), dtype=np.float32)
        for i in range(NUM_BLOCKS):
            i_s, i_e = i * TOKENS_PER_BLOCK, (i + 1) * TOKENS_PER_BLOCK
            for j in range(NUM_BLOCKS):
                j_s, j_e = j * TOKENS_PER_BLOCK, (j + 1) * TOKENS_PER_BLOCK
                A[i, j] = attn_model[i_s:i_e, j_s:j_e].mean()

        none_attn = avg_attn[1:, 0]
        none_model = none_attn[model_to_reveal]
        none_block = np.array([none_model[i * TOKENS_PER_BLOCK:(i + 1) * TOKENS_PER_BLOCK].mean()
                               for i in range(NUM_BLOCKS)])
        A += none_block[np.newaxis, :] * 0.1
        np.fill_diagonal(A, 0.0)
        A_list.append(torch.from_numpy(A))

    return torch.stack(A_list, dim=0).to(device)  # (B, 32, 32)


# ── Edge-greedy order source ──

def sample_edge_greedy_order(A_batch):
    B, N, _ = A_batch.shape
    device = A_batch.device
    orders = torch.zeros(B, N, dtype=torch.long, device=device)
    visited = torch.zeros(B, dtype=torch.long, device=device)

    for t in range(N):
        if t == 0:
            scores = A_batch.mean(dim=2)
        else:
            last = orders[:, t - 1]
            scores = A_batch[torch.arange(B), :, last]

        # Mask visited
        for b in range(B):
            for i in range(N):
                if visited[b] & (1 << i):
                    scores[b, i] = float('-inf')

        chosen = scores.argmax(dim=1)
        orders[:, t] = chosen
        for b in range(B):
            visited[b] |= (1 << chosen[b].item())

    return orders


# ── ON32 order source ──

def load_on32(ckpt_path, device):
    on = CrossAttentionOrderNetwork(num_blocks=NUM_BLOCKS, d_edge=64, d_model=64).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    on.load_state_dict(ckpt["model_state_dict"])
    on.eval()
    for p in on.parameters():
        p.requires_grad = False
    return on


@torch.no_grad()
def sample_on_order(on_model, A_batch, temperature=1.0):
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

class WikiTextDataset(torch.utils.data.Dataset):
    """Loads pre-tokenized wikitext-103 chunks. Returns (tokens, seq_id)."""

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
        return tokens, i  # i is the global index, used as cache key


# ── Training ──

def get_alpha(step, alpha_target, alpha_warmup_steps):
    if alpha_target <= 0 or alpha_warmup_steps <= 0:
        return alpha_target
    return alpha_target * min(1.0, step / alpha_warmup_steps)


@torch.no_grad()
def estimate_losses(model, eval_loader, ctx, eval_iters, device, effective_block_len):
    model.eval()
    losses = {'ar': [], 'random': [], 'l2r': []}
    for k, (tokens, _) in enumerate(eval_loader):
        if k >= eval_iters:
            break
        tokens = tokens.to(device)
        B = tokens.shape[0]

        with ctx:
            _, ar_loss = model(tokens, mode='AR')
        losses['ar'].append(ar_loss.item())

        with ctx:
            _, rand_loss = model(tokens, mode='Random')
        losses['random'].append(rand_loss.item())

    model.train()
    return {k: float(np.mean(v)) if v else float('nan') for k, v in losses.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--alpha-warmup", type=int, default=500)
    parser.add_argument("--order-source", type=str, default="on32",
                        choices=["edge_greedy", "on32", "random"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--on-ckpt", type=str, default="")
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out-dir", type=str, default=OUT_DIR)
    parser.add_argument("--refresh-interval", type=int, default=100,
                        help="Every K steps, re-extract A and refresh orders")
    parser.add_argument("--order-cache-size", type=int, default=50000,
                        help="Max number of cached orders")
    parser.add_argument("--tokens-path", type=str, default=TOKENS_PATH)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    ptdtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    effective_block_len = TOKENS_PER_BLOCK

    print(f"ON-mixed v2: alpha={args.alpha}, source={args.order_source}, "
          f"τ={args.temperature}, refresh_every={args.refresh_interval}")
    print(f"Tokens: {args.tokens_path}")

    # ── Load frozen A-extractor ──
    print("Loading frozen A-extractor...")
    extractor, extractor_bp, extractor_inv = load_extractor(AO_GPT_CKPT, device)
    print(f"  Extractor loaded, block_perm[:8]={extractor_bp[:8].tolist()}")

    # ── Load ON32 ──
    on_model = None
    if args.order_source == 'on32':
        assert args.on_ckpt, "--on-ckpt required for on32 source"
        print(f"Loading ON32 from {args.on_ckpt}...")
        on_model = load_on32(args.on_ckpt, device)
        print(f"  ON32 loaded")

    # ── Load training AO-GPT ──
    print("Loading AO-GPT for training...")
    ckpt = torch.load(AO_GPT_CKPT, map_location=device, weights_only=False)
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
    print(f"  Model has {sum(p.numel() for p in model.parameters()):,} params")

    # ── Load data ──
    if not os.path.exists(args.tokens_path):
        print(f"Tokens file not found: {args.tokens_path}")
        print("Run: python tokenize_wikitext103.py")
        sys.exit(1)
    print(f"Loading data from {args.tokens_path}...")
    train_ds = WikiTextDataset(args.tokens_path, split='train')
    val_ds = WikiTextDataset(args.tokens_path, split='val')
    print(f"  train={len(train_ds)}, val={len(val_ds)}")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=True)

    def cycle_loader(loader):
        while True:
            for batch in loader:
                yield batch

    train_iter = cycle_loader(train_loader)

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    best_val_loss = float('inf')

    # ── Order cache ──
    order_cache = {}  # seq_id → block_order (32,) long tensor on CPU

    def get_orders_for_batch(tokens, seq_ids):
        """Get block orders for a batch. Extract A on cache miss or periodic refresh."""
        B = tokens.shape[0]
        device = tokens.device

        orders = torch.zeros(B, NUM_BLOCKS, dtype=torch.long, device=device)

        # Per-sample: ON-guided or random?
        use_on = np.zeros(B, dtype=bool)
        for b in range(B):
            use_on[b] = (np.random.random() < alpha)

        for b in range(B):
            sid = int(seq_ids[b])
            do_refresh = (iter_num % args.refresh_interval == 0)

            if use_on[b]:
                if sid not in order_cache or do_refresh:
                    # Extract A → ON order
                    A_b = extract_batch_A(
                        extractor, tokens[b:b + 1].cpu(),
                        extractor_bp, device)  # (1, 32, 32)

                    if args.order_source == 'edge_greedy':
                        ord_b = sample_edge_greedy_order(A_b)
                    elif args.order_source == 'on32':
                        ord_b = sample_on_order(on_model, A_b, args.temperature)
                    else:
                        ord_b = torch.randperm(NUM_BLOCKS, device=device).unsqueeze(0)

                    order_cache[sid] = ord_b[0].cpu()

                    # Evict oldest if cache full
                    if len(order_cache) > args.order_cache_size:
                        oldest_key = next(iter(order_cache))
                        del order_cache[oldest_key]

                orders[b] = order_cache[sid].to(device)
            else:
                orders[b] = torch.randperm(NUM_BLOCKS, device=device)

        return orders

    # ── Training loop ──
    alpha = get_alpha(0, args.alpha, args.alpha_warmup)
    iter_num = 0
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"Training: alpha ramp 0→{args.alpha} over {args.alpha_warmup} steps")
    print(f"Refresh interval: {args.refresh_interval} steps")
    print(f"Data: {len(train_ds)} train seqs (full wikitext-103)")
    print(f"{'='*60}")

    while iter_num < args.max_iters:
        # Eval
        if iter_num % args.eval_interval == 0:
            alpha = get_alpha(iter_num, args.alpha, args.alpha_warmup)
            val_losses = estimate_losses(
                model, val_loader, ctx, args.eval_iters, device, effective_block_len)

            print(f"\nStep {iter_num} (α={alpha:.3f}, cache={len(order_cache)}):")
            for k, v in val_losses.items():
                marker = " <<<" if k == 'ar' else ""
                print(f"  val_{k}: {v:.4f}{marker}")

            if val_losses['ar'] < best_val_loss:
                best_val_loss = val_losses['ar']
                ckpt_path = os.path.join(args.out_dir, f'{args.order_source}_v2_ckpt.pt')
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'alpha': alpha,
                }, ckpt_path)
                print(f"  -> saved {ckpt_path}")

        # Train step
        alpha = get_alpha(iter_num, args.alpha, args.alpha_warmup)

        for micro_step in range(args.grad_accum):
            tokens, seq_ids = next(train_iter)
            tokens = tokens.to(device)

            with ctx:
                block_orders = get_orders_for_batch(tokens, seq_ids)
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
            n_cache = len(order_cache)
            print(f"  iter {iter_num}: loss={loss.item() * args.grad_accum:.4f}, "
                  f"α={alpha:.3f}, cache={n_cache}, {elapsed:.0f}s")

        iter_num += 1

    print(f"\nDone. Best val ar loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
