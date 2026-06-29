"""
Adjacent-pair NLL probe: for each sequence, swap each adjacent pair in best_blocks
and measure NLL cost. Outputs per-block "rigidity" for Plackett-Luce soft sampling.

Usage:
    python -u probe_adjacent.py --hc-results probe_results/m_step_v3k/hc_golden_orders.npz
"""

import os, sys, argparse, time
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

SEQ_LEN = 256
N_BLOCKS = 64
BLOCK_LEN = 4

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


# ── Utilities ─────────────────────────────────────────────────────────────────
def block_order_to_token_order(block_order, block_len=BLOCK_LEN):
    N = len(block_order)
    token_order = np.zeros(N * block_len, dtype=np.int64)
    for i, blk in enumerate(block_order):
        for k in range(block_len):
            token_order[i * block_len + k] = blk * block_len + k
    return token_order


def phys_to_model_idx(idx_phys, inv_perm):
    B, T = idx_phys.shape
    blk_size = T // len(inv_perm)
    idx_model = torch.zeros_like(idx_phys)
    for s in range(T):
        model_block = inv_perm[s // blk_size].item()
        offset = s % blk_size
        model_pos = model_block * blk_size + offset
        idx_model[:, model_pos] = idx_phys[:, s]
    return idx_model


@torch.no_grad()
def compute_per_seq_nll(model, idx, orders):
    B, T = idx.shape
    device = idx.device
    pos = torch.arange(0, T + 1, dtype=torch.long, device=device)
    batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, T)
    tok_emb = model.transformer.wte(idx)
    tok_emb = tok_emb[batch_indices, orders]
    none_emb = model.transformer.wnonee(torch.tensor([[0]], device=device)).expand(B, -1, -1)
    tok_emb = torch.cat([none_emb, tok_emb], dim=1)
    pos_emb = model.transformer.wpe(pos).unsqueeze(0).expand(B, -1, -1)
    pos_emb_prefix = pos_emb[:, :1, :]
    pos_emb_postfix = pos_emb[:, 1:, :][batch_indices, orders]
    pos_emb_final = torch.cat([pos_emb_prefix, pos_emb_postfix], dim=1)
    tgt_emb = model.transformer.wtpe(pos[:T]).unsqueeze(0).expand(B, -1, -1)
    tgt_emb_prefix = tgt_emb[batch_indices, orders]
    tgt_emb_postfix = torch.zeros(B, 1, tgt_emb.shape[-1], device=device)
    c = torch.cat([tgt_emb_prefix, tgt_emb_postfix], dim=1)
    targets = idx[batch_indices, orders]
    x = tok_emb + pos_emb_final
    x = model.transformer.drop(x)
    for block in model.transformer.h:
        x = block(x, c)
    x = model.transformer.final_layer(x, c)
    logits = model.lm_head(x)[:, :-1, :].contiguous()
    ce_per_token = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
        reduction="none", ignore_index=-1,
    ).reshape(B, T)
    return ce_per_token.mean(dim=1)


# ── Adjacent probe ───────────────────────────────────────────────────────────
@torch.no_grad()
def probe_adjacent_rigidity(model, idx_all, best_blocks, batch_size=20, device="cuda"):
    """
    For each sequence, swap each adjacent pair in best_blocks and compute NLL cost.
    Returns: rigidity (num_seqs, N) — per-block "how bad to move from best position"

    rigidity[s, k] = NLL(swap blocks at pos k, k+1 in best_blocks) - NLL(best_blocks)
    For last block at pos N-1: copy rigidity from N-2.
    """
    num_seqs, N = best_blocks.shape
    T = N * BLOCK_LEN
    model_device = model.transformer.wte.weight.device
    model.eval()

    # Baseline NLL with best ordering
    print("  Computing best-order NLL baseline...", flush=True)
    best_nlls = np.zeros(num_seqs, dtype=np.float32)
    for b_start in range(0, num_seqs, batch_size):
        b_end = min(b_start + batch_size, num_seqs)
        B = b_end - b_start
        idx_b = idx_all[b_start:b_end].to(model_device)
        orders_b = torch.zeros(B, T, dtype=torch.long, device=model_device)
        for s in range(B):
            orders_b[s] = torch.as_tensor(
                block_order_to_token_order(best_blocks[b_start + s]), dtype=torch.long
            )
        nll_b = compute_per_seq_nll(model, idx_b, orders_b)
        best_nlls[b_start:b_end] = nll_b.cpu().numpy()

    # Evaluate adjacent swaps
    rigidity = np.zeros((num_seqs, N), dtype=np.float32)
    n_adjacent = N - 1  # 63 pairs per sequence
    total_pairs = num_seqs * n_adjacent

    print(f"  Probing {num_seqs} × {n_adjacent} = {total_pairs} adjacent swaps...", flush=True)
    t_start = time.perf_counter()

    # Process one adjacent position at a time across all sequences (efficient batching)
    for adj_pos in range(n_adjacent):
        # Build candidates: swap blocks at adj_pos and adj_pos+1
        candidate_blocks = best_blocks.copy()
        candidate_blocks[:, [adj_pos, adj_pos + 1]] = candidate_blocks[:, [adj_pos + 1, adj_pos]]

        swapped_nlls = np.zeros(num_seqs, dtype=np.float32)
        for b_start in range(0, num_seqs, batch_size):
            b_end = min(b_start + batch_size, num_seqs)
            B = b_end - b_start
            idx_b = idx_all[b_start:b_end].to(model_device)
            orders_b = torch.zeros(B, T, dtype=torch.long, device=model_device)
            for s in range(B):
                orders_b[s] = torch.as_tensor(
                    block_order_to_token_order(candidate_blocks[b_start + s]), dtype=torch.long
                )
            nll_b = compute_per_seq_nll(model, idx_b, orders_b)
            swapped_nlls[b_start:b_end] = nll_b.cpu().numpy()

        cost = np.maximum(swapped_nlls - best_nlls, 0)  # should always be >= 0 (HC optimal)
        # Distribute cost to both blocks in the swapped pair
        rigidity[:, adj_pos] += cost
        rigidity[:, adj_pos + 1] += cost

        if (adj_pos + 1) % 20 == 0 or adj_pos == 0:
            elapsed = time.perf_counter() - t_start
            eta = elapsed / (adj_pos + 1) * n_adjacent - elapsed
            mean_cost = cost.mean()
            print(f"  Adj {adj_pos + 1:3d}/{n_adjacent} | mean_cost={mean_cost:.6f} | "
                  f"{elapsed:.1f}s elapsed | ~{eta:.0f}s remaining", flush=True)

    elapsed = time.perf_counter() - t_start
    print(f"  Done. {elapsed:.1f}s total, {elapsed/n_adjacent:.1f}s per adjacent position", flush=True)

    # Normalize: each adjacent cost contributes to 2 blocks, so divide by 2
    rigidity /= 2.0

    return rigidity, best_nlls


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hc-results", required=True, help="Path to HC npz (contains best_orders)")
    p.add_argument("--output", default="", help="Output path for rigidity.npz")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--n-seqs", type=int, default=2000, help="Number of seqs to probe (from HC results)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device(args.device)
    Config.seed = args.seed
    Config.set_seed()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load HC results
    hc_data = np.load(args.hc_results, allow_pickle=True)
    best_orders = hc_data["best_orders"]  # (num_seqs, T) token orders
    if "train_indices" in hc_data:
        train_indices = hc_data["train_indices"]
    else:
        train_indices = np.arange(len(best_orders))

    n_available = len(best_orders)
    n_probe = min(args.n_seqs, n_available)
    best_orders = best_orders[:n_probe]
    train_indices = train_indices[:n_probe]
    T = best_orders.shape[1]
    N = T // BLOCK_LEN

    # Convert token orders back to block orders
    best_blocks = np.zeros((n_probe, N), dtype=np.int64)
    for s in range(n_probe):
        for k in range(N):
            best_blocks[s, k] = best_orders[s, k * BLOCK_LEN] // BLOCK_LEN
    print(f"Loaded {n_probe} sequences from {args.hc_results}")
    print(f"Best blocks shape: {best_blocks.shape}")

    # Load data
    print("\nLoading data...", flush=True)
    tok = GPT2TokenizerFast.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    tokens_all = []
    for shard in ["wikitext-train-00000-of-00002.arrow", "wikitext-train-00001-of-00002.arrow"]:
        arrow_file = os.path.join(WIKITEXT_DIR, shard)
        ds = Dataset.from_file(arrow_file)
        for ex in ds:
            raw = tok.encode(ex["text"])
            if len(raw) >= SEQ_LEN:
                tokens_all.append(raw[:SEQ_LEN])
    tokens_all = tokens_all[:5200]
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(tokens_all))
    train_idx = perm[:5000]
    idx_phys = torch.tensor(tokens_all, dtype=torch.long)
    idx_phys_hc = idx_phys[train_idx[:n_probe]]

    # Load model
    print("Loading AO-GPT...", flush=True)
    ckpt = torch.load(AO_GPT_CKPT, map_location=device, weights_only=False)
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
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    inv_perm = torch.tensor(ckpt["data_permutation"]["inverse_block_perm"], dtype=torch.long)

    # Convert to model coordinates
    print("Converting to model coordinates...", flush=True)
    idx_list = [phys_to_model_idx(idx_phys_hc[i:i + 1], inv_perm) for i in range(n_probe)]
    idx_model = torch.cat(idx_list, dim=0)

    # Probe
    print(f"\n{'='*60}")
    print(f"Adjacent probe: {n_probe} seqs × {N-1} = {n_probe*(N-1)} swaps")
    print(f"{'='*60}")
    rigidity, best_nlls = probe_adjacent_rigidity(
        model, idx_model, best_blocks,
        batch_size=args.batch_size, device=device
    )

    print(f"\nRigidity stats:")
    print(f"  Mean: {rigidity.mean():.6f}")
    print(f"  Median: {np.median(rigidity):.6f}")
    print(f"  Max: {rigidity.max():.6f}")
    print(f"  Pct zero: {(rigidity < 1e-7).mean()*100:.1f}%")

    # Save
    out_dir = Path(args.output) if args.output else Path(args.hc_results).parent
    if out_dir.is_dir():
        out_path = out_dir / "adjacent_rigidity.npz"
    else:
        out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        rigidity=rigidity,
        best_nlls=best_nlls,
        best_blocks=best_blocks,
        train_indices=train_indices,
        n_probe=n_probe,
        source_hc=args.hc_results,
    )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
