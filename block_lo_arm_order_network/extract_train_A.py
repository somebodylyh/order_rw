"""
Extract NxN attention A matrices from wikitext-103 TRAIN set chunks.

For each 256-token chunk:
  1. Generate random N-block order (N=16 or N=64)
  2. Convert to token order in model coordinates
  3. One AO-GPT forward with return_attentions=True
  4. Aggregate attention -> NxN matrix in physical coordinates
  5. Add [None] signal, zero diagonal

Usage:
    python -u extract_train_A.py --n-chunks 10000 --n-blocks 64 --device cuda:0
"""

import os, sys, time, argparse
from pathlib import Path

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "AO-GPT-MDM"))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from model_AOGPT_AdaLN6_NoRep_cond_128_trunc_qknorm import AOGPT, AOGPTConfig

# ── Paths ──
AO_GPT_CKPT = os.path.expanduser(
    "~/ych/nanogpt-learned-order/out/base/permute/seq256/block64/"
    "out-wikitext103-seq256-random-b64-permute-block-50000-iters/ckpt.pt"
)
CACHE_DIR = os.path.expanduser("~/cache")
OUTPUT_DIR = "probe_results"

SEQ_LEN = 256
N64 = 64
BLOCK_LEN = 4  # tokens per atomic N64 block (model granularity)


def load_aogpt(ckpt_path, device):
    """Load frozen AO-GPT. Returns (model, block_perm, inv_perm)."""
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


def build_token_order(rand_order, bp, n_blocks):
    """Build model-coordinate token order from random block permutation.

    rand_order: (n_blocks,) random permutation of [0..n_blocks-1], physical indices
    bp: (64,) block_perm[phys64] = model64
    n_blocks: number of ON-action blocks (16 or 64)
    Returns: (256,) token indices in model coordinates
    """
    tokens_per_block = SEQ_LEN // n_blocks
    n64_per_step = N64 // n_blocks  # 4 for n_blocks=16, 1 for n_blocks=64
    token_order = torch.zeros(SEQ_LEN, dtype=torch.long)
    for t in range(n_blocks):
        p = rand_order[t].item()
        for a in range(n64_per_step):
            phys_n64 = p * n64_per_step + a
            model_n64 = int(bp[phys_n64])
            for k in range(BLOCK_LEN):
                pos = t * tokens_per_block + a * BLOCK_LEN + k
                token_order[pos] = model_n64 * BLOCK_LEN + k
    return token_order


def build_model_to_phys(bp):
    """Build model_position -> physical_position mapping.

    bp: (64,) block_perm[phys64] = model64
    Returns: (256,) array, model_to_phys[model_pos] = phys_pos
    """
    bp_i = np.zeros(64, dtype=np.int64)
    for i in range(64):
        bp_i[bp[i]] = i  # inv_perm: model64 -> phys64
    model_to_phys = np.zeros(256, dtype=np.int64)
    for mp in range(256):
        mb = mp // BLOCK_LEN         # model N64 block (0..63)
        phys_n64 = bp_i[mb]          # physical N64 block (0..63)
        off = mp % BLOCK_LEN         # offset within block (0..3)
        model_to_phys[mp] = phys_n64 * BLOCK_LEN + off
    return model_to_phys


@torch.no_grad()
def extract_one_A(model, idx_tokens, bp, model_to_phys, device, n_blocks=16):
    """Extract n_blocks×n_blocks attention A matrix for one chunk.

    Returns: (n_blocks, n_blocks) float32 numpy array in physical coordinates.
    """
    idx = idx_tokens.to(device).unsqueeze(0)  # (1, 256)
    tokens_per_block = SEQ_LEN // n_blocks

    # Random block order -> token order
    rand_order = torch.randperm(n_blocks, device=device)
    token_order = build_token_order(rand_order, bp, n_blocks).to(device)
    token_order = token_order.unsqueeze(0)  # (1, 256)

    # Forward with attentions
    _, _, attn_list = model.forward_fn(idx, token_order, return_attentions=True)
    attn_stack = torch.stack(attn_list).squeeze(1).cpu().numpy()  # (L, H, 257, 257)

    L, H = attn_stack.shape[:2]

    # Select top-4 highest-variance heads
    head_vars = np.zeros(H)
    for h in range(H):
        content = attn_stack[:, h, 1:, 1:]  # skip [None] row/col
        offdiag = content[:, ~np.eye(256, dtype=bool)].reshape(L, 256, 255)
        head_vars[h] = float(np.var(offdiag))
    top_heads = np.argsort(head_vars)[-4:]
    avg_attn = attn_stack[:, top_heads, :, :].mean(axis=(0, 1))  # (257, 257)

    # Remap reveal->physical
    reveal_to_model = token_order[0].cpu().numpy()
    reveal_to_phys = model_to_phys[reveal_to_model]

    attn_content = avg_attn[1:, 1:]  # (256, 256) content-content attention
    attn_phys = np.zeros((256, 256), dtype=np.float32)
    for rq in range(256):
        pq = reveal_to_phys[rq]
        for rk in range(256):
            attn_phys[pq, reveal_to_phys[rk]] += attn_content[rq, rk]

    # Aggregate to n_blocks×n_blocks
    A = np.zeros((n_blocks, n_blocks), dtype=np.float32)
    for i in range(n_blocks):
        for j in range(n_blocks):
            i_s, i_e = i * tokens_per_block, (i + 1) * tokens_per_block
            j_s, j_e = j * tokens_per_block, (j + 1) * tokens_per_block
            A[i, j] = attn_phys[i_s:i_e, j_s:j_e].mean()

    # [None] signal (each block's attention to [None])
    none_attn = avg_attn[1:, 0]
    none_block = np.array([none_attn[i * tokens_per_block:(i + 1) * tokens_per_block].mean()
                           for i in range(n_blocks)])
    A += none_block[np.newaxis, :] * 0.1
    np.fill_diagonal(A, 0.0)

    return A


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-chunks", type=int, default=10000)
    parser.add_argument("--n-blocks", type=int, default=16,
                        help="Number of ON-action blocks (16 or 64)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--ckpt", type=str, default=AO_GPT_CKPT,
        help="AOGPT checkpoint path (default: ych Random_CL ckpt)",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = f"{OUTPUT_DIR}/A_train_n{args.n_blocks}_10k.npy"

    device = torch.device(args.device)
    print(f"Device: {device}, n_blocks={args.n_blocks}, target: {args.n_chunks} chunks",
          flush=True)

    # ── Load AO-GPT ──
    print("Loading AO-GPT...", flush=True)
    model, bp, inv_perm = load_aogpt(args.ckpt, device)
    model_to_phys = build_model_to_phys(bp)
    print(f"Model loaded, block_perm[:8]={bp[:8].tolist()}", flush=True)

    # ── Load & tokenize wikitext-103 train set ──
    print("Loading & tokenizing wikitext-103 train set...", flush=True)
    from datasets import Dataset
    from transformers import GPT2TokenizerFast

    TOKENIZER_DIR = os.path.expanduser(
        "~/.cache/huggingface/hub/models--gpt2/snapshots/"
        "607a30d783dfa663caf39e06633721c8d4cfcd7e"
    )
    WIKITEXT_CACHE = os.path.expanduser(
        "~/.cache/huggingface/datasets/wikitext/wikitext-103-raw-v1/0.0.0/"
        "b08601e04326c79dfdd32d625aee71d232d685c3"
    )

    tok = GPT2TokenizerFast.from_pretrained(TOKENIZER_DIR, local_files_only=True)

    # Load raw text from both train arrow shards and chunk in one pass
    print("Loading & chunking wikitext-103 train set...", flush=True)
    chunks = []
    buffer_ids = []  # accumulate token ids from small texts to combine into 256-token chunks
    total_texts = 0
    for shard in ["wikitext-train-00000-of-00002.arrow", "wikitext-train-00001-of-00002.arrow"]:
        ds = Dataset.from_file(os.path.join(WIKITEXT_CACHE, shard))
        for ex in tqdm(ds, desc=f"Chunking {shard}", unit=" texts"):
            total_texts += 1
            ids = tok.encode(ex["text"])
            if len(ids) < SEQ_LEN:
                # Short text: accumulate into buffer for combined chunks
                buffer_ids.extend(ids)
                while len(buffer_ids) >= SEQ_LEN:
                    chunks.append(buffer_ids[:SEQ_LEN])
                    buffer_ids = buffer_ids[SEQ_LEN:]
                    if len(chunks) >= args.n_chunks:
                        break
            else:
                # Long text: extract sliding-window chunks
                for start in range(0, len(ids) - SEQ_LEN + 1, SEQ_LEN // 2):
                    chunks.append(ids[start:start + SEQ_LEN])
                    if len(chunks) >= args.n_chunks:
                        break
            if len(chunks) >= args.n_chunks:
                break
        if len(chunks) >= args.n_chunks:
            break
    print(f"Processed {total_texts} texts", flush=True)

    train_chunks = chunks[:args.n_chunks]
    print(f"Got {len(train_chunks)} chunks of {SEQ_LEN} tokens", flush=True)

    # ── Extract A matrices ──
    n_available = min(args.n_chunks, len(train_chunks))
    print(f"Extracting {n_available} A matrices ({args.n_blocks}x{args.n_blocks})...", flush=True)

    A_list = []
    t0 = time.time()

    for i in tqdm(range(n_available), desc="Extracting A"):
        tokens = torch.tensor(train_chunks[i], dtype=torch.long)
        A = extract_one_A(model, tokens, bp, model_to_phys, device, n_blocks=args.n_blocks)
        A_list.append(A)

        # Log speed every 500
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_available - i - 1) / rate
            print(f"  {i+1}/{n_available} | {rate:.1f} chunks/s | ETA {eta:.0f}s", flush=True)

    # ── Save ──
    A_all = np.stack(A_list, axis=0).astype(np.float32)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, A_all)

    elapsed = time.time() - t0
    print(f"Saved: {output_path} | shape={A_all.shape} | "
          f"range=[{A_all.min():.4f}, {A_all.max():.4f}] | "
          f"time={elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
