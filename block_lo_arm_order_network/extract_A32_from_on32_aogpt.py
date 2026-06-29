"""Extract A32 attention matrices from an ON32-trained AO-GPT checkpoint.

The ON32 continual checkpoint only stores model weights. This script loads the
base block32 AO-GPT checkpoint for model args and data permutation, then overlays
the finetuned weights before extracting attention on token chunks.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.expanduser("~/ych/nanogpt-learned-order"))

from AOGPT import AOGPT, AOGPTConfig

SEQ_LEN = 256
NUM_BLOCKS = 32
TOKENS_PER_BLOCK = SEQ_LEN // NUM_BLOCKS

BASE_CKPT = os.path.expanduser(
    "~/ych/nanogpt-learned-order/out/base/permute/seq256/block32/"
    "out-wikitext103-seq256-random-b32-permute-block/ckpt.pt"
)
FINETUNED_CKPT = (
    "probe_results/on_mixed_training/phase1_global_fixed_on32_valar/ckpt.pt"
)
TOKENS_PATH = "probe_results/wikitext103_train_tokens.npy"
OUTPUT_DIR = "probe_results/on32_aogpt_attention"


def _clean_state_dict(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        cleaned[key.replace("_orig_mod.", "")] = value
    return cleaned


def load_model(base_ckpt_path, finetuned_ckpt_path, device):
    base = torch.load(base_ckpt_path, map_location=device, weights_only=False)
    sig = list(AOGPTConfig.__init__.__code__.co_varnames)
    valid = {k: v for k, v in dict(base["model_args"]).items() if k in sig}
    model = AOGPT(AOGPTConfig(**valid))
    model.load_state_dict(_clean_state_dict(base["model"]))

    finetuned = torch.load(finetuned_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(_clean_state_dict(finetuned["model"]))
    model.crop_block_size(SEQ_LEN)
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False

    block_perm = np.array(base["data_permutation"]["block_perm"], dtype=np.int64)
    return model, block_perm


def build_model_to_phys(block_perm):
    inv = np.zeros(NUM_BLOCKS, dtype=np.int64)
    for phys_block, model_block in enumerate(block_perm):
        inv[model_block] = phys_block

    model_to_phys = np.zeros(SEQ_LEN, dtype=np.int64)
    for model_pos in range(SEQ_LEN):
        model_block = model_pos // TOKENS_PER_BLOCK
        offset = model_pos % TOKENS_PER_BLOCK
        phys_block = inv[model_block]
        model_to_phys[model_pos] = phys_block * TOKENS_PER_BLOCK + offset
    return model_to_phys


def build_token_order(block_order, block_perm, device):
    token_order = torch.zeros(SEQ_LEN, dtype=torch.long, device=device)
    for step, phys_block in enumerate(block_order.tolist()):
        model_block = int(block_perm[phys_block])
        for offset in range(TOKENS_PER_BLOCK):
            token_order[step * TOKENS_PER_BLOCK + offset] = (
                model_block * TOKENS_PER_BLOCK + offset
            )
    return token_order


@torch.no_grad()
def extract_one_A(model, tokens, block_perm, model_to_phys, device):
    idx = tokens.to(device).unsqueeze(0)
    block_order = torch.randperm(NUM_BLOCKS, device=device)
    token_order = build_token_order(block_order, block_perm, device).unsqueeze(0)

    _, _, attn_list = model.forward_fn(idx, token_order, return_attentions=True)
    attn_stack = torch.stack(attn_list).squeeze(1).cpu().numpy()

    num_heads = attn_stack.shape[1]
    head_vars = np.zeros(num_heads)
    offdiag_mask = ~np.eye(SEQ_LEN, dtype=bool)
    for head in range(num_heads):
        content = attn_stack[:, head, 1:, 1:]
        offdiag = content[:, offdiag_mask].reshape(attn_stack.shape[0], SEQ_LEN, SEQ_LEN - 1)
        head_vars[head] = float(np.var(offdiag))
    top_heads = np.argsort(head_vars)[-4:]
    avg_attn = attn_stack[:, top_heads, :, :].mean(axis=(0, 1))

    reveal_to_model = token_order[0].cpu().numpy()
    reveal_to_phys = model_to_phys[reveal_to_model]
    attn_content = avg_attn[1:, 1:]
    attn_phys = np.zeros((SEQ_LEN, SEQ_LEN), dtype=np.float32)
    for reveal_q in range(SEQ_LEN):
        phys_q = reveal_to_phys[reveal_q]
        attn_phys[phys_q, reveal_to_phys] += attn_content[reveal_q]

    A = np.zeros((NUM_BLOCKS, NUM_BLOCKS), dtype=np.float32)
    for i in range(NUM_BLOCKS):
        i_s, i_e = i * TOKENS_PER_BLOCK, (i + 1) * TOKENS_PER_BLOCK
        for j in range(NUM_BLOCKS):
            j_s, j_e = j * TOKENS_PER_BLOCK, (j + 1) * TOKENS_PER_BLOCK
            A[i, j] = attn_phys[i_s:i_e, j_s:j_e].mean()

    none_attn = avg_attn[1:, 0]
    none_phys = np.zeros(SEQ_LEN, dtype=np.float32)
    none_phys[reveal_to_phys] = none_attn
    none_block = np.array([
        none_phys[i * TOKENS_PER_BLOCK:(i + 1) * TOKENS_PER_BLOCK].mean()
        for i in range(NUM_BLOCKS)
    ])
    A += none_block[np.newaxis, :] * 0.1
    np.fill_diagonal(A, 0.0)
    return A


def nn_greedy(W, start):
    remaining = set(range(W.shape[0]))
    remaining.remove(start)
    order = [start]
    current = start
    while remaining:
        nxt = max(remaining, key=lambda j: W[current, j])
        order.append(nxt)
        remaining.remove(nxt)
        current = nxt
    return np.array(order, dtype=np.int16)


def nn_best_endpoint(W, n_candidates=2):
    degree = W.sum(axis=1)
    candidates = np.argsort(degree)[:n_candidates]
    best_order = None
    best_weight = -np.inf
    for start in candidates:
        order = nn_greedy(W, int(start))
        weight = sum(W[order[i], order[i + 1]] for i in range(len(order) - 1))
        if weight > best_weight:
            best_weight = weight
            best_order = order
    return best_order, best_weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ckpt", default=BASE_CKPT)
    parser.add_argument("--finetuned-ckpt", default=FINETUNED_CKPT)
    parser.add_argument("--tokens-path", default=TOKENS_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--n-chunks", type=int, default=2000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"Device: {device}, n_chunks={args.n_chunks}", flush=True)

    print("Loading ON32-trained AO-GPT...", flush=True)
    model, block_perm = load_model(args.base_ckpt, args.finetuned_ckpt, device)
    model_to_phys = build_model_to_phys(block_perm)
    print(f"  block_perm[:8]={block_perm[:8].tolist()}", flush=True)

    tokens_all = np.load(args.tokens_path, mmap_mode="r")
    n = min(args.n_chunks, len(tokens_all))
    A_all = np.zeros((n, NUM_BLOCKS, NUM_BLOCKS), dtype=np.float32)
    paths = np.zeros((n, NUM_BLOCKS), dtype=np.int16)
    weights = np.zeros(n, dtype=np.float32)

    t0 = time.time()
    for i in tqdm(range(n), desc="Extract A32"):
        tokens = torch.from_numpy(tokens_all[i].astype(np.int64))
        A = extract_one_A(model, tokens, block_perm, model_to_phys, device)
        A_all[i] = A
        W = 0.5 * (A + A.T)
        np.fill_diagonal(W, 0.0)
        order, weight = nn_best_endpoint(W)
        paths[i] = order
        weights[i] = weight

    np.save(output_dir / "A32_on32_aogpt.npy", A_all)
    np.save(output_dir / "NN32_on32_aogpt_paths.npy", paths)
    np.save(output_dir / "NN32_on32_aogpt_weights.npy", weights)
    np.save(output_dir / "tokens.npy", np.array(tokens_all[:n], dtype=np.int32))

    from scipy.stats import kendalltau
    l2r = np.arange(NUM_BLOCKS)
    taus = np.array([kendalltau(l2r, paths[i])[0] for i in range(n)])
    print(f"Saved to {output_dir}", flush=True)
    print(f"Time: {time.time() - t0:.1f}s", flush=True)
    print(f"NN32 tau vs L2R: mean={taus.mean():+.4f}, std={taus.std():.4f}, "
          f"median={np.median(taus):+.4f}, >0.5={(taus > 0.5).mean()*100:.1f}%",
          flush=True)
    print(f"First path: {paths[0].tolist()}", flush=True)


if __name__ == "__main__":
    main()
