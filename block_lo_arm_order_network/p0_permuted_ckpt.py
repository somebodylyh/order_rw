"""
P0: 从 ych permute_data=True checkpoint 提取 delta-attention A 矩阵 (N=64).

关键区别 vs p0_extract_delta_attention.py:
  - 数据按 permuted 顺序喂入 (block_perm, 匹配训练分布)
  - attention 先解 random reveal → model coords, 再用 inverse_block_perm → physical coords
  - N=64 blocks × 4 tokens = 256 (匹配模型的 block_order_block_len=4)

用法:
    python p0_permuted_ckpt.py --num_seqs 100 --output probe_results/real_A_permuted_n64.npy
    python p0_permuted_ckpt.py --num_seqs 5 --debug
"""

import os
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'AO-GPT-MDM'))

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import Dataset
from transformers import GPT2TokenizerFast

from model_AOGPT_AdaLN6_NoRep_cond_128_trunc_qknorm import AOGPT, AOGPTConfig

# ---------------------------------------------------------------------------
# 配置: N=64 blocks × 4 tokens = 256 seq_len (匹配模型 block_order_block_len=4)
# ---------------------------------------------------------------------------
NUM_BLOCKS = 64
BLOCK_LEN = 4
SEQ_LEN = NUM_BLOCKS * BLOCK_LEN  # 256

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


# ---------------------------------------------------------------------------
# Model & Data
# ---------------------------------------------------------------------------

def load_model_and_perm(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    sig_params = list(AOGPTConfig.__init__.__code__.co_varnames)
    valid_args = {k: v for k, v in dict(checkpoint["model_args"]).items() if k in sig_params}
    model = AOGPT(AOGPTConfig(**valid_args))
    state_dict = checkpoint["model"]
    for key in list(state_dict.keys()):
        clean = key.replace("_orig_mod.", "")
        if clean != key:
            state_dict[clean] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.crop_block_size(SEQ_LEN)
    model.to(device)
    model.eval()

    block_perm = torch.tensor(checkpoint["data_permutation"]["block_perm"], dtype=torch.long)
    inv_perm = torch.tensor(checkpoint["data_permutation"]["inverse_block_perm"], dtype=torch.long)

    return model, block_perm, inv_perm


def load_sequences(min_seq_len=SEQ_LEN, max_sequences=500):
    ds = Dataset.from_file(WIKITEXT_ARROW)
    tokenizer = GPT2TokenizerFast.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    sequences = []
    for example in ds:
        ids = tokenizer.encode(example["text"])
        if len(ids) >= min_seq_len:
            sequences.append(ids[:min_seq_len])
        if len(sequences) >= max_sequences:
            break
    return sequences


# ---------------------------------------------------------------------------
# Token-level permutation
# ---------------------------------------------------------------------------

def permute_sequence(seq, block_perm, block_len=BLOCK_LEN):
    """
    将物理顺序的 token 序列按 block_perm 重排为模型坐标顺序。
    seq: (T,) in physical order
    block_perm: (NUM_BLOCKS,) — block_perm[phys_b] = model_b
    返回: (T,) in model-coordinate order
    """
    T = len(seq)
    num_blocks = T // block_len
    seq_2d = seq.view(num_blocks, block_len)  # (N, BL)
    permuted = seq_2d[block_perm]              # (N, BL) in model-block order
    return permuted.flatten()                   # (T,)


def reorder_attention_model_to_physical(attn, inv_perm, block_len=BLOCK_LEN):
    """
    将 attention 从模型坐标映射回物理坐标。
    attn: (T+1, T+1) or (H, T+1, T+1) — in model coords
    inv_perm: (NUM_BLOCKS,) — inv_perm[model_b] = phys_b

    返回: same shape, in physical coords (preserving within-block token order).
    [None] 在位置 0 保持不动。
    """
    if attn.dim() == 3:
        H = attn.shape[0]
        reordered = torch.zeros_like(attn)
        for h in range(H):
            reordered[h] = reorder_attention_model_to_physical(attn[h], inv_perm, block_len)
        return reordered

    # 2D: (T+1, T+1)
    Tp1 = attn.shape[0]
    T = Tp1 - 1
    num_blocks = T // block_len

    # 构建 token-level inverse permutation
    # model_pos t (1-indexed after [None]) → order_block = (t-1)//block_len
    # → phys_block = inv_perm[order_block] → phys_pos = phys_block*block_len + (t-1)%block_len + 1
    token_inv_perm = torch.zeros(T, dtype=torch.long)
    for t in range(T):
        order_block = t // block_len
        offset = t % block_len
        phys_block = inv_perm[order_block].item()
        token_inv_perm[t] = phys_block * block_len + offset

    reordered = torch.zeros_like(attn)
    reordered[0, 0] = attn[0, 0]           # [None]→[None]
    reordered[0, 1:] = attn[0, 1:][token_inv_perm]           # [None]→tokens
    reordered[1:, 0] = attn[1:, 0][token_inv_perm]           # tokens→[None]
    reordered[1:, 1:] = attn[1:, 1:][token_inv_perm][:, token_inv_perm]  # tokens→tokens

    return reordered


def reorder_attention_from_reveal(attn, orders):
    """
    attn: (H, T+1, T+1) — last layer, in reveal (shuffled) coords
    orders: (T,) — reveal order
    → (H, T+1, T+1) in ascending (model position) coords
    """
    H = attn.shape[0]
    reordered = torch.zeros_like(attn)
    for h in range(H):
        a = attn[h]
        r = torch.zeros_like(a)
        r[0, 0] = a[0, 0]
        r[0, 1:] = a[0, 1:][orders]
        r[1:, 0] = a[1:, 0][orders]
        r[1:, 1:] = a[1:, 1:][orders][:, orders]
        reordered[h] = r
    return reordered


# ---------------------------------------------------------------------------
# Delta attention → A matrix
# ---------------------------------------------------------------------------

def delta_from_uniform(matrix):
    """matrix: (T+1, T+1) — 减去下三角均匀基线"""
    Tp1 = matrix.size(0)
    baseline = torch.zeros_like(matrix)
    for i in range(Tp1):
        baseline[i, :i+1] = 1.0 / (i + 1)
    return matrix - baseline


def extract_A_matrix(model, seq, block_perm, inv_perm, device, dtype,
                     num_blocks=NUM_BLOCKS, block_len=BLOCK_LEN):
    """
    对一条物理顺序序列提取 A 矩阵 (N=64).

    流程:
    1. seq (物理顺序) → permute to model coords → 喂入模型
    2. mode=None, random orders → last-layer attention (reveal coords)
    3. reorder from reveal → model coords
    4. reorder from model → physical coords (via inv_perm)
    5. delta from uniform → token-to-token → block aggregation
    6. A[i,j] = cosine_sim (cross-block only, mask within-block)
    """
    T = num_blocks * block_len
    seq_tensor = torch.tensor(seq, dtype=torch.long, device=device)  # physical order

    # Permute to model-coordinate order
    seq_model = permute_sequence(seq_tensor, block_perm.to(device), block_len)  # (T,)

    # Random reveal orders
    orders = torch.randperm(T, device=device).unsqueeze(0)  # (1, T)

    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype) \
        if "cuda" in device else torch.no_grad()

    with torch.no_grad():
        with autocast_ctx:
            _, _, attn_list = model(
                seq_model.unsqueeze(0),
                mode=None,
                orders=orders,
                return_attentions=True,
            )
            last_attn = attn_list[-1]  # (1, H, T+1, T+1) — last layer, reveal coords

    H = last_attn.shape[1]

    # Step 1: reveal coords → model (ascending) coords
    attn_model = reorder_attention_from_reveal(last_attn[0], orders[0])  # (H, T+1, T+1)

    # Step 2: model coords → physical coords
    attn_phys = reorder_attention_model_to_physical(attn_model, inv_perm, block_len)  # (H, T+1, T+1)

    # Per-head block delta (cross-block only, mask within-block)
    mask_self = ~torch.eye(num_blocks, dtype=bool)
    head_profiles = []
    for h in range(H):
        dh = delta_from_uniform(attn_phys[h].to(torch.float64)).float()
        dh_tt = dh[1:, 1:]  # (T, T) token-to-token
        dh_4d = dh_tt.view(num_blocks, block_len, num_blocks, block_len)
        dh_block = dh_4d.mean(dim=(1, 3))  # (N, N)
        dh_masked = dh_block[mask_self].view(num_blocks, num_blocks - 1)  # (N, N-1)
        head_profiles.append(dh_masked)

    # Concatenate all heads: (N, H*(N-1))
    profiles = torch.cat(head_profiles, dim=-1)

    # Cosine similarity A matrix
    profiles_norm = F.normalize(profiles, dim=-1)
    A = torch.mm(profiles_norm, profiles_norm.t())  # (N, N)
    A.fill_diagonal_(0.0)

    # Block delta matrix for debug (mean over heads)
    attn_mean = attn_phys.mean(dim=0)
    delta_mean = delta_from_uniform(attn_mean.to(torch.float64)).float()
    delta_tt = delta_mean[1:, 1:]
    delta_4d = delta_tt.view(num_blocks, block_len, num_blocks, block_len)
    block_delta = delta_4d.mean(dim=(1, 3))  # (N, N)

    return A.cpu().numpy().astype(np.float32), block_delta.cpu().numpy()


# ---------------------------------------------------------------------------
# Signal check
# ---------------------------------------------------------------------------

def check_attention_signal(A_matrices):
    """检查 A 矩阵中相邻/近/远的 signal strength"""
    batch_size, N, _ = A_matrices.shape

    all_adj, all_mid, all_far = [], [], []
    for i in range(batch_size):
        A = A_matrices[i]
        adj_vals, mid_vals, far_vals = [], [], []
        for r in range(N):
            for c in range(N):
                if r == c:
                    continue
                d = abs(r - c)
                if d == 1:
                    adj_vals.append(A[r, c])
                elif d <= 4:
                    mid_vals.append(A[r, c])
                else:
                    far_vals.append(A[r, c])
        all_adj.append(np.mean(adj_vals))
        all_mid.append(np.mean(mid_vals))
        all_far.append(np.mean(far_vals))

    adj_mean, adj_std = float(np.mean(all_adj)), float(np.std(all_adj))
    mid_mean = float(np.mean(all_mid))
    far_mean, far_std = float(np.mean(all_far)), float(np.std(all_far))
    adj_far_diff = adj_mean - far_mean
    signal_ready = adj_far_diff > 0.02

    return {
        'adjacent_mean': adj_mean, 'adjacent_std': adj_std,
        'mid_mean': mid_mean,
        'far_mean': far_mean, 'far_std': far_std,
        'adj_vs_far_diff': adj_far_diff,
        'signal_ready': signal_ready,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="P0: Delta-Attention A Matrix (permuted ckpt, N=64)")
    parser.add_argument("--num_seqs", type=int, default=100)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output", type=str, default="probe_results/real_A_permuted_n64.npy")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.debug:
        args.num_seqs = 5

    print("=" * 60)
    print("P0: Delta-Attention A Matrix (permute_data=True, N=64)")
    print("=" * 60)
    print(f"  Checkpoint: {CKPT_PATH}")
    print(f"  Num blocks: {NUM_BLOCKS} × {BLOCK_LEN} = {SEQ_LEN} tokens")
    print(f"  Num sequences: {args.num_seqs}")
    print(f"  Device: {args.device}")

    # Load model + permutation
    print("\n[1/3] Loading model + permutation...")
    device = args.device
    model, block_perm, inv_perm = load_model_and_perm(CKPT_PATH, device)
    print(f"  Model: {model.config.n_layer}L/{model.config.n_head}H/{model.config.n_embd}D")
    print(f"  block_perm: {block_perm[:10].tolist()}...")
    print(f"  inv_perm:   {inv_perm[:10].tolist()}...")
    print(f"  permute_data=True, permute_mode=block, permute_seed=42")

    # Load data
    print("\n[2/3] Loading wikitext-103 test sequences...")
    sequences = load_sequences(min_seq_len=SEQ_LEN, max_sequences=args.num_seqs)
    print(f"  Loaded {len(sequences)} sequences")

    # Extract
    print(f"\n[3/3] Extracting delta-attention A matrices (N=64)...")
    dtype = torch.bfloat16

    A_list = []
    for i in tqdm(range(len(sequences)), desc="Extracting"):
        A, block_delta = extract_A_matrix(model, sequences[i], block_perm, inv_perm, device, dtype)
        A_list.append(A)

        if args.debug:
            print(f"\n{'='*70}")
            print(f"Seq {i}: A Matrix stats (N=64)")
            print(f"{'='*70}")
            # Adjacent vs mid vs far
            adj_vals, mid_vals, far_vals = [], [], []
            for r in range(NUM_BLOCKS):
                for c in range(NUM_BLOCKS):
                    if r == c: continue
                    d = abs(r - c)
                    if d == 1: adj_vals.append(A[r, c])
                    elif d <= 4: mid_vals.append(A[r, c])
                    else: far_vals.append(A[r, c])
            print(f"  Adjacent (|i-j|=1):  mean={np.mean(adj_vals):.4f}, std={np.std(adj_vals):.4f}")
            print(f"  Mid (2≤|i-j|≤4):     mean={np.mean(mid_vals):.4f}, std={np.std(mid_vals):.4f}")
            print(f"  Far (|i-j|≥5):       mean={np.mean(far_vals):.4f}, std={np.std(far_vals):.4f}")
            print(f"  Adj vs Far diff:      {np.mean(adj_vals) - np.mean(far_vals):.4f}")
            # Show top off-diagonal pairs (non-adjacent)
            pairs = []
            for r in range(NUM_BLOCKS):
                for c in range(NUM_BLOCKS):
                    if r != c and abs(r - c) > 1:
                        pairs.append((r, c, A[r, c]))
            pairs.sort(key=lambda x: -x[2])
            print(f"  Top 10 non-adjacent pairs:")
            for r, c, v in pairs[:10]:
                print(f"    ({r:2d},{c:2d}) dist={abs(r-c):2d} val={v:.4f}")

    A_matrices = np.stack(A_list, axis=0).astype(np.float32)
    print(f"\n  A_matrices shape: {A_matrices.shape}, dtype: {A_matrices.dtype}")
    print(f"  Value range: [{A_matrices.min():.4f}, {A_matrices.max():.4f}]")

    # Signal check
    stats = check_attention_signal(A_matrices)
    print(f"\n{'='*50}")
    print(f"Cold-Start Signal Monitoring (N={NUM_BLOCKS})")
    print(f"{'='*50}")
    print(f"  Adjacent (|i-j|=1):     {stats['adjacent_mean']:.4f} ± {stats['adjacent_std']:.4f}")
    print(f"  Mid (2≤|i-j|≤4):        {stats['mid_mean']:.4f}")
    print(f"  Far (|i-j|≥5):          {stats['far_mean']:.4f} ± {stats['far_std']:.4f}")
    print(f"  Adj vs Far diff:        {stats['adj_vs_far_diff']:.4f}")
    print(f"  Signal ready:           {stats['signal_ready']}")

    if not args.debug:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        np.save(args.output, A_matrices)
        print(f"\n  Saved to {args.output}")

    if stats['signal_ready']:
        print(f"\n  ✓ P0 done. N=64 A matrices ready for P1 DP solver.")
    else:
        print(f"\n  ⚠ Signal may be weak. Check debug output.")


if __name__ == "__main__":
    main()
