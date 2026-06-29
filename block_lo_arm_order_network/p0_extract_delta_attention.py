"""
P0: 从 AO-GPT checkpoint 提取 delta-attention-based A 矩阵。

A[i, j] = cosine similarity of block i and j's cross-block delta attention profiles.
信号源: delta token-to-token attention (masked within-block) → physical coords

用法:
    python p0_extract_delta_attention.py --num_seqs 100 --output probe_results/real_A_delta_n16.npy
    python p0_extract_delta_attention.py --num_seqs 5 --debug  # 调试模式
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
# 配置: N=16 blocks × 16 tokens = 256 seq_len
# ---------------------------------------------------------------------------
NUM_BLOCKS = 16
BLOCK_LEN = 16
SEQ_LEN = NUM_BLOCKS * BLOCK_LEN  # 256

CKPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..',
    'AO-GPT-MDM', 'checkpoints', 'aogpt-small-ckpt_250000.pt'
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

def load_model(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_args = dict(checkpoint["model_args"])
    model = AOGPT(AOGPTConfig(**model_args))
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.crop_block_size(SEQ_LEN)  # 256
    model.to(device)
    model.eval()
    return model


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
# Delta attention → A matrix
# ---------------------------------------------------------------------------

def reorder_to_physical(attention, orders):
    """
    attention: (H, T+1, T+1) — single sample, last layer, shuffled coords
    orders: (T,) — shuffle order
    → (H, T+1, T+1) in physical (original document) coordinates
    """
    H = attention.shape[0]
    reordered = torch.zeros_like(attention)
    for h in range(H):
        a = attention[h]  # (T+1, T+1)
        r = torch.zeros_like(a)
        r[0, 0] = a[0, 0]                       # [None]→[None]
        r[0, 1:] = a[0, 1:][orders]              # [None]→tokens
        r[1:, 0] = a[1:, 0][orders]              # tokens→[None]
        r[1:, 1:] = a[1:, 1:][orders][:, orders] # tokens→tokens
        reordered[h] = r
    return reordered


def delta_from_uniform(matrix):
    """matrix: (T+1, T+1) — 减去下三角均匀基线"""
    Tp1 = matrix.size(0)
    baseline = torch.zeros_like(matrix)
    for i in range(Tp1):
        baseline[i, :i+1] = 1.0 / (i + 1)
    return matrix - baseline


def extract_A_matrix(model, seq, device, dtype, num_blocks=NUM_BLOCKS, block_len=BLOCK_LEN):
    """
    对一条序列提取 delta-attention-based A 矩阵。

    用随机 orders (mode=None, 自己生成以便逆映射) 跑模型 → last-layer attention,
    reorder 回物理坐标 → delta-from-uniform → token-to-token → block aggregation
    → cross-block cosine sim.
    """
    block_seqs = torch.tensor(seq, dtype=torch.long, device=device).view(num_blocks, block_len)
    T = num_blocks * block_len

    # 随机 orders —— 确保每个 block 都有机会 attend 其他 block
    orders = torch.randperm(T, device=device).unsqueeze(0)  # (1, T)

    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype) \
        if "cuda" in device else torch.no_grad()

    with torch.no_grad():
        with autocast_ctx:
            _, _, attn_list = model(
                block_seqs.flatten().unsqueeze(0),
                mode=None,
                orders=orders,
                return_attentions=True,
            )
            # attn_list: list of L tensors, each (1, H, T+1, T+1)
            last_attn = attn_list[-1]  # (1, H, T+1, T+1) — last layer

    H = last_attn.shape[1]

    # Reorder from shuffled coords to physical coords using known orders
    attn_phys = reorder_to_physical(last_attn[0], orders[0])  # (H, T+1, T+1)

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
# Cold-start monitoring
# ---------------------------------------------------------------------------

def check_attention_signal(A_matrices):
    """
    检查 A 矩阵中相邻/近/远的 signal strength。
    返回 stats dict，含 signal_ready 布尔值。
    """
    batch_size, N, _ = A_matrices.shape

    all_adj, all_near, all_far = [], [], []
    for i in range(batch_size):
        A = A_matrices[i]
        adj_vals, near_vals, far_vals = [], [], []
        for r in range(N):
            for c in range(N):
                if r == c: continue
                d = abs(r - c)
                if d == 1: adj_vals.append(A[r, c])
                elif d <= 3: near_vals.append(A[r, c])
                else: far_vals.append(A[r, c])
        all_adj.append(np.mean(adj_vals))
        all_near.append(np.mean(near_vals))
        all_far.append(np.mean(far_vals))

    adj_mean, adj_std = float(np.mean(all_adj)), float(np.std(all_adj))
    near_mean = float(np.mean(all_near))
    far_mean, far_std = float(np.mean(all_far)), float(np.std(all_far))
    adj_far_diff = adj_mean - far_mean
    signal_ready = adj_far_diff > 0.05

    return {
        'adjacent_mean': adj_mean, 'adjacent_std': adj_std,
        'near_mean': near_mean,
        'far_mean': far_mean, 'far_std': far_std,
        'adj_vs_far_diff': adj_far_diff,
        'signal_ready': signal_ready,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="P0: Delta-Attention A Matrix Extraction")
    parser.add_argument("--num_seqs", type=int, default=100, help="Number of sequences")
    parser.add_argument("--debug", action="store_true", help="Debug mode: 5 seqs, print matrices")
    parser.add_argument("--output", type=str, default="probe_results/real_A_delta.npy")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.debug:
        args.num_seqs = 5

    print("=" * 60)
    print("P0: Delta-Attention A Matrix Extraction")
    print("=" * 60)
    print(f"  Checkpoint: {CKPT_PATH}")
    print(f"  Num blocks: {NUM_BLOCKS} × {BLOCK_LEN} = {SEQ_LEN} tokens")
    print(f"  Num sequences: {args.num_seqs}")
    print(f"  Device: {args.device}")
    print(f"  Debug: {args.debug}")

    # Load model
    print("\n[1/3] Loading model...")
    device = args.device
    model = load_model(CKPT_PATH, device)
    print(f"  Model: {model.config.n_layer}L/{model.config.n_head}H/{model.config.n_embd}D")

    # Load data
    print("\n[2/3] Loading wikitext-103 test sequences...")
    sequences = load_sequences(min_seq_len=SEQ_LEN, max_sequences=args.num_seqs)
    print(f"  Loaded {len(sequences)} sequences")

    # Extract
    print(f"\n[3/3] Extracting delta-attention A matrices...")
    dtype = torch.bfloat16

    A_list = []
    for i in tqdm(range(len(sequences)), desc="Extracting"):
        A, block_delta = extract_A_matrix(model, sequences[i], device, dtype)
        A_list.append(A)

        if args.debug:
            print(f"\n{'='*60}")
            print(f"Sequence {i}: A Matrix ({NUM_BLOCKS}x{NUM_BLOCKS}, cosine similarity)")
            print(f"{'='*60}")
            # Print A matrix with heatmap-like formatting (show all 16 blocks compactly)
            for r in range(NUM_BLOCKS):
                vals = " ".join(f"{A[r, c]:7.2f}" for c in range(NUM_BLOCKS))
                print(f"  B{r:2d}: {vals}")
            adj_vals = [A[r, r+1] for r in range(NUM_BLOCKS-1)]
            nonadj_vals = []
            far_vals = []
            for r in range(NUM_BLOCKS):
                for c in range(NUM_BLOCKS):
                    if r == c: continue
                    if abs(r-c) == 1: pass  # already in adj_vals
                    elif abs(r-c) <= 3: nonadj_vals.append(A[r, c])
                    else: far_vals.append(A[r, c])
            print(f"  Adjacent (|i-j|=1):      mean={np.mean(adj_vals):.4f}")
            print(f"  Near (2≤|i-j|≤3):        mean={np.mean(nonadj_vals):.4f}")
            print(f"  Far (|i-j|≥4):           mean={np.mean(far_vals):.4f}")
            print(f"  Adj vs Far diff:          {np.mean(adj_vals) - np.mean(far_vals):.4f}")

    A_matrices = np.stack(A_list, axis=0).astype(np.float32)
    print(f"\n  A_matrices shape: {A_matrices.shape}, dtype: {A_matrices.dtype}")
    print(f"  Value range: [{A_matrices.min():.4f}, {A_matrices.max():.4f}]")

    # Signal check
    stats = check_attention_signal(A_matrices)
    print(f"\n{'='*50}")
    print(f"Cold-Start Signal Monitoring (N={NUM_BLOCKS})")
    print(f"{'='*50}")
    print(f"  Adjacent (|i-j|=1):     {stats['adjacent_mean']:.4f} ± {stats['adjacent_std']:.4f}")
    print(f"  Near (2≤|i-j|≤3):       {stats['near_mean']:.4f}")
    print(f"  Far (|i-j|≥4):          {stats['far_mean']:.4f} ± {stats['far_std']:.4f}")
    print(f"  Adj vs Far diff:        {stats['adj_vs_far_diff']:.4f}")
    print(f"  Signal ready:           {stats['signal_ready']}")

    if not args.debug:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        np.save(args.output, A_matrices)
        print(f"\n  Saved to {args.output}")

    if stats['signal_ready']:
        print(f"\n  ✓ P0 done. A matrices ready for P1 DP solver.")
    else:
        print(f"\n  ⚠ Adjacent signal weak — may need more sequences or different extraction.")


if __name__ == "__main__":
    main()
