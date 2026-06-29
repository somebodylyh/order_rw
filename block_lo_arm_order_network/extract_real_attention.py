"""Extract real block-level attention maps from AO-GPT checkpoint.

Key design:
- Uses validation set only (no data leakage with train ON data)
- M=3 random-order forward passes per sequence (matching mock data simulation)
- Selects top structural heads by attention variance
- Aggregates token-level attention → (16, 16) block-level A matrix
- Saves to probe_results/real_attention_val.npy

Usage:
    python extract_real_attention.py --num_seqs 100 --output probe_results/

Requires the AO-GPT checkpoint at AO-GPT-MDM/checkpoints/aogpt-small-ckpt_250000.pt
"""

from __future__ import annotations

import os
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'AO-GPT-MDM'))

import argparse
import numpy as np
import torch
from tqdm import tqdm
from datasets import Dataset
from transformers import GPT2TokenizerFast


# ── Config ───────────────────────────────────────────────────────────────────

NUM_BLOCKS = 16
BLOCK_LEN = 16
SEQ_LEN = NUM_BLOCKS * BLOCK_LEN  # 256
M_FORWARD_PASSES = 3  # random-order forward passes per sequence

AO_GPT_CKPT = os.path.expanduser(
    "~/ych/aogpt_unbiased_probe/out-wikitext103-random-attn-probe-bs256/ckpt.pt"
)

WIKITEXT_ARROW = os.path.expanduser(
    "~/.cache/huggingface/datasets/wikitext/wikitext-103-raw-v1/0.0.0/"
    "b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-test.arrow"
)

TOKENIZER_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--gpt2/snapshots/"
    "607a30d783dfa663caf39e06633721c8d4cfcd7e"
)

# ── Model loading ────────────────────────────────────────────────────────────


def load_aogpt_model(ckpt_path: str, device: str = "cuda", model_source: str = "ych"):
    """Load AO-GPT checkpoint and return (model, config_dict)."""
    if model_source == "ych":
        from aogpt_ych import AOGPT as AOGPTCollab, AOGPTConfig as AOGPTConfigCollab
        checkpoint = torch.load(ckpt_path, map_location=device)
        model_args = dict(checkpoint["model_args"])
        orig_block_size = model_args["block_size"]
        model = AOGPTCollab(AOGPTConfigCollab(**model_args))
        model_source_str = "ych"
    else:
        from model_AOGPT_AdaLN6_NoRep_cond_128_trunc_qknorm import AOGPT, AOGPTConfig
        checkpoint = torch.load(ckpt_path, map_location=device)
        model_args = dict(checkpoint["model_args"])
        orig_block_size = model_args["block_size"]
        model = AOGPT(AOGPTConfig(**model_args))
        model_source_str = "ours"

    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)

    model.load_state_dict(state_dict)
    if SEQ_LEN < orig_block_size:
        model.crop_block_size(SEQ_LEN)
    model.to(device)
    model.eval()
    return model, model_args, model_source_str


# ── Data loading ─────────────────────────────────────────────────────────────


def load_validation_sequences(min_seq_len: int = SEQ_LEN, max_sequences: int = 500):
    """Load wikitext-103 test set from Arrow cache, return tokenized sequences."""
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


# ── Structural head selection ────────────────────────────────────────────────


def select_structural_heads(
    attn_maps: np.ndarray,  # (L, H, T, T)
    top_k: int = 4,
    debug: bool = False,
) -> dict:
    """
    Select heads with highest OFF-DIAGONAL attention variance.

    Key design: variance is computed on non-diagonal token positions only,
    to avoid selecting heads that just attend to themselves or [None].

    Returns:
        dict with keys:
            head_mask: (H,) bool array
            diagnostics: per-head stats (variance, offdiag_ratio, none_ratio)
    """
    L, H, T, _ = attn_maps.shape
    T_content = T - 1  # exclude [None] token at position 0
    head_variances = np.zeros(H, dtype=np.float64)
    diag_ratios = np.zeros(H, dtype=np.float64)
    none_ratios = np.zeros(H, dtype=np.float64)

    for h in range(H):
        head_attn = attn_maps[:, h, :, :]  # (L, T, T)
        # Attention TO [None] token (column 0, rows 1:)
        none_attn = head_attn[:, 1:, 0]
        # Attention to content tokens (rows 1:, cols 1:)
        content_attn = head_attn[:, 1:, 1:]
        # Diagonal (self-attention within same token)
        diag_mask = np.eye(T_content, dtype=bool)
        diag_vals = content_attn[:, diag_mask]
        offdiag_mask = ~diag_mask
        offdiag_vals = content_attn[:, offdiag_mask]

        head_variances[h] = float(np.var(offdiag_vals))
        diag_ratios[h] = float(diag_vals.mean() / max(offdiag_vals.mean(), 1e-10))
        none_ratios[h] = float(none_attn.mean() / max(content_attn.mean(), 1e-10))

    # Composite score: reward high off-diagonal variance, penalize
    # heads where diagonal or [None]-token attention dominates.
    # Pure variance-max fails because self-attn heads and [None] sinks
    # have the highest absolute variance in their off-diagonal residuals.
    composite_score = np.zeros(H, dtype=np.float64)
    for h in range(H):
        # Penalize heads with extreme diag or none bias
        diag_penalty = max(diag_ratios[h], 1.0)  # >=1, higher = worse
        none_penalty = max(none_ratios[h], 1.0)
        composite_score[h] = head_variances[h] / (diag_penalty * none_penalty)

    top_indices = np.argsort(composite_score)[-top_k:]
    head_mask = np.zeros(H, dtype=bool)
    head_mask[top_indices] = True

    if debug:
        print(f"\n  Per-head diagnostics (L={L}, T={T}, top_k={top_k}):")
        print(f"  {'Head':>6s}  {'Variance':>12s}  {'DiagRatio':>10s}  "
              f"{'NoneRatio':>10s}  {'Composite':>12s}  {'Selected':>8s}")
        for h in range(H):
            sel = "✓" if head_mask[h] else ""
            print(f"  {h:6d}  {head_variances[h]:12.8f}  {diag_ratios[h]:10.3f}  "
                  f"{none_ratios[h]:10.3f}  {composite_score[h]:12.3e}  "
                  f"{sel:>8s}")

        print(f"\n  WARNING signs to watch:")
        if np.any(diag_ratios[head_mask] > 5.0):
            print(f"    ⚠ High diag ratio in selected heads (self-attention bias)")
        if np.any(none_ratios[head_mask] > 10.0):
            print(f"    ⚠ High [None] ratio in selected heads (sink collapse)")
        if head_variances.max() < 1e-6:
            print(f"    ⚠ Very low variance overall (uniform attention?)")
    else:
        print(f"  Selected heads (top-{top_k} by composite score): "
              f"{top_indices.tolist()}")
        print(f"  Head variances: min={head_variances.min():.6f} "
              f"median={np.median(head_variances):.6f} max={head_variances.max():.6f}")
        print(f"  Selected diag ratios: "
              f"{[f'{diag_ratios[h]:.1f}' for h in top_indices]}")
        print(f"  Selected none ratios: "
              f"{[f'{none_ratios[h]:.1f}' for h in top_indices]}")

    return {
        'head_mask': head_mask,
        'diagnostics': {
            'variances': head_variances,
            'diag_ratios': diag_ratios,
            'none_ratios': none_ratios,
            'composite_scores': composite_score,
            'selected': top_indices,
        }
    }


# ── Token → Block aggregation ───────────────────────────────────────────────


def aggregate_to_block_attention(
    attn_maps: np.ndarray,    # (L, H, T+1, T+1) or (H, T+1, T+1)
    head_mask: np.ndarray,    # (H,) bool
    num_blocks: int = NUM_BLOCKS,
    block_len: int = BLOCK_LEN,
) -> np.ndarray:
    """
    Aggregate token-level attention to block-level (N, N) matrix.

    Uses raw attention values. Diagonal (self-attention) is zeroed out.
    [None] token (position 0) is retained as a column — it carries
    structural signal in AO-GPT (per TiT approach).
    """
    if attn_maps.ndim == 4:
        # (L, H, T+1, T+1) → average over layers
        attn = attn_maps[:, head_mask, :, :]  # (L, H_sel, T+1, T+1)
        attn = attn.mean(axis=(0, 1))  # (T+1, T+1)
    else:
        attn = attn_maps[head_mask, :, :]  # (H_sel, T+1, T+1)
        attn = attn.mean(axis=0)  # (T+1, T+1)

    # Keep [None] token (position 0) in the key dimension — it carries signal.
    # Remove [None] from query dimension (row 0) since [None] doesn't predict.
    attn_tok = attn[1:, :]  # (T, T+1) — T content queries × (T+1) keys including [None]

    # Reshape: (N, block_len, T+1) → (N, block_len, T+1)
    # We average content tokens within each block for the query dimension,
    # and keep all key positions (including [None]) for block aggregation.
    attn_blocks_q = attn_tok.reshape(num_blocks, block_len, -1)  # (N, block_len, T+1)

    # Split key dimension into [None] + content blocks
    none_col = attn_blocks_q[:, :, 0:1]  # (N, block_len, 1) — attention TO [None]
    content_keys = attn_blocks_q[:, :, 1:]  # (N, block_len, T)
    content_keys_blocked = content_keys.reshape(num_blocks, block_len, num_blocks, block_len)
    content_keys_blocked = content_keys_blocked.mean(axis=(1, 3))  # (N, N)

    # [None]→block attention: average over query tokens in each block
    none_block = none_col.mean(axis=1).squeeze(-1)  # (N,)

    # Combine: block→block attention + broadcast [None]→block across rows
    A_block = content_keys_blocked.astype(np.float32)
    # Add [None] signal as a column bias (same for all query rows)
    A_block += none_block[np.newaxis, :] * 0.1  # scale down [None] contribution
    A_block = A_block.astype(np.float32)

    # Zero out diagonal
    np.fill_diagonal(A_block, 0.0)

    return A_block


# ── Simulate sparse extraction (matching mock data simulation) ───────────────


def extract_attention_for_sequence(
    model,
    block_seqs: torch.Tensor,  # (N, block_len) int64 tokens
    device: str,
    model_source: str = "ych",
    num_blocks: int = NUM_BLOCKS,
    M: int = M_FORWARD_PASSES,
    debug: bool = False,
) -> tuple:
    """
    Run M random-order forward passes through AO-GPT, extract + aggregate
    attention maps into a (num_blocks, num_blocks) A matrix.

    Returns:
        A: (num_blocks, num_blocks) float32 block-level attention matrix.
        diag_info: dict with head diagnostics (only first pass, for debug).
    """
    N, blen = block_seqs.shape
    all_A = np.zeros((M, N, N), dtype=np.float32)
    diag_info = {}

    rng = np.random.default_rng()
    for m in range(M):
        order = rng.permutation(N)  # random reveal order
        revealed_tokens = block_seqs[order].flatten()  # (N * blen,)
        reveal_order = torch.arange(N * blen, device=device)

        with torch.no_grad():
            if model_source == "ych":
                # ych model: use return_probe_data + return_all_attentions
                # to get attention from every transformer layer.
                _, _, probe_data = model(
                    revealed_tokens.unsqueeze(0),
                    mode=None,
                    orders=reveal_order.unsqueeze(0),
                    return_probe_data=True,
                    return_all_attentions=True,
                )
                # all_attentions: list[L] of (1, H, T+1, T+1)
                attn_stack = torch.stack(probe_data["all_attentions"]).squeeze(1).cpu().numpy()
            else:
                _, _, attn_outputs = model(
                    revealed_tokens.unsqueeze(0),
                    mode=None,
                    orders=reveal_order.unsqueeze(0),
                    return_attentions=True,
                )
                attn_stack = torch.stack(attn_outputs).squeeze(1).cpu().numpy()  # (L, H, T+1, T+1)

        # Select structural heads from first pass
        if m == 0:
            result = select_structural_heads(attn_stack, top_k=4, debug=debug)
            head_mask = result['head_mask']
            diag_info = result['diagnostics']

        # Aggregate to block level
        A_block = aggregate_to_block_attention(
            attn_stack, head_mask, num_blocks=N, block_len=blen,
        )

        # Remap from reveal-order coordinates to physical coordinates
        A_phys = np.zeros((N, N), dtype=np.float32)
        for qi in range(N):
            pi = order[qi]
            for qj in range(N):
                pj = order[qj]
                A_phys[pi, pj] = A_block[qi, qj]

        all_A[m] = A_phys

    # Average over M passes
    final_A = all_A.mean(axis=0)
    np.fill_diagonal(final_A, 0.0)
    final_A = np.clip(final_A, a_min=0.0, a_max=None)

    return final_A.astype(np.float32), diag_info


# ── Global normalization ─────────────────────────────────────────────────────


def normalize_attention_scale(A_matrices: np.ndarray) -> np.ndarray:
    """
    Scale up small attention values for numerical stability.
    Real model attention can be in the 0.001 range.

    Performs per-matrix min-max scaling to [0, 1], then multiples by a
    fixed factor so Q(σ) values are in a reasonable range.
    """
    batch_size, N, _ = A_matrices.shape
    scaled = np.zeros_like(A_matrices)
    for i in range(batch_size):
        A = A_matrices[i].copy()
        A_max = A.max()
        if A_max > 0:
            A = A / A_max  # scale to [0, 1]
        scaled[i] = A
    return scaled


# ── Main extraction loop ─────────────────────────────────────────────────────


def extract_real_attention_dataset(
    model,
    sequences: list,
    device: str,
    model_source: str = "ych",
    num_blocks: int = NUM_BLOCKS,
    block_len: int = BLOCK_LEN,
    M: int = M_FORWARD_PASSES,
    output_path: str = "probe_results/real_attention_val.npy",
    debug_batch: int = 0,
) -> np.ndarray:
    """
    Extract block-level attention matrices for all validation sequences.

    Args:
        debug_batch: if > 0, print detailed diagnostics for first N sequences
                     and exit without saving.

    Returns:
        A_matrices: (num_sequences, num_blocks, num_blocks) float32.
    """
    n_seqs = len(sequences)
    if debug_batch > 0:
        n_seqs = min(n_seqs, debug_batch)
    A_list = []

    for i in tqdm(range(n_seqs), desc="Extracting attention"):
        seq = sequences[i]
        block_seqs = torch.tensor(seq, dtype=torch.long, device=device).view(num_blocks, block_len)
        debug_this = (debug_batch > 0)
        A, diag_info = extract_attention_for_sequence(
            model, block_seqs, device, model_source=model_source,
            num_blocks=num_blocks, M=M, debug=debug_this,
        )

        if debug_this:
            print(f"\n{'='*60}")
            print(f"Sequence {i}: block-level A matrix ({num_blocks}x{num_blocks})")
            print(f"{'='*60}")
            # Print matrix with 3 decimal precision
            for r in range(num_blocks):
                row_str = "  ".join(f"{A[r, c]:6.3f}" for c in range(num_blocks))
                print(f"  B{r:2d}: {row_str}")

            # Off-diagonal vs diagonal block ratio
            diag_block = np.trace(A)
            offdiag_sum = A.sum() - diag_block
            n_pairs = num_blocks * num_blocks - num_blocks
            print(f"\n  Diag (self-block) mean:    {diag_block / num_blocks:.6f}")
            print(f"  Off-diag mean:             {offdiag_sum / max(n_pairs, 1):.6f}")
            print(f"  Off-diag / Diag ratio:     {offdiag_sum / max(diag_block, 1e-10):.3f}")
            print(f"  Value range: [{A.min():.6f}, {A.max():.6f}]")

            if diag_info:
                print(f"\n  Head selection quality:")
                sel = diag_info['selected']
                for h in sel:
                    print(f"    H{h}: var={diag_info['variances'][h]:.8f}  "
                          f"diag_ratio={diag_info['diag_ratios'][h]:.3f}  "
                          f"none_ratio={diag_info['none_ratios'][h]:.3f}")

        A_list.append(A)

    A_matrices = np.stack(A_list, axis=0)

    if debug_batch > 0:
        print(f"\n  Debug mode: processed {n_seqs} sequences, not saving.")
        return A_matrices

    A_matrices = normalize_attention_scale(A_matrices)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.save(output_path, A_matrices)
    print(f"\nSaved {n_seqs} attention matrices to {output_path}")
    print(f"  Shape: {A_matrices.shape}, dtype: {A_matrices.dtype}")
    print(f"  Value range: [{A_matrices.min():.6f}, {A_matrices.max():.6f}]")

    return A_matrices


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Extract real AO-GPT attention maps")
    parser.add_argument("--num_seqs", type=int, default=200,
                        help="Number of validation sequences (default: 200)")
    parser.add_argument("--debug_batch", type=int, default=0,
                        help="If >0, print per-sequence A matrix + head diag for "
                             "first N sequences and exit without saving")
    parser.add_argument("--output", type=str, default="probe_results/real_attention_val.npy")
    parser.add_argument("--ckpt", type=str, default=AO_GPT_CKPT)
    parser.add_argument("--model_source", type=str, default="ych",
                        choices=["ours", "ych"])
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--arrow", type=str, default=WIKITEXT_ARROW)
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Real Attention Extraction from AO-GPT")
    print("=" * 60)
    print(f"  Checkpoint: {args.ckpt}")
    print(f"  Num blocks: {NUM_BLOCKS}, block_len: {BLOCK_LEN}, seq_len: {SEQ_LEN}")
    print(f"  Forward passes per seq (M): {M_FORWARD_PASSES}")
    print(f"  Num sequences: {args.num_seqs}")
    print(f"  Device: {args.device}")
    print(f"  Debug batch: {args.debug_batch if args.debug_batch > 0 else 'off'}")

    # Load model
    print("\n[1/3] Loading AO-GPT model...")
    model, model_args, model_source = load_aogpt_model(
        args.ckpt, device=args.device, model_source=args.model_source,
    )
    print(f"  Model args: {model_args}")
    print(f"  Model source: {model_source}")

    # Load data
    print("\n[2/3] Loading validation data...")
    sequences = load_validation_sequences(
        min_seq_len=SEQ_LEN, max_sequences=args.num_seqs,
    )
    print(f"  Loaded {len(sequences)} sequences (len >= {SEQ_LEN})")

    # Extract attention
    print("\n[3/3] Extracting attention maps...")
    A_matrices = extract_real_attention_dataset(
        model, sequences, args.device,
        model_source=model_source,
        num_blocks=NUM_BLOCKS, block_len=BLOCK_LEN, M=M_FORWARD_PASSES,
        output_path=args.output,
        debug_batch=args.debug_batch,
    )

    # Quick signal check
    from attention_extractor import check_attention_signal, print_signal_report
    stats = check_attention_signal(A_matrices)
    print_signal_report(stats)

    if not stats['signal_ready']:
        print("\n  WARNING: Signal not ready! Adjacent-block signal may be weak.")
        print("  Consider: (1) more sequences, (2) different head selection, "
              "(3) per-layer instead of per-head selection.")


if __name__ == '__main__':
    main()
