"""P0: Attention map extraction from AOGPT + mock data generation.

Key design decisions:
- Mock data gives adjacent blocks MUCH higher attention (0.5 vs 0.01)
  to exaggerate signal for initial debugging, per reviewer advice.
- Unobserved pairs are filled with global mean (not 0) to avoid
  biasing the DP solver away from unexplored paths.
- Cold-start monitoring checks if the A matrices have enough structure
  before proceeding to DP.
"""

import numpy as np
import torch
from typing import Tuple, Optional

from config import Config


# ── Mock attention generation ──────────────────────────────────────────────

def mock_attention_matrix(num_blocks: int = 16,
                          adj_mean: float = 0.5,
                          nonadj_mean: float = 0.01,
                          noise_std: float = 0.02,
                          rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Generate a synthetic block-level attention matrix with strong
    adjacent-block signal for debugging.

    A[i, j] = attention strength of block i (query/later) to block j (key/earlier).
    The matrix is NOT symmetric because attention is directional.

    Adjacent block pairs get adj_mean + noise; non-adjacent get nonadj_mean + noise.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    A = rng.normal(loc=nonadj_mean, scale=noise_std, size=(num_blocks, num_blocks))
    # Boost adjacent pairs (both directions)
    for i in range(num_blocks - 1):
        A[i, i + 1] += (adj_mean - nonadj_mean)
        A[i + 1, i] += (adj_mean - nonadj_mean)
    A = np.clip(A, a_min=0.0, a_max=None)
    np.fill_diagonal(A, 0.0)  # self-connection = 0
    return A.astype(np.float32)


def generate_mock_dataset(num_sequences: int,
                          num_blocks: int = 16,
                          seed: int = 42) -> np.ndarray:
    """
    Generate a batch of synthetic A matrices.

    Returns:
        A_matrices: (num_sequences, num_blocks, num_blocks) float32
    """
    rng = np.random.default_rng(seed)
    A_list = []
    cfg = Config()
    for i in range(num_sequences):
        seq_rng = np.random.default_rng(seed + i + 1)
        A = mock_attention_matrix(
            num_blocks=num_blocks,
            adj_mean=cfg.mock_adj_mean,
            nonadj_mean=cfg.mock_nonadj_mean,
            noise_std=cfg.mock_noise_std,
            rng=seq_rng,
        )
        A_list.append(A)
    return np.stack(A_list, axis=0)


# ── Sparse attention aggregation (simulating M forward passes) ─────────────

def simulate_sparse_extraction(A_true: np.ndarray,
                                M: int = 3,
                                seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate the sparse extraction of A through M random-order forward passes.

    In a real setup, each forward pass only reveals attention values for pairs
    where the query block appears AFTER the key block in the causal ordering.
    This function simulates that sparsity.

    Args:
        A_true: (num_blocks, num_blocks) true attention matrix.
        M: number of random-order forward passes.
        seed: random seed for reproducibility.

    Returns:
        final_A: (num_blocks, num_blocks) reconstructed A after sparse filling.
        visibility_mask: (num_blocks, num_blocks) bool, which pairs were observed.
    """
    N = A_true.shape[0]
    rng = np.random.default_rng(seed)
    accumulated = np.zeros((N, N), dtype=np.float64)
    counts = np.zeros((N, N), dtype=np.int32)

    for _ in range(M):
        order = rng.permutation(N)
        # In each forward pass, block at position pos_i can attend to blocks
        # at positions < pos_i in this ordering.
        for pos_i in range(N):
            qi = order[pos_i]  # query block (physical index)
            for pos_j in range(pos_i):
                kj = order[pos_j]  # key block (physical index)
                accumulated[qi, kj] += A_true[qi, kj]
                counts[qi, kj] += 1

    # Compute final A with global-mean fill for unobserved pairs
    observed_mask = counts > 0
    diag_mask = ~np.eye(N, dtype=bool)
    observed_nondiag = observed_mask & diag_mask

    if observed_nondiag.sum() > 0:
        global_mean = accumulated[observed_nondiag].sum() / counts[observed_nondiag].sum()
    else:
        global_mean = 0.0

    safe_counts = np.where(observed_mask, counts, 1)
    final_A = np.where(observed_mask, accumulated / safe_counts, global_mean)
    np.fill_diagonal(final_A, 0.0)

    return final_A.astype(np.float32), observed_mask


def simulate_sparse_extraction_batch(A_matrices: np.ndarray,
                                      M: int = 3,
                                      seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Batched version of simulate_sparse_extraction.

    Args:
        A_matrices: (num_sequences, N, N) true attention matrices.
        M: number of random-order forward passes per sequence.

    Returns:
        final_A_batch: (num_sequences, N, N) reconstructed A matrices.
        visibility_batch: (num_sequences, N, N) bool visibility masks.
    """
    num_seq, N, _ = A_matrices.shape
    final_batch = np.zeros_like(A_matrices)
    vis_batch = np.zeros((num_seq, N, N), dtype=bool)
    for i in range(num_seq):
        final_batch[i], vis_batch[i] = simulate_sparse_extraction(
            A_matrices[i], M=M, seed=seed + i
        )
    return final_batch, vis_batch


# ── Cold-start monitoring ──────────────────────────────────────────────────

def check_attention_signal(A_matrices: np.ndarray) -> dict:
    """
    Check if A matrices have sufficient structural signal (non-uniform).

    Args:
        A_matrices: (batch_size, N, N) attention matrices.

    Returns:
        dict with keys: variance, delta_from_uniform, adjacent_vs_nonadjacent, signal_ready.
    """
    batch_size, N, _ = A_matrices.shape
    diag_mask = ~np.eye(N, dtype=bool)

    # Per-sequence statistics
    variances = []
    deltas = []
    adj_diffs = []

    uniform_val = 1.0 / N

    for i in range(batch_size):
        A = A_matrices[i]
        nondiag = A[diag_mask]
        variances.append(float(np.var(nondiag)))
        deltas.append(float(np.mean(np.abs(nondiag - uniform_val))))

        # adjacent vs non-adjacent
        adj_vals = []
        nonadj_vals = []
        for r in range(N):
            for c in range(N):
                if r == c:
                    continue
                if abs(r - c) == 1:
                    adj_vals.append(A[r, c])
                else:
                    nonadj_vals.append(A[r, c])
        adj_mean = np.mean(adj_vals) if adj_vals else 0.0
        nonadj_mean = np.mean(nonadj_vals) if nonadj_vals else 0.0
        adj_diffs.append(adj_mean - nonadj_mean)

    mean_delta = float(np.mean(deltas))
    mean_adj_diff = float(np.mean(adj_diffs))
    signal_ready = mean_delta > Config.signal_threshold

    return {
        'variance': float(np.mean(variances)),
        'delta_from_uniform': mean_delta,
        'adjacent_vs_nonadjacent': mean_adj_diff,
        'signal_ready': signal_ready,
    }


def print_signal_report(stats: dict) -> None:
    """Pretty-print the cold-start monitoring report."""
    print("=" * 50)
    print("Cold-Start Attention Signal Monitoring")
    print("=" * 50)
    print(f"  Variance:              {stats['variance']:.6f}")
    print(f"  Delta from uniform:    {stats['delta_from_uniform']:.6f}  (threshold: {Config.signal_threshold})")
    print(f"  Adjacent vs non-adj:   {stats['adjacent_vs_nonadjacent']:.6f}")
    print(f"  Signal ready:          {stats['signal_ready']}")
    print("=" * 50)
