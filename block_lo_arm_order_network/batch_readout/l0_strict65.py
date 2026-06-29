"""Model-frame strict-65 L0 all-head attention extraction.

Label-free, no physical coordinates.  This module deliberately does NOT accept
``inv_perm``, ``clean_perm``, or ``block_perm`` — the extraction stays entirely
in the model's own coordinate frame.  Physical remapping is only applied posthoc
when translating the final sigma_model → sigma_phys for scoring against physical
L2R, and that translation lives outside this module.

Node convention (strict-65):
  node 0      = None / BOS
  node 1 + i  = model-frame content block i

The returned B is a 65×65 directed graph where B[source, target] follows the
same convention as ``none_separated_block_graph.build_none_separated_B``.
"""

from __future__ import annotations

import numpy as np

from none_separated_block_graph import build_none_separated_B
from per_head_order_scan import _attn_to_A_block_loss_aligned_with_none_model_vec


def build_model_frame_strict65(
    attn_l0: np.ndarray,
    probe_orders: np.ndarray,
) -> np.ndarray:
    """Convert L0 token attention to model-frame strict-65 B graphs.

    Uses the loss-aligned attention slice (attn[:T, :T]), keeps model-block
    labels without any inv_perm / physical remap, then builds the 65-node
    None-separated block graph via ``build_none_separated_B``.

    Args:
        attn_l0: (sample, head, T+1, T+1) float array.
            T = 256 (SEQ_LEN), so shape is (S, H, 257, 257).
        probe_orders: (sample, T) int array, model-coordinate token reveal
            orders.  Each row is a permutation of [0, T).

    Returns:
        B: (sample, head, 65, 65) float32.  Row/col 0 is None node.
           B[0, 1:] = None → content edges; B[1:, 1:] = content → content.
           B[:, 0] = 0 (no edges into None).  Diagonal is zero.
    """
    attn_l0 = np.asarray(attn_l0, dtype=np.float32)
    probe_orders = np.asarray(probe_orders, dtype=np.int64)

    if attn_l0.ndim != 4:
        raise ValueError(
            f"attn_l0 must be 4D (sample, head, T+1, T+1), got {attn_l0.shape}"
        )
    n_samples, n_heads, Tp1, _ = attn_l0.shape
    if Tp1 != 257:
        raise ValueError(f"expected T+1=257, got {Tp1}")
    if probe_orders.shape != (n_samples, 256):
        raise ValueError(
            f"probe_orders must be ({n_samples}, 256), got {probe_orders.shape}"
        )

    output = np.empty((n_samples, n_heads, 65, 65), dtype=np.float32)
    for s in range(n_samples):
        # _attn_to_A_block_loss_aligned_with_none_model_vec vectorises over
        # arbitrary leading dims, so we can pass all H heads at once.
        A = _attn_to_A_block_loss_aligned_with_none_model_vec(
            attn_l0[s],            # (H, 257, 257)
            probe_orders[s],       # (256,)
        )  # → (H, 64, 65)  — columns: [None], model_block_0..63
        for h in range(n_heads):
            output[s, h] = build_none_separated_B(A[h])  # → (65, 65)

    return output


def batch_mean_heads(B: np.ndarray, batch_mean_size: int) -> np.ndarray:
    """Group per-sample graphs into batch-mean graphs.

    Args:
        B: (total_samples, head, 65, 65) float array.
        batch_mean_size: number of per-sample graphs averaged into one
            batch-mean graph.  ``total_samples`` must be divisible by
            ``batch_mean_size``.

    Returns:
        B_mean: (total_samples // batch_mean_size, head, 65, 65) float32.
    """
    B = np.asarray(B, dtype=np.float32)
    total = B.shape[0]
    if batch_mean_size <= 0 or total % batch_mean_size != 0:
        raise ValueError(
            f"sample count {total} must be divisible by "
            f"batch_mean_size {batch_mean_size}"
        )
    groups = total // batch_mean_size
    return B.reshape(groups, batch_mean_size, *B.shape[1:]).mean(axis=1)
