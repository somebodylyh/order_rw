"""Model-frame strict-65 L0 all-head attention extraction (ported).

Label-free, no physical coordinates. Node 0 = None/BOS, node 1+i = model-frame
content block i. B[source, target], zero diagonal, no edges into None.
"""

from __future__ import annotations

import numpy as np

from orderhead_v3.none_separated_block_graph import build_none_separated_B
from orderhead_v3.per_head_order_scan import _attn_to_A_block_loss_aligned_with_none_model_vec
from orderhead_v3.constants import assert_layout, N, BLOCK_LEN


def build_model_frame_strict65(
    attn_l0: np.ndarray,
    probe_orders: np.ndarray,
) -> np.ndarray:
    """Convert L0 token attention to model-frame strict-65 B graphs.

    Args:
        attn_l0: (sample, head, T+1, T+1) float array, T+1 = 257.
        probe_orders: (sample, 256) int, model-coordinate token reveal orders.

    Returns:
        B: (sample, head, 65, 65) float32. Row/col 0 = None; B[:, 0] = 0; diag = 0.
    """
    attn_l0 = np.asarray(attn_l0, dtype=np.float32)
    probe_orders = np.asarray(probe_orders, dtype=np.int64)

    if attn_l0.ndim != 4:
        raise ValueError(
            f"attn_l0 must be 4D (sample, head, T+1, T+1), got {attn_l0.shape}"
        )
    n_samples, n_heads, Tp1, _ = attn_l0.shape
    assert_layout(N, BLOCK_LEN, n_heads)
    if Tp1 != 257:
        raise ValueError(f"expected T+1=257, got {Tp1}")
    if probe_orders.shape != (n_samples, 256):
        raise ValueError(
            f"probe_orders must be ({n_samples}, 256), got {probe_orders.shape}"
        )

    output = np.empty((n_samples, n_heads, 65, 65), dtype=np.float32)
    for s in range(n_samples):
        A = _attn_to_A_block_loss_aligned_with_none_model_vec(
            attn_l0[s],            # (H, 257, 257)
            probe_orders[s],       # (256,)
        )  # → (H, 64, 65)
        for h in range(n_heads):
            output[s, h] = build_none_separated_B(A[h])  # → (65, 65)

    return output


def batch_mean_heads(B: np.ndarray, batch_mean_size: int) -> np.ndarray:
    """Group per-sample graphs into batch-mean graphs."""
    B = np.asarray(B, dtype=np.float32)
    total = B.shape[0]
    if batch_mean_size <= 0 or total % batch_mean_size != 0:
        raise ValueError(
            f"sample count {total} must be divisible by batch_mean_size {batch_mean_size}"
        )
    groups = total // batch_mean_size
    return B.reshape(groups, batch_mean_size, *B.shape[1:]).mean(axis=1)
