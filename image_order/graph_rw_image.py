"""Graph-RW v2 wrapper for image patches — N-agnostic policy with image-specific aux stats."""
import os
import sys
import numpy as np
import torch
from typing import Dict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "block_lo_arm_order_network"))

from directed_graph_policy import build_directed_graph, sample_order, sample_orders, compute_source

POLICY = "progressive_rw"

IMAGE_RW_PARAMS_DEFAULT = dict(
    tau_start=0.10, tau_step=0.10,
    alpha_dep=0.5,  alpha_pr=0.85,
    beta_sup=1.0,   beta_fut=0.5,
    beta_src=0.2,   beta_loc=0.5,
    top_k=4,        epsilon_uniform=0.0,
)


def make_B_from_attention(A_global: np.ndarray) -> np.ndarray:
    """Build directed graph B from attention matrix A_global.

    Zeros the diagonal of A_global first, then returns build_directed_graph(A_global).
    """
    assert isinstance(A_global, np.ndarray), "A_global must be numpy array"
    assert A_global.ndim == 2, f"A_global must be 2-D, got shape {A_global.shape}"
    assert A_global.shape[0] == A_global.shape[1], f"A_global must be square, got {A_global.shape}"

    A_copy = A_global.astype(np.float64).copy()
    np.fill_diagonal(A_copy, 0.0)
    B = build_directed_graph(A_copy)

    assert np.allclose(np.diag(B), 0.0), "B diagonal should be zero"
    return B


def sample_image_orders_batch(
    B: np.ndarray,
    params: Dict[str, float],
    batch_size: int,
    seed_base: int,
    step: int,
    device=None,
) -> torch.LongTensor:
    """Sample batch_size orders using deterministic seeding.

    Loops sample_order with seed = seed_base * 100000 + step * batch_size + b.
    Returns (batch_size, N) torch.long tensor, optionally moved to device.
    """
    N = B.shape[0]
    orders = np.zeros((batch_size, N), dtype=np.int64)

    for b in range(batch_size):
        seed = seed_base * 100000 + step * batch_size + b
        order, _ = sample_order(B, POLICY, params, seed)
        orders[b] = order

    orders_t = torch.from_numpy(orders).long()
    if device is not None:
        orders_t = orders_t.to(device)

    return orders_t


def mean_manhattan_step(orders: np.ndarray, grid: int = 8) -> float:
    """Compute mean Manhattan distance between consecutive patches on grid × grid lattice.

    orders: (K, N) int array.
    Converts each patch index to (r, c) = (idx // grid, idx % grid).
    Returns mean distance across all K*(N-1) transitions.
    """
    orders = np.asarray(orders, dtype=np.int64)
    K, N = orders.shape

    distances = []
    for k in range(K):
        for t in range(N - 1):
            idx_t = int(orders[k, t])
            idx_next = int(orders[k, t + 1])

            r_t = idx_t // grid
            c_t = idx_t % grid
            r_next = idx_next // grid
            c_next = idx_next % grid

            dist = abs(r_t - r_next) + abs(c_t - c_next)
            distances.append(dist)

    return float(np.mean(distances)) if distances else 0.0


def sample_orders_for_eval(
    B: np.ndarray,
    params: Dict[str, float],
    K: int,
    seed_base: int = 42,
) -> dict:
    """Sample K orders and return diagnostics dict augmented with mean_manhattan_step.

    Calls sample_orders(B, POLICY, params, K, seed_base) and adds 'mean_manhattan_step' key.
    """
    result = sample_orders(B, POLICY, params, K, seed_base)
    result['mean_manhattan_step'] = mean_manhattan_step(result['orders'])
    return result


def random_orders(K: int, N: int, seed_base: int = 42) -> np.ndarray:
    """Generate K random permutations of N elements.

    Returns (K, N) int array; each row is seeded by np.random.default_rng(seed_base * 10000 + k).
    """
    orders = np.zeros((K, N), dtype=np.int64)
    for k in range(K):
        rng = np.random.default_rng(seed_base * 10000 + k)
        orders[k] = rng.permutation(N)
    return orders


def raster_orders(K: int, N: int) -> np.ndarray:
    """Generate K copies of the raster (left-to-right, top-to-bottom) scan order.

    Returns (K, N) int array; every row = np.arange(N).
    """
    return np.tile(np.arange(N, dtype=np.int64), (K, 1))


if __name__ == "__main__":
    # Smoke test
    np.random.seed(0)
    A_global = np.random.RandomState(0).rand(64, 64).astype(np.float32)

    # Test make_B_from_attention
    B = make_B_from_attention(A_global)
    assert B.shape == (64, 64), f"Expected B shape (64, 64), got {B.shape}"
    assert np.allclose(np.diag(B), 0.0), "B diagonal should be zero"

    # Test sample_image_orders_batch
    orders_t = sample_image_orders_batch(B, IMAGE_RW_PARAMS_DEFAULT, batch_size=8, seed_base=42, step=0)
    assert orders_t.shape == (8, 64), f"Expected shape (8, 64), got {orders_t.shape}"
    assert orders_t.dtype == torch.long, f"Expected torch.long, got {orders_t.dtype}"
    for b in range(8):
        assert sorted(orders_t[b].numpy().tolist()) == list(range(64)), f"Row {b} is not a valid permutation"

    # Test sample_orders_for_eval
    result = sample_orders_for_eval(B, IMAGE_RW_PARAMS_DEFAULT, K=200, seed_base=42)
    assert result['legal_rate'] == 1.0, f"Expected legal_rate=1.0, got {result['legal_rate']}"
    assert result['orders'].shape == (200, 64), f"Expected orders shape (200, 64), got {result['orders'].shape}"

    # Compute baselines
    manhattan_rw = result['mean_manhattan_step']
    manhattan_random = mean_manhattan_step(random_orders(200, 64))
    manhattan_raster = mean_manhattan_step(raster_orders(200, 64))

    # Print results
    tau_vs_l2r_mean = result['tau_vs_l2r_mean']
    pairwise_tau_mean = result['pairwise_tau_mean']

    print(f"OK graph_rw_image: K=200, top_k=4, manhattan_rw={manhattan_rw:.3f}, manhattan_random={manhattan_random:.3f}, manhattan_raster={manhattan_raster:.3f}")
