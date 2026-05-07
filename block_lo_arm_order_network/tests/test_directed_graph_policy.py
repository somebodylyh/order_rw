"""Tests for directed_graph_policy.py — pure NumPy stochastic order policy."""

import numpy as np
import pytest

from directed_graph_policy import (
    _softmax,
    build_directed_graph,
    compute_source,
    pagerank,
    progressive_rw_step,
    sample_order,
    sample_orders,
)

# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def A_global():
    rng = np.random.default_rng(42)
    A = rng.random((64, 64)).astype(np.float32) * 0.1
    np.fill_diagonal(A, 0.0)
    return A


@pytest.fixture
def B(A_global):
    return build_directed_graph(A_global)


# ──────────────────────────────────────────────────────────────────────
# Test 1: B construction
# ──────────────────────────────────────────────────────────────────────


def test_B_construction(A_global, B):
    """B[i,j] == A_global[j,i] for i!=j, and B[i,i] == 0.0 for all i."""
    N = A_global.shape[0]
    for i in range(N):
        assert B[i, i] == 0.0, f"B[{i},{i}] should be 0.0"
        for j in range(N):
            if i != j:
                assert B[i, j] == pytest.approx(A_global[j, i]), (
                    f"B[{i},{j}]={B[i,j]} != A_global[{j},{i}]={A_global[j,i]}"
                )


# ──────────────────────────────────────────────────────────────────────
# Test 2: progressive_rw always returns valid permutations
# ──────────────────────────────────────────────────────────────────────


def test_progressive_rw_legal(B):
    """500 sampled orders from progressive_rw must all be valid permutations."""
    params = {
        'beta_sup': 1.0,
        'beta_fut': 0.5,
        'beta_src': 0.2,
        'beta_loc': 0.5,
        'tau_start': 1.0,
        'tau_step': 1.0,
    }

    result = sample_orders(B, 'progressive_rw', params, K=500, seed_base=1)
    orders = result['orders']
    N = B.shape[0]

    for k, order in enumerate(orders):
        assert len(set(order)) == N, f"order[{k}] has duplicate elements"
        assert order.min() >= 0, f"order[{k}] min < 0"
        assert order.max() < N, f"order[{k}] max >= N"


# ──────────────────────────────────────────────────────────────────────
# Test 3: progressive_rw is seedable
# ──────────────────────────────────────────────────────────────────────


def test_progressive_rw_seedable(B):
    """Same seed → bit-identical output."""
    params = {
        'beta_sup': 1.0,
        'beta_fut': 0.5,
        'beta_src': 0.2,
        'beta_loc': 0.5,
        'tau_start': 1.0,
        'tau_step': 1.0,
    }

    order1, lp1 = sample_order(B, 'progressive_rw', params, seed=12345)
    order2, lp2 = sample_order(B, 'progressive_rw', params, seed=12345)

    np.testing.assert_array_equal(order1, order2)
    assert lp1 == pytest.approx(lp2)


# ──────────────────────────────────────────────────────────────────────
# Test 4: tie-break with no identity bias
# ──────────────────────────────────────────────────────────────────────


def test_tie_break_no_id_bias():
    """All-zero B → first node distribution should be near-uniform (no index bias).

    Uses progressive_rw with K=5000 orders. Chi-squared test: expected = K/64
    per node, assert chi^2 < 150.
    """
    N = 64
    B_zero = np.zeros((N, N), dtype=np.float64)
    params = {
        'beta_sup': 1.0,
        'beta_fut': 0.5,
        'beta_src': 0.2,
        'beta_loc': 0.5,
        'tau_start': 1.0,
        'tau_step': 1.0,
    }

    K = 5000
    result = sample_orders(B_zero, 'progressive_rw', params, K=K, seed_base=2)
    first_nodes = result['orders'][:, 0]

    counts = np.bincount(first_nodes, minlength=N)
    expected = K / N
    chi2 = np.sum((counts - expected) ** 2 / expected)
    assert chi2 < 150, f"chi2={chi2:.2f} >= 150, first-node distribution is biased"


# ──────────────────────────────────────────────────────────────────────
# Test 5: PageRank converges
# ──────────────────────────────────────────────────────────────────────


def test_pagerank_converges(B):
    """PageRank on random B with uniform q should return (N,) array summing to ~1.0.

    Second call should give the same result (deterministic).
    """
    N = B.shape[0]
    q = np.ones(N, dtype=np.float64) / N

    r1 = pagerank(B, q)
    r2 = pagerank(B, q)

    assert r1.shape == (N,)
    assert r1.sum() == pytest.approx(1.0, rel=1e-5)
    np.testing.assert_array_almost_equal(r1, r2)


# ──────────────────────────────────────────────────────────────────────
# Test 6: logprob consistency
# ──────────────────────────────────────────────────────────────────────


def test_logprob_consistency(B):
    """Sample one order with seed=77777 then recompute logprob step-by-step.

    The recomputed logprob must match the original.
    """
    params = {
        'beta_sup': 1.0,
        'beta_fut': 0.5,
        'beta_src': 0.2,
        'beta_loc': 0.5,
        'tau_start': 1.0,
        'tau_step': 1.0,
    }
    seed = 77777

    order, logprob = sample_order(B, 'progressive_rw', params, seed=seed)

    # Replay step-by-step using the same policy functions
    rng = np.random.default_rng(seed)
    N = B.shape[0]

    tau_start = float(params.get('tau_start', 1.0))
    tau_step = float(params.get('tau_step', 1.0))
    alpha_dep = float(params.get('alpha_dep', 0.5))

    source, _, _ = compute_source(B, alpha_dep)

    # Step 0
    p0 = _softmax(source, tau_start, rng)
    idx0 = order[0]
    replay_lp = float(np.log(max(p0[idx0], 1e-300)))

    S = np.array([idx0], dtype=np.int64)
    U = np.setdiff1d(np.arange(N, dtype=np.int64), S)
    last = idx0

    betas = {
        'sup': float(params.get('beta_sup', 1.0)),
        'fut': float(params.get('beta_fut', 0.5)),
        'src': float(params.get('beta_src', 0.2)),
        'loc': float(params.get('beta_loc', 0.5)),
    }

    for t in range(1, N):
        p_t, _ = progressive_rw_step(B, S, U, last, betas, tau_step, source, rng)
        node_t = order[t]
        # Find index of node_t in U
        idx_t = int(np.where(U == node_t)[0][0])
        replay_lp += float(np.log(max(p_t[idx_t], 1e-300)))

        S = np.append(S, node_t)
        U = U[U != node_t]
        last = node_t

    assert replay_lp == pytest.approx(logprob, rel=1e-5), (
        f"Replayed logprob {replay_lp:.6f} != original {logprob:.6f}"
    )


# ──────────────────────────────────────────────────────────────────────
# Test 7: sample_one vs sample_orders
# ──────────────────────────────────────────────────────────────────────


def test_sample_one_vs_K(B):
    """sample_order(seed=S) matches sample_orders(K=N, seed_base=SB)[orders][i]
    when S == SB*10000 + i."""
    params = {
        'beta_sup': 1.0,
        'beta_fut': 0.5,
        'beta_src': 0.2,
        'beta_loc': 0.5,
        'tau_start': 1.0,
        'tau_step': 1.0,
    }

    seed_base = 5
    K = 10  # enough to have an index 3

    batch = sample_orders(B, 'progressive_rw', params, K=K, seed_base=seed_base)
    i = 3
    expected_order = batch['orders'][i]
    expected_lp = batch['logprobs'][i]

    order_single, lp_single = sample_order(
        B, 'progressive_rw', params, seed=seed_base * 10000 + i
    )

    np.testing.assert_array_equal(order_single, expected_order)
    assert lp_single == pytest.approx(expected_lp)
