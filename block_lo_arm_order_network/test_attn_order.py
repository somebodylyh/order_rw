"""TDD tests for the dynamic attention-order feature builder (Task 1) and the
unified C-D+L teacher (Task 2).

Run: pytest block_lo_arm_order_network/test_attn_order.py -q
"""
import numpy as np
import pytest

from attn_order_features import build_features, FEATURE_NAMES
from attn_order_teacher import teacher_step, teacher_scores, rollout_order
from attn_order_diagnostics import (
    forward_reverse_ratio, free_running_agreement, start_end_distribution, locality_stats,
)


# A fixed 4x4 directed graph with zero diagonal for hand-computed checks.
#   B[u, v] = edge u -> v
B4 = np.array(
    [
        [0.0, 1.0, 2.0, 3.0],
        [4.0, 0.0, 5.0, 6.0],
        [7.0, 8.0, 0.0, 9.0],
        [10.0, 11.0, 12.0, 0.0],
    ],
    dtype=np.float64,
)


# --------------------------------------------------------------------------
# Task 1 — feature builder
# --------------------------------------------------------------------------

def test_shape_and_feature_names():
    X, cand, names = build_features(B4, S_t=[0], U_t=[1, 2, 3], last=0, t=1, N=4)
    assert X.shape == (3, 12)
    assert list(cand) == [1, 2, 3]
    assert len(names) == 12
    assert names == FEATURE_NAMES
    assert not np.isnan(X).any()


def test_numeric_values_mid_rollout():
    # S={0}, U={1,2,3}, last=0, t=1, N=4 -- all 12 features hand-computed.
    X, cand, _ = build_features(B4, S_t=[0], U_t=[1, 2, 3], last=0, t=1, N=4)
    expected = {
        1: [1, 1, 4, 4, 9.5, 11, 5.5, 6, 1, 4, 0.25, 0.75],
        2: [2, 2, 7, 7, 8.5, 12, 8.5, 9, 2, 7, 0.25, 0.75],
        3: [3, 3, 10, 10, 7.5, 9, 11.5, 12, 3, 10, 0.25, 0.75],
    }
    for row, v in enumerate(cand):
        np.testing.assert_allclose(X[row], expected[v], rtol=0, atol=1e-9,
                                   err_msg=f"candidate {v}")


def test_t0_empty_selected_and_no_last():
    # t=0: S empty, last=None, U = all nodes.
    X, cand, names = build_features(B4, S_t=[], U_t=[0, 1, 2, 3], last=None, t=0, N=4)
    assert X.shape == (4, 12)
    # features 1-4 (selected-set) must be zero when S empty
    assert np.all(X[:, 0:4] == 0.0)
    # features 9,10 (last) must be zero when last is None
    assert np.all(X[:, 8:10] == 0.0)
    # progress: t/N = 0, |U|/N = 1
    assert np.all(X[:, 10] == 0.0)
    assert np.all(X[:, 11] == 1.0)
    # feature 5 for v=0: mean over {1,2,3} of B[u,0] = mean(4,7,10) = 7
    row0 = list(cand).index(0)
    assert X[row0, 4] == pytest.approx(7.0)


def test_single_candidate_unselected_set_empty():
    # |U|=1 -> U\{v} empty -> features 5-8 must be zero.
    X, cand, _ = build_features(B4, S_t=[0, 2, 3], U_t=[1], last=3, t=3, N=4)
    assert X.shape == (1, 12)
    assert np.all(X[:, 4:8] == 0.0)
    # but last features still populated: B[3,1]=11, B[1,3]=6
    assert X[0, 8] == pytest.approx(11.0)
    assert X[0, 9] == pytest.approx(6.0)


def test_random_graph_no_nan_various_steps():
    rng = np.random.default_rng(0)
    N = 8
    B = rng.random((N, N))
    np.fill_diagonal(B, 0.0)
    perm = rng.permutation(N)
    for t in range(N - 1):
        S = perm[:t].tolist()
        U = perm[t:].tolist()
        last = int(perm[t - 1]) if t > 0 else None
        X, cand, _ = build_features(B, S, U, last, t, N)
        assert X.shape == (len(U), 12)
        assert not np.isnan(X).any()
        assert list(cand) == U


# --------------------------------------------------------------------------
# Task 2 — unified C-D+L teacher
# --------------------------------------------------------------------------

def test_teacher_scores_main():
    # q(v) = C(v) - D(v) + L(v) with S={0}, U={1,2,3}, last=0
    q, cand = teacher_scores(B4, S_t=[0], U_t=[1, 2, 3], last=0, mode="C-D+L")
    # C=[1,2,3], D=[9.5,8.5,7.5], L=[1,2,3] => q=[-7.5,-4.5,-1.5]
    np.testing.assert_allclose(q, [-7.5, -4.5, -1.5], atol=1e-9)
    assert list(cand) == [1, 2, 3]


def test_teacher_step_probs_and_entropy():
    out = teacher_step(B4, S_t=[0], U_t=[1, 2, 3], last=0, tau_T=1.0, mode="C-D+L")
    p = out["probs"]
    assert p.shape == (3,)
    assert p.sum() == pytest.approx(1.0)
    # q=[-7.5,-4.5,-1.5] -> highest prob on candidate 3 (last position)
    assert np.argmax(p) == 2
    # entropy in nats, must match -sum p log p
    assert out["entropy"] == pytest.approx(float(-(p * np.log(p)).sum()))
    # top-k candidates ranked by score (top-1 should be candidate 3)
    assert out["topk_candidates"][0] == 3


def test_teacher_ablation_modes_differ():
    kw = dict(B=B4, S_t=[0], U_t=[1, 2, 3], last=0)
    q_c, _ = teacher_scores(mode="C", **kw)
    q_l, _ = teacher_scores(mode="L", **kw)
    q_cd, _ = teacher_scores(mode="C-D", **kw)
    q_cl, _ = teacher_scores(mode="C+L", **kw)
    q_main, _ = teacher_scores(mode="C-D+L", **kw)
    np.testing.assert_allclose(q_c, [1, 2, 3], atol=1e-9)          # C only
    np.testing.assert_allclose(q_l, [1, 2, 3], atol=1e-9)          # L only (B[0,v])
    np.testing.assert_allclose(q_cd, [1 - 9.5, 2 - 8.5, 3 - 7.5], atol=1e-9)
    np.testing.assert_allclose(q_cl, [2, 4, 6], atol=1e-9)         # C+L
    np.testing.assert_allclose(q_main, [-7.5, -4.5, -1.5], atol=1e-9)


def test_teacher_edge_cases_empty_S_and_no_last():
    # S empty -> C=0; last None -> L=0; q = -D
    q, cand = teacher_scores(B4, S_t=[], U_t=[0, 1, 2, 3], last=None, mode="C-D+L")
    # D(0) = mean over {1,2,3} of B[u,0] = mean(4,7,10) = 7
    row0 = list(cand).index(0)
    assert q[row0] == pytest.approx(-7.0)


# --------------------------------------------------------------------------
# rollout (shared by Phase 0 / Phase 1)
# --------------------------------------------------------------------------

def test_rollout_returns_valid_permutation():
    rng = np.random.default_rng(1)
    N = 16
    B = rng.random((N, N)); np.fill_diagonal(B, 0.0)
    order = rollout_order(B, tau_T=1.0, seed=3, mode="C-D+L")
    assert order.shape == (N,)
    assert sorted(order.tolist()) == list(range(N))   # a true permutation


def test_rollout_greedy_is_deterministic():
    rng = np.random.default_rng(2)
    N = 16
    B = rng.random((N, N)); np.fill_diagonal(B, 0.0)
    o1 = rollout_order(B, seed=0, mode="C-D+L", greedy=True)
    o2 = rollout_order(B, seed=999, mode="C-D+L", greedy=True)
    np.testing.assert_array_equal(o1, o2)   # greedy ignores seed


def test_rollout_entropy_reported_per_step():
    rng = np.random.default_rng(4)
    N = 8
    B = rng.random((N, N)); np.fill_diagonal(B, 0.0)
    order, ent = rollout_order(B, tau_T=1.0, seed=0, return_entropy=True)
    assert sorted(order.tolist()) == list(range(N))
    # entropy recorded for the steps that had >1 candidate (t=0..N-2)
    assert len(ent) == N - 1
    assert all(e >= 0.0 for e in ent)


# --------------------------------------------------------------------------
# Phase 1.5 orientation diagnostics
# --------------------------------------------------------------------------

def test_forward_reverse_ratio_pure_directions():
    N = 8
    fwd = np.arange(N)[None, :]
    rev = np.arange(N)[::-1][None, :]
    f, r, a = forward_reverse_ratio(fwd)
    assert (f, r) == (1.0, 0.0) and a == pytest.approx(1.0)
    f, r, a = forward_reverse_ratio(rev)
    assert (f, r) == (0.0, 1.0) and a == pytest.approx(1.0)
    # a 50/50 mix -> abs_tau still ~1 but fwd=rev=0.5
    mix = np.stack([np.arange(N), np.arange(N)[::-1]])
    f, r, a = forward_reverse_ratio(mix)
    assert f == pytest.approx(0.5) and r == pytest.approx(0.5) and a == pytest.approx(1.0)


def test_free_running_agreement_teacher_equivalent_is_perfect():
    # teacher q = C - D + L = feature[0] - feature[4] + feature[8] EXACTLY.
    # A student that scores with that exact combination must agree 100% along its own path.
    rng = np.random.default_rng(7)
    N = 8
    B = rng.random((N, N)); np.fill_diagonal(B, 0.0)
    teacher_equiv = lambda X: X[:, 0] - X[:, 4] + X[:, 8]
    t1, t4 = free_running_agreement(B, teacher_equiv, N, k=4)
    assert t1 == pytest.approx(1.0) and t4 == pytest.approx(1.0)


def test_free_running_agreement_antiteacher_is_low():
    rng = np.random.default_rng(8)
    N = 8
    B = rng.random((N, N)); np.fill_diagonal(B, 0.0)
    anti = lambda X: -(X[:, 0] - X[:, 4] + X[:, 8])
    t1, _ = free_running_agreement(B, anti, N, k=4)
    assert t1 < 0.5   # picks (near) the teacher's least-preferred candidate


def test_start_end_distribution_counts():
    orders = np.array([[3, 1, 2, 0], [3, 0, 1, 2], [5, 2, 1, 0]])
    d = start_end_distribution(orders)
    assert d["top_start"][0][0] == 3        # node 3 starts 2/3 of orders
    assert d["start_entropy"] >= 0.0


def test_locality_stats_raster_grid():
    # single raster order on 8x8 grid: consecutive steps are mostly +1 col (dist 1),
    # with a larger jump at each row wrap. P(d<=1) should be high but < 1.
    raster = np.arange(64)[None, :]
    s = locality_stats(raster)
    assert 0.0 <= s["p_le1"] <= 1.0 and s["mean_manh"] > 0.0
