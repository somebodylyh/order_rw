import pathlib
import sys

import numpy as np
import pytest

_HERE = pathlib.Path(__file__).resolve().parent
_PKG = _HERE.parent
sys.path.insert(0, str(_PKG))

import quick_head_selector as qhs


def _chain_A(n=8, forward=True):
    """Physical graph A whose B=A.T encodes a clean directed chain.

    We want B[v, v+1] large (v points forward to v+1) so readiness r is high at
    low index, low at high index. Since B = A.T, that means A[v+1, v] large.
    """
    A = np.zeros((n, n), dtype=np.float64)
    for v in range(n - 1):
        if forward:
            A[v + 1, v] = 1.0      # B[v, v+1] = 1  -> v -> v+1
        else:
            A[v, v + 1] = 1.0      # B[v+1, v] = 1  -> v+1 -> v  (reversed chain)
    return A


def test_c1_raw_sign_separates_forward_vs_reversed():
    fwd = qhs.head_cheap_scores(_chain_A(8, forward=True))
    rev = qhs.head_cheap_scores(_chain_A(8, forward=False))
    # raw C1 must have OPPOSITE signs for forward vs reversed chains
    assert np.sign(fwd["C1"]) == -np.sign(rev["C1"])
    assert abs(fwd["C1"]) > 0.3 and abs(rev["C1"]) > 0.3


def test_c4_zero_for_symmetric_graph():
    A = np.array([[0.0, 1.0, 0.5],
                  [1.0, 0.0, 1.0],
                  [0.5, 1.0, 0.0]])  # symmetric -> B symmetric -> C4 ~ 0
    s = qhs.head_cheap_scores(A)
    assert s["C4"] < 1e-6


def test_c4_high_for_directed_graph():
    s = qhs.head_cheap_scores(_chain_A(8, forward=True))
    assert s["C4"] > 0.5


def test_c3_zero_for_dead_head():
    A = np.zeros((6, 6))            # no structure -> readiness flat -> C3 ~ 0
    s = qhs.head_cheap_scores(A)
    assert s["C3"] < 1e-9
    # degenerate graph must not crash C1/C2
    assert np.isfinite(s["C1"]) and np.isfinite(s["C2"])


def test_cheap_head_scores_shapes_and_chunk_mean():
    rng = np.random.default_rng(0)
    A_lh = rng.random((3, 4, 8, 8, 8))  # (n_chunks, L, H, N, N)
    out = qhs.cheap_head_scores(A_lh)
    for k in ("C1", "C2", "C3", "C4"):
        assert out[k].shape == (4, 8)
    # passing the pre-meaned (L,H,N,N) yields identical result
    out2 = qhs.cheap_head_scores(A_lh.mean(axis=0))
    assert np.allclose(out["C1"], out2["C1"])


def test_readiness_matches_cdl_convention():
    """head_cheap_scores' internal readiness must equal readiness_vector(B)."""
    from attn_order_mlp_policy import readiness_vector
    A = _chain_A(8, forward=True)
    B = A.T.copy()
    np.fill_diagonal(B, 0.0)
    r_ref = readiness_vector(B, alpha_dep=0.5)
    r_got = B.sum(axis=1) - 0.5 * B.sum(axis=0)
    assert np.allclose(r_ref, r_got)


def test_calibrate_sign_flips_when_anticorrelated():
    raw = np.array([[3.0, 1.0, -1.0, -3.0]])      # (1,4)
    tau = np.array([[-0.9, -0.3, 0.3, 0.9]])      # raw is ANTI-correlated with tau
    sign, cal = qhs.calibrate_sign(raw, tau)
    assert sign == -1.0
    from scipy.stats import spearmanr
    rho, _ = spearmanr(cal.ravel(), tau.ravel())
    assert rho > 0                                  # calibrated now positively aligned


def test_calibrate_sign_keeps_when_correlated():
    raw = np.array([[-3.0, -1.0, 1.0, 3.0]])
    tau = np.array([[-0.9, -0.3, 0.3, 0.9]])
    sign, cal = qhs.calibrate_sign(raw, tau)
    assert sign == 1.0
    assert np.allclose(cal, raw)


def test_calibrate_sign_degenerate_returns_identity():
    raw = np.zeros((2, 2))
    tau = np.array([[0.1, -0.2], [0.3, 0.4]])
    sign, cal = qhs.calibrate_sign(raw, tau)
    assert sign == 1.0
    assert np.allclose(cal, raw)


def _scores_with(c1, c3=None, c4=None):
    c1 = np.asarray(c1, dtype=np.float64)
    if c3 is None:
        c3 = np.ones_like(c1)        # all heads pass dead filter
    if c4 is None:
        c4 = np.ones_like(c1)        # all heads pass symmetry filter
    return {"C1": c1, "C2": c1.copy(), "C3": np.asarray(c3, float),
            "C4": np.asarray(c4, float)}


def test_select_heads_never_picks_argmax_abs():
    # negative head has the LARGEST magnitude; best_positive must still be the
    # positive head, NOT the |score| winner.
    scores = _scores_with([[0.4, -0.9, 0.1]])
    sel = qhs.select_heads(scores, sign=1.0, rank_score="C1",
                           dead_thresh=0.0, sym_thresh=0.0, rule="pool", k=2)
    assert sel["best_positive"] == (0, 0, 1)        # (layer, head, sign=+1)
    assert sel["best_negative"][0] == (0, 1, -1)    # strongest anti-L2R head


def test_select_heads_pool_contents():
    scores = _scores_with([[0.4, -0.9, -0.7, 0.2]])
    sel = qhs.select_heads(scores, sign=1.0, rank_score="C1",
                           dead_thresh=0.0, sym_thresh=0.0, rule="pool", k=2)
    pool_heads = {(l, h) for (l, h, _s) in sel["pool"]}
    assert (0, 0) in pool_heads                      # best_positive
    assert (0, 1) in pool_heads and (0, 2) in pool_heads   # top-2 negatives
    assert len(sel["pool"]) == 3


def test_select_heads_single_rule():
    scores = _scores_with([[0.4, -0.9, 0.1]])
    sel = qhs.select_heads(scores, sign=1.0, rank_score="C1", rule="single")
    assert sel["pool"] == [(0, 0, 1)]


def test_select_heads_masks_dead_and_symmetric():
    # head (0,0) has top C1 but is DEAD (C3=0); head (0,2) is SYMMETRIC (C4=0).
    # Only (0,1) survives the masks -> it becomes best_positive.
    scores = {"C1": np.array([[0.9, 0.5, 0.8]]),
              "C2": np.array([[0.9, 0.5, 0.8]]),
              "C3": np.array([[0.0, 1.0, 1.0]]),
              "C4": np.array([[1.0, 1.0, 0.0]])}
    sel = qhs.select_heads(scores, sign=1.0, rank_score="C1",
                           dead_thresh=1e-6, sym_thresh=1e-6, rule="pool", k=2)
    assert sel["best_positive"] == (0, 1, 1)
