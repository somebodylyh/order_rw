"""Tests for label-free dynamic CDL consensus teacher.

Covers: rank conversion, standardized rollout margin, structure-preserving
destroy, destroyed gap, rank-based agreement, headwise z-scoring, dynamic
teacher weights, and soft pairwise teacher construction.

All tests are label-free: no physical coordinates, inv_perm, block_perm,
or L2R labels are used.
"""

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))

from batch_readout.label_free_cdl_teacher import (  # noqa: E402
    build_dynamic_teacher,
    cdl_rollout_with_standardized_margin,
    compute_label_free_quality,
    destroy_strict65,
    destroyed_gap,
    dynamic_teacher_weights,
    headwise_zscore,
    order_to_rank,
    rank_agreement,
    soft_pairwise_teacher,
)
from none_separated_block_graph import build_none_separated_B  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic strict-65 helpers
# ---------------------------------------------------------------------------

def _make_chain_65(chain_strength=1.0):
    """Create a 65×65 B where block i → block i+1 is strong (chain structure).

    Node 0 is None; content blocks form a clear directed chain 0→1→2→...→63.
    """
    B = np.zeros((65, 65), dtype=np.float64)
    # None → block 0
    B[0, 1] = chain_strength
    # Chain: block i → block i+1
    for i in range(1, 64):
        B[i, i + 1] = chain_strength
    return B


def _make_synthetic_B_heads(H=8, seed=0):
    """Create H synthetic strict-65 graphs with varied structure."""
    rng = np.random.default_rng(seed)
    B_heads = np.zeros((H, 65, 65), dtype=np.float64)
    for h in range(H):
        # Base: random attention
        B = rng.random((65, 65)).astype(np.float64) * 0.1
        # Inject a per-head chain with different orientation
        shift = h % 64
        for i in range(1, 65):
            j = 1 + (i - 1 + shift) % 64
            if i != j:
                B[i, j] += 0.5
        # Inject None→content signal
        B[0, 1 + h % 8] += 1.0
        # Zero diagonal and content-to-None column
        np.fill_diagonal(B, 0.0)
        B[1:, 0] = 0.0
        B_heads[h] = B
    return B_heads


# ============================================================================
# order_to_rank
# ============================================================================

def test_order_to_rank_is_inverse_permutation():
    order = np.array([2, 0, 3, 1], dtype=np.int64)
    expected = np.array([1, 3, 0, 2], dtype=np.int64)
    np.testing.assert_array_equal(order_to_rank(order), expected)


def test_order_to_rank_roundtrip():
    rng = np.random.default_rng(42)
    for _ in range(10):
        order = rng.permutation(64)
        rank = order_to_rank(order)
        # rank[order[t]] == t
        np.testing.assert_array_equal(rank[order], np.arange(64))
        # order[np.argsort(rank)] == order... equivalent to the above
        recovered = np.argsort(rank)
        np.testing.assert_array_equal(recovered, order)


# ============================================================================
# CDL rollout with standardized margin
# ============================================================================

def test_cdl_rollout_on_chain_recovers_identity_order():
    B = _make_chain_65(chain_strength=5.0)
    res = cdl_rollout_with_standardized_margin(B)
    # A perfect chain should produce order [0, 1, 2, ..., 63].
    np.testing.assert_array_equal(res["content_order"], np.arange(64))
    np.testing.assert_array_equal(res["rank"], np.arange(64))


def test_standardized_margin_is_scale_invariant():
    B = _make_chain_65(chain_strength=1.0)
    res1 = cdl_rollout_with_standardized_margin(B)
    res100 = cdl_rollout_with_standardized_margin(B * 100.0)
    assert res1["avg_margin"] == pytest.approx(res100["avg_margin"], rel=1e-5)


def test_standardized_margin_handles_zero_variance():
    """When all candidate utilities are identical, margin must be 0, not NaN."""
    B = np.zeros((65, 65), dtype=np.float64)
    # All utilities will be 0 at every step → std=0 → margin=0.
    res = cdl_rollout_with_standardized_margin(B)
    assert res["avg_margin"] == 0.0
    assert not np.isnan(res["avg_margin"])
    # All per-step margins should be 0.
    assert all(m == 0.0 for m in res["margins"])


def test_cdl_rollout_output_is_content_blocks_0_to_63():
    B = _make_chain_65()
    res = cdl_rollout_with_standardized_margin(B)
    order = res["content_order"]
    assert order.shape == (64,)
    assert order.dtype == np.int64
    assert set(order.tolist()) == set(range(64))
    assert 0 not in (order + 1)  # None (node 0) never appears


def test_cdl_rollout_rejects_wrong_shape():
    with pytest.raises(ValueError):
        cdl_rollout_with_standardized_margin(np.zeros((64, 64)))


# ============================================================================
# Structure-preserving destroy
# ============================================================================

def test_destroy_preserves_none_column_bit_identical():
    """Content-to-None column B[1:, 0] must be preserved byte-for-byte."""
    rng = np.random.default_rng(7)
    B = rng.random((65, 65))
    np.fill_diagonal(B, 0.0)
    # Set content-to-None column to KNOWN non-zero values (violating
    # strict-65 convention on purpose, to test preservation).
    B[1:, 0] = rng.random(64)

    Bd = destroy_strict65(B, np.random.default_rng(99))
    np.testing.assert_array_equal(Bd[1:, 0], B[1:, 0])


def test_destroy_preserves_row_value_multisets():
    """Each row's set of values (excluding preserved column and diagonal)
    must be preserved — only the assignment to columns is shuffled."""
    rng = np.random.default_rng(3)
    B = rng.random((65, 65))
    np.fill_diagonal(B, 0.0)
    B[1:, 0] = 0.0  # strict-65 convention

    Bd = destroy_strict65(B, np.random.default_rng(42))

    # None→content row: value set preserved
    np.testing.assert_allclose(
        np.sort(Bd[0, 1:]), np.sort(B[0, 1:]), atol=1e-12
    )

    # Content→content rows: value set preserved (excluding diagonal position)
    for i in range(1, 65):
        di = i - 1  # self-index in content-to-content slice
        row_orig = B[i, 1:].copy()
        row_dest = Bd[i, 1:].copy()
        # Drop the diagonal entries (both should be zero).
        mask = np.ones(64, dtype=bool)
        mask[di] = False
        np.testing.assert_allclose(
            np.sort(row_dest[mask]), np.sort(row_orig[mask]), atol=1e-12
        )
        # Diagonal must be zero.
        assert row_dest[di] == 0.0


def test_destroy_preserves_zero_diagonal():
    rng = np.random.default_rng(5)
    B = rng.random((65, 65))
    np.fill_diagonal(B, 0.0)
    B[1:, 0] = 0.0
    Bd = destroy_strict65(B, np.random.default_rng(11))
    np.testing.assert_allclose(np.diag(Bd), 0.0, atol=1e-15)


def test_destroy_actually_destroys_block_identity():
    """Destroy must CHANGE the graph — it's not a no-op."""
    rng = np.random.default_rng(1)
    B = rng.random((65, 65))
    np.fill_diagonal(B, 0.0)
    B[1:, 0] = 0.0
    # Inject a strong asymmetric edge so destroy changes it.
    B[5, 10] = 100.0

    Bd = destroy_strict65(B, np.random.default_rng(12345))
    # The specific block identity (B[5,10]) should be destroyed.
    assert not np.allclose(Bd, B, atol=1e-7), (
        "destroy must change the graph; block identity should be shuffled"
    )


def test_destroy_is_deterministic_given_rng():
    B = _make_chain_65()
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    np.testing.assert_allclose(
        destroy_strict65(B, rng1), destroy_strict65(B, rng2), atol=1e-15
    )


# ============================================================================
# Destroyed gap
# ============================================================================

def test_destroyed_gap_positive_for_structured_graph():
    """Real margin should exceed destroyed margin for a graph with actual
    structure (chain)."""
    B = _make_chain_65(chain_strength=3.0)
    gap = destroyed_gap(B, base_seed=0, n_replicas=3)
    assert gap > 0.0, f"destroyed_gap should be positive for structured B, got {gap:.4f}"


def test_destroyed_gap_near_zero_for_noise():
    """For pure noise (no structure), gap should be near zero."""
    rng = np.random.default_rng(2)
    B = rng.random((65, 65))
    np.fill_diagonal(B, 0.0)
    B[1:, 0] = 0.0
    gap = destroyed_gap(B, base_seed=0, n_replicas=5)
    # Gap may be small positive or negative; it should be close to zero.
    assert abs(gap) < 1.0, f"destroyed_gap for noise should be small, got {gap:.4f}"


# ============================================================================
# Rank agreement
# ============================================================================

def test_rank_agreement_uses_precedence_not_order_values():
    """Two heads with different order VALUES but identical precedence
    (both produce the same ordering) must have agreement = 1.0."""
    ranks = np.array([
        [0, 1, 2, 3],
        [0, 1, 2, 3],
    ], dtype=np.int64)
    # Pad to 64 columns to match expected shape.
    ranks64 = np.zeros((2, 64), dtype=np.int64)
    ranks64[:, :4] = ranks
    for h in range(2):
        ranks64[h, 4:] = np.arange(4, 64)
    ag = rank_agreement(ranks64)
    assert ag[0] == pytest.approx(1.0, abs=1e-6)
    assert ag[1] == pytest.approx(1.0, abs=1e-6)


def test_rank_agreement_different_from_order_agreement():
    """Demonstrate that rank-based agreement correctly captures precedence:
    two heads that disagree on order should have tau < 1.0 on rank vectors."""
    # Head 0: reveal blocks in order [0, 1, 2, 3]
    order0 = np.array([0, 1, 2, 3], dtype=np.int64)
    # Head 1: reveal blocks in order [2, 1, 0, 3] — different order
    order1 = np.array([2, 1, 0, 3], dtype=np.int64)

    rank0 = order_to_rank(order0)  # [0, 1, 2, 3]
    rank1 = order_to_rank(order1)  # [2, 1, 0, 3]

    ranks64 = np.zeros((2, 64), dtype=np.int64)
    ranks64[0, :4] = rank0
    ranks64[1, :4] = rank1
    for h in range(2):
        ranks64[h, 4:] = np.arange(4, 64)
    ag = rank_agreement(ranks64)
    # They disagree, so agreement should be < 1.0.
    assert ag[0] < 1.0, f"agreement should be < 1.0 for disagreeing heads, got {ag[0]}"


def test_rank_agreement_perfect_antialignment():
    """Two heads with perfectly opposite orders should have tau ≈ -1.0."""
    order0 = np.arange(64, dtype=np.int64)
    order1 = np.arange(63, -1, -1, dtype=np.int64)  # reversed
    rank0 = order_to_rank(order0)
    rank1 = order_to_rank(order1)
    ranks = np.stack([rank0, rank1])
    ag = rank_agreement(ranks)
    # Each head sees the other as perfectly reversed → mean tau ≈ -1.0.
    assert ag[0] == pytest.approx(-1.0, abs=0.05)
    assert ag[1] == pytest.approx(-1.0, abs=0.05)


# ============================================================================
# Headwise z-score
# ============================================================================

def test_headwise_zscore_zero_mean_unit_std():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float64)
    z = headwise_zscore(values)
    assert z.mean() == pytest.approx(0.0, abs=1e-9)
    assert z.std(ddof=0) == pytest.approx(1.0, abs=1e-9)


def test_headwise_zscore_constant_input_returns_zeros():
    values = np.full(8, 3.14, dtype=np.float64)
    z = headwise_zscore(values)
    np.testing.assert_allclose(z, 0.0, atol=1e-12)


# ============================================================================
# Dynamic teacher weights
# ============================================================================

def test_dynamic_teacher_weights_sum_to_one():
    rng = np.random.default_rng(0)
    for _ in range(20):
        quality = rng.normal(size=8).astype(np.float64)
        w = dynamic_teacher_weights(quality, temperature=1.0, smoothing=0.05)
        assert w.sum() == pytest.approx(1.0, abs=1e-12)


def test_dynamic_teacher_weights_all_nonnegative():
    quality = np.array([-10.0, -5.0, 0.0, 5.0, 10.0, 3.0, -3.0, 0.0])
    w = dynamic_teacher_weights(quality)
    assert (w >= 0.0).all()


def test_dynamic_teacher_weights_uniform_when_quality_equal():
    quality = np.ones(8, dtype=np.float64)
    w = dynamic_teacher_weights(quality, temperature=1.0, smoothing=0.0)
    np.testing.assert_allclose(w, np.full(8, 1.0 / 8), atol=1e-12)


def test_dynamic_teacher_weights_smoothing_floor():
    """Even with extreme quality differences, each weight ≥ smoothing / H."""
    quality = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    w = dynamic_teacher_weights(quality, temperature=1.0, smoothing=0.05)
    assert (w >= 0.05 / 8 - 1e-12).all()


# ============================================================================
# Soft pairwise teacher
# ============================================================================

def test_soft_pairwise_teacher_antisymmetric():
    """Y[i,j] + Y[j,i] must equal 1 for all i ≠ j."""
    rng = np.random.default_rng(1)
    for _ in range(5):
        ranks = np.zeros((8, 64), dtype=np.int64)
        for h in range(8):
            ranks[h] = rng.permutation(64)
        # Use equal weights for simplicity
        w = np.full(8, 1.0 / 8, dtype=np.float64)
        Y = soft_pairwise_teacher(ranks, w)
        assert Y.shape == (64, 64)
        np.testing.assert_allclose(Y + Y.T, np.ones((64, 64)), atol=1e-10)


def test_soft_pairwise_teacher_diagonal_is_half():
    ranks = np.zeros((2, 64), dtype=np.int64)
    ranks[0] = np.arange(64)
    ranks[1] = np.arange(63, -1, -1)
    w = np.array([0.5, 0.5], dtype=np.float64)
    Y = soft_pairwise_teacher(ranks, w)
    np.testing.assert_allclose(np.diag(Y), 0.5, atol=1e-12)


def test_soft_pairwise_teacher_values_in_0_to_1():
    rng = np.random.default_rng(3)
    ranks = np.zeros((8, 64), dtype=np.int64)
    for h in range(8):
        ranks[h] = rng.permutation(64)
    w = rng.random(8).astype(np.float64)
    w = w / w.sum()
    Y = soft_pairwise_teacher(ranks, w)
    assert Y.min() >= 0.0
    assert Y.max() <= 1.0


def test_soft_pairwise_teacher_consistent_with_consensus():
    """When all heads agree on order 0..63, Y[i,j] must be:
      1.0 if i < j, 0.0 if i > j, 0.5 if i == j."""
    ranks = np.tile(np.arange(64, dtype=np.int64), (8, 1))
    w = np.full(8, 1.0 / 8, dtype=np.float64)
    Y = soft_pairwise_teacher(ranks, w)
    for i in range(64):
        for j in range(64):
            if i < j:
                assert Y[i, j] == pytest.approx(1.0, abs=1e-10)
            elif i > j:
                assert Y[i, j] == pytest.approx(0.0, abs=1e-10)
            else:
                assert Y[i, j] == pytest.approx(0.5, abs=1e-10)


# ============================================================================
# compute_label_free_quality — end-to-end per-sample
# ============================================================================

def test_compute_label_free_quality_output_shapes():
    B_heads = _make_synthetic_B_heads(H=8, seed=0)
    q = compute_label_free_quality(B_heads, destroy_seed=1, n_destroy_replicas=2)
    H = 8
    assert q["orders"].shape == (H, 64)
    assert q["ranks"].shape == (H, 64)
    assert q["margin"].shape == (H,)
    assert q["destroyed_gap"].shape == (H,)
    assert q["agreement"].shape == (H,)
    assert q["quality"].shape == (H,)


def test_compute_label_free_quality_no_nan():
    B_heads = _make_synthetic_B_heads(H=8, seed=1)
    q = compute_label_free_quality(B_heads, destroy_seed=2, n_destroy_replicas=2)
    for key in ("margin", "destroyed_gap", "agreement", "quality"):
        assert not np.isnan(q[key]).any(), f"{key} contains NaN"


def test_compute_label_free_quality_agreement_is_symmetric():
    """rank_agreement[i] computed by the module should match the
    externally computed mean pairwise tau."""
    B_heads = _make_synthetic_B_heads(H=8, seed=3)
    q = compute_label_free_quality(B_heads, destroy_seed=4, n_destroy_replicas=2)
    # Recompute agreement independently.
    ag2 = rank_agreement(q["ranks"])
    np.testing.assert_allclose(q["agreement"], ag2, atol=1e-10)


# ============================================================================
# build_dynamic_teacher — full pipeline
# ============================================================================

def test_build_dynamic_teacher_end_to_end():
    B_heads = _make_synthetic_B_heads(H=8, seed=5)
    result = build_dynamic_teacher(
        B_heads,
        destroy_seed=6,
        n_destroy_replicas=3,
        teacher_temperature=1.0,
        teacher_smoothing=0.05,
    )
    # Weights sum to 1.
    assert result["weights"].sum() == pytest.approx(1.0, abs=1e-12)
    # Pairwise is 64×64.
    assert result["pairwise"].shape == (64, 64)
    # Pairwise is antisymmetric.
    np.testing.assert_allclose(
        result["pairwise"] + result["pairwise"].T,
        np.ones((64, 64)),
        atol=1e-10,
    )
    # Diagonal is 0.5.
    np.testing.assert_allclose(
        np.diag(result["pairwise"]), 0.5, atol=1e-12,
    )
    # Content orders are 0..63 permutations.
    for h in range(8):
        assert set(result["orders"][h].tolist()) == set(range(64))


def test_build_dynamic_teacher_rejects_wrong_shape():
    with pytest.raises(ValueError):
        build_dynamic_teacher(np.zeros((8, 64, 64)))


# ============================================================================
# B3-style: label-free structural guard
# ============================================================================

def test_label_free_teacher_module_has_no_physical_coordinate_parameters():
    """Every public function in the teacher module must not accept
    inv_perm, clean_perm, or block_perm."""
    import inspect
    from batch_readout import label_free_cdl_teacher as m

    forbidden = {"inv_perm", "clean_perm", "block_perm", "phys_to_model",
                 "model_to_phys", "physical_order", "l2r_order"}
    for name, obj in inspect.getmembers(m, inspect.isfunction):
        if name.startswith("_"):
            continue
        sig = inspect.signature(obj)
        overlap = set(sig.parameters.keys()) & forbidden
        assert not overlap, (
            f"{name} must not accept physical-coordinate parameters; "
            f"found: {sorted(overlap)}"
        )
