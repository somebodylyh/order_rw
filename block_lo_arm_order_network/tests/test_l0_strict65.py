"""Tests for model-frame strict-65 L0 all-head extraction.

Coverage:
  - shape, finiteness, zero-diagonal, None-node invariants
  - deterministic reproduction
  - B3: function signature enforces label-free (no inv_perm / clean_perm accepted)
  - B3 negative: model-frame output DIFFERS from explicit physical-frame remap
  - batch-mean grouping
"""

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))

from batch_readout.l0_strict65 import (  # noqa: E402
    batch_mean_heads,
    build_model_frame_strict65,
)
from none_separated_block_graph import build_none_separated_B  # noqa: E402
from per_head_order_scan import (  # noqa: E402
    _attn_to_A_block_loss_aligned_with_none_model_vec,
    _attn_to_A_block_loss_aligned_with_none_vec,
)
from training_utils import N  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_attn(n_samples=2, n_heads=8, seed=0):
    """Deterministic random attention with some block-level structure."""
    rng = np.random.default_rng(seed)
    # Base: random attention
    attn = rng.random((n_samples, n_heads, 257, 257), dtype=np.float32)
    # Inject weak block-diagonal structure so block aggregation isn't pure noise
    for s in range(n_samples):
        for h in range(n_heads):
            for b in range(64):
                rs, re = 1 + b * 4, 1 + (b + 1) * 4
                cs, ce = 1 + b * 4, 1 + (b + 1) * 4
                attn[s, h, rs:re, cs:ce] += 0.5
    # Zero the diagonal
    for s in range(n_samples):
        for h in range(n_heads):
            np.fill_diagonal(attn[s, h], 0.0)
    return attn


def _random_probe_orders(n_samples=2, seed=1):
    """Deterministic random model-coordinate token orders."""
    rng = np.random.default_rng(seed)
    orders = np.empty((n_samples, 256), dtype=np.int64)
    for i in range(n_samples):
        orders[i] = rng.permutation(256)
    return orders


# ---------------------------------------------------------------------------
# Shape, finiteness, and invariant tests
# ---------------------------------------------------------------------------

def test_build_model_frame_strict65_shape():
    attn = _synthetic_attn(n_samples=2, n_heads=8)
    probe = _random_probe_orders(n_samples=2)
    out = build_model_frame_strict65(attn, probe)
    assert out.shape == (2, 8, 65, 65)
    assert out.dtype == np.float32


def test_all_values_finite():
    attn = _synthetic_attn(n_samples=1, n_heads=4)
    probe = _random_probe_orders(n_samples=1)
    out = build_model_frame_strict65(attn, probe)
    assert np.isfinite(out).all()


def test_zero_diagonal():
    attn = _synthetic_attn(n_samples=2, n_heads=3)
    probe = _random_probe_orders(n_samples=2)
    out = build_model_frame_strict65(attn, probe)
    for s in range(out.shape[0]):
        for h in range(out.shape[1]):
            diag = np.diag(out[s, h])
            np.testing.assert_allclose(diag, 0.0, atol=1e-7)


def test_none_row_nonzero_content_to_none_column_zero():
    """Node 0 = None: B[0, 1:] may be non-zero; B[1:, 0] must be zero."""
    attn = _synthetic_attn(n_samples=1, n_heads=2)
    probe = _random_probe_orders(n_samples=1)
    out = build_model_frame_strict65(attn, probe)
    # None → content edges exist (attention to [None] key is non-zero)
    assert np.any(out[..., 0, 1:] != 0.0), "None row should have non-zero edges"
    # Content → None edges are always zero (None is never a target)
    assert np.all(out[..., 1:, 0] == 0.0), "content-to-None column must be zero"


def test_deterministic_reproduction():
    """Same inputs → bit-identical outputs."""
    attn = _synthetic_attn(n_samples=1, n_heads=2, seed=42)
    probe = _random_probe_orders(n_samples=1, seed=42)
    first = build_model_frame_strict65(attn, probe)
    second = build_model_frame_strict65(attn, probe)
    np.testing.assert_allclose(first, second, atol=1e-7)


# ---------------------------------------------------------------------------
# B3: Function-signature-level label-free enforcement
# ---------------------------------------------------------------------------

def test_function_signature_does_not_accept_physical_permutation():
    """Structural guard: build_model_frame_strict65 has no inv_perm parameter.

    Use inspect to verify the parameter list — this is stronger than a
    runtime test because it catches the mistake at code-review time.
    """
    import inspect

    sig = inspect.signature(build_model_frame_strict65)
    param_names = set(sig.parameters.keys())
    forbidden = {"inv_perm", "clean_perm", "block_perm", "phys_to_model",
                 "model_to_phys", "permutation"}
    overlap = param_names & forbidden
    assert not overlap, (
        f"build_model_frame_strict65 must not accept physical-permutation "
        f"parameters; found: {sorted(overlap)}"
    )


# ---------------------------------------------------------------------------
# B3 negative test: model-frame ≠ physical-frame under non-identity remap
# ---------------------------------------------------------------------------

def test_model_frame_differs_from_explicit_physical_remap():
    """Model-frame output must differ from physical-frame output when
    inv_perm is NOT the identity — proving the extraction genuinely ignores
    physical coordinates rather than accidentally getting the same result."""
    rng = np.random.default_rng(99)
    # 1 sample, 1 head — keep it simple
    attn = rng.random((1, 257, 257), dtype=np.float32)
    # Inject block-diagonal structure so aggregation is meaningful
    for b in range(64):
        rs, re = 1 + b * 4, 1 + (b + 1) * 4
        attn[0, rs:re, rs:re] += 0.3
    np.fill_diagonal(attn[0], 0.0)

    probe = np.stack([rng.permutation(256)])

    # --- model-frame extraction (the correct, label-free path) ---
    # Wrapping a single (1,1,257,257) → (1,1,65,65)
    attn_4d = attn[np.newaxis, :, :, :]  # (1, 1, 257, 257)
    out_model = build_model_frame_strict65(attn_4d, probe)  # (1, 1, 65, 65)

    # --- explicit physical-frame extraction ---
    # Use a non-identity inv_perm: cyclic shift by 7.
    inv_perm = np.roll(np.arange(N, dtype=np.int64), 7)
    assert not np.array_equal(inv_perm, np.arange(N)), (
        "test precondition: inv_perm must not be identity"
    )

    A_phys = _attn_to_A_block_loss_aligned_with_none_vec(
        attn[0],               # (257, 257) — 2D, no leading dim
        probe[0],              # (256,)
        inv_perm,              # non-identity!
    )  # → (64, 65)
    B_phys = build_none_separated_B(A_phys)  # → (65, 65)

    # They MUST differ — otherwise the model-frame claim is vacuous.
    assert not np.allclose(out_model[0, 0], B_phys, atol=1e-7), (
        "Model-frame and physical-frame outputs are identical under "
        "non-identity inv_perm — the extraction is NOT label-free."
    )


# ---------------------------------------------------------------------------
# Model-frame extraction is internally consistent with its own building blocks
# ---------------------------------------------------------------------------

def test_model_frame_extraction_matches_direct_model_vec_call():
    """build_model_frame_strict65 must produce the same B as calling
    _attn_to_A_block_loss_aligned_with_none_model_vec + build_none_separated_B
    directly — verifying no hidden transformations inside the wrapper."""
    attn = _synthetic_attn(n_samples=1, n_heads=2, seed=7)
    probe = _random_probe_orders(n_samples=1, seed=7)
    out = build_model_frame_strict65(attn, probe)

    for h in range(2):
        A_direct = _attn_to_A_block_loss_aligned_with_none_model_vec(
            attn[0, h], probe[0],
        )
        B_direct = build_none_separated_B(A_direct)
        np.testing.assert_allclose(out[0, h], B_direct, atol=1e-7,
                                   err_msg=f"head {h} mismatch")


# ---------------------------------------------------------------------------
# batch_mean_heads
# ---------------------------------------------------------------------------

def test_batch_mean_heads_groups_correctly():
    x = np.arange(4 * 3 * 65 * 65, dtype=np.float32).reshape(4, 3, 65, 65)
    out = batch_mean_heads(x, batch_mean_size=2)
    assert out.shape == (2, 3, 65, 65)
    np.testing.assert_allclose(out[0], x[:2].mean(axis=0), atol=1e-6)
    np.testing.assert_allclose(out[1], x[2:].mean(axis=0), atol=1e-6)


def test_batch_mean_rejects_indivisible():
    x = np.zeros((5, 2, 65, 65), dtype=np.float32)
    with pytest.raises(ValueError):
        batch_mean_heads(x, batch_mean_size=2)


def test_batch_mean_rejects_zero_or_negative_size():
    x = np.zeros((4, 2, 65, 65), dtype=np.float32)
    with pytest.raises(ValueError):
        batch_mean_heads(x, batch_mean_size=0)
    with pytest.raises(ValueError):
        batch_mean_heads(x, batch_mean_size=-1)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_rejects_wrong_attn_ndim():
    with pytest.raises(ValueError, match="4D"):
        build_model_frame_strict65(np.zeros((2, 8, 257)), np.zeros((2, 256), dtype=np.int64))


def test_rejects_wrong_probe_shape():
    attn = np.zeros((2, 8, 257, 257), dtype=np.float32)
    with pytest.raises(ValueError, match="probe_orders"):
        build_model_frame_strict65(attn, np.zeros((3, 256), dtype=np.int64))


# ---------------------------------------------------------------------------
# D1: AR loss-aligned slice direction — B[source, target] is correct
# ---------------------------------------------------------------------------

def test_ar_direction_source_target_is_correct():
    """Prove B[source, target] = attention from queries predicting *target*
    to keys in *source*, using a hand-crafted causal attention matrix.

    Setup: probe_order places block 7 at sequence positions 0..3 and
    block 3 at positions 4..7.  Queries predicting block 3 attend to
    block 7's keys.  Block 7 comes first, so its queries cannot attend
    to block 3 (causal mask).

    Expected: B[7→3]  > 0  and  B[3→7] == 0, confirming the edge
    direction is source→target, not reversed.
    """
    attn = np.zeros((1, 1, 257, 257), dtype=np.float32)

    # Custom probe order: block 7 first, block 3 second.
    probe = np.arange(256, dtype=np.int64)
    probe[0:4] = [7 * 4 + i for i in range(4)]   # block 7 tokens at pos 0..3
    probe[4:8] = [3 * 4 + i for i in range(4)]   # block 3 tokens at pos 4..7
    probe_orders = probe[np.newaxis, :]            # (1, 256)

    # Fill: queries predicting block 3 (rows 4..7) attend to block 7 keys
    # (cols 1..4, since col 0 is [None]).
    attn[0, 0, 4:8, 1:5] = 1.0
    # Queries predicting block 7 attend to [None].
    attn[0, 0, 0:4, 0] = 2.0

    out = build_model_frame_strict65(attn, probe_orders)  # (1, 1, 65, 65)
    B = out[0, 0]

    # Block 7 → block 3 edge must be positive (source=block 7, target=block 3).
    assert B[7 + 1, 3 + 1] > 0.0, (
        f"B[8,4]={B[7+1, 3+1]:.4f} — expected block7→block3 edge to be positive"
    )
    # Reverse edge (block 3 → block 7) must be zero — causal mask prevents it.
    assert B[3 + 1, 7 + 1] == 0.0, (
        f"B[4,8]={B[3+1, 7+1]:.4f} — block3→block7 should be zero (causal)"
    )
    # None → block 7 edge should reflect the attention to [None].
    assert B[0, 7 + 1] > 0.0, (
        f"B[0,8]={B[0, 7+1]:.4f} — None→block7 should be positive"
    )


def test_ar_direction_self_attention_is_symmetric_when_blocks_adjacent():
    """Within a block, queries and keys share the same model block.
    Self-edges (diagonal) must be zero by construction; off-diagonal
    within-block attention is not tested here (block-mean collapses it)."""
    attn = np.ones((1, 1, 257, 257), dtype=np.float32)
    np.fill_diagonal(attn[0, 0], 0.0)
    probe = np.arange(256, dtype=np.int64)
    probe_orders = probe[np.newaxis, :]

    out = build_model_frame_strict65(attn, probe_orders)
    B = out[0, 0]
    # Diagonal is always zero.
    diag = np.diag(B)
    np.testing.assert_allclose(diag, 0.0, atol=1e-7)


# ---------------------------------------------------------------------------
# D2: strict-65 content-to-None column invariant
# ---------------------------------------------------------------------------

def test_strict65_content_to_none_column_is_always_zero():
    """No edge may point from a content block into None — None is only a
    source, never a target.  B[1:, 0] must be exactly zero."""
    rng = np.random.default_rng(13)
    attn = rng.random((3, 8, 257, 257), dtype=np.float32)
    for s in range(3):
        for h in range(8):
            np.fill_diagonal(attn[s, h], 0.0)
    probe = np.stack([rng.permutation(256) for _ in range(3)])
    out = build_model_frame_strict65(attn, probe)
    assert np.all(out[..., 1:, 0] == 0.0), (
        "B[content, None] must be zero — no edges into None"
    )


# ---------------------------------------------------------------------------
# Edge cases: shape variants, dtype, zero input, NaN-free guarantee
# ---------------------------------------------------------------------------

def test_single_sample_single_head():
    attn = _synthetic_attn(n_samples=1, n_heads=1, seed=0)
    probe = _random_probe_orders(n_samples=1, seed=0)
    out = build_model_frame_strict65(attn, probe)
    assert out.shape == (1, 1, 65, 65)
    assert np.isfinite(out).all()


def test_many_heads():
    attn = _synthetic_attn(n_samples=1, n_heads=16, seed=0)
    # Reshape: the _synthetic_attn helper creates 8 heads, so we manually build
    rng = np.random.default_rng(1)
    attn16 = rng.random((1, 16, 257, 257), dtype=np.float32)
    for h in range(16):
        np.fill_diagonal(attn16[0, h], 0.0)
    probe = _random_probe_orders(n_samples=1, seed=0)
    out = build_model_frame_strict65(attn16, probe)
    assert out.shape == (1, 16, 65, 65)


def test_output_dtype_is_float32():
    attn = _synthetic_attn(n_samples=1, n_heads=2)
    probe = _random_probe_orders(n_samples=1)
    out = build_model_frame_strict65(attn, probe)
    assert out.dtype == np.float32


def test_all_zero_attention_produces_finite_output():
    """All-zero attention should produce zero B (no NaN from div-by-zero)."""
    attn = np.zeros((1, 2, 257, 257), dtype=np.float32)
    probe = np.arange(256, dtype=np.int64)[np.newaxis, :]
    out = build_model_frame_strict65(attn, probe)
    assert out.shape == (1, 2, 65, 65)
    assert np.isfinite(out).all()
    # The B matrix should be all zeros since attention is zero everywhere.
    np.testing.assert_allclose(out, 0.0, atol=1e-7)


def test_output_contains_no_nan():
    """With random finite input, output must never contain NaN."""
    rng = np.random.default_rng(77)
    for _ in range(5):
        attn = rng.random((2, 4, 257, 257), dtype=np.float32)
        for s in range(2):
            for h in range(4):
                np.fill_diagonal(attn[s, h], 0.0)
        probe = np.stack([rng.permutation(256) for _ in range(2)])
        out = build_model_frame_strict65(attn, probe)
        assert not np.isnan(out).any(), "output must not contain NaN"


def test_large_batch():
    """Extraction should handle a realistic batch size without error."""
    attn = _synthetic_attn(n_samples=32, n_heads=8, seed=0)
    probe = _random_probe_orders(n_samples=32, seed=0)
    out = build_model_frame_strict65(attn, probe)
    assert out.shape == (32, 8, 65, 65)
    assert np.isfinite(out).all()


def test_rejects_wrong_seq_len():
    attn = np.zeros((1, 1, 300, 300), dtype=np.float32)
    probe = np.zeros((1, 256), dtype=np.int64)
    with pytest.raises(ValueError, match="257"):
        build_model_frame_strict65(attn, probe)


# ---------------------------------------------------------------------------
# Output does NOT expose physical-frame fields (structural)
# ---------------------------------------------------------------------------

def test_output_is_pure_numpy_array_no_metadata_leak():
    """The return value is a plain ndarray — no dict with extra keys that
    could accidentally carry clean_perm / inv_perm / block_perm."""
    attn = _synthetic_attn(n_samples=1, n_heads=2)
    probe = _random_probe_orders(n_samples=1)
    out = build_model_frame_strict65(attn, probe)
    assert isinstance(out, np.ndarray), (
        "build_model_frame_strict65 must return a plain ndarray, "
        "not a dict or tuple that could leak metadata"
    )
