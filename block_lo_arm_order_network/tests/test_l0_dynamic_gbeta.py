"""Tests for L0 dynamic g_beta: normalisation, block scorer, dynamic gate.

Covers: forward shapes, alpha invariants, head-permutation equivariance,
gate scale-invariance, head dropout, and structural label-free guarantees.
"""

import pathlib
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))

from batch_readout.l0_dynamic_gbeta import (  # noqa: E402
    L0DynamicGBeta,
    SharedBlockScorer,
    SharedDynamicGate,
    build_row_col_features,
    normalize_strict65,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_B(batch=4, heads=8, seed=0):
    """Create synthetic strict-65 B with non-trivial structure."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    B = torch.rand(batch, heads, 65, 65, generator=rng)
    # strict-65 invariants
    for b in range(batch):
        for h in range(heads):
            B[b, h].fill_diagonal_(0.0)
            B[b, h, 1:, 0] = 0.0
    return B


# ============================================================================
# normalize_strict65
# ============================================================================

def test_normalize_strict65_shape():
    B = _synthetic_B(batch=3, heads=8)
    channels = normalize_strict65(B)
    assert channels.shape == (3, 8, 4, 65, 65)


def test_normalize_strict65_all_finite():
    B = _synthetic_B(batch=2, heads=4)
    # Inject some zeros to test edge cases.
    B[0, 0, :, :] = 0.0
    channels = normalize_strict65(B)
    assert torch.isfinite(channels).all()


def test_prob_channel_row_sums_to_one():
    B = _synthetic_B(batch=2, heads=3)
    channels = normalize_strict65(B)
    prob = channels[:, :, 1]  # (B, H, 65, 65)
    row_sum = prob.sum(dim=-1)  # (B, H, 65)
    # Rows with at least some mass should sum to ~1.
    nonzero_rows = B.sum(dim=-1) > 1e-8
    if nonzero_rows.any():
        assert torch.allclose(
            row_sum[nonzero_rows],
            torch.ones_like(row_sum[nonzero_rows]),
            atol=1e-5,
        )


def test_normalize_with_all_zero_rows():
    """Zero rows should not produce NaN."""
    B = torch.zeros(1, 1, 65, 65)
    channels = normalize_strict65(B)
    assert torch.isfinite(channels).all()


# ============================================================================
# build_row_col_features
# ============================================================================

def test_build_row_col_features_shape():
    B = _synthetic_B(batch=2, heads=8)
    channels = normalize_strict65(B)
    feat = build_row_col_features(channels)
    # 4 channels × 2 (row+col) × 65 = 520
    assert feat.shape == (2, 8, 64, 520)


def test_build_row_col_features_finite():
    B = _synthetic_B(batch=1, heads=2)
    channels = normalize_strict65(B)
    feat = build_row_col_features(channels)
    assert torch.isfinite(feat).all()


# ============================================================================
# L0DynamicGBeta — forward shapes
# ============================================================================

def test_forward_shapes():
    torch.manual_seed(0)
    model = L0DynamicGBeta(heads=8)
    model.eval()
    B = _synthetic_B(batch=4, heads=8)
    scores, aux = model(B, apply_head_dropout=False)
    assert scores.shape == (4, 64)
    assert aux["scores_per_head"].shape == (4, 8, 64)
    assert aux["gate_logits"].shape == (4, 8)
    assert aux["alpha"].shape == (4, 8)
    assert aux["dropped_head"] is None


def test_alpha_sums_to_one():
    torch.manual_seed(1)
    model = L0DynamicGBeta(heads=8)
    model.eval()
    B = _synthetic_B(batch=16, heads=8)
    _, aux = model(B, apply_head_dropout=False)
    torch.testing.assert_close(
        aux["alpha"].sum(dim=1), torch.ones(16), atol=1e-6, rtol=1e-6,
    )


def test_alpha_varies_across_samples():
    """alpha must not be a constant per-head value across all samples."""
    torch.manual_seed(2)
    model = L0DynamicGBeta(heads=8)
    model.eval()
    B = _synthetic_B(batch=32, heads=8)
    _, aux = model(B, apply_head_dropout=False)
    # Std of alpha across the batch, per head
    alpha_std_per_head = aux["alpha"].std(dim=0)  # (8,)
    # At least one head should have non-trivial variation.
    assert (alpha_std_per_head > 1e-4).any(), (
        "alpha should vary across samples"
    )


def test_all_outputs_finite():
    torch.manual_seed(3)
    model = L0DynamicGBeta(heads=8)
    model.eval()
    B = _synthetic_B(batch=8, heads=8)
    scores, aux = model(B, apply_head_dropout=False)
    assert torch.isfinite(scores).all()
    assert torch.isfinite(aux["scores_per_head"]).all()
    assert torch.isfinite(aux["gate_logits"]).all()
    assert torch.isfinite(aux["alpha"]).all()


def test_forward_with_different_batch_sizes():
    torch.manual_seed(4)
    model = L0DynamicGBeta(heads=8)
    model.eval()
    for bs in (1, 2, 8, 16):
        B = _synthetic_B(batch=bs, heads=8)
        scores, _ = model(B, apply_head_dropout=False)
        assert scores.shape == (bs, 64)


# ============================================================================
# Head-permutation equivariance
# ============================================================================

def test_head_permutation_equivariance():
    """Permuting heads → scores invariant, alpha permuted accordingly."""
    torch.manual_seed(5)
    model = L0DynamicGBeta(heads=8)
    model.eval()
    B = _synthetic_B(batch=4, heads=8)

    perm = torch.tensor([3, 0, 7, 1, 5, 2, 6, 4])
    inv_perm = torch.argsort(perm)

    scores_a, aux_a = model(B, apply_head_dropout=False)
    scores_b, aux_b = model(B[:, perm], apply_head_dropout=False)

    # Final scores must be invariant.
    torch.testing.assert_close(scores_a, scores_b, atol=1e-5, rtol=1e-5)

    # Alpha must be permuted (apply inverse to get back to original order).
    torch.testing.assert_close(
        aux_a["alpha"], aux_b["alpha"][:, inv_perm], atol=1e-5, rtol=1e-5,
    )

    # Per-head scores must also be permuted.
    torch.testing.assert_close(
        aux_a["scores_per_head"],
        aux_b["scores_per_head"][:, inv_perm],
        atol=1e-5, rtol=1e-5,
    )


# ============================================================================
# Gate scale-invariance (no raw channel)
# ============================================================================

def test_gate_does_not_read_raw_absolute_scale():
    """Multiplying B_raw by a constant should not change gate logits.

    prob = row / row_sum: exactly invariant to global scaling.
    rowz = (x - mean) / std: exactly invariant to global scaling.
    logz: log(M*x + eps) ≈ log(M) + log(x + eps/M); the additive log(M)
          constant cancels in z-score.  For values well above eps the
          approximation is tight; we use atol=1e-3 to absorb numerical
          differences from the eps floor on near-zero entries.
    """
    torch.manual_seed(6)
    model = L0DynamicGBeta(heads=8)
    model.eval()
    # Use B values ≥ 0.1 to stay well above the eps floor.
    B = torch.rand(4, 8, 65, 65) * 0.9 + 0.1
    for b in range(4):
        for h in range(8):
            B[b, h].fill_diagonal_(0.0)
            B[b, h, 1:, 0] = 0.0

    _, aux_orig = model(B, apply_head_dropout=False)
    B_scaled = B * 100.0
    _, aux_scaled = model(B_scaled, apply_head_dropout=False)

    torch.testing.assert_close(
        aux_orig["gate_logits"], aux_scaled["gate_logits"],
        atol=1e-3, rtol=1e-3,
    )


def test_gate_logits_change_when_structure_changes():
    """Gate SHOULD respond to actual structural changes in B."""
    torch.manual_seed(7)
    model = L0DynamicGBeta(heads=8)
    model.eval()
    B = _synthetic_B(batch=4, heads=8)
    _, aux_a = model(B, apply_head_dropout=False)

    # Permute the content blocks (swap block 0 and block 1's rows/cols).
    B_permuted = B.clone()
    B_permuted[:, :, [1, 2]] = B[:, :, [2, 1]]
    B_permuted[:, :, :, [1, 2]] = B[:, :, :, [2, 1]]
    # Fix diagonal
    for b in range(4):
        for h in range(8):
            B_permuted[b, h].fill_diagonal_(0.0)
            B_permuted[b, h, 1:, 0] = 0.0

    _, aux_b = model(B_permuted, apply_head_dropout=False)
    # Gate should respond — logits should differ.
    assert not torch.allclose(
        aux_a["gate_logits"], aux_b["gate_logits"], atol=1e-6, rtol=1e-6,
    ), "gate should respond to structural changes in B"


# ============================================================================
# Head dropout
# ============================================================================

def test_head_dropout_masks_one_head_per_sample():
    torch.manual_seed(8)
    model = L0DynamicGBeta(heads=8)
    model.eval()
    B = _synthetic_B(batch=32, heads=8)
    _, aux = model(B, apply_head_dropout=True)
    dropped = aux["dropped_head"]
    assert dropped is not None
    assert dropped.shape == (32,)
    # The dropped head's alpha should be exactly 0.
    alpha = aux["alpha"]
    for b in range(32):
        assert alpha[b, dropped[b]] == 0.0, (
            f"sample {b}: dropped head {dropped[b]} should have alpha=0"
        )


def test_head_dropout_inactive_when_disabled():
    torch.manual_seed(9)
    model = L0DynamicGBeta(heads=8)
    model.eval()
    B = _synthetic_B(batch=4, heads=8)
    _, aux = model(B, apply_head_dropout=False)
    assert aux["dropped_head"] is None


def test_head_dropout_active_in_training_mode_by_default():
    torch.manual_seed(10)
    model = L0DynamicGBeta(heads=8)
    model.train()  # training mode → dropout ON by default
    B = _synthetic_B(batch=4, heads=8)
    _, aux = model(B)
    assert aux["dropped_head"] is not None


def test_head_dropout_inactive_in_eval_mode_by_default():
    torch.manual_seed(11)
    model = L0DynamicGBeta(heads=8)
    model.eval()  # eval mode → dropout OFF by default
    B = _synthetic_B(batch=4, heads=8)
    _, aux = model(B)
    assert aux["dropped_head"] is None


def test_dropped_head_alpha_is_zero_others_renormalized():
    torch.manual_seed(12)
    model = L0DynamicGBeta(heads=8)
    model.eval()
    B = _synthetic_B(batch=16, heads=8)
    _, aux = model(B, apply_head_dropout=True)
    alpha = aux["alpha"]
    # Sum should still be 1 (renormalised over remaining 7 heads).
    torch.testing.assert_close(
        alpha.sum(dim=1), torch.ones(16), atol=1e-6, rtol=1e-6,
    )
    # Dropped head alpha is exactly 0.
    for b in range(16):
        assert alpha[b, aux["dropped_head"][b]] == 0.0


# ============================================================================
# No head identity
# ============================================================================

def test_model_has_no_head_identity_parameter():
    """The model must not contain any parameter named with 'head_id' or
    per-head embeddings."""
    model = L0DynamicGBeta(heads=8)
    for name, _ in model.named_parameters():
        assert "head_id" not in name, f"found head identity param: {name}"
        assert "head_embed" not in name, f"found head embed param: {name}"


def test_model_constructor_accepts_heads_but_no_head_idx():
    """The model takes 'heads' (count) but must not take a specific head_idx."""
    import inspect
    sig = inspect.signature(L0DynamicGBeta.__init__)
    param_names = set(sig.parameters.keys())
    assert "heads" in param_names
    forbidden = {"head_idx", "head_id", "head_index", "selected_head"}
    overlap = param_names & forbidden
    assert not overlap, f"L0DynamicGBeta must not accept head identity params: {overlap}"


# ============================================================================
# SharedBlockScorer / SharedDynamicGate
# ============================================================================

def test_shared_block_scorer_shapes():
    scorer = SharedBlockScorer(in_features=520)
    x = torch.randn(16, 520)
    out = scorer(x)
    assert out.shape == (16, 1)


def test_shared_dynamic_gate_shapes():
    gate = SharedDynamicGate(n_channels=3, hidden=32)
    x = torch.randn(2, 8, 3, 65, 65)
    logits = gate(x)
    assert logits.shape == (2, 8)


def test_shared_dynamic_gate_rejects_raw_channel_count():
    """Gate constructor only accepts 3 channels (no raw)."""
    gate = SharedDynamicGate(n_channels=3)
    # Forward with 3 channels works
    x = torch.randn(1, 8, 3, 65, 65)
    out = gate(x)
    assert out.shape == (1, 8)
    # Forward with 4 channels would fail due to shape mismatch
    # (12 features expected, 4*4=16 provided → Linear weight shape error)
    x4 = torch.randn(1, 8, 4, 65, 65)
    with pytest.raises(RuntimeError):
        gate(x4)


# ============================================================================
# Edge cases
# ============================================================================

def test_zero_input_produces_finite_output():
    torch.manual_seed(13)
    model = L0DynamicGBeta(heads=8)
    model.eval()
    B = torch.zeros(2, 8, 65, 65)
    scores, aux = model(B, apply_head_dropout=False)
    assert torch.isfinite(scores).all()
    assert torch.isfinite(aux["scores_per_head"]).all()
    assert torch.isfinite(aux["alpha"]).all()


def test_rejects_wrong_head_count():
    model = L0DynamicGBeta(heads=8)
    model.eval()
    B = torch.zeros(2, 6, 65, 65)  # 6 heads, expected 8
    with pytest.raises(ValueError, match="expected H=8"):
        model(B)


def test_scorer_parameters_shared_not_per_head():
    """Verify that the block scorer has a small, fixed parameter count
    independent of number of heads — proving it's shared."""
    m8 = L0DynamicGBeta(heads=8)
    m16 = L0DynamicGBeta(heads=16)
    n8 = sum(p.numel() for p in m8.parameters())
    n16 = sum(p.numel() for p in m16.parameters())
    # The only difference should be... actually, the scorer and gate are both
    # shared, so parameter count should be IDENTICAL regardless of H.
    assert n8 == n16, (
        f"shared model: param count must not depend on H. "
        f"H=8: {n8}, H=16: {n16}"
    )
