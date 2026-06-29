"""Tests for soft pairwise BCE loss, accuracy, and regularisation."""

import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))

from batch_readout.soft_pairwise import (  # noqa: E402
    entropy_floor_loss,
    gate_entropy,
    non_tie_mask,
    pairwise_accuracy,
    pairwise_mask,
    per_head_aux_loss,
    soft_pairwise_bce_loss,
    total_pretrain_loss,
)


# ============================================================================
# pairwise_mask
# ============================================================================

def test_pairwise_mask_is_strict_upper_triangle():
    mask = pairwise_mask(5)
    assert mask.shape == (5, 5)
    # Diagonal is False.
    assert not mask[0, 0].item()
    assert not mask[4, 4].item()
    # Upper triangle i<j is True.
    assert mask[0, 1].item()
    assert mask[1, 4].item()
    # Lower triangle i>j is False.
    assert not mask[1, 0].item()
    assert not mask[4, 1].item()


# ============================================================================
# non_tie_mask
# ============================================================================

def test_non_tie_mask_excludes_05():
    Y = torch.tensor([[[0.5, 1.0], [0.0, 0.5]]])
    nt = non_tie_mask(Y)
    assert nt[0, 0, 0].item() is False  # 0.5
    assert nt[0, 0, 1].item() is True   # 1.0
    assert nt[0, 1, 0].item() is True   # 0.0
    assert nt[0, 1, 1].item() is False  # 0.5


# ============================================================================
# soft_pairwise_bce_loss
# ============================================================================

def test_soft_pairwise_bce_prefers_correct_ordering():
    """Loss should be lower when scores match the teacher preferences."""
    Y = torch.tensor([[
        [0.5, 1.0, 1.0],
        [0.0, 0.5, 1.0],
        [0.0, 0.0, 0.5],
    ]])  # teacher: 0 < 1 < 2
    correct = torch.tensor([[3.0, 2.0, 1.0]])   # 0 > 1 > 2 scores
    wrong = -correct                               # 0 < 1 < 2 scores
    assert soft_pairwise_bce_loss(correct, Y) < soft_pairwise_bce_loss(wrong, Y)


def test_soft_pairwise_bce_finite_and_backward():
    scores = torch.randn(4, 64, requires_grad=True)
    Y = torch.sigmoid(torch.randn(4, 64, 64))
    # Make antisymmetric
    Y = Y / (Y + Y.transpose(-1, -2).clamp_min(1e-6))
    Y[:, range(64), range(64)] = 0.5
    loss = soft_pairwise_bce_loss(scores, Y)
    assert torch.isfinite(loss)
    loss.backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()


def test_soft_pairwise_bce_diagonal_does_not_affect_loss():
    """Changing the diagonal of Y should not change the loss because
    the pairwise mask excludes it."""
    scores = torch.tensor([[float(i) for i in range(64)]])
    Y = torch.full((1, 64, 64), 0.5)
    for i in range(64):
        for j in range(i + 1, 64):
            Y[0, i, j] = 1.0
            Y[0, j, i] = 0.0
    loss1 = soft_pairwise_bce_loss(scores, Y)
    # Set diagonal to absurd values — should be ignored.
    Y_diag = Y.clone()
    Y_diag[:, range(64), range(64)] = 99.0
    loss2 = soft_pairwise_bce_loss(scores, Y_diag)
    torch.testing.assert_close(loss1, loss2)


def test_soft_pairwise_bce_rejects_wrong_shape():
    scores = torch.randn(2, 64)
    Y = torch.randn(2, 63, 63)
    with pytest.raises(ValueError):
        soft_pairwise_bce_loss(scores, Y)


# ============================================================================
# pairwise_accuracy
# ============================================================================

def test_pairwise_accuracy_perfect():
    Y = torch.tensor([[
        [0.5, 1.0, 1.0],
        [0.0, 0.5, 1.0],
        [0.0, 0.0, 0.5],
    ]])
    scores = torch.tensor([[3.0, 2.0, 1.0]])
    assert pairwise_accuracy(scores, Y) == 1.0


def test_pairwise_accuracy_zero_for_reversed():
    Y = torch.tensor([[
        [0.5, 1.0, 1.0],
        [0.0, 0.5, 1.0],
        [0.0, 0.0, 0.5],
    ]])
    scores = torch.tensor([[1.0, 2.0, 3.0]])  # reversed
    assert pairwise_accuracy(scores, Y) == 0.0


def test_pairwise_accuracy_excludes_diagonal():
    """Diagonal entries (always 0.5) must not contribute to accuracy."""
    Y = torch.full((1, 4, 4), 0.5)
    for i in range(4):
        for j in range(i + 1, 4):
            Y[0, i, j] = 1.0
            Y[0, j, i] = 0.0
    scores = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    # All off-diagonal predictions should be correct.
    assert pairwise_accuracy(scores, Y) == 1.0


def test_pairwise_accuracy_excludes_ties():
    """Entries where Y ≈ 0.5 (teacher has no preference) must be excluded."""
    Y = torch.zeros(2, 64, 64)
    for b in range(2):
        Y[b, range(64), range(64)] = 0.5
    # Only set ONE off-diagonal pair with a clear preference.
    Y[0, 0, 1] = 1.0
    Y[0, 1, 0] = 0.0
    Y[1, 5, 10] = 0.0
    Y[1, 10, 5] = 1.0
    # All other off-diagonal entries are 0.0 (which != 0.5, so they count!)
    # Actually 0.0 != 0.5, so they DO count as evaluable.  Let me fix:
    # Set everything except the test pairs to 0.5.
    Y[:, :, :] = 0.5
    Y[0, 0, 1] = 1.0; Y[0, 1, 0] = 0.0
    Y[1, 5, 10] = 0.0; Y[1, 10, 5] = 1.0

    # Sample 0: scores[0]=high, scores[1]=low → correct (Y[0,1]=1.0)
    # Sample 1: scores[5]=low, scores[10]=high → correct (Y[5,10]=0.0 → want 10 before 5 → score[10] > score[5])
    scores = torch.zeros(2, 64)
    scores[0, 0] = 5.0; scores[0, 1] = 1.0  # 0 > 1 → Y[0,1]=1.0 ✓
    scores[1, 5] = 1.0; scores[1, 10] = 5.0  # 10 > 5 → Y[5,10]=0.0 (5 before 10? No: Y[5,10]=0.0 means rank[5] > rank[10], so 5 AFTER 10, so score[10] > score[5]) ✓

    assert pairwise_accuracy(scores, Y) == 1.0


def test_pairwise_accuracy_all_ties_returns_1():
    """When all off-diagonal entries are 0.5, there are no evaluable pairs."""
    Y = torch.full((2, 64, 64), 0.5)
    scores = torch.randn(2, 64)
    assert pairwise_accuracy(scores, Y) == 1.0  # vacuously perfect


# ============================================================================
# per_head_aux_loss
# ============================================================================

def test_per_head_aux_loss_is_per_head():
    """The aux loss must compute BCE independently per head, not pool first."""
    # Teacher: 0 < 1 < 2 (upper triangle = 1.0).
    Y = torch.tensor([[
        [0.5, 1.0, 1.0],
        [0.0, 0.5, 1.0],
        [0.0, 0.0, 0.5],
    ]]).float()

    # Head 0: scores [3,2,1] → order 0,1,2 → matches teacher → low loss.
    # Head 1: scores [1,3,2] → order 1,2,0 → mixed.
    # Head 2: scores [1,2,3] → order 2,1,0 → reversed → high loss.
    scores = torch.tensor([[
        [3.0, 2.0, 1.0],  # head 0: correct
        [1.0, 3.0, 2.0],  # head 1: mixed
        [1.0, 2.0, 3.0],  # head 2: reversed
    ]])

    # Per-head losses should differ.
    p_mask = pairwise_mask(3)
    losses = []
    for h in range(3):
        losses.append(
            soft_pairwise_bce_loss(scores[0:1, h, :], Y, pair_mask=p_mask).item()
        )
    # Head 0 (correct) should have lowest loss, head 2 (reversed) highest.
    assert losses[0] < losses[2], (
        f"correct head loss {losses[0]:.4f} should be < reversed loss {losses[2]:.4f}"
    )

    # The per_head_aux_loss should equal the mean of individual losses.
    aux = per_head_aux_loss(scores, Y)
    torch.testing.assert_close(aux, torch.tensor(losses).mean())


def test_per_head_aux_loss_finite_and_backward():
    scores = torch.randn(2, 8, 64, requires_grad=True)
    Y = torch.full((2, 64, 64), 0.5)
    for b in range(2):
        for i in range(64):
            for j in range(i + 1, 64):
                v = float((i + j) % 3) / 2.0
                Y[b, i, j] = v
                Y[b, j, i] = 1.0 - v
    loss = per_head_aux_loss(scores, Y)
    assert torch.isfinite(loss)
    loss.backward()
    assert scores.grad is not None


# ============================================================================
# gate_entropy / entropy_floor_loss
# ============================================================================

def test_gate_entropy_uniform_is_max():
    H = 8
    alpha = torch.full((4, H), 1.0 / H)
    ent = gate_entropy(alpha)  # (4,)
    expected = torch.full((4,), torch.tensor(H).log().item())
    torch.testing.assert_close(ent, expected, atol=1e-5, rtol=1e-5)


def test_gate_entropy_collapsed_is_zero():
    alpha = torch.zeros(4, 8)
    alpha[:, 0] = 1.0
    ent = gate_entropy(alpha)
    torch.testing.assert_close(ent, torch.zeros(4), atol=1e-5, rtol=1e-5)


def test_entropy_floor_loss_penalises_collapsed():
    uniform = torch.full((4, 8), 1.0 / 8)
    collapsed = torch.zeros(4, 8)
    collapsed[:, 0] = 1.0
    assert entropy_floor_loss(collapsed, min_entropy=1.5) > 0.0
    assert entropy_floor_loss(uniform, min_entropy=1.5) == 0.0


def test_entropy_floor_loss_finite_and_backward():
    logits = torch.randn(4, 8, requires_grad=True)
    alpha = logits.softmax(dim=1)
    alpha.retain_grad()  # softmax output is non-leaf
    loss = entropy_floor_loss(alpha, min_entropy=1.5)
    assert torch.isfinite(loss)
    loss.backward()
    assert alpha.grad is not None
    assert logits.grad is not None


# ============================================================================
# total_pretrain_loss
# ============================================================================

def test_total_pretrain_loss_keys():
    scores = torch.randn(4, 64)
    scores_h = torch.randn(4, 8, 64)
    alpha = torch.randn(4, 8).softmax(dim=1)
    outputs = {"scores": scores, "scores_per_head": scores_h, "alpha": alpha}
    Y = torch.full((4, 64, 64), 0.5)
    for b in range(4):
        for i in range(64):
            for j in range(i + 1, 64):
                Y[b, i, j] = 1.0
                Y[b, j, i] = 0.0

    result = total_pretrain_loss(outputs, Y, lambda_aux=0.05, lambda_ent=0.001)
    for key in ("loss", "loss_final", "loss_aux", "loss_ent", "pairwise_acc"):
        assert key in result, f"missing key: {key}"
    assert torch.isfinite(result["loss"])
    assert 0.0 <= result["pairwise_acc"] <= 1.0


def test_total_pretrain_loss_finite_and_backward():
    scores = torch.randn(2, 64, requires_grad=True)
    scores_h = torch.randn(2, 8, 64, requires_grad=True)
    alpha = torch.randn(2, 8, requires_grad=True).softmax(dim=1)
    outputs = {"scores": scores, "scores_per_head": scores_h, "alpha": alpha}
    Y = torch.full((2, 64, 64), 0.5)
    for b in range(2):
        for i in range(64):
            for j in range(i + 1, 64):
                Y[b, i, j] = float(j > i)
                Y[b, j, i] = float(i > j)

    result = total_pretrain_loss(outputs, Y)
    result["loss"].backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()
