"""BR-1 Task 6: tests for pairwise + Plackett-Luce losses."""
import sys
import pathlib
import math

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))


def test_pairwise_direction():
    """Correct direction = low loss; reversed direction = high loss."""
    from batch_readout.loss import pairwise_logistic_loss
    rank = torch.tensor([[0, 1, 2]])  # earliest = node 0
    z_correct = torch.tensor([[10.0, 0.0, -10.0]])
    z_wrong = torch.tensor([[-10.0, 0.0, 10.0]])
    assert pairwise_logistic_loss(z_correct, rank).item() < 0.1
    assert pairwise_logistic_loss(z_wrong, rank).item() > 5.0


def test_pairwise_zero_diff_yields_log2():
    """At z_i == z_j, -log sigmoid(0) == log 2 for each ordered pair."""
    from batch_readout.loss import pairwise_logistic_loss
    rank = torch.tensor([[0, 1, 2]])
    z = torch.zeros(1, 3)
    loss = pairwise_logistic_loss(z, rank).item()
    assert math.isclose(loss, math.log(2.0), rel_tol=1e-5)


def test_pl_closed_form_n3():
    """For N=3, sigma=[0,1,2], z=[2,1,0], tau=1:
       NLL = logsumexp([2,1,0]) - 2 + logsumexp([1,0]) - 1
    """
    from batch_readout.loss import plackett_luce_nll
    z = torch.tensor([[2.0, 1.0, 0.0]])
    sig = torch.tensor([[0, 1, 2]])
    expected = (torch.logsumexp(z[0], 0) - z[0, 0]) + (torch.logsumexp(z[0, 1:], 0) - z[0, 1])
    got = plackett_luce_nll(z, sig, tau=1.0)
    assert math.isclose(got.item(), expected.item(), rel_tol=1e-6)


def test_pl_temperature_scaling_lowers_nll_for_matched_order():
    """When teacher order matches argsort(-z), colder tau (sharper softmax)
    should give a LOWER NLL than hotter tau."""
    from batch_readout.loss import plackett_luce_nll
    z = torch.tensor([[2.0, 1.0, 0.0]])
    sig = torch.tensor([[0, 1, 2]])  # argsort(-z)
    hot = plackett_luce_nll(z, sig, tau=2.0).item()
    cold = plackett_luce_nll(z, sig, tau=0.5).item()
    assert cold < hot, (cold, hot)


def test_pl_rejects_nonpositive_tau():
    from batch_readout.loss import plackett_luce_nll
    z = torch.zeros(1, 3)
    sig = torch.tensor([[0, 1, 2]])
    with pytest.raises(ValueError, match="tau must be > 0"):
        plackett_luce_nll(z, sig, tau=0.0)


def test_pl_gradient_pushes_toward_teacher():
    """Optimising the PL NLL on a fixed teacher should monotonically lower it.
    Smoke check: one Adam step with the right sign on z's gradient."""
    from batch_readout.loss import plackett_luce_nll
    torch.manual_seed(0)
    z = torch.zeros(1, 4, requires_grad=True)
    sig = torch.tensor([[2, 0, 3, 1]])  # arbitrary teacher
    opt = torch.optim.Adam([z], lr=0.1)
    losses = []
    for _ in range(50):
        loss = plackett_luce_nll(z, sig, tau=1.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] - 0.5
    # And argsort(-z) should now agree with teacher
    assert torch.equal(torch.argsort(-z.detach(), dim=-1), sig)
