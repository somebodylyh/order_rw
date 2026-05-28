"""NR-1 Task 7: tests for pairwise_logistic_loss.

Pins:
  - random init (s=0) -> loss = log(2) (sanity baseline)
  - perfect anti-rank scores -> loss -> 0
  - perfectly inverted scores -> large loss
  - gradient direction: if rank[i] < rank[j], optimization pushes s[i] up and s[j] down
"""
import sys
import math
import pathlib

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))


def test_loss_random_baseline_near_log2():
    """All-zero scores: every pair contributes -log sigma(0) = log 2 ~ 0.693."""
    from neural_readout.loss import pairwise_logistic_loss
    N = 64
    rank = torch.arange(N).unsqueeze(0).long()
    s = torch.zeros(1, N)
    loss = pairwise_logistic_loss(s, rank)
    assert abs(loss.item() - math.log(2)) < 1e-4, loss.item()


def test_loss_zero_when_scores_perfectly_anti_rank():
    """If s = -rank (earliest gets highest score), all pairs are correctly ordered.
    With large score gap the logistic loss approaches 0.
    """
    from neural_readout.loss import pairwise_logistic_loss
    N = 64
    rank = torch.arange(N).unsqueeze(0).float()
    s = -10.0 * rank
    loss = pairwise_logistic_loss(s, rank.long())
    assert loss.item() < 0.001, loss.item()


def test_loss_max_when_scores_inverted():
    """If s = +rank (earliest gets LOWEST score), every pair is wrong.
    Loss should be far above the random-init baseline log(2).
    """
    from neural_readout.loss import pairwise_logistic_loss
    N = 64
    rank = torch.arange(N).unsqueeze(0).float()
    s = 10.0 * rank
    loss = pairwise_logistic_loss(s, rank.long())
    assert loss.item() > 5.0, loss.item()


def test_loss_gradient_pushes_correct_direction():
    """Starting from s=0, gradient w.r.t. s should be more negative for
    earlier-ranked nodes (pushing their score up) and more positive for
    later-ranked nodes (pushing their score down).
    """
    from neural_readout.loss import pairwise_logistic_loss
    s = torch.zeros(1, 4, requires_grad=True)
    rank = torch.tensor([[0, 1, 2, 3]])  # node 0 earliest, node 3 latest
    loss = pairwise_logistic_loss(s, rank)
    loss.backward()
    grad = s.grad[0]
    assert grad[0] < grad[1] < grad[2] < grad[3], grad


def test_shape_mismatch_raises():
    from neural_readout.loss import pairwise_logistic_loss
    import pytest
    with pytest.raises(ValueError, match="shape mismatch"):
        pairwise_logistic_loss(torch.zeros(1, 4), torch.zeros(1, 5).long())
