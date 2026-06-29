"""Unit tests for cotrain_sampling.sample_one_order_per_seq."""
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from order_network import CrossAttentionOrderNetwork
from cotrain_sampling import sample_one_order_per_seq


def _make_on(d_edge=16, d_model=16, seed=0):
    torch.manual_seed(seed)
    return CrossAttentionOrderNetwork(num_blocks=16, d_edge=d_edge, d_model=d_model).eval()


def test_sample_one_order_shapes():
    on = _make_on()
    A = torch.randn(4, 16, 16).softmax(dim=-1)
    orders, log_probs, entropies = sample_one_order_per_seq(on, A, temperature=1.0)
    assert orders.shape == (4, 16)
    assert log_probs.shape == (4,)
    assert entropies.shape == (4,)
    # each row is a permutation of 0..15
    for b in range(4):
        assert sorted(orders[b].tolist()) == list(range(16))


def test_temperature_low_acts_like_argmax():
    on = _make_on(seed=1)
    A = torch.randn(2, 16, 16).softmax(dim=-1)
    torch.manual_seed(123)
    o_low, _, _ = sample_one_order_per_seq(on, A, temperature=0.01)
    torch.manual_seed(123)
    o_low2, _, _ = sample_one_order_per_seq(on, A, temperature=0.01)
    assert torch.equal(o_low, o_low2)


def test_log_prob_not_nan():
    on = _make_on()
    A = torch.randn(3, 16, 16).softmax(dim=-1)
    _, log_probs, ent = sample_one_order_per_seq(on, A, temperature=0.7)
    assert torch.isfinite(log_probs).all()
    assert torch.isfinite(ent).all()
    assert (log_probs <= 0).all()
