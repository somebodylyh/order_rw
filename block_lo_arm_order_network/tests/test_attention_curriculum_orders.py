"""Tests for attention-only curriculum order generation."""

import inspect
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attention_curriculum_orders import (
    global_a_greedy,
    global_consensus_greedy,
    mixed_a_greedy,
    old_nn_greedy,
    set_aware_mixed_a_greedy,
)
from order_diagnostics import evaluate_order_diagnostics


FORBIDDEN_GENERATION_PARAMS = {
    "l2r",
    "l2r_order",
    "original_order",
    "ori_order",
    "original_position",
    "position_prior",
    "index_prior",
}


def _path_attention(path, n=None, weight=10.0):
    """Create A where each later block strongly attends to the prior path block."""
    if n is None:
        n = len(path)
    A = np.zeros((n, n), dtype=np.float32)
    for prev, nxt in zip(path[:-1], path[1:]):
        A[nxt, prev] = weight
    np.fill_diagonal(A, 0.0)
    return A


def _assert_permutation(order, n):
    assert sorted(np.asarray(order).tolist()) == list(range(n))


def test_generation_api_does_not_accept_l2r_or_position_priors():
    generation_functions = [
        old_nn_greedy,
        global_a_greedy,
        mixed_a_greedy,
        set_aware_mixed_a_greedy,
        global_consensus_greedy,
    ]

    for fn in generation_functions:
        params = set(inspect.signature(fn).parameters)
        assert not (params & FORBIDDEN_GENERATION_PARAMS), fn.__name__


def test_adversarial_index_leakage_set_aware_follows_attention_not_raw_index():
    target_path = np.asarray([3, 1, 4, 0, 2], dtype=np.int64)
    A = _path_attention(target_path)

    order = set_aware_mixed_a_greedy(A, A, lambda_mix=1.0)

    _assert_permutation(order, 5)
    assert order.tolist() == target_path.tolist()
    assert order.tolist() != list(range(5))


def test_global_a_greedy_uses_global_attention_for_all_samples():
    global_path = np.asarray([4, 2, 0, 3, 1], dtype=np.int64)
    sample_path = np.asarray([1, 3, 0, 2, 4], dtype=np.int64)
    A_global = _path_attention(global_path)
    A_sample = _path_attention(sample_path)

    order_from_sample = global_a_greedy(A_sample, A_global)
    order_from_other_sample = global_a_greedy(np.zeros_like(A_sample), A_global)

    assert order_from_sample.tolist() == old_nn_greedy(A_global).tolist()
    assert order_from_other_sample.tolist() == old_nn_greedy(A_global).tolist()


def test_mixed_a_greedy_computes_lambda_mix_attention():
    sample_path = np.asarray([3, 1, 4, 0, 2], dtype=np.int64)
    global_path = np.asarray([4, 2, 0, 3, 1], dtype=np.int64)
    A_sample = _path_attention(sample_path, weight=10.0)
    A_global = _path_attention(global_path, weight=1.0)

    sample_order = mixed_a_greedy(A_sample, A_global, lambda_mix=1.0)
    global_order = mixed_a_greedy(A_sample, A_global, lambda_mix=0.0)

    assert sample_order.tolist() == old_nn_greedy(A_sample).tolist()
    assert global_order.tolist() == old_nn_greedy(A_global).tolist()


def test_mixed_a_greedy_is_denoised_old_nn_not_set_aware():
    A = np.asarray([
        [0.0, 0.2697867155, 0.0409735255, 0.0165276360, 0.8132702112],
        [0.9127555490, 0.0, 0.7294965386, 0.5436249971, 0.9350724220],
        [0.8158535361, 0.0027385002, 0.0, 0.0335855745, 0.7296554446],
        [0.1756556183, 0.8631789088, 0.5414612293, 0.0, 0.4226872325],
        [0.0283196718, 0.1242832765, 0.6706244349, 0.6471894979, 0.0],
    ], dtype=np.float32)

    mixed_order = mixed_a_greedy(A, A, lambda_mix=1.0)
    set_aware_order = set_aware_mixed_a_greedy(A, A, lambda_mix=1.0)

    assert mixed_order.tolist() == old_nn_greedy(A).tolist()
    assert mixed_order.tolist() != set_aware_order.tolist()


def test_global_consensus_uses_sigma_global_rank_not_raw_block_index():
    global_path = np.asarray([4, 2, 0, 3, 1], dtype=np.int64)
    sample_path = np.asarray([1, 3, 0, 2, 4], dtype=np.int64)
    A_global = _path_attention(global_path, weight=10.0)
    A_sample = _path_attention(sample_path, weight=10.0)

    order = global_consensus_greedy(
        A_sample,
        A_global,
        lambda_mix=1.0,
        beta_seen=0.0,
        beta_unseen=0.0,
        beta_local=0.0,
        beta_source=0.0,
        beta_consensus=1.0,
    )

    _assert_permutation(order, 5)
    assert order.tolist() == global_path.tolist()
    assert order.tolist() != list(range(5))


def test_l2r_tau_is_diagnostics_only():
    order = np.asarray([3, 1, 4, 0, 2], dtype=np.int64)

    summary = evaluate_order_diagnostics(order)

    assert summary["tau_vs_l2r"] != 1.0
    assert summary["is_valid_permutation"]
