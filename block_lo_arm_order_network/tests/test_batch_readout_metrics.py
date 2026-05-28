"""BR-1 Task 8: tests for matching metrics."""
import sys
import pathlib

import numpy as np
import torch
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))


def test_identity_metrics():
    from batch_readout.eval_metrics import (
        kendall_tau_batch, pairwise_acc, spearman_rho_batch, top1_acc, first_k_overlap,
    )
    sig = np.tile(np.arange(64), (5, 1))
    assert kendall_tau_batch(sig, sig) == pytest.approx(1.0, abs=1e-9)
    assert spearman_rho_batch(sig, sig) == pytest.approx(1.0, abs=1e-9)
    # rank where node 0 = earliest -> highest logit; pair is correct iff z[i] > z[j]
    rank = sig.copy()
    z = torch.tensor(np.linspace(1, -1, 64))[None].repeat(5, 1)
    assert pairwise_acc(z, torch.from_numpy(rank)) == 1.0
    assert top1_acc(torch.from_numpy(sig), torch.from_numpy(sig)) == 1.0
    assert first_k_overlap(torch.from_numpy(sig), torch.from_numpy(sig), k=3) == 1.0


def test_reversed_kendall_minus_one():
    from batch_readout.eval_metrics import kendall_tau_batch
    sig = np.tile(np.arange(64), (5, 1))
    rev = sig[:, ::-1].copy()
    assert kendall_tau_batch(sig, rev) == -1.0


def test_top1_partial_credit():
    from batch_readout.eval_metrics import top1_acc
    pred = torch.tensor([[2, 0, 1], [0, 1, 2], [1, 0, 2]])
    teacher = torch.tensor([[2, 1, 0], [1, 2, 0], [1, 0, 2]])
    # row 0: 2==2 ✓; row 1: 0!=1 ✗; row 2: 1==1 ✓ -> 2/3
    assert top1_acc(pred, teacher) == pytest.approx(2 / 3)


def test_first_3_overlap_known_value():
    from batch_readout.eval_metrics import first_k_overlap
    pred = torch.tensor([[2, 0, 1, 3, 4]])
    teacher = torch.tensor([[2, 1, 0, 4, 3]])
    # first-3 of pred: {2,0,1}; first-3 of teacher: {2,1,0} -> overlap 3, ratio 1.0
    assert first_k_overlap(pred, teacher, k=3) == 1.0


def test_pairwise_acc_known_value():
    """3 nodes, rank = [0, 1, 2] (node 0 earliest). 3 ordered pairs total
    (i, j) where rank_i < rank_j; with z = [3, 1, 2] all three diffs (z0>z1,
    z0>z2, z1<z2) -> 2/3 correct."""
    from batch_readout.eval_metrics import pairwise_acc
    rank = torch.tensor([[0, 1, 2]])
    z = torch.tensor([[3.0, 1.0, 2.0]])
    assert pairwise_acc(z, rank) == pytest.approx(2 / 3, abs=1e-6)
