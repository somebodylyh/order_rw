import numpy as np
import pytest
from batch_readout.order_tau_readout import per_head_tau, layer_head_tau_table, derived_views


def _l2r_B65():
    # 65-node graph whose greedy None-start rollout yields identity order 0..63.
    # Strong forward edge i->i+1, plus None->0.
    B = np.zeros((65, 65), dtype=np.float64)
    B[0, 1] = 3.0  # None -> physical block 0 (weaker to allow anti to be strongest)
    for i in range(1, 64):
        B[i, i + 1] = 3.0  # block (i-1) -> block i  (node i -> node i+1)
    np.fill_diagonal(B, 0.0)
    return B


def test_per_head_tau_forward_l2r_is_plus_one():
    info = per_head_tau(_l2r_B65(), method="C-D+L")
    assert info["tau_vs_l2r"] > 0.99
    assert info["first_is_phys0"] is True


def _anti_l2r_B65():
    # Greedy None-start rollout yields reverse order 63..0 -> tau ~ -1.
    B = np.zeros((65, 65), dtype=np.float64)
    B[0, 64] = 10.0  # None -> physical block 63 (stronger to break ties)
    for i in range(64, 1, -1):
        B[i, i - 1] = 10.0  # node i -> node i-1 (stronger)
    np.fill_diagonal(B, 0.0)
    return B


def test_layer_head_table_shapes_and_signs():
    L, H = 2, 2
    B = np.zeros((L, H, 65, 65))
    B[0, 0] = _l2r_B65()      # forward  tau ~ +1
    B[0, 1] = _anti_l2r_B65()  # reverse  tau ~ -1
    B[1, 0] = _l2r_B65()
    B[1, 1] = _l2r_B65()
    table = layer_head_tau_table(B, methods=("C-D+L",))
    assert table["tau"].shape == (2, 2, 1)
    assert table["tau"][0, 0, 0] > 0.99
    assert table["tau"][0, 1, 0] < -0.99


def test_derived_views_keep_abs_and_signed():
    L, H = 2, 2
    B = np.zeros((L, H, 65, 65))
    B[0, 0] = _anti_l2r_B65()   # strongest |tau| in layer 0 is the anti head
    B[0, 1] = _l2r_B65()
    B[1, 0] = _l2r_B65()
    B[1, 1] = _l2r_B65()
    table = layer_head_tau_table(B, methods=("C-D+L",))
    d = derived_views(table)
    # max_signed in layer 0 is the +1 head (head 1); max_abs picks the anti head's |−1| (head 0)
    assert d["max_signed_tau_per_layer"][0, 0] > 0.99
    assert d["max_abs_tau_per_layer"][0, 0] > 0.99
    # the anti head must not be invisible: signed best != abs-best head here
    assert d["best_head_per_layer"]["signed"][0, 0] == 1
    assert d["best_head_per_layer"]["abs"][0, 0] == 0
