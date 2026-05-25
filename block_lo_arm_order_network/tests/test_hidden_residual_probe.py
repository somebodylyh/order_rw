# tests/test_hidden_residual_probe.py
import numpy as np
from hidden_residual_probe import representation_probe

def test_probe_beats_control_when_hidden_encodes_label():
    rng = np.random.default_rng(0)
    n, E = 400, 8
    labels = rng.integers(0, 4, size=n)
    onehot = np.eye(4)[labels]
    H = onehot @ rng.normal(0, 1, (4, E)) + rng.normal(0, 0.05, (n, E))  # hidden encodes label
    res = representation_probe(H, labels, kind="classification", seed=0)
    assert res["score"] > res["shuffled_control"] + 0.2
    assert res["score"] > 0.8

def test_probe_no_signal_when_hidden_random():
    rng = np.random.default_rng(1)
    n, E = 400, 8
    labels = rng.integers(0, 4, size=n)
    H = rng.normal(0, 1, (n, E))  # no relation
    res = representation_probe(H, labels, kind="classification", seed=0)
    assert abs(res["score"] - res["shuffled_control"]) < 0.15
