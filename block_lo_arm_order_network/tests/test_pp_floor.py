"""Floor baselines: B-level uniform-causal (via strict65, zeroed diagonal) and
random-B null."""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.position_prior_decomp import (  # noqa: E402
    synthetic_uniform_causal_B, random_B, rollout_order, tau_vs_arange, floor_taus,
)


def test_uniform_causal_B_zeroed_diagonal():
    B = synthetic_uniform_causal_B()
    assert B.shape == (65, 65)
    assert np.allclose(np.diag(B), 0)   # strict65 zeroes the diagonal
    assert np.allclose(B[:, 0], 0)      # no edges into None


def test_uniform_causal_floor_high_and_above_null():
    tau_uc = tau_vs_arange(rollout_order(synthetic_uniform_causal_B()))
    taus_rand = [tau_vs_arange(rollout_order(random_B(rng=np.random.default_rng(i))))
                 for i in range(8)]
    assert tau_uc > 0.9                  # pure causal mask+readout already ascends
    assert tau_uc > np.mean(taus_rand)   # above the readout null


def test_floor_taus_keys():
    f = floor_taus()
    assert "uniform_causal" in f and "random_B" in f
    assert f["uniform_causal"] > f["random_B"]
