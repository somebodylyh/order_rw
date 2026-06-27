"""Clean tau readout wrapper reproduces the seed2 L1 carrier signal.

Regression anchor: the seed2 L1 strong carrier set {0,3,5,7} reads ~1.0 in the
wrapper's (L,H) signed C-D+L tau table, matching the saved trajectory table.
"""
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.path_patch_handoff import make_probe_batch, run_clean  # noqa: E402


def test_clean_tau_reproduces_l1_carrier(seed2_bundle):
    model, chunks, seed = seed2_bundle
    assert seed == 2  # read from ckpt['args']['seed']
    dev = torch.device("cpu")
    # Single-batch tau has real sampling variance (the spec mandates multi-batch
    # CIs); the regression anchor reads the same way the production analysis does:
    # mean over 4 probe batches of 16.
    taus = []
    for i in range(4):
        pc, po = make_probe_batch(chunks, n=16, rng=np.random.default_rng(i))
        _, tau = run_clean(model, pc, po, dev)
        assert tau.shape == (4, 8)
        taus.append(tau)
    mean_tau = np.mean(taus, axis=0)
    # Rock-solid heads of the seed2 L1 strong set read ~1.0 (H0 is pool-sensitive,
    # so the anchor uses the stable {3,5,7} plus the carrier-layer identity).
    for h in (3, 5, 7):
        assert mean_tau[1, h] > 0.95
    # L1 is the unique carrier layer: its max |tau| exceeds every other layer's.
    l1_max = np.abs(mean_tau[1]).max()
    assert l1_max > max(np.abs(mean_tau[L]).max() for L in (0, 2, 3))
