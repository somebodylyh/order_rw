"""A1: concentration metrics + robust event timing."""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.emergence_characterization import concentration_metrics, event_timing  # noqa: E402


def test_entropy_high_diffuse_low_concentrated():
    S = 5
    abs_tau = np.ones((S, 4, 8))
    abs_tau[-1] = 0.01
    abs_tau[-1, 1, 0] = 1.0  # last step concentrated in (L1,H0)
    traj = {"steps": np.arange(S) * 200, "tau": abs_tau, "abs_tau": abs_tau}
    m = concentration_metrics(traj)
    assert m["mass_entropy"][0] > m["mass_entropy"][-1]  # diffuse -> concentrated
    assert m["mass_entropy"].shape == (S,)


def test_diffuse_count_collapses_after_pruning():
    # diffuse_count = cells |tau|>=0.7 model-wide; high (diffuse) early, low after
    S = 4
    A = np.zeros((S, 4, 8))
    A[:2] = 0.9            # first 2 steps: all 32 cells diffuse-high
    A[2:] = 0.1; A[2:, 1, [0, 3]] = 0.9   # later: only 2 cells survive
    traj = {"steps": np.arange(S) * 200, "tau": A, "abs_tau": A}
    m = concentration_metrics(traj)
    assert m["diffuse_count"][0] == 32
    assert m["diffuse_count"][-1] == 2


def test_event_timing_finds_drop_step():
    steps = np.arange(6) * 200
    ent = np.array([2.0, 2.0, 1.9, 0.5, 0.4, 0.4])  # steep drop between idx2->3
    ev = event_timing(steps, ent)
    assert ev["midpoint"] == 600  # step at largest negative diff (idx3)
    assert ev["onset"] <= 600 <= ev["completion"]
