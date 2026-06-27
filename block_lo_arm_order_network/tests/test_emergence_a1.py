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


def test_event_timing_finds_drop_step():
    steps = np.arange(6) * 200
    ent = np.array([2.0, 2.0, 1.9, 0.5, 0.4, 0.4])  # steep drop between idx2->3
    ev = event_timing(steps, ent)
    assert ev["midpoint"] == 600  # step at largest negative diff (idx3)
    assert ev["onset"] <= 600 <= ev["completion"]
