"""A2: winner predictability (early-bias vs contingent)."""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.emergence_characterization import winner_predictability  # noqa: E402


def _traj_from(abs_tau, steps):
    return {"steps": np.array(steps), "tau": abs_tau, "abs_tau": abs_tau}


def test_early_bias_when_winners_lead_from_start():
    S = 6
    A = np.full((S, 4, 8), 0.1)
    A[:, 1, [0, 3, 5, 7]] = 0.9  # winners lead at ALL steps incl. early
    traj = _traj_from(A, [0, 200, 600, 1000, 1400, 2000])
    w = {"winning_layer": 1, "winner_heads": [0, 3, 5, 7], "tier": "strong"}
    out = winner_predictability(traj, w)
    assert out["auc"][200] > 0.95
    assert out["verdict"] == "early-bias"


def test_contingent_when_winners_tied_until_late():
    S = 6
    A = np.full((S, 4, 8), 0.5)  # everyone tied early
    A[-1, 1, [0, 3, 5, 7]] = 1.0
    A[-1, 1, [1, 2, 4, 6]] = 0.2  # split only at the end
    traj = _traj_from(A, [0, 200, 600, 1000, 1400, 2000])
    w = {"winning_layer": 1, "winner_heads": [0, 3, 5, 7], "tier": "strong"}
    out = winner_predictability(traj, w)
    assert abs(out["auc"][200] - 0.5) < 0.2
    assert out["verdict"] == "contingent"
