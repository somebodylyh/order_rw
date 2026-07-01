import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def _base_metrics():
    """Healthy joint_group that passes both tiers."""
    return {
        "nan": False,
        "entropy_ok": True,
        "orderhead_param_delta": 0.5,
        "loss_joint_group_final": 3.30,
        "loss_frozen_gbeta_final": 3.30,
        "delta_probe_group_best_joint": -0.05,
        "delta_probe_group_best_frozen": +0.10,
        "real_beats_controls": True,
        "loss_curve_joint": [3.40, 3.35, 3.32, 3.30],
        "loss_curve_frozen": [3.40, 3.36, 3.34, 3.33],
    }


def test_gate_pass_on_mechanism_life():
    """Healthy metrics with mechanism life → PASS."""
    from analyses.v3_phase1_runner import decide_gate

    r = decide_gate(_base_metrics())
    assert r["tier1_pass"] is True
    assert r["tier2_pass"] is True
    assert r["gate"] == "PASS"
    assert len(r["reasons"]) >= 1


def test_gate_fail_when_catastrophically_worse():
    """joint_group >> frozen_gbeta on loss → Tier 1 fails → FAIL."""
    from analyses.v3_phase1_runner import decide_gate

    m = _base_metrics()
    m["loss_joint_group_final"] = 3.30 + 0.5  # exceeds ε_loss=0.01
    r = decide_gate(m)
    assert r["tier1_pass"] is False
    assert r["gate"] == "FAIL"


def test_gate_fail_when_nan():
    """NaN in any component → Tier 1 fails."""
    from analyses.v3_phase1_runner import decide_gate

    m = _base_metrics()
    m["nan"] = True
    r = decide_gate(m)
    assert r["tier1_pass"] is False


def test_gate_fail_when_entropy_unhealthy():
    """Entropy collapsed → Tier 1 fails."""
    from analyses.v3_phase1_runner import decide_gate

    m = _base_metrics()
    m["entropy_ok"] = False
    r = decide_gate(m)
    assert r["tier1_pass"] is False


def test_gate_fail_when_no_orderhead_movement():
    """OrderHead params didn't move → Tier 1 fails."""
    from analyses.v3_phase1_runner import decide_gate

    m = _base_metrics()
    m["orderhead_param_delta"] = 0.0
    r = decide_gate(m)
    assert r["tier1_pass"] is False


def test_gate_fail_when_no_life_sign():
    """Neither mechanism nor payoff life → Tier 2 fails → FAIL."""
    from analyses.v3_phase1_runner import decide_gate

    m = _base_metrics()
    m["delta_probe_group_best_joint"] = +0.11  # worse than frozen +0.10
    m["real_beats_controls"] = False
    m["loss_joint_group_final"] = 3.31  # worse than frozen but within ε
    m["loss_frozen_gbeta_final"] = 3.30
    m["loss_curve_joint"] = [3.40, 3.40, 3.38, 3.31]  # worse AUC
    m["loss_curve_frozen"] = [3.40, 3.36, 3.34, 3.33]
    r = decide_gate(m)
    assert r["tier2_pass"] is False
    assert r["gate"] == "FAIL"


def test_gate_pass_on_payoff_life_alone():
    """Mechanism fails but PPL curve beats frozen → Tier 2 passes on payoff."""
    from analyses.v3_phase1_runner import decide_gate

    m = _base_metrics()
    m["delta_probe_group_best_joint"] = +0.11  # mechanism fails
    m["real_beats_controls"] = False
    # But payoff: joint matches frozen at final and better AUC
    m["loss_joint_group_final"] = 3.30
    m["loss_frozen_gbeta_final"] = 3.30  # matches
    m["loss_curve_joint"] = [3.40, 3.33, 3.31, 3.30]
    m["loss_curve_frozen"] = [3.40, 3.36, 3.34, 3.33]
    r = decide_gate(m)
    assert r["tier2_pass"] is True
    assert r["gate"] == "PASS"


def test_gate_fail_closed_on_missing_keys():
    """Missing required keys → clear error, not silent pass/fail."""
    from analyses.v3_phase1_runner import decide_gate

    with pytest.raises(KeyError):
        decide_gate({})
