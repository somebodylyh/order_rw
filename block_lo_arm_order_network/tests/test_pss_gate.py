# tests/test_pss_gate.py
import pathlib, sys
import pytest
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.physical_signal_source import carrier_valid_filter

def test_gate_keeps_only_valid():
    B = [object() for _ in range(5)]
    taus = [1.0, 0.2, 0.9, -0.9, -0.89]
    Bv, idx = carrier_valid_filter(B, taus, thr=0.9)
    assert idx == [0, 2, 3]  # exact positive/negative boundary is included
    assert Bv[0] is B[0]
    assert Bv[1] is B[2]
    assert Bv[2] is B[3]


@pytest.mark.parametrize("n_B,n_tau", [(1, 2), (2, 1)])
def test_gate_rejects_mismatched_lengths(n_B, n_tau):
    with pytest.raises(ValueError, match="same length"):
        carrier_valid_filter([object()] * n_B, [0.9] * n_tau)


@pytest.mark.parametrize("thr", [float("nan"), float("inf"), -0.01, 1.01, True, "0.9"])
def test_gate_rejects_invalid_threshold(thr):
    with pytest.raises(ValueError, match=r"thr.*finite number.*\[0, 1\]"):
        carrier_valid_filter([object()], [0.9], thr=thr)


@pytest.mark.parametrize("tau", [float("nan"), float("inf"), -1.01, 1.01, True, "0.9"])
def test_gate_rejects_invalid_tau(tau):
    with pytest.raises(ValueError, match=r"tau_list\[0\].*finite number.*\[-1, 1\]"):
        carrier_valid_filter([object()], [tau])
