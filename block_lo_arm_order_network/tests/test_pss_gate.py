# tests/test_pss_gate.py
import pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.physical_signal_source import carrier_valid_filter

def test_gate_keeps_only_valid():
    B = [np.zeros((65, 65)) for _ in range(4)]
    taus = [1.0, 0.2, 0.95, -0.99]
    Bv, idx = carrier_valid_filter(B, taus, thr=0.9)
    assert idx == [0, 2, 3]                                # |tau|>=0.9 (incl anti)
    assert len(Bv) == 3
