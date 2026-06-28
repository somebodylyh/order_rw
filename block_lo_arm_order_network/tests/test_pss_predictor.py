# tests/test_pss_predictor.py
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.physical_signal_source import (
    slot_only_r2, valid_edge_mask, synthetic_content_invariant, synthetic_content_randomized)

def test_predictor_high_on_invariant_low_on_random():
    m = valid_edge_mask(65)
    assert slot_only_r2(synthetic_content_invariant(8), m) > 0.95   # fixed table explains all
    assert slot_only_r2(synthetic_content_randomized(40), m) < 0.5  # content-free table fails
