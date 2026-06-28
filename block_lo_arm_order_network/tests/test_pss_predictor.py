# tests/test_pss_predictor.py
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.physical_signal_source import (
    slot_only_r2, valid_edge_mask, synthetic_content_invariant, synthetic_content_randomized)

def test_predictor_high_on_invariant_low_on_random():
    m = valid_edge_mask(65)
    invariant_r2 = slot_only_r2(synthetic_content_invariant(8), m)
    randomized_r2 = slot_only_r2(synthetic_content_randomized(40), m)
    assert invariant_r2 > 0.95                    # fixed table explains all
    assert 0.6 < randomized_r2 < 0.8              # global held-out R² calibrates near 0.70
    assert invariant_r2 > randomized_r2
