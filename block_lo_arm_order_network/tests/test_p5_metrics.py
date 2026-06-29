import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.p5_utility_controller import classify_p5


def test_classify_requires_all_three():
    gain = {"delta_nll": -0.05, "h_shuffle_drop": 0.04, "zero_match": False, "mean_match": False}
    assert classify_p5(gain) == "utility_gain"
    # shuffle keeps the gain -> not content-driven -> no_gain
    nshuf = {"delta_nll": -0.05, "h_shuffle_drop": 0.0, "zero_match": False, "mean_match": False}
    assert classify_p5(nshuf) == "no_gain"
    # no delta -> no_gain
    nd = {"delta_nll": 0.01, "h_shuffle_drop": 0.04, "zero_match": False, "mean_match": False}
    assert classify_p5(nd) == "no_gain"
