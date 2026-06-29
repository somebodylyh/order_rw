import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.p5_utility_controller import headroom_stats


def test_headroom_positive_when_other_candidate_better():
    # sigma_B nll=1.0 but 'phys' is better (0.5) every sample -> positive headroom
    samples = [{"sigma_B": 1.0, "phys": 0.5, "random_0": 1.2} for _ in range(20)]
    r = headroom_stats(samples)
    assert r["abs_mean"] > 0.4 and r["gate_pass"] is True
    assert r["best_dist"]["phys"] == 1.0


def test_headroom_zero_when_sigmaB_is_best():
    samples = [{"sigma_B": 0.5, "phys": 1.0} for _ in range(20)]
    r = headroom_stats(samples)
    assert r["abs_mean"] <= 0.0 and r["gate_pass"] is False
