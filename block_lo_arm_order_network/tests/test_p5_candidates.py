import pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.p5_utility_controller import candidate_orders


def test_candidate_pool_diverse_and_valid():
    rng = np.random.default_rng(0)
    sigB = rng.permutation(64)
    pool = candidate_orders(sigB, rng, n_random=4, n_noisy=4)
    assert "sigma_B" in pool and "phys" in pool and "reverse_phys" in pool
    assert sum(k.startswith("random_") for k in pool) == 4
    assert sum(k.startswith("noisy_B_") for k in pool) == 4
    for s in pool.values():
        assert sorted(s.tolist()) == list(range(64))     # all valid perms
    assert np.array_equal(pool["sigma_B"], sigB)
    # noisy_B is a small perturbation of sigma_B (not identical, not random)
    assert not np.array_equal(pool["noisy_B_0"], sigB)
