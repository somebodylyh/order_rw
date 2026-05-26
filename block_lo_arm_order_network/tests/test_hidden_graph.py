import numpy as np
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from hidden_graph import cosine_graph


def test_cosine_graph_shapes_and_diag_zero():
    rng = np.random.default_rng(0)
    H = rng.standard_normal((5, 4, 8)).astype(np.float32)  # (n,N,E)
    per_sample, mean = cosine_graph(H)
    assert per_sample.shape == (5, 4, 4)
    assert mean.shape == (4, 4)
    assert np.allclose(np.diagonal(per_sample, axis1=1, axis2=2), 0.0)
    assert np.allclose(np.diag(mean), 0.0)


def test_cosine_graph_values_are_cosine_offdiag():
    H = np.zeros((1, 2, 3), dtype=np.float32)
    H[0, 0] = [1.0, 0.0, 0.0]
    H[0, 1] = [1.0, 1.0, 0.0]
    per_sample, mean = cosine_graph(H)
    assert np.isclose(per_sample[0, 0, 1], 1.0 / np.sqrt(2.0), atol=1e-6)
    assert np.isclose(mean[0, 1], 1.0 / np.sqrt(2.0), atol=1e-6)


def test_cosine_graph_handles_zero_vector():
    H = np.zeros((1, 2, 3), dtype=np.float32)  # both zero -> cos undefined -> 0
    per_sample, _ = cosine_graph(H)
    assert np.all(np.isfinite(per_sample))
    assert per_sample[0, 0, 1] == 0.0
