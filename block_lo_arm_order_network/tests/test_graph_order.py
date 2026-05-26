import numpy as np, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from graph_order import cdl_order, graph_mix_order, score_mix_order


def _ring(N, w=5.0):
    B = np.zeros((N, N))
    for i in range(N):
        B[i, (i + 1) % N] = w
    return B + B.T


def test_cdl_order_is_permutation():
    B = _ring(8)
    o = cdl_order(B, greedy=True)
    assert sorted(o.tolist()) == list(range(8))


def test_graph_mix_lambda_endpoints_match_single_graphs():
    A = _ring(8, 5.0); H = _ring(8, 1.0)[::-1].copy()
    assert np.array_equal(graph_mix_order(A, H, lam=1.0, greedy=True), cdl_order(A, greedy=True))
    assert np.array_equal(graph_mix_order(A, H, lam=0.0, greedy=True), cdl_order(H, greedy=True))


def test_score_mix_gamma_zero_equals_A_only():
    A = _ring(8, 5.0); Hr = _ring(8, 1.0)
    o_gamma0 = score_mix_order(A, Hr, gamma=0.0, greedy=True, seed=0)
    o_Aonly = cdl_order(A, greedy=True)
    assert np.array_equal(o_gamma0, o_Aonly)


def test_score_mix_gamma_changes_order_when_H_disagrees():
    N = 8
    A = _ring(N, 5.0)
    Hr = np.zeros((N, N))
    for i in range(N):
        Hr[i, (i + 3) % N] = 5.0
    Hr = Hr + Hr.T
    o0 = score_mix_order(A, Hr, gamma=0.0, greedy=True, seed=0)
    o2 = score_mix_order(A, Hr, gamma=2.0, greedy=True, seed=0)
    assert not np.array_equal(o0, o2)
