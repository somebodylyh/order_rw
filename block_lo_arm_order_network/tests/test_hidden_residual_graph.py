# tests/test_hidden_residual_graph.py
import numpy as np
import pytest
from hidden_residual_graph import build_B_set, PHYS_FRAME

def test_build_B_set_is_transpose_zero_diag_and_tagged():
    rng = np.random.default_rng(0)
    A_all = rng.random((5, 4, 4)).astype(np.float32)  # (n, N, N)
    B_x_list, B_G, frame = build_B_set(A_all, frame=PHYS_FRAME)
    assert frame == PHYS_FRAME
    assert len(B_x_list) == 5
    # B = A.T, zero diag
    np.testing.assert_allclose(B_x_list[0], A_all[0].T * (1 - np.eye(4)), rtol=1e-5)
    np.testing.assert_allclose(B_G, A_all.mean(0).T * (1 - np.eye(4)), rtol=1e-5)

def test_build_B_set_rejects_non_physical_frame():
    A_all = np.zeros((2, 4, 4), dtype=np.float32)
    with pytest.raises(AssertionError):
        build_B_set(A_all, frame="model_block")

def test_canonical_states_fixed_and_shaped():
    from hidden_residual_graph import canonical_states, N_BLOCKS
    rng = np.random.default_rng(1)
    B_G = rng.random((N_BLOCKS, N_BLOCKS)); np.fill_diagonal(B_G, 0.0)
    st = canonical_states(B_G, t_list=[0, 16, 32, 48])
    # deterministic: same B_G -> identical states
    st2 = canonical_states(B_G, t_list=[0, 16, 32, 48])
    for t in [0, 16, 32, 48]:
        S, U, last = st[t]
        assert len(S) == t and len(U) == N_BLOCKS - t
        assert set(S).isdisjoint(set(U)) and len(set(S) | set(U)) == N_BLOCKS
        assert (last is None) == (t == 0)
        assert st2[t][0] == S  # fixed/deterministic across calls
