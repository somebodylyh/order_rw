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

def test_residual_target_zero_when_Bx_equals_BG_and_nonzero_when_differ():
    from hidden_residual_graph import canonical_states, residual_target, PHYS_FRAME, N_BLOCKS
    rng = np.random.default_rng(2)
    B_G = rng.random((N_BLOCKS, N_BLOCKS)); np.fill_diagonal(B_G, 0.0)
    states = canonical_states(B_G, t_list=[0, 16])
    # identical B_x -> r ~ 0
    out_same = residual_target(B_G.copy(), B_G, states, PHYS_FRAME, PHYS_FRAME)
    for t in [0, 16]:
        assert np.allclose(out_same[t]["r"], 0.0, atol=1e-9)
        assert np.array_equal(out_same[t]["U"], np.asarray(states[t][1]))
    # different B_x -> nonzero r
    B_x = B_G + rng.normal(0, 0.5, B_G.shape); np.fill_diagonal(B_x, 0.0)
    out_diff = residual_target(B_x, B_G, states, PHYS_FRAME, PHYS_FRAME)
    assert np.abs(out_diff[16]["r"]).max() > 1e-6

def test_residual_target_frame_guard():
    from hidden_residual_graph import canonical_states, residual_target, PHYS_FRAME, N_BLOCKS
    B_G = np.zeros((N_BLOCKS, N_BLOCKS))
    states = canonical_states(B_G, t_list=[0])
    with pytest.raises(AssertionError):
        residual_target(B_G, B_G, states, "model_block", PHYS_FRAME)
