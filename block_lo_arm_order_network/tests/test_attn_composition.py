import numpy as np
from batch_readout.attn_composition import head_weights, composition_scores


def test_head_weights_shapes():
    n_embd, n_head = 8, 2
    head_dim = n_embd // n_head
    c_attn_w = np.random.RandomState(0).randn(3 * n_embd, n_embd)
    c_proj_w = np.random.RandomState(1).randn(n_embd, n_embd)
    w = head_weights(c_attn_w, c_proj_w, n_head, head=1, n_embd=n_embd)
    assert w["W_Q"].shape == (n_embd, head_dim)
    assert w["W_O"].shape == (head_dim, n_embd)


def test_composition_scores_in_unit_interval():
    n_embd, n_head = 8, 2
    rs = np.random.RandomState(2)
    c_attn_w = rs.randn(3 * n_embd, n_embd)
    c_proj_w = rs.randn(n_embd, n_embd)
    up = head_weights(c_attn_w, c_proj_w, n_head, 0, n_embd)
    down = head_weights(c_attn_w, c_proj_w, n_head, 1, n_embd)
    s = composition_scores(up, down)
    for key in ("Q", "K", "V"):
        assert 0.0 <= s[key] <= 1.0 + 1e-9


def test_layer_pair_composition_shapes():
    from batch_readout.attn_composition import layer_pair_composition

    n_embd, n_head, L = 8, 2, 3
    rs = np.random.RandomState(3)
    c_attn_ws = [rs.randn(3 * n_embd, n_embd) for _ in range(L)]
    c_proj_ws = [rs.randn(n_embd, n_embd) for _ in range(L)]
    pairs = [(0, 1), (0, 2), (1, 2)]
    out = layer_pair_composition(c_attn_ws, c_proj_ws, n_head, n_embd, pairs)
    assert set(out.keys()) == set(pairs)
    assert out[(0, 1)].shape == (n_head, n_head, 3)  # (up_head, down_head, {Q,K,V})
