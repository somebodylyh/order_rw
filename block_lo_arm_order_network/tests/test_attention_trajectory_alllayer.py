import numpy as np
from attention_trajectory import extract_all_layer_B


def test_extract_all_layer_B_shape():
    L, S, H, T = 4, 2, 8, 256
    rng = np.random.default_rng(0)
    # causal-ish attention over T+1 tokens; rows sum ~1 not required for shape
    attn_list = [rng.random((S, H, T + 1, T + 1)).astype(np.float32) for _ in range(L)]
    probe_orders = np.tile(np.arange(T, dtype=np.int64), (S, 1))
    B = extract_all_layer_B(attn_list, probe_orders)
    assert B.shape == (L, S, H, 65, 65)
    # diagonal must be zero (none-separated convention)
    assert np.allclose(np.diagonal(B, axis1=3, axis2=4), 0.0)
