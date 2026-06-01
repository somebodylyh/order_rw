import pathlib
import numpy as np

from per_head_order_scan import _attn_to_A_block_b0_vec
from training_utils import SEQ_LEN, N, BLOCK_LEN

# Load the per-chunk reference agg_b0 from b0_fast.py without running its main loop.
_B0_FAST = pathlib.Path(__file__).resolve().parents[1] / "batch_readout/logs/per_head_scan/b0_fast.py"


def _load_agg_b0_reference():
    import sys
    src = _B0_FAST.read_text().split("idx_phys = get_idx_phys()")[0]  # cut before the main loop
    ns = {}
    saved = sys.argv
    sys.argv = [str(_B0_FAST)]  # b0_fast.py parses sys.argv[2] at module top; avoid pytest's argv
    try:
        exec(compile(src, str(_B0_FAST), "exec"), ns)
    finally:
        sys.argv = saved
    return ns["agg_b0"]


def _random_inputs(L=4, H=8, seed=0):
    rng = np.random.RandomState(seed)
    attn = rng.rand(L, H, SEQ_LEN + 1, SEQ_LEN + 1).astype(np.float32)
    attn /= attn.sum(axis=-1, keepdims=True)  # row-stochastic like real attention
    g = np.random.RandomState(seed + 1)
    reveal_tokens = g.permutation(SEQ_LEN).astype(np.int64)
    inv_perm = g.permutation(N).astype(np.int64)
    return attn, reveal_tokens, inv_perm


def test_b0_vec_matches_per_chunk_reference():
    agg_b0_ref = _load_agg_b0_reference()
    attn, reveal_tokens, inv_perm = _random_inputs()
    got = _attn_to_A_block_b0_vec(attn, reveal_tokens, inv_perm)   # (L,H,N,N)
    ref = agg_b0_ref(attn, reveal_tokens, inv_perm)                # (L,H,N,N)
    assert got.shape == (4, 8, N, N)
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)


def test_b0_vec_zero_diagonal_and_no_lead_dim():
    attn, reveal_tokens, inv_perm = _random_inputs()
    single = _attn_to_A_block_b0_vec(attn[0, 0], reveal_tokens, inv_perm)  # (N,N), no lead dim
    assert single.shape == (N, N)
    assert np.allclose(np.diag(single), 0.0)
