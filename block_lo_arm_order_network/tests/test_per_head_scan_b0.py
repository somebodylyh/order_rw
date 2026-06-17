import pathlib
import numpy as np
import pytest

from per_head_order_scan import (
    _attn_to_A_block_b0_vec,
    _attn_to_A_block_b1_vec,
    _attn_to_A_block_loss_aligned_content_vec,
    _attn_to_A_block_loss_aligned_with_none_vec,
    _attn_to_A_block_predictor_vec,
    _per_sample_A,
    _attn_to_A_block_vec,
)
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


def test_per_sample_A_none_mode_dispatch():
    attn, reveal_tokens, inv_perm = _random_inputs()           # (L,H,T+1,T+1)
    a_old, _ = _per_sample_A(attn, reveal_tokens, inv_perm, n_top=4, none_mode="old")
    a_b0, _ = _per_sample_A(attn, reveal_tokens, inv_perm, n_top=4, none_mode="b0")
    a_loss, _ = _per_sample_A(attn, reveal_tokens, inv_perm, n_top=4, none_mode="loss_aligned")
    # OLD path must equal the existing OLD aggregator exactly (no behavior change).
    np.testing.assert_allclose(a_old, _attn_to_A_block_vec(attn, reveal_tokens, inv_perm), rtol=1e-5, atol=1e-6)
    # B0 path must equal the new B0 aggregator and differ from OLD.
    np.testing.assert_allclose(a_b0, _attn_to_A_block_b0_vec(attn, reveal_tokens, inv_perm), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        a_loss,
        _attn_to_A_block_loss_aligned_content_vec(attn, reveal_tokens, inv_perm),
        rtol=1e-5,
        atol=1e-6,
    )
    assert not np.allclose(a_old, a_b0)


def test_per_sample_A_unknown_none_mode_raises():
    attn, reveal_tokens, inv_perm = _random_inputs()
    with pytest.raises(ValueError):
        _per_sample_A(attn, reveal_tokens, inv_perm, n_top=4, none_mode="bogus")


def test_predictor_vec_uses_shifted_predictor_blocks_without_physical_remap():
    seq_len = 12
    num_blocks = 3
    block_len = 4
    attn = np.zeros((seq_len + 1, seq_len + 1), dtype=np.float32)
    for i in range(seq_len + 1):
        for j in range(seq_len + 1):
            attn[i, j] = 100 * i + j

    reveal_tokens = np.array([8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3], dtype=np.int64)
    inv_perm = np.array([2, 1, 0], dtype=np.int64)

    got = _attn_to_A_block_predictor_vec(
        attn,
        reveal_tokens,
        inv_perm,
        seq_len=seq_len,
        num_blocks=num_blocks,
        block_len=block_len,
    )
    expected = attn[:-1, :-1].reshape(
        num_blocks, block_len, num_blocks, block_len
    ).mean(axis=(1, 3))
    np.fill_diagonal(expected, 0.0)

    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_b1_vec_maps_shifted_query_zero_to_none_block():
    seq_len = 8
    num_blocks = 2
    block_len = 4
    attn = np.zeros((seq_len + 1, seq_len + 1), dtype=np.float32)
    for i in range(seq_len + 1):
        for j in range(seq_len + 1):
            attn[i, j] = 100 * i + j

    reveal_tokens = np.array([4, 5, 6, 7, 0, 1, 2, 3], dtype=np.int64)
    inv_perm = np.array([1, 0], dtype=np.int64)

    got = _attn_to_A_block_b1_vec(
        attn,
        reveal_tokens,
        inv_perm,
        seq_len=seq_len,
        num_blocks=num_blocks,
        block_len=block_len,
    )

    labels = np.empty(seq_len, dtype=np.int64)
    labels[0] = 0
    labels[1:] = inv_perm[reveal_tokens[:-1] // block_len]
    counts = np.bincount(labels, minlength=num_blocks).astype(np.float64)
    assert np.all(counts > 0)
    S = np.zeros((num_blocks, seq_len), dtype=np.float64)
    S[labels, np.arange(seq_len)] = 1.0
    S = S / counts[:, None]

    expected = np.einsum("bt,tu,cu->bc", S, attn[:-1, :-1], S, optimize=True)
    np.fill_diagonal(expected, 0.0)

    np.testing.assert_allclose(got, expected.astype(np.float32), rtol=1e-6, atol=1e-6)


def test_loss_aligned_uses_target_query_and_source_key_labels_asymmetrically():
    seq_len = 8
    num_blocks = 2
    block_len = 4
    attn = np.zeros((seq_len + 1, seq_len + 1), dtype=np.float32)
    # Make each query-key cell uniquely identifiable.
    for q in range(seq_len + 1):
        for k in range(seq_len + 1):
            attn[q, k] = 100 * q + k

    reveal_tokens = np.arange(seq_len, dtype=np.int64)
    inv_perm = np.arange(num_blocks, dtype=np.int64)

    got_with_none = _attn_to_A_block_loss_aligned_with_none_vec(
        attn,
        reveal_tokens,
        inv_perm,
        seq_len=seq_len,
        num_blocks=num_blocks,
        block_len=block_len,
    )
    got_content = _attn_to_A_block_loss_aligned_content_vec(
        attn,
        reveal_tokens,
        inv_perm,
        seq_len=seq_len,
        num_blocks=num_blocks,
        block_len=block_len,
    )

    # Query block 1 must be rows q=4..7 (predicting x4..x7). Key/source block 0
    # must be key positions k=1..4 ([x0..x3]), not the same q-side label set.
    expected_t1_s0 = attn[:seq_len, :seq_len][4:8, 1:5].mean()
    expected_t0_none = attn[:seq_len, :seq_len][0:4, 0].mean()
    assert got_with_none.shape == (num_blocks, num_blocks + 1)
    np.testing.assert_allclose(got_with_none[1, 1], expected_t1_s0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(got_content[1, 0], expected_t1_s0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(got_with_none[0, 0], expected_t0_none, rtol=1e-6, atol=1e-6)

    # The legacy B1 fold groups q=4 with source block0 under the old symmetric
    # label convention; the loss-aligned value must differ on this constructed map.
    old_b1 = _attn_to_A_block_b1_vec(
        attn,
        reveal_tokens,
        inv_perm,
        seq_len=seq_len,
        num_blocks=num_blocks,
        block_len=block_len,
    )
    assert not np.isclose(old_b1[1, 0], got_content[1, 0])


def test_model_vec_equals_b0_with_identity_inv_perm():
    """With identity inv_perm, model-mode B must equal B0 exactly."""
    from per_head_order_scan import _attn_to_A_block_model_vec
    N_small, bl = 4, 4
    sl = N_small * bl
    rng = np.random.RandomState(42)
    attn = rng.rand(sl + 1, sl + 1).astype(np.float32)
    reveal_tokens = np.arange(sl, dtype=np.int64)
    inv_perm_id = np.arange(N_small, dtype=np.int64)

    A_b0 = _attn_to_A_block_b0_vec(attn, reveal_tokens, inv_perm_id, seq_len=sl, num_blocks=N_small, block_len=bl)
    A_model = _attn_to_A_block_model_vec(attn, reveal_tokens, inv_perm_id, seq_len=sl, num_blocks=N_small, block_len=bl)
    np.testing.assert_allclose(A_b0, A_model, rtol=1e-6, atol=1e-6)


def test_model_vec_invariant_to_inv_perm():
    """Model-mode B must NOT depend on inv_perm (it ignores it)."""
    from per_head_order_scan import _attn_to_A_block_model_vec
    N_small, bl = 4, 4
    sl = N_small * bl
    rng = np.random.RandomState(42)
    attn = rng.rand(sl + 1, sl + 1).astype(np.float32)
    reveal_tokens = np.arange(sl, dtype=np.int64)

    inv_perm_a = np.array([0, 1, 2, 3], dtype=np.int64)
    inv_perm_b = np.array([2, 0, 3, 1], dtype=np.int64)

    A_a = _attn_to_A_block_model_vec(attn, reveal_tokens, inv_perm_a, seq_len=sl, num_blocks=N_small, block_len=bl)
    A_b = _attn_to_A_block_model_vec(attn, reveal_tokens, inv_perm_b, seq_len=sl, num_blocks=N_small, block_len=bl)
    np.testing.assert_allclose(A_a, A_b, rtol=1e-6, atol=1e-6)


def test_model_vec_differs_from_b0_with_swapped_inv_perm():
    """With non-identity inv_perm, model-mode and B0 must differ."""
    from per_head_order_scan import _attn_to_A_block_model_vec
    N_small, bl = 4, 4
    sl = N_small * bl
    rng = np.random.RandomState(42)
    attn = rng.rand(sl + 1, sl + 1).astype(np.float32)
    reveal_tokens = np.arange(sl, dtype=np.int64)
    inv_perm = np.array([2, 0, 3, 1], dtype=np.int64)

    A_b0 = _attn_to_A_block_b0_vec(attn, reveal_tokens, inv_perm, seq_len=sl, num_blocks=N_small, block_len=bl)
    A_model = _attn_to_A_block_model_vec(attn, reveal_tokens, inv_perm, seq_len=sl, num_blocks=N_small, block_len=bl)
    assert not np.allclose(A_b0, A_model)
