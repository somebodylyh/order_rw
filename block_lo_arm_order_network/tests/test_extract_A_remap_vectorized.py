"""Engineering hotfix: vectorize the per-chunk physical-frame remap inside
extract_A_matrices (train_clean_aogpt.py:113-117).

This is an INDEPENDENT performance fix, NOT part of NR-1 scientific design.
Constraint: identical output as the original 256x256 Python double-loop on
the same (attn_content, phys_tokens) inputs.

The remap is:
    attn_phys[phys_tokens[rq], phys_tokens[rk]] += attn_content[rq, rk]
for every (rq, rk) in [0, SEQ_LEN) x [0, SEQ_LEN).

This test pins bit-identity between the legacy double-loop and the
vectorized np.add.at form, across:
  - random attn_content
  - random permutation phys_tokens
  - identity permutation phys_tokens (sanity)
  - both float32 and float64 dtypes
"""
import numpy as np
import pytest


SEQ_LEN = 256


def _legacy_double_loop(attn_content, phys_tokens):
    """Reference (slow) implementation copied verbatim from
    train_clean_aogpt.py:113-117. Used only as a bit-identity oracle.
    """
    out = np.zeros_like(attn_content)
    for rq in range(SEQ_LEN):
        pq = phys_tokens[rq]
        for rk in range(SEQ_LEN):
            out[pq, phys_tokens[rk]] += attn_content[rq, rk]
    return out


def _vectorized(attn_content, phys_tokens):
    """np.add.at fancy-index assignment. O(SEQ_LEN^2) work but vectorized in C."""
    out = np.zeros_like(attn_content)
    np.add.at(out, (phys_tokens[:, None], phys_tokens[None, :]), attn_content)
    return out


@pytest.mark.parametrize("seed", [0, 1, 42, 12345])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_vectorized_matches_legacy(seed, dtype):
    rng = np.random.default_rng(seed)
    attn_content = rng.standard_normal((SEQ_LEN, SEQ_LEN)).astype(dtype)
    phys_tokens = rng.permutation(SEQ_LEN).astype(np.int64)

    a_loop = _legacy_double_loop(attn_content, phys_tokens)
    a_vec = _vectorized(attn_content, phys_tokens)

    np.testing.assert_array_equal(a_loop, a_vec)


def test_identity_permutation_returns_input_unchanged():
    rng = np.random.default_rng(0)
    attn_content = rng.standard_normal((SEQ_LEN, SEQ_LEN)).astype(np.float32)
    phys_tokens = np.arange(SEQ_LEN, dtype=np.int64)

    a_loop = _legacy_double_loop(attn_content, phys_tokens)
    a_vec = _vectorized(attn_content, phys_tokens)

    np.testing.assert_array_equal(a_loop, attn_content)
    np.testing.assert_array_equal(a_vec, attn_content)


def test_diagonal_handled_consistently():
    """On a phys_tokens permutation, attn_content[i, i] maps to
    out[phys_tokens[i], phys_tokens[i]] -- a diagonal element of the
    permuted matrix. Verify both implementations agree on this.
    """
    rng = np.random.default_rng(7)
    attn_content = rng.standard_normal((SEQ_LEN, SEQ_LEN)).astype(np.float32)
    phys_tokens = rng.permutation(SEQ_LEN).astype(np.int64)

    a_loop = _legacy_double_loop(attn_content, phys_tokens)
    a_vec = _vectorized(attn_content, phys_tokens)

    diag_loop = a_loop[phys_tokens, phys_tokens]
    diag_vec = a_vec[phys_tokens, phys_tokens]
    expected_diag = np.diag(attn_content)
    np.testing.assert_array_equal(diag_loop, expected_diag)
    np.testing.assert_array_equal(diag_vec, expected_diag)


def test_accumulation_semantics_preserved():
    """The legacy loop uses += so if two distinct (rq, rk) pairs ever mapped to
    the same destination (pq, phys_tokens[rk]), they would accumulate. In
    extract_A_matrices the mapping is unique because phys_tokens is a
    permutation, so this never happens in practice -- but np.add.at preserves
    the += semantics in case future maintenance introduces collisions.

    We synthesize a collision by passing phys_tokens with duplicates and
    verify both versions still agree.
    """
    rng = np.random.default_rng(11)
    attn_content = rng.standard_normal((SEQ_LEN, SEQ_LEN)).astype(np.float32)
    # Force a collision: map all even rows to the same physical row 0
    phys_tokens = np.arange(SEQ_LEN, dtype=np.int64)
    phys_tokens[::2] = 0  # rows 0, 2, 4, ... all map to physical row 0

    a_loop = _legacy_double_loop(attn_content, phys_tokens)
    a_vec = _vectorized(attn_content, phys_tokens)
    np.testing.assert_array_equal(a_loop, a_vec)


def test_vectorized_is_faster():
    """Sanity check that the vectorized form is meaningfully faster.
    Not a hard gate; the point is to flag regressions where the
    vectorized version somehow becomes slower than the loop.
    """
    import time
    rng = np.random.default_rng(0)
    attn_content = rng.standard_normal((SEQ_LEN, SEQ_LEN)).astype(np.float32)
    phys_tokens = rng.permutation(SEQ_LEN).astype(np.int64)

    # Warm up Python's JIT-ish caches once
    _legacy_double_loop(attn_content, phys_tokens)
    _vectorized(attn_content, phys_tokens)

    t0 = time.perf_counter()
    for _ in range(5):
        _legacy_double_loop(attn_content, phys_tokens)
    t_loop = (time.perf_counter() - t0) / 5

    t0 = time.perf_counter()
    for _ in range(5):
        _vectorized(attn_content, phys_tokens)
    t_vec = (time.perf_counter() - t0) / 5

    # Conservative bound: vectorized should be at least 5x faster.
    assert t_vec * 5 < t_loop, (
        f"vectorized ({t_vec*1000:.2f}ms) not >= 5x faster than loop ({t_loop*1000:.2f}ms)"
    )
