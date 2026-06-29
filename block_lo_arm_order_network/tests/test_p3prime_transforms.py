import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.p3prime_causal_verify import causal_uniform_transform


def test_uniform_rows_sum_to_one_on_support_and_untouched_others():
    rng = np.random.default_rng(0)
    L, H, T = 2, 3, 5
    attn = np.zeros((L, H, T, T))
    tril = np.tril(np.ones((T, T)))
    for l in range(L):
        for h in range(H):
            raw = rng.random((T, T)) * tril
            attn[l, h] = raw / raw.sum(axis=-1, keepdims=True)
    out = causal_uniform_transform([1])(attn.copy(), layer=0)
    row = out[0, 1, 3]
    assert np.allclose(row[:4], 0.25) and np.allclose(row[4:], 0.0)
    assert np.allclose(out[0, 0], attn[0, 0])
    assert np.allclose(out[1], attn[1])
