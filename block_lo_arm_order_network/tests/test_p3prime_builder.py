import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.p3prime_causal_verify import p3_canonical_b65
from analyses.physical_signal_source import carrier_b65_per_text


CKPT = ROOT / "runs/handoff_overnight/seed2/ckpt_step10000.pt"


@pytest.mark.skipif(not CKPT.exists(), reason="seed2 checkpoint artifact is absent")
def test_p3_builder_matches_carrier_b65_with_no_interventions():
    head = 2
    got = p3_canonical_b65(
        str(CKPT),
        0,
        [head],
        M=4,
        n_reveals=4,
        fixed_reveal_seed=0,
    )[head]
    ref = carrier_b65_per_text(
        str(CKPT),
        0,
        head,
        M=4,
        n_reveals=4,
        fixed_reveal_seed=0,
    )

    got_B, got_tau = got
    ref_B, ref_tau = ref

    assert np.allclose(np.asarray(got_tau), np.asarray(ref_tau), atol=1e-9)
    assert len(got_B) == len(ref_B)
    for got_item, ref_item in zip(got_B, ref_B):
        assert np.allclose(got_item, ref_item, atol=1e-9)


@pytest.mark.skipif(not CKPT.exists(), reason="seed2 checkpoint artifact is absent")
def test_uniform_causal_attention_transform_reduces_mean_abs_tau():
    head = 2

    def uniform_causal(attn_LH, layer):
        out = attn_LH.copy()
        support = np.tril(np.ones_like(out[layer, head], dtype=np.float64))
        denom = support.sum(axis=-1, keepdims=True)
        out[layer, head] = support / np.maximum(denom, 1.0)
        return out

    base_tau = np.asarray(
        p3_canonical_b65(
            str(CKPT),
            0,
            [head],
            M=4,
            n_reveals=4,
            fixed_reveal_seed=0,
        )[head][1]
    )
    patched_tau = np.asarray(
        p3_canonical_b65(
            str(CKPT),
            0,
            [head],
            M=4,
            n_reveals=4,
            fixed_reveal_seed=0,
            attn_transform=uniform_causal,
        )[head][1]
    )

    assert np.mean(np.abs(patched_tau)) < np.mean(np.abs(base_tau))
