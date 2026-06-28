"""Frame-aware tau helpers: model-slot vs physical frame."""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.position_prior_decomp import tau_vs_arange, tau_two_frames  # noqa: E402
from batch_readout.l0_strict65 import build_model_frame_strict65  # noqa: E402


def _uniform_causal_B():
    """Convention-correct ascending B via the real strict65 pipeline (diag zeroed)."""
    T = 257
    att = np.zeros((1, 1, T, T), dtype=np.float32)
    for i in range(T):
        att[0, 0, i, : i + 1] = 1.0 / (i + 1)
    po = np.tile(np.arange(256, dtype=np.int64), (1, 1))
    return build_model_frame_strict65(att, po)[0, 0]  # (65,65)


def test_tau_vs_arange_perfect_and_reversed():
    assert tau_vs_arange(np.arange(10)) == 1.0
    assert tau_vs_arange(np.arange(10)[::-1]) == -1.0


def test_two_frames_translation():
    B = _uniform_causal_B()
    inv = np.arange(64)[::-1]  # reverse the 64 content blocks
    out = tau_two_frames(B, inv)
    assert out["tau_model_slot"] > 0.9          # uniform-causal rolls out ascending
    assert out["tau_physical"] < out["tau_model_slot"]  # reversed frame lowers it
