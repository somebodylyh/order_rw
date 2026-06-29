import pathlib
import sys

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from search_raw_none_separated_65_heads import (  # noqa: E402
    _attn_to_B65_raw_b1_with_none_vec,
    _attn_to_A_raw_loss_aligned_with_none_vec,
    _metrics_after_model_to_phys,
)


def test_raw_b1_with_none_keeps_none_as_node_zero():
    attn = np.zeros((5, 5), dtype=np.float32)
    attn[1, 0] = 2.0
    reveal_tokens = np.array([0, 1, 2, 3], dtype=np.int64)

    B65 = _attn_to_B65_raw_b1_with_none_vec(
        attn, reveal_tokens, seq_len=4, num_blocks=2, block_len=2
    )

    assert B65.shape == (3, 3)
    assert B65[0, 1] == 2.0


def test_raw_loss_aligned_labels_targets_by_model_block():
    attn = np.zeros((5, 5), dtype=np.float32)
    attn[2, 1] = 3.0
    reveal_tokens = np.array([0, 1, 2, 3], dtype=np.int64)

    A = _attn_to_A_raw_loss_aligned_with_none_vec(
        attn, reveal_tokens, seq_len=4, num_blocks=2, block_len=2
    )

    assert A.shape == (2, 3)
    assert A[1, 1] == 3.0


def test_metrics_after_model_to_phys_evaluates_only_after_rollout():
    metrics = _metrics_after_model_to_phys([2, 0, 1], np.array([1, 2, 0]))

    assert metrics["order_phys"] == [0, 1, 2]
    assert metrics["tau_vs_l2r"] == 1.0
    assert metrics["first_block"] == 0
