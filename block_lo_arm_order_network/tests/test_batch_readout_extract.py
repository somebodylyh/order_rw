"""BR-1 Task 2: tests for batch-mean attention extractor.

Mirrors the NR-1 test layout: heavy properties (shape, mean correctness,
seed reproducibility) consolidated into ONE GPU-touching test to avoid
paying the wikitext tokenization cost (~2 min per call) multiple times.
Two fast tests cover input validation without loading the model.
"""
import sys
import pathlib

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))


CKPT_5K = str(
    ROOT / "block_lo_arm_order_network/probe_results/attention_order_mlp/alt_from0_random/ckpt_step5000.pt"
)
# The actual file lives at repo-root/probe_results/...; fall back if symlink absent.
if not pathlib.Path(CKPT_5K).exists():
    CKPT_5K = str(ROOT / "probe_results/attention_order_mlp/alt_from0_random/ckpt_step5000.pt")
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


def test_invalid_M_rejected_fast():
    from batch_readout.extract_b_batch import extract_batch_mean_B
    with pytest.raises(ValueError, match="must be positive"):
        extract_batch_mean_B(CKPT_5K, M=0, batch_size=4, seed=0, device=DEV)


def test_invalid_batch_size_rejected_fast():
    from batch_readout.extract_b_batch import extract_batch_mean_B
    with pytest.raises(ValueError, match="must be positive"):
        extract_batch_mean_B(CKPT_5K, M=4, batch_size=0, seed=0, device=DEV)


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA + wikitext")
def test_extract_batch_mean_smoke():
    """One heavy test exercising shape, mean-correctness, and seed reproducibility.

    Calls extract_batch_mean_B twice with the same seed (must be bit-identical)
    and once with a different seed (must differ on chunks).
    """
    from batch_readout.extract_b_batch import extract_batch_mean_B

    M, B = 2, 4
    out1 = extract_batch_mean_B(
        CKPT_5K, M=M, batch_size=B, seed=42, device=DEV, return_per_sample=True
    )
    out2 = extract_batch_mean_B(
        CKPT_5K, M=M, batch_size=B, seed=42, device=DEV, return_per_sample=False
    )
    out3 = extract_batch_mean_B(
        CKPT_5K, M=M, batch_size=B, seed=43, device=DEV, return_per_sample=False
    )

    # Shapes & dtypes
    assert out1["B_batch"].shape == (M, 64, 64)
    assert out1["B_batch"].dtype == np.float32
    assert out1["chunks"].shape == (M, B)
    assert out1["chunks"].dtype == np.int64
    assert out1["B_per_sample"].shape == (M, B, 64, 64)
    assert out1["split"] == "train"

    # Diagonal zeroed
    diag = np.diagonal(out1["B_batch"], axis1=1, axis2=2)
    np.testing.assert_array_equal(diag, np.zeros_like(diag))

    # B_batch equals the mean of B_per_sample (within float32 round-trip noise)
    expected_mean = out1["B_per_sample"].astype(np.float64).mean(axis=1).astype(np.float32)
    expected_mean[:, np.arange(64), np.arange(64)] = 0.0  # match diagonal re-zero
    np.testing.assert_allclose(out1["B_batch"], expected_mean, rtol=0, atol=0)

    # Bit-reproducible with same seed
    np.testing.assert_array_equal(out1["B_batch"], out2["B_batch"])
    np.testing.assert_array_equal(out1["chunks"], out2["chunks"])

    # Different seed -> different chunks
    assert not np.array_equal(out1["chunks"], out3["chunks"])
