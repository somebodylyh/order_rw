"""BR-1 Task 7: tests for Plackett-Luce sampler + argsort."""
import sys
import pathlib

import pytest
import torch
from scipy.stats import kendalltau

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))


def test_argsort_is_valid_permutation_and_earliest_is_argmax():
    from batch_readout.pl_sampling import pl_argsort
    z = torch.tensor([[3.0, 1.0, 5.0, 0.0]])
    sig = pl_argsort(z)
    assert sig.shape == (1, 4)
    assert sig.dtype == torch.int64
    assert sorted(sig[0].tolist()) == [0, 1, 2, 3]
    assert sig[0, 0].item() == 2  # argmax


def test_sample_is_valid_permutation():
    from batch_readout.pl_sampling import pl_sample
    torch.manual_seed(0)
    z = torch.randn(4, 64)
    sig = pl_sample(z, tau=1.0)
    assert sig.shape == (4, 64)
    assert sig.dtype == torch.int64
    for row in sig:
        assert sorted(row.tolist()) == list(range(64))


def test_sample_seed_determinism():
    from batch_readout.pl_sampling import pl_sample
    z = torch.randn(2, 32)
    a = pl_sample(z, tau=0.5, generator=torch.Generator().manual_seed(0))
    b = pl_sample(z, tau=0.5, generator=torch.Generator().manual_seed(0))
    assert torch.equal(a, b)


def test_sample_distinct_under_different_seeds():
    from batch_readout.pl_sampling import pl_sample
    z = torch.randn(2, 32)
    a = pl_sample(z, tau=0.5, generator=torch.Generator().manual_seed(0))
    b = pl_sample(z, tau=0.5, generator=torch.Generator().manual_seed(1))
    assert not torch.equal(a, b)


def test_cold_temperature_approaches_argsort():
    from batch_readout.pl_sampling import pl_sample, pl_argsort
    torch.manual_seed(0)
    z = torch.randn(50, 32)
    expected = pl_argsort(z)
    sig = pl_sample(z, tau=1e-5, generator=torch.Generator().manual_seed(0))
    taus = [kendalltau(sig[i].numpy(), expected[i].numpy())[0] for i in range(50)]
    assert sum(taus) / len(taus) > 0.99, sum(taus) / len(taus)


def test_rejects_cuda_input():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA to construct the bad-input case")
    from batch_readout.pl_sampling import pl_sample
    z = torch.randn(2, 8, device="cuda:0")
    with pytest.raises(ValueError, match="expects z on CPU"):
        pl_sample(z, tau=1.0)


def test_rejects_nonpositive_tau():
    from batch_readout.pl_sampling import pl_sample
    z = torch.zeros(1, 4)
    with pytest.raises(ValueError, match="tau must be > 0"):
        pl_sample(z, tau=0.0)
