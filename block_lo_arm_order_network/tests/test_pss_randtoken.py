import pathlib
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.physical_signal_source import random_token_chunk


def test_random_token_chunk_preserves_shape_dtype_device_and_bounds():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chunk = torch.arange(1024, dtype=torch.int32, device=device).reshape(32, 32)

    out = random_token_chunk(chunk, vocab_size=7, rng=np.random.default_rng(0))

    assert out.shape == chunk.shape
    assert out.dtype == chunk.dtype
    assert out.device == chunk.device
    assert int(out.min()) == 0
    assert int(out.max()) == 6


def test_random_token_chunk_is_reproducible_and_advances_generator():
    chunk = torch.zeros(128, dtype=torch.int64)
    rng = np.random.default_rng(123)

    first = random_token_chunk(chunk, 50, rng)
    second = random_token_chunk(chunk, 50, rng)
    replay = np.random.default_rng(123)

    assert torch.equal(first, random_token_chunk(chunk, 50, replay))
    assert torch.equal(second, random_token_chunk(chunk, 50, replay))
    assert not torch.equal(first, second)


def test_random_token_chunk_does_not_mutate_or_force_changes():
    chunk = torch.zeros((2, 3), dtype=torch.uint8)
    before = chunk.clone()

    out = random_token_chunk(chunk, vocab_size=1, rng=np.random.default_rng(9))

    assert torch.equal(chunk, before)
    assert torch.equal(out, chunk)
    assert out.data_ptr() != chunk.data_ptr()


@pytest.mark.parametrize(
    "chunk",
    [
        [1, 2],
        torch.empty(0, dtype=torch.long),
        torch.tensor([False, True]),
        torch.tensor([1.0]),
        torch.tensor([1 + 2j]),
    ],
)
def test_random_token_chunk_rejects_invalid_chunk(chunk):
    with pytest.raises((TypeError, ValueError)):
        random_token_chunk(chunk, 5, np.random.default_rng(0))


@pytest.mark.parametrize("vocab_size", [True, 1.5, 0, -1])
def test_random_token_chunk_rejects_invalid_vocab_size(vocab_size):
    with pytest.raises((TypeError, ValueError)):
        random_token_chunk(
            torch.zeros(2, dtype=torch.int16),
            vocab_size,
            np.random.default_rng(0),
        )


def test_random_token_chunk_rejects_unrepresentable_vocab_range():
    with pytest.raises(ValueError, match="represent"):
        random_token_chunk(
            torch.zeros(2, dtype=torch.int8),
            vocab_size=torch.iinfo(torch.int8).max + 2,
            rng=np.random.default_rng(0),
        )


@pytest.mark.parametrize("rng", [None, np.random.RandomState(0), object()])
def test_random_token_chunk_requires_numpy_generator(rng):
    with pytest.raises(TypeError, match="Generator"):
        random_token_chunk(torch.zeros(2, dtype=torch.long), 5, rng)
