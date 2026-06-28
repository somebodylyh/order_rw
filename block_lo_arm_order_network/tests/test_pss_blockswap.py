import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.physical_signal_source import block_swap_chunk, cross_sample_replace


def test_block_swap_swaps_disjoint_blocks_without_mutating_input():
    chunk = torch.arange(24)
    original = chunk.clone()

    out = block_swap_chunk(chunk, swaps=[(0, 5), (1, 3)], block_len=4)

    assert out.tolist() == [
        20, 21, 22, 23,
        12, 13, 14, 15,
        8, 9, 10, 11,
        4, 5, 6, 7,
        16, 17, 18, 19,
        0, 1, 2, 3,
    ]
    assert torch.equal(chunk, original)
    assert out.data_ptr() != chunk.data_ptr()


def test_cross_sample_replace_copies_selected_blocks_only_without_mutation():
    chunk = torch.arange(24)
    donor = torch.arange(100, 124)
    chunk_original = chunk.clone()
    donor_original = donor.clone()

    out = cross_sample_replace(chunk, donor, blocks=[1, 4], block_len=4)

    expected = chunk.clone()
    expected[4:8] = donor[4:8]
    expected[16:20] = donor[16:20]
    assert torch.equal(out, expected)
    assert torch.equal(chunk, chunk_original)
    assert torch.equal(donor, donor_original)
    assert out.data_ptr() != chunk.data_ptr()


@pytest.mark.parametrize("operation", ["swap", "replace"])
def test_empty_selection_returns_an_independent_clone(operation):
    chunk = torch.arange(8)
    if operation == "swap":
        out = block_swap_chunk(chunk, swaps=[], block_len=4)
    else:
        out = cross_sample_replace(chunk, chunk.clone(), blocks=[], block_len=4)

    assert torch.equal(out, chunk)
    assert out.data_ptr() != chunk.data_ptr()


@pytest.mark.parametrize("bad_chunk", [[1, 2, 3, 4], torch.ones(2, 4), torch.tensor([])])
@pytest.mark.parametrize("operation", ["swap", "replace"])
def test_chunk_must_be_a_nonempty_one_dimensional_tensor(bad_chunk, operation):
    with pytest.raises((TypeError, ValueError)):
        if operation == "swap":
            block_swap_chunk(bad_chunk, swaps=[], block_len=4)
        else:
            donor = bad_chunk.clone() if isinstance(bad_chunk, torch.Tensor) else bad_chunk
            cross_sample_replace(bad_chunk, donor, blocks=[], block_len=4)


@pytest.mark.parametrize("bad_block_len", [True, False, 0, -1, 2.5, "4", 3])
@pytest.mark.parametrize("operation", ["swap", "replace"])
def test_block_len_must_be_positive_integral_and_divide_chunk_length(
        bad_block_len, operation):
    chunk = torch.arange(8)
    with pytest.raises((TypeError, ValueError)):
        if operation == "swap":
            block_swap_chunk(chunk, swaps=[], block_len=bad_block_len)
        else:
            cross_sample_replace(chunk, chunk.clone(), blocks=[], block_len=bad_block_len)


@pytest.mark.parametrize("bad_pair", [
    (0, 0),       # self-pair
    (-1, 0),      # negative endpoint
    (0, 2),       # endpoint equal to block count
    (0.0, 1),     # non-integral endpoint
    (True, 1),    # bool is not an index
])
def test_block_swap_rejects_invalid_pairs(bad_pair):
    with pytest.raises((TypeError, ValueError)):
        block_swap_chunk(torch.arange(8), swaps=[bad_pair], block_len=4)


@pytest.mark.parametrize("swaps", [
    [(0, 1), (0, 2)],
    [(0, 1), (2, 1)],
    [(0, 1), (1, 2)],
])
def test_block_swap_rejects_reused_endpoints(swaps):
    with pytest.raises(ValueError, match="reused|disjoint"):
        block_swap_chunk(torch.arange(12), swaps=swaps, block_len=4)


@pytest.mark.parametrize("bad_block", [-1, 2, 0.0, True])
def test_cross_sample_replace_rejects_invalid_block_indices(bad_block):
    chunk = torch.arange(8)
    with pytest.raises((TypeError, ValueError)):
        cross_sample_replace(chunk, chunk.clone(), blocks=[bad_block], block_len=4)


def test_cross_sample_replace_rejects_duplicate_blocks():
    chunk = torch.arange(8)
    with pytest.raises(ValueError, match="unique|duplicate"):
        cross_sample_replace(chunk, chunk.clone(), blocks=[1, 1], block_len=4)


@pytest.mark.parametrize("donor", [
    torch.arange(4),
    torch.arange(8, dtype=torch.float32),
])
def test_cross_sample_replace_requires_matching_donor_shape_and_dtype(donor):
    chunk = torch.arange(8)
    with pytest.raises(ValueError):
        cross_sample_replace(chunk, donor, blocks=[0], block_len=4)


def test_cross_sample_replace_requires_tensor_donor():
    with pytest.raises(TypeError):
        cross_sample_replace(torch.arange(8), list(range(8)), blocks=[0], block_len=4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cross_sample_replace_requires_matching_donor_device():
    chunk = torch.arange(8)
    donor = chunk.cuda()
    with pytest.raises(ValueError, match="device"):
        cross_sample_replace(chunk, donor, blocks=[0], block_len=4)
