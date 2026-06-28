# tests/test_pss_blockswap.py
import pathlib, sys
import torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.physical_signal_source import block_swap_chunk

def test_block_swap_swaps_content():
    c = torch.arange(256)
    out = block_swap_chunk(c, swaps=[(0, 5)], block_len=4)
    assert out[0:4].tolist() == [20, 21, 22, 23]          # block5 content now at block0
    assert out[20:24].tolist() == [0, 1, 2, 3]            # and vice versa
    assert out[8:12].tolist() == c[8:12].tolist()         # untouched blocks unchanged
