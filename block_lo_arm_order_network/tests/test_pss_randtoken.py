# tests/test_pss_randtoken.py
import pathlib, sys
import numpy as np, torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.physical_signal_source import random_token_chunk

def test_random_token_in_range_and_changed():
    c = torch.arange(256)
    out = random_token_chunk(c, vocab_size=50, rng=np.random.default_rng(0))
    assert out.shape == c.shape and int(out.max()) < 50
    assert not torch.equal(out, c)
