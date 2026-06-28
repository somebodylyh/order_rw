"""Relayout chunks across layouts + training-anchor round-trip guard."""
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.position_prior_decomp import (  # noqa: E402
    relayout_chunks, clean_perm_from_layout, make_layouts)
from clean_training_protocol import build_clean_block_permutation  # noqa: E402


def test_relayout_to_training_is_identity():
    train = build_clean_block_permutation(64, seed=7)
    layouts = make_layouts(train, K=3, seed_base=10)
    chunks = torch.randint(0, 100, (4, 256))
    out0 = relayout_chunks(chunks, train, clean_perm_from_layout(layouts[0]))
    assert torch.equal(out0, chunks)  # layout_0 == training -> identity


def test_relayout_changes_under_different_layout():
    train = build_clean_block_permutation(64, seed=7)
    layouts = make_layouts(train, K=3, seed_base=10)
    chunks = torch.randint(0, 100, (4, 256))
    out2 = relayout_chunks(chunks, train, clean_perm_from_layout(layouts[2]))
    assert not torch.equal(out2, chunks)
