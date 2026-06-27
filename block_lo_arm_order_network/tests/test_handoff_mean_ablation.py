"""Mean-ablation hook replaces exactly the target head columns with their
position-wise batch mean, and an empty ablation is bit-identical to clean.
"""
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.path_patch_handoff import (  # noqa: E402
    mean_ablation_prehook,
    run_with_ablation,
    run_clean,
    make_probe_batch,
)


def test_prehook_replaces_only_target_head_columns_with_batch_mean():
    n_head, hs = 8, 4
    C = n_head * hs
    B, T = 3, 5
    y = torch.randn(B, T, C)
    hook = mean_ablation_prehook(head_indices=[2], n_head=n_head)
    (y2,) = hook(module=None, args=(y.clone(),))
    sl = slice(2 * hs, 3 * hs)
    # target head: position-wise batch mean (avg over B, keep T)
    exp = y[:, :, sl].mean(dim=0, keepdim=True).expand(B, -1, -1)
    assert torch.allclose(y2[:, :, sl], exp)
    # every other column untouched
    mask = torch.ones(C, dtype=torch.bool)
    mask[sl] = False
    assert torch.allclose(y2[:, :, mask], y[:, :, mask])


def test_empty_ablation_is_a_noop():
    n_head, hs = 8, 4
    y = torch.randn(2, 3, n_head * hs)
    hook = mean_ablation_prehook(head_indices=[], n_head=n_head)
    assert hook(module=None, args=(y,)) is None  # no-op -> hook returns None


def test_empty_ablation_is_bit_identical(seed2_bundle):
    model, chunks, _ = seed2_bundle
    dev = torch.device("cpu")
    pc, po = make_probe_batch(chunks, n=16, rng=np.random.default_rng(0))
    _, tau_clean = run_clean(model, pc, po, dev)
    _, tau_empty = run_with_ablation(
        model, layer=1, head_indices=[], probe_chunks=pc, probe_orders=po, device=dev
    )
    assert np.array_equal(tau_clean, tau_empty)  # Delta == 0 exactly
