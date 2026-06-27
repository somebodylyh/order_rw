"""Stage-1 ablation: hook locality (ablating layer L leaves layers < L
bit-identical) and the single/LOO/full intervention partition.
"""
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.path_patch_handoff import (  # noqa: E402
    run_with_ablation,
    run_clean,
    make_probe_batch,
    stage1_variants,
)


def test_ablating_layer_l_leaves_earlier_layers_bit_identical(seed2_bundle):
    model, chunks, _ = seed2_bundle
    dev = torch.device("cpu")
    pc, po = make_probe_batch(chunks, n=16, rng=np.random.default_rng(1))
    attn_clean, tau_clean = run_clean(model, pc, po, dev)
    attn_abl, tau_abl = run_with_ablation(
        model, layer=1, head_indices=[0, 3, 5, 7],
        probe_chunks=pc, probe_orders=po, device=dev,
    )
    # layer 0 (< ablate layer 1) attention must be bit-identical
    assert torch.allclose(attn_clean[0], attn_abl[0])
    assert np.array_equal(tau_clean[0], tau_abl[0])
    # a downstream layer (2) should change
    assert not np.allclose(tau_clean[2], tau_abl[2])


def test_stage1_variants_partition():
    v = stage1_variants([0, 3, 5, 7])
    assert v["full"] == [[0, 3, 5, 7]]
    assert [sorted(x) for x in v["loo"]] == [[3, 5, 7], [0, 5, 7], [0, 3, 7], [0, 3, 5]]
    assert v["single"] == [[0], [3], [5], [7]]
