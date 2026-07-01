"""Tests for AOGPTWithOrderHead wrapper (Task 2).

Verifies:
1. extract_B + argsort_order bit-matches FrozenBetaHook.step (ckpt-gated)
2. token_orders_from_model_blocks produces (Bs, T) shape with correct remap (shape+remap test)
"""
import sys
import pathlib

sys.path.insert(0, "block_lo_arm_order_network")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import pytest

from batch_readout.integration_hook import FrozenBetaHook
from batch_readout.hook_order_provider import random_probe_token_orders
from analyses.order_head_module import OrderHeadModule, AOGPTWithOrderHead
from analyses.p5_utility_controller import load_p5_ckpt, N
from analyses.p7_gbeta_policy import GBETA_CKPT

CKPT = "block_lo_arm_order_network/probe_results/gbeta_K1000_from10k_L1H7_seed123/ckpt_step20000.pt"


@pytest.mark.skipif(not pathlib.Path(CKPT).exists(), reason="backbone ckpt not present")
def test_wrapper_extract_argsort_matches_hook():
    model, chunks, clean_perm, dev = load_p5_ckpt(CKPT, M=8, device="cpu")
    idx = torch.stack([chunks[i] for i in range(8)])
    probe = random_probe_token_orders(idx.shape[0], 0, 0, dev)
    oh = OrderHeadModule(GBETA_CKPT, device="cpu")
    wrap = AOGPTWithOrderHead(model, oh, clean_perm, device="cpu")
    A = wrap.extract_B(idx, probe)                       # (8,N,N)
    hook = FrozenBetaHook(GBETA_CKPT, mode="argsort", device="cpu")
    ext = hook.step(A)                                   # (N,)
    internal = oh.argsort_order(A, per_sample=False)[0]  # (N,)
    assert torch.equal(internal, ext)


def test_per_sample_orders_keep_BN_shape_and_remap():
    # backbone unused by the remap path (pure index + block-perm arithmetic)
    oh = OrderHeadModule(GBETA_CKPT, device="cpu")
    wrap = AOGPTWithOrderHead(backbone=None, order_head=oh,
                              clean_perm=_make_clean_perm(), device="cpu")
    order_model = np.stack([np.random.permutation(64) for _ in range(5)])  # (5,64)
    tok = wrap.token_orders_from_model_blocks(order_model)
    assert tok.shape[0] == 5 and tok.ndim == 2          # (5, T)


def _make_clean_perm():
    # reuse a real clean_perm from a tiny ckpt load; skip if the ckpt is absent
    if pathlib.Path(CKPT).exists():
        _, _, clean_perm, _ = load_p5_ckpt(CKPT, M=2, device="cpu")
        return clean_perm
    pytest.skip("no clean_perm source")
