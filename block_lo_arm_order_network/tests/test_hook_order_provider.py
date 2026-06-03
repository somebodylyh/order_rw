"""Phase 2 hook: in-loop selected-head B0 extraction matches the offline path.

The frozen g_β hook must be fed the SAME physical-frame B0 block graph it was
pretrained on (per_head_order_scan none_mode='b0', one selected head). The new
in-loop helper must therefore, for head (l,h), reproduce exactly
`_attn_to_A_block_b0_vec(attn[l,h], reveal_tokens, inv_perm)` per sample —
picking the right head and aligning each sample's own reveal order. A wrong head
index or a mis-aligned reveal would silently feed g_β out-of-distribution input.
"""
import sys
import pathlib

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))

from training_utils import SEQ_LEN, N, BLOCK_LEN  # noqa: E402
from clean_training_protocol import CleanPermutation, expand_model_blocks_to_token_order  # noqa: E402
from per_head_order_scan import _attn_to_A_block_b0_vec  # noqa: E402


class _FakeAOGPT:
    """forward_fn returns a preset attention stack, ignoring idx/orders."""
    def __init__(self, attn):  # attn: (L, B, H, T+1, T+1) torch
        self._attn = attn

    def eval(self):
        return self

    def forward_fn(self, idx, orders, return_attentions=False):
        L = self._attn.shape[0]
        attn_list = [self._attn[l] for l in range(L)]  # each (B, H, T+1, T+1)
        return None, None, attn_list


def _random_token_orders(B, seed):
    """B random physical-block permutations -> model token orders (B, SEQ_LEN)."""
    out = torch.empty((B, SEQ_LEN), dtype=torch.long)
    for b in range(B):
        g = torch.Generator(device="cpu"); g.manual_seed(seed + b)
        blocks = torch.randperm(N, generator=g)
        out[b] = expand_model_blocks_to_token_order(blocks.unsqueeze(0), BLOCK_LEN)[0]
    return out


def test_extract_matches_offline_agg_for_selected_head():
    from batch_readout.hook_order_provider import extract_selected_head_A_for_batch
    L, B, H = 2, 2, 3
    rng = np.random.default_rng(0)
    attn = torch.from_numpy(rng.random((L, B, H, SEQ_LEN + 1, SEQ_LEN + 1), dtype=np.float64))
    model = _FakeAOGPT(attn)
    clean_perm = CleanPermutation.from_model_to_phys(torch.arange(N))  # identity remap
    idx_batch = torch.zeros((B, SEQ_LEN), dtype=torch.long)
    probe = _random_token_orders(B, seed=7)

    head = (1, 2)
    out = extract_selected_head_A_for_batch(model, idx_batch, head, clean_perm,
                                            torch.device("cpu"), probe)
    assert out.shape == (B, N, N)
    inv_perm = clean_perm.inv_perm_model_to_phys.numpy()
    for b in range(B):
        ref = _attn_to_A_block_b0_vec(attn[head[0], b, head[1]].numpy(),
                                      probe[b].numpy(), inv_perm)
        assert np.allclose(out[b].numpy(), ref, atol=1e-6)


def test_different_head_selects_different_attention():
    from batch_readout.hook_order_provider import extract_selected_head_A_for_batch
    L, B, H = 2, 1, 3
    rng = np.random.default_rng(1)
    attn = torch.from_numpy(rng.random((L, B, H, SEQ_LEN + 1, SEQ_LEN + 1), dtype=np.float64))
    model = _FakeAOGPT(attn)
    clean_perm = CleanPermutation.from_model_to_phys(torch.arange(N))
    idx_batch = torch.zeros((B, SEQ_LEN), dtype=torch.long)
    probe = _random_token_orders(B, seed=3)
    a00 = extract_selected_head_A_for_batch(model, idx_batch, (0, 0), clean_perm, torch.device("cpu"), probe)
    a11 = extract_selected_head_A_for_batch(model, idx_batch, (1, 1), clean_perm, torch.device("cpu"), probe)
    assert not np.allclose(a00.numpy(), a11.numpy())
