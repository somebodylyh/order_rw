"""Phase 2 hook: in-loop selected-head B1 extraction matches the offline path.

The frozen g_β hook must be fed the SAME physical-frame B1 block graph it was
pretrained on (per_head_order_scan none_mode='b1', one selected head). The new
in-loop helper must therefore, for head (l,h), reproduce exactly
`_attn_to_A_block_b1_vec(attn[l,h], reveal_tokens, inv_perm)` per sample —
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
from per_head_order_scan import _attn_to_A_block_b1_vec  # noqa: E402


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
        ref = _attn_to_A_block_b1_vec(attn[head[0], b, head[1]].numpy(),
                                      probe[b].numpy(), inv_perm)
        assert np.allclose(out[b].numpy(), ref, atol=1e-6)


# ---------------------------------------------------------------------------
# All-layer extraction tests
# ---------------------------------------------------------------------------

def test_extract_all_layers_all_heads_shape():
    """extract_all_layers_all_heads_A_for_batch returns (B, L*H, N, N)."""
    from batch_readout.hook_order_provider import extract_all_layers_all_heads_A_for_batch
    L, B, H = 4, 2, 3  # 4 layers, batch=2, 3 heads/layer → 12 total heads
    rng = np.random.default_rng(4)
    attn = torch.from_numpy(rng.random((L, B, H, SEQ_LEN + 1, SEQ_LEN + 1), dtype=np.float64))
    model = _FakeAOGPT(attn)
    clean_perm = CleanPermutation.from_model_to_phys(torch.arange(N))
    idx_batch = torch.zeros((B, SEQ_LEN), dtype=torch.long)
    probe = _random_token_orders(B, seed=9)

    out = extract_all_layers_all_heads_A_for_batch(
        model, idx_batch, clean_perm,
        torch.device("cpu"), probe, none_mode="b1",
    )
    assert out.shape == (B, L * H, N, N)
    # Diagonal is zero
    for bi in range(B):
        for hi in range(L * H):
            assert torch.all(out[bi, hi].diagonal() == 0.0)


def test_all_layer_extraction_is_deterministic():
    """Same inputs → same output."""
    from batch_readout.hook_order_provider import extract_all_layers_all_heads_A_for_batch
    L, B, H = 2, 1, 2
    rng = np.random.default_rng(5)
    attn = torch.from_numpy(rng.random((L, B, H, SEQ_LEN + 1, SEQ_LEN + 1), dtype=np.float64))
    model = _FakeAOGPT(attn)
    clean_perm = CleanPermutation.from_model_to_phys(torch.arange(N))
    idx_batch = torch.zeros((B, SEQ_LEN), dtype=torch.long)
    probe = _random_token_orders(B, seed=11)

    out1 = extract_all_layers_all_heads_A_for_batch(
        model, idx_batch, clean_perm, torch.device("cpu"), probe, none_mode="b1")
    out2 = extract_all_layers_all_heads_A_for_batch(
        model, idx_batch, clean_perm, torch.device("cpu"), probe, none_mode="b1")
    assert torch.allclose(out1, out2, atol=1e-6)


def test_all_layer_extraction_aggregates_all_layers():
    """Verify that changing any layer's attention changes the output, proving
    all layers contribute."""
    from batch_readout.hook_order_provider import extract_all_layers_all_heads_A_for_batch
    L, B, H = 4, 1, 2
    rng = np.random.default_rng(6)
    attn = torch.from_numpy(rng.random((L, B, H, SEQ_LEN + 1, SEQ_LEN + 1), dtype=np.float64))
    model = _FakeAOGPT(attn)
    clean_perm = CleanPermutation.from_model_to_phys(torch.arange(N))
    idx_batch = torch.zeros((B, SEQ_LEN), dtype=torch.long)
    probe = _random_token_orders(B, seed=13)

    out_all = extract_all_layers_all_heads_A_for_batch(
        model, idx_batch, clean_perm, torch.device("cpu"), probe, none_mode="b1")

    # Modify L3 head 0, verify L3*H+0 entry changes
    attn_modified = attn.clone()
    attn_modified[3, 0, 0] = torch.randn(SEQ_LEN + 1, SEQ_LEN + 1, dtype=torch.float64)
    model2 = _FakeAOGPT(attn_modified)
    out_mod = extract_all_layers_all_heads_A_for_batch(
        model2, idx_batch, clean_perm, torch.device("cpu"), probe, none_mode="b1")

    # The modified head (index = 3*H+0 = 6) should differ
    modified_head_idx = 3 * H + 0
    assert not torch.allclose(out_all[0, modified_head_idx], out_mod[0, modified_head_idx], atol=1e-6)
    # But unmodified heads should be unchanged (L0H0 = index 0)
    assert torch.allclose(out_all[0, 0], out_mod[0, 0], atol=1e-6)


# ---------------------------------------------------------------------------
# H-override test (checkpoint compatibility)
# ---------------------------------------------------------------------------

def test_head_gated_gbeta_h_override():
    """Constructing HeadGatedGBeta with different H than checkpoint should work
    via strict=False load, because gate_net and readout are H-agnostic."""
    from batch_readout.head_gated_gbeta import build_head_gated_gbeta
    import tempfile, os

    # Build a small model with H=4 and save it
    model_h4 = build_head_gated_gbeta("head_gated", N=64, H=4, gate_mode="topk", topk=2,
                                       readout_hidden=(64,), gate_hidden=(32,))
    state = {"model": model_h4.state_dict(), "config": {"variant": "head_gated", "H": 4,
              "gate_mode": "topk", "topk": 2, "n_layers": 4,
              "readout_hidden": (64,), "gate_hidden": (32,)}}
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(state, f.name)
        ckpt_path = f.name

    try:
        # Load with H=32 (all-layer)
        state_loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_h32 = build_head_gated_gbeta("head_gated", N=64, H=32, gate_mode="topk", topk=4,
                                            readout_hidden=(64,), gate_hidden=(32,))
        missing, unexpected = model_h32.load_state_dict(state_loaded["model"], strict=False)
        # Should have no missing keys (gate_net and readout shapes are H-independent)
        assert not missing, f"Unexpected missing keys: {missing}"
        # gate_net operates per-head: input=(N*N,), output=scalar. Same for any H.
        model_h32.eval()

        # Verify forward pass works
        B_heads = torch.randn(2, 32, 64, 64)
        scores, aux = model_h32(B_heads)
        assert scores.shape == (2, 64)
        assert aux["gate_weights"].shape == (2, 32)
    finally:
        os.unlink(ckpt_path)


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
