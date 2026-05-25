# tests/test_hidden_residual_hidden.py
import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "nanogpt-learned-order"))
from AOGPT import AOGPT, AOGPTConfig
from clean_training_protocol import build_clean_block_permutation, expand_model_blocks_to_token_order
from hidden_residual_hidden import extract_oracle_hidden, extract_causal_hidden

def _tiny_model(N=4, block_len=2, E=16):
    cfg = AOGPTConfig(vocab_size=32, n_layer=1, n_head=2, n_embd=E, dropout=0.0,
                      bias=False, block_size=N * block_len, block_order_block_len=block_len,
                      order_impl="block")
    m = AOGPT(cfg); m.eval(); return m, cfg

def test_oracle_hidden_shape_and_physical_remap():
    torch.manual_seed(0)
    N, bl, E = 4, 2, 16
    m, cfg = _tiny_model(N, bl, E)
    clean_perm = build_clean_block_permutation(N, seed=42)
    idx_model = torch.randint(0, 32, (3, N * bl))
    order_model = torch.arange(N).unsqueeze(0).expand(3, -1).contiguous()
    H = extract_oracle_hidden(m, idx_model, clean_perm, torch.device("cpu"), order_model)
    assert H.shape == (3, N, E)  # (n, N_blocks, n_embd), physical-block frame
    assert np.isfinite(H).all()

def test_causal_hidden_per_t_shape():
    torch.manual_seed(0)
    N, bl, E = 4, 2, 16
    m, cfg = _tiny_model(N, bl, E)
    idx_model = torch.randint(0, 32, (3, N * bl))
    canon = torch.arange(N).unsqueeze(0).expand(3, -1).contiguous()
    Hc = extract_causal_hidden(m, idx_model, torch.device("cpu"), canon, t_list=[0, 2])
    assert set(Hc.keys()) == {0, 2}
    assert Hc[0].shape == (3, E) and Hc[2].shape == (3, E)


def test_oracle_hidden_remap_follows_clean_perm():
    """Verify the model->physical block remap is semantically correct (not just shape).

    The model forward does NOT depend on clean_perm; it only affects the final
    h_blk_phys[:, inv, :] = h_blk_model scatter.  Running extract_oracle_hidden
    twice with identical inputs but different perms yields the same h_blk_model
    remapped to different physical slots.

    With identity perm:   H_id[:, m, :]      == h_model[:, m, :]
    With swap [1,0,2,3]:  H_sw[:, inv[m], :] == h_model[:, m, :]
    Therefore:            H_sw[:, inv_sw, :]  == H_id  (for inv_sw = [1,0,2,3])
    """
    from clean_training_protocol import CleanPermutation
    torch.manual_seed(0)
    N, bl, E = 4, 2, 16
    m, cfg = _tiny_model(N, bl, E)
    idx_model = torch.randint(0, 32, (3, N * bl))
    order_model = torch.arange(N).unsqueeze(0).expand(3, -1).contiguous()
    dev = torch.device("cpu")

    # identity perm: model block m -> physical slot m
    perm_id = CleanPermutation.from_model_to_phys(torch.arange(N))
    H_id = extract_oracle_hidden(m, idx_model, perm_id, dev, order_model)

    # swap perm: model block 0 -> phys 1, model block 1 -> phys 0
    perm_sw = CleanPermutation.from_model_to_phys(torch.tensor([1, 0, 2, 3]))
    H_sw = extract_oracle_hidden(m, idx_model, perm_sw, dev, order_model)

    inv_sw = perm_sw.inv_perm_model_to_phys.numpy()   # [1, 0, 2, 3]

    # After inverting the swap, both outputs must agree in every value
    np.testing.assert_allclose(
        H_sw[:, inv_sw, :], H_id,
        rtol=1e-5, atol=1e-6,
        err_msg="model->phys remap is wrong: H_sw[:, inv_sw, :] != H_id",
    )

    # Sanity: the two outputs must NOT be trivially identical (blocks 0 and 1 moved)
    assert not np.allclose(H_sw, H_id), (
        "H_sw and H_id are identical — swap perm had no effect, remap is likely identity regardless"
    )
