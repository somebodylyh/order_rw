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
    clean_perm = build_clean_block_permutation(N, seed=42)
    idx_model = torch.randint(0, 32, (3, N * bl))
    canon = torch.arange(N).unsqueeze(0).expand(3, -1).contiguous()
    Hc = extract_causal_hidden(m, idx_model, clean_perm, torch.device("cpu"), canon, t_list=[0, 2])
    assert set(Hc.keys()) == {0, 2}
    assert Hc[0].shape == (3, E) and Hc[2].shape == (3, E)
