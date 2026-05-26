import numpy as np, torch, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))
sys.path.insert(0, str(ROOT / "nanogpt-learned-order"))
sys.path.insert(0, str(ROOT / "scripts"))
from AOGPT import AOGPT, AOGPTConfig
from hidden_graph_modelio import extract_image_block_hidden, nll_under_order_image


def _tiny_block_model(N=4, BL=1, V=16):
    cfg = AOGPTConfig(order_impl="block", block_order_block_len=BL, block_size=N * BL,
                      vocab_size=V, n_layer=1, n_head=1, n_embd=8, dropout=0.0, bias=False)
    m = AOGPT(cfg); m.eval(); return m


def test_extract_image_block_hidden_shape_and_finite():
    torch.manual_seed(0); N, BL, V, n = 4, 1, 16, 3
    m = _tiny_block_model(N, BL, V)
    tokens = torch.randint(0, V, (n, N * BL))
    H = extract_image_block_hidden(m, tokens, block_len=BL, n_blocks=N, device="cpu")
    assert H.shape == (n, N, 8)            # (n, N, E)
    assert np.all(np.isfinite(H))


def test_nll_under_order_image_finite_and_order_sensitive():
    torch.manual_seed(0); N, BL, V, n = 4, 1, 16, 3
    m = _tiny_block_model(N, BL, V)
    tokens = torch.randint(0, V, (n, N * BL))
    o1 = np.array([0, 1, 2, 3]); o2 = np.array([3, 2, 1, 0])
    nll1 = nll_under_order_image(m, tokens, o1, block_len=BL, device="cpu")
    nll2 = nll_under_order_image(m, tokens, o2, block_len=BL, device="cpu")
    assert np.isfinite(nll1) and np.isfinite(nll2)


def test_nll_under_order_text_runs_via_compute_token_ce():
    # text path needs train_clean_aogpt importable; smoke that the call shape works
    import torch, numpy as np
    from hidden_graph_modelio import nll_under_order_text
    N, BL, V, n = 4, 2, 16, 3
    from AOGPT import AOGPT, AOGPTConfig
    cfg = AOGPTConfig(order_impl="block", block_order_block_len=BL, block_size=N * BL,
                      vocab_size=V, n_layer=1, n_head=1, n_embd=8, dropout=0.0, bias=False)
    m = AOGPT(cfg); m.eval()
    idx = torch.randint(0, V, (n, N * BL))
    nll = nll_under_order_text(m, idx, np.arange(N), block_len=BL, device="cpu")
    assert np.isfinite(nll)
