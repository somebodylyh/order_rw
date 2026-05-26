import numpy as np, torch, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in ["block_lo_arm_order_network", "nanogpt-learned-order", "scripts"]:
    sys.path.insert(0, str(ROOT / p))
from AOGPT import AOGPT, AOGPTConfig
import causal_hidden_rollout as CHR


def _tiny(N=4, BL=2, V=16, E=8):
    cfg = AOGPTConfig(order_impl="block", block_order_block_len=BL, block_size=N * BL,
                      vocab_size=V, n_layer=1, n_head=1, n_embd=E, dropout=0.0, bias=False)
    m = AOGPT(cfg); m.eval(); return m


def test_context_hidden_at_step_shape():
    torch.manual_seed(0); N, BL, V, E, n = 4, 2, 16, 8, 3
    m = _tiny(N, BL, V, E)
    idx = torch.randint(0, V, (n, N * BL))
    prefix = [0, 1]
    completion = [2, 3]
    c = CHR.context_hidden_at_step(m, idx, prefix, completion, BL, device="cpu")
    assert c.shape == (n, E)
    assert np.all(np.isfinite(c))


def test_context_hidden_is_completion_invariant():
    torch.manual_seed(0); N, BL, V, E, n = 4, 2, 16, 8, 3
    m = _tiny(N, BL, V, E)
    idx = torch.randint(0, V, (n, N * BL))
    prefix = [2, 0]
    c1 = CHR.context_hidden_at_step(m, idx, prefix, [1, 3], BL, device="cpu")
    c2 = CHR.context_hidden_at_step(m, idx, prefix, [3, 1], BL, device="cpu")
    cos = (c1 * c2).sum(1) / (np.linalg.norm(c1, axis=1) * np.linalg.norm(c2, axis=1) + 1e-12)
    assert float(cos.min()) > 0.99999


def test_causal_invariance_check_passes_on_tiny():
    torch.manual_seed(0); N, BL, V, E = 4, 2, 16, 8
    m = _tiny(N, BL, V, E)
    idx = torch.randint(0, V, (5, N * BL))
    rep = CHR.causal_invariance_check(m, idx, N, BL, device="cpu",
                                      t_list=(0, 1, 2, 3), n_patterns=3, seed=0)
    assert rep["passed"] is True
    assert rep["min_cosine"] > 0.99999
