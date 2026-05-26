# block_lo_arm_order_network/tests/test_causal_hidden_rollout.py
import numpy as np, torch, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in ["block_lo_arm_order_network", "nanogpt-learned-order", "scripts"]:
    sys.path.insert(0, str(ROOT / p))
from AOGPT import AOGPT, AOGPTConfig
import causal_hidden_rollout as CHR
import graph_order as GO


def _tiny(N=4, BL=2, V=16, E=8, n_layer=2, n_head=2):
    cfg = AOGPTConfig(order_impl="block", block_order_block_len=BL, block_size=N * BL,
                      vocab_size=V, n_layer=n_layer, n_head=n_head, n_embd=E, dropout=0.0, bias=False)
    m = AOGPT(cfg); m.eval(); return m


def test_candidate_conditioned_hidden_shape():
    torch.manual_seed(0); N, BL, V, E, n = 4, 2, 16, 8, 3
    m = _tiny(N, BL, V, E)
    idx = torch.randint(0, V, (n, N * BL))
    c = CHR.candidate_conditioned_hidden(m, idx, [0, 1], next_block=2, other_blocks=[3],
                                         block_len=BL, device="cpu")
    assert c.shape == (n, E) and np.all(np.isfinite(c))


def test_ct_invariant_to_rest_given_next():
    # fix (S_t, sigma(t+1)=v); vary the rest -> c_t^(v) invariant (the Path-X precondition)
    torch.manual_seed(0); N, BL, V, E, n = 8, 2, 32, 16, 4
    m = _tiny(N, BL, V, E)
    idx = torch.randint(0, V, (n, N * BL))
    c1 = CHR.candidate_conditioned_hidden(m, idx, [2, 0], 5, [1, 3, 4, 6, 7], BL, "cpu")
    c2 = CHR.candidate_conditioned_hidden(m, idx, [2, 0], 5, [7, 6, 4, 3, 1], BL, "cpu")
    cos = (c1 * c2).sum(1) / (np.linalg.norm(c1, axis=1) * np.linalg.norm(c2, axis=1) + 1e-12)
    assert float(cos.min()) > 0.99999


def test_ct_depends_on_next_block():
    # varying sigma(t+1)=v DOES change c_t (intrinsic candidate-conditioning; documents the finding)
    torch.manual_seed(0); N, BL, V, E, n = 8, 2, 32, 16, 4
    m = _tiny(N, BL, V, E)
    idx = torch.randint(0, V, (n, N * BL))
    ca = CHR.candidate_conditioned_hidden(m, idx, [2, 0], 5, [1, 3, 4, 6, 7], BL, "cpu")
    cb = CHR.candidate_conditioned_hidden(m, idx, [2, 0], 1, [5, 3, 4, 6, 7], BL, "cpu")
    assert np.abs(ca - cb).max() > 1e-4


def test_causal_invariance_check_passes_on_tiny():
    torch.manual_seed(0); N, BL, V, E = 8, 2, 32, 16
    m = _tiny(N, BL, V, E)
    idx = torch.randint(0, V, (5, N * BL))
    rep = CHR.causal_invariance_check(m, idx, N, BL, device="cpu",
                                      t_list=(0, 1, 2, 3, 4), n_patterns=3, seed=0)
    assert rep["passed"] is True
    assert rep["n_compare"] > 0 and rep["min_cosine"] > 0.99999


def test_pooled_context_hidden_shape_and_completion_invariant():
    torch.manual_seed(0); N, BL, V, E, n = 8, 2, 32, 16, 4
    m = _tiny(N, BL, V, E)
    idx = torch.randint(0, V, (n, N * BL))
    p1 = CHR.pooled_context_hidden(m, idx, [2, 0, 5], [1, 3, 4, 6, 7], BL, "cpu", pool="mean")
    p2 = CHR.pooled_context_hidden(m, idx, [2, 0, 5], [7, 6, 4, 3, 1], BL, "cpu", pool="mean")
    assert p1.shape == (n, E)
    cos = (p1 * p2).sum(1) / (np.linalg.norm(p1, axis=1) * np.linalg.norm(p2, axis=1) + 1e-12)
    assert float(cos.min()) > 0.99999          # Path-Y context depends only on S_t
    pe = CHR.pooled_context_hidden(m, idx, [], [0, 1, 2, 3, 4, 5, 6, 7], BL, "cpu")
    assert pe.shape == (n, E) and np.allclose(pe, 0.0)


def test_candidate_embeddings_content_token_shape_and_no_pos():
    torch.manual_seed(0); N, BL, V, E, n = 4, 2, 16, 8, 3
    m = _tiny(N, BL, V, E)
    idx = torch.randint(0, V, (n, N * BL))
    E_cand = CHR.candidate_embeddings(m, idx, list(range(N)), BL, mode="content_token", device="cpu")
    assert E_cand.shape == (n, N, E)
    blk0_tokens = idx[:, 0:BL]                               # model-frame block 0 tokens
    expect0 = m.transformer.wte(blk0_tokens).mean(dim=1).detach().numpy()
    assert np.allclose(E_cand[:, 0, :], expect0, atol=1e-5)


def test_candidate_embeddings_content_free_is_sample_invariant():
    torch.manual_seed(0); N, BL, V, E, n = 4, 2, 16, 8, 3
    m = _tiny(N, BL, V, E)
    idx = torch.randint(0, V, (n, N * BL))
    E_free = CHR.candidate_embeddings(m, idx, list(range(N)), BL, mode="content_free", device="cpu")
    assert E_free.shape == (n, N, E)
    assert np.allclose(E_free[0], E_free[1]) and np.allclose(E_free[0], E_free[2])


def test_dynamic_score_cos_direction():
    # Path Y: one shared context p, broadcast over candidates
    p = np.array([1.0, 0.0, 0.0])
    E_cand = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    s = CHR.dynamic_score_cos(p, E_cand)
    assert s.shape == (3,)
    assert s[0] > s[1] > s[2]
    assert np.isclose(s[0], 1.0) and np.isclose(s[2], -1.0)


def test_paired_cos_per_candidate():
    # Path X: each candidate has its own context row; cosine is row-wise paired
    C = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    E = np.array([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    s = CHR.paired_cos(C, E)
    assert s.shape == (3,)
    assert np.isclose(s[0], 1.0)      # aligned
    assert np.isclose(s[1], 0.0)      # orthogonal
    assert np.isclose(s[2], -1.0)     # anti-aligned


def _ring(N, w=5.0):
    B = np.zeros((N, N))
    for i in range(N):
        B[i, (i + 1) % N] = w
    return B + B.T


def test_gamma0_equals_static_cdl_order_pathX():
    N = 6; B_A = _ring(N, 5.0); B_pos = _ring(N, 1.0)
    ctx_fn = lambda S, cand: np.zeros((len(cand), 4))      # X: (M,E)
    cand_emb_fn = lambda cand: np.zeros((len(cand), 4))
    o = CHR.causal_score_mix_rollout(B_A, B_pos, 0.0, "X", ctx_fn, cand_emb_fn)
    assert np.array_equal(o, GO.cdl_order(B_A, greedy=True))


def test_gamma0_equals_static_cdl_order_pathY():
    N = 6; B_A = _ring(N, 5.0); B_pos = _ring(N, 1.0)
    ctx_fn = lambda S, cand: np.zeros(4)                   # Y: (E,)
    cand_emb_fn = lambda cand: np.zeros((len(cand), 4))
    o = CHR.causal_score_mix_rollout(B_A, B_pos, 0.0, "Y", ctx_fn, cand_emb_fn)
    assert np.array_equal(o, GO.cdl_order(B_A, greedy=True))


def test_pathX_gamma_changes_order_when_hidden_disagrees():
    N = 6; B_A = _ring(N, 5.0); B_pos = np.zeros((N, N))
    def ctx_fn(S, cand):                                    # c_t^(v): align the largest-id candidate
        C = np.zeros((len(cand), 2))
        for j, v in enumerate(cand):
            C[j] = [1.0, 0.0] if v == max(cand) else [-1.0, 0.0]
        return C
    cand_emb_fn = lambda cand: np.tile([1.0, 0.0], (len(cand), 1)).astype(float)  # all e_v=[1,0]
    o0 = CHR.causal_score_mix_rollout(B_A, B_pos, 0.0, "X", ctx_fn, cand_emb_fn)
    o2 = CHR.causal_score_mix_rollout(B_A, B_pos, 2.0, "X", ctx_fn, cand_emb_fn)
    assert not np.array_equal(o0, o2)
