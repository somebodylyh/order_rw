import sys, os
import numpy as np
import torch

CHENHE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CHENHE)
from orderhead_v3.l0_dynamic_gbeta import L0DynamicGBeta
from orderhead_v3.cdl_teacher import build_dynamic_teacher, consensus_order_from_pairwise


def test_gbeta_forward_shape():
    B = torch.rand(3, 8, 65, 65)
    gb = L0DynamicGBeta(heads=8, nodes=65).eval()
    with torch.no_grad():
        scores, _aux = gb(B, apply_head_dropout=False)
    assert scores.shape == (3, 64)
    assert torch.isfinite(scores).all()
    sigma = scores.argsort(dim=1, descending=True)
    assert all(sorted(sigma[i].tolist()) == list(range(64)) for i in range(3))


def test_teacher_is_model_frame():
    # build_dynamic_teacher operates PER SAMPLE on (H, 65, 65) and emits
    # orders/ranks/pairwise over model-frame content indices [0,63] — no
    # inverse_block_perm applied anywhere.
    rng = np.random.default_rng(0)
    B_heads = rng.random((8, 65, 65)).astype(np.float64)
    for h in range(8):
        np.fill_diagonal(B_heads[h], 0.0)
        B_heads[h][:, 0] = 0.0                     # no edges into None
    teacher = build_dynamic_teacher(B_heads)
    # per-head orders are permutations of the 64 model-frame blocks
    assert teacher["orders"].shape == (8, 64)
    for h in range(8):
        assert sorted(teacher["orders"][h].tolist()) == list(range(64))
    # soft pairwise is (64,64), antisymmetric around 0.5, diagonal 0.5
    Y = teacher["pairwise"]
    assert Y.shape == (64, 64)
    off = ~np.eye(64, dtype=bool)
    assert np.allclose((Y + Y.T)[off], 1.0)
    assert np.allclose(np.diag(Y), 0.5)
    # derived consensus order is a model-frame [0,63] permutation
    order = consensus_order_from_pairwise(Y)
    assert sorted(order.tolist()) == list(range(64))
