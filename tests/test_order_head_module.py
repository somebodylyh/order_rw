# tests/test_order_head_module.py
import sys, pathlib
sys.path.insert(0, "block_lo_arm_order_network")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from batch_readout.integration_hook import FrozenBetaHook
from analyses.order_head_module import OrderHeadModule
from analyses.p7_gbeta_policy import GBETA_CKPT

def test_batchmean_argsort_bitmatches_frozenbetahook():
    torch.manual_seed(0)
    N = 64                                         # gβ requires 64x64 B (verified)
    A = torch.rand(8, N, N)                        # synthetic per-sample attention
    hook = FrozenBetaHook(GBETA_CKPT, mode="argsort", device="cpu")
    ext_sigma = hook.step(A)                       # (64,) model-frame block order
    oh = OrderHeadModule(GBETA_CKPT, device="cpu")
    int_sigma = oh.argsort_order(A, per_sample=False)[0]   # (64,)
    assert int_sigma.shape == ext_sigma.shape
    assert torch.equal(int_sigma, ext_sigma), "internal batch-mean argsort must bit-match the hook"

def test_per_sample_scores_shape():
    torch.manual_seed(0)
    A = torch.rand(6, 64, 64)
    oh = OrderHeadModule(GBETA_CKPT, device="cpu")
    z_bm = oh.scores(A, per_sample=False)
    z_ps = oh.scores(A, per_sample=True)
    assert z_bm.shape == (1, 64)
    assert z_ps.shape == (6, 64)
