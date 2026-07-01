# tests/test_v3_feasibility_smoke.py
import sys, pathlib, math
sys.path.insert(0, "block_lo_arm_order_network")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import pytest
from analyses.order_head_module import OrderHeadModule, AOGPTWithOrderHead
from analyses.p5_utility_controller import load_p5_ckpt
from analyses.p7_gbeta_policy import GBETA_CKPT
from batch_readout.hook_order_provider import random_probe_token_orders

CKPT = "block_lo_arm_order_network/probe_results/gbeta_K1000_from10k_L1H7_seed123/ckpt_step20000.pt"
pytestmark = pytest.mark.skipif(not pathlib.Path(CKPT).exists(), reason="backbone ckpt absent")

def test_grad_routing_pg_does_not_touch_backbone():
    model, chunks, clean_perm, dev = load_p5_ckpt(CKPT, M=8, device="cpu")
    oh = OrderHeadModule(GBETA_CKPT, device="cpu")
    for p in oh.gbeta.parameters(): p.requires_grad_(True)
    wrap = AOGPTWithOrderHead(model, oh, clean_perm, device="cpu")
    idx = torch.stack([chunks[i] for i in range(8)])
    probe = random_probe_token_orders(8, 0, 0, dev)
    z = wrap.compute_order_logits(idx, probe, per_sample=True)   # B detached inside
    # PG-only surrogate: depends on OrderHead params, not backbone
    pg = z.sum()
    model.zero_grad(); [p.grad and p.grad.zero_() for p in oh.gbeta.parameters()]
    pg.backward()
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in model.parameters()), \
        "PG term must not produce backbone gradient (B is detached)"
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in oh.gbeta.parameters()), \
        "PG term must update OrderHead params"

def test_smoke_runs_and_params_move():
    from analyses.v3_feasibility_smoke import run_smoke
    res = run_smoke(CKPT, arm="per_sample", n_steps=3, batch_size=4, held=6,
                    device="cpu", out_dir="runs/v3_feasibility_test")
    assert res["nan"] is False
    assert res["orderhead_param_delta"] > 0.0
    assert math.isfinite(res["entropy_last"])
    assert 0.0 < res["entropy_last"]                       # entropy did not collapse to 0
    assert res["denoise_before"] is not None and res["denoise_after"] is not None

def test_per_sample_orders_are_diverse_BN():
    model, chunks, clean_perm, dev = load_p5_ckpt(CKPT, M=8, device="cpu")
    oh = OrderHeadModule(GBETA_CKPT, device="cpu")
    wrap = AOGPTWithOrderHead(model, oh, clean_perm, device="cpu")
    idx = torch.stack([chunks[i] for i in range(8)])
    probe = random_probe_token_orders(8, 0, 0, dev)
    A = wrap.extract_B(idx, probe)
    orders = oh.argsort_order(A, per_sample=True).cpu().numpy()  # (8,N)
    assert orders.shape[0] == 8
    assert not np.all(orders == orders[0]), "per-sample orders must not be identical"
