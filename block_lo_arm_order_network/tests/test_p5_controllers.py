import pathlib, sys
import torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.p5_utility_controller import BOnlyController, ScaffoldedController


def test_scaffolded_freezes_gB_and_small_init_residual():
    gB = BOnlyController(b_dim=130)
    for p in gB.parameters():
        p.requires_grad_(False)
    sc = ScaffoldedController(gB, h_dim=8, alpha_init=0.01)
    B = torch.randn(64, 130); H = torch.randn(64, 8)
    z = sc(B, H)
    assert z.shape == (64,)
    # gB is frozen inside the scaffolded controller
    assert all(not p.requires_grad for p in sc.g_B.parameters())
    assert any(p.requires_grad for p in sc.g_H.parameters())
    assert sc.residual_ratio(B, H) < 0.5               # residual is small at init
