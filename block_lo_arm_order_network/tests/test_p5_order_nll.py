import pathlib, sys
import numpy as np, torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.p5_utility_controller import order_nll, utility_pool, load_p5_ckpt

CKPT = str(ROOT / "runs/handoff_overnight/seed123/ckpt_step10000.pt")


def test_order_nll_matches_forward_fn_and_pool_orders():
    model, chunks, clean_perm, dev = load_p5_ckpt(CKPT, M=2, device="cpu")
    idx = chunks[0:1]
    l2r = np.arange(64)
    rev = np.arange(64)[::-1].copy()
    nll_l2r = order_nll(model, idx, l2r, clean_perm, dev)
    pool = utility_pool(model, idx, [l2r, rev], clean_perm, dev)
    assert isinstance(nll_l2r, float) and nll_l2r > 0
    assert abs(pool[0] - nll_l2r) < 1e-5      # pool reuses order_nll
    assert pool[0] != pool[1]                 # different orders -> different NLL
