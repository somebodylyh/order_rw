import sys
from pathlib import Path
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_HERE))

from directed_graph_policy import build_directed_graph
from train_attn_order_mlp import OrderMLP
from attn_order_distill import distill_order_mlp
import attn_order_image_diag as D


def _local_A(N=64, grid=8, seed=0):
    """A 64x64 with locality: each node attends mostly to its 4-neighbours on an 8x8 grid."""
    rng = np.random.default_rng(seed)
    A = rng.uniform(0, 0.01, size=(N, N))
    for i in range(N):
        r, c = divmod(i, grid)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid and 0 <= nc < grid:
                A[i, nr * grid + nc] += 1.0
    np.fill_diagonal(A, 0.0)
    return A.astype(np.float32)


def test_image_refresh_diagnostics_keys_and_locality():
    B = build_directed_graph(_local_A())
    mlp, _ = distill_order_mlp(B, mlp=None, n_orders=20, epochs=3, seed=0, device="cpu")
    rec = D.image_refresh_diagnostics(B, mlp, tau=0.5, top_k=4, seed=0, K=32, device="cpu")
    for k in ("p_le1", "p_le2", "top4_follow", "B_edge_ratio",
              "rollout_entropy", "rollout_unique", "teacher_p_le1", "teacher_top4_follow"):
        assert k in rec, f"missing key {k}"
    # on a locality graph the student rollout must beat the random floor on both metrics
    assert rec["p_le1"] > 0.10, rec["p_le1"]
    assert rec["B_edge_ratio"] > 1.3, rec["B_edge_ratio"]
    assert 0 < rec["rollout_unique"] <= 32


# ---- Task 2: extraction wrapper tests ----

import importlib.util
import pytest as _pytest_module  # noqa: E402


def _load_trainer():
    spec = importlib.util.spec_from_file_location(
        "train_vq64_alternating", str(_REPO / "scripts" / "train_vq64_alternating.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CKPT = _REPO / "probe_results_image/e2_vq_round2_20k/cont_random/ckpt_step20000.pt"
VAL  = _REPO / "block_lo_arm_order_network/data/Imagenet32VQ_f4_800k_seq64/val.bin"


@_pytest_module.mark.skipif(not (CKPT.exists() and VAL.exists()), reason="needs image ckpt+data")
def test_extract_B_from_model_shape_and_determinism():
    T = _load_trainer()
    import torch
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, model_args = T.build_or_load_model_for_extraction(str(CKPT), dev)
    data = np.memmap(str(VAL), dtype=np.uint16, mode="r")
    B1 = T.extract_B_from_model(model, data, tokens_per_image=64, n_images=20, m_passes=2,
                                device=dev, seed=123)
    B2 = T.extract_B_from_model(model, data, tokens_per_image=64, n_images=20, m_passes=2,
                                device=dev, seed=123)
    assert B1.shape == (64, 64)
    assert np.allclose(np.diag(B1), 0.0)
    assert np.isfinite(B1).all()
    assert np.allclose(B1, B2), "same seed must give identical B"


# ---- Task 3: alpha schedule ----

def test_alpha_schedule_delayed_warmup():
    T = _load_trainer()
    # warmup_start=3000, ramp=10000, alpha_max=0.9  -> 0 until 3k, 0.9 at 13k, flat after
    f = lambda s: T.get_alpha_alt(s, warmup_start=3000, ramp=10000, alpha_max=0.9)
    assert f(0) == 0.0
    assert f(2999) == 0.0
    assert f(3000) == 0.0
    assert abs(f(8000) - 0.45) < 1e-6   # halfway through ramp
    assert abs(f(13000) - 0.9) < 1e-6
    assert abs(f(30000) - 0.9) < 1e-6
