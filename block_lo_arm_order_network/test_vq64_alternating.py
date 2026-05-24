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
