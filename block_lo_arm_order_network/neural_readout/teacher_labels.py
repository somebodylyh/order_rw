"""NR-1 Task 3: CDL-source-start teacher -> (sigma, rank, pairwise Y).

Convention (load-bearing; asserted in tests):
  - rank[v] = position of v in the reveal order
  - rank[sigma[0]] == 0  (earliest revealed = rank 0)
  - pairwise Y[i, j] = 1 iff rank[i] < rank[j]   (i revealed before j)
  - score convention (used in loss.py): earliest should get the HIGHEST score.

Teacher name: "CDL-source-start" -- the first node is anchored by readiness
(out(v) - alpha_dep * in(v)), then sequential C-D+L greedy rollout fills the
rest. Pure C-D+L without the source anchor can roll a reversed chain on text
substrates, so the anchor is part of the label spec (NR-1 §2.3).

NR-1 spec lock-in:
  - Only attention-derived information is used to build the label.
  - No NLL, no L2R / raster oracle, no per-token loss appears anywhere here.
"""
import sys
import pathlib

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "block_lo_arm_order_network"))

from attn_order_teacher import rollout_order
from attn_order_mlp_policy import readiness_vector


def generate_teacher_label(B, alpha_dep=0.5):
    """Return (sigma, rank, pairwise_Y) for one graph B.

    Args:
        B: (N, N) array_like, directed attention graph. Must have zero diagonal.
        alpha_dep: source-start readiness weight, source(v) = out(v) - alpha_dep * in(v).
                   Default 0.5 matches the value used by alternating runtime
                   (configs/text/wikitext103_seq256_block64_v3_readiness.py:71
                   and attn_order_mlp_policy.py:149).

    Returns:
        sigma: (N,) int64, reveal order. sigma[0] is the source-start node.
        rank:  (N,) int64, rank[v] = position of v in sigma; rank 0 = earliest.
        pairwise_Y: (N, N) uint8, Y[i, j] = 1 iff rank[i] < rank[j], diag = 0.
    """
    B = np.asarray(B, dtype=np.float64)
    if B.ndim != 2 or B.shape[0] != B.shape[1]:
        raise ValueError(f"B must be square 2D, got {B.shape}")
    if not np.all(np.diag(B) == 0.0):
        raise ValueError("B must have zero diagonal")

    N = B.shape[0]

    # Source-start anchor: pick the node with max out-degree (minus alpha*in)
    r = readiness_vector(B, alpha_dep=alpha_dep)
    start = int(np.argmax(r))

    sigma = rollout_order(B, mode="C-D+L", greedy=True, start=start)
    sigma = np.asarray(sigma, dtype=np.int64)
    assert sigma.shape == (N,), f"unexpected sigma shape {sigma.shape}"

    # rank = inverse permutation; rank[sigma[t]] = t so rank 0 = earliest
    rank = np.empty(N, dtype=np.int64)
    rank[sigma] = np.arange(N, dtype=np.int64)

    # Pairwise precedence: Y[i, j] = 1 iff i revealed before j
    Y = (rank[:, None] < rank[None, :]).astype(np.uint8)
    np.fill_diagonal(Y, 0)

    return sigma, rank, Y
