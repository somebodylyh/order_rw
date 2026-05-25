#!/usr/bin/env python3
"""Print the MLP-selected block order (greedy + a few samples) as an 8x8 grid."""
import sys
from pathlib import Path
import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "block_lo_arm_order_network"))
sys.path.insert(0, str(_REPO / "nanogpt-learned-order"))
sys.path.insert(0, str(_REPO / "scripts"))

from train_attn_order_mlp import OrderMLP
import attn_order_mlp_policy as P

GRID = 8

run = sys.argv[1] if len(sys.argv) > 1 else "probe_results_image/vq64_alt_from0_mlp_patch2x2"
step = sys.argv[2] if len(sys.argv) > 2 else "30000"
A = np.load(_REPO / run / f"A_global_step{step}.npy")
B = A.T.copy(); np.fill_diagonal(B, 0.0)

mlp = OrderMLP(); mlp.load_state_dict(torch.load(_REPO / run / f"beta_step{step}.pt", map_location="cpu")); mlp.eval()

def grid_of(order):
    g = [["  " for _ in range(GRID)] for _ in range(GRID)]
    for t, idx in enumerate(order):
        r, c = int(idx) // GRID, int(idx) % GRID
        g[r][c] = f"{t:2d}"
    return "\n".join(" ".join(row) for row in g)

def manh(order):
    coords = [(int(i)//GRID, int(i)%GRID) for i in order]
    steps = [abs(coords[k][0]-coords[k-1][0])+abs(coords[k][1]-coords[k-1][1]) for k in range(1, len(coords))]
    return float(np.mean(steps)), float(np.mean([s == 1 for s in steps]))

# greedy (canonical) order
g_order = P.sample_orders_batched_mlp(B, 1, mlp, "original", base_seed=0,
        device=torch.device("cpu"), tau=0.5, top_k=4, greedy=True)[0].numpy()
m, p1 = manh(g_order)
print(f"#### GREEDY order  (mean_manh={m:.3f}  P(step==1 adjacent)={p1:.3f}) ####")
print("first idx:", int(g_order[0]), " seq:", " ".join(str(int(i)) for i in g_order))
print(grid_of(g_order))

# a few stochastic samples (training tau/top_k)
print("\n#### 3 STOCHASTIC samples (tau=0.5 top_k=4) ####")
ss = P.sample_orders_batched_mlp(B, 3, mlp, "original", base_seed=42,
        device=torch.device("cpu"), tau=0.5, top_k=4)
for j in range(3):
    o = ss[j].numpy(); m, p1 = manh(o)
    print(f"-- sample {j}: start={int(o[0])} mean_manh={m:.3f} P(adj)={p1:.3f}")
    print(grid_of(o))
