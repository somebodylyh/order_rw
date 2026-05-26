"""Position/locality baseline graphs B_pos for the hidden-graph diagnostic."""
import numpy as np


def text_position_graph(N=64, tau=1.0):
    """B_pos(u,v) = exp(-|u-v|/tau), diag 0. Text = block index distance."""
    idx = np.arange(N)
    d = np.abs(idx[:, None] - idx[None, :]).astype(np.float64)
    B = np.exp(-d / float(tau))
    np.fill_diagonal(B, 0.0)
    return B


def image_manhattan_graph(side=8, tau=1.0):
    """B_pos(u,v) = exp(-manhattan(u,v)/tau), diag 0, over a side x side raster grid."""
    n = side * side
    rows = np.arange(n) // side
    cols = np.arange(n) % side
    d = (np.abs(rows[:, None] - rows[None, :]) +
         np.abs(cols[:, None] - cols[None, :])).astype(np.float64)
    B = np.exp(-d / float(tau))
    np.fill_diagonal(B, 0.0)
    return B
