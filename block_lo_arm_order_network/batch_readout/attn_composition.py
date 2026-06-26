"""Weight-based CANDIDATE composition scores (composition compatibility).

NOT causal handoff evidence: per-head RMSNorm(q,k) and AdaLN modulation make
these approximate under this architecture. A high score only nominates an
upstream->downstream edge for later path-patching verification.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def head_weights(c_attn_w, c_proj_w, n_head, head, n_embd) -> dict:
    c_attn_w = np.asarray(c_attn_w, dtype=np.float64)  # (3*n_embd, n_embd)
    c_proj_w = np.asarray(c_proj_w, dtype=np.float64)  # (n_embd, n_embd)
    hd = n_embd // n_head
    r = slice(head * hd, (head + 1) * hd)
    q_block = c_attn_w[0 * n_embd:1 * n_embd][r, :]  # (hd, n_embd)
    k_block = c_attn_w[1 * n_embd:2 * n_embd][r, :]
    v_block = c_attn_w[2 * n_embd:3 * n_embd][r, :]
    W_Q = q_block.T  # (n_embd, hd)
    W_K = k_block.T
    W_V = v_block.T
    W_O = c_proj_w[:, r].T  # (hd, n_embd)
    return {"W_Q": W_Q, "W_K": W_K, "W_V": W_V, "W_O": W_O}


def _normed_frob(prod, a, b) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + _EPS
    return float(np.linalg.norm(prod) / denom)


def composition_scores(up: dict, down: dict) -> dict:
    W_OV_up = up["W_V"] @ up["W_O"]          # (n_embd, n_embd)
    W_QK_down = down["W_Q"] @ down["W_K"].T   # (n_embd, n_embd)
    W_OV_down = down["W_V"] @ down["W_O"]      # (n_embd, n_embd)
    return {
        "Q": _normed_frob(W_QK_down @ W_OV_up, W_QK_down, W_OV_up),
        "K": _normed_frob(W_QK_down.T @ W_OV_up, W_QK_down, W_OV_up),
        "V": _normed_frob(W_OV_down @ W_OV_up, W_OV_down, W_OV_up),
    }


def layer_pair_composition(c_attn_ws, c_proj_ws, n_head, n_embd, pairs) -> dict:
    out = {}
    for (i, j) in pairs:
        mat = np.zeros((n_head, n_head, 3), dtype=np.float64)
        ups = [head_weights(c_attn_ws[i], c_proj_ws[i], n_head, a, n_embd) for a in range(n_head)]
        downs = [head_weights(c_attn_ws[j], c_proj_ws[j], n_head, b, n_embd) for b in range(n_head)]
        for a in range(n_head):
            for b in range(n_head):
                s = composition_scores(ups[a], downs[b])
                mat[a, b] = [s["Q"], s["K"], s["V"]]
        out[(i, j)] = mat
    return out


def extract_layer_weights(model):
    """Read per-layer c_attn/c_proj weights from an AOGPT model.

    Args:
        model: AOGPT model instance

    Returns:
        tuple of (c_attn_ws, c_proj_ws, n_head, n_embd) where:
        - c_attn_ws: list of numpy arrays, one per layer
        - c_proj_ws: list of numpy arrays, one per layer
        - n_head: number of attention heads
        - n_embd: embedding dimension
    """
    blocks = model.transformer.h
    c_attn_ws, c_proj_ws = [], []
    for blk in blocks:
        c_attn_ws.append(blk.attn.c_attn.weight.detach().cpu().numpy())
        c_proj_ws.append(blk.attn.c_proj.weight.detach().cpu().numpy())
    n_head = blocks[0].attn.n_head
    n_embd = blocks[0].attn.n_embd
    return c_attn_ws, c_proj_ws, int(n_head), int(n_embd)
