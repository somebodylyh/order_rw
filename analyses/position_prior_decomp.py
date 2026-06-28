"""Pillar-5-core: position-prior decomposition + content-binding. No new training.

See docs/superpowers/specs/2026-06-28-position-prior-decomposition-design.md
and docs/superpowers/plans/2026-06-28-position-prior-decomposition.md.

Key empirical note (decided during execution): a uniform-causal attention pushed
through the real ``build_model_frame_strict65`` pipeline rolls out to ascending
order with tau=+1.0 — so the synthetic floor is generated via that pipeline, not by
hand-stamping a 65x65 B (hand-stamping mis-specifies the C-D+L sign convention and
yields tau=-1). The strict65 builder enforces the zeroed-diagonal convention.
"""
import contextlib
import pathlib
import sys

import numpy as np
import torch

_BLOCK_DIR = pathlib.Path(__file__).resolve().parents[1] / "block_lo_arm_order_network"
if str(_BLOCK_DIR) not in sys.path:
    sys.path.insert(0, str(_BLOCK_DIR))

from none_separated_block_graph import rollout_by_method  # noqa: E402


# ── Task 1: frame-aware tau helpers ──────────────────────────────────────────

def rollout_order(B65, method="C-D+L"):
    """Rolled-out content order (length 64, None excluded), model frame."""
    return np.asarray(rollout_by_method(np.asarray(B65, dtype=np.float64), method))


def _kendall_tau(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    n = len(a)
    c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
            if s > 0:
                c += 1
            elif s < 0:
                d += 1
    return (c - d) / (0.5 * n * (n - 1))


def tau_vs_arange(order):
    order = np.asarray(order)
    return float(_kendall_tau(order, np.arange(len(order))))


def tau_two_frames(B65, inv_perm_content, method="C-D+L"):
    """tau of the rolled-out order in the model-slot frame and (posthoc) the
    physical-block frame (apply inv_perm_content to the order)."""
    order = rollout_order(B65, method)
    inv = np.asarray(inv_perm_content)
    order_phys = inv[order]
    return {"tau_model_slot": tau_vs_arange(order),
            "tau_physical": tau_vs_arange(order_phys)}


def model_mask_allows_self(model):
    """This AOGPT uses tril (includes diagonal) -> self attention allowed."""
    blk = model.transformer.h[0].attn
    return bool(getattr(blk, "bias", None) is not None and blk.bias[0, 0, 0, 0] == 1)


# ── Task 2: synthetic floor baselines ────────────────────────────────────────

from batch_readout.l0_strict65 import build_model_frame_strict65  # noqa: E402


def synthetic_uniform_causal_B():
    """Uniform-causal attention (att[i,j]=1/(i+1) for j<=i) pushed through the real
    strict65 builder (which zeroes the B diagonal). This is the pure mask+readout
    floor; it rolls out ascending (tau~+1.0)."""
    T = 257
    att = np.zeros((1, 1, T, T), dtype=np.float32)
    for i in range(T):
        att[0, 0, i, : i + 1] = 1.0 / (i + 1)
    po = np.tile(np.arange(256, dtype=np.int64), (1, 1))
    return build_model_frame_strict65(att, po)[0, 0]  # (65,65)


def random_B(rng=None):
    """Structureless readout null: random 65x65 B, zeroed diagonal, no None in-edges."""
    rng = rng or np.random.default_rng(0)
    B = rng.random((65, 65))
    np.fill_diagonal(B, 0.0)
    B[:, 0] = 0.0
    return B


def floor_taus(rng=None):
    uc = tau_vs_arange(rollout_order(synthetic_uniform_causal_B()))
    rb = float(np.mean([tau_vs_arange(rollout_order(random_B(np.random.default_rng(i))))
                        for i in range(8)]))
    return {"uniform_causal": float(uc), "random_B": rb}


# ── Task 3: reversible PE-ablation context manager ───────────────────────────

@contextlib.contextmanager
def pe_ablation(model, which):
    """Temporarily zero wpe / wtpe / both embedding outputs via forward hooks.
    which='none' registers nothing (bit-identical). Hooks removed on exit."""
    def _zero_hook(_m, _inp, out):
        return torch.zeros_like(out)

    targets = []
    if which in ("wpe", "both"):
        targets.append(model.transformer.wpe)
    if which in ("wtpe", "both"):
        targets.append(model.transformer.wtpe)
    handles = [m.register_forward_hook(_zero_hook) for m in targets]
    try:
        yield model
    finally:
        for h in handles:
            h.remove()
