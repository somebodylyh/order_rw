"""Fixed-order validation loss/PPL (val_ori_l2r_block): teacher-forced AO-NLL under
the fixed physical L2R block order, identical eval set across all arms."""

from __future__ import annotations

import math
import sys
import pathlib
from typing import Dict, List, Union

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
BLOCK_ROOT = ROOT / "block_lo_arm_order_network"
for path in (ROOT, BLOCK_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyses.p5_utility_controller import order_nll, N  # noqa: E402

L2R = np.arange(N, dtype=np.int64)


@torch.no_grad()
def fixed_order_val_loss(model, eval_chunks, clean_perm, device):
    """Mean NLL of ``eval_chunks`` under the fixed physical L2R block order.

    Parameters
    ----------
    model : nn.Module
        AO-GPT backbone with ``forward_fn(idx, token_order) -> (logits, loss)``.
    eval_chunks : List[Tensor]
        List of token index tensors, each shape ``(T,)`` or ``(1, T)``.
    clean_perm : CleanPermutation
        Permutation metadata (must have ``block_perm_phys_to_model``).
    device : str or torch.device
        Torch device.

    Returns
    -------
    float
        Mean NLL (lower = better).  PPL = ``math.exp(loss)``.
    """
    nlls = []
    for chunk in eval_chunks:
        row = chunk.unsqueeze(0) if chunk.dim() == 1 else chunk
        nll = order_nll(model, row, L2R, clean_perm, device)
        nlls.append(nll)
    return float(np.mean(nlls))


def eval_curve(ckpt_paths, eval_chunks, clean_perm, device):
    """Evaluate fixed-order val loss at multiple checkpoints.

    Parameters
    ----------
    ckpt_paths : Dict[int, str]
        Mapping from step number to checkpoint path.
    eval_chunks : List[Tensor]
        Fixed evaluation chunks (same across all checkpoints).
    clean_perm : CleanPermutation
        Permutation metadata shared across all checkpoints.
    device : str or torch.device

    Returns
    -------
    Dict[int, Dict[str, float]]
        Mapping from step to ``{"loss": ..., "ppl": ...}``.
    """
    from analyses.p5_utility_controller import load_p5_ckpt  # local to avoid cycle

    out: Dict[int, Dict[str, float]] = {}
    for step, path in sorted(ckpt_paths.items()):
        model, _chunks, cp, dev = load_p5_ckpt(str(path), 2, device=str(device))
        # Use the caller's clean_perm, not the ckpt's — ensures frame consistency
        loss = fixed_order_val_loss(model, eval_chunks, clean_perm, device)
        out[int(step)] = {"loss": loss, "ppl": math.exp(loss)}
    return out


__all__ = ["fixed_order_val_loss", "eval_curve", "L2R"]
