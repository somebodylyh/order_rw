"""Pillar-3 causal handoff path-patching.

Mean-ablation + path-restricted QK-patch interventions with tau-only readouts,
on the converged (step-10000) checkpoints of seed2/seed42/seed123.

See spec: docs/superpowers/specs/2026-06-27-handoff-causal-path-patching-design.md
and plan: docs/superpowers/plans/2026-06-27-handoff-causal-path-patching.md

Why tau-only (not NLL): any-order training makes the objective order-insensitive,
so the order signal lives in the attention map (model-frame B -> tau_vs_l2r), not
the loss. tau is built from att = softmax(q.k^T), which depends only on q,k.
"""
import pathlib
import sys

import numpy as np
import torch

# Make block_lo_arm_order_network modules importable regardless of CWD.
_BLOCK_DIR = pathlib.Path(__file__).resolve().parents[1] / "block_lo_arm_order_network"
if str(_BLOCK_DIR) not in sys.path:
    sys.path.insert(0, str(_BLOCK_DIR))

from attention_trajectory import extract_all_layer_B  # noqa: E402
from batch_readout.order_tau_readout import layer_head_tau_table  # noqa: E402
from neural_readout.extract_b import _load_model_and_chunks  # noqa: E402


# ── Task 2: probe loader + tau-readout wrapper ────────────────────────────────

def load_model_and_chunks_seed(ckpt_path, total, device, split="train"):
    """Load the AOGPT model (eval mode), eval-token chunks, and the ckpt seed.

    device may be a torch.device or a string; the underlying loader wants a string.
    Seed is read from ckpt['args']['seed'] (integrity: never trust dir names).
    """
    dev_str = str(device) if isinstance(device, torch.device) else device
    model, chunks, _clean_perm, _dev, _ci = _load_model_and_chunks(
        ckpt_path, total, seed=0, device=dev_str, split=split
    )
    model.eval()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    seed = int(ckpt["args"]["seed"])
    return model, chunks, seed


def make_probe_batch(eval_model_tokens, n, rng):
    """Draw n random rows + identity probe orders (model-frame == L2R reference)."""
    idx = rng.choice(len(eval_model_tokens), size=n, replace=False)
    chunks = eval_model_tokens[idx]
    probe_orders = np.tile(np.arange(256, dtype=np.int64), (n, 1))
    return chunks, probe_orders


def tau_table_from_attn(attn_list, probe_orders):
    """attn_list (per-layer (S,H,257,257)) -> (L,H) signed tau for method C-D+L."""
    B_all = extract_all_layer_B(attn_list, probe_orders)  # (L,S,H,65,65)
    B_lhn = B_all.mean(axis=1)                            # (L,H,65,65) batch-mean
    tbl = layer_head_tau_table(B_lhn, methods=("C-D+L", "L"))
    return tbl["tau"][:, :, 0]                            # (L,H) signed, C-D+L


@torch.no_grad()
def run_clean(model, probe_chunks, probe_orders, device):
    """Clean forward -> (attn_list, tau[L,H])."""
    pc = probe_chunks.to(device)
    po = torch.from_numpy(probe_orders).to(device)
    _, _, attn_list = model.forward_fn(pc, po, return_attentions=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return attn_list, tau_table_from_attn(attn_list, probe_orders)
