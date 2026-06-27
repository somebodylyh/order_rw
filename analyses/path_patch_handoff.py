"""Pillar-3 causal handoff path-patching.

Mean-ablation + path-restricted QK-patch interventions with tau-only readouts,
on the converged (step-10000) checkpoints of seed2/seed42/seed123.

See spec: docs/superpowers/specs/2026-06-27-handoff-causal-path-patching-design.md
and plan: docs/superpowers/plans/2026-06-27-handoff-causal-path-patching.md

Why tau-only (not NLL): any-order training makes the objective order-insensitive,
so the order signal lives in the attention map (model-frame B -> tau_vs_l2r), not
the loss. tau is built from att = softmax(q.k^T), which depends only on q,k.
"""
import csv as _csv
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


# ── Task 3: mean-ablation hook (forward_pre_hook on attn.c_proj) ──────────────

def mean_ablation_prehook(head_indices, n_head):
    """forward_pre_hook(module, args) replacing target heads' columns of y
    (c_proj input, shape (B,T,C)) with their position-wise batch mean.

    Only the batch dim is averaged; the token/block (position) dim is preserved.
    Empty head set -> returns None (a true no-op, bit-identical).
    """
    heads = list(head_indices)

    def _hook(module, args):
        if not heads:
            return None
        y = args[0]
        B, T, C = y.shape
        hs = C // n_head
        y = y.clone()
        for h in heads:
            sl = slice(h * hs, (h + 1) * hs)
            y[:, :, sl] = y[:, :, sl].mean(dim=0, keepdim=True)  # position-wise batch mean
        return (y,) + tuple(args[1:])

    return _hook


@torch.no_grad()
def run_with_ablation(model, layer, head_indices, probe_chunks, probe_orders, device):
    """Forward with target heads mean-ablated at `layer` -> (attn_list, tau[L,H])."""
    n_head = model.transformer.h[layer].attn.n_head
    handle = model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(
        mean_ablation_prehook(head_indices, n_head)
    )
    try:
        pc = probe_chunks.to(device)
        po = torch.from_numpy(probe_orders).to(device)
        _, _, attn_list = model.forward_fn(pc, po, return_attentions=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    finally:
        handle.remove()
    return attn_list, tau_table_from_attn(attn_list, probe_orders)


# ── Task 4: Stage-1 intervention variants + runner ───────────────────────────

def stage1_variants(strong_set):
    """Redundancy ladder: full set, leave-one-out, and single-head ablations."""
    s = list(strong_set)
    return {
        "full": [list(s)],
        "loo": [[h for h in s if h != drop] for drop in s],
        "single": [[h] for h in s],
    }


@torch.no_grad()
def run_stage1(model, ablate_layer, target_heads, readout_layers, probe_batches, device):
    """Ablate `target_heads` at `ablate_layer`; return mean/std Delta-tau[L,H] over
    probe batches. `readout_layers` is recorded (must be downstream of ablate_layer).
    """
    per_batch = []
    for pc, po in probe_batches:
        _, tau_clean = run_clean(model, pc, po, device)
        _, tau_abl = run_with_ablation(model, ablate_layer, target_heads, pc, po, device)
        per_batch.append(tau_abl - tau_clean)
    arr = np.stack(per_batch)  # (Nbatch, L, H)
    return {
        "ablate_layer": ablate_layer,
        "target_heads": list(target_heads),
        "dtau_mean": arr.mean(axis=0),
        "dtau_std": arr.std(axis=0),
        "n_batch": len(per_batch),
        "readout_layers": list(readout_layers),
    }


# ── Task 5: multiplicity-collapse readout + Stage-1 table writer ──────────────

def multiplicity_collapse(tau_before_LH, tau_after_LH, layer, strong=0.95):
    """How many heads at `layer` are strong (|tau|>=strong) before vs after, plus
    the layer's mean signed tau before/after."""
    b = np.abs(tau_before_LH[layer])
    a = np.abs(tau_after_LH[layer])
    return {
        "n_strong_before": int((b >= strong).sum()),
        "n_strong_after": int((a >= strong).sum()),
        "mean_tau_before": float(tau_before_LH[layer].mean()),
        "mean_tau_after": float(tau_after_LH[layer].mean()),
    }


def consensus_tau(tau_after_LH, layer, carrier_heads):
    """Mean signed tau over a carrier set at `layer` (carrier-set aggregation)."""
    if not carrier_heads:
        return float("nan")
    return float(np.mean([tau_after_LH[layer, h] for h in carrier_heads]))


_STAGE1_COLS = [
    "seed", "stage", "intervention", "target_layer", "target_heads",
    "mean_tau_before", "mean_tau_after", "n_strong_before",
    "n_strong_after", "delta_global_tau",
]


def write_stage1_table(rows, out_csv):
    with open(out_csv, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=_STAGE1_COLS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
