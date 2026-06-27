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
import math
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

# Make block_lo_arm_order_network modules importable regardless of CWD.
_BLOCK_DIR = pathlib.Path(__file__).resolve().parents[1] / "block_lo_arm_order_network"
if str(_BLOCK_DIR) not in sys.path:
    sys.path.insert(0, str(_BLOCK_DIR))

from attention_trajectory import extract_all_layer_B  # noqa: E402
from batch_readout.order_tau_readout import layer_head_tau_table  # noqa: E402
from neural_readout.extract_b import _load_model_and_chunks  # noqa: E402
from model_AOGPT_AdaLN6_NoRep_cond_128_trunc_qknorm import modulate  # noqa: E402


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


# ── Task 6: path-restricted L0->L1 QK patch helpers ──────────────────────────

def capture_block_input(model, layer):
    """Pre-hook recording the residual `x` and conditioning `c` entering a block.

    Block.forward is called positionally as block(x, cond, return_attn=...), so
    args = (x, cond). Returns (handle, store) where store['x'], store['cond'] are
    filled after a forward pass; remove the handle when done.
    """
    store = {}
    blk = model.transformer.h[layer]

    def _pre(module, args, kwargs):
        store["x"] = args[0].detach().clone()
        store["cond"] = args[1].detach().clone()
        return None

    handle = blk.register_forward_pre_hook(_pre, with_kwargs=True)
    return handle, store


def _l1_qkv_from_residual(model, x_resid, cond):
    """Recompute L1's per-head q,k,v from a residual (applies ln_1 + AdaLN msa
    modulation -> c_attn -> head split -> q_norm/k_norm). Returns q,k,v,(hs)."""
    blk = model.transformer.h[1]
    attn = blk.attn
    shift, scale, *_ = blk.adaLN(cond).chunk(6, dim=-1)  # shift_msa, scale_msa, ...
    xn = modulate(blk.ln_1(x_resid), shift, scale)
    B, T, C = xn.shape
    nh = attn.n_head
    hs = C // nh
    q, k, v = attn.c_attn(xn).split(C, dim=2)
    q = q.view(B, T, nh, hs).transpose(1, 2)
    k = k.view(B, T, nh, hs).transpose(1, 2)
    v = v.view(B, T, nh, hs).transpose(1, 2)
    q, k = attn.q_norm(q), attn.k_norm(k)
    return q, k, v, hs


def _causal_softmax(scores):
    """scores (B,T,T) -> causal softmax att (B,T,T)."""
    T = scores.size(-1)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=scores.device))
    scores = scores.masked_fill(~causal.view(1, T, T), float("-inf"))
    return F.softmax(scores, dim=-1)


def l1_attn_from_residual(model, x_resid, cond, dst_head, device):
    """L1 destination head's attention map (B,T,T) recomputed from a residual."""
    q, k, _v, hs = _l1_qkv_from_residual(model, x_resid, cond)
    scores = (q[:, dst_head] @ k[:, dst_head].transpose(-2, -1)) * (1.0 / math.sqrt(hs))
    return _causal_softmax(scores)


def patched_l1_head_output(model, x_clean_L1, x_corr_L1, dst_head, cond, device):
    """Path-restricted patch: Q/K from the corrupted residual, V from the CLEAN
    residual -> att @ v for the destination head. Returns (B,T,hs)."""
    q_c, k_c, _vc, hs = _l1_qkv_from_residual(model, x_corr_L1, cond)
    _qc2, _kc2, v_clean, _hs2 = _l1_qkv_from_residual(model, x_clean_L1, cond)
    scores = (q_c[:, dst_head] @ k_c[:, dst_head].transpose(-2, -1)) * (1.0 / math.sqrt(hs))
    att = _causal_softmax(scores)
    return att @ v_clean[:, dst_head]


# ── Task 7: Stage-2 runner (readout 2a + downstream path_fraction) ────────────

def global_tau(tau_LH):
    """Global order signal = max |tau| over all (layer, head)."""
    return float(np.abs(tau_LH).max())


def redundancy_ordering(means):
    """means: {'single':x,'loo':y,'full':z}. Reports the expected single<=LOO<=full
    trend (not a hard gate)."""
    s, l, f = means["single"], means["loo"], means["full"]
    return {"single": s, "loo": l, "full": f, "monotone": bool(s <= l <= f)}


def _patched_cproj_prehook(patched_by_head, n_head):
    """c_proj pre-hook replacing each dst head's y-slice with a precomputed patched
    head output (corr-QK, clean-V). Other heads untouched -> path-restricted."""
    def _hook(module, args):
        y = args[0].clone()
        C = y.shape[-1]
        hs = C // n_head
        for h, y_patch in patched_by_head.items():
            y[:, :, h * hs:(h + 1) * hs] = y_patch
        return (y,) + tuple(args[1:])
    return _hook


@torch.no_grad()
def _run_path_restricted(model, dst_heads_L1, x_clean_L1, x_corr_L1, cond,
                         probe_chunks, probe_orders, device):
    """Clean forward with only L1 dst heads' outputs replaced by the corr-QK/clean-V
    patch; propagate to L2/L3 -> tau[L,H]."""
    n_head = model.transformer.h[1].attn.n_head
    patched_by_head = {
        h: patched_l1_head_output(model, x_clean_L1, x_corr_L1, h, cond, device)
        for h in dst_heads_L1
    }
    handle = model.transformer.h[1].attn.c_proj.register_forward_pre_hook(
        _patched_cproj_prehook(patched_by_head, n_head)
    )
    try:
        pc = probe_chunks.to(device)
        po = torch.from_numpy(probe_orders).to(device)
        _, _, attn_list = model.forward_fn(pc, po, return_attentions=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    finally:
        handle.remove()
    return tau_table_from_attn(attn_list, probe_orders)


def _layer_delta(tau_a, tau_b, layer):
    """Mean |tau_a - tau_b| over a layer's heads."""
    return float(np.abs(tau_a[layer] - tau_b[layer]).mean())


def _path_fraction(tau_clean, tau_full, tau_path, layer_or_global,
                   downstream_layers=(2, 3), eps=1e-9):
    """|Delta_path| / |Delta_full| for a downstream readout. layer_or_global is an
    int layer index or the string 'global'.

    The global readout aggregates over downstream layers (mean of per-layer
    mean-|Delta|): a max-|tau| global is pinned at ~1.0 by the robust L1 carrier and
    is insensitive to L0 ablation, so it is NOT used here. Returns nan if
    |Delta_full| < eps.
    """
    if layer_or_global == "global":
        d_path = float(np.mean([_layer_delta(tau_path, tau_clean, L) for L in downstream_layers]))
        d_full = float(np.mean([_layer_delta(tau_full, tau_clean, L) for L in downstream_layers]))
    else:
        L = layer_or_global
        d_path = _layer_delta(tau_path, tau_clean, L)
        d_full = _layer_delta(tau_full, tau_clean, L)
    return float(d_path / d_full) if d_full >= eps else float("nan")


@torch.no_grad()
def run_stage2(model, seed, src_heads_L0, dst_heads_L1, probe_batches, device,
               downstream_layers=(2, 3), null_src_heads=None):
    """Stage-2: per probe batch run clean, full-L0-ablation, and path-restricted
    L0->L1-QK forwards; report readout 2a (L1-dst Delta-tau under full L0 ablation)
    and readout 2b (downstream/global path_fraction), plus an optional null-path
    control (src = L0 null heads)."""
    rows = {"clean": [], "full": [], "path": [], "path_null": []}
    for pc, po in probe_batches:
        # 1. clean + capture L1 input
        h1, store_clean = capture_block_input(model, 1)
        _, tau_clean = run_clean(model, pc, po, device)
        h1.remove()
        x_clean_L1, cond = store_clean["x"], store_clean["cond"]
        # 2. full L0-source ablation + capture L1 input
        h2, store_corr = capture_block_input(model, 1)
        _, tau_full = run_with_ablation(model, 0, src_heads_L0, pc, po, device)
        h2.remove()
        x_corr_L1 = store_corr["x"]
        # 3. path-restricted L0->L1 QK
        tau_path = _run_path_restricted(model, dst_heads_L1, x_clean_L1, x_corr_L1,
                                        cond, pc, po, device)
        rows["clean"].append(tau_clean)
        rows["full"].append(tau_full)
        rows["path"].append(tau_path)
        # optional null-path control
        if null_src_heads is not None:
            h3, store_null = capture_block_input(model, 1)
            _, _tau_full_null = run_with_ablation(model, 0, null_src_heads, pc, po, device)
            h3.remove()
            x_corr_null = store_null["x"]
            rows["path_null"].append(
                _run_path_restricted(model, dst_heads_L1, x_clean_L1, x_corr_null,
                                     cond, pc, po, device)
            )

    tc = np.mean(rows["clean"], axis=0)
    tf = np.mean(rows["full"], axis=0)
    tp = np.mean(rows["path"], axis=0)
    dst = list(dst_heads_L1)
    out = {
        "seed": int(seed),
        "src_heads_L0": list(src_heads_L0),
        "dst_heads_L1": dst,
        # readout 2a: L1-dst tau collapse under full L0 ablation
        "dtau_2a_l1_dst_mean": float((tf[1, dst] - tc[1, dst]).mean()),
        "tau_clean_l1_dst": [float(tc[1, h]) for h in dst],
        "tau_full_l1_dst": [float(tf[1, h]) for h in dst],
        # readout 2b: downstream path_fraction (global aggregates downstream layers)
        "path_fraction_global": _path_fraction(tc, tf, tp, "global", downstream_layers),
        "n_batch": len(probe_batches),
    }
    for L in downstream_layers:
        out[f"path_fraction_L{L}"] = _path_fraction(tc, tf, tp, L, downstream_layers)
    if rows["path_null"]:
        tpn = np.mean(rows["path_null"], axis=0)
        out["path_fraction_global_null"] = _path_fraction(tc, tf, tpn, "global", downstream_layers)
    return out
