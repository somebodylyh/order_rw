"""P3': causal verification helpers for the L0 physical-order carrier."""
import contextlib
import csv as _csv
import json as _json
import numbers
import pathlib
import sys

import numpy as np
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_BLOCK_DIR = _ROOT / "block_lo_arm_order_network"
for _path in (str(_ROOT), str(_BLOCK_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from analyses.physical_signal_source import (  # noqa: E402
    P2_CARRIERS,
    _stack,
    block_swap_chunk,
    carrier_b65_per_text,
    row_normalize_l1,
    slot_only_r2,
    valid_edge_mask,
    within_text_noise_floor,
)
from analyses.position_prior_decomp import pe_ablation  # noqa: E402
from none_separated_block_graph import build_none_separated_B, rollout_by_method, discovery_metrics  # noqa: E402
from neural_readout.extract_b import _load_model_and_chunks  # noqa: E402
from analyses.canonical_reanalysis import canonical_scan, random_reveal_orders  # noqa: E402
from per_head_order_scan import _attn_to_A_block_loss_aligned_with_none_vec  # noqa: E402


_ALLOWED_PE_MODES = {"none", "wpe", "wtpe", "both"}


def _validate_heads(heads):
    heads = list(heads)
    if not heads:
        raise ValueError("heads must not be empty")
    out = []
    for index, head in enumerate(heads):
        if isinstance(head, bool) or not isinstance(head, numbers.Integral):
            raise TypeError(f"heads[{index}] must be a non-bool integer")
        head = int(head)
        if head < 0:
            raise ValueError(f"heads[{index}] must be nonnegative")
        out.append(head)
    if len(set(out)) != len(out):
        raise ValueError("heads must be unique")
    return out


def _validate_layer(layer):
    if isinstance(layer, bool) or not isinstance(layer, numbers.Integral):
        raise TypeError("layer must be a non-bool integer")
    layer = int(layer)
    if layer < 0:
        raise ValueError("layer must be nonnegative")
    return layer


def _run_intervened_b65(
    ckpt_path,
    layer,
    heads,
    M=24,
    n_reveals=32,
    fixed_reveal_seed=0,
    device="cpu",
    pe_mode=None,
    corrupt_fn=None,
    attn_transform=None,
    bundle=None,
):
    if bundle is None:
        model, chunks, clean_perm, dev, _ = _load_model_and_chunks(
            ckpt_path, M, seed=0, device=device, split="train"
        )
    else:
        model, chunks, clean_perm, dev = bundle
    inv = clean_perm.inv_perm_model_to_phys.cpu().numpy()
    if corrupt_fn is not None:
        chunks = corrupt_fn(chunks)
        if not isinstance(chunks, torch.Tensor):
            raise TypeError("corrupt_fn must return a torch.LongTensor")
        if chunks.shape != (M, 256):
            raise ValueError(f"corrupt_fn must preserve chunk shape {(M, 256)}")

    reveals = random_reveal_orders(n_reveals, fixed_reveal_seed)
    results = {head: ([], []) for head in heads}
    ctx = contextlib.nullcontext(model) if pe_mode is None else pe_ablation(model, pe_mode)

    with ctx:
        for text_index in range(M):
            accum = None
            for reveal in reveals:
                po = torch.from_numpy(np.asarray(reveal)[None, :]).to(dev)
                _, _, attn_list = model.forward_fn(
                    chunks[text_index : text_index + 1].to(dev),
                    po,
                    return_attentions=True,
                )
                attn = torch.stack(attn_list, dim=0).cpu().numpy()[:, 0]
                if attn_transform is not None:
                    attn = attn_transform(attn, layer)
                A = _attn_to_A_block_loss_aligned_with_none_vec(attn, reveal, inv)
                accum = A.astype(np.float64) if accum is None else accum + A

            A_mean = accum / float(n_reveals)
            for head in heads:
                B = build_none_separated_B(A_mean[layer, head])
                tau = float(discovery_metrics(rollout_by_method(B, "C-D+L"))["tau_vs_l2r"])
                results[head][0].append(B)
                results[head][1].append(tau)

    return results


@torch.no_grad()
def p3_canonical_b65(
    ckpt_path,
    layer,
    heads,
    M=24,
    n_reveals=32,
    fixed_reveal_seed=0,
    device="cpu",
    pe_mode=None,
    corrupt_fn=None,
    attn_transform=None,
    bundle=None,
):
    """Intervention-aware canonical B65 builder.

    With no interventions, this delegates to `carrier_b65_per_text` exactly.
    Otherwise it runs the same canonical readout after applying optional
    position-path, token-corruption, and attention-map interventions.
    """
    layer = _validate_layer(layer)
    heads = _validate_heads(heads)
    if pe_mode is not None and pe_mode not in _ALLOWED_PE_MODES:
        raise ValueError(f"pe_mode must be one of {_ALLOWED_PE_MODES} or None")

    if pe_mode is None and corrupt_fn is None and attn_transform is None:
        if bundle is not None:
            return _p3_canonical_b65_from_bundle(
                bundle,
                layer,
                heads,
                M=M,
                n_reveals=n_reveals,
                fixed_reveal_seed=fixed_reveal_seed,
            )
        return {
            head: carrier_b65_per_text(
                ckpt_path,
                layer,
                head,
                M=M,
                n_reveals=n_reveals,
                fixed_reveal_seed=fixed_reveal_seed,
                device=device,
            )
            for head in heads
        }

    return _run_intervened_b65(
        ckpt_path,
        layer,
        heads,
        M=M,
        n_reveals=n_reveals,
        fixed_reveal_seed=fixed_reveal_seed,
        device=device,
        pe_mode=pe_mode,
        corrupt_fn=corrupt_fn,
        attn_transform=attn_transform,
        bundle=bundle,
    )


def _load_p3_bundle(ckpt_path, M, device="cpu"):
    model, chunks, clean_perm, dev, _ = _load_model_and_chunks(
        ckpt_path, M, seed=0, device=device, split="train"
    )
    return model, chunks, clean_perm, dev


@torch.no_grad()
def _p3_canonical_b65_from_bundle(
    bundle,
    layer,
    heads,
    M=24,
    n_reveals=32,
    fixed_reveal_seed=0,
    pe_mode=None,
    corrupt_fn=None,
    attn_transform=None,
):
    model, chunks, clean_perm, dev = bundle
    inv = clean_perm.inv_perm_model_to_phys.cpu().numpy()
    reveals = random_reveal_orders(n_reveals, fixed_reveal_seed)
    results = {head: ([], []) for head in heads}
    ctx = contextlib.nullcontext(model) if pe_mode is None else pe_ablation(model, pe_mode)
    if corrupt_fn is not None:
        chunks = corrupt_fn(chunks)
    with ctx:
        for text_index in range(M):
            accum = None
            for reveal in reveals:
                po = torch.from_numpy(np.asarray(reveal)[None, :]).to(dev)
                _, _, attn_list = model.forward_fn(
                    chunks[text_index : text_index + 1].to(dev),
                    po,
                    return_attentions=True,
                )
                attn = torch.stack(attn_list, dim=0).detach().cpu().numpy()[:, 0]
                if attn_transform is not None:
                    attn = attn_transform(attn, layer)
                A = _attn_to_A_block_loss_aligned_with_none_vec(attn, reveal, inv)
                accum = A.astype(np.float64) if accum is None else accum + A
            A_mean = accum / float(n_reveals)
            for head in heads:
                B = build_none_separated_B(A_mean[layer, head])
                tau = float(discovery_metrics(rollout_by_method(B, "C-D+L"))["tau_vs_l2r"])
                results[head][0].append(B)
                results[head][1].append(tau)
    return results


@torch.no_grad()
def _carrier_b65_per_text_bundle(bundle, layer, head, M=24, n_reveals=32,
                                 fixed_reveal_seed=0, return_halves=False):
    model, chunks, clean_perm, dev = bundle
    inv = clean_perm.inv_perm_model_to_phys.cpu().numpy()
    reveals = random_reveal_orders(n_reveals, fixed_reveal_seed)
    half = n_reveals // 2
    B_list, tau_list = [], []
    halfA_list, halfB_list = [], []
    for text_index in range(M):
        accum = None
        accum_a = None
        accum_b = None
        for reveal_index, reveal in enumerate(reveals):
            po = torch.from_numpy(np.asarray(reveal)[None, :]).to(dev)
            _, _, attn_list = model.forward_fn(
                chunks[text_index : text_index + 1].to(dev),
                po,
                return_attentions=True,
            )
            attn = torch.stack(attn_list, dim=0).detach().cpu().numpy()[:, 0]
            A = _attn_to_A_block_loss_aligned_with_none_vec(attn, reveal, inv)
            accum = A.astype(np.float64) if accum is None else accum + A
            if return_halves:
                if reveal_index < half:
                    accum_a = A.astype(np.float64) if accum_a is None else accum_a + A
                else:
                    accum_b = A.astype(np.float64) if accum_b is None else accum_b + A
        B = build_none_separated_B((accum / n_reveals)[layer, head])
        B_list.append(B)
        tau_list.append(float(discovery_metrics(rollout_by_method(B, "C-D+L"))["tau_vs_l2r"]))
        if return_halves:
            nA = max(half, 1)
            nB = max(n_reveals - half, 1)
            halfA_list.append(build_none_separated_B((accum_a / nA)[layer, head]))
            halfB_list.append(build_none_separated_B((accum_b / nB)[layer, head]))
    if return_halves:
        return B_list, tau_list, halfA_list, halfB_list
    return B_list, tau_list


def causal_uniform_transform(heads):
    """Return an attention transform that mean-patches the listed heads.

    The affected heads are replaced with a uniform distribution over their
    causal support; all other layers/heads are left untouched.
    """
    heads = _validate_heads(heads)

    def _transform(attn_LH, layer):
        out = attn_LH.copy()
        for head in heads:
            support = np.tril(np.ones_like(out[layer, head], dtype=np.float64))
            out[layer, head] = support / np.maximum(support.sum(axis=-1, keepdims=True), 1.0)
        return out

    return _transform


def _mean_abs(tau_list):
    return float(np.mean(np.abs(np.asarray(tau_list)))) if len(tau_list) else 0.0


def a0_self_qk_calibration(ckpt_path, layer, heads, M=8, n_reveals=8, device="cpu"):
    """Calibration only: mean-patch each carrier head's own QK and read itself."""
    clean = p3_canonical_b65(ckpt_path, layer, heads, M=M, n_reveals=n_reveals, device=device)
    per_head = []
    for head in heads:
        patched = p3_canonical_b65(
            ckpt_path,
            layer,
            [head],
            M=M,
            n_reveals=n_reveals,
            device=device,
            attn_transform=causal_uniform_transform([head]),
        )
        per_head.append(
            {
                "head": head,
                "tau_clean": _mean_abs(clean[head][1]),
                "tau_selfpatch": _mean_abs(patched[head][1]),
            }
        )
    return {"layer": layer, "load_bearing": False, "per_head": per_head}


def a1_head_local(ckpt_path, layer, heads, M=8, n_reveals=8, device="cpu"):
    """Patch head h's QK, read head h' != h."""
    clean = p3_canonical_b65(ckpt_path, layer, heads, M=M, n_reveals=n_reveals, device=device)
    tau_clean = {head: _mean_abs(clean[head][1]) for head in heads}
    delta = {head: {} for head in heads}
    off_diag = []
    for patched_head in heads:
        patched = p3_canonical_b65(
            ckpt_path,
            layer,
            heads,
            M=M,
            n_reveals=n_reveals,
            device=device,
            attn_transform=causal_uniform_transform([patched_head]),
        )
        for read_head in heads:
            diff = tau_clean[read_head] - _mean_abs(patched[read_head][1])
            delta[patched_head][read_head] = diff
            if patched_head != read_head:
                off_diag.append(abs(diff))
    return {
        "layer": layer,
        "delta": delta,
        "parallel_independent": bool(max(off_diag) < 0.1) if off_diag else True,
    }


def _cluster_tau(B_by_head, kept):
    """Per-text mean-B over kept heads -> mean |tau(C-D+L)| over texts."""
    n_text = len(next(iter(B_by_head.values())))
    taus = []
    for text_index in range(n_text):
        B = np.mean([B_by_head[head][text_index] for head in kept], axis=0)
        taus.append(abs(float(discovery_metrics(rollout_by_method(B, "C-D+L"))["tau_vs_l2r"])))
    return float(np.mean(taus))


def a1_leave_k_out(ckpt_path, layer, heads, M=8, n_reveals=8, device="cpu"):
    """No-retrain redundancy ladder on the aggregated clean B_cluster."""
    clean = p3_canonical_b65(ckpt_path, layer, heads, M=M, n_reveals=n_reveals, device=device)
    B_by_head = {head: clean[head][0] for head in heads}
    full = _cluster_tau(B_by_head, heads)
    subsets = []
    if len(heads) > 1:
        for drop in heads:
            kept = [head for head in heads if head != drop]
            subsets.append({"kept": kept, "tau": _cluster_tau(B_by_head, kept)})
    return {"layer": layer, "full_tau": full, "subset_tau": subsets, "single_head": bool(len(heads) == 1)}


def _b_verdict(tau_none, tau_abl, r2_none, r2_abl, drop=0.3):
    tau_collapsed = (tau_none - tau_abl) > drop
    r2_collapsed = (r2_none - r2_abl) > drop
    if tau_collapsed and r2_collapsed:
        return "base_map_collapses"
    if tau_collapsed:
        return "mixed_tau_only"
    if r2_collapsed:
        return "mixed_r2_only"
    return "intact"


def _base_map_cross_r2(B_table_src, B_eval, mask, normalize=False):
    """How well the CLEAN slot-mean fixed-map table predicts `B_eval`.

    B_hat is the content-free slot-pair mean over `B_table_src` (the clean carrier
    B65). Predicting held-out CLEAN B keeps R² high; predicting the position-ablated
    B drops R² IF the ablation actually disrupts that fixed map. This is the correct
    base-map-survival metric: unlike a self-referential slot_only_r2(B_abl), a
    degenerate-but-cross-text-consistent ablated pattern does NOT score high here,
    because it is scored against the *clean* table, not its own."""
    Xtr = _stack(B_table_src, mask, normalize)
    Xte = _stack(B_eval, mask, normalize)
    B_hat = Xtr.mean(axis=0)
    ss_res = ((Xte - B_hat) ** 2).sum()
    ss_tot = ((Xte - Xte.mean()) ** 2).sum() + 1e-12
    return float(1.0 - ss_res / ss_tot)


def _b_per_head_row(head, clean_B, abl_B, tau_none, tau_abl, mask):
    """Build one P3'-B per-head row using the clean-predict base-map R².

    tau_abl/abl_B MUST come from the POSITION ablation (pe_mode='both'), never from
    the A0 self-QK patch. r2_none/r2_abl both score against the SAME clean fixed-map
    table (a train-half of the clean carrier B65): r2_none predicts the held-out
    clean half, r2_abl predicts the position-ablated B. A drop in r2_abl means the
    clean fixed map no longer explains the attention -> the base map was disrupted.
    (r2_abl_selfref keeps the old self-referential value for transparency; NOT used
    in the verdict.)"""
    n_tr = max(1, len(clean_B) // 2)
    table_src = clean_B[:n_tr]
    clean_eval = clean_B[n_tr:] if len(clean_B) - n_tr >= 1 else clean_B
    r2_none = _base_map_cross_r2(table_src, clean_eval, mask)
    r2_abl = _base_map_cross_r2(table_src, abl_B, mask)
    return {
        "head": head,
        "tau_none": tau_none,
        "tau_abl": tau_abl,
        "r2_none": r2_none,
        "r2_abl": r2_abl,
        "r2_abl_selfref": slot_only_r2(abl_B, mask, normalize=False),
        "verdict": _b_verdict(tau_none, tau_abl, r2_none, r2_abl),
    }


def b_position_ablation(ckpt_path, layer, heads, M=12, n_reveals=8, device="cpu"):
    """Ablate the position path and require joint tau + base-map collapse."""
    mask = valid_edge_mask(65)
    none = p3_canonical_b65(ckpt_path, layer, heads, M=M, n_reveals=n_reveals, device=device)
    abl = p3_canonical_b65(
        ckpt_path,
        layer,
        heads,
        M=M,
        n_reveals=n_reveals,
        device=device,
        pe_mode="both",
    )
    per_head = [
        _b_per_head_row(head, none[head][0], abl[head][0],
                        _mean_abs(none[head][1]), _mean_abs(abl[head][1]), mask)
        for head in heads
    ]
    return {"layer": layer, "per_head": per_head}


def make_block_swap_corrupt(swaps, block_len=4):
    """Return a chunk corruption fn that swaps the given fixed-length blocks."""

    def _corrupt(chunks):
        out = chunks.clone()
        for index in range(out.shape[0]):
            row = out[index].detach().cpu()
            swapped = block_swap_chunk(row, swaps, block_len=block_len)
            out[index] = torch.as_tensor(swapped, device=out.device)
        return out

    return _corrupt


def _resid_change(B_clean, B_corr, mask):
    deltas = []
    for clean_B, corr_B in zip(B_clean, B_corr):
        diff = row_normalize_l1(corr_B, mask) - row_normalize_l1(clean_B, mask)
        deltas.append(float((diff[mask] ** 2).mean()))
    return float(np.mean(deltas))


def c_content_residual(ckpt_path, layer, heads, swaps=((1, 2),), M=12, n_reveals=8, device="cpu"):
    """Corrupt content at fixed slots and measure residual change above the floor."""
    mask = valid_edge_mask(65)
    corrupt = make_block_swap_corrupt(swaps)
    per_head = []
    for head in heads:
        clean_B, clean_tau, halfA, halfB = carrier_b65_per_text(
            ckpt_path,
            layer,
            head,
            M=M,
            n_reveals=n_reveals,
            device=device,
            return_halves=True,
        )
        floor = within_text_noise_floor(halfA, halfB, mask)
        corr = p3_canonical_b65(
            ckpt_path,
            layer,
            [head],
            M=M,
            n_reveals=n_reveals,
            device=device,
            corrupt_fn=corrupt,
        )
        corr_B, corr_tau = corr[head]
        resid_change = _resid_change(clean_B, corr_B, mask)
        ratio = float(resid_change / (floor + 1e-9))
        per_head.append(
            {
                "head": head,
                "resid_change": resid_change,
                "noise_floor": floor,
                "ratio": ratio,
                "dtau": _mean_abs(clean_tau) - _mean_abs(corr_tau),
                "content_driven": bool(ratio > 1.0),
            }
        )
    return {"layer": layer, "per_head": per_head}


def canonical_null_heads(ckpt_path, layer, carriers, k=2, M=4, device="cpu"):
    """The k lowest-|tau| heads at `layer`, excluding the carrier heads."""
    carrier_set = set(carriers)
    rows = [
        row
        for row in canonical_scan(
            ckpt_path,
            M=M,
            batch_size=8,
            device=device,
            methods=("C-D+L",),
        )
        if row["layer"] == layer and row["method"] == "C-D+L" and row["head"] not in carrier_set
    ]
    rows.sort(key=lambda row: row["abs_tau"])
    return [row["head"] for row in rows[:k]]


def _canonical_head_tau(ckpt_path, layer, head, M, device, ablate=None):
    rows = canonical_scan(
        ckpt_path,
        M=M,
        batch_size=8,
        device=device,
        methods=("C-D+L",),
        ablate=ablate,
    )
    for row in rows:
        if row["layer"] == layer and row["head"] == head and row["method"] == "C-D+L":
            return abs(row["tau_vs_l2r"])
    raise ValueError("head not found")


def d_locus(ckpt_path, layer, head, M=8, n_reveals=8, device="cpu"):
    """Verify the same-head QK intervention is causal while OV/output is degenerate."""
    tau_clean = _mean_abs(
        p3_canonical_b65(ckpt_path, layer, [head], M=M, n_reveals=n_reveals, device=device)[head][1]
    )
    tau_pe = _mean_abs(
        p3_canonical_b65(
            ckpt_path,
            layer,
            [head],
            M=M,
            n_reveals=n_reveals,
            device=device,
            pe_mode="both",
        )[head][1]
    )
    tau_clean_scan = _canonical_head_tau(ckpt_path, layer, head, M, device, ablate=None)
    tau_ov = _canonical_head_tau(ckpt_path, layer, head, M, device, ablate=(layer, [head]))
    return {
        "tau_clean": tau_clean,
        "tau_pe": tau_pe,
        "tau_ovablate": tau_ov,
        "qk_changes": bool((tau_clean - tau_pe) > 0.3),
        "ov_degenerate": bool(abs(tau_clean_scan - tau_ov) < 0.1),
    }


def classify_p3prime(b_result, c_result):
    """B+ confirmed iff majority base-map collapse and at least one content-driven head."""
    b_verdicts = [row["verdict"] for row in b_result["per_head"]]
    majority_base = sum(verdict == "base_map_collapses" for verdict in b_verdicts) > len(b_verdicts) / 2
    any_content = any(row["content_driven"] for row in c_result["per_head"])
    return "B+ confirmed" if (majority_base and any_content) else "mixed/departure"


def run_seed(seed, root="runs/handoff_overnight", out_dir=None, M=8, n_reveals=8, device="cpu"):
    """Full P3′ per seed: A0/A1/B/C/D + null-head controls + verdict."""
    out = pathlib.Path(out_dir or f"runs/p3prime_causal/seed{seed}")
    out.mkdir(parents=True, exist_ok=True)
    layer, heads = P2_CARRIERS[seed]
    ckpt = f"{root}/seed{seed}/ckpt_step10000.pt"
    null_heads = canonical_null_heads(ckpt, layer, heads, k=2, M=max(2, M // 2), device=device)
    bundle = _load_p3_bundle(ckpt, M, device=device)

    clean = _p3_canonical_b65_from_bundle(
        bundle,
        layer,
        heads,
        M=M,
        n_reveals=n_reveals,
        fixed_reveal_seed=0,
    )
    patched_all = _p3_canonical_b65_from_bundle(
        bundle,
        layer,
        heads,
        M=M,
        n_reveals=n_reveals,
        fixed_reveal_seed=0,
        attn_transform=causal_uniform_transform(heads),
    )
    a0 = {
        "layer": layer,
        "load_bearing": False,
        "per_head": [
            {
                "head": head,
                "tau_clean": _mean_abs(clean[head][1]),
                "tau_selfpatch": _mean_abs(patched_all[head][1]),
            }
            for head in heads
        ],
    }
    patched_by_head = {
        patched_head: _p3_canonical_b65_from_bundle(
            bundle,
            layer,
            heads,
            M=M,
            n_reveals=n_reveals,
            fixed_reveal_seed=0,
            attn_transform=causal_uniform_transform([patched_head]),
        )
        for patched_head in heads
    }
    a1_local = {
        "layer": layer,
        "delta": {
            patched_head: {
                read_head: _mean_abs(clean[read_head][1]) - _mean_abs(patched_by_head[patched_head][read_head][1])
                for read_head in heads
            }
            for patched_head in heads
        },
        "parallel_independent": True,
    }
    off_diag = [
        abs(a1_local["delta"][patched_head][read_head])
        for patched_head in heads
        for read_head in heads
        if patched_head != read_head
    ]
    a1_local["parallel_independent"] = bool(max(off_diag) < 0.1) if off_diag else True
    a1_loko = {"layer": layer, "full_tau": _cluster_tau({h: clean[h][0] for h in heads}, heads),
               "subset_tau": [{"kept": [h for h in heads if h != drop],
                                "tau": _cluster_tau({h: clean[h][0] for h in heads},
                                                    [h for h in heads if h != drop])}
                               for drop in heads] if len(heads) > 1 else [],
               "single_head": bool(len(heads) == 1)}
    abl = _p3_canonical_b65_from_bundle(
        bundle,
        layer,
        heads,
        M=M,
        n_reveals=n_reveals,
        fixed_reveal_seed=0,
        pe_mode="both",
    )
    b = {
        "layer": layer,
        "per_head": [
            _b_per_head_row(
                head,
                clean[head][0],
                abl[head][0],
                _mean_abs(clean[head][1]),
                _mean_abs(abl[head][1]),   # POSITION-ablation tau, not the A0 self-QK patch
                valid_edge_mask(65),
            )
            for head in heads
        ],
    }
    c = {
        "layer": layer,
        "per_head": [],
    }
    corrupt = make_block_swap_corrupt(((1, 2),))
    for head in heads:
        clean_B, clean_tau, halfA, halfB = _carrier_b65_per_text_bundle(
            bundle,
            layer,
            head,
            M=M,
            n_reveals=n_reveals,
            fixed_reveal_seed=0,
            return_halves=True,
        )
        floor = within_text_noise_floor(halfA, halfB, valid_edge_mask(65))
        corr = _p3_canonical_b65_from_bundle(
            bundle,
            layer,
            [head],
            M=M,
            n_reveals=n_reveals,
            fixed_reveal_seed=0,
            corrupt_fn=corrupt,
        )
        corr_B, corr_tau = corr[head]
        resid_change = _resid_change(clean_B, corr_B, valid_edge_mask(65))
        ratio = float(resid_change / (floor + 1e-9))
        c["per_head"].append(
            {
                "head": head,
                "resid_change": resid_change,
                "noise_floor": floor,
                "ratio": ratio,
                "dtau": _mean_abs(clean_tau) - _mean_abs(corr_tau),
                "content_driven": bool(ratio > 1.0),
            }
        )
    d = {head: d_locus(ckpt, layer, head, M=M, n_reveals=n_reveals, device=device) for head in heads}
    null_clean = _p3_canonical_b65_from_bundle(
        bundle,
        layer,
        null_heads,
        M=M,
        n_reveals=n_reveals,
        fixed_reveal_seed=0,
    )
    null_abl = _p3_canonical_b65_from_bundle(
        bundle,
        layer,
        null_heads,
        M=M,
        n_reveals=n_reveals,
        fixed_reveal_seed=0,
        pe_mode="both",
    )
    b_null = {
        "layer": layer,
        "per_head": [
            _b_per_head_row(
                head,
                null_clean[head][0],
                null_abl[head][0],
                _mean_abs(null_clean[head][1]),
                _mean_abs(null_abl[head][1]),
                valid_edge_mask(65),
            )
            for head in null_heads
        ],
    }
    verdict = classify_p3prime(b, c)

    summary = {
        "seed": seed,
        "carrier": {"layer": layer, "heads": heads},
        "null_heads": null_heads,
        "A0_calibration": a0,
        "A1_head_local": a1_local,
        "A1_leave_k_out": a1_loko,
        "B_position": b,
        "C_content": c,
        "D_locus": d,
        "B_null_control": b_null,
        "verdict": verdict,
    }
    _json.dump(summary, open(out / "p3prime.json", "w"), indent=2, default=float)

    c_by_head = {row["head"]: row for row in c["per_head"]}
    with open(out / "p3prime.csv", "w", newline="") as handle:
        writer = _csv.DictWriter(
            handle,
            fieldnames=[
                "seed",
                "layer",
                "head",
                "tau_none",
                "tau_abl",
                "r2_none",
                "r2_abl",
                "b_verdict",
                "resid_change",
                "noise_floor",
                "ratio",
                "content_driven",
            ],
        )
        writer.writeheader()
        for row in b["per_head"]:
            c_row = c_by_head[row["head"]]
            writer.writerow(
                {
                    "seed": seed,
                    "layer": layer,
                    "head": row["head"],
                    "tau_none": round(row["tau_none"], 4),
                    "tau_abl": round(row["tau_abl"], 4),
                    "r2_none": round(row["r2_none"], 4),
                    "r2_abl": round(row["r2_abl"], 4),
                    "b_verdict": row["verdict"],
                    "resid_change": round(c_row["resid_change"], 6),
                    "noise_floor": round(c_row["noise_floor"], 6),
                    "ratio": round(c_row["ratio"], 3),
                    "content_driven": c_row["content_driven"],
                }
            )
    return summary
