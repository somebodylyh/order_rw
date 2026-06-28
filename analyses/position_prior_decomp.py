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

TRAJ_ROOT = "runs/handoff_overnight"


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


# ── Task 4: Part-1 step-0 4-arm tau table ────────────────────────────────────

from analyses.path_patch_handoff import (  # noqa: E402
    load_model_and_chunks_seed, make_probe_batch, run_clean)


def tau_table_under(model, chunks, which, n_batches=4, bs_mean=16, device="cpu"):
    dev = torch.device(device)
    accum = []
    for i in range(n_batches):
        pc, po = make_probe_batch(chunks, bs_mean, np.random.default_rng(i))
        with pe_ablation(model, which):
            _, tau = run_clean(model, pc, po, dev)
        accum.append(tau)
    return np.mean(accum, axis=0)


# Report arm name -> pe_ablation token. (Bug guard: pe_ablation only knows
# none/wpe/wtpe/both; passing the report names directly registers NO hooks and
# makes every arm equal to full.)
_ARM_TO_ABL = {"full": "none", "zero_wpe": "wpe", "zero_wtpe": "wtpe", "zero_both": "both"}


def part1_arms(seed, root, n_batches=4, bs_mean=16, device="cpu"):
    ckpt = f"{root}/seed{seed}/ckpt_step0.pt"
    model, chunks, _ = load_model_and_chunks_seed(
        ckpt, max(64, bs_mean * n_batches), torch.device(device))
    return {arm: tau_table_under(model, chunks, _ARM_TO_ABL[arm], n_batches, bs_mean, device)
            for arm in ("full", "zero_wpe", "zero_wtpe", "zero_both")}


# ── Task 5: Part-1 summary (ablation table + floor + frame sanity) ───────────

import csv as _csv  # noqa: E402
import json as _json  # noqa: E402


def _training_inv_perm(ckpt_path):
    """Length-64 inv_perm_model_to_phys from the ckpt clean_protocol."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return np.asarray(ck["clean_protocol"]["inv_perm_model_to_phys"], dtype=int)


def _winner_for_seed(seed, root):
    from analyses.emergence_characterization import load_tau_trajectory, winner
    traj = load_tau_trajectory(seed, root)
    return winner(traj, seed)


def run_part1(seed, root, out_dir, n_batches=4, bs_mean=16, device="cpu"):
    from analyses.emergence_characterization import extract_carrier_B
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt0 = f"{root}/seed{seed}/ckpt_step0.pt"

    arms = part1_arms(seed, root, n_batches, bs_mean, device)
    floor = floor_taus()

    w = _winner_for_seed(seed, root)
    L = w["winning_layer"]
    heads = w["winner_heads"]
    inv = _training_inv_perm(ckpt0)
    perm_baseline = tau_vs_arange(inv[np.arange(64)])

    B0 = extract_carrier_B(seed, 0, L, root=root, bs_mean=bs_mean,
                           n_batches=n_batches, device=device)  # (8,65,65)
    ms, ph = [], []
    for h in heads:
        fr = tau_two_frames(B0[h], inv)
        ms.append(fr["tau_model_slot"])
        ph.append(fr["tau_physical"])
    frame_sanity = {"tau_model_slot": float(np.mean(ms)),
                    "tau_physical": float(np.mean(ph)),
                    "tau_perm_baseline": float(perm_baseline)}

    # carrier-head max/mean tau per arm (winning layer)
    def carrier_stat(tab):
        vals = [tab[L, h] for h in heads]
        return float(np.max(vals)), float(np.mean(vals))

    with open(out / "part1_ablation.csv", "w", newline="") as f:
        wr = _csv.writer(f)
        wr.writerow(["arm", "carrier_max_tau", "carrier_mean_tau"])
        for arm in ("full", "zero_wpe", "zero_wtpe", "zero_both"):
            mx, mn = carrier_stat(arms[arm])
            wr.writerow([arm, round(mx, 4), round(mn, 4)])
        wr.writerow(["uniform_causal_floor", round(floor["uniform_causal"], 4), ""])
        wr.writerow(["random_B_null", round(floor["random_B"], 4), ""])

    fmax = {arm: carrier_stat(arms[arm])[0] for arm in arms}
    summary = {
        "seed": seed, "winning_layer": L, "carrier_heads": heads, "tier": w["tier"],
        "arms_carrier_max": fmax,
        "delta_wpe": fmax["full"] - fmax["zero_wpe"],
        "delta_wtpe": fmax["full"] - fmax["zero_wtpe"],
        "delta_both": fmax["full"] - fmax["zero_both"],
        "floor": floor, "frame_sanity": frame_sanity,
    }
    _json.dump(summary, open(out / "part1.json", "w"), indent=2, default=float)
    return summary


# ── Task 6: layout generator (training anchor + fixed random) ────────────────

from clean_training_protocol import build_clean_block_permutation  # noqa: E402


def _perm_dict(layout_id, clean_perm, is_train, rng_seed):
    return {"layout_id": int(layout_id),
            "perm": clean_perm.block_perm_phys_to_model.tolist(),
            "inv_perm": clean_perm.inv_perm_model_to_phys.tolist(),
            "is_training_layout": bool(is_train), "rng_seed": rng_seed}


def make_layouts(training_clean_perm, K=8, seed_base=1000):
    layouts = [_perm_dict(0, training_clean_perm, True, None)]
    for i in range(1, K):
        cp = build_clean_block_permutation(64, seed=seed_base + i)
        layouts.append(_perm_dict(i, cp, False, seed_base + i))
    return layouts


def save_layouts(layouts, path):
    _json.dump(layouts, open(path, "w"), indent=2)


def load_layouts(path):
    return _json.load(open(path))


# ── Task 7: relayout chunks across layouts ───────────────────────────────────

from clean_training_protocol import (  # noqa: E402
    CleanPermutation, model_to_phys_idx_clean, phys_to_model_idx_clean)


def clean_perm_from_layout(layout_dict):
    return CleanPermutation(
        block_perm_phys_to_model=torch.tensor(layout_dict["perm"], dtype=torch.long),
        inv_perm_model_to_phys=torch.tensor(layout_dict["inv_perm"], dtype=torch.long))


def relayout_chunks(chunks_model, training_clean_perm, layout_clean_perm):
    """Model-frame chunks (trained layout) -> physical -> model-frame under layout."""
    idx_phys = model_to_phys_idx_clean(chunks_model, training_clean_perm)
    return phys_to_model_idx_clean(idx_phys, layout_clean_perm)


# ── Task 8: binding scores (tau_pos / tau_content) + anchor-validity gate ─────

from attention_trajectory import extract_all_layer_B  # noqa: E402


def _training_clean_perm(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    proto = ck["clean_protocol"]
    return CleanPermutation(
        block_perm_phys_to_model=torch.tensor(proto["block_perm_phys_to_model"], dtype=torch.long),
        inv_perm_model_to_phys=torch.tensor(proto["inv_perm_model_to_phys"], dtype=torch.long))


def _layout_binding(model, chunks, training_perm, layout, winning_layer, carrier_heads,
                    n_batches, bs_mean, device):
    """Mean tau_pos / tau_content over carrier heads & batches for one layout."""
    dev = torch.device(device)
    lp = clean_perm_from_layout(layout)
    inv_k = np.asarray(layout["inv_perm"], dtype=int)
    pos_acc, cont_acc = [], []
    for i in range(n_batches):
        pc, po = make_probe_batch(chunks, bs_mean, np.random.default_rng(i))
        pc_re = relayout_chunks(pc, training_perm, lp)
        attn_list, _ = run_clean(model, pc_re, po, dev)
        B = extract_all_layer_B(attn_list, po)[winning_layer].mean(axis=0)  # (8,65,65)
        for h in carrier_heads:
            order = rollout_order(B[h])
            pos_acc.append(tau_vs_arange(order))
            cont_acc.append(tau_vs_arange(inv_k[order]))
    return float(np.mean(pos_acc)), float(np.mean(cont_acc))


def binding_scores(seed, ckpt_step, layouts, winning_layer, carrier_heads, tier,
                   root=TRAJ_ROOT, bs_mean=16, n_batches=4, device="cpu",
                   collapse=0.3):
    ckpt = f"{root}/seed{seed}/ckpt_step{ckpt_step}.pt"
    model, chunks, _ = load_model_and_chunks_seed(
        ckpt, max(64, bs_mean * n_batches), torch.device(device))
    training_perm = _training_clean_perm(ckpt)
    per_layout = []
    for lay in layouts:
        tp, tc = _layout_binding(model, chunks, training_perm, lay, winning_layer,
                                 carrier_heads, n_batches, bs_mean, device)
        per_layout.append({"layout_id": lay["layout_id"], "is_training": lay["is_training_layout"],
                           "tau_pos": tp, "tau_content": tc})
    anchor = next(p for p in per_layout if p["is_training"])
    relay = [p for p in per_layout if not p["is_training"]]
    relay_pos = float(np.mean([p["tau_pos"] for p in relay])) if relay else float("nan")
    relay_cont = float(np.mean([p["tau_content"] for p in relay])) if relay else float("nan")
    anchor_thr = 0.95 if tier == "strong" else 0.60
    anchor_valid = bool(anchor["tau_pos"] >= anchor_thr)
    if not anchor_valid:
        verdict = "invalid"
    elif relay_pos < collapse and relay_cont < collapse:
        verdict = "ood-break"
    elif relay_cont > relay_pos:
        verdict = "content-bound"
    else:
        verdict = "slot-scaffold"
    return {"seed": seed, "ckpt_step": ckpt_step, "winning_layer": winning_layer,
            "carrier_heads": list(carrier_heads), "tier": tier,
            "anchor_tau_pos": anchor["tau_pos"], "anchor_tau_content": anchor["tau_content"],
            "relayout_mean_pos": relay_pos, "relayout_mean_content": relay_cont,
            "relayout_drop": relay_pos - anchor["tau_pos"],
            "anchor_valid": anchor_valid, "verdict": verdict, "per_layout": per_layout}


# ── Task 9: Part-2 driver across steps ───────────────────────────────────────

def run_part2(seed, root, out_dir, K=8, steps=(0, 2000, 10000), bs_mean=16,
              n_batches=4, device="cpu"):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    w = _winner_for_seed(seed, root)
    L, heads, tier = w["winning_layer"], w["winner_heads"], w["tier"]

    training_perm = _training_clean_perm(f"{root}/seed{seed}/ckpt_step10000.pt")
    layouts = make_layouts(training_perm, K=K, seed_base=1000)
    save_layouts(layouts, str(out / "layouts.json"))

    by_step = {}
    for st in steps:
        by_step[str(st)] = binding_scores(seed, st, layouts, L, heads, tier,
                                          root=root, bs_mean=bs_mean,
                                          n_batches=n_batches, device=device)

    with open(out / "part2_binding.csv", "w", newline="") as f:
        wr = _csv.writer(f)
        wr.writerow(["step", "anchor_tau_pos", "anchor_tau_content",
                     "relayout_mean_pos", "relayout_mean_content", "relayout_drop",
                     "anchor_valid", "verdict"])
        for st in steps:
            b = by_step[str(st)]
            wr.writerow([st, round(b["anchor_tau_pos"], 4), round(b["anchor_tau_content"], 4),
                         round(b["relayout_mean_pos"], 4), round(b["relayout_mean_content"], 4),
                         round(b["relayout_drop"], 4), b["anchor_valid"], b["verdict"]])

    summary = {"seed": seed, "winning_layer": L, "carrier_heads": heads, "tier": tier,
               "by_step": by_step}
    _json.dump(summary, open(out / "part2.json", "w"), indent=2, default=float)
    return summary
