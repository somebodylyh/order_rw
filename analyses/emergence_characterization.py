"""Spec A: characterize order-carrier emergence (diffuse -> pruning) from the
existing every-200-step trajectories.

See docs/superpowers/specs/2026-06-27-emergence-characterization-design.md
and docs/superpowers/plans/2026-06-27-emergence-characterization.md.

After Pillar 3 falsified the cross-layer handoff, the mechanism question becomes:
how is the carrier layer/head set *selected*? The order signal first appears
diffusely (all layers/heads, step 0-1200), then a short winner-take-all pruning
event (~1200-2000) leaves a seed-specific layer-local redundant carrier.
"""
import csv as _csv
import glob
import pathlib
import sys

import numpy as np

# Make block_lo_arm_order_network modules importable regardless of CWD.
_BLOCK_DIR = pathlib.Path(__file__).resolve().parents[1] / "block_lo_arm_order_network"
if str(_BLOCK_DIR) not in sys.path:
    sys.path.insert(0, str(_BLOCK_DIR))

TRAJ_ROOT = "runs/handoff_overnight"
_METHOD_I = 0  # C-D+L


# ── Task 1: tau-trajectory loader + winner determination ─────────────────────

def load_tau_trajectory(seed, root=TRAJ_ROOT):
    """Stack the 51 per-step tau_table.npz into steps + signed tau[S,4,8] (C-D+L)."""
    pat = f"{root}/seed{seed}/attention_trajectory/raw/step_*/tau_table.npz"
    files = sorted(glob.glob(pat))
    steps, taus = [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        step = int(f.split("step_")[1].split("/")[0])
        steps.append(step)
        taus.append(d["tau"][:, :, _METHOD_I])  # (4,8) signed C-D+L
    steps = np.array(steps, dtype=int)
    tau = np.stack(taus).astype(float)           # (S,4,8)
    order = np.argsort(steps)
    return {"steps": steps[order], "tau": tau[order], "abs_tau": np.abs(tau[order])}


def winner(seed_traj, seed, strong=0.95):
    """Winning layer = argmax_L sum_h|tau| at the final step. Winner set = that
    layer's |tau|>=strong heads, or its frozen weak carrier set if none (seed42)."""
    from analyses.handoff_carrier_config import load_carrier_sets
    final = seed_traj["abs_tau"][-1]             # (4,8)
    winning_layer = int(final.sum(axis=1).argmax())
    strong_heads = [h for h in range(8) if final[winning_layer, h] >= strong]
    if strong_heads:
        return {"winning_layer": winning_layer, "winner_heads": strong_heads,
                "tier": "strong"}
    weak = load_carrier_sets(seed)[winning_layer]["weak"]
    return {"winning_layer": winning_layer, "winner_heads": list(weak), "tier": "weak"}


# ── Task 2: A1 concentration metrics + event timing ──────────────────────────

def _entropy(p, eps=1e-12):
    p = p / (p.sum() + eps)
    return float(-(p * np.log(p + eps)).sum())


def concentration_metrics(seed_traj, strong=0.95, diffuse_tau=0.7):
    """Order-signal concentration over training.

    Empirically the winner-take-all event is sharpest in the *diffuse_count* — the
    number of (layer,head) cells with |tau|>=diffuse_tau (0.7) — which collapses
    32 -> ~10 (matching the Pillar-1/2 strong_pass story). |tau|-mass entropy and
    layer-mass entropy barely move (pruned heads keep moderate |tau|, mass stays
    near-uniform), so they are reported as secondary/robustness signals and the
    *event* is timed on diffuse_count, not entropy.
    """
    A = seed_traj["abs_tau"]                                   # (S,4,8)
    S = A.shape[0]
    strong_head_count = (A >= strong).sum(axis=2)             # (S,4)  |tau|>=0.95
    diffuse_count = (A >= diffuse_tau).sum(axis=(1, 2))       # (S,)   |tau|>=0.7 model-wide
    mass_entropy = np.array([_entropy(A[s].ravel()) for s in range(S)])
    softmax_entropy = np.array([
        _entropy(np.exp(A[s].ravel()) / np.exp(A[s].ravel()).sum()) for s in range(S)])
    layer_mass = A.sum(axis=2)                                # (S,4)
    top1_layer_share = layer_mass.max(axis=1) / (layer_mass.sum(axis=1) + 1e-12)
    return {"strong_head_count": strong_head_count, "diffuse_count": diffuse_count,
            "mass_entropy": mass_entropy, "softmax_entropy": softmax_entropy,
            "layer_mass": layer_mass, "top1_layer_share": top1_layer_share}


def _smooth3(x):
    if len(x) < 3:
        return x.copy()
    out = x.copy()
    out[1:-1] = (x[:-2] + x[1:-1] + x[2:]) / 3.0
    return out


def event_timing(steps, mass_entropy, plateau_eps=0.1):
    """onset/midpoint/completion of the winner-take-all entropy drop, robust to
    one-step noise (smoothed curve; midpoint = steepest drop; onset/completion =
    pre/post plateau boundaries)."""
    steps = np.asarray(steps)
    ent = _smooth3(np.asarray(mass_entropy, dtype=float))
    diffs = np.diff(ent)
    mid_i = int(np.argmin(diffs)) + 1            # step after the steepest drop
    # Local plateaus around the event (robust to late wander / dips in the tail).
    pre_plateau = float(np.median(ent[max(0, mid_i - 4):mid_i])) if mid_i > 0 else ent[0]
    post_plateau = float(np.median(ent[mid_i:mid_i + 4]))
    band = plateau_eps * (pre_plateau - post_plateau + 1e-12)
    # Scan-based plateau crossings (tolerate one-step noise, no contiguous walk):
    # onset = last step <= midpoint still at the pre-drop plateau;
    # completion = first step >= midpoint at the post-drop plateau.
    pre_idx = [i for i in range(mid_i + 1) if ent[i] >= pre_plateau - band]
    post_idx = [i for i in range(mid_i, len(ent)) if ent[i] <= post_plateau + band]
    onset_i = max(pre_idx) if pre_idx else mid_i
    comp_i = min(post_idx) if post_idx else mid_i
    return {"onset": int(steps[onset_i]), "midpoint": int(steps[mid_i]),
            "completion": int(steps[comp_i])}


# ── Task 3: A2 winner predictability ─────────────────────────────────────────

def _auc(scores, labels):
    """Rank-based ROC-AUC; nan if single-class (e.g. all-zero label)."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    pos = labels == 1
    neg = ~pos
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2)
                 / (pos.sum() * neg.sum()))


def _spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ra = a.argsort().argsort().astype(float)
    rb = b.argsort().argsort().astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum()) + 1e-12
    return float((ra * rb).sum() / denom)


def winner_predictability(seed_traj, winner_dict, early_steps=(0, 200, 600, 1000)):
    """Does early |tau| rank predict final carrier membership? Verdict = early-bias
    (winners lead early) vs contingent (tied until the pruning window)."""
    steps = seed_traj["steps"]
    A = seed_traj["abs_tau"]
    L = winner_dict["winning_layer"]
    wh = set(winner_dict["winner_heads"])
    final = A[-1, L]                                  # (8,)
    labels = np.array([1 if h in wh else 0 for h in range(8)])
    step_idx = {int(s): i for i, s in enumerate(steps)}
    auc, spearman = {}, {}
    for s in early_steps:
        i = step_idx[s]
        early = A[i, L]                              # (8,) within winning layer
        auc[s] = _auc(early, labels)
        spearman[s] = _spearman(early, final)
    layer_rank = {}
    for s in early_steps:
        i = step_idx[s]
        sums = A[i].sum(axis=1)
        layer_rank[s] = int((sums > sums[L]).sum() + 1)   # 1 = highest
    early_auc = np.nanmean([auc[s] for s in early_steps if s <= 600])
    verdict = "early-bias" if early_auc >= 0.8 else "contingent"
    return {"auc": auc, "spearman": spearman,
            "winning_layer_rank_by_step": layer_rank, "verdict": verdict,
            "tier": winner_dict["tier"]}


# ── Task 4: A3 schedule/loss overlay ─────────────────────────────────────────

def load_eval_curve(seed, root=TRAJ_ROOT):
    path = f"{root}/seed{seed}/eval_curve.tsv"
    with open(path) as f:
        rows = list(_csv.DictReader(f, delimiter="\t"))

    def col(name):
        return np.array([float(x[name]) if x[name] not in ("", "nan") else np.nan
                         for x in rows])

    return {"step": col("step").astype(int), "lr": col("lr"),
            "val_train_objective": col("val_train_objective"),
            "val_model_order": col("val_model_order")}


def schedule_loss_overlay(eval_curve, event, lr_rel_eps=0.02, loss_rel_eps=0.15):
    """Overlay pruning window against LR + loss. Correlational only: classify as
    aligns-with-LR / aligns-with-loss-transition / intrinsic (no alignment)."""
    s = eval_curve["step"]
    win = (s >= event["onset"]) & (s <= event["completion"])
    lr = eval_curve["lr"]
    lrw = lr[win]
    lr_range = float((lrw.max() - lrw.min()) / (np.nanmean(lr) + 1e-12))
    loss = eval_curve["val_train_objective"]
    full_drop = np.nanmax(loss) - np.nanmin(loss) + 1e-12
    win_drop_frac = float((np.nanmax(loss[win]) - np.nanmin(loss[win])) / full_drop)
    # Excess loss-drop rate: window's share of the total drop vs its share of steps.
    # A smooth (linear) decline gives ~1; a transition concentrated in the window
    # gives >> 1. This avoids flagging a wide window that simply covers most of
    # the curve.
    step_frac = float(win.sum()) / float(len(s))
    excess = win_drop_frac / (step_frac + 1e-12)
    if lr_range > lr_rel_eps:
        cls = "aligns-with-LR"
    elif excess > 1.0 + loss_rel_eps:
        cls = "aligns-with-loss-transition"
    else:
        cls = "intrinsic (no alignment)"
    return {"lr_range_in_window": lr_range, "loss_drop_frac_in_window": win_drop_frac,
            "loss_drop_excess_rate": excess, "classification": cls}


# ── Task 5: per-seed driver + table/JSON writers ─────────────────────────────

def run_seed_emergence(seed, root=TRAJ_ROOT, out_dir=None):
    """Run A1-A3 for one seed; write concentration.csv + summary.json; return summary."""
    import json

    out = pathlib.Path(out_dir or f"runs/emergence/seed{seed}")
    out.mkdir(parents=True, exist_ok=True)
    traj = load_tau_trajectory(seed, root)
    w = winner(traj, seed)
    conc = concentration_metrics(traj)
    # Time the event on the diffuse_count collapse (mass entropy is too flat).
    event = event_timing(traj["steps"], conc["diffuse_count"])
    pred = winner_predictability(traj, w)
    ec = load_eval_curve(seed, root)
    sched = schedule_loss_overlay(ec, event)

    with open(out / "concentration.csv", "w", newline="") as f:
        wcsv = _csv.writer(f)
        wcsv.writerow(["step", "diffuse_count", "mass_entropy", "softmax_entropy",
                       "top1_layer_share", *[f"strong_L{L}" for L in range(4)]])
        for i, st in enumerate(traj["steps"]):
            wcsv.writerow([int(st), int(conc["diffuse_count"][i]),
                           round(float(conc["mass_entropy"][i]), 4),
                           round(float(conc["softmax_entropy"][i]), 4),
                           round(float(conc["top1_layer_share"][i]), 4),
                           *[int(conc["strong_head_count"][i, L]) for L in range(4)]])

    summary = {"seed": seed, "winner": w, "event": event,
               "predictability": pred, "schedule": sched}
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    return summary
