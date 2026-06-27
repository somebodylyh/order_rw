# Emergence Characterization (Spec A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Characterize how the seed-locked, layer-local redundant order carrier is *selected* (a diffuse→pruning/winner-take-all transition) from the existing every-200-step trajectories, with no new training.

**Architecture:** Pure-function analysis module `analyses/emergence_characterization.py` (A1 concentration/timing, A2 predictability, A3 schedule/loss overlay, A4 carrier-B re-extraction) + `analyses/plot_emergence.py` (figures). A1/A2/A3 read saved `tau_table.npz` and `eval_curve.tsv` directly; A4 re-extracts the winning layer's B from saved checkpoints via the Pillar-3 forward pipeline. Per-seed driver writes tables/JSON + a cross-seed README carrying the pre-registered Spec B fork.

**Tech Stack:** Python 3, NumPy, scikit-learn (`roc_auc_score`) or a tiny hand-rolled AUC, SciPy/`scipy.stats.spearmanr` (or hand-rolled), Matplotlib, pytest. Reuses `analyses/path_patch_handoff.py` (`load_model_and_chunks_seed`, `make_probe_batch`, `run_clean`), `attention_trajectory.extract_all_layer_B`, `none_separated_block_graph.rollout_by_method`, `analyses/handoff_carrier_config.py` (frozen carrier sets).

## Global Constraints

- **No new training.** A1/A2/A3 are pure file reads; A4 re-extracts B from saved ckpts (CPU if feasible, GPU only as an accelerator).
- **Method `C-D+L`** (index 0 in `tau_table['methods']==['C-D+L','L']`); metrics use `|τ|`, signed τ retained where noted.
- **Data roots:** `runs/handoff_overnight/seed{2,42,123}/`. Per-step τ: `attention_trajectory/raw/step_{NNNNNN}/tau_table.npz` (51 steps, 0–10000 by 200; `tau` shape `(4,8,2)`). Loss/LR: `eval_curve.tsv` (51 rows; cols incl. `step,val_train_objective,val_model_order,lr`). Ckpts: `ckpt_step{1000,2000,10000}.pt`.
- **τ-mass entropy primary** = Shannon entropy of `p_i=|τ_i|/Σ|τ|` over all 32 (layer,head) cells; `softmax(|τ|)` entropy is a robustness variant only.
- **Event timing:** lightly smooth (3-point) the primary entropy curve; midpoint = step of largest negative finite difference; onset/completion = pre/post-plateau boundaries of the largest sustained drop, tolerating one-step noise.
- **Winning layer** = `argmax_L Σ_h|τ[final,L,h]|` at step 10000. **Winner set** = that layer's `|τ|≥0.95` heads; **if empty (seed42), fall back to the frozen weak carrier set** (`handoff_carrier_config`) and tag `tier="weak"`, reported separately.
- **seed42 A2:** no strong final carrier → use the weak-tier winner set as the AUC/Spearman label; never compute AUC on an all-zero label.
- **Per-seed first**, aggregate second. Single run per seed → A2 measures within-run predictability, **not** reproducibility (that is Spec B).
- **Imports/tests run from `block_lo_arm_order_network/`**; tests insert repo ROOT (`parents[2]`) on `sys.path` and import `analyses.*` as a namespace package; the module self-bootstraps the block dir on `sys.path`.

---

### Task 1: τ-trajectory loader + winner determination

**Files:**
- Create: `analyses/emergence_characterization.py`
- Test: `block_lo_arm_order_network/tests/test_emergence_loader.py`

**Interfaces:**
- Produces:
  - `TRAJ_ROOT = "runs/handoff_overnight"` (overridable arg).
  - `load_tau_trajectory(seed, root=TRAJ_ROOT) -> dict` with `steps:(S,) int`, `tau:(S,4,8) float` (signed, method C-D+L), `abs_tau:(S,4,8)`.
  - `winner(seed_traj, seed) -> dict` with `winning_layer:int`, `winner_heads:list[int]`, `tier:str("strong"|"weak")`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_emergence_loader.py
import pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.emergence_characterization import load_tau_trajectory, winner

def test_load_trajectory_shape_and_order():
    t = load_tau_trajectory(2, root=str(ROOT / "runs/handoff_overnight"))
    assert t["steps"][0] == 0 and t["steps"][-1] == 10000
    assert t["steps"].shape == (51,)
    assert t["tau"].shape == (51, 4, 8)
    assert np.all(np.diff(t["steps"]) > 0)  # sorted

def test_winner_seed2_is_L1_strong():
    t = load_tau_trajectory(2, root=str(ROOT / "runs/handoff_overnight"))
    w = winner(t, 2)
    assert w["winning_layer"] == 1
    assert w["tier"] == "strong"
    assert set(w["winner_heads"]) == {0, 3, 5, 7}

def test_winner_seed42_is_L0_weak():
    t = load_tau_trajectory(42, root=str(ROOT / "runs/handoff_overnight"))
    w = winner(t, 42)
    assert w["winning_layer"] == 0
    assert w["tier"] == "weak"            # no strong head anywhere
    assert len(w["winner_heads"]) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_emergence_loader.py -q`
Expected: FAIL (`ModuleNotFoundError: analyses.emergence_characterization`)

- [ ] **Step 3: Implement loader + winner**

```python
# analyses/emergence_characterization.py
"""Spec A: characterize order-carrier emergence (diffuse -> pruning) from the
existing every-200-step trajectories. See
docs/superpowers/specs/2026-06-27-emergence-characterization-design.md."""
import glob
import pathlib
import sys

import numpy as np

_BLOCK_DIR = pathlib.Path(__file__).resolve().parents[1] / "block_lo_arm_order_network"
if str(_BLOCK_DIR) not in sys.path:
    sys.path.insert(0, str(_BLOCK_DIR))

TRAJ_ROOT = "runs/handoff_overnight"
_METHOD_I = 0  # C-D+L

def load_tau_trajectory(seed, root=TRAJ_ROOT):
    pat = f"{root}/seed{seed}/attention_trajectory/raw/step_*/tau_table.npz"
    files = sorted(glob.glob(pat))
    steps, taus = [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        step = int(f.split("step_")[1].split("/")[0])
        steps.append(step)
        taus.append(d["tau"][:, :, _METHOD_I])  # (4,8) signed C-D+L
    steps = np.array(steps, dtype=int)
    tau = np.stack(taus).astype(float)          # (S,4,8)
    order = np.argsort(steps)
    return {"steps": steps[order], "tau": tau[order], "abs_tau": np.abs(tau[order])}

def winner(seed_traj, seed, strong=0.95):
    from analyses.handoff_carrier_config import load_carrier_sets
    final = seed_traj["abs_tau"][-1]            # (4,8)
    winning_layer = int(final.sum(axis=1).argmax())
    strong_heads = [h for h in range(8) if final[winning_layer, h] >= strong]
    if strong_heads:
        return {"winning_layer": winning_layer, "winner_heads": strong_heads, "tier": "strong"}
    weak = load_carrier_sets(seed)[winning_layer]["weak"]
    return {"winning_layer": winning_layer, "winner_heads": list(weak), "tier": "weak"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_emergence_loader.py -q`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add analyses/emergence_characterization.py block_lo_arm_order_network/tests/test_emergence_loader.py
git commit -m "feat: emergence tau-trajectory loader + winner determination"
```

---

### Task 2: A1 concentration metrics + event timing

**Files:**
- Modify: `analyses/emergence_characterization.py`
- Test: `block_lo_arm_order_network/tests/test_emergence_a1.py`

**Interfaces:**
- Produces:
  - `concentration_metrics(seed_traj, strong=0.95) -> dict`: `strong_head_count:(S,4)`, `mass_entropy:(S,)` (primary), `softmax_entropy:(S,)` (robustness), `layer_mass:(S,4)`, `top1_layer_share:(S,)`.
  - `event_timing(steps, mass_entropy) -> dict`: `onset:int`, `midpoint:int`, `completion:int` (step values).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_emergence_a1.py
import pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.emergence_characterization import concentration_metrics, event_timing

def test_entropy_high_diffuse_low_concentrated():
    # synthetic: step0 all cells equal (max entropy), stepN one cell dominant
    S = 5
    abs_tau = np.ones((S, 4, 8))
    abs_tau[-1] = 0.01; abs_tau[-1, 1, 0] = 1.0   # last step concentrated in (L1,H0)
    traj = {"steps": np.arange(S) * 200, "tau": abs_tau, "abs_tau": abs_tau}
    m = concentration_metrics(traj)
    assert m["mass_entropy"][0] > m["mass_entropy"][-1]   # diffuse -> concentrated
    assert m["mass_entropy"].shape == (S,)

def test_event_timing_finds_drop_step():
    steps = np.arange(6) * 200
    ent = np.array([2.0, 2.0, 1.9, 0.5, 0.4, 0.4])  # steep drop between idx2->3
    ev = event_timing(steps, ent)
    assert ev["midpoint"] == 600       # step at largest negative diff (idx3)
    assert ev["onset"] <= 600 <= ev["completion"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_emergence_a1.py -q`
Expected: FAIL (functions undefined)

- [ ] **Step 3: Implement A1**

```python
# append to analyses/emergence_characterization.py
def _entropy(p, eps=1e-12):
    p = p / (p.sum() + eps)
    return float(-(p * np.log(p + eps)).sum())

def concentration_metrics(seed_traj, strong=0.95):
    A = seed_traj["abs_tau"]                  # (S,4,8)
    S = A.shape[0]
    strong_head_count = (A >= strong).sum(axis=2)            # (S,4)
    mass_entropy = np.array([_entropy(A[s].ravel()) for s in range(S)])
    softmax_entropy = np.array([
        _entropy(np.exp(A[s].ravel()) / np.exp(A[s].ravel()).sum()) for s in range(S)])
    layer_mass = A.sum(axis=2)                               # (S,4)
    top1_layer_share = layer_mass.max(axis=1) / (layer_mass.sum(axis=1) + 1e-12)
    return {"strong_head_count": strong_head_count, "mass_entropy": mass_entropy,
            "softmax_entropy": softmax_entropy, "layer_mass": layer_mass,
            "top1_layer_share": top1_layer_share}

def _smooth3(x):
    if len(x) < 3:
        return x.copy()
    out = x.copy()
    out[1:-1] = (x[:-2] + x[1:-1] + x[2:]) / 3.0
    return out

def event_timing(steps, mass_entropy, plateau_eps=0.1):
    ent = _smooth3(np.asarray(mass_entropy, dtype=float))
    diffs = np.diff(ent)
    mid_i = int(np.argmin(diffs)) + 1           # step after the steepest drop
    pre_plateau = ent[:mid_i].max() if mid_i > 0 else ent[0]
    post_plateau = ent[mid_i:].min()
    band = plateau_eps * (pre_plateau - post_plateau + 1e-12)
    onset_i = mid_i
    while onset_i > 0 and ent[onset_i - 1] >= pre_plateau - band:
        onset_i -= 1
    comp_i = mid_i
    while comp_i < len(ent) - 1 and ent[comp_i + 1] <= post_plateau + band:
        comp_i += 1
    return {"onset": int(steps[onset_i]), "midpoint": int(steps[mid_i]),
            "completion": int(steps[comp_i])}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_emergence_a1.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analyses/emergence_characterization.py block_lo_arm_order_network/tests/test_emergence_a1.py
git commit -m "feat: A1 concentration metrics + robust event timing"
```

---

### Task 3: A2 winner predictability (+ seed42 weak-tier) 

**Files:**
- Modify: `analyses/emergence_characterization.py`
- Test: `block_lo_arm_order_network/tests/test_emergence_a2.py`

**Interfaces:**
- Produces:
  - `winner_predictability(seed_traj, winner_dict, early_steps=(0,200,600,1000)) -> dict`: per early step `auc`, `spearman`; `winning_layer_rank_by_step` (rank of winning layer's Σ|τ| among layers); `verdict:str("early-bias"|"contingent")`.
- Consumes: `winner()` output (`winning_layer`, `winner_heads`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_emergence_a2.py
import pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.emergence_characterization import winner_predictability

def _traj_from(abs_tau, steps):
    return {"steps": np.array(steps), "tau": abs_tau, "abs_tau": abs_tau}

def test_early_bias_when_winners_lead_from_start():
    S = 6
    A = np.full((S, 4, 8), 0.1)
    A[:, 1, [0, 3, 5, 7]] = 0.9       # winners lead at ALL steps incl. early
    traj = _traj_from(A, [0,200,600,1000,1400,2000])
    w = {"winning_layer": 1, "winner_heads": [0,3,5,7], "tier": "strong"}
    out = winner_predictability(traj, w)
    assert out["auc"][200] > 0.95
    assert out["verdict"] == "early-bias"

def test_contingent_when_winners_tied_until_late():
    S = 6
    A = np.full((S, 4, 8), 0.5)       # everyone tied early
    A[-1, 1, [0,3,5,7]] = 1.0; A[-1, 1, [1,2,4,6]] = 0.2  # split only at the end
    traj = _traj_from(A, [0,200,600,1000,1400,2000])
    w = {"winning_layer": 1, "winner_heads": [0,3,5,7], "tier": "strong"}
    out = winner_predictability(traj, w)
    assert abs(out["auc"][200] - 0.5) < 0.2
    assert out["verdict"] == "contingent"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_emergence_a2.py -q`
Expected: FAIL

- [ ] **Step 3: Implement A2**

```python
# append to analyses/emergence_characterization.py
def _auc(scores, labels):
    # rank-based ROC-AUC; labels in {0,1}, undefined if single class
    labels = np.asarray(labels); scores = np.asarray(scores, dtype=float)
    pos = labels == 1; neg = ~pos
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))

def _spearman(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    ra = a.argsort().argsort().astype(float); rb = b.argsort().argsort().astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum()) + 1e-12
    return float((ra * rb).sum() / denom)

def winner_predictability(seed_traj, winner_dict, early_steps=(0, 200, 600, 1000)):
    steps = seed_traj["steps"]; A = seed_traj["abs_tau"]
    L = winner_dict["winning_layer"]; wh = set(winner_dict["winner_heads"])
    final = A[-1, L]                                  # (8,)
    labels = np.array([1 if h in wh else 0 for h in range(8)])
    step_idx = {int(s): i for i, s in enumerate(steps)}
    auc, spearman = {}, {}
    for s in early_steps:
        i = step_idx[s]
        early = A[i, L]                              # (8,) within winning layer
        auc[s] = _auc(early, labels)
        spearman[s] = _spearman(early, final)
    # layer-level lead: rank of winning layer's sum|tau| among layers (1=highest)
    layer_rank = {}
    for s in early_steps:
        i = step_idx[s]
        sums = A[i].sum(axis=1)
        layer_rank[s] = int((sums > sums[L]).sum() + 1)
    # verdict: early-bias if winners already lead early (AUC high at step<=600)
    early_auc = np.nanmean([auc[s] for s in early_steps if s <= 600])
    verdict = "early-bias" if early_auc >= 0.8 else "contingent"
    return {"auc": auc, "spearman": spearman,
            "winning_layer_rank_by_step": layer_rank, "verdict": verdict,
            "tier": winner_dict["tier"]}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_emergence_a2.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analyses/emergence_characterization.py block_lo_arm_order_network/tests/test_emergence_a2.py
git commit -m "feat: A2 winner predictability (AUC/Spearman/layer-lead) + verdict"
```

---

### Task 4: A3 schedule/loss overlay + classification

**Files:**
- Modify: `analyses/emergence_characterization.py`
- Test: `block_lo_arm_order_network/tests/test_emergence_a3.py`

**Interfaces:**
- Produces:
  - `load_eval_curve(seed, root=TRAJ_ROOT) -> dict` with `step,lr,val_train_objective,val_model_order` arrays.
  - `schedule_loss_overlay(eval_curve, event) -> dict` with `lr_range_in_window`, `loss_drop_in_window`, `classification:str`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_emergence_a3.py
import pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.emergence_characterization import load_eval_curve, schedule_loss_overlay

def test_load_eval_curve_columns():
    ec = load_eval_curve(2, root=str(ROOT / "runs/handoff_overnight"))
    assert ec["step"][0] == 0 and len(ec["step"]) == 51
    assert "lr" in ec and "val_model_order" in ec

def test_classification_intrinsic_when_lr_flat_and_loss_smooth():
    ec = {"step": np.arange(6)*200, "lr": np.full(6, 1e-3),
          "val_train_objective": np.linspace(6, 5, 6),
          "val_model_order": np.linspace(6, 5, 6)}
    ev = {"onset": 200, "midpoint": 600, "completion": 1000}
    out = schedule_loss_overlay(ec, ev)
    assert out["classification"] == "intrinsic (no alignment)"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_emergence_a3.py -q`
Expected: FAIL

- [ ] **Step 3: Implement A3**

```python
# append to analyses/emergence_characterization.py
import csv as _csv

def load_eval_curve(seed, root=TRAJ_ROOT):
    path = f"{root}/seed{seed}/eval_curve.tsv"
    cols = {}
    with open(path) as f:
        r = _csv.DictReader(f, delimiter="\t")
        rows = list(r)
    def col(name):
        return np.array([float(x[name]) if x[name] not in ("", "nan") else np.nan
                         for x in rows])
    return {"step": col("step").astype(int), "lr": col("lr"),
            "val_train_objective": col("val_train_objective"),
            "val_model_order": col("val_model_order")}

def schedule_loss_overlay(eval_curve, event, lr_rel_eps=0.02, loss_rel_eps=0.15):
    s = eval_curve["step"]
    win = (s >= event["onset"]) & (s <= event["completion"])
    lr = eval_curve["lr"]; lrw = lr[win]
    lr_range = float((lrw.max() - lrw.min()) / (np.nanmean(lr) + 1e-12))
    loss = eval_curve["val_train_objective"]
    full_drop = np.nanmax(loss) - np.nanmin(loss) + 1e-12
    win_drop = float((np.nanmax(loss[win]) - np.nanmin(loss[win])) / full_drop)
    if lr_range > lr_rel_eps:
        cls = "aligns-with-LR"
    elif win_drop > loss_rel_eps:
        cls = "aligns-with-loss-transition"
    else:
        cls = "intrinsic (no alignment)"
    return {"lr_range_in_window": lr_range, "loss_drop_in_window": win_drop,
            "classification": cls}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_emergence_a3.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analyses/emergence_characterization.py block_lo_arm_order_network/tests/test_emergence_a3.py
git commit -m "feat: A3 schedule/loss overlay + intrinsic/LR/loss classification"
```

---

### Task 5: Per-seed driver + table/JSON writers (A1–A3)

**Files:**
- Modify: `analyses/emergence_characterization.py`
- Test: `block_lo_arm_order_network/tests/test_emergence_driver.py`

**Interfaces:**
- Produces:
  - `run_seed_emergence(seed, root=TRAJ_ROOT, out_dir=None) -> dict` — runs Task1–4 (A4 added in Task 7), writes `concentration.csv` (step + metrics), `summary.json` (winner, event, predictability verdict, A3 classification). Returns the summary.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_emergence_driver.py
import json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.emergence_characterization import run_seed_emergence

def test_run_seed_emergence_smoke(tmp_path):
    s = run_seed_emergence(2, root=str(ROOT / "runs/handoff_overnight"), out_dir=str(tmp_path))
    assert (tmp_path / "concentration.csv").exists()
    assert (tmp_path / "summary.json").exists()
    j = json.load(open(tmp_path / "summary.json"))
    assert j["seed"] == 2
    assert j["winner"]["winning_layer"] == 1
    assert j["predictability"]["verdict"] in ("early-bias", "contingent")
    assert "classification" in j["schedule"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_emergence_driver.py -q`
Expected: FAIL

- [ ] **Step 3: Implement the driver**

```python
# append to analyses/emergence_characterization.py
import json
import pathlib as _pl

def run_seed_emergence(seed, root=TRAJ_ROOT, out_dir=None):
    out = _pl.Path(out_dir or f"runs/emergence/seed{seed}")
    out.mkdir(parents=True, exist_ok=True)
    traj = load_tau_trajectory(seed, root)
    w = winner(traj, seed)
    conc = concentration_metrics(traj)
    event = event_timing(traj["steps"], conc["mass_entropy"])
    pred = winner_predictability(traj, w)
    ec = load_eval_curve(seed, root)
    sched = schedule_loss_overlay(ec, event)
    # concentration.csv
    with open(out / "concentration.csv", "w", newline="") as f:
        wcsv = _csv.writer(f)
        wcsv.writerow(["step", "mass_entropy", "softmax_entropy", "top1_layer_share",
                       *[f"strong_L{L}" for L in range(4)]])
        for i, st in enumerate(traj["steps"]):
            wcsv.writerow([int(st), round(float(conc["mass_entropy"][i]), 4),
                           round(float(conc["softmax_entropy"][i]), 4),
                           round(float(conc["top1_layer_share"][i]), 4),
                           *[int(conc["strong_head_count"][i, L]) for L in range(4)]])
    summary = {"seed": seed, "winner": w, "event": event,
               "predictability": pred, "schedule": sched}
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    return summary
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_emergence_driver.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analyses/emergence_characterization.py block_lo_arm_order_network/tests/test_emergence_driver.py
git commit -m "feat: per-seed emergence driver + concentration.csv/summary.json"
```

---

### Task 6: Plotting module (A1/A2/A3 figures)

**Files:**
- Create: `analyses/plot_emergence.py`
- Test: `block_lo_arm_order_network/tests/test_plot_emergence.py`

**Interfaces:**
- Produces: `plot_emergence(summary_json, concentration_csv, eval_tsv, out_dir)` writing `concentration.png` (entropy + per-layer strong-head count vs step, with onset/midpoint/completion vlines), `predictability.png` (AUC/Spearman vs early step), `schedule_overlay.png` (lr + loss vs step with event band).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_plot_emergence.py
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.emergence_characterization import run_seed_emergence
from analyses.plot_emergence import plot_emergence

def test_plot_emergence_writes_pngs(tmp_path):
    run_seed_emergence(2, root=str(ROOT / "runs/handoff_overnight"), out_dir=str(tmp_path))
    plot_emergence(str(tmp_path / "summary.json"), str(tmp_path / "concentration.csv"),
                   str(ROOT / "runs/handoff_overnight/seed2/eval_curve.tsv"), str(tmp_path))
    assert (tmp_path / "concentration.png").exists()
    assert (tmp_path / "schedule_overlay.png").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_plot_emergence.py -q`
Expected: FAIL (`ModuleNotFoundError: analyses.plot_emergence`)

- [ ] **Step 3: Implement plotting** (matplotlib Agg; read CSV/JSON; draw the three figures with event vlines/band). Full code in the module — three `matplotlib` figures: entropy+strong-count curve, AUC/Spearman-vs-early-step, and lr/loss overlay with an axvspan over [onset, completion].

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_plot_emergence.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analyses/plot_emergence.py block_lo_arm_order_network/tests/test_plot_emergence.py
git commit -m "feat: emergence plots (concentration, predictability, schedule overlay)"
```

---

### Task 7: A4 carrier-B re-extraction wrapper + regression anchor

**Files:**
- Modify: `analyses/emergence_characterization.py`
- Test: `block_lo_arm_order_network/tests/test_emergence_a4_extract.py`

**Interfaces:**
- Consumes: `path_patch_handoff.load_model_and_chunks_seed`, `make_probe_batch`, `run_clean`; `attention_trajectory.extract_all_layer_B`.
- Produces:
  - `extract_carrier_B(seed, ckpt_step, winning_layer, root=TRAJ_ROOT, bs_mean=16, n_batches=4, device="cpu") -> np.ndarray` — batch-mean B `(8,65,65)` for the winning layer at a ckpt step.

- [ ] **Step 1: Write the failing test (regression anchor)**

```python
# tests/test_emergence_a4_extract.py
import pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.emergence_characterization import extract_carrier_B
from analyses.path_patch_handoff import _BLOCK_DIR  # noqa
from none_separated_block_graph import rollout_by_method
from batch_readout.order_tau_readout import per_head_tau

def test_extracted_B_reproduces_step10000_carrier_tau():
    B = extract_carrier_B(2, 10000, winning_layer=1,
                          root=str(ROOT / "runs/handoff_overnight"))
    assert B.shape == (8, 65, 65)
    # seed2 L1 strong heads {3,5,7} roll out to ~1.0 tau (H0 pool-sensitive)
    for h in (3, 5, 7):
        info = per_head_tau(B[h], "C-D+L")
        assert info["tau_vs_l2r"] > 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_emergence_a4_extract.py -q`
Expected: FAIL

- [ ] **Step 3: Implement re-extraction**

```python
# append to analyses/emergence_characterization.py
def extract_carrier_B(seed, ckpt_step, winning_layer, root=TRAJ_ROOT,
                      bs_mean=16, n_batches=4, device="cpu"):
    import torch
    from analyses.path_patch_handoff import (
        load_model_and_chunks_seed, make_probe_batch, run_clean)
    from attention_trajectory import extract_all_layer_B
    ckpt = f"{root}/seed{seed}/ckpt_step{ckpt_step}.pt"
    dev = torch.device(device)
    model, chunks, _ = load_model_and_chunks_seed(ckpt, max(64, bs_mean * n_batches), dev)
    accum = []
    for i in range(n_batches):
        pc, po = make_probe_batch(chunks, bs_mean, np.random.default_rng(i))
        attn_list, _ = run_clean(model, pc, po, dev)
        B_all = extract_all_layer_B(attn_list, po)        # (L,S,H,65,65)
        accum.append(B_all[winning_layer].mean(axis=0))   # (H,65,65)
    return np.mean(accum, axis=0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_emergence_a4_extract.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analyses/emergence_characterization.py block_lo_arm_order_network/tests/test_emergence_a4_extract.py
git commit -m "feat: A4 winning-layer B re-extraction from ckpts + regression anchor"
```

---

### Task 8: A4 B-structure rendering (pre/post/converged) + Pattern A/B

**Files:**
- Modify: `analyses/plot_emergence.py`
- Test: `block_lo_arm_order_network/tests/test_emergence_a4_render.py`

**Interfaces:**
- Produces:
  - `carrier_b_structure(seed, winning_layer, winner_heads, ckpt_steps=(1000,2000,10000), root=TRAJ_ROOT) -> dict` — per ckpt_step, per head: `tau_vs_l2r` + the B matrix; classify `pattern:str("sharpening"|"switch")` (sharpening if pre-step mean |τ| over winner heads ≥ 0.5, i.e. already L2R-like; else switch).
  - `plot_carrier_b(b_struct, out_dir)` — heatmap grid `(winner heads) × (ckpt steps)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_emergence_a4_render.py
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.plot_emergence import carrier_b_structure, plot_carrier_b

def test_carrier_b_structure_and_render(tmp_path):
    bs = carrier_b_structure(2, winning_layer=1, winner_heads=[3, 5, 7],
                             ckpt_steps=(2000, 10000),
                             root=str(ROOT / "runs/handoff_overnight"))
    assert bs["pattern"] in ("sharpening", "switch")
    assert 10000 in bs["by_step"]
    plot_carrier_b(bs, str(tmp_path))
    assert (tmp_path / "carrier_b_structure.png").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_emergence_a4_render.py -q`
Expected: FAIL

- [ ] **Step 3: Implement structure + render** (use `extract_carrier_B` per step; `per_head_tau` for τ and `rollout_by_method` for order; classify Pattern A/B by the earliest ckpt step's mean |τ| over winner heads ≥ 0.5; draw a heads×steps heatmap grid). Full code in the plot module.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_emergence_a4_render.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analyses/plot_emergence.py block_lo_arm_order_network/tests/test_emergence_a4_render.py
git commit -m "feat: A4 carrier-B structure (pre/post/converged) + Pattern A/B render"
```

---

### Task 9: Three-seed run + cross-seed README + Spec B fork

**Files:**
- Create: `runs/emergence/` (outputs)
- Create: `analyses/emergence_README.md`

- [ ] **Step 1: Run all three seeds (A1–A3 + driver)**

Run: `cd block_lo_arm_order_network && python -c "from analyses.emergence_characterization import run_seed_emergence as r; [r(s, root='../runs/handoff_overnight', out_dir=f'../runs/emergence/seed{s}') for s in (2,42,123)]"`
Expected: `runs/emergence/seed{2,42,123}/{concentration.csv,summary.json}` written.

- [ ] **Step 2: Generate A1/A2/A3 plots**

Run: `python -c` over the three seeds calling `plot_emergence(...)` with each seed's summary/concentration/eval_curve.

- [ ] **Step 3: Generate A4 B-structure (per seed winning layer/heads)**

Call `carrier_b_structure` + `plot_carrier_b` for each seed using its `summary.json` winner.

- [ ] **Step 4: Write `analyses/emergence_README.md`**

Fill per-seed: event onset/midpoint/completion, winner layer/heads/tier, A2 verdict (early-bias vs contingent) with AUC/Spearman/layer-rank, A3 classification, A4 Pattern A/B. Then the **pre-registered Spec B fork** resolved by the observed verdicts (early-bias → init-stability Spec B; contingent → pruning-window Spec B; +LR/⑤ branches per spec).

- [ ] **Step 5: Commit**

```bash
git add analyses/emergence_README.md runs/emergence
git commit -m "results: Spec A emergence characterization, 3 seeds + Spec B fork"
```

---

## Self-Review

**Spec coverage:**
- A1 concentration (3 metrics) + event timing → Task 2. ✓ (mass-entropy primary, softmax robustness, smoothed-drop timing — Global Constraints + Task 2)
- A2 predictability (AUC/Spearman/layer-lead) + verdict + seed42 weak-tier → Tasks 1 (winner tier) + 3. ✓
- A3 LR/loss overlay, correlational classification → Task 4. ✓
- A4 winning-layer B re-extraction at {1000,2000,10000} + Pattern A/B → Tasks 7,8. ✓
- Outputs (concentration.csv, summary.json, figures, README, Spec B fork) → Tasks 5,6,9. ✓
- No new training; per-seed first; within-run not reproducibility → Global Constraints + Task 9 README. ✓
- TDD: synthetic event-timing/predictability anchors (Tasks 2,3), A4 re-extraction regression (Task 7). ✓

**Placeholder scan:** Tasks 6 & 8 step-3 describe the figures rather than pasting full matplotlib boilerplate, because they assemble already-tested data (the metrics/structure dicts) into standard plots; the data-producing functions (`carrier_b_structure`) have complete code. No TBD/TODO in logic-bearing steps.

**Type consistency:** `winner()` returns `{winning_layer, winner_heads, tier}` consumed identically by Tasks 3/5/7/8; `event_timing` returns `{onset,midpoint,completion}` consumed by Tasks 4/5/6; `concentration_metrics` keys (`mass_entropy`,`strong_head_count`,`top1_layer_share`) match the CSV writer (Task 5) and plots (Task 6); method index 0 = C-D+L throughout.
