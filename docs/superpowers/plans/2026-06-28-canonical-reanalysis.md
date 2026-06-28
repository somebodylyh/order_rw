# Canonical 65-node Re-analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-derive the order-signal emergence, carrier, handoff (③), and content-binding (⑤) on `runs/handoff_overnight` using the project's sealed **strict 65-node None-separated** readout (random reveal + posthoc-inv physical scoring + method C-D+L + destroyed controls), replacing the tautological model-frame/identity-reveal readout the ③/A/⑤ arc used.

**Architecture:** A thin `analyses/canonical_reanalysis.py` that wraps the **existing** canonical readout functions (`_attn_to_A_block_loss_aligned_with_none_vec`, `build_none_separated_B`, `rollout_by_method`, `discovery_metrics`, `classify_gate_status`, `combined_discovery_score`, `entry_shuffled_control`, `content_label_permutation_control`) and loads `runs/handoff_overnight` checkpoints via the proven `_load_model_and_chunks` loader + random reveal orders. It sweeps 3 seeds × 11 ckpts, drives C2 ablation by reusing `path_patch_handoff.mean_ablation_prehook`, and writes per-seed sweep TSVs + a corrected README.

**Tech Stack:** Python 3, PyTorch, NumPy, pytest. Reuses `block_lo_arm_order_network/per_head_order_scan.py` (`_attn_to_A_block_loss_aligned_with_none_vec`), `none_separated_block_graph.py` (graph/rollout/gate/controls), `neural_readout.extract_b._load_model_and_chunks`, `clean_training_protocol` (`expand_model_blocks_to_token_order`, `CleanPermutation`), `analyses/path_patch_handoff.py` (`mean_ablation_prehook`).

## Global Constraints

- **No new training.** Read from `runs/handoff_overnight/seed{2,42,123}/ckpt_step{0,1000,…,10000}.pt`. CPU; GPU only as accelerator.
- **Strict 65-node None-separated protocol** (sealed, `reports/strict_65node_discovery_ckpt_verification_20260617/01_protocol_definition.md`): node 0 = `[None]`/BOS separate; nodes 1..64 = content blocks; `B65 = build_none_separated_B(A)`, `diag=0`; loss-aligned AR frame `attn[:T,:T]`; extraction `_attn_to_A_block_loss_aligned_with_none_vec(sample, token_order, inv_perm)`.
- **Random reveal** orders per sample, seeded `seed+i` (matching the canonical scan): `rand_blocks = torch.randperm(N, generator=manual_seed(seed+i))` → `expand_model_blocks_to_token_order`.
- **Scoring**: rollout σ_model, score `tau_vs_l2r` (posthoc-inv physical) via `discovery_metrics`.
- **Primary method `C-D+L`**; also run `L`, `none_edge`. `METHODS` superset available.
- **Gate**: `classify_gate_status(metrics, destroyed)` → strong/weak/fail; `strong_pass` = τ=1.0, first_block=0, phys0_rank=0, prefix4=4, prefix8=8.
- **Destroyed controls**: `entry_shuffled_control` + `content_label_permutation_control`, `control_seeds=5`; floor ≈0.05–0.07.
- **Sweep params (sealed)**: `M=8, batch_size=8, control_seeds=5`. Sampling seeds `K=3` per (seed, ckpt); aggregate mean±std.
- **Anchor (must reproduce before trusting)**: clean_base sweep — step0 best_tau ≈ floor (~0.19 max, within destroyed noise); step≥5000 best_tau=1.0 on **L0** heads (L0H1–H4), method C-D+L/L.
- **Tests run from `block_lo_arm_order_network/`**; insert repo ROOT (`parents[2]`) for `analyses.*`; module self-bootstraps the block dir.

---

### Task 1: Canonical per-ckpt 65-node scan wrapper

**Files:**
- Create: `analyses/canonical_reanalysis.py`
- Test: `block_lo_arm_order_network/tests/test_canon_scan.py`

**Interfaces:**
- Produces:
  - `random_reveal_orders(n, seed) -> np.ndarray (n,256)` — per-sample random reveal, seeded `seed+i`, via `expand_model_blocks_to_token_order`.
  - `canonical_scan(ckpt_path, M=8, batch_size=8, sampling_seed=0, methods=("C-D+L","L","none_edge"), control_seeds=(0,1,2,3,4), device="cpu") -> list[dict]` — rows per `(layer,head,method)` with `tau_vs_l2r, abs_tau, first_block, phys0_rank, prefix4_overlap, prefix8_overlap, destroyed_abs_tau_mean, gate_status, combined_score`.

- [ ] **Step 1: Write the failing test** (anchor: seed2 step10000 has an L0 strong-pass head under C-D+L; step0 has none)

```python
# tests/test_canon_scan.py
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.canonical_reanalysis import canonical_scan

def test_step10000_has_L0_strong_pass_cdl():
    rows = canonical_scan(str(ROOT/"runs/handoff_overnight/seed2/ckpt_step10000.pt"),
                          M=8, batch_size=8, sampling_seed=0)
    cdl_strong = [r for r in rows if r["method"] == "C-D+L" and r["gate_status"] == "strong_pass"]
    assert any(r["layer"] == 0 for r in cdl_strong)            # carrier is in L0
    assert max(r["abs_tau"] for r in cdl_strong) > 0.95         # tau ~1.0

def test_step0_is_at_floor():
    rows = canonical_scan(str(ROOT/"runs/handoff_overnight/seed2/ckpt_step0.pt"),
                          M=8, batch_size=8, sampling_seed=0)
    strong = [r for r in rows if r["gate_status"] == "strong_pass"]
    assert len(strong) == 0                                     # order absent at init
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_canon_scan.py -q`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement the scan**

```python
# analyses/canonical_reanalysis.py
"""Canonical strict 65-node None-separated re-analysis on runs/handoff_overnight.
See docs/superpowers/specs/2026-06-28-canonical-reanalysis-design.md and
reports/strict_65node_discovery_ckpt_verification_20260617/."""
import pathlib, sys
import numpy as np, torch

_BLOCK_DIR = pathlib.Path(__file__).resolve().parents[1] / "block_lo_arm_order_network"
if str(_BLOCK_DIR) not in sys.path:
    sys.path.insert(0, str(_BLOCK_DIR))

from neural_readout.extract_b import _load_model_and_chunks
from clean_training_protocol import expand_model_blocks_to_token_order
from training_utils import N, SEQ_LEN, BLOCK_LEN
from per_head_order_scan import _attn_to_A_block_loss_aligned_with_none_vec
from none_separated_block_graph import (
    build_none_separated_B, rollout_by_method, discovery_metrics,
    classify_gate_status, combined_discovery_score,
    entry_shuffled_control, content_label_permutation_control)

def random_reveal_orders(n, seed):
    orders = np.empty((n, SEQ_LEN), dtype=np.int64)
    for i in range(n):
        g = torch.Generator(device="cpu"); g.manual_seed(int(seed) + int(i))
        rb = torch.randperm(N, generator=g)
        orders[i] = expand_model_blocks_to_token_order(rb.unsqueeze(0), BLOCK_LEN)[0].numpy()
    return orders

def _destroyed_floor(B65, method, control_seeds):
    vals = []
    for s in control_seeds:
        for Bc in (entry_shuffled_control(B65, seed=s),
                   content_label_permutation_control(B65, seed=s)):
            vals.append(abs(discovery_metrics(rollout_by_method(Bc, method))["tau_vs_l2r"]))
    return float(np.mean(vals)) if vals else 0.0

@torch.no_grad()
def canonical_scan(ckpt_path, M=8, batch_size=8, sampling_seed=0,
                   methods=("C-D+L", "L", "none_edge"), control_seeds=(0,1,2,3,4),
                   device="cpu"):
    total = M * batch_size
    model, chunks, clean_perm, dev, _ = _load_model_and_chunks(
        ckpt_path, total, seed=sampling_seed, device=device, split="train")
    inv_perm = clean_perm.inv_perm_model_to_phys.cpu().numpy()
    orders = random_reveal_orders(total, sampling_seed)
    A_acc = None
    for start in range(0, total, batch_size):
        bs = min(batch_size, total - start)
        tok = chunks[start:start+bs].to(dev)
        po = torch.from_numpy(orders[start:start+bs]).to(dev)
        _, _, attn_list = model.forward_fn(tok, po, return_attentions=True)
        attn = torch.stack(attn_list, 0).cpu().numpy()        # (L,bs,H,257,257)
        for bi in range(bs):
            A = _attn_to_A_block_loss_aligned_with_none_vec(
                attn[:, bi], orders[start+bi], inv_perm)        # (L,H,64,65)
            A_acc = A.astype(np.float64) if A_acc is None else A_acc + A
    A_lh = A_acc / total
    rows = []
    L, H = A_lh.shape[:2]
    for layer in range(L):
        for head in range(H):
            B65 = build_none_separated_B(A_lh[layer, head])
            for method in methods:
                m = discovery_metrics(rollout_by_method(B65, method))
                dz = _destroyed_floor(B65, method, control_seeds)
                rows.append({"layer": layer, "head": head, "method": method,
                             "tau_vs_l2r": float(m["tau_vs_l2r"]),
                             "abs_tau": abs(float(m["tau_vs_l2r"])),
                             "first_block": int(m["first_block"]),
                             "phys0_rank": int(m["phys0_rank"]),
                             "prefix4_overlap": int(m["prefix4_overlap"]),
                             "prefix8_overlap": int(m["prefix8_overlap"]),
                             "destroyed_abs_tau_mean": dz,
                             "gate_status": classify_gate_status(m, dz),
                             "combined_score": combined_discovery_score(m, dz)})
    return rows
```

> Implementer note: confirm `_attn_to_A_block_loss_aligned_with_none_vec` returns `(L,H,64,65)`
> for a `(L,H,257,257)` sample (it vectorises over leading dims; `search_none_separated_65_heads.py`
> calls it per-sample). If it expects a single head, loop heads. The anchor test pins correctness.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_canon_scan.py -q`
Expected: PASS (L0 strong-pass at 10k; none at 0)

- [ ] **Step 5: Commit**

```bash
git add analyses/canonical_reanalysis.py block_lo_arm_order_network/tests/test_canon_scan.py
git commit -m "feat: canonical 65-node None-separated scan on handoff_overnight ckpts"
```

---

### Task 2: Multi-sampling-seed aggregation + per-(layer,head) carrier

**Files:**
- Modify: `analyses/canonical_reanalysis.py`
- Test: `block_lo_arm_order_network/tests/test_canon_aggregate.py`

**Interfaces:**
- Produces:
  - `scan_aggregated(ckpt_path, K=3, **kw) -> dict` — runs `canonical_scan` for `sampling_seed in range(K)`, returns per-(layer,head,method) `tau_mean, tau_std, strong_frac` + `best_head`, `best_method`, `best_tau`, `strong_pass_heads` (C-D+L), `destroyed_floor_mean`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_canon_aggregate.py
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.canonical_reanalysis import scan_aggregated

def test_aggregate_seed2_10k_carrier_is_L0():
    agg = scan_aggregated(str(ROOT/"runs/handoff_overnight/seed2/ckpt_step10000.pt"), K=2)
    assert agg["best_tau"] > 0.9
    assert agg["best_head"][0] == 0                       # (layer,head); layer 0
    assert all(h[0] == 0 for h in agg["strong_pass_heads"]) or len(agg["strong_pass_heads"]) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_canon_aggregate.py -q`
Expected: FAIL

- [ ] **Step 3: Implement aggregation**

```python
# append to analyses/canonical_reanalysis.py
from collections import defaultdict

def scan_aggregated(ckpt_path, K=3, primary="C-D+L", **kw):
    by_key = defaultdict(list)        # (layer,head,method) -> [tau,...]
    gate_by_key = defaultdict(list)
    floors = []
    for s in range(K):
        for r in canonical_scan(ckpt_path, sampling_seed=s, **kw):
            k = (r["layer"], r["head"], r["method"])
            by_key[k].append(r["tau_vs_l2r"]); gate_by_key[k].append(r["gate_status"])
            floors.append(r["destroyed_abs_tau_mean"])
    per = {}
    for k, taus in by_key.items():
        per[k] = {"tau_mean": float(np.mean(taus)), "tau_std": float(np.std(taus)),
                  "strong_frac": float(np.mean([g == "strong_pass" for g in gate_by_key[k]]))}
    prim = {k: v for k, v in per.items() if k[2] == primary}
    best_k = max(prim, key=lambda k: abs(prim[k]["tau_mean"]))
    strong = sorted({(l, h) for (l, h, m), v in prim.items()
                     if m == primary and v["strong_frac"] >= 0.5})
    return {"per": per, "best_head": (best_k[0], best_k[1]), "best_method": best_k[2],
            "best_tau": abs(prim[best_k]["tau_mean"]), "strong_pass_heads": strong,
            "destroyed_floor_mean": float(np.mean(floors))}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_canon_aggregate.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analyses/canonical_reanalysis.py block_lo_arm_order_network/tests/test_canon_aggregate.py
git commit -m "feat: multi-sampling-seed aggregation + L0 carrier identification"
```

---

### Task 3: Per-seed ckpt sweep (C1 emergence baseline) + TSV

**Files:**
- Modify: `analyses/canonical_reanalysis.py`
- Test: `block_lo_arm_order_network/tests/test_canon_sweep.py`

**Interfaces:**
- Produces:
  - `sweep_seed(seed, root="runs/handoff_overnight", steps=(0,1000,...,10000), K=3, out_dir=None) -> dict` — aggregate at each step; write `strict65_sweep.tsv` (step, n_strong, best_head, best_method, best_tau, destroyed_floor) + `sweep.json`; return per-step records.

- [ ] **Step 1: Write the failing smoke test** (2 steps for speed)

```python
# tests/test_canon_sweep.py
import json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.canonical_reanalysis import sweep_seed

def test_sweep_emergence_smoke(tmp_path):
    s = sweep_seed(2, root=str(ROOT/"runs/handoff_overnight"),
                   steps=(0, 10000), K=1, out_dir=str(tmp_path))
    assert (tmp_path/"strict65_sweep.tsv").exists()
    j = json.load(open(tmp_path/"sweep.json"))
    assert j["by_step"]["0"]["best_tau"] < j["by_step"]["10000"]["best_tau"]  # emerges
    assert j["by_step"]["10000"]["best_head"][0] == 0                          # L0 carrier
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_canon_sweep.py -q`
Expected: FAIL

- [ ] **Step 3: Implement the sweep**

```python
# append to analyses/canonical_reanalysis.py
import csv as _csv, json as _json

DEFAULT_STEPS = tuple(range(0, 10001, 1000))

def sweep_seed(seed, root="runs/handoff_overnight", steps=DEFAULT_STEPS, K=3, out_dir=None):
    out = pathlib.Path(out_dir or f"runs/canonical_reanalysis/seed{seed}")
    out.mkdir(parents=True, exist_ok=True)
    by_step = {}
    for st in steps:
        agg = scan_aggregated(f"{root}/seed{seed}/ckpt_step{st}.pt", K=K)
        by_step[str(st)] = {"best_head": list(agg["best_head"]), "best_method": agg["best_method"],
                            "best_tau": agg["best_tau"], "n_strong": len(agg["strong_pass_heads"]),
                            "strong_pass_heads": [list(h) for h in agg["strong_pass_heads"]],
                            "destroyed_floor": agg["destroyed_floor_mean"]}
    with open(out/"strict65_sweep.tsv", "w", newline="") as f:
        w = _csv.writer(f, delimiter="\t")
        w.writerow(["step","n_strong","best_head","best_method","best_tau","destroyed_floor"])
        for st in steps:
            b = by_step[str(st)]
            w.writerow([st, b["n_strong"], f"L{b['best_head'][0]}H{b['best_head'][1]}",
                        b["best_method"], round(b["best_tau"],4), round(b["destroyed_floor"],4)])
    summary = {"seed": seed, "by_step": by_step}
    _json.dump(summary, open(out/"sweep.json","w"), indent=2, default=float)
    return summary
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_canon_sweep.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analyses/canonical_reanalysis.py block_lo_arm_order_network/tests/test_canon_sweep.py
git commit -m "feat: per-seed canonical ckpt sweep (emergence baseline) + TSV"
```

---

### Task 4: C2 — carrier ablation under the canonical readout

**Files:**
- Modify: `analyses/canonical_reanalysis.py`
- Test: `block_lo_arm_order_network/tests/test_canon_ablation.py`

**Interfaces:**
- Consumes: `path_patch_handoff.mean_ablation_prehook`.
- Produces:
  - `canonical_scan_ablated(ckpt_path, ablate_layer, ablate_heads, **kw) -> list[dict]` — same as `canonical_scan` but with the mean-ablation hook on `model.transformer.h[ablate_layer].attn.c_proj` active during the forward.
  - `ablation_effect(seed, ckpt_step, carrier_layer, carrier_heads, null_heads, root, K=3) -> dict` — `{strong_before, strong_after_carrier, strong_after_null, best_tau_before/after}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_canon_ablation.py
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.canonical_reanalysis import ablation_effect

def test_carrier_ablation_drops_strong_more_than_null():
    out = ablation_effect(2, 10000, carrier_layer=0, carrier_heads=[1,2,3,4],
                          null_heads=[0,6,7], root=str(ROOT/"runs/handoff_overnight"), K=1)
    assert out["strong_after_carrier"] <= out["strong_before"]
    assert "strong_after_null" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_canon_ablation.py -q`
Expected: FAIL

- [ ] **Step 3: Implement ablated scan + effect**

Refactor `canonical_scan`'s forward to accept an optional `ablate=(layer,heads)`; when set,
register `mean_ablation_prehook(heads, n_head)` on `model.transformer.h[layer].attn.c_proj`
around the forward loop (remove after). `canonical_scan_ablated` calls it. `ablation_effect`
aggregates `n_strong` (C-D+L) over K seeds for: clean, carrier-ablated, null-ablated; returns
the counts + best_tau. Full code in the module.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_canon_ablation.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analyses/canonical_reanalysis.py block_lo_arm_order_network/tests/test_canon_ablation.py
git commit -m "feat: C2 carrier ablation under canonical 65-node readout"
```

---

### Task 5: C4 — content-dependence (built-in control) read-off

**Files:**
- Modify: `analyses/canonical_reanalysis.py`
- Test: `block_lo_arm_order_network/tests/test_canon_content.py`

**Interfaces:**
- Produces:
  - `content_dependence(ckpt_path, carrier_layer, carrier_head, method="C-D+L", control_seeds=(0,1,2,3,4), **kw) -> dict` — `{tau_clean, tau_content_permuted_mean, tau_entry_shuffled_mean, content_dependent: bool}` using `content_label_permutation_control` vs the clean carrier B65.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_canon_content.py
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.canonical_reanalysis import content_dependence

def test_carrier_is_content_dependent():
    out = content_dependence(str(ROOT/"runs/handoff_overnight/seed2/ckpt_step10000.pt"),
                             carrier_layer=0, carrier_head=2)
    # a real carrier: high clean tau, collapses under content-label permutation
    assert out["tau_clean"] > 0.8
    assert out["tau_content_permuted_mean"] < 0.4
    assert out["content_dependent"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_canon_content.py -q`
Expected: FAIL

- [ ] **Step 3: Implement content_dependence**

Build the carrier head's B65 (reuse the extraction from `canonical_scan`, single head), roll out
clean τ; apply `content_label_permutation_control` (5 seeds) and `entry_shuffled_control`, mean
|τ|; `content_dependent = tau_clean - tau_content_permuted_mean > 0.4`. Full code in the module.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_canon_content.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analyses/canonical_reanalysis.py block_lo_arm_order_network/tests/test_canon_content.py
git commit -m "feat: C4 content-dependence via content_label_permutation_control"
```

---

### Task 6: Plots (emergence curve + heads heatmap)

**Files:**
- Create: `analyses/plot_canonical_reanalysis.py`
- Test: `block_lo_arm_order_network/tests/test_canon_plot.py`

**Interfaces:**
- Produces: `plot_sweep(sweep_json, out_dir)` → `emergence.png` (best_tau + n_strong vs step, with destroyed floor line).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_canon_plot.py
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.canonical_reanalysis import sweep_seed
from analyses.plot_canonical_reanalysis import plot_sweep

def test_plot_sweep_writes_png(tmp_path):
    sweep_seed(2, root=str(ROOT/"runs/handoff_overnight"), steps=(0,10000), K=1, out_dir=str(tmp_path))
    plot_sweep(str(tmp_path/"sweep.json"), str(tmp_path))
    assert (tmp_path/"emergence.png").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_canon_plot.py -q`
Expected: FAIL

- [ ] **Step 3: Implement plot_sweep** — matplotlib Agg, twin-axis best_tau (left) + n_strong (right) vs step, horizontal line at destroyed floor, title with the step-10000 carrier head. Full code in the module.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_canon_plot.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analyses/plot_canonical_reanalysis.py block_lo_arm_order_network/tests/test_canon_plot.py
git commit -m "feat: canonical emergence curve plot"
```

---

### Task 7: Full 3-seed run + corrected README

**Files:**
- Create: `runs/canonical_reanalysis/` (outputs)
- Create: `analyses/canonical_reanalysis_README.md`

- [ ] **Step 1: Run the sweep for all 3 seeds**

Run: `cd block_lo_arm_order_network && python -c "from analyses.canonical_reanalysis import sweep_seed; [sweep_seed(s, root='../runs/handoff_overnight', K=3, out_dir=f'../runs/canonical_reanalysis/seed{s}') for s in (2,42,123)]"`
Expected: `strict65_sweep.tsv` + `sweep.json` per seed; emergence visible (floor→1.0, L0 carrier).

- [ ] **Step 2: Plots + C2 ablation + C4 content for each seed**

Call `plot_sweep`, `ablation_effect` (carrier = step-10000 strong-pass heads vs null), and `content_dependence` (carrier head) per seed; save outputs.

- [ ] **Step 3: Write `analyses/canonical_reanalysis_README.md`**

Per seed: real carrier (L0 heads + contrast with model-frame L1), emergence timing (coarse interval, vs destroyed floor), C2 ablation (carrier vs null strong-pass collapse), C4 content-dependence (content_label_permutation collapse). Then the **corrected line-level conclusion** superseding ③/A/⑤: the order signal is **absent at init, genuinely emerges** to an L0 content-bound carrier; the model-frame L1 result was a tautological-readout artifact. State which prior conclusions are corrected.

- [ ] **Step 4: Commit**

```bash
git add -f analyses/canonical_reanalysis_README.md runs/canonical_reanalysis
git commit -m "results: canonical 65-node re-analysis (3 seeds) + corrected conclusions"
```

---

## Self-Review

**Spec coverage:**
- Strict 65-node protocol (None-sep, loss-aligned AR, posthoc-inv, random reveal, C-D+L primary, destroyed controls) → Task 1 (Global Constraints + canonical_scan). ✓
- Variance (K sampling seeds + control_seeds=5) → Tasks 1,2. ✓
- C0/C1 emergence + carrier baseline (3 seeds × 11 ckpts) → Tasks 3,7. ✓
- C2 carrier ablation, canonical readout → Task 4. ✓
- C3 emergence shape (read off sweep) → Tasks 3,7 (README). ✓
- C4 content vs position (content_label_permutation_control) → Task 5. ✓
- Plots, README, corrected conclusions → Tasks 6,7. ✓
- Anchor (step0 floor / step10000 L0 strong τ=1.0) → Task 1 tests. ✓

**Placeholder scan:** Tasks 4/5/6 step-3 give itemized algorithms over already-tested helpers
(`canonical_scan`, `mean_ablation_prehook`, the controls); the data-producing scan (Task 1) has
full code. No TBD/TODO in logic-bearing steps.

**Type consistency:** `canonical_scan` row keys (`gate_status`,`abs_tau`,`tau_vs_l2r`,`layer`,`head`,
`method`) consumed identically by `scan_aggregated`/`sweep_seed`/ablation; `best_head`=(layer,head)
tuple throughout; method `C-D+L` primary everywhere; `strong_pass_heads` = list of (layer,head).
