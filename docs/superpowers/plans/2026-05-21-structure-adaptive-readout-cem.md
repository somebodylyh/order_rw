# Structure-Conditioned Adaptive Readout — CEM/ES Oracle Capstone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether the readout parameters that our hand-designed graph-regime rule prescribes (text→readiness, image→proximity/coverage, VQ-uniform→random fallback) can be **automatically discovered from graph structure** by a black-box optimizer (CEM/ES). If yes, the work upgrades from "we hand-found that different graph regimes need different readouts" to "attention-graph structure *induces* an optimal readout regime that is automatically recoverable."

**Architecture:** Three layers, built bottom-up. (1) A single **unified continuous readout family** `score = β_sup·C − β_dep·D + ρ·r + γ_B·B̃_local − γ_d·d̃ + fallback_mix·uniform` that can instantiate readiness-Graph-RW, Bcov-proximity, distance-only, and random by parameter choice alone. (2) A **CEM/ES search** over those parameters per graph, with a cheap two-tier fitness (frozen order-specific eval for screening, short continuation for confirmation). (3) An optional **MLP hypernetwork** that imitates `g(B) → w*` to show the regime decision can be automated. GRPO fine-tuning and AttnTNT are documented as future, not implemented here.

**Tech Stack:** PyTorch 2.x, NumPy, existing `directed_graph_policy`, `graph_rw_image`, `graph_regime_diagnostic.py`, `readout_order_diagnostic.py`, `train_imagelarge_graph_rw.py` (frozen-eval + short-continuation harness), nanogpt-learned-order `AOGPT` (BlockAOGPT). Pure-CPU for sampling/diagnostics; GPU only for continuation fitness.

---

## Context for the implementer (read once)

This is an ML research plan; "tests" are sanity gates (assertions on shapes / monotonic responses / control behavior), not unit tests of business logic. Each task still follows write→run→verify→commit.

**Hard-won facts from the prior stage (do not relitigate):**
- The baseline image ckpt was trained with `permute_data=True`. Any **physical-frame** block order MUST be remapped to model frame via `inverse_block_perm` before being fed to the model. This remap lives in `train_imagelarge_graph_rw.py::_forward_with_block_orders`. Reuse it; never re-implement per-arm.
- `build_directed_graph(A)` returns `B = A^T` diag-zeroed, **not** row-normalized (row sums ~0.19). Samplers consume raw B via tau-softmax.
- Single-step softmax Graph-RW (v1/v2/v3) cannot produce spatially-local orders on the E3 proximity graph; the `support` term dominates. Proven, archived. We are NOT trying to fix v1/v3.
- `locality` (mean_manh, P(d≤1)) is a **diagnostic**, not the objective. Final fitness is task loss / cross-order robustness.
- Regime classifier (`graph_regime_diagnostic.py`) already classifies: text→readiness_dominant, E3→proximity_dominant, E2/shuffled/random→uniform_noisy, using `directionality` to split readiness (1D directional) from proximity (2D multi-directional).

**Inputs that already exist (verify in Task 0.1):**
| What | Path |
|---|---|
| E3-control-small A_block (image proximity) | `probe_results_image_large/imagenet64_vqf4_full_l4h8e256_patch2x2_control/A_block_8x8.npy` |
| E3-large/fixed A_global | `probe_results_image_large/imagenet64_vqf4_full_l8h8e512_patch2x2_fixed/A_global.npy` |
| E2 single-token VQ A_global | `probe_results_image/e2_imagenet32_vqf4_seq64/attention/A_global.npy` |
| text clean-method A_global | `block_lo_arm_order_network/probe_results/clean_method_graph_rw_a10_to_a095_30k60k/A_global_step50000.npy` |
| baseline ckpt (for fitness) | `nanogpt-learned-order/out/image_large/imagenet64_vq_f4_800k/seq256/patch2x2_full_baseline_l4h8e256/ckpt.pt` |
| val/train data | `nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch8x8/{val,train}.bin,meta.pkl` |
| **Round-2 Bcov multi-seed results (ran externally — PATH NEEDED)** | _user to provide; see Task 1.2_ |

**Commit policy:** commit code, configs, small TSV/JSON/MD result tables. NEVER commit `ckpt*.pt`, `A_global.npy`, or large `.npy` order pools.

---

## File structure

### Create
- `block_lo_arm_order_network/unified_readout.py` — the one continuous readout family + a deterministic and a stochastic sampler. The single source of truth for "turn B + params into an order." (~180 LOC)
- `block_lo_arm_order_network/cem_readout_search.py` — CEM/ES optimizer over the readout params for one graph, with pluggable fitness. (~250 LOC)
- `block_lo_arm_order_network/readout_fitness.py` — fitness functions: `frozen_eval_fitness` (fast screening) and `short_continuation_fitness` (confirm). Wraps the baseline model + the inverse_block_perm remap. (~160 LOC)
- `scripts/run_cem_oracle.sh` — orchestration: run CEM per graph (text / E3 / E3-large / E2 / controls), each to its own output dir.
- `scripts/regime_threshold_sensitivity.py` — Phase-1 threshold sweep over the regime rule.
- `scripts/round2_bcov_ci_summary.py` — Phase-1 multi-seed CI + paired sign-count consolidation of the externally-run Round-2 Bcov results.
- `block_lo_arm_order_network/mlp_readout_hypernet.py` — Phase-3 (optional) MLP imitation of `g(B)→w*`.
- `probe_results_image_large/structure_adaptive/` — output root: `threshold_sensitivity/`, `round2_bcov_ci/`, `cem/<graph>/`, `hypernet/`.

### Reuse (do NOT modify except where a task says Modify)
- `block_lo_arm_order_network/graph_regime_diagnostic.py` — diagnostics `g(B)`; Modify in Task 1.1 to add a sweep entry point.
- `block_lo_arm_order_network/readout_order_diagnostic.py` — L1 order metrics (mean_manh, P(d≤1), directionality helpers); import its stat functions.
- `block_lo_arm_order_network/train_imagelarge_graph_rw.py` — `load_baseline_model`, `_forward_with_block_orders` (the inverse_block_perm remap), data loading. Import, do not duplicate.
- `block_lo_arm_order_network/directed_graph_policy.py`, `image_order/graph_rw_image.py` — graph + sampler primitives.

---

## Phase 0 — Freeze prior stage, wire inputs

### Task 0.1: Verify all input artifacts load and freeze the prior conclusions

**Files:**
- Create: `probe_results_image_large/structure_adaptive/inputs_preflight.txt`

- [ ] **Step 1: Confirm every input path exists and has expected shape**

Run:
```bash
cd /home/admin/lyuyuhuan/order_lyu
python3 - <<'EOF'
import numpy as np, os
paths = {
 "E3_ctrl_small": "probe_results_image_large/imagenet64_vqf4_full_l4h8e256_patch2x2_control/A_block_8x8.npy",
 "E3_large_fixed": "probe_results_image_large/imagenet64_vqf4_full_l8h8e512_patch2x2_fixed/A_global.npy",
 "E2_singletoken": "probe_results_image/e2_imagenet32_vqf4_seq64/attention/A_global.npy",
 "text_clean": "block_lo_arm_order_network/probe_results/clean_method_graph_rw_a10_to_a095_30k60k/A_global_step50000.npy",
 "baseline_ckpt": "nanogpt-learned-order/out/image_large/imagenet64_vq_f4_800k/seq256/patch2x2_full_baseline_l4h8e256/ckpt.pt",
 "val_bin": "nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch8x8/val.bin",
}
for k,p in paths.items():
    ok = os.path.exists(p)
    extra = ""
    if ok and p.endswith(".npy"):
        a = np.load(p); extra = f"shape={a.shape}"
    print(f"{'OK ' if ok else 'MISSING'} {k:16s} {extra}  {p}")
EOF
```

Expected: all `OK`; the four `.npy` are `(64,64)`. If any MISSING, STOP and report (the graph set must be complete or the regime story has a gap).

- [ ] **Step 2: Write freeze note + commit**

Run:
```bash
mkdir -p probe_results_image_large/structure_adaptive
cat > probe_results_image_large/structure_adaptive/inputs_preflight.txt <<'EOF'
Structure-adaptive readout CEM capstone — frozen prior stage.
PRIOR STAGE (sealed, do not reoptimize):
- Stage-1 v1/v3 readout mismatch: SUMMARY.md, STAGE1_FINDINGS.md
- Graph diagnostics + rule-based regime: graph_regime/README.md
- Frozen Bcov_balanced spec: round2_bcov_balanced_frozen.md (gamma_B=gamma_d=1.0)
- Round-2 Bcov multi-seed continuation: RAN EXTERNALLY (results path to be imported in Task 1.2)
NEXT: CEM/ES oracle search over a unified readout family to test whether the
regime decision is automatically recoverable from graph structure.
EOF
git add probe_results_image_large/structure_adaptive/inputs_preflight.txt
git commit -m "structure-adaptive: phase-0 inputs preflight + freeze note"
```

---

## Phase 1 — Diagnostic robustness (no training)

### Task 1.1: Threshold sensitivity analysis for the regime rule

**Files:**
- Create: `scripts/regime_threshold_sensitivity.py`
- Modify: `block_lo_arm_order_network/graph_regime_diagnostic.py` (expose `diagnose` + the graph list for import)

- [ ] **Step 1: Make `graph_regime_diagnostic.py` importable without running main**

Modify `block_lo_arm_order_network/graph_regime_diagnostic.py`: add a module-level function that returns the standard graph list so other scripts reuse it (DRY). Add near the top, after imports:

```python
def standard_graphs():
    """(name, path, topology, grid) for the canonical graph set."""
    return [
        ("E3_ctrl_small_image", "probe_results_image_large/imagenet64_vqf4_full_l4h8e256_patch2x2_control/A_block_8x8.npy", "grid2d", 8),
        ("E2_singletoken_vq_image", "probe_results_image/e2_imagenet32_vqf4_seq64/attention/A_global.npy", "grid2d", 8),
        ("E2_large_image", "probe_results_image/e2_large_imagenet32_vqf4_seq64_l8h8e512/attention/A_global.npy", "grid2d", 8),
        ("E3_large_fixed_image", "probe_results_image_large/imagenet64_vqf4_full_l8h8e512_patch2x2_fixed/A_global.npy", "grid2d", 8),
        ("text_clean_method", "block_lo_arm_order_network/probe_results/clean_method_graph_rw_a10_to_a095_30k60k/A_global_step50000.npy", "seq1d", 0),
    ]
```

Then change `main()`'s local `graphs = [...]` to `graphs = standard_graphs()` (keep the derived shuffled/random controls block as-is).

- [ ] **Step 2: Add a `classify_regime_with_thresholds` variant**

Modify `graph_regime_diagnostic.py`: refactor the hard-coded thresholds in `classify_regime` into parameters with the current values as defaults:

```python
def classify_regime(diag, loc_thr=0.5, pnbr_thr=0.4, direc_thr=0.7,
                    ent_thr=0.93, t1_thr=0.05):
    loc = diag["locality_score"]; ent = diag["row_entropy"]
    t1 = diag["top1_mass"]; direc = diag["directionality"]
    if loc >= loc_thr and diag["p_nbr_le1"] >= pnbr_thr:
        if direc >= direc_thr:
            return "readiness_dominant", "readiness-guided Graph-RW (progressive_rw_v3, high rho); directional 1D edges"
        return "proximity_dominant", "coverage_readout (Bcov_balanced / Hilbert); single-step Graph-RW insufficient"
    elif ent >= ent_thr and t1 <= t1_thr:
        return "uniform_noisy", "no structured readout — random/mixed fallback; do not over-trust B"
    return "weak_mixed", "mild structure — random-mixed continuation, validate before trusting B"
```

(Behavior at defaults is identical to the frozen version — verify in Step 4.)

- [ ] **Step 3: Write the sensitivity sweep script**

Create `scripts/regime_threshold_sensitivity.py`:

```python
#!/usr/bin/env python3
"""Sweep regime-rule thresholds; report regime stability per graph."""
import sys, itertools, json
from pathlib import Path
import numpy as np
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "block_lo_arm_order_network"))
from directed_graph_policy import build_directed_graph
import graph_regime_diagnostic as grd

OUT = _REPO / "probe_results_image_large/structure_adaptive/threshold_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)

LOC = [0.4, 0.5, 0.6]; PNBR = [0.3, 0.4, 0.5]; DIREC = [0.6, 0.7, 0.8]

def diag_for(path, topo, grid):
    A = np.load(_REPO / path).astype(np.float64)
    return grd.diagnose(build_directed_graph(A), topo, grid)

def main():
    rng = np.random.default_rng(0)
    graphs = grd.standard_graphs()
    diags = {name: diag_for(p, t, g) for name, p, t, g in graphs}
    # derived controls reuse E3 B
    A3 = np.load(_REPO / graphs[0][1]).astype(np.float64); B3 = build_directed_graph(A3)
    diags["E3_shuffled_cols"] = grd.diagnose(np.stack([B3[i, rng.permutation(64)] for i in range(64)]), "grid2d", 8)
    diags["random_B"] = grd.diagnose((lambda r: (np.fill_diagonal(r, 0.0) or r))(rng.random((64,64))), "grid2d", 8)

    rows = []
    for name, d in diags.items():
        regimes = []
        for lt, pt, dt in itertools.product(LOC, PNBR, DIREC):
            r, _ = grd.classify_regime(d, loc_thr=lt, pnbr_thr=pt, direc_thr=dt)
            regimes.append(r)
        from collections import Counter
        c = Counter(regimes); top, cnt = c.most_common(1)[0]
        rows.append((name, top, cnt, len(regimes), dict(c)))

    with open(OUT / "sensitivity.tsv", "w") as f:
        f.write("graph\tmajority_regime\tstable_count\ttotal\tdistribution\n")
        for name, top, cnt, tot, dist in rows:
            f.write(f"{name}\t{top}\t{cnt}\t{tot}\t{json.dumps(dist)}\n")
    print(f"{'graph':24s}{'majority':20s}{'stable':>8}{'total':>7}  distribution")
    for name, top, cnt, tot, dist in rows:
        print(f"{name:24s}{top:20s}{cnt:8d}{tot:7d}  {dist}")
    print(f"\nwrote {OUT}/sensitivity.tsv")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run + verify default behavior unchanged and stability**

Run:
```bash
cd /home/admin/lyuyuhuan/order_lyu
python block_lo_arm_order_network/graph_regime_diagnostic.py   # default thresholds, must match frozen result
python scripts/regime_threshold_sensitivity.py
```

Expected:
- The plain run still classifies text→readiness_dominant, E3_ctrl_small→proximity_dominant, E2/E3_large/shuffled/random→uniform_noisy (defaults unchanged).
- Sensitivity: `text_clean_method` and `E3_ctrl_small_image` should be stable (majority == frozen regime across most/all of the 27 threshold combos). If `E3_large_fixed` flips between uniform_noisy and proximity near the boundary, that is a **finding** (motivates continuous adaptive readout over hard thresholds), not a failure — record it.

- [ ] **Step 5: Commit**

```bash
git add scripts/regime_threshold_sensitivity.py block_lo_arm_order_network/graph_regime_diagnostic.py \
        probe_results_image_large/structure_adaptive/threshold_sensitivity/sensitivity.tsv
git commit -m "structure-adaptive: regime threshold sensitivity analysis (parameterized thresholds)"
```

### Task 1.2: Consolidate the externally-run Round-2 Bcov multi-seed CI + sign-count

**Files:**
- Create: `scripts/round2_bcov_ci_summary.py`
- Reads: externally-produced Round-2 result files (path provided by user)

> **BLOCKER until path provided.** The Round-2 Bcov multi-seed continuation was run outside this repo/session. Before this task, the user must place (or point to) the per-seed result files. Expected form: one row per (arm, seed) with columns including `seed, val_cross_avg, val_structured_avg, val_noisy_avg` for arms `Bcov_balanced, random, distance_only, shuffled_Bcov`. If the layout differs, adapt the loader in Step 1 and note it. Do NOT hard-code the user's quoted numbers (−0.0029±0.0007 etc.) — compute them from the files.

- [ ] **Step 1: Write the CI / sign-count summary script**

Create `scripts/round2_bcov_ci_summary.py`:

```python
#!/usr/bin/env python3
"""Multi-seed CI + paired sign-count for Round-2 Bcov vs controls.
Reads a TSV of per-(arm,seed) eval aggregates; computes paired deltas, mean,
95% CI (t-based), and #seeds where Bcov_balanced wins (delta<0)."""
import argparse, csv, math
from pathlib import Path
from collections import defaultdict

def t95(n):  # two-sided 95% t multiplier, small-n table
    return {2:12.71,3:4.303,4:3.182,5:2.776,6:2.571,7:2.447,8:2.365,9:2.306,10:2.262}.get(n, 1.96)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="TSV: arm,seed,val_cross_avg,val_structured_avg,val_noisy_avg")
    ap.add_argument("--baseline-arm", default="Bcov_balanced")
    ap.add_argument("--out", default="probe_results_image_large/structure_adaptive/round2_bcov_ci/summary.md")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.results), delimiter="\t"))
    by = defaultdict(dict)  # (arm)->{seed: {metric:val}}
    metrics = ["val_cross_avg", "val_structured_avg", "val_noisy_avg"]
    for r in rows:
        by[r["arm"]][int(r["seed"])] = {m: float(r[m]) for m in metrics}
    base = by[args.baseline_arm]
    out = ["# Round-2 Bcov multi-seed CI + sign-count\n\n",
           f"baseline arm = **{args.baseline_arm}** (negative delta = Bcov better)\n\n",
           "| comparison | metric | mean Δ | 95% CI | seeds Bcov wins |\n|---|---|--:|--:|--:|\n"]
    for arm in by:
        if arm == args.baseline_arm: continue
        seeds = sorted(set(base) & set(by[arm]))
        for m in metrics:
            deltas = [base[s][m] - by[arm][s][m] for s in seeds]  # Bcov - control
            n = len(deltas); mean = sum(deltas)/n
            sd = math.sqrt(sum((d-mean)**2 for d in deltas)/(n-1)) if n>1 else 0.0
            ci = t95(n) * sd / math.sqrt(n) if n>1 else 0.0
            wins = sum(1 for d in deltas if d < 0)
            out.append(f"| {args.baseline_arm} vs {arm} | {m} | {mean:+.4f} | ±{ci:.4f} | {wins}/{n} |\n")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(out)); print("".join(out)); print(f"wrote {args.out}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run against the provided results and verify**

Run (substitute the real path the user provides):
```bash
cd /home/admin/lyuyuhuan/order_lyu
python scripts/round2_bcov_ci_summary.py --results <PATH_TO_ROUND2_RESULTS.tsv>
```

Expected: a table with mean Δ, 95% CI, and sign-count (e.g. "Bcov beats distance_only on cross in N/N seeds"). Sanity: the mean Δ signs should match the user's reported direction (Bcov better than random/distance/shuffled on cross & structured). If signs disagree with the user's external numbers, STOP and reconcile (wrong file or arm mislabel).

- [ ] **Step 3: Commit**

```bash
git add scripts/round2_bcov_ci_summary.py probe_results_image_large/structure_adaptive/round2_bcov_ci/summary.md
git commit -m "structure-adaptive: Round-2 Bcov multi-seed CI + paired sign-count summary"
```

---

## Phase 2 — Unified readout family + CEM/ES machinery

### Task 2.1: Implement the unified continuous readout family

**Files:**
- Create: `block_lo_arm_order_network/unified_readout.py`

- [ ] **Step 1: Write the readout family**

Create `block_lo_arm_order_network/unified_readout.py`:

```python
#!/usr/bin/env python3
"""Unified continuous readout family. One scoring function instantiates
readiness-Graph-RW, Bcov-proximity, distance-only, and random by params alone.

score_t(v) = beta_sup*Csup(v) - beta_dep*Dfut(v) + rho*r(v)
             + gamma_B*localB(v) - gamma_d*dist(v)
Each term is min-max normalized over the current candidate set so weights are
commensurable across graphs of different scale. fallback_mix blends the final
policy with uniform. Orders are PHYSICAL frame (remap to model frame downstream).
"""
import numpy as np

PARAM_KEYS = ["beta_sup", "beta_dep", "rho", "gamma_B", "gamma_d", "tau", "fallback_mix"]
PARAM_BOUNDS = {  # (lo, hi) for CEM clipping
    "beta_sup": (0.0, 2.0), "beta_dep": (0.0, 2.0), "rho": (0.0, 2.0),
    "gamma_B": (0.0, 2.0), "gamma_d": (0.0, 2.0), "tau": (0.03, 0.5),
    "fallback_mix": (0.0, 1.0),
}

def _coords(N, topology, grid):
    if topology == "grid2d":
        idx = np.arange(N); return np.stack([idx // grid, idx % grid], 1).astype(float)
    return np.arange(N).reshape(N, 1).astype(float)

def _minmax(x):
    x = np.asarray(x, float); r = x.ptp()
    return (x - x.min()) / (r + 1e-12) if r > 0 else np.zeros_like(x)

def sample_order(B, source, coords, w, rng, has_grid):
    """One order (length N) in physical frame from params w (dict)."""
    N = B.shape[0]
    U = list(range(N)); S = []; last = -1; order = []
    for _ in range(N):
        U_arr = np.array(U)
        Csup = _minmax(B[S].sum(0)[U_arr]) if S else np.zeros(len(U))
        Dfut = _minmax(B[U_arr].sum(0)[U_arr]) if len(U) > 1 else np.zeros(len(U))
        r = _minmax(source[U_arr])
        localB = _minmax(B[last, U_arr]) if last >= 0 else np.zeros(len(U))
        if has_grid and last >= 0:
            d = np.abs(coords[U_arr] - coords[last]).sum(1); dist = _minmax(d)
        else:
            dist = np.zeros(len(U))
        score = (w["beta_sup"]*Csup - w["beta_dep"]*Dfut + w["rho"]*r
                 + w["gamma_B"]*localB - w["gamma_d"]*dist)
        score = score - score.max()
        p = np.exp(score / max(w["tau"], 1e-3)); p = p / p.sum()
        p = (1.0 - w["fallback_mix"]) * p + w["fallback_mix"] * (1.0 / len(U))
        p = p / p.sum()
        ci = int(rng.choice(len(U), p=p)); v = U[ci]
        order.append(v); S.append(v); last = v; U.pop(ci)
    return np.array(order, dtype=np.int64)

def sample_orders_batch(B, source, coords, w, k, seed, has_grid):
    rng = np.random.default_rng(seed)
    return np.stack([sample_order(B, source, coords, w, rng, has_grid) for _ in range(k)])

def clip_params(w):
    return {k: float(np.clip(w[k], *PARAM_BOUNDS[k])) for k in PARAM_KEYS}

def make_coords(N, topology, grid):
    return _coords(N, topology, grid), (topology == "grid2d")
```

- [ ] **Step 2: Sanity test — params reproduce known regimes**

Run:
```bash
cd /home/admin/lyuyuhuan/order_lyu
python3 - <<'EOF'
import sys, numpy as np
sys.path.insert(0,"block_lo_arm_order_network")
from directed_graph_policy import build_directed_graph, compute_source
import unified_readout as ur
from readout_order_diagnostic import order_stats   # mean_manh / p_d_le1
A=np.load("probe_results_image_large/imagenet64_vqf4_full_l4h8e256_patch2x2_control/A_block_8x8.npy").astype(float)
B=build_directed_graph(A); src,_,_=compute_source(B,0.5); coords,has=ur.make_coords(64,"grid2d",8)
def manh(w): 
    o=ur.sample_orders_batch(B,src,coords,w,8,0,has); return order_stats(o)["mean_manh"]
base=dict(beta_sup=0,beta_dep=0,rho=0,gamma_B=0,gamma_d=0,tau=0.1,fallback_mix=0)
print("proximity-like (gamma_B,gamma_d high):", round(manh({**base,"gamma_B":1.5,"gamma_d":1.0}),3), "(expect ~2-3, << random 5.3)")
print("support-like (beta_sup high):        ", round(manh({**base,"beta_sup":1.5}),3), "(expect ~5, near random)")
print("fallback random (fallback_mix=1):    ", round(manh({**base,"fallback_mix":1.0}),3), "(expect ~5.3)")
EOF
```

Expected: proximity-like config gives clearly lower mean_manh (~2–3.5) than support-like (~5) and random (~5.3). This proves the family can express the proximity regime via params alone.

- [ ] **Step 3: Commit**

```bash
git add block_lo_arm_order_network/unified_readout.py
git commit -m "structure-adaptive: unified continuous readout family (one scoring fn for all regimes)"
```

### Task 2.2: Implement fitness functions (frozen-eval screening + short-continuation confirm)

**Files:**
- Create: `block_lo_arm_order_network/readout_fitness.py`

- [ ] **Step 1: Write the fitness module**

Create `block_lo_arm_order_network/readout_fitness.py`:

```python
#!/usr/bin/env python3
"""Fitness for CEM readout search. Two tiers:
  frozen_eval_fitness     — no training; CE under sampled orders on frozen model (fast).
  short_continuation_fitness — N-step continuation from baseline, then cross-order eval.
Both reuse train_imagelarge_graph_rw for model load + the inverse_block_perm remap.
Higher fitness = better = NEGATIVE loss.
"""
import sys, math
from pathlib import Path
import numpy as np, torch
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "block_lo_arm_order_network"))
import train_imagelarge_graph_rw as T
import unified_readout as ur
from directed_graph_policy import build_directed_graph, compute_source

SEQ = 256; N = 64

def _load(ckpt, device):
    model, margs, ckptd, ftp, ibp = T.load_baseline_model(ckpt, device)
    return model, ftp, ibp

def _orders_phys_to_model_blocks(orders_phys, ibp, device):
    o = torch.as_tensor(orders_phys, dtype=torch.long, device=device)
    return o  # _forward_with_block_orders applies ibp remap internally

@torch.no_grad()
def _cross_eval(model, val_tokens, B, source, coords, has_grid, w, ftp, ibp,
                device, k_batches=8, bs=16, seed=123):
    model.eval()
    losses = []
    for bi in range(k_batches):
        x = val_tokens[bi*bs:(bi+1)*bs].to(device)
        if x.shape[0] == 0: break
        ob = ur.sample_orders_batch(B, source, coords, w, x.shape[0], seed+bi, has_grid)
        bo = torch.as_tensor(ob, dtype=torch.long, device=device)
        loss = T._forward_with_block_orders(model, x, bo, fixed_token_perm=ftp, inv_block_perm=ibp)
        losses.append(float(loss.item()))
    return float(np.mean(losses))

def frozen_eval_fitness(ckpt, A_block_path, val_bin, meta, topology, grid, device="cuda:0"):
    model, ftp, ibp = _load(ckpt, device)
    A = np.load(A_block_path).astype(np.float64); B = build_directed_graph(A)
    source, _, _ = compute_source(B, 0.5); coords, has = ur.make_coords(N, topology, grid)
    import pickle; meta_d = pickle.load(open(meta, "rb")); assert int(meta_d["tokens_per_image"]) == SEQ
    vm = np.memmap(val_bin, dtype=np.uint16, mode="r"); nval = min(512, len(vm)//SEQ)
    val = torch.from_numpy(np.asarray(vm[:nval*SEQ], dtype=np.int64).reshape(nval, SEQ))
    def fitness(w):
        return -_cross_eval(model, val, B, source, coords, has, ur.clip_params(w), ftp, ibp, device)
    return fitness

def short_continuation_fitness(ckpt, A_block_path, data_dir, topology, grid,
                               steps=200, device="cuda:0"):
    """Fitness = -cross_avg after `steps` continuation under orders from w.
    Implemented by importing T.main-style loop is heavy; here we do a compact
    in-process continuation. See run_cem_oracle.sh for the wired invocation."""
    raise NotImplementedError("Wired in Task 2.4; frozen_eval is the default screening fitness.")
```

> Design note: Phase-2 CEM uses `frozen_eval_fitness` as the default (cheap, no training). `short_continuation_fitness` is deferred to Task 2.4 where it is wired as a confirm-only re-rank of the top-K CEM candidates, to keep GPU cost bounded.

- [ ] **Step 2: Sanity — frozen fitness ranks proximity > random on E3**

Run:
```bash
cd /home/admin/lyuyuhuan/order_lyu
CUDA_VISIBLE_DEVICES=0 python3 - <<'EOF'
import sys; sys.path.insert(0,"block_lo_arm_order_network")
from readout_fitness import frozen_eval_fitness
f = frozen_eval_fitness(
  "nanogpt-learned-order/out/image_large/imagenet64_vq_f4_800k/seq256/patch2x2_full_baseline_l4h8e256/ckpt.pt",
  "probe_results_image_large/imagenet64_vqf4_full_l4h8e256_patch2x2_control/A_block_8x8.npy",
  "nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch8x8/val.bin",
  "nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch8x8/meta.pkl",
  "grid2d", 8)
base=dict(beta_sup=0,beta_dep=0,rho=0,gamma_B=0,gamma_d=0,tau=0.1,fallback_mix=0)
prox=f({**base,"gamma_B":1.5,"gamma_d":1.0}); rnd=f({**base,"fallback_mix":1.0})
print(f"frozen fitness proximity={prox:.4f} random={rnd:.4f}  (proximity should be >= random within ~0.01)")
EOF
```

Expected: frozen fitness runs without the inverse_block_perm error; proximity config ≳ random (the frozen baseline was trained on random orders, so the gap may be small — the point is the harness works and is ordered sensibly, not a large gap).

- [ ] **Step 3: Commit**

```bash
git add block_lo_arm_order_network/readout_fitness.py
git commit -m "structure-adaptive: readout fitness (frozen-eval screening) reusing inverse_block_perm remap"
```

### Task 2.3: Implement the CEM/ES search

**Files:**
- Create: `block_lo_arm_order_network/cem_readout_search.py`

- [ ] **Step 1: Write the CEM optimizer**

Create `block_lo_arm_order_network/cem_readout_search.py`:

```python
#!/usr/bin/env python3
"""CEM search over unified-readout params for one graph. Fitness is pluggable."""
import argparse, json, sys
from pathlib import Path
import numpy as np
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "block_lo_arm_order_network"))
import unified_readout as ur

def cem(fitness, pop=16, elite=4, gens=5, seed=0, init=None, log=print):
    rng = np.random.default_rng(seed)
    keys = ur.PARAM_KEYS
    lo = np.array([ur.PARAM_BOUNDS[k][0] for k in keys])
    hi = np.array([ur.PARAM_BOUNDS[k][1] for k in keys])
    mu = (lo + hi) / 2 if init is None else np.array([init[k] for k in keys])
    sigma = (hi - lo) / 4
    history = []
    best = (-1e18, None)
    for g in range(gens):
        cand = rng.normal(mu, sigma, size=(pop, len(keys)))
        cand = np.clip(cand, lo, hi)
        fits = []
        for c in cand:
            w = {k: float(v) for k, v in zip(keys, c)}
            fits.append(fitness(ur.clip_params(w)))
        fits = np.array(fits)
        order = np.argsort(-fits)
        elite_idx = order[:elite]
        mu = cand[elite_idx].mean(0); sigma = cand[elite_idx].std(0) + 1e-3
        gbest = fits[order[0]]
        if gbest > best[0]:
            best = (float(gbest), {k: float(v) for k, v in zip(keys, cand[order[0]])})
        history.append(dict(gen=g, best_fit=float(gbest), mean_fit=float(fits.mean()),
                            mu={k: float(v) for k, v in zip(keys, mu)}))
        log(f"  gen {g}: best={gbest:.4f} mean={fits.mean():.4f} mu={ {k: round(float(v),3) for k,v in zip(keys,mu)} }")
    return best, history

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graph-name", required=True)
    p.add_argument("--a-path", required=True)
    p.add_argument("--topology", choices=["grid2d", "seq1d"], required=True)
    p.add_argument("--grid", type=int, default=8)
    p.add_argument("--fitness", choices=["frozen", "dryrun"], default="frozen")
    p.add_argument("--ckpt", default="")
    p.add_argument("--val-bin", default="")
    p.add_argument("--meta", default="")
    p.add_argument("--pop", type=int, default=16)
    p.add_argument("--elite", type=int, default=4)
    p.add_argument("--gens", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    if args.fitness == "dryrun":
        # cheap surrogate: reward low mean_manh (grid) — verifies the loop end-to-end
        from directed_graph_policy import build_directed_graph, compute_source
        from readout_order_diagnostic import order_stats
        A = np.load(args.a_path).astype(np.float64); B = build_directed_graph(A)
        src, _, _ = compute_source(B, 0.5); coords, has = ur.make_coords(B.shape[0], args.topology, args.grid)
        def fitness(w):
            o = ur.sample_orders_batch(B, src, coords, ur.clip_params(w), 6, 0, has)
            return -order_stats(o)["mean_manh"]
    else:
        from readout_fitness import frozen_eval_fitness
        fitness = frozen_eval_fitness(args.ckpt, args.a_path, args.val_bin, args.meta,
                                      args.topology, args.grid)

    logf = open(out / "cem_log.txt", "w")
    def log(m): print(m, flush=True); logf.write(m + "\n"); logf.flush()
    log(f"[CEM] graph={args.graph_name} fitness={args.fitness} pop={args.pop} elite={args.elite} gens={args.gens}")
    best, hist = cem(fitness, args.pop, args.elite, args.gens, args.seed, log=log)
    json.dump(dict(graph=args.graph_name, best_fit=best[0], best_w=best[1], history=hist),
              open(out / "cem_result.json", "w"), indent=2)
    log(f"[CEM] best_fit={best[0]:.4f} best_w={best[1]}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Dry-run on E3 (surrogate fitness, CPU, verifies the whole loop)**

Run:
```bash
cd /home/admin/lyuyuhuan/order_lyu
python block_lo_arm_order_network/cem_readout_search.py \
  --graph-name E3_dryrun \
  --a-path probe_results_image_large/imagenet64_vqf4_full_l4h8e256_patch2x2_control/A_block_8x8.npy \
  --topology grid2d --grid 8 --fitness dryrun --gens 5 --pop 16 --elite 4 \
  --outdir probe_results_image_large/structure_adaptive/cem/E3_dryrun
cat probe_results_image_large/structure_adaptive/cem/E3_dryrun/cem_result.json | python -m json.tool | head -30
```

Expected: `cem_log.txt` shows best_fit improving (mean_manh decreasing) across 5 gens; `best_w` has high `gamma_B`/`gamma_d`, low `fallback_mix` (the surrogate rewards locality). `cem_result.json` saved. No ckpt/npy written.

- [ ] **Step 3: Commit**

```bash
git add block_lo_arm_order_network/cem_readout_search.py \
        probe_results_image_large/structure_adaptive/cem/E3_dryrun/cem_result.json \
        probe_results_image_large/structure_adaptive/cem/E3_dryrun/cem_log.txt
git commit -m "structure-adaptive: CEM readout search + dry-run (surrogate) verifying the loop"
```

### Task 2.4: Wire short-continuation confirm fitness (top-K re-rank)

**Files:**
- Modify: `block_lo_arm_order_network/readout_fitness.py`

- [ ] **Step 1: Implement compact in-process short continuation**

Modify `readout_fitness.py`: replace the `short_continuation_fitness` stub with a real implementation that loads the baseline, runs `steps` of continuation feeding orders from `w` (mixing with random per the existing α schedule = fixed α here), then returns `-cross_eval`. Reuse `T.load_baseline_model`, `T._forward_with_block_orders`, `T.get_lr`. Key body:

```python
def short_continuation_fitness(ckpt, A_block_path, data_dir, topology, grid,
                               steps=200, alpha=0.9, lr=1e-4, bs=16, grad_accum=4,
                               device="cuda:0"):
    import pickle
    model, margs, ckptd, ftp, ibp = T.load_baseline_model(ckpt, device)
    opt = model.configure_optimizers(0.1, lr, (0.9, 0.99), device.split(":")[0])
    A = np.load(A_block_path).astype(np.float64); B = build_directed_graph(A)
    source, _, _ = compute_source(B, 0.5); coords, has = ur.make_coords(N, topology, grid)
    meta = pickle.load(open(f"{data_dir}/meta.pkl","rb")); assert int(meta["tokens_per_image"])==SEQ
    tm = np.memmap(f"{data_dir}/train.bin", dtype=np.uint16, mode="r"); ntr = len(tm)//SEQ
    vm = np.memmap(f"{data_dir}/val.bin", dtype=np.uint16, mode="r"); nval=min(512,len(vm)//SEQ)
    val = torch.from_numpy(np.asarray(vm[:nval*SEQ],dtype=np.int64).reshape(nval,SEQ))
    def fitness(w):
        w = ur.clip_params(w)
        # fresh weights each eval would be ideal but costly; reload to avoid contamination
        m2,_,_,ftp2,ibp2 = T.load_baseline_model(ckpt, device)
        o2 = m2.configure_optimizers(0.1, lr, (0.9,0.99), device.split(":")[0]); m2.train()
        rng = np.random.default_rng(0)
        for step in range(1, steps+1):
            o2.zero_grad(set_to_none=True)
            for _ in range(grad_accum):
                idx = rng.integers(0, ntr, size=bs)
                x = torch.from_numpy(np.stack([np.asarray(tm[i*SEQ:(i+1)*SEQ],dtype=np.int64) for i in idx])).to(device)
                if rng.random() > alpha:
                    bo = torch.stack([torch.randperm(N,device=device) for _ in range(bs)])
                else:
                    ob = ur.sample_orders_batch(B, source, coords, w, bs, step, has)
                    bo = torch.as_tensor(ob, dtype=torch.long, device=device)
                loss = T._forward_with_block_orders(m2, x, bo, fixed_token_perm=ftp2, inv_block_perm=ibp2)
                (loss/grad_accum).backward()
            o2.step()
        return -_cross_eval(m2, val, B, source, coords, has, w, ftp2, ibp2, device)
    return fitness
```

- [ ] **Step 2: Smoke (1 candidate, 50 steps) to verify it trains + evals**

Run:
```bash
cd /home/admin/lyuyuhuan/order_lyu
CUDA_VISIBLE_DEVICES=0 python3 - <<'EOF'
import sys; sys.path.insert(0,"block_lo_arm_order_network")
from readout_fitness import short_continuation_fitness
f=short_continuation_fitness(
  "nanogpt-learned-order/out/image_large/imagenet64_vq_f4_800k/seq256/patch2x2_full_baseline_l4h8e256/ckpt.pt",
  "probe_results_image_large/imagenet64_vqf4_full_l4h8e256_patch2x2_control/A_block_8x8.npy",
  "nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch8x8", "grid2d", 8, steps=50)
base=dict(beta_sup=0,beta_dep=0,rho=0,gamma_B=1.5,gamma_d=1.0,tau=0.1,fallback_mix=0)
print("short-continuation fitness (proximity, 50 steps):", round(f(base),4))
EOF
```

Expected: returns a finite fitness (~ −7.2x), no errors. Confirms the confirm-tier harness trains and evals.

- [ ] **Step 3: Commit**

```bash
git add block_lo_arm_order_network/readout_fitness.py
git commit -m "structure-adaptive: short-continuation confirm fitness for CEM top-K re-rank"
```

---

## Phase 3 — Run CEM oracle across the graph set (the decider)

### Task 3.1: CEM oracle orchestration script

**Files:**
- Create: `scripts/run_cem_oracle.sh`

- [ ] **Step 1: Write the orchestration**

Create `scripts/run_cem_oracle.sh`:

```bash
#!/bin/bash
# CEM oracle readout search per graph. frozen-eval fitness (screening).
set -euo pipefail
REPO=/home/admin/lyuyuhuan/order_lyu
CKPT=$REPO/nanogpt-learned-order/out/image_large/imagenet64_vq_f4_800k/seq256/patch2x2_full_baseline_l4h8e256/ckpt.pt
DATA=$REPO/nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch8x8
ROOT=$REPO/probe_results_image_large/structure_adaptive/cem
GPU="${GPU:-0}"; POP="${POP:-16}"; GENS="${GENS:-5}"
cd "$REPO"

run() {  # name a_path topology grid
  CUDA_VISIBLE_DEVICES="$GPU" python block_lo_arm_order_network/cem_readout_search.py \
    --graph-name "$1" --a-path "$2" --topology "$3" --grid "$4" \
    --fitness frozen --ckpt "$CKPT" --val-bin "$DATA/val.bin" --meta "$DATA/meta.pkl" \
    --pop "$POP" --gens "$GENS" --outdir "$ROOT/$1" 2>&1 | tee "$ROOT/$1.log"
}

# image graphs use the image baseline ckpt for frozen fitness; text graph uses dryrun
# (no text ckpt wired here) — text oracle is structural-only via dryrun surrogate.
run E3_ctrl_small  probe_results_image_large/imagenet64_vqf4_full_l4h8e256_patch2x2_control/A_block_8x8.npy grid2d 8
run E2_singletoken probe_results_image/e2_imagenet32_vqf4_seq64/attention/A_global.npy grid2d 8
run E3_large_fixed probe_results_image_large/imagenet64_vqf4_full_l8h8e512_patch2x2_fixed/A_global.npy grid2d 8
echo "[done] CEM oracle (image graphs)"
```

> Note: text-graph oracle and E3-large fitness use the **image** baseline ckpt only where the graph is from that model. E3-large's A is from the l8h8e512 model — its frozen fitness needs the l8h8e512 ckpt; if unavailable, run E3-large with `--fitness dryrun` (structural surrogate) and label it accordingly in the summary. Confirm ckpt availability in Step 2 before trusting E3-large loss-based fitness.

- [ ] **Step 2: Confirm ckpt-graph pairing before running loss-based fitness**

Run:
```bash
cd /home/admin/lyuyuhuan/order_lyu
ls -la nanogpt-learned-order/out/image_large/*/*/*/ckpt.pt 2>/dev/null | grep -iE "l8h8e512|l4h8e256" || echo "check which model ckpts exist"
```

Expected: identify whether an l8h8e512 ckpt exists for E3-large loss-based fitness. If only l4h8e256 exists, run E3-large with `--fitness dryrun` and mark it structural-only. **This protects against the apples-to-oranges trap (using the wrong model's loss for a graph from a different model).**

- [ ] **Step 3: Run CEM oracle for the image graphs**

```bash
cd /home/admin/lyuyuhuan/order_lyu
chmod +x scripts/run_cem_oracle.sh
GPU=0 POP=16 GENS=5 bash scripts/run_cem_oracle.sh
```

Expected: each graph produces `cem/<name>/cem_result.json` with `best_w` + `history`. Per-graph runtime with frozen fitness is modest (pop×gens×~512-token eval); if too slow, lower POP to 12.

- [ ] **Step 4: Commit (results tables only)**

```bash
git add scripts/run_cem_oracle.sh probe_results_image_large/structure_adaptive/cem/*/cem_result.json \
        probe_results_image_large/structure_adaptive/cem/*/cem_log.txt
git commit -m "structure-adaptive: CEM oracle search across graph set (frozen-eval fitness)"
```

### Task 3.2: E3-large hard-question decision + cross-graph oracle summary

**Files:**
- Create: `scripts/cem_oracle_summary.py`

- [ ] **Step 1: Write the summary + decision script**

Create `scripts/cem_oracle_summary.py`:

```python
#!/usr/bin/env python3
"""Aggregate CEM oracle results: best_w per graph, which terms activated, and the
E3-large decision (no-structure fallback vs readout-mismatch)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "block_lo_arm_order_network")
import unified_readout as ur

ROOT = Path("probe_results_image_large/structure_adaptive/cem")
graphs = [d.name for d in ROOT.iterdir() if (d / "cem_result.json").exists()]

def dominant_terms(w):
    weights = {k: w[k] for k in ["beta_sup","beta_dep","rho","gamma_B","gamma_d"]}
    tot = sum(abs(v) for v in weights.values()) + 1e-9
    return {k: round(v/tot, 2) for k, v in sorted(weights.items(), key=lambda kv:-kv[1])}

md = ["# CEM oracle cross-graph summary\n\n",
      "| graph | best_fit | dominant terms (normalized) | fallback_mix | tau |\n|---|--:|---|--:|--:|\n"]
rows = {}
for g in sorted(graphs):
    r = json.load(open(ROOT / g / "cem_result.json")); w = r["best_w"]; rows[g] = r
    md.append(f"| {g} | {r['best_fit']:.4f} | {dominant_terms(w)} | {w['fallback_mix']:.2f} | {w['tau']:.2f} |\n")

md.append("\n## E3-large decision\n")
if "E3_large_fixed" in rows:
    r = rows["E3_large_fixed"]; w = r["best_w"]
    structured = (w["gamma_B"] + w["gamma_d"] + w["rho"]) > 0.6 and w["fallback_mix"] < 0.5
    md.append(
        "- CEM found a structured w* (gamma/rho active, low fallback) → E3-large is NOT "
        "no-structure; Bcov merely mismatched. Diagnostic/readout family needs revisiting.\n"
        if structured else
        "- CEM did NOT find a structured w* (high fallback / no gain) → E3-large is genuinely "
        "no-structure / fallback, consistent with the regime diagnostic.\n")
open(ROOT.parent / "cem_oracle_summary.md", "w").write("".join(md))
print("".join(md)); print(f"wrote {ROOT.parent/'cem_oracle_summary.md'}")
```

- [ ] **Step 2: Run + interpret**

```bash
cd /home/admin/lyuyuhuan/order_lyu
python scripts/cem_oracle_summary.py
```

Expected reads (the scientific payoff):
- `E3_ctrl_small`: best_w has gamma_B/gamma_d dominant, low fallback → CEM auto-recovers the proximity/coverage regime.
- `E2_singletoken`: best_w has high fallback_mix or no fit gain → CEM auto-recovers the no-structure fallback.
- `E3_large_fixed`: the decision branch printed — records whether E3-large is genuinely no-structure or a readout mismatch (this sets the paper's boundary claim).

- [ ] **Step 3: Commit**

```bash
git add scripts/cem_oracle_summary.py probe_results_image_large/structure_adaptive/cem_oracle_summary.md
git commit -m "structure-adaptive: CEM oracle cross-graph summary + E3-large decision"
```

---

## Phase 4 — (Optional) MLP hypernetwork imitation

> Only do this if Phase 3 shows the per-graph `best_w` cluster by regime (proximity vs readiness vs fallback). If there are too few graphs or no clear pattern, STOP and write "CEM/ES suggests learnability; hypernetwork left as future work" — do not force-train an MLP on <6 points.

### Task 4.1: MLP imitation of g(B) → w*

**Files:**
- Create: `block_lo_arm_order_network/mlp_readout_hypernet.py`

- [ ] **Step 1: Write the imitation trainer**

Create `block_lo_arm_order_network/mlp_readout_hypernet.py`:

```python
#!/usr/bin/env python3
"""Tiny MLP imitating CEM oracle: g(B) -> w*. Proof that the regime decision is automatable.
Inputs g(B): [readiness_strength, asymmetry, row_entropy, top1_mass, top4_mass,
              locality_score, p_nbr_le1, directionality]. Targets: normalized w*."""
import json, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, "block_lo_arm_order_network")
import graph_regime_diagnostic as grd
import unified_readout as ur
from directed_graph_policy import build_directed_graph

CEM = Path("probe_results_image_large/structure_adaptive/cem")
FEATS = ["readiness_strength","asymmetry","row_entropy","top1_mass","top4_mass",
         "locality_score","p_nbr_le1","directionality"]
# map graph dir name -> (a_path, topology, grid)
GRAPHS = {n.replace("_image","").replace("_fixed",""): (p, t, g) for n,p,t,g in grd.standard_graphs()}

def build_dataset():
    X, Y, names = [], [], []
    for d in CEM.iterdir():
        rj = d / "cem_result.json"
        if not rj.exists(): continue
        w = json.load(open(rj))["best_w"]
        # find matching graph spec by prefix
        key = next((k for k in GRAPHS if d.name.startswith(k)), None)
        if key is None: continue
        path, topo, grid = GRAPHS[key]
        A = np.load(path).astype(np.float64); diag = grd.diagnose(build_directed_graph(A), topo, grid)
        X.append([diag[f] for f in FEATS]); Y.append([w[k] for k in ur.PARAM_KEYS]); names.append(d.name)
    return np.array(X), np.array(Y), names

def main():
    X, Y, names = build_dataset()
    if len(X) < 6:
        print(f"only {len(X)} graphs with oracle w*; too few to train MLP. "
              "Conclusion: CEM/ES suggests learnability; hypernetwork = future work.")
        return
    Xn = (X - X.mean(0)) / (X.std(0)+1e-6); Yn = (Y - Y.mean(0)) / (Y.std(0)+1e-6)
    Xt, Yt = torch.tensor(Xn,dtype=torch.float32), torch.tensor(Yn,dtype=torch.float32)
    net = nn.Sequential(nn.Linear(len(FEATS),32), nn.ReLU(), nn.Linear(32,len(ur.PARAM_KEYS)))
    opt = torch.optim.Adam(net.parameters(), 1e-2)
    for ep in range(2000):
        opt.zero_grad(); loss = ((net(Xt)-Yt)**2).mean(); loss.backward(); opt.step()
    print(f"final imitation MSE (normalized) = {loss.item():.4f} on {len(X)} graphs")
    pred = net(Xt).detach().numpy()*Y.std(0)+Y.mean(0)
    for i,n in enumerate(names):
        print(f"  {n:22s} pred_w={ {k:round(float(pred[i][j]),2) for j,k in enumerate(ur.PARAM_KEYS)} }")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run**

```bash
cd /home/admin/lyuyuhuan/order_lyu
python block_lo_arm_order_network/mlp_readout_hypernet.py
```

Expected: either "<6 graphs → future work" message, OR a low imitation MSE with predicted w that recovers proximity-heavy for E3, fallback-heavy for E2.

- [ ] **Step 3: Commit**

```bash
git add block_lo_arm_order_network/mlp_readout_hypernet.py
git commit -m "structure-adaptive: optional MLP hypernetwork imitating CEM oracle g(B)->w*"
```

---

## Future (documented, NOT implemented here)
- **GRPO fine-tuning** of `π_φ(w | g(B))`: sample K param candidates per graph, group-relative advantage on short-eval reward, clipped-ratio update. Only after CEM oracle + MLP show a clusterable, optimizable parameter space; reward must use frozen/short-continuation eval (full continuation too expensive/noisy).
- **AttnTNT** per-step learned order controller `π_φ(v_t | B, S_t)`: highest variance/cost; long-term.

---

## Decision matrix (after Phase 3)
| Pattern | Conclusion |
|---|---|
| CEM auto-recovers proximity-w for E3, fallback-w for E2, (dryrun) readiness for text | **Strong:** graph structure induces the optimal readout regime, automatically recoverable → structure-conditioned adaptive readout paper. |
| CEM recovers regimes but E3-large gets a structured w* beating random | E3-large is readout-mismatch, not no-structure → revise diagnostic/readout family boundary; honest limitation. |
| CEM finds no w* better than random/fallback on any image graph | Parameter space not optimizable with this family/fitness → remains diagnostic/proof-of-concept paper; reconsider readout family or fitness. |

## Self-review checklist (applied)
- **Spec coverage:** Phase 0 freeze ✓; Phase 1 threshold sensitivity (1.1) + multi-seed CI/sign-count (1.2) ✓; Phase 2 unified family (2.1) + fitness (2.2/2.4) + CEM (2.3) ✓; Phase 3 oracle run (3.1) + E3-large decision (3.2) ✓; Phase 4 MLP optional (4.1) ✓; GRPO/AttnTNT documented as future ✓.
- **Reordering vs user spec:** user placed "E3-large oracle CEM" in Phase 1; it depends on the CEM machinery, so it is Phase 3 Task 3.2. Intent (E3-large is the boundary-decider) preserved.
- **Coordinate-trap guards:** all model-fed orders go through `_forward_with_block_orders` (inverse_block_perm); Task 3.1 Step 2 guards against using the wrong model ckpt for E3-large.
- **Placeholders:** none — every code/command step has concrete content. `short_continuation_fitness` stub in 2.2 is explicitly completed in 2.4.
- **Open dependency:** Task 1.2 needs the externally-run Round-2 result file path (flagged as a blocker, numbers computed not hard-coded).
- **Commit hygiene:** no ckpt/large-npy committed anywhere.

---

# AMENDMENT 2026-05-21 — Phase 1.5 results: CEM input definition + bug audit

This amendment is binding. It supersedes any conflicting text above and in the deprecated
graph-diversity REPORT. It locks the CEM input spec so Phase 2 cannot reinterpret it.

## A. Deprecation list (specific prior claim → replacing evidence)

These came from the per-sample aggregation bug (contiguous `block_len=4` pooling scrambled
the patch2x2 spatial structure; correct mapping is spatial `token_to_patch_indices`,
16×16 token grid → 8×8 patch grid). All DEPRECATED:

| Deprecated claim | Replaced by (corrected evidence) |
|---|---|
| "E3 per-sample g(B) ≈ random on locality" (p_nbr≤1≈0.06, locality_score≈0) | per-sample E3 **p_nbr≤1 = 0.55 ± 0.06, locality_score = 0.43 ± 0.06** (spatial agg, 300 samples) |
| "regime structure is purely population-level emergent" | structure is **robustly aggregable**: knee at n≈5, ~95% of global by n≈30 (`phase1_5_20260521/aggregation_knee.tsv`) |
| "controller must be checkpoint-level / full-population global B" | **batch-level (n≈30) is sufficient**; per-sample carries signal-with-noise |
| The paper-ready "modality-dependent **sample-level visibility** is asymmetric (text 0.53 vs 0.98, image 0.019 vs 0.77)" narrative | **DELETED in full.** The image "0.019 per-sample" number was a pure artifact. Do not reuse this framing anywhere. |
| Old per-head result (`layer_head_scan/...step20000.tsv`, all heads ≈random) | per-head locality is **head-concentrated**: 5/32 heads (all layer-0) have locality_score>0.5, best 0.80 (`phase1_5_20260521/PER_HEAD_FINDING.md`) |

Corrected Case-A status: STILL HOLDS and is cleaner — all 12 g(B) metrics ratio<0.3, only
e2_small↔e2_large cross the boundary (1/300), and the **locality metrics now also separate
setups** (p_nbr≤1 ratio 0.127), so separation is no longer driven only by noise-amplified
metrics. See the corrected banner in `analyses/graph_diversity_20260521/REPORT.md`.

## B. CEM input — PRECISE specification (do NOT reinterpret in Phase 2)

The default CEM/oracle graph input is:
```
For a fixed batch of N_BATCH = 30 val images:
  per (sample, layer, head): A_token[256,256] in PHYSICAL frame
    (= extract_image_attention_e2 path: feed token data, reveal under random order,
       remap reveal-order back via argsort — gives physical token positions)
  B_input = mean over (sample ∈ batch_30, layer ∈ ALL layers, head ∈ ALL heads) of A_token
  A_block = spatial aggregation of B_input via token_to_patch_indices  (16x16 -> 8x8)
            [the UNIFIED utility diagnose_e3_control_dual_level.aggregate_token_to_block;
             NEVER contiguous block_len=4 for patch2x2]
  B = build_directed_graph(A_block)            # = A_block^T, diag zeroed, NOT row-normalized
```
Explicitly NOT permitted without a new amendment: "all heads but layer-0 only",
"per-layer averaged then concatenated", "single-sample B", "contiguous block_len=4".
N_BATCH=30 is chosen from the knee (≥95% of global locality). Global B (all 500 images)
is the offline upper-reference only.

## C. Bug-audit-ongoing note (BINDING)

> Two configuration/aggregation bugs were found and fixed during Phase 1.5:
> (1) `extract_per_sample_gB.py` used contiguous `block_len=4` pooling for patch2x2 (fixed →
> spatial `token_to_patch_indices`); (2) `layer_head_locality_scan.py` used the same wrong
> pooling AND hardcoded `N_LAYERS=8` for a 4-layer model. Neither was a typo — both were
> silent contamination producing plausible-but-wrong numbers. Audit of token→block mapping
> and model-configuration constants across the codebase is **ongoing**. Any analysis
> predating this audit is conditionally valid until re-verified with the unified spatial
> aggregation utility.

## D. Round-2 Bcov dependency — AUDITED, gate PASSED (with one confirmation needed)

> The Round-2 Bcov empirical core was audited for the aggregation bug (code-reading, Task Y).
> Result: **case (a) — Round-2 uses only the GLOBAL `A_block_8x8.npy`.**
> `scripts/run_round2_e3_5arm.sh` passes `--a-block-path .../A_block_8x8.npy` to
> `block_lo_arm_order_network/train_imagelarge_round2.py`, which at line 400–402 does
> `A_block = np.load(a_block_path); B_real = build_directed_graph(A_block)` and performs
> **no token→block aggregation** (it never re-aggregates per-sample 256-token attention).
> The global A_block was produced with the correct `token_to_patch_indices` (verified:
> p_nbr≤1=0.984). Therefore Round-2 Bcov-vs-control numbers are UNAFFECTED by the bug, and
> CEM keeps its hand-designed reference point.
>
> **Remaining confirmation (does NOT block Phase 2):** the audit covers the *committed*
> Round-2 code. If the external run used a modified copy, confirm it matches. Task 1.2 (the
> multi-seed CI/sign-count consolidation) still needs the result-file path, but the
> aggregation gate for Phase 2 is satisfied.

## E. Optional ablation — REDEFINED (head-selection is a noise-floor test, not a sharpener)

Per-head finding shows global all-head locality (0.783) ≈ the layer-0 local heads (0.80) —
the non-local heads contribute ~neutrally, so all-head averaging is already implicitly
≈ local-head-only. Therefore head-selection cannot meaningfully "sharpen the readout" at
batch scale. Redefine:

> **Ablation A (per-sample regime exposure / noise floor):** compare layer-0-selected B vs
> all-head B at **n=1**. Tests whether head selection materially reduces per-sample noise.
> Expected: layer-0 wins at n=1, gap closes by n≈10, indistinguishable by n≈30.
>
> **Ablation B (NOT recommended in baseline):** layer-0-selected B at n=30 as alternative
> CEM input. Run only if Ablation A shows a large persistent gap. Skip by default.

The earlier "test whether head-selected B sharpens readout (at n=30)" motivation is removed
— the data would dismiss it.

## F. Phase-2 gate status
- Aggregation bug: fixed for per-sample + per-head paths; codebase audit ongoing (C).
- Round-2 reference: audited clean, case (a) (D).
- CEM input: locked (B), N_BATCH=30 all-head spatial-aggregated B.
- **Phase 2 may start** once this amendment is reviewed. Ablation per (E). Task 1.2 CI
  consolidation proceeds in parallel when the result-file path is provided.
