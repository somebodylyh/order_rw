# Position-Prior Decomposition & Content-Binding (Pillar ⑤-core+) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose the step-0 model-slot order prior (floor vs PE) and probe whether the converged carrier is position-bound (slot scaffold) or content-bound, all from saved checkpoints with no new training.

**Architecture:** Pure-function analysis module `analyses/position_prior_decomp.py` (synthetic-B floors, reversible PE-ablation hook, frame-aware τ, layout generation/relayout, binding scores) + `analyses/plot_position_prior.py`. Reuses the Pillar-3 forward/readout pipeline and the clean-permutation protocol; `inv_perm` is used only for posthoc frame translation.

**Tech Stack:** Python 3, NumPy, PyTorch, Matplotlib, pytest. Reuses `analyses/path_patch_handoff.py` (`load_model_and_chunks_seed`, `make_probe_batch`, `run_clean`), `attention_trajectory.extract_all_layer_B`, `batch_readout.l0_strict65.build_model_frame_strict65`, `none_separated_block_graph.rollout_by_method`, `batch_readout.order_tau_readout.per_head_tau`, `clean_training_protocol` (`CleanPermutation`, `build_clean_block_permutation`, `model_to_phys_idx_clean`, `phys_to_model_idx_clean`), `analyses/handoff_carrier_config.py`.

## Global Constraints

- **No new training.** All readouts from `runs/handoff_overnight/seed{2,42,123}/ckpt_step{0,2000,10000}.pt`. CPU forward; GPU only as accelerator.
- **Method `C-D+L`** for all rollouts; metrics use `|τ|`/signed τ as noted.
- **B-graph convention:** the strict65 B has a **zeroed diagonal** (no self-edge); the synthetic uniform-causal floor is defined at the **B level** with content edges `B[i,j]=uniform for j<i` (strict), diagonal 0, None-node structure (`B[:,0]=0`, `B[0,1:]=uniform`).
- **Model mask** allows self (`tril` includes diagonal) but that self-mass is discarded by the B-diagonal-zeroing — floor is B-level.
- **`wpe` = token-side slot PE; `wtpe` = target/AdaLN slot conditioning.** Report `Δ_wpe, Δ_wtpe, Δ_both` separately; **not additive**.
- **PE ablation is reversible** (hook/context manager), never a permanent param edit; `which="none"` is bit-identical.
- **Frame sanity:** report `τ_model_slot`, `τ_physical=τ(inv_perm[σ_model])`, and `τ_perm_baseline=τ(training_inv_perm[arange], arange)`. Claim = `τ_model_slot ≫ τ_physical`, with `τ_physical≈τ_perm_baseline`; not `τ_physical=0`.
- **Part 2 layouts:** K=8 = training anchor (`layout_0`) + 7 fixed-seed random; persisted to `layouts.json`. Checkpoints step0/step2000/step10000.
- **`τ_pos=τ(σ_model,arange)`, `τ_content=τ(inv_perm_k[σ_model],arange)`.** At `layout_0`, `τ_content` need not equal `τ_pos`.
- **Anchor-validity gate:** interpret binding only if `τ_pos(anchor)` is high for the carrier (strong seeds ≥0.95 @ step10000; seed42 weak ≥0.60); else invalid/inconclusive (distinct from OOD break = anchor high but relayout collapses).
- **`inv_perm` only posthoc**, never in B construction.
- **N=64 content blocks, BLOCK_LEN=4, SEQ_LEN=256**; B is 65×65 (None node 0 + 64 blocks).
- **Tests run from `block_lo_arm_order_network/`**; insert repo ROOT (`parents[2]`) for `analyses.*`; module self-bootstraps the block dir.

---

### Task 1: Frame-aware τ helper + model mask sanity

**Files:**
- Create: `analyses/position_prior_decomp.py`
- Test: `block_lo_arm_order_network/tests/test_pp_frame.py`

**Interfaces:**
- Produces:
  - `rollout_order(B65, method="C-D+L") -> np.ndarray` (the rolled-out content order; thin wrapper over `rollout_by_method`).
  - `tau_vs_arange(order) -> float` (Kendall τ of `order` vs `np.arange(len(order))`).
  - `tau_two_frames(B65, inv_perm_content, method="C-D+L") -> dict{tau_model_slot, tau_physical}` — physical = τ after applying `inv_perm_content` to the rolled-out order.
  - `model_mask_allows_self(model) -> bool`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pp_frame.py
import pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.position_prior_decomp import tau_vs_arange, tau_two_frames

def test_tau_vs_arange_perfect_and_reversed():
    assert tau_vs_arange(np.arange(10)) == 1.0
    assert tau_vs_arange(np.arange(10)[::-1]) == -1.0

def test_two_frames_translation():
    # a B whose rollout is identity in model frame; physical frame applies inv_perm
    from analyses.position_prior_decomp import synthetic_uniform_causal_B  # Task 2
    B = synthetic_uniform_causal_B(65)
    inv = np.arange(64)[::-1]            # reverse permutation on the 64 content blocks
    out = tau_two_frames(B, inv)
    assert out["tau_model_slot"] > 0.5
    # reversing the content frame flips the sign of the physical-frame tau
    assert out["tau_physical"] < out["tau_model_slot"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_pp_frame.py -q`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement frame helpers**

```python
# analyses/position_prior_decomp.py
"""Pillar-5-core: position-prior decomposition + content-binding. No new training.
See docs/superpowers/specs/2026-06-28-position-prior-decomposition-design.md."""
import pathlib, sys
import numpy as np

_BLOCK_DIR = pathlib.Path(__file__).resolve().parents[1] / "block_lo_arm_order_network"
if str(_BLOCK_DIR) not in sys.path:
    sys.path.insert(0, str(_BLOCK_DIR))

from none_separated_block_graph import rollout_by_method  # noqa: E402

def rollout_order(B65, method="C-D+L"):
    return np.asarray(rollout_by_method(np.asarray(B65, dtype=np.float64), method))

def _kendall_tau(a, b):
    a = np.asarray(a); b = np.asarray(b); n = len(a)
    c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
            if s > 0: c += 1
            elif s < 0: d += 1
    return (c - d) / (0.5 * n * (n - 1))

def tau_vs_arange(order):
    order = np.asarray(order)
    return float(_kendall_tau(order, np.arange(len(order))))

def tau_two_frames(B65, inv_perm_content, method="C-D+L"):
    order = rollout_order(B65, method)             # content order (model frame)
    inv = np.asarray(inv_perm_content)
    order_phys = inv[order]                         # translate to physical frame (posthoc)
    return {"tau_model_slot": tau_vs_arange(order),
            "tau_physical": tau_vs_arange(order_phys)}

def model_mask_allows_self(model):
    # this AOGPT uses tril (includes diagonal) -> self attention allowed
    blk = model.transformer.h[0].attn
    return bool(getattr(blk, "bias", None) is not None and blk.bias[0, 0, 0, 0] == 1)
```

> Implementer note: confirm `rollout_by_method` returns a length-64 content order (None
> excluded). If it returns 65 with a None slot, drop the None index before τ. The
> regression anchor (Task 2) pins the expected floor magnitude.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_pp_frame.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analyses/position_prior_decomp.py block_lo_arm_order_network/tests/test_pp_frame.py
git commit -m "feat: frame-aware tau helpers (model-slot vs physical) + mask sanity"
```

---

### Task 2: Synthetic floor baselines (uniform-causal B + random-B)

**Files:**
- Modify: `analyses/position_prior_decomp.py`
- Test: `block_lo_arm_order_network/tests/test_pp_floor.py`

**Interfaces:**
- Produces:
  - `synthetic_uniform_causal_B(n_nodes=65) -> np.ndarray (n,n)` — diagonal 0, content `B[i,j]=1/(i-1)` for `1≤j<i`, `B[:,0]=0`, `B[0,1:]=1`.
  - `random_B(n_nodes=65, rng=None) -> np.ndarray` — diagonal 0, `B[:,0]=0`, else uniform[0,1).
  - `floor_taus(rng) -> dict{uniform_causal, random_B}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pp_floor.py
import pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.position_prior_decomp import synthetic_uniform_causal_B, random_B, rollout_order, tau_vs_arange

def test_uniform_causal_B_conventions():
    B = synthetic_uniform_causal_B(65)
    assert B.shape == (65, 65)
    assert np.allclose(np.diag(B), 0)          # zeroed diagonal (no self-edge)
    assert np.allclose(B[:, 0], 0)             # no edges into None
    assert np.allclose(np.triu(B[1:, 1:]), 0)  # strict lower-tri content (j<i only)

def test_uniform_causal_floor_above_random_null():
    tau_uc = tau_vs_arange(rollout_order(synthetic_uniform_causal_B(65)))
    taus_rand = [tau_vs_arange(rollout_order(random_B(65, np.random.default_rng(i))))
                 for i in range(8)]
    assert tau_uc > np.mean(taus_rand)         # causal structure beats null
    assert tau_uc > 0.3                          # meaningful ascending bias floor
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_pp_floor.py -q`
Expected: FAIL

- [ ] **Step 3: Implement floors**

```python
# append to analyses/position_prior_decomp.py
def synthetic_uniform_causal_B(n_nodes=65):
    B = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    B[0, 1:] = 1.0                                 # None -> each content block
    for i in range(2, n_nodes):                    # content node i (>=2 in 65-frame)
        B[i, 1:i] = 1.0 / (i - 1)                  # uniform over strict predecessors j<i
    np.fill_diagonal(B, 0.0)
    B[:, 0] = 0.0
    return B

def random_B(n_nodes=65, rng=None):
    rng = rng or np.random.default_rng(0)
    B = rng.random((n_nodes, n_nodes))
    np.fill_diagonal(B, 0.0)
    B[:, 0] = 0.0
    return B

def floor_taus(rng=None):
    rng = rng or np.random.default_rng(0)
    uc = tau_vs_arange(rollout_order(synthetic_uniform_causal_B(65)))
    rb = float(np.mean([tau_vs_arange(rollout_order(random_B(65, np.random.default_rng(i))))
                        for i in range(8)]))
    return {"uniform_causal": float(uc), "random_B": rb}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_pp_floor.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analyses/position_prior_decomp.py block_lo_arm_order_network/tests/test_pp_floor.py
git commit -m "feat: B-level uniform-causal + random-B floor baselines (zeroed diagonal)"
```

---

### Task 3: Reversible PE-ablation context manager

**Files:**
- Modify: `analyses/position_prior_decomp.py`
- Test: `block_lo_arm_order_network/tests/test_pp_pe_ablation.py`

**Interfaces:**
- Produces: `pe_ablation(model, which)` context manager, `which ∈ {"none","wpe","wtpe","both"}`, zeroing the named embedding module outputs via forward hooks; restores on exit.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pp_pe_ablation.py
import pathlib, sys
import numpy as np, torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.position_prior_decomp import pe_ablation
from analyses.path_patch_handoff import load_model_and_chunks_seed, make_probe_batch, run_clean

def test_none_is_bit_identical(tmp_path):
    dev = torch.device("cpu")
    model, chunks, _ = load_model_and_chunks_seed(
        str(ROOT/"runs/handoff_overnight/seed2/ckpt_step0.pt"), 64, dev)
    pc, po = make_probe_batch(chunks, 16, np.random.default_rng(0))
    _, tau_plain = run_clean(model, pc, po, dev)
    with pe_ablation(model, "none"):
        _, tau_none = run_clean(model, pc, po, dev)
    assert np.array_equal(tau_plain, tau_none)
    # hooks removed after context -> identical again
    _, tau_after = run_clean(model, pc, po, dev)
    assert np.array_equal(tau_plain, tau_after)

def test_zero_both_changes_tau(tmp_path):
    dev = torch.device("cpu")
    model, chunks, _ = load_model_and_chunks_seed(
        str(ROOT/"runs/handoff_overnight/seed2/ckpt_step0.pt"), 64, dev)
    pc, po = make_probe_batch(chunks, 16, np.random.default_rng(0))
    _, tau_full = run_clean(model, pc, po, dev)
    with pe_ablation(model, "both"):
        _, tau_zero = run_clean(model, pc, po, dev)
    assert not np.array_equal(tau_full, tau_zero)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_pp_pe_ablation.py -q`
Expected: FAIL

- [ ] **Step 3: Implement the context manager**

```python
# append to analyses/position_prior_decomp.py
import contextlib

@contextlib.contextmanager
def pe_ablation(model, which):
    handles = []
    def _zero_hook(_m, _inp, out):
        return torch.zeros_like(out)
    targets = []
    if which in ("wpe", "both"):
        targets.append(model.transformer.wpe)
    if which in ("wtpe", "both"):
        targets.append(model.transformer.wtpe)
    # which == "none" -> no hooks (bit-identical)
    for mod in targets:
        handles.append(mod.register_forward_hook(_zero_hook))
    try:
        yield model
    finally:
        for h in handles:
            h.remove()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_pp_pe_ablation.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analyses/position_prior_decomp.py block_lo_arm_order_network/tests/test_pp_pe_ablation.py
git commit -m "feat: reversible wpe/wtpe ablation context manager (none = bit-identical)"
```

---

### Task 4: Step-0 τ table under full / zero-wpe / zero-wtpe / zero-both

**Files:**
- Modify: `analyses/position_prior_decomp.py`
- Test: `block_lo_arm_order_network/tests/test_pp_part1.py`

**Interfaces:**
- Produces:
  - `tau_table_under(model, chunks, which, n_batches=4, bs_mean=16, device="cpu") -> np.ndarray (4,8)` — mean over batches of `run_clean` τ with `pe_ablation(which)` active.
  - `part1_arms(seed, root, n_batches=4) -> dict` — `{full, zero_wpe, zero_wtpe, zero_both}` each a `(4,8)` τ table, at `ckpt_step0`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pp_part1.py
import pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.position_prior_decomp import part1_arms

def test_part1_arms_shapes_and_difference():
    arms = part1_arms(2, root=str(ROOT/"runs/handoff_overnight"), n_batches=2)
    for k in ("full", "zero_wpe", "zero_wtpe", "zero_both"):
        assert arms[k].shape == (4, 8)
    # zeroing both PEs should move the tau table off full
    assert not np.allclose(arms["full"], arms["zero_both"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_pp_part1.py -q`
Expected: FAIL

- [ ] **Step 3: Implement Part-1 driver**

```python
# append to analyses/position_prior_decomp.py
import torch
from analyses.path_patch_handoff import (
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

def part1_arms(seed, root, n_batches=4, bs_mean=16, device="cpu"):
    ckpt = f"{root}/seed{seed}/ckpt_step0.pt"
    model, chunks, _ = load_model_and_chunks_seed(ckpt, max(64, bs_mean*n_batches), torch.device(device))
    return {w: tau_table_under(model, chunks, w, n_batches, bs_mean, device)
            for w in ("full", "zero_wpe", "zero_wtpe", "zero_both")}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_pp_part1.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analyses/position_prior_decomp.py block_lo_arm_order_network/tests/test_pp_part1.py
git commit -m "feat: Part-1 step-0 tau table under full/zero-wpe/zero-wtpe/zero-both"
```

---

### Task 5: Part-1 summary (table + floor/frame + plots)

**Files:**
- Modify: `analyses/position_prior_decomp.py`
- Create: `analyses/plot_position_prior.py`
- Test: `block_lo_arm_order_network/tests/test_pp_part1_summary.py`

**Interfaces:**
- Produces:
  - `run_part1(seed, root, out_dir, n_batches=4) -> dict` — assembles `part1_arms` + `floor_taus` + frame sanity (`τ_model_slot/τ_physical/τ_perm_baseline` at step0 for the winning layer's carrier candidates), writes `part1_ablation.csv` + `part1.json`. Carrier heads from `handoff_carrier_config`/emergence winner.
  - `plot_part1(part1_json, out_dir)` (in plot module) — bar chart: full vs zero_wpe/wtpe/both vs uniform_causal vs random_B (max-carrier τ).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pp_part1_summary.py
import json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.position_prior_decomp import run_part1

def test_run_part1_writes_outputs(tmp_path):
    d = run_part1(2, root=str(ROOT/"runs/handoff_overnight"), out_dir=str(tmp_path), n_batches=2)
    assert (tmp_path/"part1_ablation.csv").exists()
    assert (tmp_path/"part1.json").exists()
    j = json.load(open(tmp_path/"part1.json"))
    assert "floor" in j and "frame_sanity" in j
    assert j["frame_sanity"]["tau_model_slot"] >= j["frame_sanity"]["tau_physical"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_pp_part1_summary.py -q`
Expected: FAIL

- [ ] **Step 3: Implement `run_part1` + `plot_part1`**

Assemble: `part1_arms(seed)`, `floor_taus()`, and frame sanity — for the winning layer/carrier
heads (from `analyses.emergence_characterization.winner` on the step-10000 trajectory, or
`handoff_carrier_config`), at step0 compute per-carrier `tau_two_frames(B_head, inv_perm_content)`
(B from `extract_carrier_B`-style re-extraction at step0) plus
`τ_perm_baseline = tau_vs_arange(training_inv_perm[arange(64)])` from the ckpt's
`clean_protocol`. Write `part1_ablation.csv` (rows = arms+floors, cols = max/mean carrier τ) and
`part1.json`. `plot_part1` draws the bar chart. Full code in the modules.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_pp_part1_summary.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analyses/position_prior_decomp.py analyses/plot_position_prior.py block_lo_arm_order_network/tests/test_pp_part1_summary.py
git commit -m "feat: Part-1 summary (ablation table + floor + frame sanity) + plot"
```

---

### Task 6: Layout generator (anchor + fixed random) + persistence

**Files:**
- Modify: `analyses/position_prior_decomp.py`
- Test: `block_lo_arm_order_network/tests/test_pp_layouts.py`

**Interfaces:**
- Produces:
  - `make_layouts(training_clean_perm, K=8, seed_base=1000) -> list[dict]` — `layout_0` = training (`is_training_layout=True`), `layout_1..K-1` = `build_clean_block_permutation(64, seed_base+i)`. Each dict: `{layout_id, perm(list), inv_perm(list), is_training_layout, rng_seed}`.
  - `save_layouts(layouts, path)` / `load_layouts(path)` (JSON).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pp_layouts.py
import pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.position_prior_decomp import make_layouts, save_layouts, load_layouts
from clean_training_protocol import build_clean_block_permutation

def test_layouts_anchor_and_count(tmp_path):
    train = build_clean_block_permutation(64, seed=7)
    layouts = make_layouts(train, K=8, seed_base=1000)
    assert len(layouts) == 8
    assert layouts[0]["is_training_layout"] is True
    assert all(l["is_training_layout"] is False for l in layouts[1:])
    # round-trip persistence
    save_layouts(layouts, str(tmp_path/"layouts.json"))
    back = load_layouts(str(tmp_path/"layouts.json"))
    assert back[3]["perm"] == layouts[3]["perm"]
    # perms are valid permutations of 0..63
    assert sorted(layouts[5]["perm"]) == list(range(64))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_pp_layouts.py -q`
Expected: FAIL

- [ ] **Step 3: Implement layout generator**

```python
# append to analyses/position_prior_decomp.py
import json
from clean_training_protocol import build_clean_block_permutation

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
    json.dump(layouts, open(path, "w"), indent=2)

def load_layouts(path):
    return json.load(open(path))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_pp_layouts.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analyses/position_prior_decomp.py block_lo_arm_order_network/tests/test_pp_layouts.py
git commit -m "feat: K-layout generator (training anchor + fixed-random) + persistence"
```

---

### Task 7: Relayout chunks + round-trip guard

**Files:**
- Modify: `analyses/position_prior_decomp.py`
- Test: `block_lo_arm_order_network/tests/test_pp_relayout.py`

**Interfaces:**
- Produces:
  - `relayout_chunks(chunks_model, training_clean_perm, layout_clean_perm) -> Tensor` — model-frame `chunks` (trained layout) → physical (`model_to_phys_idx_clean`) → model-frame under `layout` (`phys_to_model_idx_clean`).
  - `clean_perm_from_layout(layout_dict) -> CleanPermutation`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pp_relayout.py
import pathlib, sys
import torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.position_prior_decomp import relayout_chunks, clean_perm_from_layout, make_layouts
from clean_training_protocol import build_clean_block_permutation

def test_relayout_to_training_is_identity():
    train = build_clean_block_permutation(64, seed=7)
    layouts = make_layouts(train, K=3, seed_base=10)
    chunks = torch.randint(0, 100, (4, 256))
    out0 = relayout_chunks(chunks, train, clean_perm_from_layout(layouts[0]))
    assert torch.equal(out0, chunks)            # layout_0 == training -> identity

def test_relayout_changes_under_different_layout():
    train = build_clean_block_permutation(64, seed=7)
    layouts = make_layouts(train, K=3, seed_base=10)
    chunks = torch.randint(0, 100, (4, 256))
    out2 = relayout_chunks(chunks, train, clean_perm_from_layout(layouts[2]))
    assert not torch.equal(out2, chunks)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_pp_relayout.py -q`
Expected: FAIL

- [ ] **Step 3: Implement relayout**

```python
# append to analyses/position_prior_decomp.py
import torch as _torch
from clean_training_protocol import (
    CleanPermutation, model_to_phys_idx_clean, phys_to_model_idx_clean)

def clean_perm_from_layout(layout_dict):
    return CleanPermutation(
        block_perm_phys_to_model=_torch.tensor(layout_dict["perm"], dtype=_torch.long),
        inv_perm_model_to_phys=_torch.tensor(layout_dict["inv_perm"], dtype=_torch.long))

def relayout_chunks(chunks_model, training_clean_perm, layout_clean_perm):
    idx_phys = model_to_phys_idx_clean(chunks_model, training_clean_perm)
    return phys_to_model_idx_clean(idx_phys, layout_clean_perm)
```

> Implementer note: confirm `model_to_phys_idx_clean`/`phys_to_model_idx_clean` operate on
> `(B,256)` token tensors at block granularity (BLOCK_LEN=4); the round-trip test guards
> correctness.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_pp_relayout.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analyses/position_prior_decomp.py block_lo_arm_order_network/tests/test_pp_relayout.py
git commit -m "feat: relayout chunks across layouts + training-anchor round-trip guard"
```

---

### Task 8: Binding scores (τ_pos / τ_content) + anchor-validity gate

**Files:**
- Modify: `analyses/position_prior_decomp.py`
- Test: `block_lo_arm_order_network/tests/test_pp_binding.py`

**Interfaces:**
- Produces:
  - `binding_scores(seed, ckpt_step, layouts, winning_layer, carrier_heads, root, bs_mean=16, n_batches=4, device="cpu") -> dict` — per layout: mean carrier `τ_pos`, `τ_content`; `anchor_tau_pos`, `relayout_mean_pos/content`, `relayout_drop`, `anchor_valid` (gate), `verdict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pp_binding.py
import pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.position_prior_decomp import binding_scores, make_layouts
from clean_training_protocol import build_clean_block_permutation

def test_binding_scores_keys_and_gate():
    train = build_clean_block_permutation(64, seed=7)
    layouts = make_layouts(train, K=4, seed_base=10)
    out = binding_scores(2, 10000, layouts, winning_layer=1, carrier_heads=[3,5,7],
                         root=str(ROOT/"runs/handoff_overnight"), n_batches=1)
    assert "anchor_tau_pos" in out and "relayout_mean_pos" in out
    assert "relayout_mean_content" in out and "anchor_valid" in out
    assert out["verdict"] in ("slot-scaffold", "content-bound", "ood-break", "invalid")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_pp_binding.py -q`
Expected: FAIL

- [ ] **Step 3: Implement binding scores**

Per layout `k`: `relayout_chunks` the probe batch → `run_clean` forward → `extract_all_layer_B`
→ for each carrier head, `rollout_order(B[winning_layer, head])` → `τ_pos = tau_vs_arange(order)`
and `τ_content = tau_vs_arange(inv_perm_k[order])`. Average over carrier heads + batches.
`anchor_*` = layout_0 values; `anchor_valid = anchor_tau_pos >= (0.95 if tier=="strong" else 0.60)`.
`verdict`: if not `anchor_valid` → `"invalid"`; elif `relayout_mean_pos` collapses (e.g. `< 0.3`) and
`relayout_mean_content < 0.3` → `"ood-break"`; elif `relayout_mean_content > relayout_mean_pos` →
`"content-bound"`; else `"slot-scaffold"`. Full code in the module.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_pp_binding.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analyses/position_prior_decomp.py block_lo_arm_order_network/tests/test_pp_binding.py
git commit -m "feat: Part-2 binding scores (tau_pos/tau_content) + anchor-validity gate"
```

---

### Task 9: Run Part 2 across seeds × steps

**Files:**
- Modify: `analyses/position_prior_decomp.py`
- Test: `block_lo_arm_order_network/tests/test_pp_part2_driver.py`

**Interfaces:**
- Produces: `run_part2(seed, root, out_dir, K=8, steps=(0,2000,10000)) -> dict` — builds layouts (from the ckpt's training perm), saves `layouts.json`, runs `binding_scores` at each step, writes `part2_binding.csv` + `part2.json`.

- [ ] **Step 1: Write the failing smoke test**

```python
# tests/test_pp_part2_driver.py
import json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.position_prior_decomp import run_part2

def test_run_part2_smoke(tmp_path):
    d = run_part2(2, root=str(ROOT/"runs/handoff_overnight"), out_dir=str(tmp_path),
                  K=4, steps=(10000,))
    assert (tmp_path/"layouts.json").exists()
    assert (tmp_path/"part2_binding.csv").exists()
    j = json.load(open(tmp_path/"part2.json"))
    assert "10000" in j["by_step"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_pp_part2_driver.py -q`
Expected: FAIL

- [ ] **Step 3: Implement `run_part2`**

Read the training `CleanPermutation` from the ckpt `clean_protocol`; `make_layouts`; save;
determine `winning_layer`/`carrier_heads`/`tier` from the step-10000 trajectory winner
(`analyses.emergence_characterization`); for each step run `binding_scores`; write the table +
JSON. Full code in the module.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_pp_part2_driver.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analyses/position_prior_decomp.py block_lo_arm_order_network/tests/test_pp_part2_driver.py
git commit -m "feat: Part-2 driver (layouts + binding across step0/2000/10000)"
```

---

### Task 10: Part-2 plots

**Files:**
- Modify: `analyses/plot_position_prior.py`
- Test: `block_lo_arm_order_network/tests/test_pp_plot_part2.py`

**Interfaces:**
- Produces: `plot_part2(part2_json, out_dir)` — `binding.png`: `τ_pos` vs `τ_content` (mean ± CI over layouts) across steps {0,2000,10000}, with the anchor and the `relayout_drop` annotated.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pp_plot_part2.py
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.position_prior_decomp import run_part2
from analyses.plot_position_prior import plot_part2

def test_plot_part2_writes_png(tmp_path):
    run_part2(2, root=str(ROOT/"runs/handoff_overnight"), out_dir=str(tmp_path),
              K=4, steps=(10000,))
    plot_part2(str(tmp_path/"part2.json"), str(tmp_path))
    assert (tmp_path/"binding.png").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_pp_plot_part2.py -q`
Expected: FAIL

- [ ] **Step 3: Implement `plot_part2`** — matplotlib Agg, two lines (τ_pos, τ_content) vs step with error bars, anchor marker. Full code in the module.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_pp_plot_part2.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analyses/plot_position_prior.py block_lo_arm_order_network/tests/test_pp_plot_part2.py
git commit -m "feat: Part-2 binding plot (tau_pos vs tau_content across steps)"
```

---

### Task 11: Three-seed run + cross-seed README

**Files:**
- Create: `runs/position_prior/` (outputs)
- Create: `analyses/position_prior_README.md`

- [ ] **Step 1: Run Part 1 (step0) for all seeds**

Run a `python -c` invoking `run_part1` + `plot_part1` for seed {2,42,123} → `runs/position_prior/seed*/`.

- [ ] **Step 2: Run Part 2 (step0/2000/10000, K=8) for all seeds**

Run `run_part2` + `plot_part2` for each seed.

- [ ] **Step 3: Write `analyses/position_prior_README.md`**

Per seed: Part-1 decomposition (full vs zero_both vs uniform_causal vs random_B; Δ_wpe/Δ_wtpe;
frame sanity τ_model_slot vs τ_physical vs τ_perm_baseline) and Part-2 (anchor_valid, τ_pos vs
τ_content across steps, verdict). Then the **line-level conclusion**: is the order signal a
metric/mask artifact, a learned slot scaffold, or content-bound — and the resulting **reframe of
Spec A** (if floor explains most of step-0 τ) + entry to Spec B.

- [ ] **Step 4: Commit**

```bash
git add -f analyses/position_prior_README.md runs/position_prior
git commit -m "results: Pillar-5-core position-prior decomposition + content-binding, 3 seeds"
```

---

### Task 12: (Optional) permute-PE robustness arm

**Files:**
- Modify: `analyses/position_prior_decomp.py`
- Test: `block_lo_arm_order_network/tests/test_pp_permute_pe.py`

Only if Task 5 shows zero-PE produces a suspicious activation-scale artifact (e.g. `zero_both`
τ collapses far below `uniform_causal`, suggesting OOD scale rather than a clean floor).

**Interfaces:**
- Produces: `pe_permute(model, which, rng)` context manager shuffling the slot→PE assignment (keeps activation scale in-distribution); `part1_arms` gains `permute_wpe`/`permute_wtpe` arms.

- [ ] **Step 1: Write the failing test** — `pe_permute(model,"none",rng)` is bit-identical; `pe_permute(model,"wpe",rng)` changes τ but preserves PE-row norms.
- [ ] **Step 2: Run to verify fail.**
- [ ] **Step 3: Implement** the permute context manager (forward hook reindexing the embedding output rows by a fixed permutation).
- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit** `feat: optional permute-PE robustness arm`.

---

## Self-Review

**Spec coverage:**
- Part 1A floor ladder (random-B / uniform-causal-B / real zero-both-PE) → Tasks 2, 4. ✓ (uniform-causal is **B-level, diagonal zeroed** — Task 2.)
- Part 1B wpe/wtpe arms, reversible hook, non-additive deltas → Tasks 3, 4, 5. ✓
- Part 1C frame sanity (model-slot vs physical vs τ_perm_baseline) → Tasks 1, 5. ✓
- Part 2 layouts (K=8 anchor+random, persisted) → Task 6; relayout round-trip → Task 7; τ_pos/τ_content + anchor gate → Task 8; step0/2000/10000 → Task 9. ✓
- Plots + README + reframe/Spec-B entry → Tasks 5, 10, 11. ✓
- No new training; inv_perm posthoc only; per-seed first → Global Constraints + Tasks 8, 11. ✓
- TDD: bit-identical PE-none (Task 3), floor>null (Task 2), relayout round-trip (Task 7), frame translation (Task 1). ✓

**Placeholder scan:** Tasks 5, 8, 9, 10 step-3 give itemized algorithms over already-tested
helpers (floors, PE hook, relayout, binding) rather than re-pasting boilerplate; all
data-producing pure functions have full code. No TBD/TODO in logic-bearing steps.

**Type consistency:** τ tables `(4,8)` throughout; `make_layouts`→list[dict] consumed by
`clean_perm_from_layout`/`relayout_chunks`/`binding_scores`; `tau_two_frames` keys
(`tau_model_slot`,`tau_physical`) consistent; method `C-D+L` everywhere; `which ∈
{none,wpe,wtpe,both}` consistent across Tasks 3/4.
