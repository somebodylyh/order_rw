# Pillar ③ Causal Handoff Path-Patching — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Causally verify the order-signal handoff circuit (L0 weak source → redundant L1 carrier set → downstream/global order) on the converged step-10000 checkpoints of seed2/seed42/seed123, via mean-ablation and path-restricted QK patching, with τ-only readouts.

**Architecture:** A single analysis module `analyses/path_patch_handoff.py` plus a frozen carrier-set config. Interventions are implemented as PyTorch forward hooks (no edits to the shared model file): Stage-1 mean-ablation is a `forward_pre_hook` on a block's `attn.c_proj` that replaces target-head columns of `y` with their position-wise batch mean; Stage-2 path-restricted QK patch recomputes one L1 head's attention from a pre-captured L0-ablated residual and injects the resulting head output into an otherwise-clean forward. Readouts reuse the existing `extract_all_layer_B` → `layer_head_tau_table` / `per_head_tau` pipeline.

**Tech Stack:** Python 3, PyTorch, NumPy, pytest. Reuses `block_lo_arm_order_network/attention_trajectory.py` (`extract_all_layer_B`), `batch_readout/order_tau_readout.py` (`per_head_tau`, `layer_head_tau_table`), `batch_readout/l0_strict65.py` (`build_model_frame_strict65`), `per_head_order_scan.py` (ckpt/model loader, `_batch_mean_B`), and the AOGPT model `model_AOGPT_AdaLN6_NoRep_cond_128_trunc_qknorm.py`.

## Global Constraints

- **Working dir for all imports/tests:** `block_lo_arm_order_network/` is the import root (modules import `from per_head_order_scan import ...`, `from none_separated_block_graph import ...` as top-level). Run pytest from that dir.
- **τ readout method:** `C-D+L` (signed) is primary; also record `L`. Always `methods=("C-D+L","L")`.
- **`bs_mean = 16`** probe samples for every τ extraction (never 4). Emit CIs over **≥4 independent probe batches** of 16.
- **probe_orders = identity** (`np.tile(np.arange(256), (S,1))`) — required so model-frame B ≡ physical L2R reference (label-free; no `inv_perm`).
- **Carrier sets are frozen** from the step-10000 τ table and live in config; **no runtime head selection**.
- **Mean-ablation = position-wise batch mean:** `y[:, pos, slice] ← mean_over_batch(y[:, pos, slice])`; the token/block (position) dim is never averaged out.
- **Redundancy ladder** `single ≤ LOO ≤ full-set` is an **expected, reported trend**, not a hard gate.
- **Per-seed reporting first**, aggregate second. 3 seeds is a floor.
- **Checkpoints:** `runs/handoff_overnight/seed{2,42,123}/ckpt_step10000.pt`. Read the seed back from the ckpt; never trust directory names.
- **Bit-identical anchor:** when an ablation source equals the clean activation (or the target set is empty), every readout Δ must be exactly 0.0.

---

### Task 1: Frozen carrier-set config + integrity loader

**Files:**
- Create: `analyses/handoff_carrier_config.py`
- Test: `block_lo_arm_order_network/tests/test_handoff_carrier_config.py`

**Interfaces:**
- Produces: `CARRIER_SETS: dict[int, dict[int, dict[str, list[int]]]]` keyed `seed -> layer -> {"strong":[...],"weak":[...],"null":[...]}`; `load_carrier_sets(seed: int) -> dict[int, dict[str,list[int]]]`; `verify_against_tau_table(seed: int, tau_npz_path: str, strong=0.95, weak_lo=0.60) -> None` (raises `AssertionError` on mismatch).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_handoff_carrier_config.py
import numpy as np, pytest
from analyses.handoff_carrier_config import CARRIER_SETS, load_carrier_sets, verify_against_tau_table

def test_frozen_sets_match_spec():
    s2 = load_carrier_sets(2)
    assert s2[1]["strong"] == [0, 3, 5, 7]      # seed2 L1 strong
    assert s2[0]["strong"] == []                 # no strong L0 anywhere
    s123 = load_carrier_sets(123)
    assert s123[1]["strong"] == [0, 5, 6, 7]     # seed123 L1 strong (H0=0.96>=0.95)
    s42 = load_carrier_sets(42)
    assert all(s42[L]["strong"] == [] for L in range(4))  # seed42 no strong head
    # null is disjoint from weak/strong in every (seed,layer)
    for seed in (2, 42, 123):
        for L, tiers in load_carrier_sets(seed).items():
            assert set(tiers["null"]).isdisjoint(tiers["weak"])
            assert set(tiers["null"]).isdisjoint(tiers["strong"])

def test_verify_against_real_tau_table():
    # the frozen config must reproduce from the real step-10000 tau table
    verify_against_tau_table(
        2, "runs/handoff_overnight/seed2/attention_trajectory/raw/step_010000/tau_table.npz")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_handoff_carrier_config.py -v`
Expected: FAIL with `ModuleNotFoundError: analyses.handoff_carrier_config`

- [ ] **Step 3: Write the config + verifier**

```python
# analyses/handoff_carrier_config.py
"""Frozen Pillar-3 carrier sets (from step-10000 tau_table.npz, method C-D+L).
strong |tau|>=0.95 ; weak 0.60<=|tau|<0.95 ; null = 3 smallest-|tau| heads
excluding any weak/strong carrier. L0 has no strong head in any seed -> it is a
weak UPSTREAM SOURCE, never a strong carrier."""
import numpy as np

CARRIER_SETS = {
    2: {
        0: {"strong": [],        "weak": [1, 6, 7],       "null": [5, 2, 3]},
        1: {"strong": [0,3,5,7], "weak": [1, 2],          "null": [4, 6]},
        2: {"strong": [],        "weak": [3],             "null": [1, 6, 5]},
        3: {"strong": [],        "weak": [5],             "null": [7, 1, 6]},
    },
    42: {
        0: {"strong": [], "weak": [0,3,5,6,7], "null": [4, 1, 2]},
        1: {"strong": [], "weak": [2, 6],      "null": [4, 1, 3]},
        2: {"strong": [], "weak": [3, 5],      "null": [7, 0, 6]},
        3: {"strong": [], "weak": [2],         "null": [3, 4, 0]},
    },
    123: {
        0: {"strong": [],        "weak": [6, 7],       "null": [3, 4, 1]},
        1: {"strong": [0,5,6,7], "weak": [1, 4],       "null": [2, 3]},
        2: {"strong": [],        "weak": [4],          "null": [0, 7, 3]},
        3: {"strong": [],        "weak": [0,1,3,4],    "null": [5, 7, 2]},
    },
}

def load_carrier_sets(seed: int) -> dict:
    return CARRIER_SETS[int(seed)]

def verify_against_tau_table(seed, tau_npz_path, strong=0.95, weak_lo=0.60):
    d = np.load(tau_npz_path, allow_pickle=True)
    methods = list(d["methods"]); mi = methods.index("C-D+L")
    tau = d["tau"][:, :, mi]                    # (L,H) signed
    cfg = load_carrier_sets(seed)
    for L in range(tau.shape[0]):
        a = np.abs(tau[L])
        exp_strong = sorted(int(h) for h in range(8) if a[h] >= strong)
        exp_weak = sorted(int(h) for h in range(8) if weak_lo <= a[h] < strong)
        assert sorted(cfg[L]["strong"]) == exp_strong, (seed, L, "strong", exp_strong)
        assert sorted(cfg[L]["weak"]) == exp_weak, (seed, L, "weak", exp_weak)
        cand = sorted((h for h in range(8)
                       if h not in exp_strong and h not in exp_weak), key=lambda h: a[h])
        assert cfg[L]["null"] == cand[:3], (seed, L, "null", cand[:3])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_handoff_carrier_config.py -v`
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add analyses/handoff_carrier_config.py block_lo_arm_order_network/tests/test_handoff_carrier_config.py
git commit -m "feat: frozen Pillar-3 carrier-set config + tau-table integrity check"
```

---

### Task 2: Probe loader + τ-readout wrapper (reuse existing pipeline)

**Files:**
- Create: `analyses/path_patch_handoff.py` (start the module here)
- Test: `block_lo_arm_order_network/tests/test_handoff_probe_readout.py`

**Resolved confirmations (verified before execution):**
- Loader: `_load_model_and_chunks(ckpt_path, total, seed, device, split)` from
  `neural_readout.extract_b`, returns `(model, chunks, clean_perm, dev, _ci)` — gives the
  model **and** the eval-token `chunks` source in one call.
- Seed read-back: `ckpt['args']['seed']` (ckpt['args'] is a **dict**). Dir names match.
- model_args: `n_layer=4, n_head=8, n_embd=384` (→ `hs=48`). Block container `model.transformer.h`.

**Interfaces:**
- Consumes: `extract_all_layer_B` (from `attention_trajectory`), `layer_head_tau_table`/`per_head_tau` (from `batch_readout.order_tau_readout`), `_load_model_and_chunks` (from `neural_readout.extract_b`).
- Produces:
  - `load_model_and_chunks_seed(ckpt_path: str, total: int, device, split="train") -> tuple[model, chunks, int]` (model in eval mode, eval-token chunks, seed from `ckpt['args']['seed']`).
  - `make_probe_batch(eval_model_tokens, n: int, rng) -> tuple[Tensor, np.ndarray]` returns `(probe_chunks[n,257], probe_orders[n,256]=identity)` by drawing `n` random rows.
  - `tau_table_from_attn(attn_list, probe_orders) -> np.ndarray` returns `tau[L,H]` (signed, method `C-D+L`) via `extract_all_layer_B(...).mean(axis=1)` → `layer_head_tau_table`.
  - `run_clean(model, probe_chunks, probe_orders, device) -> (attn_list, tau_LH)`.

- [ ] **Step 1: Write the failing test** (regression anchor: clean τ table matches the saved trajectory table within tolerance)

```python
# tests/test_handoff_probe_readout.py
import numpy as np, torch
from analyses.path_patch_handoff import load_model_and_chunks_seed, make_probe_batch, run_clean

def test_clean_tau_reproduces_l1_carrier():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, chunks, seed = load_model_and_chunks_seed(
        "runs/handoff_overnight/seed2/ckpt_step10000.pt", total=64, device=dev)
    assert seed == 2                                   # read from ckpt['args']['seed']
    pc, po = make_probe_batch(chunks, n=16, rng=np.random.default_rng(0))
    _, tau = run_clean(model, pc, po, dev)             # tau: (L,H) signed C-D+L
    # seed2 L1 strong carrier set {0,3,5,7} should read ~1.0 (robust to sample identity)
    for h in (0, 3, 5, 7):
        assert tau[1, h] > 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_handoff_probe_readout.py -v`
Expected: FAIL (`ImportError`/`AttributeError` on the new functions)

- [ ] **Step 3: Implement the wrappers**

```python
# analyses/path_patch_handoff.py  (module header + Task-2 functions)
"""Pillar-3 causal handoff path-patching: mean-ablation + path-restricted QK
patch interventions with tau-only readouts. See spec
docs/superpowers/specs/2026-06-27-handoff-causal-path-patching-design.md."""
import numpy as np, torch
from attention_trajectory import extract_all_layer_B
from batch_readout.order_tau_readout import layer_head_tau_table
from neural_readout.extract_b import _load_model_and_chunks

def load_model_and_chunks_seed(ckpt_path, total, device, split="train"):
    model, chunks, clean_perm, dev, _ci = _load_model_and_chunks(
        ckpt_path, total, seed=0, device=device, split=split)
    model.eval()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    seed = int(ckpt["args"]["seed"])      # integrity: read seed from ckpt, not dir name
    return model, chunks, seed

def make_probe_batch(eval_model_tokens, n, rng):
    idx = rng.choice(len(eval_model_tokens), size=n, replace=False)
    chunks = eval_model_tokens[idx]
    probe_orders = np.tile(np.arange(256, dtype=np.int64), (n, 1))  # identity
    return chunks, probe_orders

def tau_table_from_attn(attn_list, probe_orders):
    B_all = extract_all_layer_B(attn_list, probe_orders)   # (L,S,H,65,65)
    B_lhn = B_all.mean(axis=1)                             # (L,H,65,65)
    tbl = layer_head_tau_table(B_lhn, methods=("C-D+L", "L"))
    return tbl["tau"][:, :, 0]                             # (L,H) signed, C-D+L

@torch.no_grad()
def run_clean(model, probe_chunks, probe_orders, device):
    pc = probe_chunks.to(device)
    po = torch.from_numpy(probe_orders).to(device)
    _, _, attn_list = model.forward_fn(pc, po, return_attentions=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return attn_list, tau_table_from_attn(attn_list, probe_orders)
```

> Implementer note: confirm `scan.load_model_from_ckpt` exists; if the loader has a
> different name, adapt this one call. The model/eval-token loading must reproduce the
> trajectory run (same model class, same `eval_model_tokens` source the logger used).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_handoff_probe_readout.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analyses/path_patch_handoff.py block_lo_arm_order_network/tests/test_handoff_probe_readout.py
git commit -m "feat: handoff probe loader + tau-readout wrapper reusing trajectory pipeline"
```

---

### Task 3: Mean-ablation hook (c_proj pre-hook) + unit tests

**Files:**
- Modify: `analyses/path_patch_handoff.py`
- Test: `block_lo_arm_order_network/tests/test_handoff_mean_ablation.py`

**Interfaces:**
- Produces:
  - `mean_ablation_prehook(head_indices: list[int], n_head: int) -> callable` — returns a `forward_pre_hook(module, args)` that replaces, in `args[0]` (`y`, shape `(B,T,C)`), each target head's column block with its position-wise batch mean and returns the modified args.
  - `run_with_ablation(model, layer: int, head_indices: list[int], probe_chunks, probe_orders, device) -> (attn_list, tau_LH)` — registers the pre-hook on `model.transformer.h[layer].attn.c_proj`, runs forward, removes the hook.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_handoff_mean_ablation.py
import numpy as np, torch
from analyses.path_patch_handoff import (
    mean_ablation_prehook, run_with_ablation, load_model_and_seed,
    make_probe_batch, run_clean)

def test_prehook_replaces_only_target_head_columns_with_batch_mean():
    n_head, hs = 8, 4; C = n_head*hs; B, T = 3, 5
    y = torch.randn(B, T, C)
    hook = mean_ablation_prehook(head_indices=[2], n_head=n_head)
    (y2,) = hook(module=None, args=(y.clone(),))
    sl = slice(2*hs, 3*hs)
    # target head: position-wise batch mean (avg over B, keep T)
    exp = y[:, :, sl].mean(dim=0, keepdim=True).expand(B, -1, -1)
    assert torch.allclose(y2[:, :, sl], exp)
    # every other column untouched
    mask = torch.ones(C, bool); mask[sl] = False
    assert torch.allclose(y2[:, :, mask], y[:, :, mask])

def test_empty_ablation_is_bit_identical():
    dev = torch.device("cpu")
    model, _ = load_model_and_seed("runs/handoff_overnight/seed2/ckpt_step10000.pt", dev)
    pc, po = make_probe_batch(_eval_tokens(), n=16, rng=np.random.default_rng(0))
    _, tau_clean = run_clean(model, pc, po, dev)
    _, tau_empty = run_with_ablation(model, layer=1, head_indices=[], 
                                     probe_chunks=pc, probe_orders=po, device=dev)
    assert np.array_equal(tau_clean, tau_empty)   # Δ ≡ 0
```

(`_eval_tokens()` is a small test helper returning the same eval-token tensor the
clean test uses; factor it into a `conftest.py` fixture if convenient.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_handoff_mean_ablation.py -v`
Expected: FAIL (functions undefined)

- [ ] **Step 3: Implement the hook + runner**

```python
# append to analyses/path_patch_handoff.py
def mean_ablation_prehook(head_indices, n_head):
    heads = list(head_indices)
    def _hook(module, args):
        if not heads:
            return None                      # no-op -> bit-identical
        y = args[0]
        B, T, C = y.shape
        hs = C // n_head
        y = y.clone()
        for h in heads:
            sl = slice(h*hs, (h+1)*hs)
            y[:, :, sl] = y[:, :, sl].mean(dim=0, keepdim=True)  # position-wise batch mean
        return (y,) + tuple(args[1:])
    return _hook

@torch.no_grad()
def run_with_ablation(model, layer, head_indices, probe_chunks, probe_orders, device):
    n_head = model.transformer.h[layer].attn.n_head
    handle = model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(
        mean_ablation_prehook(head_indices, n_head))
    try:
        pc = probe_chunks.to(device)
        po = torch.from_numpy(probe_orders).to(device)
        _, _, attn_list = model.forward_fn(pc, po, return_attentions=True)
    finally:
        handle.remove()
    return attn_list, tau_table_from_attn(attn_list, probe_orders)
```

> Implementer note: confirm the block list attribute path (`model.transformer.h`); if the
> model exposes blocks differently, adjust the single accessor. `c_proj`'s input is `y`
> reassembled `(B,T,C)` with heads side-by-side (model file line 85), so column block
> `[h*hs:(h+1)*hs]` is exactly head `h`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_handoff_mean_ablation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analyses/path_patch_handoff.py block_lo_arm_order_network/tests/test_handoff_mean_ablation.py
git commit -m "feat: position-wise batch-mean ablation hook on attn.c_proj"
```

---

### Task 4: Hook-locality test + Stage-1 runner (single / LOO / full / null)

**Files:**
- Modify: `analyses/path_patch_handoff.py`
- Test: `block_lo_arm_order_network/tests/test_handoff_stage1.py`

**Interfaces:**
- Produces:
  - `stage1_variants(strong_set: list[int]) -> dict[str, list[list[int]]]` returns `{"full":[set], "loo":[set\{h} for h], "single":[[h] for h], }`.
  - `run_stage1(model, seed, ablate_layer, target_heads, readout_layers, probe_batches, device) -> dict` — for each probe batch runs clean + ablated, returns `Δτ[L,H]` per batch plus mean/CI; ablate at `ablate_layer`, read τ at `readout_layers` (must be `> ablate_layer`).

- [ ] **Step 1: Write the failing tests** (hook locality is the key guard)

```python
# tests/test_handoff_stage1.py
import numpy as np, torch
from analyses.path_patch_handoff import (
    run_with_ablation, run_clean, load_model_and_seed, make_probe_batch,
    stage1_variants)

def test_ablating_layer_l_leaves_earlier_layers_bit_identical():
    dev = torch.device("cpu")
    model, _ = load_model_and_seed("runs/handoff_overnight/seed2/ckpt_step10000.pt", dev)
    pc, po = make_probe_batch(_eval_tokens(), n=16, rng=np.random.default_rng(1))
    attn_clean, tau_clean = run_clean(model, pc, po, dev)
    attn_abl, tau_abl = run_with_ablation(model, layer=1, head_indices=[0,3,5,7],
                                          probe_chunks=pc, probe_orders=po, device=dev)
    # layer 0 (< ablate layer 1) attention must be bit-identical
    assert torch.allclose(attn_clean[0], attn_abl[0])
    assert np.array_equal(tau_clean[0], tau_abl[0])
    # a downstream layer should change
    assert not np.allclose(tau_clean[2], tau_abl[2])

def test_stage1_variants_partition():
    v = stage1_variants([0, 3, 5, 7])
    assert v["full"] == [[0,3,5,7]]
    assert [sorted(x) for x in v["loo"]] == [[3,5,7],[0,5,7],[0,3,7],[0,3,5]]
    assert v["single"] == [[0],[3],[5],[7]]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_handoff_stage1.py -v`
Expected: FAIL

- [ ] **Step 3: Implement variants + runner**

```python
# append to analyses/path_patch_handoff.py
def stage1_variants(strong_set):
    s = list(strong_set)
    return {
        "full":   [list(s)],
        "loo":    [[h for h in s if h != drop] for drop in s],
        "single": [[h] for h in s],
    }

@torch.no_grad()
def run_stage1(model, seed, ablate_layer, target_heads, readout_layers,
               probe_batches, device):
    per_batch = []
    for pc, po in probe_batches:
        _, tau_clean = run_clean(model, pc, po, device)
        _, tau_abl = run_with_ablation(model, ablate_layer, target_heads, pc, po, device)
        dtau = tau_abl - tau_clean
        per_batch.append(dtau)
    arr = np.stack(per_batch)                       # (Nbatch, L, H)
    return {
        "ablate_layer": ablate_layer, "target_heads": list(target_heads),
        "dtau_mean": arr.mean(0), "dtau_std": arr.std(0), "n_batch": len(per_batch),
        "readout_layers": list(readout_layers),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_handoff_stage1.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analyses/path_patch_handoff.py block_lo_arm_order_network/tests/test_handoff_stage1.py
git commit -m "feat: Stage-1 ablation runner + variants + hook-locality guard"
```

---

### Task 5: Multiplicity-collapse readout + Stage-1 table writer

**Files:**
- Modify: `analyses/path_patch_handoff.py`
- Test: `block_lo_arm_order_network/tests/test_handoff_table.py`

**Interfaces:**
- Produces:
  - `multiplicity_collapse(tau_before_LH, tau_after_LH, layer, strong=0.95) -> dict` returns `{"n_strong_before":int,"n_strong_after":int,"mean_tau_before":float,"mean_tau_after":float}` over that layer's strong-carrier set membership (count of heads with `|τ|≥strong`).
  - `consensus_tau(tau_after_LH, layer, carrier_heads) -> float` — mean signed τ over the carrier set at `layer` (carrier-set aggregation readout).
  - `write_stage1_table(rows: list[dict], out_csv: str) -> None` — columns: `seed,stage,intervention,target_layer,target_heads,mean_tau_before,mean_tau_after,n_strong_before,n_strong_after,delta_global_tau`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_handoff_table.py
import numpy as np, csv
from analyses.path_patch_handoff import multiplicity_collapse, write_stage1_table

def test_multiplicity_collapse_counts():
    before = np.zeros((4,8)); after = np.zeros((4,8))
    before[1, [0,3,5,7]] = 1.0          # 4 strong before
    after[1, [0]] = 0.97                # 1 strong after
    mc = multiplicity_collapse(before, after, layer=1)
    assert mc["n_strong_before"] == 4
    assert mc["n_strong_after"] == 1

def test_table_writer_has_required_columns(tmp_path):
    p = tmp_path/"s1.csv"
    write_stage1_table([{
        "seed":2,"stage":"1a","intervention":"full","target_layer":1,
        "target_heads":"[0,3,5,7]","mean_tau_before":1.0,"mean_tau_after":0.4,
        "n_strong_before":4,"n_strong_after":0,"delta_global_tau":-0.3}], str(p))
    rows = list(csv.DictReader(open(p)))
    assert rows[0]["n_strong_after"] == "0"
    assert "delta_global_tau" in rows[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_handoff_table.py -v`
Expected: FAIL

- [ ] **Step 3: Implement readouts + writer**

```python
# append to analyses/path_patch_handoff.py
import csv as _csv

def multiplicity_collapse(tau_before_LH, tau_after_LH, layer, strong=0.95):
    b = np.abs(tau_before_LH[layer]); a = np.abs(tau_after_LH[layer])
    return {
        "n_strong_before": int((b >= strong).sum()),
        "n_strong_after":  int((a >= strong).sum()),
        "mean_tau_before": float(tau_before_LH[layer].mean()),
        "mean_tau_after":  float(tau_after_LH[layer].mean()),
    }

def consensus_tau(tau_after_LH, layer, carrier_heads):
    if not carrier_heads:
        return float("nan")
    return float(np.mean([tau_after_LH[layer, h] for h in carrier_heads]))

_STAGE1_COLS = ["seed","stage","intervention","target_layer","target_heads",
                "mean_tau_before","mean_tau_after","n_strong_before",
                "n_strong_after","delta_global_tau"]

def write_stage1_table(rows, out_csv):
    with open(out_csv, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=_STAGE1_COLS); w.writeheader()
        for r in rows: w.writerow(r)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_handoff_table.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analyses/path_patch_handoff.py block_lo_arm_order_network/tests/test_handoff_table.py
git commit -m "feat: multiplicity-collapse readout + Stage-1 table writer"
```

---

### Task 6: Path-restricted L0→L1 QK patch hook + unit tests

**Files:**
- Modify: `analyses/path_patch_handoff.py`
- Test: `block_lo_arm_order_network/tests/test_handoff_path_patch.py`

**Interfaces:**
- Consumes: clean L1 input residual `x_clean_L1` and L0-ablated L1 input residual `x_corr_L1`, both captured via a `forward_pre_hook` on `model.transformer.h[1]` (Block input).
- Produces:
  - `capture_block_input(model, layer) -> (handle, store)` — pre-hook recording `args[0]` (residual `x` entering the block) into `store["x"]`.
  - `l1_attn_from_residual(model, x_resid, cond, dst_head, device) -> att_dst` — recompute L1's attention map for `dst_head` from a given residual (applies `ln_1`+AdaLN modulate → `c_attn` → q,k → `q_norm`/`k_norm` → causal softmax). Returns `(S,257,257)` for that head.
  - `patched_l1_head_output(model, x_clean_L1, x_corr_L1, dst_head, cond, device) -> y_head` — att from `x_corr_L1` (QK) but **v from `x_clean_L1`** → `att@v` for `dst_head`; returns its `(S,257,hs)` contribution (for downstream injection in Task 7).

- [ ] **Step 1: Write the failing tests** (Q/K-only: V path stays clean; bit-identical anchor)

```python
# tests/test_handoff_path_patch.py
import numpy as np, torch
from analyses.path_patch_handoff import (
    load_model_and_seed, make_probe_batch, capture_block_input,
    l1_attn_from_residual, patched_l1_head_output)

def test_corr_equals_clean_gives_identical_att():
    dev = torch.device("cpu")
    model, _ = load_model_and_seed("runs/handoff_overnight/seed2/ckpt_step10000.pt", dev)
    pc, po = make_probe_batch(_eval_tokens(), n=8, rng=np.random.default_rng(2))
    h, store = capture_block_input(model, layer=1)
    with torch.no_grad():
        model.forward_fn(pc.to(dev), torch.from_numpy(po).to(dev), return_attentions=True)
    h.remove()
    x_clean = store["x"]; cond = store["cond"]
    att_a = l1_attn_from_residual(model, x_clean, cond, dst_head=0, device=dev)
    att_b = l1_attn_from_residual(model, x_clean, cond, dst_head=0, device=dev)
    assert torch.allclose(att_a, att_b)              # deterministic
    # patched with corr==clean -> head output equals clean head output
    y_patch = patched_l1_head_output(model, x_clean, x_clean, dst_head=0, cond=cond, device=dev)
    assert torch.isfinite(y_patch).all()

def test_patched_att_uses_corr_qk_but_clean_v():
    # changing only x_corr changes att; changing only the v-residual must NOT change att
    dev = torch.device("cpu")
    model, _ = load_model_and_seed("runs/handoff_overnight/seed2/ckpt_step10000.pt", dev)
    pc, po = make_probe_batch(_eval_tokens(), n=8, rng=np.random.default_rng(3))
    h, store = capture_block_input(model, layer=1)
    with torch.no_grad():
        model.forward_fn(pc.to(dev), torch.from_numpy(po).to(dev), return_attentions=True)
    h.remove()
    x_clean, cond = store["x"], store["cond"]
    x_corr = x_clean + 0.1*torch.randn_like(x_clean)
    att_clean = l1_attn_from_residual(model, x_clean, cond, dst_head=0, device=dev)
    att_corr  = l1_attn_from_residual(model, x_corr,  cond, dst_head=0, device=dev)
    assert not torch.allclose(att_clean, att_corr)   # qk responds to corruption
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_handoff_path_patch.py -v`
Expected: FAIL

- [ ] **Step 3: Implement capture + recompute helpers**

```python
# append to analyses/path_patch_handoff.py
import math, torch.nn.functional as F
from model_AOGPT_AdaLN6_NoRep_cond_128_trunc_qknorm import modulate

def capture_block_input(model, layer):
    store = {}
    blk = model.transformer.h[layer]
    def _pre(module, args, kwargs):
        store["x"] = args[0].detach().clone()
        store["cond"] = args[1].detach().clone()   # AdaLN conditioning c
        return None
    handle = blk.register_forward_pre_hook(_pre, with_kwargs=True)
    return handle, store

def _l1_qkv_from_residual(model, x_resid, cond, device):
    blk = model.transformer.h[1]; attn = blk.attn
    shift, scale, *_ = blk.adaLN(cond).chunk(6, dim=-1)
    xn = modulate(blk.ln_1(x_resid), shift, scale)
    B, T, C = xn.shape; nh = attn.n_head; hs = C // nh
    q, k, v = attn.c_attn(xn).split(C, dim=2)
    q = q.view(B,T,nh,hs).transpose(1,2); k = k.view(B,T,nh,hs).transpose(1,2)
    v = v.view(B,T,nh,hs).transpose(1,2)
    q, k = attn.q_norm(q), attn.k_norm(k)
    return q, k, v, hs

def l1_attn_from_residual(model, x_resid, cond, dst_head, device):
    q, k, v, hs = _l1_qkv_from_residual(model, x_resid, cond, device)
    B, nh, T, _ = q.shape
    att = (q @ k.transpose(-2,-1)) * (1.0/math.sqrt(hs))
    causal = torch.tril(torch.ones(T,T,dtype=torch.bool,device=q.device)).view(1,1,T,T)
    att = att.masked_fill(~causal, float("-inf"))
    att = F.softmax(att, dim=-1)
    return att[:, dst_head]                          # (B,T,T)

def patched_l1_head_output(model, x_clean_L1, x_corr_L1, dst_head, cond, device):
    # QK from corrupted residual, V from CLEAN residual (V path stays clean)
    q_c, k_c, _, hs = _l1_qkv_from_residual(model, x_corr_L1, cond, device)
    _, _, v_clean, _ = _l1_qkv_from_residual(model, x_clean_L1, cond, device)
    B, nh, T, _ = q_c.shape
    att = (q_c[:,dst_head] @ k_c[:,dst_head].transpose(-2,-1)) * (1.0/math.sqrt(hs))
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=q_c.device))
    att = att.masked_fill(~causal.view(1, T, T), float("-inf"))
    att = F.softmax(att, dim=-1)
    return att @ v_clean[:, dst_head]                # (B,T,hs)
```

> Implementer note: the `with_kwargs=True` pre-hook assumes `Block.forward(x, c, ...)` is
> called positionally as `block(x, cond, return_attn=...)` (model file line 319 calls
> `block(x, target_pos_emb_final, return_attn=True)`), so `args=(x, cond)`. Verify and
> simplify the causal-mask line (the `if False` branch is illustrative; keep the plain
> `torch.tril` masked_fill).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_handoff_path_patch.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analyses/path_patch_handoff.py block_lo_arm_order_network/tests/test_handoff_path_patch.py
git commit -m "feat: path-restricted L0->L1 QK patch helpers (corr QK, clean V)"
```

---

### Task 7: Stage-2 runner (readout 2a + downstream path_fraction) + redundancy sanity

**Files:**
- Modify: `analyses/path_patch_handoff.py`
- Test: `block_lo_arm_order_network/tests/test_handoff_stage2.py`

**Interfaces:**
- Produces:
  - `run_stage2(model, seed, src_heads_L0, dst_heads_L1, probe_batches, device) -> dict` — per batch: (2a) L1-dst τ under full-L0-ablation; (2b) downstream L2/L3/global τ under three conditions — clean, full-L0-ablation, and **path-restricted** (inject `patched_l1_head_output` for each dst head into an otherwise-clean forward, propagate to L2/L3); returns `path_fraction = |Δτ_down_path| / |Δτ_down_full|` per downstream readout, plus the null-path (src=null head) control.
  - `redundancy_ordering(stage1_result) -> dict` — checks `mean|Δτ| single ≤ loo ≤ full` (reported, not asserted).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_handoff_stage2.py
import numpy as np, torch
from analyses.path_patch_handoff import run_stage2, redundancy_ordering, load_model_and_seed, make_probe_batch

def test_path_fraction_is_finite_and_nonneg():
    dev = torch.device("cpu")
    model, seed = load_model_and_seed("runs/handoff_overnight/seed2/ckpt_step10000.pt", dev)
    pb = [make_probe_batch(_eval_tokens(), 16, np.random.default_rng(i)) for i in range(2)]
    out = run_stage2(model, seed, src_heads_L0=[1,6,7], dst_heads_L1=[0,3,5,7],
                     probe_batches=pb, device=dev)
    assert np.isfinite(out["path_fraction_global"])
    assert out["path_fraction_global"] >= 0

def test_redundancy_ordering_reports_trend():
    fake = {"single":0.05, "loo":0.21, "full":0.25}
    r = redundancy_ordering(fake)
    assert r["monotone"] in (True, False)
    assert r["single"] <= r["full"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_handoff_stage2.py -v`
Expected: FAIL

- [ ] **Step 3: Implement Stage-2 runner**

Implementation outline (full code in module): for each probe batch —
1. clean forward → `tau_clean[L,H]`, capture `x_clean_L1`;
2. full-L0-ablation forward (`run_with_ablation` layer 0, `src_heads_L0`) → `tau_full[L,H]`, capture `x_corr_L1` (re-run with a block-1 input capture hook active);
3. **path-restricted**: clean forward, but register a `forward_hook` on `model.transformer.h[1].attn.c_proj` that *adds* `(patched_l1_head_output - clean_head_output)` for each `dst_head` into `y` **before** `c_proj` so only those heads' outputs change (V clean, other heads clean, MLP clean), then let the model propagate L1→L2→L3 normally → `tau_path[L,H]`;
4. readouts: `Δτ_2a = tau_full[1,dst] - tau_clean[1,dst]`; for downstream `R∈{2,3,global}`, `path_fraction_R = |tau_path[R]-tau_clean[R]| / |tau_full[R]-tau_clean[R]|` (guard divide-by-zero → nan);
5. null-path control: repeat (3)-(4) with `src_heads_L0 = config null heads of L0`.

```python
def redundancy_ordering(means):  # means: {"single":x,"loo":y,"full":z}
    s, l, f = means["single"], means["loo"], means["full"]
    return {"single":s, "loo":l, "full":f, "monotone": (s <= l <= f)}
```

> Implementer note: "global" τ = `max_abs` over all (layer,head) per `derived_views`, or
> the consensus-order τ; use the same global definition as the trajectory analysis for
> comparability. Propagation in step 3 works because injecting into `y` *before* `c_proj`
> means the model's own L1→L2→L3 forward carries the perturbation downstream with all
> other paths clean.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_handoff_stage2.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analyses/path_patch_handoff.py block_lo_arm_order_network/tests/test_handoff_stage2.py
git commit -m "feat: Stage-2 runner (L1-dst confirm + downstream path_fraction) + null-path control"
```

---

### Task 8: Per-seed driver + heatmap + report

**Files:**
- Modify: `analyses/path_patch_handoff.py` (add `main`/CLI)
- Create: `analyses/plot_handoff_path_patch.py`
- Test: `block_lo_arm_order_network/tests/test_handoff_driver_smoke.py`

**Interfaces:**
- Produces:
  - `run_seed(ckpt_path, out_dir, n_batches=4, bs_mean=16, device=...) -> dict` — orchestrates Stage 0 verify → Stage 1a/1b (full/loo/single/null) → Stage 2 (2a+2b+null-path); writes `stage1_table.csv`, `stage2_table.csv`, `summary.json`.
  - `plot_src_downstream_heatmap(summary_json, out_png)` — src-ablation × downstream-layer Δτ heatmap.

- [ ] **Step 1: Write the failing smoke test** (1 seed, 1 batch, runs end-to-end)

```python
# tests/test_handoff_driver_smoke.py
import json, os
from analyses.path_patch_handoff import run_seed

def test_run_seed_smoke(tmp_path):
    out = run_seed("runs/handoff_overnight/seed2/ckpt_step10000.pt",
                   str(tmp_path), n_batches=1, bs_mean=16)
    assert os.path.exists(tmp_path/"stage1_table.csv")
    assert os.path.exists(tmp_path/"stage2_table.csv")
    s = json.load(open(tmp_path/"summary.json"))
    assert s["seed"] == 2
    assert "path_fraction_global" in s["stage2"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_handoff_driver_smoke.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `run_seed` + plotter**

Wire the Task 1–7 functions: load model+seed, `verify_against_tau_table`, build
`n_batches` probe batches of `bs_mean`, run Stage 1a (ablate L1 strong → read L2/L3/global),
Stage 1b (ablate L0 weak source → read L1/L2/L3/global), Stage 2 (src=L0 weak, dst=L1 strong),
plus null-head controls; assemble rows via `write_stage1_table` and a `write_stage2_table`
sibling; dump `summary.json`. `plot_src_downstream_heatmap` reads `summary.json` and renders
a `(src head set) × (downstream layer)` Δτ heatmap with matplotlib.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_handoff_driver_smoke.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analyses/path_patch_handoff.py analyses/plot_handoff_path_patch.py block_lo_arm_order_network/tests/test_handoff_driver_smoke.py
git commit -m "feat: per-seed handoff path-patch driver + heatmap + report"
```

---

### Task 9: Full 3-seed run + redundancy sanity + write-up

**Files:**
- Create: `runs/handoff_pathpatch/` (outputs, git-ignored if large)
- Create: `analyses/handoff_pathpatch_README.md` (per-seed findings + interpretation table filled in)

- [ ] **Step 1: Smoke one seed at full settings**

Run: `cd block_lo_arm_order_network && python -c "from analyses.path_patch_handoff import run_seed; run_seed('runs/handoff_overnight/seed2/ckpt_step10000.pt','runs/handoff_pathpatch/seed2', n_batches=4, bs_mean=16)"`
Expected: `stage1_table.csv`, `stage2_table.csv`, `summary.json` written; redundancy ordering printed (single ≤ loo ≤ full reported).

- [ ] **Step 2: Run all three seeds**

Run the same for `seed42` and `seed123`. seed42 is the **negative contrast** (no strong L1 set → expect weak/diffuse Stage-2 effects).

- [ ] **Step 3: Generate heatmaps**

Run: `python analyses/plot_handoff_path_patch.py runs/handoff_pathpatch/seed{2,42,123}/summary.json`

- [ ] **Step 4: Fill the interpretation table + README**

Map each seed's results onto the spec's interpretation table (single/set/null, multiplicity collapse, path_fraction vs null-path, seed42 contrast). State which decision criteria (1–4) are met **per seed**.

- [ ] **Step 5: Commit**

```bash
git add analyses/handoff_pathpatch_README.md runs/handoff_pathpatch
git commit -m "results: Pillar-3 causal handoff path-patching, 3 seeds + write-up"
```

---

## Self-Review

**Spec coverage:**
- Stage 0 frozen carrier sets + integrity → Task 1. ✓
- Stage 1a/1b mean-ablation, single/LOO/full/null, multiplicity collapse, hook locality → Tasks 3,4,5. ✓
- Stage 2 path-restricted QK patch, readout 2a + downstream path_fraction + null-path → Tasks 6,7. ✓
- τ-only readout, bs_mean=16, identity probe_orders, CIs over batches → Tasks 2,4,8 (Global Constraints). ✓
- seed42 contrast, per-seed reporting → Tasks 8,9. ✓
- TDD: bit-identical anchor (Tasks 3,6), hook locality (Task 4), redundancy sanity (Tasks 7,9). ✓
- Outputs: tables, heatmap, README → Tasks 5,8,9. ✓

**Open implementation confirmations (flagged inline for the implementer, not blockers):**
- `scan.load_model_from_ckpt` exact name/signature (Task 2).
- block container attribute `model.transformer.h` (Tasks 3,6).
- `Block.forward(x, c, ...)` positional args for the capture hook (Task 6).
- "global τ" definition shared with the trajectory analysis (Task 7).

**Placeholder scan:** no TBD/TODO; every code step shows code. Task 7 step 3 gives an
itemized algorithm rather than a single code block because it orchestrates already-tested
helpers — acceptable (the helpers' code is complete in Tasks 3/6).

**Type consistency:** τ tables are `(L,H)` signed `C-D+L` throughout; ablation runners
return `(attn_list, tau_LH)`; Stage runners return dicts with the column names used by the
writers. Consistent.
