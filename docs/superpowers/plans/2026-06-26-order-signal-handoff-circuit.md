# Order-Signal Handoff Circuit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record, during training from scratch (3 seeds, 0→10k), the per-(layer,head) order signal τ and the candidate cross-layer composition scores, so we can map whether the order-bearing carrier moves across layers over training or is seed-locked, and nominate handoff edges.

**Architecture:** Two new pure modules (`order_tau_readout.py` for model-frame B→τ, `attn_composition.py` for weight-based composition) are TDD'd in isolation, then wired into the existing `AttentionTrajectoryLogger.log_snapshot` (extended from L0-only to all-layer). Training uses the existing `train_clean_aogpt` with `--eval-interval 200`; a launcher runs 3 seeds after an overhead-calibration gate. An offline analysis script produces flow heatmaps and handoff-timing curves.

**Tech Stack:** Python, numpy, torch, pytest. Reuses `none_separated_block_graph`, `batch_readout.l0_strict65`, `per_head_order_scan`.

## Global Constraints

- **Label-free boundary:** construction stays in model-frame; no `inv_perm`/`clean_perm`/`block_perm` enters graph building. τ is a posthoc comparison of σ_model against `np.arange` (valid only because probe_orders = identity). Copy this invariant into every τ-related test.
- **bs_mean:** `16` for the final map; never default to `4`. Fallback under overhead = dense `8` + 1000-step anchor `16`.
- **Seeds:** `[2, 42, 123]`, from-scratch (seed *values*, not historical lineage ckpts). Read true `--seed` back from each ckpt.
- **record_steps:** `[0, 200, 400, …, 10000]` (51 points, includes step 0).
- **ckpt_save_steps:** `[0, 1000, …, 10000]`.
- **composition_pairs:** all `i<j` by default; fallback adjacent-only (must keep `L1→L2`). composition is cheapest — cut last.
- **layers=0..3, heads=0..7, methods={C-D+L, L}.**
- **Overhead budget:** target ≤5%, hard cap ≤10% of training wall-clock.
- **Naming:** composition outputs are "candidate composition", never "handoff evidence".
- Run all `pytest` from repo root with `PYTHONPATH=block_lo_arm_order_network`.

---

### Task 1: Per-(layer,head) order-τ readout module

**Files:**
- Create: `block_lo_arm_order_network/batch_readout/order_tau_readout.py`
- Test: `block_lo_arm_order_network/tests/test_order_tau_readout.py`

**Interfaces:**
- Consumes: `none_separated_block_graph.rollout_by_method(B65, method)`, `.discovery_metrics(order)`, `.classify_gate_status(metrics, destroyed_abs_tau_mean)`.
- Produces:
  - `per_head_tau(B65: np.ndarray, method: str) -> dict` — runs rollout+metrics on one model-frame `(65,65)` B; returns `{"tau_vs_l2r", "phys0_rank", "prefix8_overlap", "first_is_phys0"}`.
  - `layer_head_tau_table(B_lhn: np.ndarray, methods=("C-D+L","L")) -> dict` — input `(L, H, 65, 65)` batch-mean model-frame B; returns `{"tau": np.ndarray (L,H,M), "phys0_rank": (L,H,M), "prefix8": (L,H,M), "methods": list}` where `M=len(methods)`.
  - `derived_views(table: dict) -> dict` — returns `{"best_head_per_layer", "best_head_global", "max_abs_tau_per_layer", "max_signed_tau_per_layer", "strong_pass_count_per_layer"}`, each keyed by method index.

- [ ] **Step 1: Write the failing test for `per_head_tau`**

```python
# block_lo_arm_order_network/tests/test_order_tau_readout.py
import numpy as np
import pytest
from batch_readout.order_tau_readout import per_head_tau, layer_head_tau_table, derived_views


def _l2r_B65():
    # 65-node graph whose greedy None-start rollout yields identity order 0..63.
    # Strong forward edge i->i+1, plus None->0.
    B = np.zeros((65, 65), dtype=np.float64)
    B[0, 1] = 5.0  # None -> physical block 0
    for i in range(1, 64):
        B[i, i + 1] = 5.0  # block (i-1) -> block i  (node i -> node i+1)
    np.fill_diagonal(B, 0.0)
    return B


def test_per_head_tau_forward_l2r_is_plus_one():
    info = per_head_tau(_l2r_B65(), method="C-D+L")
    assert info["tau_vs_l2r"] > 0.99
    assert info["first_is_phys0"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/admin/lyuyuhuan/order_lyu && PYTHONPATH=block_lo_arm_order_network pytest block_lo_arm_order_network/tests/test_order_tau_readout.py::test_per_head_tau_forward_l2r_is_plus_one -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'batch_readout.order_tau_readout'`

- [ ] **Step 3: Implement `per_head_tau`**

```python
# block_lo_arm_order_network/batch_readout/order_tau_readout.py
"""Per-(layer,head) order-τ readout from model-frame strict65 B graphs.

Label-free: input B is model-frame; τ_vs_l2r is a posthoc comparison of the
rolled-out σ_model against np.arange, valid only because the trajectory logger
uses identity probe_orders (model-frame == physical L2R reference). No inv_perm
is used anywhere here.
"""
from __future__ import annotations

import numpy as np

from none_separated_block_graph import (
    rollout_by_method,
    discovery_metrics,
    classify_gate_status,
)

DEFAULT_METHODS = ("C-D+L", "L")


def per_head_tau(B65: np.ndarray, method: str) -> dict:
    order = rollout_by_method(np.asarray(B65, dtype=np.float64), method)
    m = discovery_metrics(order)
    return {
        "tau_vs_l2r": float(m["tau_vs_l2r"]),
        "phys0_rank": int(m["phys0_rank"]),
        "prefix8_overlap": int(m["prefix8_overlap"]),
        "first_is_phys0": bool(m["first_is_phys0"]),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/admin/lyuyuhuan/order_lyu && PYTHONPATH=block_lo_arm_order_network pytest block_lo_arm_order_network/tests/test_order_tau_readout.py::test_per_head_tau_forward_l2r_is_plus_one -v`
Expected: PASS

- [ ] **Step 5: Write failing tests for `layer_head_tau_table` + `derived_views`**

```python
def _anti_l2r_B65():
    # Greedy None-start rollout yields reverse order 63..0 -> tau ~ -1.
    B = np.zeros((65, 65), dtype=np.float64)
    B[0, 64] = 5.0  # None -> physical block 63
    for i in range(64, 1, -1):
        B[i, i - 1] = 5.0  # node i -> node i-1
    np.fill_diagonal(B, 0.0)
    return B


def test_layer_head_table_shapes_and_signs():
    L, H = 2, 2
    B = np.zeros((L, H, 65, 65))
    B[0, 0] = _l2r_B65()      # forward  tau ~ +1
    B[0, 1] = _anti_l2r_B65()  # reverse  tau ~ -1
    B[1, 0] = _l2r_B65()
    B[1, 1] = _l2r_B65()
    table = layer_head_tau_table(B, methods=("C-D+L",))
    assert table["tau"].shape == (2, 2, 1)
    assert table["tau"][0, 0, 0] > 0.99
    assert table["tau"][0, 1, 0] < -0.99


def test_derived_views_keep_abs_and_signed():
    L, H = 2, 2
    B = np.zeros((L, H, 65, 65))
    B[0, 0] = _l2r_B65()
    B[0, 1] = _anti_l2r_B65()   # strongest |tau| in layer 0 is the anti head
    B[1, 0] = _l2r_B65()
    B[1, 1] = _l2r_B65()
    table = layer_head_tau_table(B, methods=("C-D+L",))
    d = derived_views(table)
    # max_signed in layer 0 is the +1 head; max_abs picks the anti head's |−1|
    assert d["max_signed_tau_per_layer"][0, 0] > 0.99
    assert d["max_abs_tau_per_layer"][0, 0] > 0.99
    # the anti head must not be invisible: signed best != abs-best head here
    assert d["best_head_per_layer"]["signed"][0, 0] == 0
    assert d["best_head_per_layer"]["abs"][0, 0] == 1
```

- [ ] **Step 6: Run to verify fail**

Run: `cd /home/admin/lyuyuhuan/order_lyu && PYTHONPATH=block_lo_arm_order_network pytest block_lo_arm_order_network/tests/test_order_tau_readout.py -v`
Expected: FAIL with `AttributeError`/`ImportError` for `layer_head_tau_table`.

- [ ] **Step 7: Implement `layer_head_tau_table` + `derived_views`**

```python
def layer_head_tau_table(B_lhn: np.ndarray, methods=DEFAULT_METHODS) -> dict:
    B_lhn = np.asarray(B_lhn, dtype=np.float64)
    L, H = B_lhn.shape[:2]
    Mn = len(methods)
    tau = np.full((L, H, Mn), np.nan, dtype=np.float64)
    rank = np.full((L, H, Mn), -1, dtype=np.int64)
    pre8 = np.zeros((L, H, Mn), dtype=np.int64)
    for li in range(L):
        for hi in range(H):
            for mi, method in enumerate(methods):
                info = per_head_tau(B_lhn[li, hi], method)
                tau[li, hi, mi] = info["tau_vs_l2r"]
                rank[li, hi, mi] = info["phys0_rank"]
                pre8[li, hi, mi] = info["prefix8_overlap"]
    return {"tau": tau, "phys0_rank": rank, "prefix8": pre8, "methods": list(methods)}


def derived_views(table: dict, strong_tau: float = 0.7) -> dict:
    tau = table["tau"]  # (L,H,M)
    L, H, Mn = tau.shape
    abs_tau = np.abs(tau)
    max_signed = tau.max(axis=1)            # (L,M)
    best_signed = tau.argmax(axis=1)        # (L,M)
    max_abs = abs_tau.max(axis=1)           # (L,M)
    best_abs = abs_tau.argmax(axis=1)       # (L,M)
    strong = (abs_tau >= strong_tau).sum(axis=1)  # (L,M)
    glob = np.array([np.unravel_index(abs_tau[:, :, m].argmax(), (L, H))
                     for m in range(Mn)])   # (M,2) -> (layer,head)
    return {
        "best_head_per_layer": {"signed": best_signed, "abs": best_abs},
        "best_head_global": glob,
        "max_abs_tau_per_layer": max_abs,
        "max_signed_tau_per_layer": max_signed,
        "strong_pass_count_per_layer": strong,
    }
```

- [ ] **Step 8: Run all Task-1 tests to verify pass**

Run: `cd /home/admin/lyuyuhuan/order_lyu && PYTHONPATH=block_lo_arm_order_network pytest block_lo_arm_order_network/tests/test_order_tau_readout.py -v`
Expected: PASS (3 tests)

- [ ] **Step 9: Commit**

```bash
git add block_lo_arm_order_network/batch_readout/order_tau_readout.py block_lo_arm_order_network/tests/test_order_tau_readout.py
git commit -m "feat: per-(layer,head) order-tau readout with abs+signed derived views"
```

---

### Task 2: Weight-based candidate composition module

**Files:**
- Create: `block_lo_arm_order_network/batch_readout/attn_composition.py`
- Test: `block_lo_arm_order_network/tests/test_attn_composition.py`

**Interfaces:**
- Produces:
  - `head_weights(c_attn_w, c_proj_w, n_head, head, n_embd) -> dict` → activation-space `{"W_Q","W_K","W_V","W_O"}` with `W_Q/W_K/W_V` shape `(n_embd, head_dim)`, `W_O` shape `(head_dim, n_embd)`.
  - `composition_scores(up: dict, down: dict) -> dict` → `{"Q": float, "K": float, "V": float}`, each in `[0,1]`.
  - `layer_pair_composition(c_attn_ws, c_proj_ws, n_head, n_embd, pairs) -> dict` → `{(i,j): np.ndarray (H,H,3)}` for `{Q,K,V}` over upstream×downstream heads; `pairs` is a list of `(i,j)` with `i<j`.

**Shape contract (from spec §B):** PyTorch `Linear.weight` is `[out, in]`. For `c_attn` (out=3·n_embd): rows `[0:n_embd]`=q, `[n_embd:2n_embd]`=k, `[2n_embd:3n_embd]`=v; head `h` takes the `head_dim` row-block within each third. `W_Q = q_block.T` → `(n_embd, head_dim)`. For `c_proj` (out=n_embd, in=n_embd): head `h` reads input cols `[h*head_dim:(h+1)*head_dim]`; `W_O = c_proj_w[:, cols].T` → `(head_dim, n_embd)`. `W_OV = W_V @ W_O` `(n_embd,n_embd)`; `W_QK = W_Q @ W_K.T` `(n_embd,n_embd)`.

- [ ] **Step 1: Write failing test for `head_weights` shapes + `composition_scores` identity case**

```python
# block_lo_arm_order_network/tests/test_attn_composition.py
import numpy as np
from batch_readout.attn_composition import head_weights, composition_scores


def test_head_weights_shapes():
    n_embd, n_head = 8, 2
    head_dim = n_embd // n_head
    c_attn_w = np.random.RandomState(0).randn(3 * n_embd, n_embd)
    c_proj_w = np.random.RandomState(1).randn(n_embd, n_embd)
    w = head_weights(c_attn_w, c_proj_w, n_head, head=1, n_embd=n_embd)
    assert w["W_Q"].shape == (n_embd, head_dim)
    assert w["W_O"].shape == (head_dim, n_embd)


def test_composition_scores_in_unit_interval():
    n_embd, n_head = 8, 2
    rs = np.random.RandomState(2)
    c_attn_w = rs.randn(3 * n_embd, n_embd)
    c_proj_w = rs.randn(n_embd, n_embd)
    up = head_weights(c_attn_w, c_proj_w, n_head, 0, n_embd)
    down = head_weights(c_attn_w, c_proj_w, n_head, 1, n_embd)
    s = composition_scores(up, down)
    for key in ("Q", "K", "V"):
        assert 0.0 <= s[key] <= 1.0 + 1e-9
```

- [ ] **Step 2: Run to verify fail**

Run: `cd /home/admin/lyuyuhuan/order_lyu && PYTHONPATH=block_lo_arm_order_network pytest block_lo_arm_order_network/tests/test_attn_composition.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `head_weights` + `composition_scores`**

```python
# block_lo_arm_order_network/batch_readout/attn_composition.py
"""Weight-based CANDIDATE composition scores (composition compatibility).

NOT causal handoff evidence: per-head RMSNorm(q,k) and AdaLN modulation make
these approximate under this architecture. A high score only nominates an
upstream->downstream edge for later path-patching verification.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def head_weights(c_attn_w, c_proj_w, n_head, head, n_embd) -> dict:
    c_attn_w = np.asarray(c_attn_w, dtype=np.float64)  # (3*n_embd, n_embd)
    c_proj_w = np.asarray(c_proj_w, dtype=np.float64)  # (n_embd, n_embd)
    hd = n_embd // n_head
    r = slice(head * hd, (head + 1) * hd)
    q_block = c_attn_w[0 * n_embd:1 * n_embd][r, :]  # (hd, n_embd)
    k_block = c_attn_w[1 * n_embd:2 * n_embd][r, :]
    v_block = c_attn_w[2 * n_embd:3 * n_embd][r, :]
    W_Q = q_block.T  # (n_embd, hd)
    W_K = k_block.T
    W_V = v_block.T
    W_O = c_proj_w[:, r].T  # (hd, n_embd)
    return {"W_Q": W_Q, "W_K": W_K, "W_V": W_V, "W_O": W_O}


def _normed_frob(prod, a, b) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + _EPS
    return float(np.linalg.norm(prod) / denom)


def composition_scores(up: dict, down: dict) -> dict:
    W_OV_up = up["W_V"] @ up["W_O"]          # (n_embd, n_embd)
    W_QK_down = down["W_Q"] @ down["W_K"].T   # (n_embd, n_embd)
    W_OV_down = down["W_V"] @ down["W_O"]      # (n_embd, n_embd)
    return {
        "Q": _normed_frob(W_QK_down @ W_OV_up, W_QK_down, W_OV_up),
        "K": _normed_frob(W_QK_down.T @ W_OV_up, W_QK_down, W_OV_up),
        "V": _normed_frob(W_OV_down @ W_OV_up, W_OV_down, W_OV_up),
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /home/admin/lyuyuhuan/order_lyu && PYTHONPATH=block_lo_arm_order_network pytest block_lo_arm_order_network/tests/test_attn_composition.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Write failing test for `layer_pair_composition`**

```python
from batch_readout.attn_composition import layer_pair_composition


def test_layer_pair_composition_shapes():
    n_embd, n_head, L = 8, 2, 3
    rs = np.random.RandomState(3)
    c_attn_ws = [rs.randn(3 * n_embd, n_embd) for _ in range(L)]
    c_proj_ws = [rs.randn(n_embd, n_embd) for _ in range(L)]
    pairs = [(0, 1), (0, 2), (1, 2)]
    out = layer_pair_composition(c_attn_ws, c_proj_ws, n_head, n_embd, pairs)
    assert set(out.keys()) == set(pairs)
    assert out[(0, 1)].shape == (n_head, n_head, 3)  # (up_head, down_head, {Q,K,V})
```

- [ ] **Step 6: Run to verify fail**

Run: `cd /home/admin/lyuyuhuan/order_lyu && PYTHONPATH=block_lo_arm_order_network pytest block_lo_arm_order_network/tests/test_attn_composition.py::test_layer_pair_composition_shapes -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 7: Implement `layer_pair_composition`**

```python
def layer_pair_composition(c_attn_ws, c_proj_ws, n_head, n_embd, pairs) -> dict:
    out = {}
    for (i, j) in pairs:
        mat = np.zeros((n_head, n_head, 3), dtype=np.float64)
        ups = [head_weights(c_attn_ws[i], c_proj_ws[i], n_head, a, n_embd) for a in range(n_head)]
        downs = [head_weights(c_attn_ws[j], c_proj_ws[j], n_head, b, n_embd) for b in range(n_head)]
        for a in range(n_head):
            for b in range(n_head):
                s = composition_scores(ups[a], downs[b])
                mat[a, b] = [s["Q"], s["K"], s["V"]]
        out[(i, j)] = mat
    return out
```

- [ ] **Step 8: Run all Task-2 tests to verify pass**

Run: `cd /home/admin/lyuyuhuan/order_lyu && PYTHONPATH=block_lo_arm_order_network pytest block_lo_arm_order_network/tests/test_attn_composition.py -v`
Expected: PASS (3 tests)

- [ ] **Step 9: Commit**

```bash
git add block_lo_arm_order_network/batch_readout/attn_composition.py block_lo_arm_order_network/tests/test_attn_composition.py
git commit -m "feat: weight-based candidate composition scores (Q/K/V)"
```

---

### Task 3: Helper to read per-layer c_attn/c_proj weights from the model

**Files:**
- Modify: `block_lo_arm_order_network/batch_readout/attn_composition.py`
- Test: `block_lo_arm_order_network/tests/test_attn_composition.py`

**Interfaces:**
- Produces: `extract_layer_weights(model) -> tuple[list, list, int, int]` → `(c_attn_ws, c_proj_ws, n_head, n_embd)`, numpy weights per layer read from `model.transformer.h[ℓ].attn.{c_attn,c_proj}.weight`.

**Note for implementer:** confirm the module path to the block list. From the model file, `AOGPT` holds layers in `self.transformer.h` (nanoGPT convention); each block's attention is `.attn` with `.c_attn`/`.c_proj`. If the attribute differs, grep `self.transformer` in `model_AOGPT_AdaLN6_NoRep_cond_128_trunc_qknorm.py` and adjust. The test below is the gate.

- [ ] **Step 1: Write failing test using a real (tiny) model instance**

```python
def test_extract_layer_weights_matches_config():
    import torch
    from model_AOGPT_AdaLN6_NoRep_cond_128_trunc_qknorm import AOGPT, AOGPTConfig
    from batch_readout.attn_composition import extract_layer_weights
    cfg = AOGPTConfig(n_layer=4, n_head=8, n_embd=64, block_size=256)
    model = AOGPT(cfg)
    c_attn_ws, c_proj_ws, n_head, n_embd = extract_layer_weights(model)
    assert len(c_attn_ws) == 4 and len(c_proj_ws) == 4
    assert n_head == 8 and n_embd == 64
    assert c_attn_ws[0].shape == (3 * 64, 64)
    assert c_proj_ws[0].shape == (64, 64)
```

- [ ] **Step 2: Run to verify fail**

Run: `cd /home/admin/lyuyuhuan/order_lyu && PYTHONPATH=block_lo_arm_order_network pytest block_lo_arm_order_network/tests/test_attn_composition.py::test_extract_layer_weights_matches_config -v`
Expected: FAIL (`ImportError`) — or, if the attribute path is wrong, `AttributeError`; fix the path then.

- [ ] **Step 3: Implement `extract_layer_weights`**

```python
def extract_layer_weights(model):
    blocks = model.transformer.h
    c_attn_ws, c_proj_ws = [], []
    for blk in blocks:
        c_attn_ws.append(blk.attn.c_attn.weight.detach().cpu().numpy())
        c_proj_ws.append(blk.attn.c_proj.weight.detach().cpu().numpy())
    n_head = blocks[0].attn.n_head
    n_embd = blocks[0].attn.n_embd
    return c_attn_ws, c_proj_ws, int(n_head), int(n_embd)
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /home/admin/lyuyuhuan/order_lyu && PYTHONPATH=block_lo_arm_order_network pytest block_lo_arm_order_network/tests/test_attn_composition.py::test_extract_layer_weights_matches_config -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/batch_readout/attn_composition.py block_lo_arm_order_network/tests/test_attn_composition.py
git commit -m "feat: read per-layer c_attn/c_proj weights for composition"
```

---

### Task 4: All-layer extraction in the trajectory logger

**Files:**
- Modify: `block_lo_arm_order_network/attention_trajectory.py` (the `attn_list[0]` extraction at lines ~442-449, `build_model_frame_strict65` call, and metadata `num_layers_extracted`/`layer`)
- Test: `block_lo_arm_order_network/tests/test_attention_trajectory_alllayer.py`

**Interfaces:**
- Produces: a new pure helper `extract_all_layer_B(attn_list, probe_orders) -> np.ndarray` → `(L, S, H, 65, 65)` model-frame B (one call to `build_model_frame_strict65` per layer; that function is already layer-agnostic).
- Consumes: `batch_readout.l0_strict65.build_model_frame_strict65`.

- [ ] **Step 1: Write failing test**

```python
# block_lo_arm_order_network/tests/test_attention_trajectory_alllayer.py
import numpy as np
from attention_trajectory import extract_all_layer_B


def test_extract_all_layer_B_shape():
    L, S, H, T = 4, 2, 8, 256
    rng = np.random.default_rng(0)
    # causal-ish attention over T+1 tokens; rows sum ~1 not required for shape
    attn_list = [rng.random((S, H, T + 1, T + 1)).astype(np.float32) for _ in range(L)]
    probe_orders = np.tile(np.arange(T, dtype=np.int64), (S, 1))
    B = extract_all_layer_B(attn_list, probe_orders)
    assert B.shape == (L, S, H, 65, 65)
    # diagonal must be zero (none-separated convention)
    assert np.allclose(np.diagonal(B, axis1=3, axis2=4), 0.0)
```

- [ ] **Step 2: Run to verify fail**

Run: `cd /home/admin/lyuyuhuan/order_lyu && PYTHONPATH=block_lo_arm_order_network pytest block_lo_arm_order_network/tests/test_attention_trajectory_alllayer.py -v`
Expected: FAIL with `ImportError: cannot import name 'extract_all_layer_B'`.

- [ ] **Step 3: Implement `extract_all_layer_B` and use it in `log_snapshot`**

Add near the top-level functions of `attention_trajectory.py`:

```python
def extract_all_layer_B(attn_list, probe_orders) -> np.ndarray:
    """Build model-frame strict65 B for every layer.

    attn_list: list of (S, H, 257, 257) per layer (torch or numpy).
    Returns (L, S, H, 65, 65) float32.
    """
    out = []
    for attn_l in attn_list:
        arr = attn_l.cpu().numpy() if hasattr(attn_l, "cpu") else np.asarray(attn_l)
        out.append(build_model_frame_strict65(arr, probe_orders))
    return np.stack(out, axis=0)
```

Then in `log_snapshot`, replace lines ~442-449:

```python
        # all-layer extraction (was L0-only)
        B_all = extract_all_layer_B(attn_list, self._probe_orders)  # (L,S,H,65,65)
        if was_training:
            model.train()
        B_model = B_all[0]  # keep existing L0 summaries/heatmaps working
```

Also update the `__init__` metadata dict: set `"num_layers_extracted": len(... )` is not known at init; change the two keys to `"num_layers_extracted": "all"` and drop the hardcoded `"layer": 0`, or leave them and add `"all_layers": True` (the test does not assert on metadata).

- [ ] **Step 4: Run to verify pass**

Run: `cd /home/admin/lyuyuhuan/order_lyu && PYTHONPATH=block_lo_arm_order_network pytest block_lo_arm_order_network/tests/test_attention_trajectory_alllayer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/attention_trajectory.py block_lo_arm_order_network/tests/test_attention_trajectory_alllayer.py
git commit -m "feat: all-layer strict65 B extraction in trajectory logger"
```

---

### Task 5: Wire τ table + composition into `log_snapshot` outputs

**Files:**
- Modify: `block_lo_arm_order_network/attention_trajectory.py` (`__init__` to accept `bs_mean`, `composition_pairs`, and a model-weights hook; `log_snapshot` to compute and save τ + composition)
- Test: `block_lo_arm_order_network/tests/test_attention_trajectory_alllayer.py`

**Interfaces:**
- Consumes: `order_tau_readout.layer_head_tau_table`, `.derived_views`; `attn_composition.extract_layer_weights`, `.layer_pair_composition`; `per_head_order_scan._batch_mean_B` (for bs_mean batching the S samples down — here S == n_attention_samples, so batch-mean is over all probe samples).
- Produces: in each `raw/step_XXXXXX/` dir, `tau_table.npz` (`tau`,`phys0_rank`,`prefix8`,`methods`) and `composition.npz` (one array per `"i_j"` key); appends `max_signed`/`max_abs`/`strong_count` per layer to the per-step `_summary_log`.

**Implementation notes:**
- Batch-mean over probe samples: `B_lhn = B_all.mean(axis=1)` gives `(L,H,65,65)` — this is the bs_mean=`n_attention_samples` mean. To honor `bs_mean=16` independent of the 8 fixed-probe count, the launcher (Task 7) sets `--attn-trajectory-samples 16`; `log_snapshot` simply averages whatever S it is given. Record the effective `bs_mean = S` in the npz.
- composition uses model weights at this step via `extract_layer_weights(model)`.

- [ ] **Step 1: Write failing test (integration of τ + composition save)**

```python
import numpy as np
import torch
from pathlib import Path
from attention_trajectory import AttentionTrajectoryLogger


def test_log_snapshot_saves_tau_and_composition(tmp_path):
    from model_AOGPT_AdaLN6_NoRep_cond_128_trunc_qknorm import AOGPT, AOGPTConfig
    cfg = AOGPTConfig(n_layer=4, n_head=8, n_embd=64, block_size=256)
    model = AOGPT(cfg)
    logger = AttentionTrajectoryLogger(
        output_root=tmp_path, run_name="t", n_attention_samples=4,
        composition_pairs=[(0, 1), (1, 2), (2, 3)], seed=2,
    )
    eval_tokens = torch.randint(0, cfg.vocab_size, (8, 256))
    logger.log_snapshot(model, eval_tokens, global_step=0, clean_perm=None,
                        device=torch.device("cpu"))
    step_dir = tmp_path / "raw" / "step_000000"
    tau = np.load(step_dir / "tau_table.npz")
    assert tau["tau"].shape == (4, 8, 2)  # (L,H,methods)
    comp = np.load(step_dir / "composition.npz")
    assert comp["0_1"].shape == (8, 8, 3)
```

- [ ] **Step 2: Run to verify fail**

Run: `cd /home/admin/lyuyuhuan/order_lyu && PYTHONPATH=block_lo_arm_order_network pytest block_lo_arm_order_network/tests/test_attention_trajectory_alllayer.py::test_log_snapshot_saves_tau_and_composition -v`
Expected: FAIL (`__init__` lacks `composition_pairs`; no `tau_table.npz`).

- [ ] **Step 3: Implement**

In `__init__`, add params `composition_pairs: Optional[list] = None` and store `self.composition_pairs = composition_pairs or []`. In `log_snapshot`, after `B_all` is built and the model is set back to train, insert before the return:

```python
        from batch_readout.order_tau_readout import layer_head_tau_table, derived_views
        from batch_readout.attn_composition import extract_layer_weights, layer_pair_composition

        # batch-mean over probe samples -> (L,H,65,65)
        B_lhn = B_all.mean(axis=1)
        tau_tbl = layer_head_tau_table(B_lhn, methods=("C-D+L", "L"))
        dviews = derived_views(tau_tbl)
        np.savez(step_dir / "tau_table.npz",
                 tau=tau_tbl["tau"], phys0_rank=tau_tbl["phys0_rank"],
                 prefix8=tau_tbl["prefix8"], methods=np.array(tau_tbl["methods"]),
                 bs_mean=np.int64(B_all.shape[1]))

        if self.composition_pairs:
            c_attn_ws, c_proj_ws, n_head, n_embd = extract_layer_weights(model)
            comp = layer_pair_composition(c_attn_ws, c_proj_ws, n_head, n_embd,
                                          self.composition_pairs)
            np.savez(step_dir / "composition.npz",
                     **{f"{i}_{j}": comp[(i, j)] for (i, j) in self.composition_pairs})

        self._summary_log[-1].update({
            "max_signed_tau_per_layer": dviews["max_signed_tau_per_layer"].tolist(),
            "max_abs_tau_per_layer": dviews["max_abs_tau_per_layer"].tolist(),
            "strong_pass_count_per_layer": dviews["strong_pass_count_per_layer"].tolist(),
        })
```

(`self._summary_log[-1]` exists because the existing code appends to it earlier in `log_snapshot`; verify the append happens before this block, otherwise append a fresh dict.)

- [ ] **Step 4: Run to verify pass**

Run: `cd /home/admin/lyuyuhuan/order_lyu && PYTHONPATH=block_lo_arm_order_network pytest block_lo_arm_order_network/tests/test_attention_trajectory_alllayer.py -v`
Expected: PASS (all tests in file)

- [ ] **Step 5: Run the full touched-module test suite (regression)**

Run: `cd /home/admin/lyuyuhuan/order_lyu && PYTHONPATH=block_lo_arm_order_network pytest block_lo_arm_order_network/tests/test_order_tau_readout.py block_lo_arm_order_network/tests/test_attn_composition.py block_lo_arm_order_network/tests/test_attention_trajectory_alllayer.py -v`
Expected: PASS (all)

- [ ] **Step 6: Commit**

```bash
git add block_lo_arm_order_network/attention_trajectory.py block_lo_arm_order_network/tests/test_attention_trajectory_alllayer.py
git commit -m "feat: save per-(layer,head) tau table + candidate composition per snapshot"
```

---

### Task 6: Pass composition_pairs + bs_mean through `train_clean_aogpt`

**Files:**
- Modify: `block_lo_arm_order_network/train_clean_aogpt.py` (the `AttentionTrajectoryLogger(...)` construction at lines ~1803-1831, and the arg parser near lines ~972-978)

**Interfaces:**
- Consumes: `AttentionTrajectoryLogger(..., composition_pairs=...)` from Task 5.
- Produces: CLI flag `--attn-composition-pairs` (default `all`), forwarded to the logger.

- [ ] **Step 1: Add the CLI flag (no test; config plumbing verified by the smoke run in Task 7)**

In the parser block (~line 978), add:

```python
    p.add_argument("--attn-composition-pairs", default="all",
                   choices=["all", "adjacent"],
                   help="layer pairs for candidate composition: all i<j, or adjacent only.")
```

- [ ] **Step 2: Build the pairs list and pass to the logger**

Where the logger is constructed (~line 1803), before it:

```python
        n_layer = model.config.n_layer
        if args.attn_composition_pairs == "adjacent":
            comp_pairs = [(i, i + 1) for i in range(n_layer - 1)]
        else:
            comp_pairs = [(i, j) for i in range(n_layer) for j in range(i + 1, n_layer)]
```

and add `composition_pairs=comp_pairs,` to the `AttentionTrajectoryLogger(...)` kwargs.

- [ ] **Step 3: Syntax/smoke check the import path**

Run: `cd /home/admin/lyuyuhuan/order_lyu && PYTHONPATH=block_lo_arm_order_network python -c "import ast; ast.parse(open('block_lo_arm_order_network/train_clean_aogpt.py').read()); print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add block_lo_arm_order_network/train_clean_aogpt.py
git commit -m "feat: forward composition-pairs config into trajectory logger"
```

---

### Task 7: Launcher with overhead-calibration gate (3 seeds)

**Files:**
- Create: `scripts/run_handoff_circuit_trajectory.sh`

**Interfaces:**
- Consumes: `train_clean_aogpt` CLI (`--attn-trajectory`, `--attn-trajectory-samples`, `--eval-interval`, `--attn-composition-pairs`, `--seed`, `--max-steps`, and the existing save-steps / run-kind / data-source flags — confirm exact names with `python block_lo_arm_order_network/train_clean_aogpt.py --help`).

- [ ] **Step 1: Write the calibration + launch script**

```bash
#!/usr/bin/env bash
# scripts/run_handoff_circuit_trajectory.sh
# Online order-signal handoff-circuit trajectory: 3 seeds, 0->10k, record every 200.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=block_lo_arm_order_network

SEEDS=(2 42 123)
COMMON=(--run-kind baseline --data-source continuous
        --attn-trajectory --attn-trajectory-samples 16
        --attn-composition-pairs all
        --eval-interval 200 --max-steps 10000)
# NOTE: confirm save-steps flag name from --help; intent is ckpt every 1000.
SAVE_FLAG="--save-steps 0,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000"
DEVICE=${DEVICE:-cuda:0}

# ── Calibration gate: seed 2 to 400 steps, measure snapshot overhead ──
echo "[calib] seed 2 -> 400 steps"
python block_lo_arm_order_network/train_clean_aogpt.py \
  --seed 2 "${COMMON[@]}" $SAVE_FLAG --max-steps 400 \
  --device "$DEVICE" --out-dir runs/handoff_calib_seed2 2>&1 | tee runs/handoff_calib_seed2.log

echo "[calib] inspect extraction_time_s in runs/handoff_calib_seed2/attention_trajectory/summaries/"
echo "[calib] If snapshot overhead > 10% of a 200-step interval, apply the spec degradation ladder before continuing."
read -r -p "Calibration acceptable? continue to full 3-seed run? [y/N] " ok
[[ "$ok" == "y" ]] || { echo "stopping at calibration gate"; exit 0; }

# ── Full runs ──
for s in "${SEEDS[@]}"; do
  echo "[run] seed $s -> 10000"
  python block_lo_arm_order_network/train_clean_aogpt.py \
    --seed "$s" "${COMMON[@]}" $SAVE_FLAG \
    --device "$DEVICE" --out-dir "runs/handoff_seed${s}" 2>&1 | tee "runs/handoff_seed${s}.log"
done
echo "done. trajectories in runs/handoff_seed{2,42,123}/attention_trajectory/"
```

- [ ] **Step 2: Make executable and shellcheck-parse**

Run: `cd /home/admin/lyuyuhuan/order_lyu && chmod +x scripts/run_handoff_circuit_trajectory.sh && bash -n scripts/run_handoff_circuit_trajectory.sh && echo ok`
Expected: `ok`

- [ ] **Step 3: Verify the real CLI flag names before running**

Run: `cd /home/admin/lyuyuhuan/order_lyu && PYTHONPATH=block_lo_arm_order_network python block_lo_arm_order_network/train_clean_aogpt.py --help 2>&1 | grep -E "save-steps|run-kind|data-source|out-dir|device"`
Expected: confirm each flag exists; fix the script's flag names if any differ. (This is a required check — do not launch with unverified flags.)

- [ ] **Step 4: Commit**

```bash
git add scripts/run_handoff_circuit_trajectory.sh
git commit -m "feat: handoff-circuit trajectory launcher with overhead calibration gate"
```

---

### Task 8: Offline analysis — flow heatmaps + handoff timing

**Files:**
- Create: `analyses/analyze_handoff_circuit.py`
- Test: `block_lo_arm_order_network/tests/test_analyze_handoff_circuit.py`

**Interfaces:**
- Produces:
  - `load_trajectory(run_dir) -> dict` → `{"steps": np.ndarray, "tau": (T,L,H,M), "composition": {(i,j): (T,H,H,3)}, "methods": list}` by reading `raw/step_*/tau_table.npz` + `composition.npz` in step order.
  - `handoff_timing(traj) -> dict` → per layer, the step of peak `max_abs_tau`; and per candidate edge, the step of peak composition — used to test the spec's handoff signature (upstream τ peak precedes downstream τ peak with rising composition).

- [ ] **Step 1: Write failing test with a synthetic two-step trajectory dir**

```python
# block_lo_arm_order_network/tests/test_analyze_handoff_circuit.py
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, "analyses")
from analyze_handoff_circuit import load_trajectory, handoff_timing


def _write_step(root, step, tau, comp01):
    d = root / "raw" / f"step_{step:06d}"
    d.mkdir(parents=True)
    np.savez(d / "tau_table.npz", tau=tau, phys0_rank=np.zeros_like(tau, dtype=np.int64),
             prefix8=np.zeros_like(tau, dtype=np.int64),
             methods=np.array(["C-D+L", "L"]), bs_mean=np.int64(16))
    np.savez(d / "composition.npz", **{"0_1": comp01})


def test_load_and_timing(tmp_path):
    L, H, M = 2, 2, 2
    # step 0: L0 head0 carries; step 200: L1 head0 carries; comp rises
    tau0 = np.zeros((L, H, M)); tau0[0, 0] = 1.0
    tau1 = np.zeros((L, H, M)); tau1[1, 0] = 1.0
    _write_step(tmp_path, 0, tau0, np.zeros((H, H, 3)))
    _write_step(tmp_path, 200, tau1, np.ones((H, H, 3)))
    traj = load_trajectory(tmp_path)
    assert traj["tau"].shape == (2, L, H, M)
    assert list(traj["steps"]) == [0, 200]
    t = handoff_timing(traj)
    # layer 0 peaks earlier than layer 1
    assert t["layer_peak_step"][0] <= t["layer_peak_step"][1]
```

- [ ] **Step 2: Run to verify fail**

Run: `cd /home/admin/lyuyuhuan/order_lyu && PYTHONPATH=block_lo_arm_order_network pytest block_lo_arm_order_network/tests/test_analyze_handoff_circuit.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `analyze_handoff_circuit.py`**

```python
# analyses/analyze_handoff_circuit.py
"""Offline analysis of handoff-circuit trajectories: flow + handoff timing."""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np

_STEP_RE = re.compile(r"step_(\d+)")


def load_trajectory(run_dir) -> dict:
    raw = Path(run_dir) / "raw"
    step_dirs = sorted(raw.glob("step_*"), key=lambda p: int(_STEP_RE.search(p.name).group(1)))
    steps, taus, comps = [], [], {}
    methods = None
    for d in step_dirs:
        step = int(_STEP_RE.search(d.name).group(1))
        tt = np.load(d / "tau_table.npz", allow_pickle=True)
        steps.append(step)
        taus.append(tt["tau"])
        methods = list(tt["methods"])
        cpath = d / "composition.npz"
        if cpath.exists():
            cz = np.load(cpath)
            for key in cz.files:
                i, j = map(int, key.split("_"))
                comps.setdefault((i, j), []).append(cz[key])
    return {
        "steps": np.array(steps),
        "tau": np.stack(taus, axis=0),
        "composition": {k: np.stack(v, axis=0) for k, v in comps.items()},
        "methods": methods,
    }


def handoff_timing(traj: dict) -> dict:
    tau = traj["tau"]            # (T,L,H,M)
    steps = traj["steps"]
    max_abs = np.abs(tau).max(axis=(2, 3))   # (T,L)
    peak_idx = max_abs.argmax(axis=0)         # (L,)
    layer_peak_step = steps[peak_idx]
    edge_peak_step = {}
    for (i, j), arr in traj["composition"].items():  # (T,H,H,3)
        edge_peak_step[(i, j)] = int(steps[arr.max(axis=(1, 2, 3)).argmax()])
    return {"layer_peak_step": layer_peak_step, "edge_peak_step": edge_peak_step}
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /home/admin/lyuyuhuan/order_lyu && PYTHONPATH=block_lo_arm_order_network pytest block_lo_arm_order_network/tests/test_analyze_handoff_circuit.py -v`
Expected: PASS

- [ ] **Step 5: Add a `main()` that writes flow heatmaps (no test; visual artifact)**

```python
def main():
    import argparse, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    traj = load_trajectory(args.run_dir)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tau = traj["tau"]; steps = traj["steps"]
    for label, data in [("max_abs", np.abs(tau).max(axis=2)),
                        ("max_signed", tau.max(axis=2))]:
        for mi, method in enumerate(traj["methods"]):
            fig, axp = plt.subplots(figsize=(8, 3))
            im = axp.imshow(data[:, :, mi].T, aspect="auto", origin="lower",
                            extent=[steps[0], steps[-1], -0.5, tau.shape[1] - 0.5],
                            cmap="RdBu_r", vmin=-1, vmax=1)
            axp.set_xlabel("step"); axp.set_ylabel("layer")
            axp.set_title(f"{label} tau ({method})")
            fig.colorbar(im, ax=axp)
            fig.savefig(out / f"flow_{label}_{method}.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
    print(f"wrote flow heatmaps to {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add analyses/analyze_handoff_circuit.py block_lo_arm_order_network/tests/test_analyze_handoff_circuit.py
git commit -m "feat: offline handoff-circuit flow + timing analysis"
```

---

## Self-Review

**Spec coverage:**
- §A all-layer extraction → Task 4. ✓
- §A per-(layer,head) signed τ + raw/derived/viz (abs+signed) → Task 1 + Task 5 (save) + Task 8 (viz). ✓
- §A bs_mean=16 → Task 5 note + Task 7 `--attn-trajectory-samples 16`. ✓
- §B candidate composition (shape contract, Q/K/V, all i<j, naming) → Task 2 + Task 3 + Task 6. ✓
- §C training run (eval-interval 200, order_policy random, ckpt every 1000, calibration gate) → Task 7. ✓ (order_policy default already random for `run_kind baseline`; the help-check in Task 7 step 3 confirms.)
- Outputs (tau npz, composition npz, heatmaps, ckpts) → Tasks 5, 7, 8. ✓
- Hypotheses/handoff signature → Task 8 `handoff_timing`. ✓
- Testing (TDD pure fns; L0 regression anchor) → Tasks 1-5,8. **Gap:** the spec's "per-(layer,head) τ matches `search_none_separated_65_heads` on a fixed ckpt for L0" regression is not a unit test (needs a real ckpt). Captured instead by Task 7 step-3 manual check + first real run cross-checked against existing `5k_signal_carrier_layer_multiseed` reports; acceptable as a runtime check, noted here so it is not forgotten.

**Placeholder scan:** no TBD/TODO; every code step has real code. Two explicit "confirm flag name"/"confirm attribute path" checks are deliberate runtime gates with the verifying command given, not placeholders.

**Type consistency:** `layer_head_tau_table` returns `tau (L,H,M)` consumed identically in Task 5 save and Task 8 load. `composition_scores` returns `{"Q","K","V"}` → `layer_pair_composition` packs `[Q,K,V]` last axis → Task 8 reads `(H,H,3)`. `extract_layer_weights` tuple order `(c_attn_ws, c_proj_ws, n_head, n_embd)` matches all call sites. ✓

## Notes For The Executor

- Run order is linear (Task N depends on N-1 for the wiring tasks). Tasks 1, 2 are independent and could be done in parallel.
- The expensive runtime risk is Task 7's calibration; honor the spec degradation ladder (composition cut last).
- Do not launch the full 3-seed run until Task 7 step-3 flag verification passes.
