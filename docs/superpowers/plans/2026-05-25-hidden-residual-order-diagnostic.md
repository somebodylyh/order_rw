# Hidden-Residual Order Diagnostic (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a GPU-free, training-free diagnostic that measures whether a clean random-trained text AOGPT carries a per-sample attention-order residual `r_x(v|S_t)=s_x−s_G` that hidden state (oracle `h_v` / causal `h_{S_t}`) can explain.

**Architecture:** Three pure-ish library modules (graph/teacher math; hidden extraction; probes) + one orchestrator script. All new code in `block_lo_arm_order_network/`, reusing existing `train_clean_aogpt`, `attn_order_teacher`, `attn_order_features`, `directed_graph_policy`, `clean_training_protocol`, `mlp_residual_policy`, and `AOGPT_block.forward_fn(return_hidden=...)`. Unit tests use tiny synthetic graphs/hidden (fast, CPU); one integration task runs the real `clean_base_random_perm/ckpt_step30000.pt`.

**Tech Stack:** Python 3.8, numpy, torch 2.4 (CPU ok), scikit-learn (Ridge/LogisticRegression), pytest. Reuses spec `docs/superpowers/specs/2026-05-25-hidden-residual-order-diagnostic-design.md`.

---

## File Structure

- `block_lo_arm_order_network/hidden_residual_graph.py` — B_x/B_G build (physical frame), canonical fixed global states, residual target, split-pass noise floor, gate. **Guards 1 & 2 live here.**
- `block_lo_arm_order_network/hidden_residual_hidden.py` — oracle `h_v` (original mode) + causal `h_{S_t}` (predictor mode) extraction, model→physical block remap.
- `block_lo_arm_order_network/hidden_residual_probe.py` — 1a representation probes (+ controls), 1b residual probes (B0/B1/B2 + oracle/causal heads), order-effect metrics. Pure numpy/sklearn.
- `scripts/run_hidden_residual_diag.py` — orchestrator/CLI on real ckpts; writes `gate.json`, `phase1a.json`, `phase1b.json`, `order_effects.json`, `SUMMARY.md`.
- Tests: `block_lo_arm_order_network/tests/test_hidden_residual_graph.py`, `..._hidden.py`, `..._probe.py`.

Constants shared across modules: `N_BLOCKS = 64`, `BLOCK_LEN = 4`, `T_LIST = [0, 16, 32, 48]`.

---

## Task 1: Graph module — physical-frame B set + frame guard

**Files:**
- Create: `block_lo_arm_order_network/hidden_residual_graph.py`
- Test: `block_lo_arm_order_network/tests/test_hidden_residual_graph.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hidden_residual_graph.py
import numpy as np
import pytest
from hidden_residual_graph import build_B_set, PHYS_FRAME

def test_build_B_set_is_transpose_zero_diag_and_tagged():
    rng = np.random.default_rng(0)
    A_all = rng.random((5, 4, 4)).astype(np.float32)  # (n, N, N)
    B_x_list, B_G, frame = build_B_set(A_all, frame=PHYS_FRAME)
    assert frame == PHYS_FRAME
    assert len(B_x_list) == 5
    # B = A.T, zero diag
    np.testing.assert_allclose(B_x_list[0], A_all[0].T * (1 - np.eye(4)), rtol=1e-5)
    np.testing.assert_allclose(B_G, A_all.mean(0).T * (1 - np.eye(4)), rtol=1e-5)

def test_build_B_set_rejects_non_physical_frame():
    A_all = np.zeros((2, 4, 4), dtype=np.float32)
    with pytest.raises(AssertionError):
        build_B_set(A_all, frame="model_block")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_hidden_residual_graph.py -v`
Expected: FAIL with `ModuleNotFoundError: hidden_residual_graph` / `cannot import build_B_set`.

- [ ] **Step 3: Write minimal implementation**

```python
# hidden_residual_graph.py
"""Phase-1 hidden-residual diagnostic: graph/teacher math (physical block frame).
See docs/superpowers/specs/2026-05-25-hidden-residual-order-diagnostic-design.md."""
import numpy as np
from directed_graph_policy import build_directed_graph

N_BLOCKS = 64
BLOCK_LEN = 4
T_LIST = [0, 16, 32, 48]
PHYS_FRAME = "physical_block"  # the ONLY frame the teacher is allowed to score in (guard 1)

def build_B_set(A_all, frame):
    """A_all: (n, N, N) attention in PHYSICAL block frame (extract_A_matrices already remaps).
    Returns (B_x_list, B_G, frame). B = build_directed_graph(A) = A.T with zero diag."""
    assert frame == PHYS_FRAME, f"B must be built in {PHYS_FRAME!r} frame, got {frame!r} (guard 1)"
    A_all = np.asarray(A_all, dtype=np.float64)
    assert A_all.ndim == 3 and A_all.shape[1] == A_all.shape[2], f"bad A_all shape {A_all.shape}"
    B_x_list = [build_directed_graph(A_all[i]) for i in range(A_all.shape[0])]
    B_G = build_directed_graph(A_all.mean(axis=0))
    return B_x_list, B_G, frame
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_hidden_residual_graph.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/hidden_residual_graph.py block_lo_arm_order_network/tests/test_hidden_residual_graph.py
git commit -m "feat(hidden-residual): physical-frame B set builder + frame guard"
```

---

## Task 2: Canonical fixed global states (guard 2)

**Files:**
- Modify: `block_lo_arm_order_network/hidden_residual_graph.py`
- Test: `block_lo_arm_order_network/tests/test_hidden_residual_graph.py`

- [ ] **Step 1: Write the failing test**

```python
def test_canonical_states_fixed_and_shaped():
    from hidden_residual_graph import canonical_states, N_BLOCKS
    rng = np.random.default_rng(1)
    B_G = rng.random((N_BLOCKS, N_BLOCKS)); np.fill_diagonal(B_G, 0.0)
    st = canonical_states(B_G, t_list=[0, 16, 32, 48])
    # deterministic: same B_G -> identical states
    st2 = canonical_states(B_G, t_list=[0, 16, 32, 48])
    for t in [0, 16, 32, 48]:
        S, U, last = st[t]
        assert len(S) == t and len(U) == N_BLOCKS - t
        assert set(S).isdisjoint(set(U)) and len(set(S) | set(U)) == N_BLOCKS
        assert (last is None) == (t == 0)
        assert st2[t][0] == S  # fixed/deterministic across calls
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_hidden_residual_graph.py::test_canonical_states_fixed_and_shaped -v`
Expected: FAIL with `cannot import canonical_states`.

- [ ] **Step 3: Write minimal implementation (append to hidden_residual_graph.py)**

```python
from attn_order_teacher import rollout_order

def canonical_states(B_G, t_list=T_LIST):
    """Guard 2: ONE global greedy C-D+L rollout over B_G defines the partial states used
    for EVERY sample. Returns {t: (S_t:tuple, U_t:tuple, last:int|None)}."""
    order = rollout_order(np.asarray(B_G, dtype=np.float64), mode="C-D+L",
                          greedy=True, standardize=True)
    order = [int(v) for v in order]
    states = {}
    for t in t_list:
        S = tuple(order[:t])
        U = tuple(order[t:])
        last = order[t - 1] if t > 0 else None
        states[t] = (S, U, last)
    return states
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_hidden_residual_graph.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/hidden_residual_graph.py block_lo_arm_order_network/tests/test_hidden_residual_graph.py
git commit -m "feat(hidden-residual): fixed global canonical states (guard 2)"
```

---

## Task 3: Residual target r_x(v|S_t) = s_x − s_G

**Files:**
- Modify: `block_lo_arm_order_network/hidden_residual_graph.py`
- Test: `block_lo_arm_order_network/tests/test_hidden_residual_graph.py`

- [ ] **Step 1: Write the failing test**

```python
def test_residual_target_zero_when_Bx_equals_BG_and_nonzero_when_differ():
    from hidden_residual_graph import canonical_states, residual_target, PHYS_FRAME, N_BLOCKS
    rng = np.random.default_rng(2)
    B_G = rng.random((N_BLOCKS, N_BLOCKS)); np.fill_diagonal(B_G, 0.0)
    states = canonical_states(B_G, t_list=[0, 16])
    # identical B_x -> r ~ 0
    out_same = residual_target(B_G.copy(), B_G, states, PHYS_FRAME, PHYS_FRAME)
    for t in [0, 16]:
        assert np.allclose(out_same[t]["r"], 0.0, atol=1e-9)
        assert np.array_equal(out_same[t]["U"], np.asarray(states[t][1]))
    # different B_x -> nonzero r
    B_x = B_G + rng.normal(0, 0.5, B_G.shape); np.fill_diagonal(B_x, 0.0)
    out_diff = residual_target(B_x, B_G, states, PHYS_FRAME, PHYS_FRAME)
    assert np.abs(out_diff[16]["r"]).max() > 1e-6

def test_residual_target_frame_guard():
    from hidden_residual_graph import canonical_states, residual_target, PHYS_FRAME, N_BLOCKS
    B_G = np.zeros((N_BLOCKS, N_BLOCKS))
    states = canonical_states(B_G, t_list=[0])
    with pytest.raises(AssertionError):
        residual_target(B_G, B_G, states, "model_block", PHYS_FRAME)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_hidden_residual_graph.py -k residual_target -v`
Expected: FAIL with `cannot import residual_target`.

- [ ] **Step 3: Write minimal implementation (append)**

```python
from attn_order_teacher import teacher_scores

def residual_target(B_x, B_G, states, frame_x, frame_g, mode="C-D+L"):
    """r_x(v|S_t) = s_x(v|S_t) - s_G(v|S_t), scored by the same teacher at the SAME state.
    Guard 1: both graphs must be in physical block frame. Returns {t: {U, r, s_x, s_g}}."""
    assert frame_x == frame_g == PHYS_FRAME, \
        f"B_x/B_G frame mismatch (guard 1): {frame_x!r} vs {frame_g!r}"
    out = {}
    for t, (S, U, last) in states.items():
        q_x, U_x = teacher_scores(B_x, S, U, last, mode=mode)
        q_g, U_g = teacher_scores(B_G, S, U, last, mode=mode)
        assert np.array_equal(U_x, U_g), "candidate sets diverged (state not fixed)"
        out[t] = {"U": U_x, "r": (q_x - q_g), "s_x": q_x, "s_g": q_g}
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_hidden_residual_graph.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/hidden_residual_graph.py block_lo_arm_order_network/tests/test_hidden_residual_graph.py
git commit -m "feat(hidden-residual): residual target r_x=s_x-s_G at fixed states"
```

---

## Task 4: Split-pass noise floor + pre-gate

**Files:**
- Modify: `block_lo_arm_order_network/hidden_residual_graph.py`
- Test: `block_lo_arm_order_network/tests/test_hidden_residual_graph.py`

- [ ] **Step 1: Write the failing test**

```python
def test_gate_fails_when_no_persample_variation_passes_when_present():
    from hidden_residual_graph import build_B_set, canonical_states, gate_metrics, PHYS_FRAME, N_BLOCKS
    rng = np.random.default_rng(3)
    base = rng.random((N_BLOCKS, N_BLOCKS)).astype(np.float32); np.fill_diagonal(base, 0.0)
    # NO per-sample variation: every sample identical (B_x == B_G) -> gate fail
    A_same = np.stack([base] * 8)
    g_same = gate_metrics(A_same, A_same[:4], A_same[4:], t_list=[0, 16], frame=PHYS_FRAME)
    assert g_same["passed"] is False
    assert g_same["r_norm"] < g_same["noise_floor"] + 1e-6
    # strong per-sample variation -> gate pass
    A_var = np.stack([base + rng.normal(0, 0.4, base.shape).astype(np.float32) for _ in range(8)])
    for a in A_var: np.fill_diagonal(a, 0.0)
    g_var = gate_metrics(A_var, A_var[:4], A_var[4:], t_list=[0, 16], frame=PHYS_FRAME)
    assert g_var["passed"] is True
    assert g_var["mean_abs_Bx_minus_BG"] > g_same["mean_abs_Bx_minus_BG"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_hidden_residual_graph.py -k gate -v`
Expected: FAIL with `cannot import gate_metrics`.

- [ ] **Step 3: Write minimal implementation (append)**

```python
def gate_metrics(A_all, A_half_a, A_half_b, t_list=T_LIST, frame=PHYS_FRAME, snr_threshold=1.0):
    """Pre-gate: does per-sample B_x vary, and is r_x above the split-pass noise floor?
    A_all: all-pass mean substrate; A_half_a/A_half_b: the two pass-halves (2-vs-2 at M=4).
    Returns dict of metrics + bool 'passed'."""
    B_x_list, B_G, fr = build_B_set(A_all, frame=frame)
    states = canonical_states(B_G, t_list=t_list)
    # signal: mean over samples of |r_x| at the largest probed t (most context)
    t_sig = max(t_list)
    r_sig = []
    for B_x in B_x_list:
        out = residual_target(B_x, B_G, states, fr, fr)
        r_sig.append(np.abs(out[t_sig]["r"]).mean())
    r_norm = float(np.mean(r_sig))
    # noise floor: same quantity from two pass-halves' graphs vs each other
    B_a, _, _ = build_B_set(A_half_a, frame=frame)
    B_b, _, _ = build_B_set(A_half_b, frame=frame)
    B_a_mean = build_directed_graph(np.asarray(A_half_a, np.float64).mean(0))
    B_b_mean = build_directed_graph(np.asarray(A_half_b, np.float64).mean(0))
    noise = np.abs(residual_target(B_a_mean, B_b_mean, states, frame, frame)[t_sig]["r"]).mean()
    noise_floor = float(noise)
    mean_abs = float(np.mean([np.abs(B_x - B_G).mean() for B_x in B_x_list]))
    corr = float(np.mean([np.corrcoef(B_x.ravel(), B_G.ravel())[0, 1] for B_x in B_x_list]))
    snr = r_norm / (noise_floor + 1e-9)
    return {"r_norm": r_norm, "noise_floor": noise_floor, "snr": snr,
            "mean_abs_Bx_minus_BG": mean_abs, "corr_Bx_BG": corr,
            "passed": bool(snr >= snr_threshold and r_norm > noise_floor)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_hidden_residual_graph.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/hidden_residual_graph.py block_lo_arm_order_network/tests/test_hidden_residual_graph.py
git commit -m "feat(hidden-residual): split-pass noise floor + pre-gate"
```

---

## Task 5: Hidden extraction (oracle h_v + causal h_{S_t}) + model→phys remap

**Files:**
- Create: `block_lo_arm_order_network/hidden_residual_hidden.py`
- Test: `block_lo_arm_order_network/tests/test_hidden_residual_hidden.py`

- [ ] **Step 1: Write the failing test** (tiny AOGPT on CPU; checks shapes + remap correctness)

```python
# tests/test_hidden_residual_hidden.py
import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "nanogpt-learned-order"))
from AOGPT import AOGPT, AOGPTConfig
from clean_training_protocol import build_clean_block_permutation, expand_model_blocks_to_token_order
from hidden_residual_hidden import extract_oracle_hidden, extract_causal_hidden

def _tiny_model(N=4, block_len=2, E=16):
    cfg = AOGPTConfig(vocab_size=32, n_layer=1, n_head=2, n_embd=E, dropout=0.0,
                      bias=False, block_size=N * block_len, block_order_block_len=block_len,
                      order_impl="block")
    m = AOGPT(cfg); m.eval(); return m, cfg

def test_oracle_hidden_shape_and_physical_remap():
    torch.manual_seed(0)
    N, bl, E = 4, 2, 16
    m, cfg = _tiny_model(N, bl, E)
    clean_perm = build_clean_block_permutation(N, seed=42)
    idx_model = torch.randint(0, 32, (3, N * bl))
    order_model = torch.arange(N).unsqueeze(0).expand(3, -1).contiguous()
    H = extract_oracle_hidden(m, idx_model, clean_perm, torch.device("cpu"), order_model)
    assert H.shape == (3, N, E)  # (n, N_blocks, n_embd), physical-block frame
    assert np.isfinite(H).all()

def test_causal_hidden_per_t_shape():
    torch.manual_seed(0)
    N, bl, E = 4, 2, 16
    m, cfg = _tiny_model(N, bl, E)
    clean_perm = build_clean_block_permutation(N, seed=42)
    idx_model = torch.randint(0, 32, (3, N * bl))
    canon = torch.arange(N).unsqueeze(0).expand(3, -1).contiguous()
    Hc = extract_causal_hidden(m, idx_model, clean_perm, torch.device("cpu"), canon, t_list=[0, 2])
    assert set(Hc.keys()) == {0, 2}
    assert Hc[0].shape == (3, E) and Hc[2].shape == (3, E)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_hidden_residual_hidden.py -v`
Expected: FAIL with `cannot import extract_oracle_hidden`.

- [ ] **Step 3: Write minimal implementation**

```python
# hidden_residual_hidden.py
"""Phase-1 hidden-residual diagnostic: hidden-state extraction.
oracle h_v   = full-context block representation (hidden_return_mode='original'), model->phys.
causal h_S_t = predictor hidden at block-step t under the FIXED canonical order (guard 2)."""
import numpy as np
import torch
from clean_training_protocol import expand_model_blocks_to_token_order

BLOCK_LEN = 4

@torch.no_grad()
def _block_len_from(model):
    return int(model.block_order_block_len)

@torch.no_grad()
def extract_oracle_hidden(model, idx_model, clean_perm, device, order_model_blocks):
    """Returns (n, N, E) hidden in PHYSICAL block frame. h_v = mean of block v's token hiddens."""
    model.eval()
    bl = _block_len_from(model)
    N = order_model_blocks.shape[1]
    token_order = expand_model_blocks_to_token_order(order_model_blocks, bl).to(device)
    out = model.forward_fn(idx_model.to(device), token_order, return_hidden=True,
                           hidden_return_mode="original")
    hidden = out[2]                       # (logits, loss, hidden); (n, T+1, E)
    h_tok = hidden[:, 1:, :]              # drop [None]; (n, T, E) aligned to idx_model (model frame)
    n, T, E = h_tok.shape
    h_blk_model = h_tok.reshape(n, N, bl, E).mean(dim=2).cpu().numpy()   # (n, N) model-block frame
    inv = clean_perm.inv_perm_model_to_phys.cpu().numpy()                # model block -> phys block
    h_blk_phys = np.empty_like(h_blk_model)
    h_blk_phys[:, inv, :] = h_blk_model                                  # guard 1: align to phys frame
    return h_blk_phys

@torch.no_grad()
def extract_causal_hidden(model, idx_model, clean_perm, device, canonical_order_model_blocks, t_list):
    """Returns {t: (n, E)} predictor hidden at the start of block-step t under the FIXED canonical
    order (same order for every sample -> guard 2). predictor hidden is in reveal-rank frame."""
    model.eval()
    bl = _block_len_from(model)
    token_order = expand_model_blocks_to_token_order(canonical_order_model_blocks, bl).to(device)
    out = model.forward_fn(idx_model.to(device), token_order, return_hidden=True,
                           hidden_return_mode="predictor")
    pred = out[2]                         # (n, T, E) indexed by reveal rank
    res = {}
    for t in t_list:
        rank = t * bl                     # first token rank of block-step t
        res[t] = pred[:, rank, :].cpu().numpy()
    return res
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_hidden_residual_hidden.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/hidden_residual_hidden.py block_lo_arm_order_network/tests/test_hidden_residual_hidden.py
git commit -m "feat(hidden-residual): oracle h_v + causal h_S_t extraction with phys remap"
```

---

## Task 6: Representation probe (1a) with controls

**Files:**
- Create: `block_lo_arm_order_network/hidden_residual_probe.py`
- Test: `block_lo_arm_order_network/tests/test_hidden_residual_probe.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hidden_residual_probe.py
import numpy as np
from hidden_residual_probe import representation_probe

def test_probe_beats_control_when_hidden_encodes_label():
    rng = np.random.default_rng(0)
    n, E = 400, 8
    labels = rng.integers(0, 4, size=n)
    onehot = np.eye(4)[labels]
    H = onehot @ rng.normal(0, 1, (4, E)) + rng.normal(0, 0.05, (n, E))  # hidden encodes label
    res = representation_probe(H, labels, kind="classification", seed=0)
    assert res["score"] > res["shuffled_control"] + 0.2
    assert res["score"] > 0.8

def test_probe_no_signal_when_hidden_random():
    rng = np.random.default_rng(1)
    n, E = 400, 8
    labels = rng.integers(0, 4, size=n)
    H = rng.normal(0, 1, (n, E))  # no relation
    res = representation_probe(H, labels, kind="classification", seed=0)
    assert abs(res["score"] - res["shuffled_control"]) < 0.15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_hidden_residual_probe.py -v`
Expected: FAIL with `cannot import representation_probe`.

- [ ] **Step 3: Write minimal implementation**

```python
# hidden_residual_probe.py
"""Phase-1 hidden-residual diagnostic: probes + order-effect metrics (pure numpy/sklearn)."""
import numpy as np
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.model_selection import cross_val_score

def representation_probe(H, labels, kind="classification", seed=0):
    """Cross-validated probe score for hidden H -> labels, with a shuffled-hidden control.
    kind='classification' -> accuracy; 'regression' -> R^2."""
    H = np.asarray(H, dtype=np.float64); labels = np.asarray(labels)
    if kind == "classification":
        est = LogisticRegression(max_iter=500, multi_class="auto")
        scoring = "accuracy"
    else:
        est = RidgeCV(alphas=np.logspace(-3, 3, 13))
        scoring = "r2"
    score = float(cross_val_score(est, H, labels, cv=5, scoring=scoring).mean())
    rng = np.random.default_rng(seed)
    H_shuf = H[rng.permutation(len(H))]
    control = float(cross_val_score(est, H_shuf, labels, cv=5, scoring=scoring).mean())
    return {"score": score, "shuffled_control": control, "kind": kind}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_hidden_residual_probe.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/hidden_residual_probe.py block_lo_arm_order_network/tests/test_hidden_residual_probe.py
git commit -m "feat(hidden-residual): 1a representation probe with shuffled control"
```

---

## Task 7: Residual probe (1b) — B0/B1/B2 baselines + oracle/causal heads

**Files:**
- Modify: `block_lo_arm_order_network/hidden_residual_probe.py`
- Test: `block_lo_arm_order_network/tests/test_hidden_residual_probe.py`

- [ ] **Step 1: Write the failing test**

```python
def test_residual_probe_oracle_beats_baselines_when_hidden_carries_r():
    from hidden_residual_probe import residual_probe, build_causal_features
    rng = np.random.default_rng(2)
    n, E, k = 600, 6, 4  # n samples, hidden dim E, k candidates per sample
    # planted: r depends on a hidden direction (per-sample), NOT on global/pos features
    w = rng.normal(0, 1, E)
    H = rng.normal(0, 1, (n, k, E))
    r = (H @ w) + rng.normal(0, 0.05, (n, k))
    phi_global = rng.normal(0, 1, (k, 6))            # B1: same across samples -> can't explain r
    phi_global = np.broadcast_to(phi_global, (n, k, 6))
    pos_id = np.broadcast_to(np.eye(k), (n, k, k))   # B2: position/id, same across samples
    res = residual_probe(r, H_oracle=H, phi_global=phi_global, pos_id=pos_id, seed=0)
    assert res["R2_oracle"] > 0.5
    assert res["R2_oracle"] > res["R2_B2"] + 0.2
    assert abs(res["R2_B1"]) < 0.1 and abs(res["R2_B2"]) < 0.1

def test_causal_interaction_features_shape():
    from hidden_residual_probe import build_causal_features
    rng = np.random.default_rng(3)
    n, k, E, P = 5, 4, 6, 4
    h_state = rng.normal(0, 1, (n, E))               # shared per sample
    cand_emb = rng.normal(0, 1, (n, k, P))
    feats = build_causal_features(h_state, cand_emb)
    # [emb (P), h_state broadcast (E), emb ⊙ (W h_state) (P)] -> per (n,k)
    assert feats.shape[0] == n * k and feats.shape[1] == P + E + P
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_hidden_residual_probe.py -k residual_probe or causal_interaction -v`
Expected: FAIL with `cannot import residual_probe`.

- [ ] **Step 3: Write minimal implementation (append to hidden_residual_probe.py)**

```python
def _r2_flat(X, y, seed=0):
    y_flat = np.asarray(y, np.float64).reshape(-1)
    X_flat = np.ascontiguousarray(np.asarray(X, np.float64)).reshape(len(y_flat), -1)
    est = RidgeCV(alphas=np.logspace(-3, 3, 13))
    return float(cross_val_score(est, X_flat, y_flat, cv=5, scoring="r2").mean())

def build_causal_features(h_state, cand_emb, W=None, seed=0):
    """Causal head input per candidate: [emb(v), h_S_t, emb(v) ⊙ (W h_S_t)].
    h_state:(n,E) shared across candidates; cand_emb:(n,k,P). Returns (n*k, P+E+P)."""
    h_state = np.asarray(h_state, np.float64); cand_emb = np.asarray(cand_emb, np.float64)
    n, k, P = cand_emb.shape; E = h_state.shape[1]
    if W is None:
        rng = np.random.default_rng(seed); W = rng.normal(0, 1 / np.sqrt(E), (E, P))
    Wh = h_state @ W                                  # (n, P)
    h_b = np.broadcast_to(h_state[:, None, :], (n, k, E))
    inter = cand_emb * Wh[:, None, :]                 # (n,k,P) candidate-state interaction
    feats = np.concatenate([cand_emb, h_b, inter], axis=2)  # (n,k,P+E+P)
    return feats.reshape(n * k, P + E + P)

def residual_probe(r, H_oracle, phi_global, pos_id, seed=0):
    """1b: predict residual target r (n,k) from baselines and oracle hidden.
    Returns R^2 for B0 (mean), B1 (global phi), B2 (pos/id), oracle ([phi_global, h_v])."""
    r = np.asarray(r, np.float64); n, k = r.shape
    y = r.reshape(-1)
    R2_B0 = 0.0  # mean-only predictor has R^2 = 0 by definition on held-out
    R2_B1 = _r2_flat(np.asarray(phi_global).reshape(n * k, -1), y, seed)
    R2_B2 = _r2_flat(np.asarray(pos_id).reshape(n * k, -1), y, seed)
    Xo = np.concatenate([np.asarray(phi_global), np.asarray(H_oracle)], axis=2).reshape(n * k, -1)
    R2_oracle = _r2_flat(Xo, y, seed)
    return {"R2_B0": R2_B0, "R2_B1": R2_B1, "R2_B2": R2_B2, "R2_oracle": R2_oracle}

def residual_probe_causal(r, causal_feats, seed=0):
    """1b causal head: R^2 of predicting r from build_causal_features output."""
    r = np.asarray(r, np.float64)
    return {"R2_causal": _r2_flat(causal_feats, r.reshape(-1), seed)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_hidden_residual_probe.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/hidden_residual_probe.py block_lo_arm_order_network/tests/test_hidden_residual_probe.py
git commit -m "feat(hidden-residual): 1b residual probe + B0/B1/B2 + causal interaction"
```

---

## Task 8: Order-effect metrics

**Files:**
- Modify: `block_lo_arm_order_network/hidden_residual_probe.py`
- Test: `block_lo_arm_order_network/tests/test_hidden_residual_probe.py`

- [ ] **Step 1: Write the failing test**

```python
def test_order_effect_zero_residual_no_change_and_large_residual_changes():
    from hidden_residual_probe import order_effect
    rng = np.random.default_rng(4)
    N = 64
    s_g = rng.normal(0, 1, N)
    # zero residual -> identical order
    z = order_effect(s_g, np.zeros(N))
    assert z["tau_vs_global"] > 0.999 and z["argmax_changed"] == 0
    # large residual -> order changes
    big = order_effect(s_g, rng.normal(0, 5, N))
    assert big["tau_vs_global"] < 0.95
    assert 0.0 <= big["topk_changed_ratio"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_hidden_residual_probe.py -k order_effect -v`
Expected: FAIL with `cannot import order_effect`.

- [ ] **Step 3: Write minimal implementation (append)**

```python
from scipy.stats import kendalltau

def order_effect(s_g, delta_h, top_k=8):
    """Compare descending-score orders from s_g vs s_g+delta_h (single state, N candidates)."""
    s_g = np.asarray(s_g, np.float64); delta_h = np.asarray(delta_h, np.float64)
    order_g = np.argsort(-s_g); order_h = np.argsort(-(s_g + delta_h))
    tau = float(kendalltau(order_g, order_h).correlation)
    rank_g = np.empty_like(order_g); rank_g[order_g] = np.arange(len(s_g))
    rank_h = np.empty_like(order_h); rank_h[order_h] = np.arange(len(s_g))
    displacement = float(np.abs(rank_g - rank_h).mean())
    topk_changed = float(len(set(order_g[:top_k].tolist()) ^ set(order_h[:top_k].tolist())) / (2 * top_k))
    argmax_changed = int(order_g[0] != order_h[0])
    # margin ratio: |delta_h| vs the gap between adjacent global scores
    gaps = np.abs(np.diff(np.sort(s_g)[::-1]))
    margin_ratio = float(np.abs(delta_h).mean() / (gaps.mean() + 1e-9))
    return {"tau_vs_global": tau, "mean_displacement": displacement,
            "topk_changed_ratio": topk_changed, "argmax_changed": argmax_changed,
            "margin_ratio": margin_ratio}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_hidden_residual_probe.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/hidden_residual_probe.py block_lo_arm_order_network/tests/test_hidden_residual_probe.py
git commit -m "feat(hidden-residual): order-effect metrics"
```

---

## Task 9: Orchestrator script (integration on real primary ckpt)

**Files:**
- Create: `scripts/run_hidden_residual_diag.py`

- [ ] **Step 1: Write the orchestrator**

```python
#!/usr/bin/env python3
"""Phase-1 hidden-residual order diagnostic — orchestrator.
Runs the staged pipeline (pre-gate -> 1a -> 1b -> order-effects) on a clean-line text ckpt.
NO training. See docs/superpowers/specs/2026-05-25-hidden-residual-order-diagnostic-design.md."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "block_lo_arm_order_network"))
sys.path.insert(0, str(_REPO / "nanogpt-learned-order"))

from AOGPT import AOGPT, AOGPTConfig
from clean_training_protocol import (build_clean_block_permutation, load_train_chunks,
                                     phys_to_model_idx_clean, expand_model_blocks_to_token_order)
from train_clean_aogpt import extract_A_matrices
import hidden_residual_graph as G
import hidden_residual_hidden as Hd
import hidden_residual_probe as P


def load_clean_ckpt(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    ma = ckpt["model_args"]
    sig = set(AOGPTConfig.__init__.__code__.co_varnames)
    model = AOGPT(AOGPTConfig(**{k: v for k, v in ma.items() if k in sig})).to(device)
    sd = ckpt["model"]
    if all(k.startswith("_orig_mod.") for k in sd):
        sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False); model.eval()
    seed = int(ckpt.get("config", {}).get("permute_seed", 42)) if isinstance(ckpt.get("config"), dict) else 42
    return model, seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--n-chunks", type=int, default=256)
    p.add_argument("--m-passes", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-root", default=str(_REPO / "block_lo_arm_order_network/probe_results/hidden_residual_diag"))
    args = p.parse_args()
    dev = torch.device(args.device)
    out = Path(args.out_root) / args.tag; out.mkdir(parents=True, exist_ok=True)

    model, perm_seed = load_clean_ckpt(args.ckpt, dev)
    clean_perm = build_clean_block_permutation(G.N_BLOCKS, seed=perm_seed)
    idx_phys = load_train_chunks(n_chunks=args.n_chunks)
    idx_model = phys_to_model_idx_clean(idx_phys, clean_perm)

    # M passes of attention -> per-sample A (physical frame) averaged over passes; keep halves
    passes = [extract_A_matrices(model, idx_model, clean_perm, dev, n_chunks=args.n_chunks)
              for _ in range(args.m_passes)]
    A_stack = np.stack(passes)                       # (M, n, N, N)
    A_all = A_stack.mean(0)                           # (n, N, N) all-pass per-sample
    half = args.m_passes // 2
    A_half_a = A_stack[:half].mean(0); A_half_b = A_stack[half:].mean(0)

    gate = G.gate_metrics(A_all, A_half_a, A_half_b, frame=G.PHYS_FRAME)
    (out / "gate.json").write_text(json.dumps(gate, indent=2))
    print("GATE:", json.dumps(gate, indent=2), flush=True)
    if not gate["passed"]:
        (out / "SUMMARY.md").write_text(
            f"# {args.tag}\n\nPRE-GATE FAILED (weak-model preliminary null). "
            f"r_norm={gate['r_norm']:.4g} <= noise_floor={gate['noise_floor']:.4g}. "
            "Per-sample B_x has no usable order residual on this ckpt. Not a final refutation.\n")
        print("Gate failed -> recorded preliminary null, stopping.", flush=True); return

    B_x_list, B_G, fr = G.build_B_set(A_all, frame=G.PHYS_FRAME)
    states = G.canonical_states(B_G)
    canon_order = G.rollout_order(B_G, mode="C-D+L", greedy=True, standardize=True)
    canon_model = clean_perm.block_perm_phys_to_model.numpy()[canon_order.astype(int)]
    canon_model_t = torch.tensor(canon_model, dtype=torch.long).unsqueeze(0).expand(args.n_chunks, -1).contiguous()
    order_model_full = torch.arange(G.N_BLOCKS).unsqueeze(0).expand(args.n_chunks, -1).contiguous()

    # residual target r_x per sample at each t
    targets = [G.residual_target(B_x, B_G, states, fr, fr) for B_x in B_x_list]

    # hidden
    h_oracle = Hd.extract_oracle_hidden(model, idx_model, clean_perm, dev, order_model_full)  # (n,N,E)
    h_causal = Hd.extract_causal_hidden(model, idx_model, clean_perm, dev, canon_model_t, list(states.keys()))

    # ---- 1a representation probe: predict residual sign at t=max over candidates (oracle h_v) ----
    t_top = max(states.keys()); U_top = np.asarray(states[t_top][1])
    H_uv = h_oracle[:, U_top, :].reshape(-1, h_oracle.shape[-1])             # (n*|U|, E)
    r_uv = np.concatenate([np.sign(t[t_top]["r"]) for t in targets]).astype(int)
    phase1a = {"residual_sign_oracle": P.representation_probe(H_uv, (r_uv > 0).astype(int),
                                                              kind="classification")}
    (out / "phase1a.json").write_text(json.dumps(phase1a, indent=2))
    print("PHASE1A:", json.dumps(phase1a, indent=2), flush=True)

    # ---- 1b residual probe (oracle + causal) at t_top ----
    r_mat = np.stack([t[t_top]["r"] for t in targets])                       # (n, |U|)
    Ho = h_oracle[:, U_top, :]                                               # (n, |U|, E)
    # B1 global phi: C-D+L components of B_G over U (broadcast across samples)
    from attn_order_teacher import teacher_components
    S, U, last = states[t_top]
    C, D, L, _ = teacher_components(B_G, S, U, last)
    phi_g = np.stack([C, D, L], axis=1)[None].repeat(len(targets), 0)        # (n,|U|,3)
    pos_id = np.eye(len(U))[None].repeat(len(targets), 0)                    # (n,|U|,|U|)
    res1b = P.residual_probe(r_mat, H_oracle=Ho, phi_global=phi_g, pos_id=pos_id)
    cand_emb = np.eye(len(U))[None].repeat(len(targets), 0)                  # candidate id emb
    feats_c = P.build_causal_features(h_causal[t_top], cand_emb)
    res1b.update(P.residual_probe_causal(r_mat, feats_c))
    res1b["R2_causal_over_oracle"] = (res1b["R2_causal"] / res1b["R2_oracle"]
                                      if res1b["R2_oracle"] > 1e-6 else None)
    (out / "phase1b.json").write_text(json.dumps(res1b, indent=2))
    print("PHASE1B:", json.dumps(res1b, indent=2), flush=True)

    # ---- order effect: oracle residual head prediction added to s_g ----
    # (use planted-free check: measure with the per-sample r itself as the upper-bound delta)
    eff = [P.order_effect(targets[i][t_top]["s_g"], targets[i][t_top]["r"]) for i in range(len(targets))]
    order_eff = {k: float(np.mean([e[k] for e in eff])) for k in eff[0]}
    (out / "order_effects.json").write_text(json.dumps(order_eff, indent=2))
    print("ORDER_EFFECTS:", json.dumps(order_eff, indent=2), flush=True)

    (out / "SUMMARY.md").write_text(
        f"# {args.tag}\n\nGate PASSED (snr={gate['snr']:.3g}).\n"
        f"- 1a residual-sign probe (oracle): score={phase1a['residual_sign_oracle']['score']:.3f} "
        f"vs control {phase1a['residual_sign_oracle']['shuffled_control']:.3f}\n"
        f"- 1b R2_oracle={res1b['R2_oracle']:.3f} (B1={res1b['R2_B1']:.3f}, B2={res1b['R2_B2']:.3f}); "
        f"R2_causal={res1b['R2_causal']:.3f}; causal/oracle={res1b['R2_causal_over_oracle']}\n"
        f"- order-effect (oracle-r upper bound): tau_vs_global={order_eff['tau_vs_global']:.3f}, "
        f"argmax_changed={order_eff['argmax_changed']:.3f}\n\n"
        "Preliminary (weak ckpt); null not a final refutation.\n")
    print("Wrote", out, flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run on the real primary ckpt (small n)**

Run:
```bash
cd /home/admin/lyuyuhuan/order_lyu && python scripts/run_hidden_residual_diag.py \
  --ckpt block_lo_arm_order_network/probe_results/clean_base_random_perm/ckpt_step30000.pt \
  --tag clean_random_30k_smoke --n-chunks 32 --m-passes 4 --device cpu
```
Expected: prints `GATE:` JSON with finite `r_norm`/`noise_floor`/`snr`; writes `gate.json`. If gate passes, also writes `phase1a.json`, `phase1b.json`, `order_effects.json`, `SUMMARY.md`. No exceptions.

- [ ] **Step 3: Verify output files exist and are finite**

Run:
```bash
ls block_lo_arm_order_network/probe_results/hidden_residual_diag/clean_random_30k_smoke/
python -c "import json,glob; [print(f, json.load(open(f))) for f in glob.glob('block_lo_arm_order_network/probe_results/hidden_residual_diag/clean_random_30k_smoke/*.json')]"
```
Expected: `gate.json` present with finite values; `SUMMARY.md` present.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_hidden_residual_diag.py
git commit -m "feat(hidden-residual): orchestrator + smoke on clean_base_random_perm@30k"
```

---

## Task 10: Full primary + secondary runs (analysis, not code)

**Files:** none (produces artifacts under `probe_results/hidden_residual_diag/`).

- [ ] **Step 1: Primary (clean random) full run**

```bash
cd /home/admin/lyuyuhuan/order_lyu && python scripts/run_hidden_residual_diag.py \
  --ckpt block_lo_arm_order_network/probe_results/clean_base_random_perm/ckpt_step30000.pt \
  --tag clean_random_30k --n-chunks 256 --m-passes 4 --device cuda:0
```
Expected: gate decision + (if passed) phase1a/1b/order_effects JSON.

- [ ] **Step 2: Secondary (MLP-sculpted) comparison run**

```bash
cd /home/admin/lyuyuhuan/order_lyu && python scripts/run_hidden_residual_diag.py \
  --ckpt probe_results/attention_order_mlp/alt_from0_mlp_finetune/ckpt_step30000.pt \
  --tag alt_mlp_30k --n-chunks 256 --m-passes 4 --device cuda:0
```

- [ ] **Step 3: If primary borderline, rerun n=512, M=6**

```bash
cd /home/admin/lyuyuhuan/order_lyu && python scripts/run_hidden_residual_diag.py \
  --ckpt block_lo_arm_order_network/probe_results/clean_base_random_perm/ckpt_step30000.pt \
  --tag clean_random_30k_n512m6 --n-chunks 512 --m-passes 6 --device cuda:0
```

- [ ] **Step 4: Report** R²_oracle (ceiling), R²_causal/oracle, gate metrics, order-effects; compare primary vs secondary (does MLP-order training sculpt the residual?). Record conclusions; null = weak-model preliminary only.
