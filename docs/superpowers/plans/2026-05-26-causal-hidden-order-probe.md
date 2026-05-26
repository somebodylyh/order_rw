# Causal-Hidden Order Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a frozen, falsifiable diagnostic that decides whether per-step CAUSAL hidden (the model's predictor state given the revealed prefix `S_t`) carries order signal beyond the static `B_A` C-D+L order — on text and image checkpoints, with no training.

**Architecture:** One model-coupled module `causal_hidden_rollout.py` (causal-invariance PRE-GATE, per-step predictor-hidden extraction, candidate embedding, two param-free dynamic scores, a per-step γ-mix greedy rollout, controls, Level-2 oracle, per-sample audit) wired by one driver `scripts/run_causal_hidden_probe.py` that mirrors `scripts/run_hidden_graph_diag.py`. The driver builds modality-specific closures (text uses `clean_perm`; image uses `fixed_token_perm`/`inv_block_perm`) so the rollout core is modality-agnostic. Reuses, unchanged: `graph_normalize`, `graph_order`, `attn_order_teacher`, `position_graph`, `hidden_graph_modelio` (NLL), `hidden_residual_*`, `train_clean_aogpt`/`train_vq64_round2` helpers.

**Tech Stack:** Python 3.8, NumPy, PyTorch 2.4.1, pytest 8.3.5. Repo root `/home/admin/lyuyuhuan/order_lyu`. Interpreter `/home/admin/anaconda3/envs/X1/bin/python`. Modules in `block_lo_arm_order_network/`, tests in `block_lo_arm_order_network/tests/`, driver in `scripts/`. Spec: `docs/superpowers/specs/2026-05-26-causal-hidden-order-probe-design.md`.

**Conventions (verified):** `N_BLOCKS=64`. Block model = `AOGPT(AOGPTConfig(order_impl="block", block_order_block_len=BL, block_size=N*BL, vocab_size=V, n_layer, n_head, n_embd, dropout=0.0, bias=False))` (dispatches to `AOGPT_block.BlockAOGPT`). `model.forward_fn(idx, token_order, return_hidden=True, hidden_return_mode="predictor")` returns a tuple whose `[2]` is `(n, T, E)` predictor hidden; the predictor state conditioned on the first `t` revealed blocks is at token-rank `t*BL`: `out[2][:, t*BL, :]`. `model.transformer.wte` is the `(vocab, n_embd)` token embedding. `model.block_order_block_len` is `BL`. `expand_model_blocks_to_token_order(model_block_orders, BL)` and `physical_blocks_to_model_blocks(phys, clean_perm)` are importable from `train_clean_aogpt` (re-exported from `clean_training_protocol`). `attn_order_teacher.teacher_scores(B, S_t, U_t, last, mode="C-D+L")` returns `(q, cand)`. `graph_order.cdl_order(B, greedy=True)` returns the static C-D+L phys order. **Tests must add `block_lo_arm_order_network`, `nanogpt-learned-order`, AND `scripts/` to `sys.path`** (the T6 gotcha — modelio imports `train_vq64_round2` from `scripts/`).

Run all tests from repo root: `cd /home/admin/lyuyuhuan/order_lyu && /home/admin/anaconda3/envs/X1/bin/python -m pytest block_lo_arm_order_network/tests/<file> -v`.

---

## REVISION (2026-05-27, after CT1 PRE-GATE — see spec §0.5)

CT1 found this is a **target-aware AO** architecture: the predictor hidden at rank `t*BL` is
AdaLN-conditioned on the next target position, so `c_t = c_t(S_t, v)` where `v=σ(t+1)` (verified:
fix `(S_t,v)`, vary rest → bit-identical; vary `v` → changes). This **supersedes** the tasks below:

- **Path X (MAIN, the verdict gate):** candidate-conditioned `c_t^{(v)} = context_hidden_at_step(S_t,
  completion=[v]+rest)` (the existing primitive already computes this — `completion[0]` IS `σ(t+1)`).
  Score `s_H(v) = paired_cos(C, E_cand)` where `C[j]=c_t^{(U[j])}`, `E_cand[j]=e_{U[j]}` (per-candidate
  cosine), or qk. **FORBID true-token-likelihood scoring** (Level-2 circularity).
- **Path Y (CONTROL only, NOT the gate):** target-neutral `p_t = pool_{u∈S_t}(partial-context original
  hidden)`; `s_H(v)=cos(p_t, e_v)` (shared context). **A Y NULL does not close the line.**
- **Reframed PRE-GATE (Task 1):** test `c_t^{(v)}` is invariant to `σ(t+2..)` holding `(S_t, v)` fixed
  (passes). NOT the old "completion-invariant `c_t`".
- **Affected tasks:** Task 1 (invariance check semantics + add Path-Y pooled helper), Task 3 (add
  `paired_cos`), Task 4 (rollout: X = per-candidate `c_t^{(v)}`; Y = shared `p_t` control), Task 6/8
  (X is the gate; Y reported as control; cost-controlled X: text first, `n_roll∈{8,16}`, eval 128–256,
  `γ∈{0,.5,1}`, cos first). Implement Y first as a cheap pipeline smoke, then X.

## File Structure

| File | Responsibility |
|---|---|
| `block_lo_arm_order_network/causal_hidden_rollout.py` | invariance check; per-step predictor hidden; candidate embedding; cos/attn dynamic scores; per-step position-residualize; γ-mix greedy rollout (+ controls/dispersion); Level-2 oracle; per-sample audit |
| `scripts/run_causal_hidden_probe.py` | driver: load ckpt → PRE-GATE (abort on fail) → Level-0/1/2 + audit → NLL eval → §10 verdict + decision → JSON/MD report; modality closures |
| `block_lo_arm_order_network/tests/test_causal_hidden_rollout.py` | smoke tests on a tiny BlockAOGPT (CPU) |

The module exposes a small, pure-where-possible surface so the driver only orchestrates. Functions take a model + numpy graphs + closures and return numpy/int arrays.

---

## Task 1: `causal_hidden_rollout.py` — per-step predictor hidden + causal-invariance PRE-GATE

**Files:**
- Create: `block_lo_arm_order_network/causal_hidden_rollout.py`
- Test: `block_lo_arm_order_network/tests/test_causal_hidden_rollout.py`

Spec §3, §4. `context_hidden_at_step` runs ONE forward with `[prefix + completion]` and reads the predictor hidden at rank `t*BL` (conditioned only on the `t`-block prefix). `causal_invariance_check` verifies that reading is completion-invariant (the precondition of the whole rollout).

- [ ] **Step 1: Write the failing test**

```python
# block_lo_arm_order_network/tests/test_causal_hidden_rollout.py
import numpy as np, torch, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in ["block_lo_arm_order_network", "nanogpt-learned-order", "scripts"]:
    sys.path.insert(0, str(ROOT / p))
from AOGPT import AOGPT, AOGPTConfig
import causal_hidden_rollout as CHR


def _tiny(N=4, BL=2, V=16, E=8):
    cfg = AOGPTConfig(order_impl="block", block_order_block_len=BL, block_size=N * BL,
                      vocab_size=V, n_layer=1, n_head=1, n_embd=E, dropout=0.0, bias=False)
    m = AOGPT(cfg); m.eval(); return m


def test_context_hidden_at_step_shape():
    torch.manual_seed(0); N, BL, V, E, n = 4, 2, 16, 8, 3
    m = _tiny(N, BL, V, E)
    idx = torch.randint(0, V, (n, N * BL))
    prefix = [0, 1]                                   # model-frame blocks revealed
    completion = [2, 3]
    c = CHR.context_hidden_at_step(m, idx, prefix, completion, BL, device="cpu")
    assert c.shape == (n, E)
    assert np.all(np.isfinite(c))


def test_context_hidden_is_completion_invariant():
    torch.manual_seed(0); N, BL, V, E, n = 4, 2, 16, 8, 3
    m = _tiny(N, BL, V, E)
    idx = torch.randint(0, V, (n, N * BL))
    prefix = [2, 0]
    c1 = CHR.context_hidden_at_step(m, idx, prefix, [1, 3], BL, device="cpu")
    c2 = CHR.context_hidden_at_step(m, idx, prefix, [3, 1], BL, device="cpu")
    cos = (c1 * c2).sum(1) / (np.linalg.norm(c1, axis=1) * np.linalg.norm(c2, axis=1) + 1e-12)
    assert float(cos.min()) > 0.99999


def test_causal_invariance_check_passes_on_tiny():
    torch.manual_seed(0); N, BL, V, E = 4, 2, 16, 8
    m = _tiny(N, BL, V, E)
    idx = torch.randint(0, V, (5, N * BL))
    rep = CHR.causal_invariance_check(m, idx, N, BL, device="cpu",
                                      t_list=(0, 1, 2, 3), n_patterns=3, seed=0)
    assert rep["passed"] is True
    assert rep["min_cosine"] > 0.99999
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/admin/lyuyuhuan/order_lyu && /home/admin/anaconda3/envs/X1/bin/python -m pytest block_lo_arm_order_network/tests/test_causal_hidden_rollout.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'causal_hidden_rollout'`.

- [ ] **Step 3: Write minimal implementation**

```python
# block_lo_arm_order_network/causal_hidden_rollout.py
"""Frozen causal-hidden order probe: per-step predictor hidden, candidate embedding,
param-free dynamic scores (cos-emb + model-attn), per-step gamma-mix greedy rollout,
controls, Level-2 oracle, per-sample audit. NO training. See
docs/superpowers/specs/2026-05-26-causal-hidden-order-probe-design.md."""
import numpy as np
import torch


def _expand_blocks_to_tokens(block_order, block_len):
    """block_order: list/1d int (model-frame blocks) -> 1d token order (len*block_len)."""
    bo = np.asarray(block_order, dtype=np.int64)
    base = np.arange(block_len, dtype=np.int64)
    return (bo[:, None] * block_len + base[None, :]).reshape(-1)


@torch.no_grad()
def context_hidden_at_step(model, idx_model, prefix_blocks, completion_blocks, block_len,
                           device, chunk_size=64):
    """Predictor hidden c_t conditioned on the prefix (model-frame blocks), read at rank
    t*block_len. completion_blocks fills positions after the prefix; by causal masking c_t
    must not depend on it (see causal_invariance_check). Returns (n, E)."""
    model.eval()
    t = len(prefix_blocks)
    full = list(prefix_blocks) + list(completion_blocks)
    tok_order = torch.as_tensor(_expand_blocks_to_tokens(full, block_len),
                                dtype=torch.long, device=device)
    rank = t * block_len
    n_total = idx_model.shape[0]
    out_chunks = []
    for i in range(0, n_total, chunk_size):
        idx = idx_model[i:i + chunk_size].to(device)
        order = tok_order.unsqueeze(0).expand(idx.shape[0], -1)
        out = model.forward_fn(idx, order, return_hidden=True, hidden_return_mode="predictor")
        out_chunks.append(out[2][:, rank, :].float().cpu().numpy())
    return np.concatenate(out_chunks, axis=0)


def causal_invariance_check(model, idx_model, n_blocks, block_len, device,
                            t_list=(0, 1, 4, 16, 32, 48, 63), n_patterns=4, seed=0):
    """PRE-GATE: c_t must depend only on the prefix S_t, not the completion. For each t and
    several prefix patterns, compare c_t under >=2 completions. Pass iff min cosine > 0.99999."""
    rng = np.random.default_rng(seed)
    min_cos, max_absdiff, fails = 1.0, 0.0, []
    for t in t_list:
        if t >= n_blocks:
            continue
        for _ in range(n_patterns):
            perm = rng.permutation(n_blocks)
            prefix = perm[:t].tolist()
            rest = perm[t:].tolist()
            comp_a = rest
            comp_b = list(reversed(rest))
            comp_c = rng.permutation(rest).tolist() if len(rest) > 1 else rest
            c_a = context_hidden_at_step(model, idx_model, prefix, comp_a, block_len, device)
            for comp in (comp_b, comp_c):
                c_b = context_hidden_at_step(model, idx_model, prefix, comp, block_len, device)
                num = (c_a * c_b).sum(1)
                den = np.linalg.norm(c_a, axis=1) * np.linalg.norm(c_b, axis=1) + 1e-12
                cos = float((num / den).min())
                ad = float(np.abs(c_a - c_b).max())
                min_cos = min(min_cos, cos); max_absdiff = max(max_absdiff, ad)
                if cos <= 0.99999:
                    fails.append({"t": int(t), "cosine": cos, "max_absdiff": ad})
    return {"passed": len(fails) == 0, "min_cosine": float(min_cos),
            "max_absdiff": float(max_absdiff), "n_fail": len(fails), "fails": fails[:10]}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/admin/lyuyuhuan/order_lyu && /home/admin/anaconda3/envs/X1/bin/python -m pytest block_lo_arm_order_network/tests/test_causal_hidden_rollout.py -v`
Expected: PASS (3 tests). If `context_hidden_at_step` returns the wrong rank (off-by-one), confirm the `t*BL` indexing against `hidden_residual_hidden.extract_causal_hidden` (line 61: `rank = t*bl`).

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/causal_hidden_rollout.py block_lo_arm_order_network/tests/test_causal_hidden_rollout.py
git commit -m "feat(causal-hidden): per-step predictor hidden + causal-invariance PRE-GATE"
```

---

## Task 2: candidate embedding (content-aware token-only + content-free)

**Files:**
- Modify: `block_lo_arm_order_network/causal_hidden_rollout.py`
- Test: `block_lo_arm_order_network/tests/test_causal_hidden_rollout.py`

Spec §5.4. Content-aware = mean of the candidate block's input **token** embeddings (`wte` only, NO positional). Content-free = the block-id positional embedding (`wpe`) for the block's first position — no candidate content.

- [ ] **Step 1: Write the failing test (append)**

```python
def test_candidate_embeddings_content_token_shape_and_no_pos():
    torch.manual_seed(0); N, BL, V, E, n = 4, 2, 16, 8, 3
    m = _tiny(N, BL, V, E)
    idx = torch.randint(0, V, (n, N * BL))
    # model-frame block ids 0..N-1 ; content-aware uses idx token embeddings
    E_cand = CHR.candidate_embeddings(m, idx, list(range(N)), BL, mode="content_token", device="cpu")
    assert E_cand.shape == (n, N, E)
    # token-only: equals mean wte over the block's tokens (no wpe added)
    blk0_tokens = idx[:, 0:BL]                               # model-frame block 0 tokens
    expect0 = m.transformer.wte(blk0_tokens).mean(dim=1).detach().numpy()
    assert np.allclose(E_cand[:, 0, :], expect0, atol=1e-5)


def test_candidate_embeddings_content_free_is_sample_invariant():
    torch.manual_seed(0); N, BL, V, E, n = 4, 2, 16, 8, 3
    m = _tiny(N, BL, V, E)
    idx = torch.randint(0, V, (n, N * BL))
    E_free = CHR.candidate_embeddings(m, idx, list(range(N)), BL, mode="content_free", device="cpu")
    assert E_free.shape == (n, N, E)
    # content-free does not depend on token content -> identical across samples
    assert np.allclose(E_free[0], E_free[1]) and np.allclose(E_free[0], E_free[2])
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `... -m pytest block_lo_arm_order_network/tests/test_causal_hidden_rollout.py -k candidate -v`
Expected: FAIL with `AttributeError: module 'causal_hidden_rollout' has no attribute 'candidate_embeddings'`.

- [ ] **Step 3: Append implementation**

```python
@torch.no_grad()
def candidate_embeddings(model, idx_model, model_block_ids, block_len, mode, device):
    """Returns (n, len(model_block_ids), E) candidate embeddings in model embedding space.
    mode='content_token': mean of the block's input token embeddings (wte only, NO positional)
                          -> content-aware (sees candidate content; label reports accordingly).
    mode='content_free' : the block's first-position positional embedding (wpe), broadcast over
                          samples -> no candidate content."""
    model.eval()
    idx = idx_model.to(device)
    n = idx.shape[0]
    bids = list(model_block_ids)
    if mode == "content_token":
        cols = []
        for b in bids:
            toks = idx[:, b * block_len:(b + 1) * block_len]          # (n, BL) model-frame
            cols.append(model.transformer.wte(toks).mean(dim=1))      # (n, E) token-only
        E_cand = torch.stack(cols, dim=1)                             # (n, M, E)
        return E_cand.float().cpu().numpy()
    if mode == "content_free":
        pos = torch.tensor([b * block_len + 1 for b in bids], device=device)  # +1: [None] offset
        emb = model.transformer.wpe(pos)                              # (M, E)
        return emb.unsqueeze(0).expand(n, -1, -1).float().cpu().numpy()
    raise ValueError(f"unknown mode {mode}")
```

- [ ] **Step 4: Run to verify pass**

Run: `... -m pytest block_lo_arm_order_network/tests/test_causal_hidden_rollout.py -k candidate -v`
Expected: PASS (2). If `wpe` index out of range, drop the `+1` (the `[None]` token uses `wnonee`; positions index `wpe` directly) and re-confirm against `AOGPT_block` forward.

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/causal_hidden_rollout.py block_lo_arm_order_network/tests/test_causal_hidden_rollout.py
git commit -m "feat(causal-hidden): candidate embeddings (content-token wte-only / content-free wpe)"
```

---

## Task 3: dynamic scores — cos-emb (required) + model-attn (best-effort)

**Files:**
- Modify: `block_lo_arm_order_network/causal_hidden_rollout.py`
- Test: `block_lo_arm_order_network/tests/test_causal_hidden_rollout.py`

Spec §5.1, §5.2. cos-emb scores candidates by cosine between the aggregated context vector and each candidate embedding. model-attn uses the model's own `c_attn` projections (split q,k,v; RMSNorm on q,k per head) to compute `q(p)·k(e_v)/sqrt(d_h)`, aggregated over heads (mean here; the driver restricts to top-4-variance heads to match `A_global`). Both operate on already-aggregated `(E,)` context and `(M, E)` candidate matrices (aggregation over rollout samples happens in the rollout).

- [ ] **Step 1: Write the failing test (append)**

```python
def test_dynamic_score_cos_direction():
    p = np.array([1.0, 0.0, 0.0])
    E_cand = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    s = CHR.dynamic_score_cos(p, E_cand)
    assert s.shape == (3,)
    assert s[0] > s[1] > s[2]                       # aligned > orthogonal > anti-aligned
    assert np.isclose(s[0], 1.0) and np.isclose(s[2], -1.0)


def test_dynamic_score_attn_shape_and_finite():
    torch.manual_seed(0); N, BL, V, E = 4, 2, 16, 8
    m = _tiny(N, BL, V, E)
    p = np.random.default_rng(0).standard_normal(E).astype(np.float32)
    E_cand = np.random.default_rng(1).standard_normal((N, E)).astype(np.float32)
    s = CHR.dynamic_score_attn(m, p, E_cand, layer=0, heads=None)
    assert s.shape == (N,)
    assert np.all(np.isfinite(s))
```

- [ ] **Step 2: Run to verify fail**

Run: `... -k "dynamic_score" -v`
Expected: FAIL (`has no attribute 'dynamic_score_cos'`).

- [ ] **Step 3: Append implementation**

```python
def dynamic_score_cos(p, E_cand):
    """p: (E,) context vector; E_cand: (M, E). Returns (M,) cosine similarity."""
    p = np.asarray(p, dtype=np.float64); E_cand = np.asarray(E_cand, dtype=np.float64)
    pn = p / (np.linalg.norm(p) + 1e-12)
    cn = E_cand / (np.linalg.norm(E_cand, axis=1, keepdims=True) + 1e-12)
    return (cn @ pn).astype(np.float64)


@torch.no_grad()
def dynamic_score_attn(model, p, E_cand, layer=0, heads=None):
    """Model-attn (best-effort): q(p).k(e_v)/sqrt(d_h) using the model's own c_attn + q/k RMSNorm.
    p: (E,), E_cand: (M, E). heads: iterable of head indices to average (None=all). Pre-softmax
    logits, mean over heads. Returns (M,)."""
    attn = model.transformer.h[layer].attn
    nh, E = attn.n_head, attn.n_embd
    hs = E // nh
    dev = next(model.parameters()).device
    P = torch.as_tensor(np.asarray(p), dtype=torch.float32, device=dev).view(1, 1, E)
    C = torch.as_tensor(np.asarray(E_cand), dtype=torch.float32, device=dev).unsqueeze(0)  # (1,M,E)
    q = attn.c_attn(P).split(E, dim=2)[0].view(1, 1, nh, hs).transpose(1, 2)               # (1,nh,1,hs)
    k = attn.c_attn(C).split(E, dim=2)[1].view(1, C.shape[1], nh, hs).transpose(1, 2)       # (1,nh,M,hs)
    q, k = attn.q_norm(q), attn.k_norm(k)
    logits = (q @ k.transpose(-2, -1)).squeeze(2).squeeze(0) / (hs ** 0.5)                   # (nh, M)
    if heads is not None:
        logits = logits[list(heads)]
    return logits.mean(dim=0).float().cpu().numpy()
```

- [ ] **Step 4: Run to verify pass**

Run: `... -k "dynamic_score" -v`
Expected: PASS (2). If `c_attn` split order differs (q,k,v vs other), confirm against `AOGPT_block.py:67` (`q, k, v = self.c_attn(x).split(self.n_embd, dim=2)`).

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/causal_hidden_rollout.py block_lo_arm_order_network/tests/test_causal_hidden_rollout.py
git commit -m "feat(causal-hidden): cos-emb + model-attn (qk) dynamic scores"
```

---

## Task 4: per-step γ-mix greedy rollout (with γ=0 ≡ static B_A C-D+L)

**Files:**
- Modify: `block_lo_arm_order_network/causal_hidden_rollout.py`
- Test: `block_lo_arm_order_network/tests/test_causal_hidden_rollout.py`

Spec §3, §6, §14.1, §14.2, §14.3. The rollout builds ONE shared phys-frame order greedily. At each step it gets the aggregated context `p_t` and candidate embeddings via injected closures (so the module stays modality-agnostic), computes `s_H` (cos or attn), **per-step** position-residualizes it against the candidate `B_pos` slice, z-scores both branches, and mixes `s = z(s_CDL) + γ·z(s_H_resid)`. `t=0` start node uses A-only (hidden branch OFF), so `γ=0` is bit-identical to `cdl_order(B_A)`.

- [ ] **Step 1: Write the failing test (append)**

```python
import graph_order as GO
from graph_normalize import shift_nonneg


def _ring(N, w=5.0):
    B = np.zeros((N, N))
    for i in range(N):
        B[i, (i + 1) % N] = w
    return B + B.T


def test_gamma0_equals_static_cdl_order():
    N = 6
    B_A = _ring(N, 5.0)
    B_pos = _ring(N, 1.0)
    # context closure returns zeros -> s_H all-equal; even so gamma=0 must ignore s_H entirely
    ctx_fn = lambda prefix, U: np.zeros(8)
    cand_fn = lambda U: np.zeros((len(U), 8))
    order0 = CHR.causal_score_mix_rollout(B_A, B_pos, gamma=0.0, ctx_fn=ctx_fn, cand_fn=cand_fn,
                                          score_kind="cos", hidden_off_at_t0=True)
    assert np.array_equal(order0, GO.cdl_order(B_A, greedy=True))


def test_gamma_changes_order_when_hidden_disagrees():
    N = 6
    B_A = _ring(N, 5.0)
    B_pos = np.zeros((N, N))
    # context/candidate fixed vectors that make cos prefer a specific block each step
    ctx_fn = lambda prefix, U: np.array([1.0, 0.0])
    def cand_fn(U):
        E = np.zeros((len(U), 2))
        for j, v in enumerate(U):
            E[j] = [1.0, 0.0] if v == max(U) else [-1.0, 0.0]   # cos prefers the largest-id candidate
        return E
    o0 = CHR.causal_score_mix_rollout(B_A, B_pos, 0.0, ctx_fn, cand_fn, "cos", hidden_off_at_t0=True)
    o2 = CHR.causal_score_mix_rollout(B_A, B_pos, 2.0, ctx_fn, cand_fn, "cos", hidden_off_at_t0=True)
    assert not np.array_equal(o0, o2)
```

- [ ] **Step 2: Run to verify fail**

Run: `... -k "gamma" -v`
Expected: FAIL (`has no attribute 'causal_score_mix_rollout'`).

- [ ] **Step 3: Append implementation**

```python
from attn_order_teacher import teacher_scores
from graph_normalize import shift_nonneg as _shift_nonneg

MODE = "C-D+L"


def _zscore(v):
    v = np.asarray(v, dtype=np.float64)
    sd = v.std()
    return (v - v.mean()) / (sd if sd > 0 else 1.0)


def _residualize_vec(y, x):
    """OLS residual of y on [1, x] (both length-M); returns y - fit."""
    y = np.asarray(y, dtype=np.float64); x = np.asarray(x, dtype=np.float64)
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def causal_score_mix_rollout(B_A, B_pos, gamma, ctx_fn, cand_fn, score_kind,
                             hidden_off_at_t0=True, model=None, attn_layer=0, attn_heads=None,
                             record=None):
    """Shared greedy phys-frame order. Per step:
      s_CDL = teacher_scores(shift_nonneg(B_A), S, U, last, 'C-D+L')   (A-only branch)
      p_t   = ctx_fn(S, U)          # aggregated context vector (E,)   [from injected closure]
      E_U   = cand_fn(U)            # candidate embeddings (|U|, E)
      s_H   = dynamic_score_cos|attn ; s_H_resid = residualize(s_H, B_pos[last,U]) per step
      score = z(s_CDL) + gamma * z(s_H_resid)        (gamma=0 -> A-only exactly)
    hidden_off_at_t0: at t=0 (S empty) use A-only only (no hidden branch) so the start node and
    the gamma=0 path match cdl_order(B_A) exactly."""
    A = _shift_nonneg(B_A)
    N = A.shape[0]
    S, U, last, order = [], list(range(N)), None, []
    for t in range(N):
        if len(U) == 1:
            v = U[0]
        else:
            qa, cand = teacher_scores(A, S, U, last, mode=MODE)
            cand = list(cand)
            score = _zscore(qa)
            use_hidden = gamma != 0.0 and not (hidden_off_at_t0 and t == 0)
            if use_hidden:
                p_t = ctx_fn(S, cand)
                E_U = cand_fn(cand)
                if score_kind == "cos":
                    s_H = dynamic_score_cos(p_t, E_U)
                else:
                    s_H = dynamic_score_attn(model, p_t, E_U, layer=attn_layer, heads=attn_heads)
                pos_vec = (B_pos[last, cand] if last is not None
                           else np.zeros(len(cand), dtype=np.float64))
                s_H_resid = _residualize_vec(s_H, pos_vec)
                score = score + gamma * _zscore(s_H_resid)
                if record is not None:
                    record.append({"t": t, "s_H_std": float(np.std(s_H)),
                                   "score_argmax": int(cand[int(np.argmax(score))])})
            v = int(cand[int(np.argmax(score))])
        order.append(v); S.append(v); U.remove(v); last = v
    return np.asarray(order, dtype=np.int64)
```

- [ ] **Step 4: Run to verify pass**

Run: `... -k "gamma" -v`
Expected: PASS (2). The `γ=0 ≡ cdl_order(B_A)` test is the load-bearing regression (spec §14.2). If it fails, the A-only branch's `teacher_scores`/`shift_nonneg` does not match `graph_order.cdl_order` — align them before proceeding.

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/causal_hidden_rollout.py block_lo_arm_order_network/tests/test_causal_hidden_rollout.py
git commit -m "feat(causal-hidden): per-step gamma-mix greedy rollout (gamma0 == static C-D+L)"
```

---

## Task 5: controls (position / shuffled-hidden) + Level-2 oracle + per-sample audit

**Files:**
- Modify: `block_lo_arm_order_network/causal_hidden_rollout.py`
- Test: `block_lo_arm_order_network/tests/test_causal_hidden_rollout.py`

Spec §6 (controls), §7 (Level-2 oracle, per-sample audit), §14.4 (dispersion). Position-control replaces `s_H` with the pure `B_pos` candidate slice. Shuffled-hidden replaces `s_H_resid` with a seeded permutation of itself. Level-2 oracle greedily picks the candidate with lowest true next-block NLL given the prefix (diagnostic-only). Per-sample audit returns order diversity over a few samples.

- [ ] **Step 1: Write the failing test (append)**

```python
def test_position_control_order_is_permutation():
    N = 6; B_A = _ring(N, 5.0); B_pos = _ring(N, 1.0)
    o = CHR.position_control_rollout(B_A, B_pos, gamma=1.0)
    assert sorted(o.tolist()) == list(range(N))


def test_shuffled_hidden_differs_from_real_with_fixed_inputs():
    N = 6; B_A = _ring(N, 5.0); B_pos = np.zeros((N, N))
    ctx_fn = lambda S, U: np.array([1.0, 0.0])
    cand_fn = lambda U: np.stack([[1.0, 0.0] if v == max(U) else [0.0, 1.0] for v in U]).astype(float)
    o_real = CHR.causal_score_mix_rollout(B_A, B_pos, 1.0, ctx_fn, cand_fn, "cos")
    o_shuf = CHR.shuffled_hidden_rollout(B_A, B_pos, 1.0, ctx_fn, cand_fn, "cos", seed=0)
    assert sorted(o_shuf.tolist()) == list(range(N))
    # with a strong, consistent hidden signal, shuffling should usually change the order
    assert not np.array_equal(o_real, o_shuf)


def test_order_diversity_metric():
    orders = np.array([[0, 1, 2, 3], [0, 1, 2, 3], [3, 2, 1, 0]])
    d = CHR.order_diversity(orders)
    assert 0.0 <= d["mean_pairwise_kendall_tau"] <= 1.0
    assert d["frac_unique"] == 2 / 3
```

- [ ] **Step 2: Run to verify fail**

Run: `... -k "control or shuffled or diversity" -v`
Expected: FAIL (missing attributes).

- [ ] **Step 3: Append implementation**

```python
def position_control_rollout(B_A, B_pos, gamma):
    """Control: s_H replaced by the pure position slice B_pos[last, U] (no hidden)."""
    ctx_fn = lambda S, U: None
    cand_fn = lambda U: None

    A = _shift_nonneg(B_A); N = A.shape[0]
    S, U, last, order = [], list(range(N)), None, []
    for t in range(N):
        if len(U) == 1:
            v = U[0]
        else:
            qa, cand = teacher_scores(A, S, U, last, mode=MODE); cand = list(cand)
            score = _zscore(qa)
            if gamma != 0.0 and last is not None:
                score = score + gamma * _zscore(B_pos[last, cand])
            v = int(cand[int(np.argmax(score))])
        order.append(v); S.append(v); U.remove(v); last = v
    return np.asarray(order, dtype=np.int64)


def shuffled_hidden_rollout(B_A, B_pos, gamma, ctx_fn, cand_fn, score_kind, seed=0,
                            model=None, attn_layer=0, attn_heads=None):
    """Control: same pipeline but s_H_resid is permuted (value-multiset matched) each step."""
    rng = np.random.default_rng(seed)
    A = _shift_nonneg(B_A); N = A.shape[0]
    S, U, last, order = [], list(range(N)), None, []
    for t in range(N):
        if len(U) == 1:
            v = U[0]
        else:
            qa, cand = teacher_scores(A, S, U, last, mode=MODE); cand = list(cand)
            score = _zscore(qa)
            if gamma != 0.0 and last is not None:
                p_t = ctx_fn(S, cand); E_U = cand_fn(cand)
                s_H = (dynamic_score_cos(p_t, E_U) if score_kind == "cos"
                       else dynamic_score_attn(model, p_t, E_U, layer=attn_layer, heads=attn_heads))
                s_H_resid = _residualize_vec(s_H, B_pos[last, cand])
                perm = rng.permutation(len(s_H_resid))
                score = score + gamma * _zscore(s_H_resid[perm])
            v = int(cand[int(np.argmax(score))])
        order.append(v); S.append(v); U.remove(v); last = v
    return np.asarray(order, dtype=np.int64)


def _kendall_tau(a, b):
    a = np.asarray(a); b = np.asarray(b); n = len(a)
    ra = np.empty(n); ra[a] = np.arange(n)
    rb = np.empty(n); rb[b] = np.arange(n)
    conc = 0; disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(ra[i] - ra[j]) * np.sign(rb[i] - rb[j])
            conc += s > 0; disc += s < 0
    tot = n * (n - 1) / 2
    return (conc - disc) / tot if tot > 0 else 1.0


def order_diversity(orders):
    """orders: (k, N). Returns mean pairwise (1-|tau|)/normalized diversity + unique fraction."""
    orders = np.asarray(orders); k = orders.shape[0]
    taus = [abs(_kendall_tau(orders[i], orders[j])) for i in range(k) for j in range(i + 1, k)]
    uniq = len({tuple(o.tolist()) for o in orders})
    return {"mean_pairwise_kendall_tau": float(np.mean(taus)) if taus else 1.0,
            "frac_unique": uniq / k}


def oracle_nll_greedy_order(per_step_nll_fn, n_blocks):
    """Level-2 DIAGNOSTIC ONLY (not in WIN gate, not a teacher). Greedily pick the candidate with
    the lowest true next-block NLL given the prefix. per_step_nll_fn(prefix, U) -> dict v->nll."""
    S, U, order = [], list(range(n_blocks)), []
    for _ in range(n_blocks):
        if len(U) == 1:
            v = U[0]
        else:
            nlls = per_step_nll_fn(S, U)
            v = min(U, key=lambda c: nlls[c])
        order.append(v); S.append(v); U.remove(v)
    return np.asarray(order, dtype=np.int64)
```

- [ ] **Step 4: Run to verify pass**

Run: `... -k "control or shuffled or diversity" -v`
Expected: PASS (3).

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/causal_hidden_rollout.py block_lo_arm_order_network/tests/test_causal_hidden_rollout.py
git commit -m "feat(causal-hidden): position/shuffled-hidden controls + order diversity + L2 oracle helper"
```

---

## Task 6: driver `run_causal_hidden_probe.py` — text path + PRE-GATE + smoke

**Files:**
- Create: `scripts/run_causal_hidden_probe.py`
- Reference (mirror): `scripts/run_hidden_graph_diag.py`

The driver builds modality closures (`ctx_fn`, `cand_fn`, `nll_fn`) and orchestrates PRE-GATE → Level-0/1 → verdict → report. Its check is a CLI smoke on small sizes, not a unit test. Build incrementally.

- [ ] **Step 1: Write loading + closures (text)**

```python
#!/usr/bin/env python3
"""Causal-Hidden Order Probe (frozen, no training). Per-checkpoint report: PRE-GATE causal
invariance; Level-0 baselines; Level-1 dynamic gamma-mix (cos-emb [+ model-attn]) vs A-only with
position + shuffled-hidden controls; WIN gate; decision. See
docs/superpowers/specs/2026-05-26-causal-hidden-order-probe-design.md."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
for p in ["block_lo_arm_order_network", "nanogpt-learned-order", "scripts"]:
    sys.path.insert(0, str(_REPO / p))

import causal_hidden_rollout as CHR
import position_graph as PG
import graph_order as GO
import hidden_graph_modelio as MIO


def build_text(args, device):
    from clean_training_protocol import build_clean_block_permutation, phys_to_model_idx_clean
    from training_utils import load_train_chunks
    from train_clean_aogpt import extract_A_matrices, physical_blocks_to_model_blocks
    import hidden_residual_graph as G
    from run_hidden_residual_diag import load_clean_ckpt

    model, perm_seed = load_clean_ckpt(args.ckpt, device)
    BL = int(model.block_order_block_len)
    clean_perm = build_clean_block_permutation(G.N_BLOCKS, seed=perm_seed)
    idx_phys = load_train_chunks(n_chunks=args.n_chunks)
    idx_model = phys_to_model_idx_clean(idx_phys, clean_perm)
    idx_roll = idx_model[:args.n_roll]

    A_all = extract_A_matrices(model, idx_model, clean_perm, device, n_chunks=args.n_chunks)
    _, B_A, _ = G.build_B_set(A_all, frame=G.PHYS_FRAME)
    B_pos = PG.text_position_graph(N=G.N_BLOCKS, tau=args.pos_tau)

    # phys-frame blocks (rollout/B_A frame) -> model-frame blocks (what the model is fed).
    # Reuse the SAME helper nll_fn uses, so framing is consistent (NOT inv_perm_model_to_phys,
    # which is the opposite direction — a classic coordinate bug in this codebase).
    def p2m(phys_blocks):
        return physical_blocks_to_model_blocks(
            torch.as_tensor(np.asarray(phys_blocks), dtype=torch.long), clean_perm).cpu().numpy()

    def ctx_fn(S_phys, U_phys):
        prefix_m = p2m(S_phys).tolist()
        comp_m = p2m(U_phys).tolist()
        c = CHR.context_hidden_at_step(model, idx_roll, prefix_m, comp_m, BL, device)
        return c.mean(axis=0)                                       # (E,)

    def cand_fn(U_phys):
        E_c = CHR.candidate_embeddings(model, idx_roll, p2m(U_phys).tolist(), BL,
                                       mode=args.ev_mode, device=device)
        return E_c.mean(axis=0)                                     # (|U|, E)

    nll_fn = lambda phys_order: MIO.nll_under_order_text(
        model, idx_model, physical_blocks_to_model_blocks(
            torch.as_tensor(phys_order, dtype=torch.long), clean_perm).cpu().numpy(),
        BL, device, batch_size=args.eval_batch_size)

    pregate = CHR.causal_invariance_check(model, idx_roll, G.N_BLOCKS, BL, device,
                                          n_patterns=args.n_patterns)
    return dict(model=model, B_A=B_A, B_pos=B_pos, ctx_fn=ctx_fn, cand_fn=cand_fn,
                nll_fn=nll_fn, pregate=pregate, N=G.N_BLOCKS)
```

- [ ] **Step 2: Add Level-0/1 + verdict + report**

Add to `run_causal_hidden_probe.py`:

```python
def run_levels(ctx, args):
    B_A, B_pos = ctx["B_A"], ctx["B_pos"]
    ctx_fn, cand_fn, nll_fn = ctx["ctx_fn"], ctx["cand_fn"], ctx["nll_fn"]

    # Level 0 baselines
    static_cdl = GO.cdl_order(B_A, greedy=True)
    rng = np.random.default_rng(0)
    rand_order = rng.permutation(ctx["N"]).astype(np.int64)
    l2r = np.arange(ctx["N"], dtype=np.int64)
    base = {"static_BA_CDL": nll_fn(static_cdl), "random": nll_fn(rand_order),
            "L2R_or_raster": nll_fn(l2r)}

    # Level 1: cos-emb gamma sweep + controls
    gammas = [0.0, 0.25, 0.5, 1.0, 2.0]
    l1 = {}
    for g in gammas:
        o = CHR.causal_score_mix_rollout(B_A, B_pos, g, ctx_fn, cand_fn, "cos",
                                         hidden_off_at_t0=True)
        l1[f"gamma{g}"] = {"nll": nll_fn(o),
                           "tau_vs_L2R": CHR._kendall_tau(o, np.arange(ctx["N"]))}
    pos_ctrl = nll_fn(CHR.position_control_rollout(B_A, B_pos, gamma=1.0))
    shuf_ctrl = nll_fn(CHR.shuffled_hidden_rollout(B_A, B_pos, 1.0, ctx_fn, cand_fn, "cos", seed=0))
    return {"baselines": base, "level1_cos": l1,
            "controls": {"position": pos_ctrl, "shuffled_hidden": shuf_ctrl}}


def verdict(results):
    a_only = results["level1_cos"]["gamma0.0"]["nll"]
    improved = {g: a_only - v["nll"] for g, v in results["level1_cos"].items()}
    adj = [improved[f"gamma{g}"] for g in (0.25, 0.5, 1.0)]
    # margin = improvement vs static must clear a small absolute floor and beat both controls
    best = max(improved.values())
    stable = sum(d >= -1e-3 for d in adj) >= 2 and best > 0.01
    beats_pos = a_only - results["controls"]["position"] < best
    beats_shuf = a_only - results["controls"]["shuffled_hidden"] < best
    # non-degenerate: winning order not ~ L2R
    best_g = max(results["level1_cos"], key=lambda k: a_only - results["level1_cos"][k]["nll"])
    not_l2r = abs(results["level1_cos"][best_g]["tau_vs_L2R"]) < 0.95
    win = all([best > 0.01, stable, beats_pos, beats_shuf, not_l2r])
    return {"verdict": "WIN" if win else "NULL", "improved_vs_Aonly": improved,
            "best_gamma": best_g, "stable": stable, "beats_position": beats_pos,
            "beats_shuffled": beats_shuf, "not_L2R_degenerate": not_l2r}


def write_report(out_dir, modality, ckpt, pregate, results, verd):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    blob = {"modality": modality, "ckpt": str(ckpt), "pregate": pregate,
            **results, "decision": verd}
    (out / "report.json").write_text(json.dumps(blob, indent=2, default=float))
    (out / "report.md").write_text(
        f"# Causal-Hidden Probe — {modality}\n- ckpt: `{ckpt}`\n"
        f"- PRE-GATE passed: {pregate['passed']} (min_cos={pregate['min_cosine']:.6f})\n"
        f"- **verdict: {verd['verdict']}** (best {verd['best_gamma']})\n"
        f"- improved vs A-only: {verd['improved_vs_Aonly']}\n"
        f"- beats_position={verd['beats_position']} beats_shuffled={verd['beats_shuffled']} "
        f"not_L2R={verd['not_L2R_degenerate']}\n")
    print(f"[report] {out/'report.json'}  verdict={verd['verdict']}")
```

- [ ] **Step 3: Add `main()` (PRE-GATE abort + dispatch)**

Add to `run_causal_hidden_probe.py`:

```python
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modality", required=True, choices=["text", "image"])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-chunks", type=int, default=16)
    p.add_argument("--n-roll", type=int, default=16)
    p.add_argument("--n-eval", type=int, default=256)        # image
    p.add_argument("--a-global-path"); p.add_argument("--data-val"); p.add_argument("--meta")
    p.add_argument("--ev-mode", default="content_token", choices=["content_token", "content_free"])
    p.add_argument("--pos-tau", type=float, default=2.0)
    p.add_argument("--n-patterns", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    device = torch.device(args.device)
    ctx = build_text(args, device) if args.modality == "text" else build_image(args, device)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    if not ctx["pregate"]["passed"]:
        (out / "PREGATE_FAILED.md").write_text(
            f"# PRE-GATE FAILED ({args.modality})\nFuture completion leaks into predictor hidden; "
            f"dynamic-hidden probe invalid. min_cosine={ctx['pregate']['min_cosine']:.6f}\n")
        print(f"[PRE-GATE FAILED] min_cosine={ctx['pregate']['min_cosine']:.6f} -> abort"); return
    results = run_levels(ctx, args)
    verd = verdict(results)
    write_report(args.out_dir, args.modality, args.ckpt, ctx["pregate"], results, verd)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Smoke-run the text path (small)**

Run:
```bash
cd /home/admin/lyuyuhuan/order_lyu && /home/admin/anaconda3/envs/X1/bin/python -u scripts/run_causal_hidden_probe.py \
  --modality text \
  --ckpt block_lo_arm_order_network/probe_results/clean_base_random_perm/ckpt_step30000.pt \
  --out-dir probe_results/causal_hidden_probe/text_clean_random_SMOKE \
  --n-chunks 2 --n-roll 2 --n-patterns 2 --device cuda:0
```
Expected: prints PRE-GATE result then `[report] …/report.json verdict=…` (or `[PRE-GATE FAILED]`). `report.json` has finite `baselines`/`level1_cos` and the `gamma0.0` NLL equals the `static_BA_CDL` baseline (γ=0 ≡ static order). Fix any frame/shape mismatch here before scaling.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_causal_hidden_probe.py
git commit -m "feat(causal-hidden): probe driver (text), PRE-GATE + Level-0/1 + verdict + report"
```

---

## Task 7: driver image path + model-attn wiring (best-effort)

**Files:**
- Modify: `scripts/run_causal_hidden_probe.py`

Spec §2, §5.2, §14.5. Image closures mirror the hidden-graph driver's `diagnose_image`; model-attn is wired as an optional second Level-1 score (top-4 variance heads to match `A_global`); if `--with-attn` is off or the hook misbehaves, the probe still completes on cos-emb alone.

- [ ] **Step 1: Add image builder**

Add to `run_causal_hidden_probe.py`:

```python
def build_image(args, device):
    import pickle
    from directed_graph_policy import build_directed_graph
    from run_hidden_graph_diag import AOGPT_from_ckpt
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = AOGPT_from_ckpt(ckpt, device)
    BL = int(pickle.load(open(args.meta, "rb")).get("block_order_block_len", 1))
    N = 64
    val = np.fromfile(args.data_val, dtype=np.uint16).reshape(-1, N * BL)
    idx_eval = torch.from_numpy(val[:args.n_eval].astype(np.int64))
    idx_roll = torch.from_numpy(val[:args.n_roll].astype(np.int64))
    B_A = build_directed_graph(np.load(args.a_global_path).astype(np.float32))
    B_pos = PG.image_manhattan_graph(side=8, tau=args.pos_tau)

    # image is identity-framed (phys == model block ids)
    ctx_fn = lambda S, U: CHR.context_hidden_at_step(model, idx_roll, list(S), list(U), BL, device).mean(0)
    cand_fn = lambda U: CHR.candidate_embeddings(model, idx_roll, list(U), BL, args.ev_mode, device).mean(0)
    nll_fn = lambda phys_order: MIO.nll_under_order_image(
        model, idx_eval, np.asarray(phys_order), BL, device, batch_size=args.eval_batch_size)
    pregate = CHR.causal_invariance_check(model, idx_roll, N, BL, device, n_patterns=args.n_patterns)
    return dict(model=model, B_A=B_A, B_pos=B_pos, ctx_fn=ctx_fn, cand_fn=cand_fn,
                nll_fn=nll_fn, pregate=pregate, N=N)
```

- [ ] **Step 2: Smoke-run the image path (small)**

Run:
```bash
cd /home/admin/lyuyuhuan/order_lyu && /home/admin/anaconda3/envs/X1/bin/python -u scripts/run_causal_hidden_probe.py \
  --modality image \
  --ckpt probe_results_image/vq64_alt_from0_mlp_patch2x2_l8h8e512/ckpt_step30000.pt \
  --a-global-path probe_results_image/vq64_alt_from0_mlp_patch2x2_l8h8e512/A_global_step30000.npy \
  --data-val nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch2x2/val.bin \
  --meta nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch2x2/meta.pkl \
  --out-dir probe_results_image/causal_hidden_probe/image_mlp_alt_SMOKE \
  --n-eval 16 --n-roll 8 --n-patterns 2 --device cuda:0
```
Expected: PRE-GATE result + `[report] … verdict=…`; finite fields. Fix frame/shape issues here.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_causal_hidden_probe.py
git commit -m "feat(causal-hidden): probe driver image path (identity-framed) + smoke"
```

---

## Task 8: Stage runs (text primary/contrast + image primary) + summary

**Files:** none new (execution + result capture). Controller runs on GPU (not a subagent).

- [ ] **Step 1: PRE-GATE + Level-1 — text primary (clean_random)**

```bash
cd /home/admin/lyuyuhuan/order_lyu && /home/admin/anaconda3/envs/X1/bin/python -u scripts/run_causal_hidden_probe.py \
  --modality text \
  --ckpt block_lo_arm_order_network/probe_results/clean_base_random_perm/ckpt_step30000.pt \
  --out-dir probe_results/causal_hidden_probe/text_clean_random_30k \
  --n-chunks 16 --n-roll 16 --n-patterns 4 --device cuda:0
```
If PRE-GATE fails → stop and record (the probe is invalid for this ckpt). Else record verdict + improved_vs_Aonly.

- [ ] **Step 2: Text contrast (alt_mlp)** — same command with `--ckpt probe_results/attention_order_mlp/alt_from0_mlp_finetune/ckpt_step30000.pt` and `--out-dir .../text_alt_mlp_30k`.

- [ ] **Step 3: Image primary (mlp_alt)**

```bash
cd /home/admin/lyuyuhuan/order_lyu && /home/admin/anaconda3/envs/X1/bin/python -u scripts/run_causal_hidden_probe.py \
  --modality image \
  --ckpt probe_results_image/vq64_alt_from0_mlp_patch2x2_l8h8e512/ckpt_step30000.pt \
  --a-global-path probe_results_image/vq64_alt_from0_mlp_patch2x2_l8h8e512/A_global_step30000.npy \
  --data-val nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch2x2/val.bin \
  --meta nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch2x2/meta.pkl \
  --out-dir probe_results_image/causal_hidden_probe/image_mlp_alt_30k \
  --n-eval 256 --n-roll 16 --n-patterns 4 --device cuda:0
```

- [ ] **Step 4: Write + commit the summary**

Write `probe_results/causal_hidden_probe/SUMMARY.md` with each PRE-GATE result, verdict, `improved_vs_Aonly`, controls, and the §10 decision (Case A/B/C; image must not block text). Then:
```bash
git add -f probe_results/causal_hidden_probe/SUMMARY.md
git commit -m "results(causal-hidden): Stage verdicts (text primary/contrast + image primary)"
```

---

## Task 9 (gated, optional): content-free Version B + per-sample audit + model-attn

**Files:** `scripts/run_causal_hidden_probe.py` (add `--ev-mode content_free`, `--with-attn`, `--audit-n` flags), execution.

Spec §7 (audit), §8 (Version B), §14.5. Run ONLY for modalities where Version A WON (per §8). Add: (a) `--ev-mode content_free` re-run; (b) per-sample audit (16–32 samples, per-sample greedy rollout via the existing rollout with `n_roll=1` per sample, report `order_diversity` + small-subset NLL); (c) `--with-attn` to add the model-attn Level-1 variant (top-4 variance heads to match `A_global`). Wire as thin branches; smoke each before the full run. Commit results to `SUMMARY.md`.

---

## Self-Review

**Spec coverage:** §0–§1 question/scope ✓ (driver docstring + verdict→decision). §2 substrates/ckpts ✓ (Task 6/7/8 reuse Stage-A ckpts; image block_len read from meta). §3 shared greedy rollout + per-step predictor hidden ✓ (Tasks 1,4). §4 PRE-GATE multi-t/multi-S_t, abort on fail ✓ (Task 1 + driver main). §5.1 cos-emb ✓ (Task 3). §5.2 model-attn best-effort ✓ (Tasks 3,9). §5.3 pooling summary=c_t ✓ (Task 1 `context_hidden_at_step`; last/mean-revealed deferred per spec). §5.4 e_v content-token (wte-only) + content-free ✓ (Task 2). §6 per-step residualize + γ-mix + position/shuffled controls ✓ (Tasks 4,5,6). §7 Level-2 oracle (diagnostic-only helper) + per-sample audit ✓ (Tasks 5,9). §8 content-aware first, Version B gated ✓ (Task 9). §9 NLL metric ✓ (reuse modelio). §10 WIN gate (6 conditions) + decision ✓ (Task 6 `verdict`; SUMMARY records Case A/B/C in Task 8). §11 framing guard ✓ (Level-2 not in `verdict`; content-aware labeled in report). §12 code units ✓. §14.1 t=0 hidden-off ✓ (Task 4 `hidden_off_at_t0`). §14.2 γ0≡static + regression test ✓ (Task 4 Step 1/4). §14.3 per-step vector residualize ✓ (Task 4 `_residualize_vec`). §14.4 dispersion ✓ (Task 4 `record`, expand in Task 8/9). §14.5 attn best-effort ✓ (Tasks 3,7,9). §14.6 image non-blocking ✓ (Task 8 SUMMARY decision).

**Placeholder scan:** Task 9 is intentionally gated/optional (depends on Version-A WIN) and references concrete flags + reused functions; no silent TODOs. The `verdict` margin (`>0.01` nats, `2σ` deferred to Task 8 dispersion) is an explicit default; Task 8 records dispersion so the margin can be tightened to `2σ` if the team prefers — flagged, not vague.

**Type consistency:** `context_hidden_at_step → (n,E)`; `candidate_embeddings → (n,M,E)`; closures aggregate to `(E,)` and `(M,E)` before `dynamic_score_*`. `causal_score_mix_rollout`/`position_control_rollout`/`shuffled_hidden_rollout` all return `(N,) int64` phys orders; `nll_fn` consumes phys orders (text converts phys→model internally). `cdl_order(B_A)` is the shared γ=0 reference. `teacher_scores(B, S, U, last, mode)` returns `(q, cand)` (verified T5). `_kendall_tau` used in both `order_diversity` and the driver `tau_vs_L2R`.

**Open verification flags (resolve at the named step):**
1. Task 1 Step 4 — predictor-hidden rank `t*BL` and completion-invariance on the real ckpts (the PRE-GATE is itself the runtime check).
2. Task 2 Step 4 — `wpe` index offset (`+1` for the `[None]` token) for content-free mode.
3. Task 3 Step 4 — `c_attn` split order (q,k,v) and per-head RMSNorm path in `AOGPT_block`.
4. Task 4 Step 4 — the `γ=0 ≡ cdl_order(B_A)` regression (load-bearing; align A-only branch with `graph_order.cdl_order`).
5. Task 6 Step 4 — text phys↔model frame: confirm `p2m` (via `physical_blocks_to_model_blocks`)
   feeds the model the SAME frame `nll_fn` uses, and the `γ0` NLL == `static_BA_CDL` NLL check in
   the smoke (both must reveal the same physical sequence).
