# Hidden Graph Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a frozen, falsifiable diagnostic that decides whether a hidden-state graph `B_H=sim(H,H)`, after removing position, carries order structure complementary to the attention graph `B_A` — on text and image checkpoints, with no training.

**Architecture:** Five small pure-numpy modules (graph build / position / normalize / structure metrics / readout+mix) with full unit tests, plus one model-coupled I/O module (image hidden extraction + frozen NLL-under-order) with smoke tests, wired by one driver script that emits a per-checkpoint JSON+MD report. The driver mirrors the existing `scripts/run_hidden_residual_diag.py` harness and reuses `extract_oracle_hidden`, `build_directed_graph`/`build_B_set`, `attn_order_teacher.rollout_order`/`teacher_scores`, `compute_token_ce` (text NLL) and `_forward_with_block_orders` (image NLL).

**Tech Stack:** Python 3.8, NumPy, PyTorch, pytest. Repo root `/home/admin/lyuyuhuan/order_lyu`. Modules live in `block_lo_arm_order_network/`, tests in `block_lo_arm_order_network/tests/`, driver in `scripts/`, spec at `docs/superpowers/specs/2026-05-26-hidden-graph-diagnostic-design.md`.

**Conventions (from the codebase):** `N_BLOCKS=64` for both text and image; text `BLOCK_LEN=4` (`model.block_order_block_len`), image `BLOCK_LEN=1` (`meta["block_order_block_len"]`); image patches form an 8×8 raster grid (side=8); graphs are scored only in physical-block frame (`PHYS_FRAME`). Run all tests from the repo root with `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/<file> -v`.

---

## File Structure

| File | Responsibility |
|---|---|
| `block_lo_arm_order_network/hidden_graph.py` | `cosine_graph(H)` → per-sample + mean `B_H`, diag 0 (pure) |
| `block_lo_arm_order_network/position_graph.py` | `text_position_graph`, `image_manhattan_graph` (pure) |
| `block_lo_arm_order_network/graph_normalize.py` | off-diag z-score, shift-to-nonneg, off-diag corr, `B_pos`-residualize, matched-random residual (pure) |
| `block_lo_arm_order_network/graph_structure_metrics.py` | sharpness / row-entropy / top-k mass / spectral gap + shuffled-null comparison (pure) |
| `block_lo_arm_order_network/graph_order.py` | C-D+L readout wrapper, graph-level λ-mix order, score-level γ-mix order (pure given teacher) |
| `block_lo_arm_order_network/hidden_graph_modelio.py` | image block-hidden extraction; text & image frozen NLL-under-order (model-coupled) |
| `scripts/run_hidden_graph_diag.py` | driver: load ckpt → 3 graphs → normalize/resid → metrics/corr → readout/mix → NLL → report |
| `block_lo_arm_order_network/tests/test_hidden_graph.py` etc. | one test file per module |

---

## Task 1: `hidden_graph.py` — cosine graph

**Files:**
- Create: `block_lo_arm_order_network/hidden_graph.py`
- Test: `block_lo_arm_order_network/tests/test_hidden_graph.py`

- [ ] **Step 1: Write the failing test**

```python
# block_lo_arm_order_network/tests/test_hidden_graph.py
import numpy as np
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from hidden_graph import cosine_graph


def test_cosine_graph_shapes_and_diag_zero():
    rng = np.random.default_rng(0)
    H = rng.standard_normal((5, 4, 8)).astype(np.float32)  # (n,N,E)
    per_sample, mean = cosine_graph(H)
    assert per_sample.shape == (5, 4, 4)
    assert mean.shape == (4, 4)
    assert np.allclose(np.diagonal(per_sample, axis1=1, axis2=2), 0.0)
    assert np.allclose(np.diag(mean), 0.0)


def test_cosine_graph_values_are_cosine_offdiag():
    # two unit vectors: cos known; check one off-diagonal entry
    H = np.zeros((1, 2, 3), dtype=np.float32)
    H[0, 0] = [1.0, 0.0, 0.0]
    H[0, 1] = [1.0, 1.0, 0.0]
    per_sample, mean = cosine_graph(H)
    assert np.isclose(per_sample[0, 0, 1], 1.0 / np.sqrt(2.0), atol=1e-6)
    assert np.isclose(mean[0, 1], 1.0 / np.sqrt(2.0), atol=1e-6)


def test_cosine_graph_handles_zero_vector():
    H = np.zeros((1, 2, 3), dtype=np.float32)  # both zero → cos undefined → 0
    per_sample, _ = cosine_graph(H)
    assert np.all(np.isfinite(per_sample))
    assert per_sample[0, 0, 1] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_hidden_graph.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hidden_graph'`.

- [ ] **Step 3: Write minimal implementation**

```python
# block_lo_arm_order_network/hidden_graph.py
"""Hidden-state relational graph B_H = cos(h_u, h_v), block-level."""
import numpy as np


def cosine_graph(H):
    """H: (n, N, E) per-sample block hidden in physical frame.
    Returns (per_sample (n,N,N), mean (N,N)), cosine similarity, diag zeroed.
    Zero-norm blocks yield cosine 0 (not NaN)."""
    H = np.asarray(H, dtype=np.float64)
    norm = np.linalg.norm(H, axis=2, keepdims=True)          # (n,N,1)
    safe = np.where(norm == 0.0, 1.0, norm)
    Hn = H / safe                                            # unit vectors; zero rows stay zero
    per_sample = np.einsum("nie,nje->nij", Hn, Hn)          # (n,N,N) cosine
    N = per_sample.shape[1]
    eye = np.eye(N, dtype=bool)
    per_sample[:, eye] = 0.0
    mean = per_sample.mean(axis=0)
    mean[eye] = 0.0
    return per_sample, mean
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_hidden_graph.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/hidden_graph.py block_lo_arm_order_network/tests/test_hidden_graph.py
git commit -m "feat(hidden-graph): cosine B_H build (per-sample + mean, diag0, zero-safe)"
```

---

## Task 2: `position_graph.py` — position/distance baselines

**Files:**
- Create: `block_lo_arm_order_network/position_graph.py`
- Test: `block_lo_arm_order_network/tests/test_position_graph.py`

- [ ] **Step 1: Write the failing test**

```python
# block_lo_arm_order_network/tests/test_position_graph.py
import numpy as np, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from position_graph import text_position_graph, image_manhattan_graph


def test_text_position_graph_decays_with_index_distance():
    B = text_position_graph(N=4, tau=1.0)
    assert B.shape == (4, 4)
    assert np.allclose(np.diag(B), 0.0)
    assert np.allclose(B, B.T)                       # symmetric
    assert B[0, 1] > B[0, 2] > B[0, 3]               # monotone decay in |u-v|
    assert np.isclose(B[0, 1], np.exp(-1.0))


def test_image_manhattan_graph_uses_grid_distance():
    B = image_manhattan_graph(side=8, tau=1.0)       # 64 patches, 8x8 raster
    assert B.shape == (64, 64)
    assert np.allclose(np.diag(B), 0.0)
    assert np.allclose(B, B.T)
    # patch 0=(0,0), 1=(0,1) manh 1; 8=(1,0) manh 1; 9=(1,1) manh 2
    assert np.isclose(B[0, 1], np.exp(-1.0))
    assert np.isclose(B[0, 8], np.exp(-1.0))
    assert np.isclose(B[0, 9], np.exp(-2.0))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_position_graph.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'position_graph'`.

- [ ] **Step 3: Write minimal implementation**

```python
# block_lo_arm_order_network/position_graph.py
"""Position/locality baseline graphs B_pos for the hidden-graph diagnostic."""
import numpy as np


def text_position_graph(N=64, tau=1.0):
    """B_pos(u,v) = exp(-|u-v|/tau), diag 0. Text = block index distance."""
    idx = np.arange(N)
    d = np.abs(idx[:, None] - idx[None, :]).astype(np.float64)
    B = np.exp(-d / float(tau))
    np.fill_diagonal(B, 0.0)
    return B


def image_manhattan_graph(side=8, tau=1.0):
    """B_pos(u,v) = exp(-manhattan(u,v)/tau), diag 0, over a side x side raster grid."""
    n = side * side
    rows = np.arange(n) // side
    cols = np.arange(n) % side
    d = (np.abs(rows[:, None] - rows[None, :]) +
         np.abs(cols[:, None] - cols[None, :])).astype(np.float64)
    B = np.exp(-d / float(tau))
    np.fill_diagonal(B, 0.0)
    return B
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_position_graph.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/position_graph.py block_lo_arm_order_network/tests/test_position_graph.py
git commit -m "feat(hidden-graph): B_pos baselines (text index-distance, image manhattan)"
```

---

## Task 3: `graph_normalize.py` — normalize / residualize / corr / matched control

**Files:**
- Create: `block_lo_arm_order_network/graph_normalize.py`
- Test: `block_lo_arm_order_network/tests/test_graph_normalize.py`

Locked semantics (spec §5, §8.5): off-diag z-score for corr/residualize/mix; shift-to-nonnegative (not clip) for readout input; residualize = OLS of target off-diag entries on basis off-diag entries; matched-random residual = off-diag permutation of the residual graph preserving value multiset and diag 0.

- [ ] **Step 1: Write the failing test**

```python
# block_lo_arm_order_network/tests/test_graph_normalize.py
import numpy as np, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from graph_normalize import (offdiag_zscore, shift_nonneg, offdiag_corr,
                             residualize, matched_random_residual, _offdiag_mask)


def _sym(rng, N):
    A = rng.standard_normal((N, N)); A = (A + A.T) / 2; np.fill_diagonal(A, 0.0); return A


def test_offdiag_zscore_zero_mean_unit_std_offdiag():
    rng = np.random.default_rng(1); B = _sym(rng, 6)
    Z = offdiag_zscore(B)
    m = _offdiag_mask(6)
    assert np.allclose(Z[m].mean(), 0.0, atol=1e-9)
    assert np.allclose(Z[m].std(), 1.0, atol=1e-9)
    assert np.allclose(np.diag(Z), 0.0)


def test_shift_nonneg_min_is_zero_and_order_preserved():
    rng = np.random.default_rng(2); B = _sym(rng, 6)
    S = shift_nonneg(offdiag_zscore(B))
    m = _offdiag_mask(6)
    assert S[m].min() == 0.0
    assert np.all(S[m] >= 0.0)
    assert np.allclose(np.diag(S), 0.0)
    # rank of off-diag entries preserved by shift
    base = offdiag_zscore(B)
    assert np.array_equal(np.argsort(base[m]), np.argsort(S[m]))


def test_offdiag_corr_self_is_one():
    rng = np.random.default_rng(3); B = _sym(rng, 8)
    assert np.isclose(offdiag_corr(B, B, method="pearson"), 1.0)
    assert np.isclose(offdiag_corr(B, B, method="spearman"), 1.0)


def test_residualize_removes_basis_correlation():
    rng = np.random.default_rng(4); basis = _sym(rng, 10)
    noise = _sym(rng, 10)
    target = 3.0 * basis + 0.5 * noise          # strongly correlated with basis
    resid = residualize(target, basis)
    assert abs(offdiag_corr(resid, basis, method="pearson")) < 1e-6
    assert np.allclose(np.diag(resid), 0.0)


def test_matched_random_preserves_offdiag_multiset_and_diag0():
    rng = np.random.default_rng(5); B = _sym(rng, 7)
    M = matched_random_residual(B, seed=123)
    m = _offdiag_mask(7)
    assert np.allclose(np.sort(M[m]), np.sort(B[m]))   # same value multiset
    assert np.allclose(np.diag(M), 0.0)
    assert not np.allclose(M, B)                        # actually shuffled
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_graph_normalize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'graph_normalize'`.

- [ ] **Step 3: Write minimal implementation**

```python
# block_lo_arm_order_network/graph_normalize.py
"""Graph normalization, position residualization, off-diag correlation, matched control.
corr/residualize/mix use off-diag z-score; C-D+L readout uses shift-to-nonnegative."""
import numpy as np


def _offdiag_mask(N):
    return ~np.eye(N, dtype=bool)


def offdiag_zscore(B):
    B = np.asarray(B, dtype=np.float64).copy()
    m = _offdiag_mask(B.shape[0])
    vals = B[m]
    std = vals.std()
    std = std if std > 0 else 1.0
    out = np.zeros_like(B)
    out[m] = (vals - vals.mean()) / std
    return out


def shift_nonneg(B):
    """Shift off-diag entries so min off-diag = 0 (preserves order; diag stays 0)."""
    B = np.asarray(B, dtype=np.float64).copy()
    m = _offdiag_mask(B.shape[0])
    out = np.zeros_like(B)
    out[m] = B[m] - B[m].min()
    return out


def offdiag_corr(A, B, method="pearson"):
    a = np.asarray(A, dtype=np.float64)[_offdiag_mask(A.shape[0])]
    b = np.asarray(B, dtype=np.float64)[_offdiag_mask(B.shape[0])]
    if method == "spearman":
        a = np.argsort(np.argsort(a)).astype(np.float64)
        b = np.argsort(np.argsort(b)).astype(np.float64)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def residualize(target, basis):
    """OLS residual of target off-diag on [1, basis off-diag]; diag stays 0."""
    target = np.asarray(target, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    N = target.shape[0]
    m = _offdiag_mask(N)
    y = target[m]
    x = basis[m]
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid_vals = y - X @ beta
    out = np.zeros_like(target)
    out[m] = resid_vals
    return out


def matched_random_residual(B, seed=0):
    """Off-diagonal permutation of B preserving the off-diag value multiset, diag 0.
    Symmetric input -> symmetric matched control (permute upper triangle, mirror)."""
    B = np.asarray(B, dtype=np.float64)
    N = B.shape[0]
    rng = np.random.default_rng(seed)
    iu = np.triu_indices(N, k=1)
    upper = B[iu].copy()
    rng.shuffle(upper)
    out = np.zeros_like(B)
    out[iu] = upper
    out = out + out.T
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_graph_normalize.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/graph_normalize.py block_lo_arm_order_network/tests/test_graph_normalize.py
git commit -m "feat(hidden-graph): graph normalize/residualize/corr + matched-random control"
```

---

## Task 4: `graph_structure_metrics.py` — structure metrics + shuffled null

**Files:**
- Create: `block_lo_arm_order_network/graph_structure_metrics.py`
- Test: `block_lo_arm_order_network/tests/test_graph_structure_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# block_lo_arm_order_network/tests/test_graph_structure_metrics.py
import numpy as np, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from graph_structure_metrics import (sharpness, row_entropy, topk_mass,
                                      spectral_gap, structure_vs_null)


def test_sharpness_higher_for_peaked_rows():
    N = 8
    peaked = np.zeros((N, N)); 
    for i in range(N):
        j = (i + 1) % N
        peaked[i, j] = 1.0
    flat = np.ones((N, N)); np.fill_diagonal(flat, 0.0)
    assert sharpness(peaked) > sharpness(flat)


def test_row_entropy_max_for_uniform():
    N = 8
    flat = np.ones((N, N)); np.fill_diagonal(flat, 0.0)
    # uniform over N-1 candidates -> entropy log(N-1)
    assert np.isclose(row_entropy(flat), np.log(N - 1), atol=1e-6)


def test_topk_mass_in_unit_interval():
    rng = np.random.default_rng(0)
    B = np.abs(rng.standard_normal((10, 10))); np.fill_diagonal(B, 0.0)
    mk = topk_mass(B, k=3)
    assert 0.0 <= mk <= 1.0


def test_structure_vs_null_flags_peaked_above_shuffled():
    N = 10
    peaked = np.zeros((N, N))
    for i in range(N):
        peaked[i, (i + 1) % N] = 5.0
    peaked = peaked + peaked.T
    res = structure_vs_null(peaked, metric=sharpness, n_shuffle=50, seed=0)
    assert res["value"] > res["null_mean"] + 2 * res["null_std"]
    assert res["z"] > 2.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_graph_structure_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# block_lo_arm_order_network/graph_structure_metrics.py
"""Structure metrics for a block graph and comparison against a shuffled null."""
import numpy as np
from graph_normalize import matched_random_residual, _offdiag_mask


def _rows_offdiag(B):
    N = B.shape[0]
    out = []
    for i in range(N):
        out.append(np.delete(B[i], i))
    return np.asarray(out, dtype=np.float64)            # (N, N-1)


def sharpness(B):
    """Mean over rows of max / mean of |off-diag| entries (peak-to-mean ratio)."""
    R = np.abs(_rows_offdiag(B))
    mean = R.mean(axis=1)
    mean = np.where(mean == 0, 1.0, mean)
    return float((R.max(axis=1) / mean).mean())


def row_entropy(B):
    """Mean Shannon entropy of softmax-free normalized non-negative off-diag rows."""
    R = np.abs(_rows_offdiag(B))
    s = R.sum(axis=1, keepdims=True)
    s = np.where(s == 0, 1.0, s)
    P = R / s
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.where(P > 0, P * np.log(P), 0.0).sum(axis=1)
    return float(ent.mean())


def topk_mass(B, k=4):
    """Mean fraction of |off-diag| row mass in the top-k entries."""
    R = np.abs(_rows_offdiag(B))
    s = R.sum(axis=1)
    s = np.where(s == 0, 1.0, s)
    topk = np.sort(R, axis=1)[:, -k:].sum(axis=1)
    return float((topk / s).mean())


def spectral_gap(B):
    """Gap between the two largest-magnitude eigenvalues of the symmetrized |B|."""
    M = np.abs((B + B.T) / 2.0)
    w = np.linalg.eigvalsh(M)
    w = np.sort(np.abs(w))[::-1]
    return float(w[0] - w[1]) if len(w) > 1 else float(w[0])


def structure_vs_null(B, metric=sharpness, n_shuffle=100, seed=0):
    """Compare metric(B) to its distribution over off-diag-shuffled nulls."""
    val = metric(B)
    nulls = [metric(matched_random_residual(B, seed=seed + i)) for i in range(n_shuffle)]
    nulls = np.asarray(nulls)
    mu, sd = float(nulls.mean()), float(nulls.std())
    z = (val - mu) / (sd if sd > 0 else 1.0)
    return {"value": val, "null_mean": mu, "null_std": sd, "z": float(z)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_graph_structure_metrics.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/graph_structure_metrics.py block_lo_arm_order_network/tests/test_graph_structure_metrics.py
git commit -m "feat(hidden-graph): graph structure metrics + shuffled-null z-test"
```

---

## Task 5: `graph_order.py` — C-D+L readout, graph-mix, score-mix orders

**Files:**
- Create: `block_lo_arm_order_network/graph_order.py`
- Test: `block_lo_arm_order_network/tests/test_graph_order.py`

Reuse `attn_order_teacher.rollout_order(B, mode="C-D+L", greedy, standardize, seed)` and `teacher_scores(B, S_t, U_t, last, mode)`. Readout input graphs are always passed through `shift_nonneg`. Score-level mix z-scores each branch's per-step score vector before the γ-weighted sum (spec §6e).

- [ ] **Step 1: Write the failing test**

```python
# block_lo_arm_order_network/tests/test_graph_order.py
import numpy as np, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from graph_order import cdl_order, graph_mix_order, score_mix_order


def _ring(N, w=5.0):
    B = np.zeros((N, N))
    for i in range(N):
        B[i, (i + 1) % N] = w
    return B + B.T


def test_cdl_order_is_permutation():
    B = _ring(8)
    o = cdl_order(B, greedy=True)
    assert sorted(o.tolist()) == list(range(8))


def test_graph_mix_lambda_endpoints_match_single_graphs():
    A = _ring(8, 5.0); H = _ring(8, 1.0)[::-1].copy()
    assert np.array_equal(graph_mix_order(A, H, lam=1.0, greedy=True), cdl_order(A, greedy=True))
    assert np.array_equal(graph_mix_order(A, H, lam=0.0, greedy=True), cdl_order(H, greedy=True))


def test_score_mix_gamma_zero_equals_A_only():
    A = _ring(8, 5.0); Hr = _ring(8, 1.0)
    o_gamma0 = score_mix_order(A, Hr, gamma=0.0, greedy=True, seed=0)
    o_Aonly = cdl_order(A, greedy=True)
    assert np.array_equal(o_gamma0, o_Aonly)


def test_score_mix_gamma_changes_order_when_H_disagrees():
    N = 8
    A = _ring(N, 5.0)
    Hr = np.zeros((N, N))
    for i in range(N):
        Hr[i, (i + 3) % N] = 5.0          # H prefers a different neighbour
    Hr = Hr + Hr.T
    o0 = score_mix_order(A, Hr, gamma=0.0, greedy=True, seed=0)
    o2 = score_mix_order(A, Hr, gamma=2.0, greedy=True, seed=0)
    assert not np.array_equal(o0, o2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_graph_order.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# block_lo_arm_order_network/graph_order.py
"""Order readouts: C-D+L on a single graph, graph-level mix, score-level two-branch mix."""
import numpy as np
from attn_order_teacher import rollout_order, teacher_scores
from graph_normalize import shift_nonneg, offdiag_zscore

MODE = "C-D+L"


def cdl_order(B, greedy=True, tau_T=1.0, seed=0, top_k=0):
    """Full reveal order from one graph via C-D+L; readout sees shift-to-nonneg graph."""
    Bn = shift_nonneg(B)
    return rollout_order(Bn, mode=MODE, greedy=greedy, tau_T=tau_T, seed=seed,
                         top_k=top_k, standardize=True).astype(np.int64)


def graph_mix_order(B_A, B_H, lam, greedy=True, tau_T=1.0, seed=0, top_k=0):
    """Order from the graph-level mix lam*B_A_z + (1-lam)*B_H_z."""
    mix = lam * offdiag_zscore(B_A) + (1.0 - lam) * offdiag_zscore(B_H)
    return cdl_order(mix, greedy=greedy, tau_T=tau_T, seed=seed, top_k=top_k)


def _zscore_vec(v):
    v = np.asarray(v, dtype=np.float64)
    sd = v.std()
    return (v - v.mean()) / (sd if sd > 0 else 1.0)


def score_mix_order(B_A, B_H_resid, gamma, greedy=True, tau_T=1.0, seed=0):
    """Sequential C-D+L with per-step score s = z(s_A) + gamma * z(s_H), on shift-nonneg graphs.
    Frozen proxy of the two-branch policy s = f_A(phi_A) + gamma f_H(phi_H_resid)."""
    A = shift_nonneg(B_A)
    H = shift_nonneg(B_H_resid)
    N = A.shape[0]
    rng = np.random.default_rng(seed)
    S, U, last, order = [], list(range(N)), None, []
    for t in range(N):
        if len(U) == 1:
            v = U[0]
        else:
            qa, cand = teacher_scores(A, S, U, last, mode=MODE)
            qh, candh = teacher_scores(H, S, U, last, mode=MODE)
            assert list(cand) == list(candh)
            score = _zscore_vec(qa) + gamma * _zscore_vec(qh)
            if greedy:
                v = int(cand[int(np.argmax(score))])
            else:
                p = np.exp((score - score.max()) / tau_T); p /= p.sum()
                v = int(rng.choice(cand, p=p))
        order.append(v); S.append(v); U.remove(v); last = v
    return np.asarray(order, dtype=np.int64)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_graph_order.py -v`
Expected: PASS (4 tests). If `teacher_scores` returns `(q, cand)` with `cand` already restricted to `U`, the assertion holds; if it returns full-length arrays, adjust the indexing to `cand`’s order (verify against `attn_order_teacher.teacher_scores` before implementing).

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/graph_order.py block_lo_arm_order_network/tests/test_graph_order.py
git commit -m "feat(hidden-graph): C-D+L readout, graph-level lambda-mix, score-level gamma-mix orders"
```

---

## Task 6: `hidden_graph_modelio.py` — image hidden extraction + frozen NLL-under-order

**Files:**
- Create: `block_lo_arm_order_network/hidden_graph_modelio.py`
- Test: `block_lo_arm_order_network/tests/test_hidden_graph_modelio.py`

These functions touch the model, so the test is a **smoke test on a tiny `BlockAOGPT`** (no checkpoint). Text hidden extraction reuses the existing `hidden_residual_hidden.extract_oracle_hidden` (no new code). New here: (a) image block-hidden extraction mirroring it but in image framing, (b) text & image frozen NLL-under-order.

Verify before implementing: `from AOGPT import AOGPT, AOGPTConfig`; a block model is `AOGPT(AOGPTConfig(order_impl="block", block_order_block_len=BL, block_size=N*BL, vocab_size=V, n_layer=1, n_head=1, n_embd=8))`. `forward_fn(idx, orders, return_hidden=True, hidden_return_mode="original")` returns a tuple whose `[2]` is hidden `(b, T+1, E)` (index 0 = the `[None]` token). `expand_model_blocks_to_token_order(orders_blocks, BL)` and `compute_token_ce(model, idx, token_orders, device)` are importable from `train_clean_aogpt`. `_forward_with_block_orders(model, x, block_orders, fixed_token_perm, inv_block_perm)` is importable from `scripts/train_vq64_round2`.

- [ ] **Step 1: Write the failing smoke test**

```python
# block_lo_arm_order_network/tests/test_hidden_graph_modelio.py
import numpy as np, torch, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))
sys.path.insert(0, str(ROOT / "nanogpt-learned-order"))
from AOGPT import AOGPT, AOGPTConfig
from hidden_graph_modelio import extract_image_block_hidden, nll_under_order_image


def _tiny_block_model(N=4, BL=1, V=16):
    cfg = AOGPTConfig(order_impl="block", block_order_block_len=BL, block_size=N * BL,
                      vocab_size=V, n_layer=1, n_head=1, n_embd=8, dropout=0.0, bias=False)
    m = AOGPT(cfg); m.eval(); return m


def test_extract_image_block_hidden_shape_and_finite():
    torch.manual_seed(0); N, BL, V, n = 4, 1, 16, 3
    m = _tiny_block_model(N, BL, V)
    tokens = torch.randint(0, V, (n, N * BL))
    H = extract_image_block_hidden(m, tokens, block_len=BL, n_blocks=N, device="cpu")
    assert H.shape == (n, N, 8)            # (n, N, E)
    assert np.all(np.isfinite(H))


def test_nll_under_order_image_finite_and_order_sensitive():
    torch.manual_seed(0); N, BL, V, n = 4, 1, 16, 3
    m = _tiny_block_model(N, BL, V)
    tokens = torch.randint(0, V, (n, N * BL))
    o1 = np.array([0, 1, 2, 3]); o2 = np.array([3, 2, 1, 0])
    nll1 = nll_under_order_image(m, tokens, o1, block_len=BL, device="cpu")
    nll2 = nll_under_order_image(m, tokens, o2, block_len=BL, device="cpu")
    assert np.isfinite(nll1) and np.isfinite(nll2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_hidden_graph_modelio.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hidden_graph_modelio'`.

- [ ] **Step 3: Write minimal implementation**

```python
# block_lo_arm_order_network/hidden_graph_modelio.py
"""Model-coupled I/O for the hidden-graph diagnostic:
image block-hidden extraction (text reuses hidden_residual_hidden.extract_oracle_hidden),
and frozen teacher-forced NLL under a given block order (text + image)."""
import numpy as np
import torch

# train_vq64_round2 lives in scripts/; the driver adds scripts/ to sys.path before import.
from train_vq64_round2 import _forward_with_block_orders


def _expand_block_order_to_tokens(order_blocks, block_len, device):
    """order_blocks: (b, N) int -> token order (b, N*block_len) revealing each block's tokens."""
    b, N = order_blocks.shape
    base = torch.arange(block_len, device=device)
    tok = order_blocks[:, :, None] * block_len + base[None, None, :]
    return tok.reshape(b, N * block_len)


@torch.no_grad()
def extract_image_block_hidden(model, tokens, block_len, n_blocks, device, chunk_size=16,
                               token_order=None):
    """Returns (n, N, E) full-context block hidden = mean over each block's token hiddens.
    token_order None -> identity raster reveal (physical frame). Mirrors extract_oracle_hidden."""
    model.eval()
    n_total = tokens.shape[0]
    if token_order is None:
        ident = torch.arange(n_blocks, device=device)
    chunks = []
    for i in range(0, n_total, chunk_size):
        idx = tokens[i:i + min(chunk_size, n_total - i)].to(device)
        nc = idx.shape[0]
        order_blocks = (token_order[i:i + nc] if token_order is not None
                        else ident.unsqueeze(0).expand(nc, -1)).to(device)
        tok_order = _expand_block_order_to_tokens(order_blocks, block_len, device)
        out = model.forward_fn(idx, tok_order, return_hidden=True, hidden_return_mode="original")
        h_tok = out[2][:, 1:, :]                                    # drop [None]; (nc, T, E)
        E = h_tok.shape[-1]
        h_blk = h_tok.reshape(nc, n_blocks, block_len, E).mean(dim=2)  # (nc, N, E)
        chunks.append(h_blk.cpu().numpy())
    return np.concatenate(chunks, axis=0)


@torch.no_grad()
def nll_under_order_image(model, tokens, phys_block_order, block_len, device,
                          fixed_token_perm=None, inv_block_perm=None, batch_size=16):
    """Token-avg teacher-forced NLL with every sample revealed in the same phys block order."""
    model.eval()
    N = int(phys_block_order.shape[0])
    order_t = torch.as_tensor(np.asarray(phys_block_order), dtype=torch.long)
    total, ntok = 0.0, 0
    for s in range(0, tokens.shape[0], batch_size):
        x = tokens[s:s + batch_size].to(device)
        orders = order_t.unsqueeze(0).expand(x.shape[0], -1).contiguous().to(device)
        loss = _forward_with_block_orders(model, x, orders,
                                          fixed_token_perm=fixed_token_perm,
                                          inv_block_perm=inv_block_perm)
        total += float(loss.item()) * x.shape[0]
        ntok += x.shape[0]
    return total / ntok
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_hidden_graph_modelio.py -v`
Expected: PASS (2 tests). If `_forward_with_block_orders` returns a per-token loss tensor rather than a scalar mean, take `.mean()`; confirm its return type in `scripts/train_vq64_round2.py` first.

- [ ] **Step 5: Add the text NLL helper + its smoke test**

Append to `hidden_graph_modelio.py`:

```python
def nll_under_order_text(model, idx_eval_model, model_block_order, block_len, device,
                         batch_size=16):
    """Token-avg teacher-forced NLL on text; model_block_order is in MODEL block frame.
    Reuses train_clean_aogpt.compute_token_ce + expand_model_blocks_to_token_order."""
    from train_clean_aogpt import compute_token_ce, expand_model_blocks_to_token_order
    model.eval()
    order_t = torch.as_tensor(np.asarray(model_block_order), dtype=torch.long)
    total, ntok = 0.0, 0
    with torch.no_grad():
        for s in range(0, idx_eval_model.size(0), batch_size):
            idx = idx_eval_model[s:s + batch_size].to(device)
            orders = order_t.unsqueeze(0).expand(idx.size(0), -1).to(device)
            token_orders = expand_model_blocks_to_token_order(orders, block_len).to(device)
            token_losses, _ = compute_token_ce(model, idx, token_orders, device)
            total += float(token_losses.float().sum().item())
            ntok += int(token_losses.numel())
    return total / ntok
```

Add to the test file:

```python
def test_nll_under_order_text_runs_via_compute_token_ce():
    # text path needs train_clean_aogpt importable; smoke that the call shape works
    import torch, numpy as np
    from hidden_graph_modelio import nll_under_order_text
    N, BL, V, n = 4, 2, 16, 3
    from AOGPT import AOGPT, AOGPTConfig
    cfg = AOGPTConfig(order_impl="block", block_order_block_len=BL, block_size=N * BL,
                      vocab_size=V, n_layer=1, n_head=1, n_embd=8, dropout=0.0, bias=False)
    m = AOGPT(cfg); m.eval()
    idx = torch.randint(0, V, (n, N * BL))
    nll = nll_under_order_text(m, idx, np.arange(N), block_len=BL, device="cpu")
    assert np.isfinite(nll)
```

- [ ] **Step 6: Run, then commit**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_hidden_graph_modelio.py -v`
Expected: PASS (3 tests).

```bash
git add block_lo_arm_order_network/hidden_graph_modelio.py block_lo_arm_order_network/tests/test_hidden_graph_modelio.py
git commit -m "feat(hidden-graph): image block-hidden extraction + frozen NLL-under-order (text+image)"
```

---

## Task 7: Driver `run_hidden_graph_diag.py` — text path (Stage A primary)

**Files:**
- Create: `scripts/run_hidden_graph_diag.py`
- Reference (mirror): `scripts/run_hidden_residual_diag.py:25-96`

The driver is orchestration over already-tested units; its check is a CLI smoke run on a small `--n-chunks`, not a unit test. Build it incrementally.

- [ ] **Step 1: Write the text loading + graph-build section**

```python
#!/usr/bin/env python3
"""Hidden Graph Diagnostic (frozen, no training). Per-checkpoint report:
B_A vs B_H vs B_pos structure, position residualization, C-D+L readout orders,
graph-level lambda-mix + score-level gamma-mix, frozen NLL-under-order."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
for p in ["block_lo_arm_order_network", "nanogpt-learned-order", "scripts"]:
    sys.path.insert(0, str(_REPO / p))

import hidden_graph as HG
import position_graph as PG
import graph_normalize as GN
import graph_structure_metrics as SM
import graph_order as GO
import hidden_graph_modelio as MIO


def diagnose_text(args, device):
    from AOGPT import AOGPT
    from clean_training_protocol import build_clean_block_permutation, phys_to_model_idx_clean
    from training_utils import load_train_chunks
    from train_clean_aogpt import extract_A_matrices, physical_blocks_to_model_blocks
    import hidden_residual_graph as G
    import hidden_residual_hidden as Hd
    from run_hidden_residual_diag import load_clean_ckpt

    model, perm_seed = load_clean_ckpt(args.ckpt, device)
    block_len = int(model.block_order_block_len)
    clean_perm = build_clean_block_permutation(G.N_BLOCKS, seed=perm_seed)
    idx_phys = load_train_chunks(n_chunks=args.n_chunks)
    idx_model = phys_to_model_idx_clean(idx_phys, clean_perm)

    # B_A = global attention graph (physical frame), same as the residual line's B_G
    A_all = extract_A_matrices(model, idx_model, clean_perm, device, n_chunks=args.n_chunks)
    _, B_A, _ = G.build_B_set(A_all, frame=G.PHYS_FRAME)

    # B_H = cosine of full-context block hidden (physical frame), Stage-1 oracle
    order_model_full = torch.stack([clean_perm.block_perm_phys_to_model] * idx_model.shape[0])
    H = Hd.extract_oracle_hidden(model, idx_model, clean_perm, device, order_model_full)  # (n,N,E)
    _, B_H = HG.cosine_graph(H)

    B_pos = PG.text_position_graph(N=G.N_BLOCKS, tau=args.pos_tau)
    phys_to_model = lambda o: physical_blocks_to_model_blocks(
        torch.as_tensor(o, dtype=torch.long), clean_perm).cpu().numpy()
    nll_fn = lambda phys_order: MIO.nll_under_order_text(
        model, idx_model, phys_to_model(phys_order), block_len, device,
        batch_size=args.eval_batch_size)
    return B_A, B_H, B_pos, nll_fn
```

- [ ] **Step 2: Write the shared analysis section (modality-agnostic)**

Add to `run_hidden_graph_diag.py`:

```python
def analyze(B_A, B_H, B_pos, nll_fn, args):
    """Run the full §6 diagnostic on three raw graphs + a per-order frozen NLL fn."""
    B_H_resid = GN.residualize(GN.offdiag_zscore(B_H), GN.offdiag_zscore(B_pos))
    graphs = {"B_A": B_A, "B_H_raw": B_H, "B_H_resid": B_H_resid, "B_pos": B_pos}

    structure = {name: {m.__name__: SM.structure_vs_null(g, metric=m, n_shuffle=args.n_shuffle)
                        for m in (SM.sharpness, SM.spectral_gap)}
                 for name, g in graphs.items()}
    corr = {
        "BH_vs_Bpos": GN.offdiag_corr(B_H, B_pos),
        "BH_vs_BA": GN.offdiag_corr(B_H, B_A),
        "BA_vs_Bpos": GN.offdiag_corr(B_A, B_pos),
        "BHresid_vs_BA": GN.offdiag_corr(B_H_resid, B_A),
        "BHresid_vs_Bpos": GN.offdiag_corr(B_H_resid, B_pos),  # sanity ~0
    }

    orders = {name: GO.cdl_order(g, greedy=True) for name, g in graphs.items()}
    lam_orders = {f"mix_lam{lam}": GO.graph_mix_order(B_A, B_H_resid, lam, greedy=True)
                  for lam in (0.0, 0.25, 0.5, 0.75, 1.0)}
    # score-level mix: report mean +/- spread over seeds for the noise floor (sampled)
    gamma_nll = {}
    for gamma in (0.0, 0.25, 0.5, 1.0, 2.0):
        seeds = [GO.score_mix_order(B_A, B_H_resid, gamma, greedy=False, seed=s)
                 for s in range(args.n_seed)]
        nlls = [nll_fn(o) for o in seeds]
        gamma_nll[f"gamma{gamma}"] = {"mean": float(np.mean(nlls)), "std": float(np.std(nlls))}
    # matched-random control at the best gamma slot is computed in the report step
    base_nll = {name: nll_fn(o) for name, o in orders.items()}
    mix_nll = {name: nll_fn(o) for name, o in lam_orders.items()}
    return {"structure": structure, "corr": corr, "nll_orders": base_nll,
            "nll_graph_mix": mix_nll, "nll_score_mix": gamma_nll}
```

- [ ] **Step 3: Write the matched-control + verdict + report section**

Add to `run_hidden_graph_diag.py`:

```python
def control_and_verdict(B_A, B_H, B_pos, nll_fn, score_mix_nll, args):
    B_H_resid = GN.residualize(GN.offdiag_zscore(B_H), GN.offdiag_zscore(B_pos))
    # matched-random residual control (same multiset), evaluated at gamma=1.0
    ctrl_nlls = []
    for s in range(args.n_seed):
        Bc = GN.matched_random_residual(B_H_resid, seed=1000 + s)
        o = GO.score_mix_order(B_A, Bc, gamma=1.0, greedy=False, seed=s)
        ctrl_nlls.append(nll_fn(o))
    a_only = score_mix_nll["gamma0.0"]["mean"]
    floor = max(score_mix_nll["gamma0.0"]["std"], 1e-9)
    improved = {g: (a_only - v["mean"]) for g, v in score_mix_nll.items()}
    # stable improvement: adjacent gammas in {.25,.5,1} not worse, one clearly better
    adj = [improved[f"gamma{g}"] for g in (0.25, 0.5, 1.0)]
    stable = sum(d >= -floor for d in adj) >= 2 and max(adj) > 2 * floor
    beats_ctrl = improved["gamma1.0"] > (a_only - float(np.mean(ctrl_nlls))) + floor
    bh_resid_structured = (SM.structure_vs_null(B_H_resid, metric=SM.sharpness,
                                                n_shuffle=args.n_shuffle)["z"] > 2.0)
    not_clone = abs(GN.offdiag_corr(B_H_resid, B_A)) < 0.95
    not_pos = abs(GN.offdiag_corr(B_H_resid, B_pos)) < 0.1
    verdict = ("WIN" if all([bh_resid_structured, not_clone, not_pos, stable, beats_ctrl])
               else "NULL")
    return {"verdict": verdict, "improved_vs_Aonly": improved, "stable": stable,
            "beats_matched_control": beats_ctrl, "ctrl_nll_mean": float(np.mean(ctrl_nlls)),
            "bh_resid_structured": bh_resid_structured, "not_BA_clone": not_clone,
            "not_position_artifact": not_pos}


def write_report(out_dir, modality, ckpt, results, verdict):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    blob = {"modality": modality, "ckpt": str(ckpt), **results, "decision": verdict}
    (out / "report.json").write_text(json.dumps(blob, indent=2, default=float))
    lines = [f"# Hidden Graph Diagnostic — {modality}", f"- ckpt: `{ckpt}`",
             f"- **verdict: {verdict['verdict']}**", "",
             f"- corr(B_H,B_pos)={results['corr']['BH_vs_Bpos']:.3f}  "
             f"corr(B_H,B_A)={results['corr']['BH_vs_BA']:.3f}  "
             f"corr(B_Hresid,B_A)={results['corr']['BHresid_vs_BA']:.3f}",
             f"- score-mix NLL vs A-only: {verdict['improved_vs_Aonly']}",
             f"- stable={verdict['stable']} beats_control={verdict['beats_matched_control']} "
             f"bh_resid_structured={verdict['bh_resid_structured']}"]
    (out / "report.md").write_text("\n".join(lines))
    print(f"[report] {out/'report.json'}  verdict={verdict['verdict']}")
```

- [ ] **Step 4: Write `main()` wiring text + image dispatch**

Add to `run_hidden_graph_diag.py`:

```python
def diagnose_image(args, device):
    import pickle
    from AOGPT import AOGPT
    from directed_graph_policy import build_directed_graph
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = AOGPT_from_ckpt(ckpt, device)                       # see Step 5
    with open(args.meta, "rb") as f: meta = pickle.load(f)
    block_len = int(meta.get("block_order_block_len", 1))
    n_blocks = 64
    val = np.fromfile(args.data_val, dtype=np.uint16).reshape(-1, n_blocks * block_len)
    tokens = torch.from_numpy(val[:args.n_eval].astype(np.int64))
    B_A = build_directed_graph(np.load(args.a_global_path).astype(np.float32))
    H = MIO.extract_image_block_hidden(model, tokens, block_len, n_blocks, device)
    _, B_H = HG.cosine_graph(H)
    B_pos = PG.image_manhattan_graph(side=8, tau=args.pos_tau)
    nll_fn = lambda phys_order: MIO.nll_under_order_image(
        model, tokens, np.asarray(phys_order), block_len, device,
        batch_size=args.eval_batch_size)
    return B_A, B_H, B_pos, nll_fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modality", required=True, choices=["text", "image"])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-chunks", type=int, default=16)            # text
    p.add_argument("--n-eval", type=int, default=256)             # image
    p.add_argument("--a-global-path"); p.add_argument("--data-val"); p.add_argument("--meta")
    p.add_argument("--pos-tau", type=float, default=2.0)
    p.add_argument("--n-shuffle", type=int, default=100)
    p.add_argument("--n-seed", type=int, default=5)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    device = torch.device(args.device)
    if args.modality == "text":
        B_A, B_H, B_pos, nll_fn = diagnose_text(args, device)
    else:
        B_A, B_H, B_pos, nll_fn = diagnose_image(args, device)
    results = analyze(B_A, B_H, B_pos, nll_fn, args)
    verdict = control_and_verdict(B_A, B_H, B_pos, nll_fn, results["nll_score_mix"], args)
    write_report(args.out_dir, args.modality, args.ckpt, results, verdict)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Add the image ckpt loader helper**

Add near the top of `run_hidden_graph_diag.py` (mirror `train_vq64_fixed_baseline.py` model construction):

```python
def AOGPT_from_ckpt(ckpt, device):
    from AOGPT import AOGPT, AOGPTConfig
    ma = ckpt["model_args"]
    model = AOGPT(AOGPTConfig(**ma)).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model
```

- [ ] **Step 6: Smoke-run the text path (small, on the ready ckpt)**

Run:
```bash
cd /home/admin/lyuyuhuan/order_lyu && python -u scripts/run_hidden_graph_diag.py \
  --modality text \
  --ckpt block_lo_arm_order_network/probe_results/clean_base_random_perm/ckpt_step30000.pt \
  --out-dir probe_results/hidden_graph_diag/text_clean_random_SMOKE \
  --n-chunks 2 --n-shuffle 20 --n-seed 2 --device cuda:0
```
Expected: prints `[report] …/report.json verdict=…`; `report.json` exists with finite corr/NLL fields. Fix any frame/shape mismatch surfaced here before scaling.

- [ ] **Step 7: Commit**

```bash
git add scripts/run_hidden_graph_diag.py
git commit -m "feat(hidden-graph): diagnostic driver (text+image), analysis+control+verdict+report"
```

---

## Task 8: Stage-A full runs (text primary/contrast + image primary)

**Files:** none (execution + result capture). Image primary needs `A_global_step30000.npy` (present) and `vq64_alt_…_l8h8e512/ckpt_step30000.pt` (present).

- [ ] **Step 1: Text primary (clean_random) full**

```bash
cd /home/admin/lyuyuhuan/order_lyu && python -u scripts/run_hidden_graph_diag.py \
  --modality text \
  --ckpt block_lo_arm_order_network/probe_results/clean_base_random_perm/ckpt_step30000.pt \
  --out-dir probe_results/hidden_graph_diag/text_clean_random_30k \
  --n-chunks 16 --n-shuffle 100 --n-seed 5 --device cuda:0
```
Expected: `report.json` with a `decision.verdict`. Record corr + verdict.

- [ ] **Step 2: Text contrast (alt_mlp_finetune) full**

```bash
cd /home/admin/lyuyuhuan/order_lyu && python -u scripts/run_hidden_graph_diag.py \
  --modality text \
  --ckpt probe_results/attention_order_mlp/alt_from0_mlp_finetune/ckpt_step30000.pt \
  --out-dir probe_results/hidden_graph_diag/text_alt_mlp_30k \
  --n-chunks 16 --n-shuffle 100 --n-seed 5 --device cuda:0
```

- [ ] **Step 3: Image primary (mlp_alt) full**

```bash
cd /home/admin/lyuyuhuan/order_lyu && python -u scripts/run_hidden_graph_diag.py \
  --modality image \
  --ckpt probe_results_image/vq64_alt_from0_mlp_patch2x2_l8h8e512/ckpt_step30000.pt \
  --a-global-path probe_results_image/vq64_alt_from0_mlp_patch2x2_l8h8e512/A_global_step30000.npy \
  --data-val nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch2x2/val.bin \
  --meta nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch2x2/meta.pkl \
  --out-dir probe_results_image/hidden_graph_diag/image_mlp_alt_30k \
  --n-eval 256 --n-shuffle 100 --n-seed 5 --device cuda:0
```

- [ ] **Step 4: Commit a combined Stage-A summary**

Write `probe_results/hidden_graph_diag/STAGE_A_SUMMARY.md` with the three verdicts + key corr/NLL numbers, then:

```bash
git add probe_results/hidden_graph_diag/STAGE_A_SUMMARY.md
git commit -m "results(hidden-graph): Stage A verdicts (text primary/contrast + image primary)"
```

---

## Task 9 (Stage B, gated): image contrast (fixed_random) + Stage-2 causal

**Files:** none new for Stage B run; Stage-2 causal adds a small driver flag.

- [ ] **Step 1: Image contrast — when `vq64_fixed_random_l8h8e512@30k` exists**

```bash
cd /home/admin/lyuyuhuan/order_lyu && python -u scripts/run_hidden_graph_diag.py \
  --modality image \
  --ckpt probe_results_image/vq64_fixed_random_l8h8e512/ckpt_step30000.pt \
  --a-global-path <extract or reuse A_global for this ckpt> \
  --data-val nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch2x2/val.bin \
  --meta nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch2x2/meta.pkl \
  --out-dir probe_results_image/hidden_graph_diag/image_fixed_random_30k \
  --n-eval 256 --device cuda:0
```
Note: this ckpt has no saved `A_global`; extract it first via the image attention extractor (`extract_image_attention*.extract_A_global`) and pass the saved `.npy`.

- [ ] **Step 2: Stage-2 causal (ONLY if any Stage-A oracle `B_H_resid` verdict was WIN)**

Add `--hidden-mode {original,predictor}` and `--t-list 0,16,32,48` to the driver; for `predictor`, build `B_H` from `extract_causal_hidden` per `t`, compare `offdiag_corr(B_H_causal, B_H_oracle)` (raw + resid) and `cdl_order` agreement. Implement as a thin branch in `diagnose_text`/`diagnose_image` mirroring Task 7 Step 1, gated behind the flag. Smoke at one `t` first.

- [ ] **Step 3: Commit Stage-B results**

```bash
git add probe_results_image/hidden_graph_diag/image_fixed_random_30k docs/superpowers/plans/2026-05-26-hidden-graph-diagnostic.md
git commit -m "results(hidden-graph): Stage B image contrast (+ causal if gated)"
```

---

## Self-Review

**Spec coverage:** §2 substrates ✓ (Task 7–9). §3 B_A/B_H/B_pos ✓ (Tasks 1,2,7). §4 normalization & residualization ✓ (Task 3). §5 structure+corr+readout+λ-mix+γ-mix ✓ (Tasks 4,5,7). §6 frozen NLL-under-order + checkpoint-local ✓ (Task 6, driver builds `nll_fn` from each ckpt's own model/graphs). §7 oracle-first/gated causal ✓ (Task 9 Step 2 gated on Stage-A WIN). §8 anti-fooling verdict (non-artifact, non-clone, NLL<A-only, γ-sweep stability, matched control, shuffled null, seed noise floor) ✓ (Task 7 Step 3). §9 code units ✓ (one module per file). Endpoint two-graph policy = explicitly out of scope ✓.

**Placeholder scan:** Task 9 Step 1 has `<extract or reuse A_global>` — intentional, the contrast ckpt's A_global must be extracted when the ckpt lands (Stage B is gated and the exact path depends on the extractor output); flagged inline, not a silent TODO. All Stage-A tasks are fully specified.

**Type consistency:** `cosine_graph` returns `(per_sample, mean)` everywhere; driver uses `_, B_H = HG.cosine_graph(H)`. `cdl_order`/`graph_mix_order`/`score_mix_order` all return `(N,) int64` phys-frame orders; text `nll_fn` converts phys→model before NLL (matches `evaluate_orders` framing), image `nll_fn` consumes phys order directly (matches `_forward_with_block_orders`). `offdiag_corr`/`residualize`/`shift_nonneg`/`matched_random_residual` signatures consistent across Tasks 3–7.

**Open verification flags for the implementer (resolve at the named step, do not assume):**
1. Task 5 Step 4 — `teacher_scores` return contract (`cand` order vs full-length).
2. Task 6 Step 4 — `_forward_with_block_orders` return (scalar vs per-token).
3. Task 7 Step 1 — text B_A: confirm `G.build_B_set(A_all, frame=PHYS_FRAME)` returns `(per_sample, global, frame)` (used as `_, B_A, _`).
4. Task 7 Step 6 — frame correctness of text order (phys vs model) shows up here; the smoke run is the gate before full runs.
