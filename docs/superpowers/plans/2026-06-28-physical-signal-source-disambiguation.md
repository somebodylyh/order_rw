# P2 Physical-Order Signal Source Disambiguation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide whether the canonical L0 physical-order signal is a fixed-layout slot→physical map (B), content-dependent recovery (C), or a map with content modulation (B+), by measuring whether the carrier's attention/B graph depends on **content** (not on the order output, which is tautological under a fixed layout).

**Architecture:** A `analyses/physical_signal_source.py` module that builds per-text carrier B65 graphs at a **fixed layout + shared reveal orders** (reusing the canonical extraction), restricts all metrics to the **valid B65 edge support**, and computes cross-text variance, pairwise similarity, and a content-free **slot-only mean-table predictor's held-out R²** (split over text samples), gated by per-text carrier validity and calibrated against synthetic content-invariant / content-randomized baselines. Block-swap + random-token perturbations and an E3 relayout diagnostic are supporting. A per-seed verdict writer + README close it.

**Tech Stack:** Python 3, PyTorch, NumPy, Matplotlib, pytest. Reuses `analyses/canonical_reanalysis.py` (`random_reveal_orders`, the canonical extraction path), `block_lo_arm_order_network/per_head_order_scan.py` (`_attn_to_A_block_loss_aligned_with_none_vec`), `none_separated_block_graph.py` (`build_none_separated_B`, `rollout_by_method`, `discovery_metrics`), `neural_readout.extract_b._load_model_and_chunks`, `analyses/position_prior_decomp.py` (`relayout_chunks`, `clean_perm_from_layout`, `make_layouts`).

## Global Constraints

- **No new training.** Read `runs/handoff_overnight/seed{2,42,123}/ckpt_step10000.pt`. CPU; GPU only as accelerator.
- **Order output cannot disambiguate** (spec): `τ_physical=1 ⟺ σ_model=block_perm`; B/C only separable at the attention/B level. τ_physical is **validity only**, never a B/C judge.
- **Carrier config (frozen, from P1):** seed2 L0{2,3,4,5}, seed42 L0{2}, seed123 L0{1,2,3,4}.
- **Valid-edge support only:** metrics computed on None→content (`B[0,1:]`) + content→content causal lower-triangle (`B[i,j], i>j≥1`); exclude into-None (`B[:,0]`), diagonal, and the structurally-zero content upper-triangle.
- **Row normalization:** `B_norm[i,:] = B[i,:] / (Σ_{valid j}|B[i,j]| + ε)`, ε=1e-9; **report raw AND row-normalized**.
- **Slot-only predictor:** `B_hat[i,j]=mean_{train texts}B[i,j]`; **train/held-out split over TEXT samples, not edges**; predictor sees no held-out-text activations. Report global total-entry held-out R² for both raw and normalized B, together with a size-matched randomized-null mean/std over deterministic seeds `0..15` and null-adjusted excess `(r2-null_mean)/(1-null_mean)`. Normalized R² alone is not strong B evidence: row normalization gives the randomized synthetic null about 0.70.
- **Carrier-validity gate:** a (text,head) enters variance/R² only if its per-text τ_physical ≥ 0.9 (strong); seed42's single weak head reported separately, soft threshold, never hard-excluded; **report all-sample AND valid-only**.
- **Calibration:** synthetic content-invariant B (variance≈0, raw/normalized R²≈1) + content-randomized B (variance high; raw R²≈-0.05, normalized total-entry R²≈0.70) anchor the metrics. R² evidence must clearly exceed its matched null in **both** raw and normalized views; disagreement or a near-null result is `mixed`, not a hard B/C call.
- **E1 = the only verdict driver; E3 relayout supporting** (survival→C; collapse→only consistent-with-B). Random-token = stress-only.
- **Tests run from `block_lo_arm_order_network/`**; insert repo ROOT (`parents[2]`) for `analyses.*`.

---

### Task 1: Per-text carrier B65 at fixed layout + shared reveal

**Files:**
- Create: `analyses/physical_signal_source.py`
- Test: `block_lo_arm_order_network/tests/test_pss_per_text.py`

**Interfaces:**
- Produces: `carrier_b65_per_text(ckpt_path, layer, head, M=24, n_reveals=8, fixed_reveal_seed=0, device="cpu", return_halves=False)` — normally returns `(B_list, tau_list)`; with `return_halves=True`, returns `(B_list, tau_list, halfA_list, halfB_list)`. The latter mode requires an even `n_reveals >= 2` and uses two equal, disjoint reveal halves, as required by Task 4's noise-floor estimator. Validation occurs before checkpoint loading.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pss_per_text.py
import pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.physical_signal_source import carrier_b65_per_text

def test_per_text_b65_shapes_and_validity():
    B_list, tau_list = carrier_b65_per_text(
        str(ROOT/"runs/handoff_overnight/seed2/ckpt_step10000.pt"), layer=0, head=2, M=6, n_reveals=2)
    assert len(B_list) == 6 and B_list[0].shape == (65, 65)
    assert len(tau_list) == 6
    # carrier head: most texts read high physical tau
    assert np.mean([abs(t) > 0.9 for t in tau_list]) >= 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_pss_per_text.py -q`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement**

```python
# analyses/physical_signal_source.py
"""P2: disambiguate the L0 physical-order signal source (B fixed-layout map vs
C content-dependent). Measures content-dependence of the carrier's attention/B,
NOT the order output (tautological under fixed layout). See spec
docs/superpowers/specs/2026-06-28-physical-signal-source-disambiguation-design.md."""
import pathlib, sys
import numpy as np, torch

_BLOCK_DIR = pathlib.Path(__file__).resolve().parents[1] / "block_lo_arm_order_network"
if str(_BLOCK_DIR) not in sys.path:
    sys.path.insert(0, str(_BLOCK_DIR))

from neural_readout.extract_b import _load_model_and_chunks
from analyses.canonical_reanalysis import random_reveal_orders
from per_head_order_scan import _attn_to_A_block_loss_aligned_with_none_vec
from none_separated_block_graph import build_none_separated_B, rollout_by_method, discovery_metrics

@torch.no_grad()
def carrier_b65_per_text(ckpt_path, layer, head, M=24, fixed_reveal_seed=0, n_reveals=4,
                         device="cpu"):
    model, chunks, clean_perm, dev, _ = _load_model_and_chunks(
        ckpt_path, M, seed=0, device=device, split="train")
    inv = clean_perm.inv_perm_model_to_phys.cpu().numpy()
    reveals = random_reveal_orders(n_reveals, fixed_reveal_seed)   # SHARED across texts
    B_list, tau_list = [], []
    for t in range(M):
        A_acc = None
        for rev in reveals:
            po = torch.from_numpy(rev[None, :]).to(dev)
            _, _, attn_list = model.forward_fn(chunks[t:t+1].to(dev), po, return_attentions=True)
            attn = torch.stack(attn_list, 0).cpu().numpy()[:, 0]   # (L,H,257,257)
            A = _attn_to_A_block_loss_aligned_with_none_vec(attn, rev, inv)  # (L,H,64,65)
            A_acc = A.astype(np.float64) if A_acc is None else A_acc + A
        B = build_none_separated_B((A_acc / n_reveals)[layer, head])
        B_list.append(B)
        tau_list.append(float(discovery_metrics(rollout_by_method(B, "C-D+L"))["tau_vs_l2r"]))
    return B_list, tau_list
```

> Implementer note: shared reveals across texts isolate the content variable (layout + reveal
> held fixed, only the text changes). `chunks` are the model-frame eval chunks; `M` distinct
> rows = `M` distinct texts.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_pss_per_text.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analyses/physical_signal_source.py block_lo_arm_order_network/tests/test_pss_per_text.py
git commit -m "feat: per-text carrier B65 at fixed layout + shared reveal"
```

---

### Task 2: Valid-edge mask + L1 row-normalization

**Files:**
- Modify: `analyses/physical_signal_source.py`
- Test: `block_lo_arm_order_network/tests/test_pss_mask_norm.py`

**Interfaces:**
- Produces:
  - `valid_edge_mask(n=65) -> np.ndarray bool (n,n)` — `True` on `[0,1:]` (None→content) and content lower-tri (`i>j≥1`); `False` elsewhere.
  - `row_normalize_l1(B, mask, eps=1e-9) -> np.ndarray` — each row divided by its valid-edge L1 mass + ε; non-valid entries zeroed.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pss_mask_norm.py
import pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.physical_signal_source import valid_edge_mask, row_normalize_l1

def test_valid_edge_mask_structure():
    m = valid_edge_mask(65)
    assert m[0, 1:].all() and not m[:, 0].any()       # None->content yes, into-None no
    assert not np.diag(m).any()                        # no diagonal
    assert m[2, 1] and not m[1, 2]                     # content lower-tri only
    assert m.sum() == 64 + 64*63//2                    # 64 None-edges + content lower-tri

def test_row_normalize_l1_sums_to_one_on_valid():
    B = np.zeros((65, 65)); B[3, 1] = 2.0; B[3, 2] = 2.0
    m = valid_edge_mask(65)
    Bn = row_normalize_l1(B, m)
    assert abs(Bn[3, 1] + Bn[3, 2] - 1.0) < 1e-6       # row 3 valid mass -> 1
    assert Bn[:, 0].sum() == 0                          # invalid zeroed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_pss_mask_norm.py -q`
Expected: FAIL

- [ ] **Step 3: Implement**

```python
# append to analyses/physical_signal_source.py
def valid_edge_mask(n=65):
    m = np.zeros((n, n), dtype=bool)
    m[0, 1:] = True                                   # None -> content
    for i in range(1, n):
        for j in range(1, i):
            m[i, j] = True                            # content causal lower-tri
    return m

def row_normalize_l1(B, mask, eps=1e-9):
    B = np.asarray(B, dtype=np.float64) * mask
    mass = np.abs(B).sum(axis=1, keepdims=True) + eps
    return B / mass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_pss_mask_norm.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analyses/physical_signal_source.py block_lo_arm_order_network/tests/test_pss_mask_norm.py
git commit -m "feat: valid-edge mask + L1 row-normalization"
```

---

### Task 3: Synthetic baselines + order-tautology unit test

**Files:**
- Modify: `analyses/physical_signal_source.py`
- Test: `block_lo_arm_order_network/tests/test_pss_synth.py`

**Interfaces:**
- Produces:
  - `synthetic_content_invariant(M, seed=0) -> list[(65,65)]` — one random valid B repeated M times (variance 0).
  - `synthetic_content_randomized(M, seed=0) -> list[(65,65)]` — M independent random valid B's.
- Documents (test): order-level lookup similarity is tautological at τ_physical=1.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pss_synth.py
import pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.physical_signal_source import (
    synthetic_content_invariant, synthetic_content_randomized, valid_edge_mask)
from none_separated_block_graph import rollout_by_method, discovery_metrics

def test_synthetic_baselines_shapes():
    inv = synthetic_content_invariant(5)
    rnd = synthetic_content_randomized(5)
    assert len(inv) == 5 and len(rnd) == 5
    assert np.allclose(inv[0], inv[4])                 # content-invariant: identical
    assert not np.allclose(rnd[0], rnd[4])             # randomized: differ

def test_order_lookup_tautology():
    # any B that rolls out to tau_physical=1 has sigma_model == arange in physical frame
    # -> order-level lookup similarity is tautologically 1; documents why E2-order was removed
    from analyses.physical_signal_source import synthetic_ascending_B
    B = synthetic_ascending_B()
    sigma = rollout_by_method(B, "C-D+L")
    assert discovery_metrics(sigma)["tau_vs_l2r"] == 1.0
    assert list(sigma) == list(range(len(sigma)))      # equals the lookup order exactly
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_pss_synth.py -q`
Expected: FAIL

- [ ] **Step 3: Implement**

```python
# append to analyses/physical_signal_source.py
def _random_valid_B(rng, n=65):
    m = valid_edge_mask(n)
    B = np.zeros((n, n)); B[m] = rng.random(int(m.sum()))
    return B

def synthetic_content_invariant(M, seed=0):
    B = _random_valid_B(np.random.default_rng(seed))
    return [B.copy() for _ in range(M)]

def synthetic_content_randomized(M, seed=0):
    return [_random_valid_B(np.random.default_rng(seed + i)) for i in range(M)]

def synthetic_ascending_B(n=65):
    # None -> block0 strongest; content i attends earlier j -> rolls out ascending, tau=1
    B = np.zeros((n, n))
    B[0, 1] = 1.0
    for i in range(2, n):
        B[i, 1:i] = 1.0 / (i - 1)
    return B
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_pss_synth.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analyses/physical_signal_source.py block_lo_arm_order_network/tests/test_pss_synth.py
git commit -m "feat: synthetic content-invariant/randomized baselines + order-tautology test"
```

---

### Task 4: Cross-text variance + pairwise similarity

**Files:**
- Modify: `analyses/physical_signal_source.py`
- Test: `block_lo_arm_order_network/tests/test_pss_variance.py`

**Interfaces:**
- Produces:
  - `cross_text_variance(B_list, mask, normalize=True) -> float` — mean over valid edges of `Var_text` (row-normalized if `normalize`).
  - `pairwise_similarity(B_list, mask, normalize=True) -> float` — mean over text pairs of Pearson r on valid edges.
  - `within_text_noise_floor(halfA_list, halfB_list, mask, normalize=True) -> float` — the **sampling-noise variance floor** for the full estimator: `0.25 * mean_edge(Var_text(halfA - halfB, ddof=0))`. `halfA_list`/`halfB_list` must be equal-length B65 lists built from two equal, disjoint reveal halves.
  - `content_variance(B_list, halfA_list, halfB_list, mask, normalize=True) -> float` — `max(0, cross_text_variance(B_list) − within_text_noise_floor(halfA, halfB))`: the between-text variance **above the sampling floor**, i.e. the content-attributable variance.

  **Why (resolved during Task 1 and corrected during Task 4):** at low n_reveals a single text's B65 is noisy (τ swings to ~−0.3); the synthetic content-invariant baseline (exact copies) has zero sampling noise and so under-estimates the real floor. For equal independent halves A and B, the full estimator is `(A+B)/2`, hence its sampling variance is `0.25 Var(A-B)`. Reveal orders are shared across texts, so a reveal-half-specific per-edge bias is common-mode: it contributes to raw `mean((A-B)^2)` but contributes zero cross-text variance. Centering via `Var_text(A-B)` removes that common-mode bias. The former `0.5 * mean((A-B)^2)` formula was therefore both incorrectly scaled for the full estimator and scientifically confounded by shared-reveal bias.

  `_stack` rejects empty input, non-2D/non-boolean/empty masks, matrix/mask shape mismatch, and non-finite values on valid edges. The floor additionally rejects unequal half-list lengths. These contracts prevent silent NumPy broadcasting or invalid variance estimates.

- [ ] **Step 1: Write the failing tests** (calibrate against the synthetic anchors)

```python
# tests/test_pss_variance.py
import pathlib, sys
import numpy as np
import pytest
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.physical_signal_source import (
    cross_text_variance, pairwise_similarity, within_text_noise_floor,
    content_variance, valid_edge_mask,
    synthetic_content_invariant, synthetic_content_randomized)

def test_variance_and_similarity_calibration():
    m = valid_edge_mask(65)
    inv = synthetic_content_invariant(8); rnd = synthetic_content_randomized(8)
    assert cross_text_variance(inv, m) < 1e-9               # invariant -> ~0
    assert cross_text_variance(rnd, m) > cross_text_variance(inv, m)
    assert pairwise_similarity(inv, m) > 0.99               # invariant -> ~1
    assert pairwise_similarity(rnd, m) < pairwise_similarity(inv, m)

def test_within_text_floor_and_content_variance():
    m = valid_edge_mask(65)
    # identical halves -> zero floor; identical B_list -> zero content variance
    inv = synthetic_content_invariant(6)
    assert within_text_noise_floor(inv, inv, m) < 1e-9
    assert content_variance(inv, inv, inv, m) < 1e-9
    # normalize=False synthetic checks also cover the estimator's scale:
    # independent half noise + a half-specific common shift gives a centered
    # floor matching Var_text((A+B)/2), with content_variance near zero.
    # Adding genuine per-text content leaves content_variance strictly positive.

def test_metric_input_contracts():
    m = np.ones((2, 2), dtype=bool)
    with pytest.raises(ValueError, match="must not be empty"):
        cross_text_variance([], m, normalize=False)
    with pytest.raises(ValueError, match="equal length"):
        within_text_noise_floor([np.zeros((2, 2))], [], m, normalize=False)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_pss_variance.py -q`
Expected: FAIL

- [ ] **Step 3: Implement**

```python
# append to analyses/physical_signal_source.py
def _stack(B_list, mask, normalize):
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("mask must be a 2D boolean array")
    if not mask.any():
        raise ValueError("mask must select at least one value")
    B_list = list(B_list)
    if not B_list:
        raise ValueError("B_list must not be empty")
    rows = []
    for i, B in enumerate(B_list):
        B = np.asarray(B, dtype=np.float64)
        if B.shape != mask.shape:
            raise ValueError(f"B_list[{i}] shape {B.shape} does not match mask shape {mask.shape}")
        if not np.isfinite(B[mask]).all():
            raise ValueError(f"B_list[{i}] contains non-finite values selected by mask")
        Bn = row_normalize_l1(B, mask) if normalize else (B * mask)
        rows.append(Bn[mask])
    return np.stack(rows)                                   # (M, n_valid)

def cross_text_variance(B_list, mask, normalize=True):
    X = _stack(B_list, mask, normalize)
    return float(X.var(axis=0).mean())

def pairwise_similarity(B_list, mask, normalize=True):
    X = _stack(B_list, mask, normalize)
    M = len(X); sims = []
    for a in range(M):
        for b in range(a + 1, M):
            r = np.corrcoef(X[a], X[b])[0, 1]
            if np.isfinite(r):
                sims.append(r)
    return float(np.mean(sims)) if sims else 1.0

def within_text_noise_floor(halfA_list, halfB_list, mask, normalize=True):
    if len(halfA_list) != len(halfB_list):
        raise ValueError("halfA_list and halfB_list must have equal length")
    XA = _stack(halfA_list, mask, normalize)
    XB = _stack(halfB_list, mask, normalize)
    # Equal independent halves: Var((A+B)/2) = 0.25 Var(A-B). Centering
    # across texts removes half-specific common bias from shared reveal orders.
    return float(0.25 * (XA - XB).var(axis=0, ddof=0).mean())

def content_variance(B_list, halfA_list, halfB_list, mask, normalize=True):
    cv = cross_text_variance(B_list, mask, normalize)
    floor = within_text_noise_floor(halfA_list, halfB_list, mask, normalize)
    return float(max(0.0, cv - floor))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_pss_variance.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analyses/physical_signal_source.py block_lo_arm_order_network/tests/test_pss_variance.py
git commit -m "feat: cross-text variance + pairwise similarity (calibrated vs synthetic)"
```

---

### Task 5: Slot-only mean-table predictor (text-level held-out R²)

**Files:**
- Modify: `analyses/physical_signal_source.py`
- Test: `block_lo_arm_order_network/tests/test_pss_predictor.py`

**Interfaces:**
- Produces: `slot_only_r2(B_list, mask, n_train=None, normalize=True) -> float` — require at least two matrices; require integer, non-bool `n_train` satisfying `1 <= n_train < M` (default `M//2`); fit `B_hat=mean_{train texts}` and evaluate one global R² over all held-out text×valid-edge entries. With row normalization, the randomized synthetic baseline calibrates near 0.70 because between-edge structure contributes to the global denominator; this normalized total-entry R² is not by itself strong B evidence.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pss_predictor.py
import pathlib, sys
import numpy as np
import pytest
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.physical_signal_source import (
    slot_only_r2, valid_edge_mask, synthetic_content_invariant, synthetic_content_randomized)

def test_predictor_high_on_invariant_low_on_random():
    m = valid_edge_mask(65)
    invariant = synthetic_content_invariant(8)
    randomized = synthetic_content_randomized(40)
    invariant_raw = slot_only_r2(invariant, m, normalize=False)
    invariant_norm = slot_only_r2(invariant, m, normalize=True)
    randomized_raw = slot_only_r2(randomized, m, normalize=False)
    randomized_norm = slot_only_r2(randomized, m, normalize=True)
    assert invariant_raw > 0.95 and invariant_norm > 0.95
    assert randomized_raw < 0.1
    # Total-entry normalized R² calibrates near 0.70; this is not by itself
    # strong evidence for a fixed map because the matched random null is high.
    assert 0.6 < randomized_norm < 0.8

@pytest.mark.parametrize("n_train", [0, -1, 4, 5, 1.5, True])
def test_predictor_rejects_invalid_split(n_train):
    m = valid_edge_mask(65)
    with pytest.raises(ValueError, match="n_train"):
        slot_only_r2(synthetic_content_invariant(4), m, n_train=n_train)

def test_predictor_requires_at_least_two_matrices():
    m = valid_edge_mask(65)
    with pytest.raises(ValueError, match="at least 2"):
        slot_only_r2(synthetic_content_invariant(1), m)

def test_predictor_accepts_valid_explicit_split():
    m = valid_edge_mask(65)
    assert np.isfinite(slot_only_r2(synthetic_content_randomized(6), m, n_train=2))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_pss_predictor.py -q`
Expected: FAIL

- [ ] **Step 3: Implement**

```python
# append to analyses/physical_signal_source.py
def slot_only_r2(B_list, mask, n_train=None, normalize=True):
    M = len(B_list)
    if M < 2:
        raise ValueError("slot_only_r2 requires at least 2 matrices")
    if n_train is None:
        n_train = M // 2
    elif isinstance(n_train, bool) or not isinstance(n_train, (int, np.integer)):
        raise ValueError("n_train must be an integer")
    n_train = int(n_train)
    if not 1 <= n_train < M:
        raise ValueError(f"n_train must satisfy 1 <= n_train < {M}")
    X = _stack(B_list, mask, normalize)                     # (M, n_valid)
    B_hat = X[:n_train].mean(axis=0)                        # content-free slot-pair table
    test = X[n_train:]
    ss_res = ((test - B_hat) ** 2).sum()
    ss_tot = ((test - test.mean()) ** 2).sum() + 1e-12
    return float(1.0 - ss_res / ss_tot)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_pss_predictor.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analyses/physical_signal_source.py block_lo_arm_order_network/tests/test_pss_predictor.py
git commit -m "feat: slot-only mean-table predictor held-out R2 (text-level split)"
```

---

### Task 6: Carrier-validity gate + per-sample τ table

**Files:**
- Modify: `analyses/physical_signal_source.py`
- Test: `block_lo_arm_order_network/tests/test_pss_gate.py`

**Interfaces:**
- Produces: `carrier_valid_filter(B_list, tau_list, thr=0.9) -> (B_valid, idx)` — keep texts where `|tau|>=thr`; returns the filtered B list + kept indices.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pss_gate.py
import pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.physical_signal_source import carrier_valid_filter

def test_gate_keeps_only_valid():
    B = [np.zeros((65, 65)) for _ in range(4)]
    taus = [1.0, 0.2, 0.95, -0.99]
    Bv, idx = carrier_valid_filter(B, taus, thr=0.9)
    assert idx == [0, 2, 3]                                # |tau|>=0.9 (incl anti)
    assert len(Bv) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_pss_gate.py -q`
Expected: FAIL

- [ ] **Step 3: Implement**

```python
# append to analyses/physical_signal_source.py
def carrier_valid_filter(B_list, tau_list, thr=0.9):
    idx = [i for i, t in enumerate(tau_list) if abs(t) >= thr]
    return [B_list[i] for i in idx], idx
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_pss_gate.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analyses/physical_signal_source.py block_lo_arm_order_network/tests/test_pss_gate.py
git commit -m "feat: carrier-validity gate (per-text tau filter)"
```

---

### Task 7: Block-swap / cross-sample content perturbation (secondary)

**Files:**
- Modify: `analyses/physical_signal_source.py`
- Test: `block_lo_arm_order_network/tests/test_pss_blockswap.py`

**Interfaces:**
- Produces: `block_swap_chunk(chunk, swaps, block_len=4) -> Tensor` — swap content of given model-block index pairs within a `(256,)` chunk (fixed eval labels); `cross_sample_replace(chunk, donor, blocks, block_len=4) -> Tensor`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pss_blockswap.py
import pathlib, sys
import torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.physical_signal_source import block_swap_chunk

def test_block_swap_swaps_content():
    c = torch.arange(256)
    out = block_swap_chunk(c, swaps=[(0, 5)], block_len=4)
    assert out[0:4].tolist() == [20, 21, 22, 23]          # block5 content now at block0
    assert out[20:24].tolist() == [0, 1, 2, 3]            # and vice versa
    assert out[8:12].tolist() == c[8:12].tolist()         # untouched blocks unchanged
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_pss_blockswap.py -q`
Expected: FAIL

- [ ] **Step 3: Implement**

```python
# append to analyses/physical_signal_source.py
def block_swap_chunk(chunk, swaps, block_len=4):
    out = chunk.clone()
    for a, b in swaps:
        sa, sb = a * block_len, b * block_len
        tmp = out[sa:sa+block_len].clone()
        out[sa:sa+block_len] = out[sb:sb+block_len]
        out[sb:sb+block_len] = tmp
    return out

def cross_sample_replace(chunk, donor, blocks, block_len=4):
    out = chunk.clone()
    for b in blocks:
        s = b * block_len
        out[s:s+block_len] = donor[s:s+block_len]
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_pss_blockswap.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analyses/physical_signal_source.py block_lo_arm_order_network/tests/test_pss_blockswap.py
git commit -m "feat: block-swap / cross-sample content perturbation helpers"
```

---

### Task 8: Random-token stress control

**Files:**
- Modify: `analyses/physical_signal_source.py`
- Test: `block_lo_arm_order_network/tests/test_pss_randtoken.py`

**Interfaces:**
- Produces: `random_token_chunk(chunk, vocab_size, rng) -> Tensor` — replace all tokens with random ids (severe OOD; stress only).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pss_randtoken.py
import pathlib, sys
import numpy as np, torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.physical_signal_source import random_token_chunk

def test_random_token_in_range_and_changed():
    c = torch.arange(256)
    out = random_token_chunk(c, vocab_size=50, rng=np.random.default_rng(0))
    assert out.shape == c.shape and int(out.max()) < 50
    assert not torch.equal(out, c)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_pss_randtoken.py -q`
Expected: FAIL

- [ ] **Step 3: Implement**

```python
# append to analyses/physical_signal_source.py
def random_token_chunk(chunk, vocab_size, rng):
    return torch.from_numpy(rng.integers(0, vocab_size, size=tuple(chunk.shape))).to(chunk.dtype)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_pss_randtoken.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analyses/physical_signal_source.py block_lo_arm_order_network/tests/test_pss_randtoken.py
git commit -m "feat: random-token stress control (OOD, stress-only)"
```

---

### Task 9: E3 relayout supporting diagnostic

**Files:**
- Modify: `analyses/physical_signal_source.py`
- Test: `block_lo_arm_order_network/tests/test_pss_relayout.py`

**Interfaces:**
- Consumes: `position_prior_decomp.relayout_chunks`, `clean_perm_from_layout`, `make_layouts`; `canonical_reanalysis.canonical_scan`-style readout.
- Produces: `relayout_diagnostic(ckpt_path, layer, carrier_heads, K=4, M=8, device="cpu") -> dict` — anchor τ_physical vs relayout-mean τ_physical + drop, on the carrier heads.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pss_relayout.py
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.physical_signal_source import relayout_diagnostic

def test_relayout_reports_anchor_and_drop():
    out = relayout_diagnostic(str(ROOT/"runs/handoff_overnight/seed2/ckpt_step10000.pt"),
                              layer=0, carrier_heads=[2,3,4,5], K=2, M=8)
    assert "anchor_tau" in out and "relayout_mean_tau" in out and "drop" in out
    assert out["anchor_tau"] > 0.8                        # training-layout anchor valid
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_pss_relayout.py -q`
Expected: FAIL

- [ ] **Step 3: Implement** — load model + training perm; build the carrier's best per-head physical τ at the training layout (anchor) and under K relayouts (via `relayout_chunks`, shared random reveals); report anchor / relayout-mean / drop. Full code in the module, reusing `carrier_b65_per_text`'s extraction with relayouted chunks.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_pss_relayout.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analyses/physical_signal_source.py block_lo_arm_order_network/tests/test_pss_relayout.py
git commit -m "feat: E3 relayout supporting diagnostic (anchor vs relayout drop)"
```

---

### Task 10: Per-seed verdict writer (B / B+ / C / mixed)

**Files:**
- Modify: `analyses/physical_signal_source.py`
- Test: `block_lo_arm_order_network/tests/test_pss_verdict.py`

**Interfaces:**
- Produces: `default_r2_null_base_seeds(M) -> tuple(i*M for i in range(16))` and `matched_r2_null(M, mask, base_seeds=None) -> dict`. Each of the 16 draws uses `synthetic_content_randomized(M, seed=base)`, which consumes seed block `[base, base+M-1]`; supplied bases must be 16 nonnegative, non-bool integers whose blocks do not overlap. The returned raw/normalized null `mean` and population `std` use the same default text split as the real sample. `null_adjusted_excess(r2, null_mean) -> (r2-null_mean)/(1-null_mean)`.
- Produces: `classify_source(content_variance, noise_floor, r2_raw_excess, r2_norm_excess, r2_raw_null_effect, r2_norm_null_effect) -> str`. A null effect is the descriptive standardized distance `(r2-null_mean)/(null_std+1e-12)` across the 16 synthetic draws; it is **not** a z-test or inferential-significance claim. Clear fixed-map evidence requires both excesses `>=0.15` and both null effects `>=2`; clear absence requires both excesses `<=0.05` and both null effects `<=1`. These conservative heuristic thresholds require agreement of two representations. Then: map evidence + content ratio `<=0.5` → `B`; map evidence + ratio `>0.5` → `B+`; map absence + ratio `>=2` → `C`; every disagreement/near-null/boundary case → `mixed`. Absolute R² never decides the class alone.
- Produces: `run_seed(seed, root, out_dir, M=24, K_relayout=4, n_reveals=32) -> dict` — full E1(+E3) per carrier head. Every row reports `r2_slot_only_raw`, `r2_slot_only_norm`, both matched-null means/stds, both null-adjusted excesses, and both descriptive null effects. Nulls are count-matched to `len(use)` after validity gating, but **not tau-selection-matched**: the synthetic draws do not reproduce carrier-validity selection and must not be described as inferential controls. Writes `source.json` + `source.csv`.
- Consumes/extends: modifies `carrier_b65_per_text` (Task 1) to add `return_halves=False` → when True also returns `(halfA_list, halfB_list)` per-text B65 from two disjoint reveal halves.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pss_verdict.py
import pathlib, sys
import pytest
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.physical_signal_source import (
    classify_source, default_r2_null_base_seeds, matched_r2_null, valid_edge_mask)

def test_default_null_seed_blocks_are_disjoint():
    M = 24
    bases = default_r2_null_base_seeds(M)
    assert len(bases) == 16
    blocks = [set(range(base, base + M)) for base in bases]
    assert all(blocks[i].isdisjoint(blocks[j])
               for i in range(16) for j in range(i + 1, 16))

def test_matched_null_rejects_overlapping_seed_blocks():
    bases = list(default_r2_null_base_seeds(4))
    bases[1] = bases[0] + 1
    with pytest.raises(ValueError, match="overlap"):
        matched_r2_null(4, valid_edge_mask(65), base_seeds=bases)

def test_classify_source_rules():
    strong_map = dict(r2_raw_excess=0.3, r2_norm_excess=0.3,
                      r2_raw_null_effect=3.0, r2_norm_null_effect=3.0)
    absent_map = dict(r2_raw_excess=0.0, r2_norm_excess=0.0,
                      r2_raw_null_effect=0.0, r2_norm_null_effect=0.0)
    assert classify_source(0.001, 0.05, **strong_map) == "B"
    assert classify_source(0.2, 0.02, **absent_map) == "C"
    assert classify_source(0.06, 0.02, **strong_map) == "B+"
    # A high normalized absolute R² cannot rescue raw/null disagreement.
    disagree = dict(r2_raw_excess=0.02, r2_norm_excess=0.5,
                    r2_raw_null_effect=0.5, r2_norm_null_effect=5.0)
    assert classify_source(0.001, 0.05, **disagree) == "mixed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_pss_verdict.py -q`
Expected: FAIL

- [ ] **Step 3: Implement**

```python
# append to analyses/physical_signal_source.py
import csv as _csv, json as _json

def null_adjusted_excess(r2, null_mean):
    return float((r2 - null_mean) / (1.0 - null_mean + 1e-12))

def default_r2_null_base_seeds(M):
    if not isinstance(M, int) or isinstance(M, bool) or M < 2:
        raise ValueError("M must be an integer >= 2")
    return tuple(i * M for i in range(16))

def matched_r2_null(M, mask, base_seeds=None):
    bases = default_r2_null_base_seeds(M) if base_seeds is None else tuple(base_seeds)
    if len(bases) != 16:
        raise ValueError("base_seeds must contain exactly 16 entries")
    if any(isinstance(b, bool) or not isinstance(b, (int, np.integer)) or b < 0
           for b in bases):
        raise ValueError("base_seeds must be nonnegative integers")
    ordered = sorted(int(b) for b in bases)
    if any(right < left + M for left, right in zip(ordered, ordered[1:])):
        raise ValueError("base_seeds define overlapping seed blocks")
    vals = {"raw": [], "norm": []}
    for base in bases:
        Bs = synthetic_content_randomized(M, seed=int(base))
        vals["raw"].append(slot_only_r2(Bs, mask, normalize=False))
        vals["norm"].append(slot_only_r2(Bs, mask, normalize=True))
    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=0))}
            for k, v in vals.items()}

def classify_source(content_variance, noise_floor, r2_raw_excess, r2_norm_excess,
                    r2_raw_null_effect, r2_norm_null_effect):
    ratio = content_variance / (noise_floor + 1e-9)
    map_evidence = (r2_raw_excess >= 0.15 and r2_norm_excess >= 0.15 and
                    r2_raw_null_effect >= 2.0 and r2_norm_null_effect >= 2.0)
    map_absent = (r2_raw_excess <= 0.05 and r2_norm_excess <= 0.05 and
                  r2_raw_null_effect <= 1.0 and r2_norm_null_effect <= 1.0)
    if map_evidence and ratio <= 0.5:
        return "B"
    if map_absent and ratio >= 2.0:
        return "C"
    if map_evidence and ratio > 0.5:
        return "B+"
    return "mixed"

# FIRST modify carrier_b65_per_text (Task 1) to add `return_halves=False`: when True it
# also returns halfA_list, halfB_list — per-text B65 built from two disjoint halves of that
# text's reveals (first n_reveals//2 vs the rest). Run with n_reveals=32 (per Task-1 finding
# that per-text B saturates at ~8-32 reveals; 32 gives a clean within-text floor).
def run_seed(seed, root="runs/handoff_overnight", out_dir=None, M=24, K_relayout=4, n_reveals=32):
    from analyses.physical_signal_source import (
        carrier_b65_per_text, carrier_valid_filter, valid_edge_mask,
        cross_text_variance, pairwise_similarity, slot_only_r2,
        content_variance, within_text_noise_floor,
        synthetic_content_invariant, synthetic_content_randomized, relayout_diagnostic,
        matched_r2_null, null_adjusted_excess)
    CARRIERS = {2: (0, [2,3,4,5]), 42: (0, [2]), 123: (0, [1,2,3,4])}
    layer, heads = CARRIERS[seed]
    out = pathlib.Path(out_dir or f"runs/physical_signal_source/seed{seed}")
    out.mkdir(parents=True, exist_ok=True)
    mask = valid_edge_mask(65)
    vi = cross_text_variance(synthetic_content_invariant(M), mask)   # synthetic anchors (0 / high)
    vr = cross_text_variance(synthetic_content_randomized(M), mask)
    ckpt = f"{root}/seed{seed}/ckpt_step10000.pt"
    rows = []
    for h in heads:
        B_list, tau_list, hA, hB = carrier_b65_per_text(
            ckpt, layer, h, M=M, n_reveals=n_reveals, return_halves=True)
        Bv, idx = carrier_valid_filter(B_list, tau_list, thr=0.9 if len(heads) > 1 else 0.6)
        use = Bv if len(Bv) >= max(4, M // 4) else B_list
        hAv = [hA[i] for i in idx] if len(Bv) == len(idx) and use is Bv else hA
        hBv = [hB[i] for i in idx] if len(Bv) == len(idx) and use is Bv else hB
        floor = within_text_noise_floor(hAv, hBv, mask)
        cvar = content_variance(use, hAv, hBv, mask)            # between-text minus sampling floor
        var = cross_text_variance(use, mask)
        r2_raw = slot_only_r2(use, mask, normalize=False)
        r2_norm = slot_only_r2(use, mask, normalize=True)
        null = matched_r2_null(len(use), mask)
        raw_excess = null_adjusted_excess(r2_raw, null["raw"]["mean"])
        norm_excess = null_adjusted_excess(r2_norm, null["norm"]["mean"])
        raw_null_effect = ((r2_raw - null["raw"]["mean"]) /
                           (null["raw"]["std"] + 1e-12))
        norm_null_effect = ((r2_norm - null["norm"]["mean"]) /
                            (null["norm"]["std"] + 1e-12))
        sim = pairwise_similarity(use, mask)
        verdict = classify_source(
            cvar, floor, raw_excess, norm_excess, raw_null_effect, norm_null_effect)
        rows.append({"seed": seed, "layer": layer, "head": h, "n_valid": len(Bv),
                     "variance": var, "noise_floor": floor, "content_variance": cvar,
                     "r2_slot_only_raw": r2_raw, "r2_slot_only_norm": r2_norm,
                     "r2_null_raw_mean": null["raw"]["mean"],
                     "r2_null_raw_std": null["raw"]["std"],
                     "r2_null_norm_mean": null["norm"]["mean"],
                     "r2_null_norm_std": null["norm"]["std"],
                     "r2_raw_excess": raw_excess, "r2_norm_excess": norm_excess,
                     "r2_raw_null_effect": raw_null_effect,
                     "r2_norm_null_effect": norm_null_effect, "pairwise_sim": sim,
                     "var_invariant": vi, "var_randomized": vr, "verdict": verdict})
    relay = relayout_diagnostic(ckpt, layer, heads, K=K_relayout, M=8)
    with open(out / "source.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow(r)
    summary = {"seed": seed, "carrier": {"layer": layer, "heads": heads},
               "per_head": rows, "relayout": relay,
               "seed_verdict": max({r["verdict"] for r in rows},
                                   key=lambda v: sum(x["verdict"] == v for x in rows))}
    _json.dump(summary, open(out / "source.json", "w"), indent=2, default=float)
    return summary
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_pss_verdict.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analyses/physical_signal_source.py block_lo_arm_order_network/tests/test_pss_verdict.py
git commit -m "feat: per-seed source verdict (B/B+/C/mixed) + run_seed driver"
```

---

### Task 11: Plots (variance heatmap, R² bars, residual)

**Files:**
- Create: `analyses/plot_physical_signal_source.py`
- Test: `block_lo_arm_order_network/tests/test_pss_plot.py`

**Interfaces:**
- Produces: `plot_source(source_json, out_dir)` → `source_metrics.png` (per-head variance vs synthetic anchors + paired raw/normalized R² bars with their matched-null means + verdict labels).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pss_plot.py
import json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.plot_physical_signal_source import plot_source

def test_plot_source_writes_png(tmp_path):
    j = {"seed": 2, "carrier": {"layer": 0, "heads": [2]},
         "per_head": [{"head": 2, "variance": 0.01,
                       "r2_slot_only_raw": 0.4, "r2_slot_only_norm": 0.9,
                       "r2_null_raw_mean": -0.05, "r2_null_norm_mean": 0.7,
                       "var_invariant": 0.001, "var_randomized": 0.2, "verdict": "B"}],
         "relayout": {"anchor_tau": 1.0, "relayout_mean_tau": 0.1, "drop": -0.9},
         "seed_verdict": "B"}
    p = tmp_path / "source.json"; json.dump(j, open(p, "w"))
    plot_source(str(p), str(tmp_path))
    assert (tmp_path / "source_metrics.png").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_pss_plot.py -q`
Expected: FAIL

- [ ] **Step 3: Implement** — matplotlib Agg; per-head bar of variance with the two synthetic-anchor lines + paired raw/normalized R² bars and matched-null reference lines + verdict text. Full code in the module.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_pss_plot.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analyses/plot_physical_signal_source.py block_lo_arm_order_network/tests/test_pss_plot.py
git commit -m "feat: physical-signal-source metric plots"
```

---

### Task 12: Three-seed run + README with P3′ fork

**Files:**
- Create: `runs/physical_signal_source/` (outputs)
- Create: `analyses/physical_signal_source_README.md`

- [ ] **Step 1: Run all 3 seeds**

Run: `cd block_lo_arm_order_network && python -c "from analyses.physical_signal_source import run_seed; from analyses.plot_physical_signal_source import plot_source; [(_:=run_seed(s, root='../runs/handoff_overnight', out_dir=f'../runs/physical_signal_source/seed{s}'), plot_source(f'../runs/physical_signal_source/seed{s}/source.json', f'../runs/physical_signal_source/seed{s}')) for s in (2,42,123)]"`
Expected: `source.json/csv/png` per seed.

- [ ] **Step 2: Write `analyses/physical_signal_source_README.md`**

Per seed: per-head variance (vs synthetic anchors), raw + normalized slot-only R² with count-matched synthetic-null mean/std, null-adjusted excess and descriptive null effect (not inferential significance; not tau-selection-matched), pairwise sim, carrier-valid fraction, relayout drop, and the **B/B+/C/mixed verdict** (seed42 single-head reported separately). The line-level statement: is the L0 physical-order signal a fixed-layout map (B), content-dependent (C), or B+. Then the **pre-registered P3′ fork**: B → positional/QK causal intervention; C → content-feature causal intervention; B+ → separate base map from modulation; never output-ablation→same-head-readout.

- [ ] **Step 3: Commit**

```bash
git add -f analyses/physical_signal_source_README.md runs/physical_signal_source
git commit -m "results: P2 physical-order signal source disambiguation (3 seeds) + P3' fork"
```

---

## Self-Review

**Spec coverage:**
- E0 negative control (model-frame vs canonical) → documented in README (Task 12), data already exists. ✓
- E1 per-text B65 (Task 1), valid-edge mask + L1 norm (Task 2), variance + similarity (Task 4), slot-only R² text-split (Task 5), carrier-validity gate (Task 6), synthetic-baseline calibration (Tasks 3,4,5,10), block-swap secondary (Task 7), random-token stress (Task 8). ✓
- E2-order removed + tautology documented (Task 3 test). ✓
- E3 relayout supporting (Task 9). ✓
- Verdict B/B+/C/mixed per seed (Task 10), plots (Task 11), README + P3′ fork (Task 12). ✓
- seed42 single-head separate threshold (Task 10 `run_seed`). ✓

**Placeholder scan:** Tasks 9/11 step-3 give itemized algorithms over already-tested helpers
(`carrier_b65_per_text`, `relayout_chunks`, the metrics); all metric/predictor functions have
full code. No TBD/TODO in logic-bearing steps.

**Type consistency:** B65 `(65,65)` and `mask` threaded through `cross_text_variance` /
`pairwise_similarity` / `slot_only_r2` identically; `B_list` = list of `(65,65)`; carrier config
`(layer, heads)` consistent (Tasks 1,9,10); verdict strings `B/B+/C/mixed` consistent (Tasks 10,11,12).
