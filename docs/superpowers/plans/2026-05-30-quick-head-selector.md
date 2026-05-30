# Quick Head Selector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a cheap (O(N²), no-rollout) per-head order-score that pre-selects the top-k order-specialized attention heads of a text AOGPT run, and validate that its top-k recalls the expensive full-CDL-scan winner across the clean_base checkpoint ladder + the alt_from0_random@5k cross-run point.

**Architecture:** A pure-numpy scoring module (`quick_head_selector.py`) computes four structural scores per head from the per-head physical block graph `A` (C1 signed readiness-position corr, C2 signed flow drift, C3 readiness spread, C4 asymmetry strength); a selection rule sign-calibrates the ranking score against expensive `tau_vs_l2r`, masks dead/symmetric heads via C3/C4, and returns `best+ / best- / pool` (never `argmax|score|`). A driver (`validate_quick_selector.py`) re-extracts attention with the SAME seeded path used by the expensive scan, computes cheap scores, and reports Spearman / Recall@k / cross-run / null / cost-ratio.

**Tech Stack:** Python, numpy, scipy.stats (`spearmanr`, `corrcoef`), pytest. Reuses `per_head_order_scan.extract_per_head_and_heavy_A`, `neural_readout.extract_b._load_model_and_chunks`, `neural_readout.teacher_labels.generate_teacher_label`, `attn_order_mlp_policy.readiness_vector`. Spec: `docs/superpowers/specs/2026-05-30-quick-head-selector-design.md`.

---

## File Structure

| File | Responsibility |
|---|---|
| `block_lo_arm_order_network/quick_head_selector.py` | Pure logic: cheap scores (C1–C4), sign calibration, selection rule, metrics (Spearman / Recall@k), expensive-JSON loader. No torch, no training. |
| `block_lo_arm_order_network/validate_quick_selector.py` | GPU driver: re-extract A_lh per (ckpt,seed) via shared seeded path, compute cheap scores, calibrate, report Spearman/Recall/cross-run/null/cost-ratio → JSON + stdout. |
| `block_lo_arm_order_network/tests/test_quick_head_selector.py` | TDD unit tests for every pure function in `quick_head_selector.py`. |

**Conventions (load-bearing, from spec §3 + teacher_labels.py):**
- Per head, take the physical block graph `A` (N×N, N=64), build `B = Aᵀ` with zero diagonal (same `B` the CDL readout consumes via `_batch_mean_B`).
- Readiness `r(v) = out(v) − α·in(v)` where `out = B.sum(axis=1)` (row sum), `in = B.sum(axis=0)` (col sum), `α = 0.5`. This matches `readiness_vector` used by `generate_teacher_label`.
- L=4, H=8 → 32 heads. `extract_per_head_and_heavy_A` returns `A_lh: (n_chunks, L, H, N, N)`; cheap scores use the grand-mean over chunks `A_lh.mean(axis=0)` → `(L,H,N,N)`.
- **Sign is never preset.** C1/C2 raw signs are calibrated so positive ↔ positive expensive `tau_vs_l2r` (spec §3.1).
- **Never select by `|score|`** (spec §4): `best+ = argmax(signed)`, `best- = argmin(signed)`.

**Prerequisite (spec §2):** expensive ground-truth JSONs from the stability spec must exist as per-head scan files (schema = `per_head_layer_sorted_by_abs_tau_vs_l2r` list of `{layer, head, tau_vs_l2r, ...}`). Existing examples: `batch_readout/logs/diag_head_layer_scan*.json`. The full clean_base ladder (9 ckpt × 5 seed) is the stability-spec deliverable; the driver (Task 6) reads whatever matches its glob and errors clearly if the ladder is incomplete. **Tasks 1–5 have zero dependency on the prerequisite** (pure functions, synthetic inputs) and can be implemented immediately.

---

### Task 1: Cheap per-head scores (C1–C4), pure numpy

**Files:**
- Create: `block_lo_arm_order_network/quick_head_selector.py`
- Test: `block_lo_arm_order_network/tests/test_quick_head_selector.py`

- [ ] **Step 1: Write the failing test**

Create `block_lo_arm_order_network/tests/test_quick_head_selector.py`:

```python
import pathlib
import sys

import numpy as np
import pytest

_HERE = pathlib.Path(__file__).resolve().parent
_PKG = _HERE.parent
sys.path.insert(0, str(_PKG))

import quick_head_selector as qhs


def _chain_A(n=8, forward=True):
    """Physical graph A whose B=A.T encodes a clean directed chain.

    We want B[v, v+1] large (v points forward to v+1) so readiness r is high at
    low index, low at high index. Since B = A.T, that means A[v+1, v] large.
    """
    A = np.zeros((n, n), dtype=np.float64)
    for v in range(n - 1):
        if forward:
            A[v + 1, v] = 1.0      # B[v, v+1] = 1  -> v -> v+1
        else:
            A[v, v + 1] = 1.0      # B[v+1, v] = 1  -> v+1 -> v  (reversed chain)
    return A


def test_c1_raw_sign_separates_forward_vs_reversed():
    fwd = qhs.head_cheap_scores(_chain_A(8, forward=True))
    rev = qhs.head_cheap_scores(_chain_A(8, forward=False))
    # raw C1 must have OPPOSITE signs for forward vs reversed chains
    assert np.sign(fwd["C1"]) == -np.sign(rev["C1"])
    assert abs(fwd["C1"]) > 0.3 and abs(rev["C1"]) > 0.3


def test_c4_zero_for_symmetric_graph():
    A = np.array([[0.0, 1.0, 0.5],
                  [1.0, 0.0, 1.0],
                  [0.5, 1.0, 0.0]])  # symmetric -> B symmetric -> C4 ~ 0
    s = qhs.head_cheap_scores(A)
    assert s["C4"] < 1e-6


def test_c4_high_for_directed_graph():
    s = qhs.head_cheap_scores(_chain_A(8, forward=True))
    assert s["C4"] > 0.5


def test_c3_zero_for_dead_head():
    A = np.zeros((6, 6))            # no structure -> readiness flat -> C3 ~ 0
    s = qhs.head_cheap_scores(A)
    assert s["C3"] < 1e-9
    # degenerate graph must not crash C1/C2
    assert np.isfinite(s["C1"]) and np.isfinite(s["C2"])


def test_cheap_head_scores_shapes_and_chunk_mean():
    rng = np.random.default_rng(0)
    A_lh = rng.random((3, 4, 8, 8, 8))  # (n_chunks, L, H, N, N)
    out = qhs.cheap_head_scores(A_lh)
    for k in ("C1", "C2", "C3", "C4"):
        assert out[k].shape == (4, 8)
    # passing the pre-meaned (L,H,N,N) yields identical result
    out2 = qhs.cheap_head_scores(A_lh.mean(axis=0))
    assert np.allclose(out["C1"], out2["C1"])


def test_readiness_matches_cdl_convention():
    """head_cheap_scores' internal readiness must equal readiness_vector(B)."""
    from attn_order_mlp_policy import readiness_vector
    A = _chain_A(8, forward=True)
    B = A.T.copy()
    np.fill_diagonal(B, 0.0)
    r_ref = readiness_vector(B, alpha_dep=0.5)
    r_got = B.sum(axis=1) - 0.5 * B.sum(axis=0)
    assert np.allclose(r_ref, r_got)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_quick_head_selector.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'quick_head_selector'`.

- [ ] **Step 3: Write minimal implementation**

Create `block_lo_arm_order_network/quick_head_selector.py`:

```python
"""Quick head selector — cheap O(N^2) per-head order scores as a top-k
pre-selector for the order-specialized attention head of a text AOGPT run.

Positioning (spec docs/superpowers/specs/2026-05-30-quick-head-selector-design.md):
    A^{l,h} -> cheap score(l,h) -> top-k candidate heads -> CDL validation -> hook
The cheap score is a PRE-SELECTOR, not a final oracle.

Red lines (spec §2): L2R tau is a diagnostic microscope only (used to validate /
sign-calibrate the cheap score). It is NOT a training label and does NOT enter the
readout, MLP teacher, order generation, NLL, or reward. The teacher remains the
attention-derived CDL pseudo-order.

Conventions: per head take physical block graph A (N,N); B = A.T (zero diagonal);
readiness r(v) = out(v) - alpha*in(v) with out = B.sum(1), in = B.sum(0), alpha=0.5
(matches attn_order_mlp_policy.readiness_vector used by generate_teacher_label).
"""
import numpy as np

ALPHA_DEP = 0.5


def head_cheap_scores(A, alpha_dep=ALPHA_DEP, eps=1e-8):
    """Cheap order scores for ONE per-head physical block graph A (N, N).

    Returns dict {C1, C2, C3, C4} of raw floats (signs NOT calibrated):
        C1 signed readiness-position correlation: corr(r(v), index)
        C2 signed flow drift: mean over rows of weighted displacement under B
        C3 readiness spread: std(r)                       [dead-head filter]
        C4 asymmetry strength: ||B - B.T||_F / ||B||_F    [directionality filter]
    """
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    B = A.T.copy()
    np.fill_diagonal(B, 0.0)

    out = B.sum(axis=1)
    inn = B.sum(axis=0)
    r = out - alpha_dep * inn
    idx = np.arange(n, dtype=np.float64)

    if np.std(r) < eps:
        c1 = 0.0
    else:
        c1 = float(np.corrcoef(r, idx)[0, 1])

    # C2: per-row weighted mean displacement (j - v) under B, averaged over rows
    v_idx = np.arange(n, dtype=np.float64)
    disp = idx[None, :] - v_idx[:, None]        # (n, n): j - v
    row_w = B.sum(axis=1)
    nz = row_w > eps
    drift_rows = (B * disp).sum(axis=1)
    c2 = float((drift_rows[nz] / row_w[nz]).mean()) if nz.any() else 0.0

    c3 = float(np.std(r))

    num = float(np.linalg.norm(B - B.T))
    den = float(np.linalg.norm(B)) + eps
    c4 = num / den

    return {"C1": c1, "C2": c2, "C3": c3, "C4": c4}


def cheap_head_scores(A_lh, alpha_dep=ALPHA_DEP):
    """Cheap scores for all heads.

    A_lh: (L, H, N, N) batch-mean per-head physical A, or (n_chunks, L, H, N, N)
          which is averaged over chunks first.
    Returns dict {C1, C2, C3, C4: (L, H) float arrays} (raw, signs NOT calibrated).
    """
    A_lh = np.asarray(A_lh, dtype=np.float64)
    if A_lh.ndim == 5:
        A_lh = A_lh.mean(axis=0)
    elif A_lh.ndim != 4:
        raise ValueError(f"A_lh must be 4D or 5D, got {A_lh.shape}")
    L, H = A_lh.shape[:2]
    out = {k: np.zeros((L, H), dtype=np.float64) for k in ("C1", "C2", "C3", "C4")}
    for l in range(L):
        for h in range(H):
            s = head_cheap_scores(A_lh[l, h], alpha_dep=alpha_dep)
            for k in out:
                out[k][l, h] = s[k]
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_quick_head_selector.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
cd /home/admin/lyuyuhuan/order_lyu
git add block_lo_arm_order_network/quick_head_selector.py block_lo_arm_order_network/tests/test_quick_head_selector.py
git commit -m "feat(quick-head-selector): Task 1 — cheap per-head scores C1-C4"
```

---

### Task 2: Sign calibration

**Files:**
- Modify: `block_lo_arm_order_network/quick_head_selector.py`
- Test: `block_lo_arm_order_network/tests/test_quick_head_selector.py`

- [ ] **Step 1: Write the failing test**

Append to `block_lo_arm_order_network/tests/test_quick_head_selector.py`:

```python
def test_calibrate_sign_flips_when_anticorrelated():
    raw = np.array([[3.0, 1.0, -1.0, -3.0]])      # (1,4)
    tau = np.array([[-0.9, -0.3, 0.3, 0.9]])      # raw is ANTI-correlated with tau
    sign, cal = qhs.calibrate_sign(raw, tau)
    assert sign == -1.0
    from scipy.stats import spearmanr
    rho, _ = spearmanr(cal.ravel(), tau.ravel())
    assert rho > 0                                  # calibrated now positively aligned


def test_calibrate_sign_keeps_when_correlated():
    raw = np.array([[-3.0, -1.0, 1.0, 3.0]])
    tau = np.array([[-0.9, -0.3, 0.3, 0.9]])
    sign, cal = qhs.calibrate_sign(raw, tau)
    assert sign == 1.0
    assert np.allclose(cal, raw)


def test_calibrate_sign_degenerate_returns_identity():
    raw = np.zeros((2, 2))
    tau = np.array([[0.1, -0.2], [0.3, 0.4]])
    sign, cal = qhs.calibrate_sign(raw, tau)
    assert sign == 1.0
    assert np.allclose(cal, raw)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_quick_head_selector.py -k calibrate_sign -v`
Expected: FAIL — `AttributeError: module 'quick_head_selector' has no attribute 'calibrate_sign'`.

- [ ] **Step 3: Write minimal implementation**

Add to the top imports of `quick_head_selector.py`:

```python
from scipy.stats import spearmanr
```

Append to `quick_head_selector.py`:

```python
def calibrate_sign(raw, expensive_tau):
    """Choose a global sign so the score is POSITIVELY aligned with expensive tau.

    raw, expensive_tau: (L, H) arrays. Returns (sign, calibrated) where sign in
    {+1.0, -1.0} and calibrated = sign * raw. Spec §3.1: positive cheap score is
    DEFINED to correlate positively with expensive tau_vs_l2r on the calibration
    split. Degenerate inputs (constant raw / too few pairs) return (+1, raw).
    """
    raw = np.asarray(raw, dtype=np.float64)
    tau = np.asarray(expensive_tau, dtype=np.float64)
    rf, tf = raw.ravel(), tau.ravel()
    mask = ~(np.isnan(rf) | np.isnan(tf))
    if mask.sum() < 3 or np.std(rf[mask]) < 1e-12 or np.std(tf[mask]) < 1e-12:
        return 1.0, raw.copy()
    rho, _ = spearmanr(rf[mask], tf[mask])
    sign = -1.0 if (rho is not None and not np.isnan(rho) and rho < 0) else 1.0
    return sign, sign * raw
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_quick_head_selector.py -k calibrate_sign -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
cd /home/admin/lyuyuhuan/order_lyu
git add block_lo_arm_order_network/quick_head_selector.py block_lo_arm_order_network/tests/test_quick_head_selector.py
git commit -m "feat(quick-head-selector): Task 2 — sign calibration vs expensive tau"
```

---

### Task 3: Selection rule (best+ / best- / pool; never |score|)

**Files:**
- Modify: `block_lo_arm_order_network/quick_head_selector.py`
- Test: `block_lo_arm_order_network/tests/test_quick_head_selector.py`

- [ ] **Step 1: Write the failing test**

Append to `block_lo_arm_order_network/tests/test_quick_head_selector.py`:

```python
def _scores_with(c1, c3=None, c4=None):
    c1 = np.asarray(c1, dtype=np.float64)
    if c3 is None:
        c3 = np.ones_like(c1)        # all heads pass dead filter
    if c4 is None:
        c4 = np.ones_like(c1)        # all heads pass symmetry filter
    return {"C1": c1, "C2": c1.copy(), "C3": np.asarray(c3, float),
            "C4": np.asarray(c4, float)}


def test_select_heads_never_picks_argmax_abs():
    # negative head has the LARGEST magnitude; best_positive must still be the
    # positive head, NOT the |score| winner.
    scores = _scores_with([[0.4, -0.9, 0.1]])
    sel = qhs.select_heads(scores, sign=1.0, rank_score="C1",
                           dead_thresh=0.0, sym_thresh=0.0, rule="pool", k=2)
    assert sel["best_positive"] == (0, 0, 1)        # (layer, head, sign=+1)
    assert sel["best_negative"][0] == (0, 1, -1)    # strongest anti-L2R head


def test_select_heads_pool_contents():
    scores = _scores_with([[0.4, -0.9, -0.7, 0.2]])
    sel = qhs.select_heads(scores, sign=1.0, rank_score="C1",
                           dead_thresh=0.0, sym_thresh=0.0, rule="pool", k=2)
    pool_heads = {(l, h) for (l, h, _s) in sel["pool"]}
    assert (0, 0) in pool_heads                      # best_positive
    assert (0, 1) in pool_heads and (0, 2) in pool_heads   # top-2 negatives
    assert len(sel["pool"]) == 3


def test_select_heads_single_rule():
    scores = _scores_with([[0.4, -0.9, 0.1]])
    sel = qhs.select_heads(scores, sign=1.0, rank_score="C1", rule="single")
    assert sel["pool"] == [(0, 0, 1)]


def test_select_heads_masks_dead_and_symmetric():
    # head (0,0) has top C1 but is DEAD (C3=0); head (0,2) is SYMMETRIC (C4=0).
    # Only (0,1) survives the masks -> it becomes best_positive.
    scores = {"C1": np.array([[0.9, 0.5, 0.8]]),
              "C2": np.array([[0.9, 0.5, 0.8]]),
              "C3": np.array([[0.0, 1.0, 1.0]]),
              "C4": np.array([[1.0, 1.0, 0.0]])}
    sel = qhs.select_heads(scores, sign=1.0, rank_score="C1",
                           dead_thresh=1e-6, sym_thresh=1e-6, rule="pool", k=2)
    assert sel["best_positive"] == (0, 1, 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_quick_head_selector.py -k select_heads -v`
Expected: FAIL — `AttributeError: module 'quick_head_selector' has no attribute 'select_heads'`.

- [ ] **Step 3: Write minimal implementation**

Append to `quick_head_selector.py`:

```python
def select_heads(scores, sign=1.0, rank_score="C1", dead_thresh=0.0,
                 sym_thresh=0.0, rule="pool", k=2):
    """Select order heads from cheap scores. NEVER argmax|score| (spec §4).

    Args:
        scores: dict of (L, H) raw arrays from cheap_head_scores.
        sign: +1/-1 sign calibration for the ranking score (from calibrate_sign).
        rank_score: "C1" or "C2" — which signed score to rank by.
        dead_thresh: heads with C3 <= dead_thresh are masked (dead/uniform).
        sym_thresh: heads with C4 <= sym_thresh are masked (symmetric/local-only).
        rule: "pool" -> {best_positive} + top-k negatives; "single" -> {best_positive}.
        k: number of negative heads in the pool.

    Returns dict:
        best_positive: (layer, head, +1)   — argmax of signed score over masked heads
        best_negative: list of (layer, head, -1), strongest anti first (len up to k)
        pool: list of (layer, head, sign) per `rule`
    """
    signed = sign * np.asarray(scores[rank_score], dtype=np.float64)
    c3 = np.asarray(scores["C3"], dtype=np.float64)
    c4 = np.asarray(scores["C4"], dtype=np.float64)
    mask = (c3 > dead_thresh) & (c4 > sym_thresh)
    if not mask.any():
        raise ValueError("all heads masked out — relax dead_thresh/sym_thresh")

    masked = np.where(mask, signed, np.nan)
    flat = masked.ravel()
    L, H = signed.shape

    pos_i = int(np.nanargmax(flat))
    best_positive = (pos_i // H, pos_i % H, 1)

    # negatives: most-negative signed scores first, among masked heads
    order = np.argsort(np.where(np.isnan(flat), np.inf, flat))  # ascending; nan last
    n_valid = int(mask.sum())
    neg_idx = [int(i) for i in order[:min(k, n_valid)]]
    best_negative = [(i // H, i % H, -1) for i in neg_idx]

    if rule == "single":
        pool = [best_positive]
    elif rule == "pool":
        pool = [best_positive] + [hd for hd in best_negative
                                  if (hd[0], hd[1]) != (best_positive[0], best_positive[1])]
    else:
        raise ValueError(f"unknown rule {rule!r}")

    return {"best_positive": best_positive,
            "best_negative": best_negative,
            "pool": pool}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_quick_head_selector.py -k select_heads -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
cd /home/admin/lyuyuhuan/order_lyu
git add block_lo_arm_order_network/quick_head_selector.py block_lo_arm_order_network/tests/test_quick_head_selector.py
git commit -m "feat(quick-head-selector): Task 3 — selection rule (best+/best-/pool, no |score|)"
```

---

### Task 4: Validation metrics — Spearman + Recall@k

**Files:**
- Modify: `block_lo_arm_order_network/quick_head_selector.py`
- Test: `block_lo_arm_order_network/tests/test_quick_head_selector.py`

- [ ] **Step 1: Write the failing test**

Append to `block_lo_arm_order_network/tests/test_quick_head_selector.py`:

```python
def test_spearman_cheap_expensive_perfect():
    cheap = np.array([[1.0, 2.0, 3.0, 4.0]])
    tau = np.array([[0.1, 0.2, 0.3, 0.4]])
    assert qhs.spearman_cheap_expensive(cheap, tau) == pytest.approx(1.0)


def test_spearman_handles_nan():
    cheap = np.array([[1.0, 2.0, np.nan, 4.0]])
    tau = np.array([[0.1, 0.2, 0.3, 0.4]])
    rho = qhs.spearman_cheap_expensive(cheap, tau)
    assert rho == pytest.approx(1.0)        # nan pair dropped, rest perfect


def test_recall_at_k_positive_hit():
    # expensive best+ is head (0,3) (max tau). cheap signed top-2 positive must
    # contain it for recall@2 to be True.
    signed = np.array([[0.1, 0.2, 0.9, 0.8]])
    tau = np.array([[0.0, 0.1, 0.5, 0.7]])  # argmax tau = head 3
    assert qhs.recall_at_k(signed, tau, k=2, which="pos") is True
    assert qhs.recall_at_k(signed, tau, k=1, which="pos") is False  # cheap top1 = head2


def test_recall_at_k_negative_hit():
    # expensive best- is argmin tau = head 0. cheap signed most-negative top-2
    # must contain it.
    signed = np.array([[-0.9, -0.8, 0.2, 0.5]])
    tau = np.array([[-0.7, -0.5, 0.1, 0.4]])
    assert qhs.recall_at_k(signed, tau, k=2, which="neg") is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_quick_head_selector.py -k "spearman or recall" -v`
Expected: FAIL — `AttributeError: ... has no attribute 'spearman_cheap_expensive'`.

- [ ] **Step 3: Write minimal implementation**

Append to `quick_head_selector.py`:

```python
def spearman_cheap_expensive(cheap_lh, expensive_tau_lh):
    """Spearman rho between a cheap (L,H) score and expensive tau (L,H).

    NaN pairs (either side) are dropped. Returns nan if < 3 valid pairs.
    """
    a = np.asarray(cheap_lh, dtype=np.float64).ravel()
    b = np.asarray(expensive_tau_lh, dtype=np.float64).ravel()
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 3:
        return float("nan")
    rho, _ = spearmanr(a[m], b[m])
    return float(rho)


def recall_at_k(signed_cheap_lh, expensive_tau_lh, k, which="pos"):
    """Does the cheap top-k contain the expensive winner?

    which="pos": winner = argmax(expensive_tau); cheap top-k = k highest signed.
    which="neg": winner = argmin(expensive_tau); cheap top-k = k lowest signed.
    NaN-safe. Returns bool.
    """
    s = np.asarray(signed_cheap_lh, dtype=np.float64).ravel()
    t = np.asarray(expensive_tau_lh, dtype=np.float64).ravel()
    tmask = ~np.isnan(t)
    if not tmask.any():
        return False
    t_filled_for_pos = np.where(tmask, t, -np.inf)
    t_filled_for_neg = np.where(tmask, t, np.inf)

    if which == "pos":
        winner = int(np.argmax(t_filled_for_pos))
        ranked = np.argsort(np.where(np.isnan(s), -np.inf, s))[::-1]  # high first
    elif which == "neg":
        winner = int(np.argmin(t_filled_for_neg))
        ranked = np.argsort(np.where(np.isnan(s), np.inf, s))          # low first
    else:
        raise ValueError(f"which must be 'pos' or 'neg', got {which!r}")

    return winner in set(int(i) for i in ranked[:k])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_quick_head_selector.py -k "spearman or recall" -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
cd /home/admin/lyuyuhuan/order_lyu
git add block_lo_arm_order_network/quick_head_selector.py block_lo_arm_order_network/tests/test_quick_head_selector.py
git commit -m "feat(quick-head-selector): Task 4 — Spearman + Recall@k metrics"
```

---

### Task 5: Expensive-JSON loader + threshold calibration helper

**Files:**
- Modify: `block_lo_arm_order_network/quick_head_selector.py`
- Test: `block_lo_arm_order_network/tests/test_quick_head_selector.py`

- [ ] **Step 1: Write the failing test**

Append to `block_lo_arm_order_network/tests/test_quick_head_selector.py`:

```python
def test_load_expensive_tau_parses_schema(tmp_path):
    import json
    payload = {"config": {"L": 2, "H": 2},
               "per_head_layer_sorted_by_abs_tau_vs_l2r": [
                   {"layer": 0, "head": 0, "tau_vs_l2r": 0.5},
                   {"layer": 0, "head": 1, "tau_vs_l2r": -0.3},
                   {"layer": 1, "head": 0, "tau_vs_l2r": 0.1},
                   {"layer": 1, "head": 1, "tau_vs_l2r": -0.7}]}
    p = tmp_path / "scan.json"
    p.write_text(json.dumps(payload))
    arr = qhs.load_expensive_tau(str(p), L=2, H=2)
    assert arr.shape == (2, 2)
    assert arr[0, 0] == 0.5 and arr[1, 1] == -0.7


def test_load_expensive_tau_missing_head_is_nan(tmp_path):
    import json
    payload = {"per_head_layer_sorted_by_abs_tau_vs_l2r": [
        {"layer": 0, "head": 0, "tau_vs_l2r": 0.4}]}
    p = tmp_path / "scan.json"
    p.write_text(json.dumps(payload))
    arr = qhs.load_expensive_tau(str(p), L=2, H=2)
    assert arr[0, 0] == 0.4
    assert np.isnan(arr[0, 1]) and np.isnan(arr[1, 0])


def test_null_quantile_threshold():
    # step0 null distribution -> threshold at the given percentile
    null_vals = np.array([[0.0, 0.1, 0.2, 0.3, 0.4, 1.0]])
    thr95 = qhs.null_quantile_threshold(null_vals, pct=95)
    thr90 = qhs.null_quantile_threshold(null_vals, pct=90)
    assert thr95 == pytest.approx(np.percentile(null_vals, 95))
    assert thr90 <= thr95
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_quick_head_selector.py -k "expensive_tau or null_quantile" -v`
Expected: FAIL — `AttributeError: ... has no attribute 'load_expensive_tau'`.

- [ ] **Step 3: Write minimal implementation**

Add to the imports of `quick_head_selector.py`:

```python
import json
```

Append to `quick_head_selector.py`:

```python
def load_expensive_tau(json_path, L=4, H=8):
    """Parse one per-head scan JSON into a (L, H) tau_vs_l2r array.

    Schema (per_head_order_scan.scan_checkpoint output):
        {"per_head_layer_sorted_by_abs_tau_vs_l2r": [{layer, head, tau_vs_l2r, ...}]}
    Missing (layer, head) entries are NaN.
    """
    with open(json_path) as f:
        d = json.load(f)
    arr = np.full((L, H), np.nan, dtype=np.float64)
    for e in d["per_head_layer_sorted_by_abs_tau_vs_l2r"]:
        arr[int(e["layer"]), int(e["head"])] = float(e["tau_vs_l2r"])
    return arr


def null_quantile_threshold(null_scores, pct=95):
    """Threshold at the `pct` percentile of a step0 null score distribution.

    Spec §3.2: dead_thresh / sym_thresh are set so only heads significantly above
    the random-init level survive. default pct=95, fallback pct=90.
    """
    vals = np.asarray(null_scores, dtype=np.float64).ravel()
    vals = vals[~np.isnan(vals)]
    return float(np.percentile(vals, pct))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_quick_head_selector.py -v`
Expected: PASS (all ~20 tests in the file).

- [ ] **Step 5: Commit**

```bash
cd /home/admin/lyuyuhuan/order_lyu
git add block_lo_arm_order_network/quick_head_selector.py block_lo_arm_order_network/tests/test_quick_head_selector.py
git commit -m "feat(quick-head-selector): Task 5 — expensive-JSON loader + null-quantile thresholds"
```

---

### Task 6: Validation driver (GPU integration)

**Files:**
- Create: `block_lo_arm_order_network/validate_quick_selector.py`

This task wires the pure functions to real checkpoints. It has no unit test (it is an orchestration script that needs a GPU + the prerequisite expensive JSONs); it is verified by a smoke run on whatever single ckpt + JSON already exists, then run in full once the stability-spec ladder is on disk.

- [ ] **Step 1: Write the driver**

Create `block_lo_arm_order_network/validate_quick_selector.py`:

```python
"""Validate the quick head selector against the expensive full-CDL per-head scan.

Pipeline (spec docs/superpowers/specs/2026-05-30-quick-head-selector-design.md §5):
  for each (ckpt, seed):
      re-extract A_lh via the SAME seeded path the expensive scan used
      (extract_per_head_and_heavy_A) -> grand-mean over chunks -> cheap_head_scores
      load expensive tau_vs_l2r from the matching scan JSON
  pool all (ckpt, head): pick rank_score in {C1, C2} by |Spearman| vs expensive tau,
      fix its sign (calibrate_sign)
  report: per-step Spearman; Recall@1/3/5 for best+ and best-; cross-run 5k recovery;
      step0/1k null behavior; cost ratio (cheap scoring vs full per-head CDL readout).

Red lines (spec §2): L2R tau is a diagnostic only; nothing here feeds training,
the MLP teacher, order generation, NLL, or reward.
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from training_utils import N
from neural_readout.extract_b import _load_model_and_chunks
from per_head_order_scan import (
    extract_per_head_and_heavy_A, _batch_mean_B, _orders_from_graphs,
)
import quick_head_selector as qhs

# clean_base ladder (spec §5); alt_from0_random has ONLY step5000 (data reality).
CLEAN_BASE_DIR = _HERE / "probe_results" / "clean_base_random_perm"
CLEAN_BASE_STEPS = [0, 1000, 5000, 10000, 20000, 30000, 40000, 50000, 60000]
ALT_FROM0_CKPT = _HERE.parent / "probe_results" / "attention_order_mlp" / \
    "alt_from0_random" / "ckpt_step5000.pt"


def cheap_scores_for_ckpt(ckpt_path, M, batch_size, seed, device):
    """Re-extract A_lh (shared seeded path) -> grand-mean -> cheap raw scores.

    Returns (scores_dict, A_lh) so the caller can reuse A_lh for cost timing.
    """
    total = M * batch_size
    model, chunks, clean_perm, dev, _ = _load_model_and_chunks(
        ckpt_path, total, seed, device, "train")
    A_lh, _A_heavy = extract_per_head_and_heavy_A(model, chunks, clean_perm, dev, seed)
    scores = qhs.cheap_head_scores(A_lh)            # uses grand-mean over chunks
    return scores, A_lh


def expensive_json_path(scan_dir, step, seed):
    """Resolve the expensive scan JSON for (step, seed). Tries the ladder naming
    `ckpt{step}_seed{seed}.json` then a flat `*step{step}*seed{seed}*.json` glob."""
    scan_dir = pathlib.Path(scan_dir)
    direct = scan_dir / f"ckpt{step}_seed{seed}.json"
    if direct.exists():
        return direct
    hits = sorted(scan_dir.glob(f"*{step}*seed{seed}*.json"))
    return hits[0] if hits else None


def cost_ratio(A_lh, M, batch_size, alpha_dep=0.5, n_top_k=3):
    """Honest post-extraction cost ratio (spec §5.5). Both methods share the
    forward extraction; the saving is the per-head CDL rollout.

    expensive = full per-head CDL readout over ALL heads x M batch-means.
    cheap     = cheap_head_scores over all heads + CDL readout for top-k heads only.
    Returns dict {t_expensive, t_cheap, ratio}.
    """
    L, H = A_lh.shape[1], A_lh.shape[2]

    t0 = time.perf_counter()
    for l in range(L):
        for h in range(H):
            B = _batch_mean_B(A_lh[:, l, h], M, batch_size)
            _orders_from_graphs(B, alpha_dep)
    t_expensive = time.perf_counter() - t0

    t0 = time.perf_counter()
    qhs.cheap_head_scores(A_lh)
    for _ in range(n_top_k):                          # CDL only on top-k survivors
        B = _batch_mean_B(A_lh[:, 0, 0], M, batch_size)
        _orders_from_graphs(B, alpha_dep)
    t_cheap = time.perf_counter() - t0

    return {"t_expensive": t_expensive, "t_cheap": t_cheap,
            "ratio": t_expensive / t_cheap if t_cheap > 0 else float("nan")}


def run(scan_dir, M, batch_size, seeds, device, out_path):
    dev = torch.device(device)
    steps_present = [s for s in CLEAN_BASE_STEPS
                     if (CLEAN_BASE_DIR / f"ckpt_step{s}.pt").exists()]
    if not steps_present:
        raise SystemExit(f"no clean_base ckpts under {CLEAN_BASE_DIR}")

    # 1) cheap raw scores per step (seed0 extraction is representative & cheap);
    #    expensive tau averaged over the available sampling seeds.
    raw_by_step, tau_by_step, missing = {}, {}, []
    A_cache = {}
    for s in steps_present:
        ckpt = CLEAN_BASE_DIR / f"ckpt_step{s}.pt"
        scores, A_lh = cheap_scores_for_ckpt(ckpt, M, batch_size, seeds[0], dev)
        raw_by_step[s] = scores
        A_cache[s] = A_lh
        taus = []
        for sd in seeds:
            jp = expensive_json_path(scan_dir, s, sd)
            if jp is not None:
                taus.append(qhs.load_expensive_tau(str(jp), L=A_lh.shape[1], H=A_lh.shape[2]))
            else:
                missing.append((s, sd))
        tau_by_step[s] = np.nanmean(np.stack(taus), axis=0) if taus else None

    usable = [s for s in steps_present if tau_by_step[s] is not None]
    if not usable:
        raise SystemExit(
            f"no expensive scan JSONs found under {scan_dir} (prerequisite: run the "
            f"stability-spec per-head scan first). missing={missing}")

    # 2) pool over usable steps to pick rank_score + sign (calibration split = all).
    pooled_raw = {c: np.concatenate([raw_by_step[s][c].ravel() for s in usable])
                  for c in ("C1", "C2")}
    pooled_tau = np.concatenate([tau_by_step[s].ravel() for s in usable])
    rho_abs = {c: abs(qhs.spearman_cheap_expensive(pooled_raw[c], pooled_tau))
               for c in ("C1", "C2")}
    rank_score = max(rho_abs, key=rho_abs.get)
    sign, _ = qhs.calibrate_sign(pooled_raw[rank_score].reshape(1, -1),
                                 pooled_tau.reshape(1, -1))

    # 3) thresholds from the step0 null (default 95th, fallback 90th).
    thresholds = {}
    if 0 in raw_by_step:
        for pct in (95, 90):
            thresholds[pct] = {
                "dead_thresh": qhs.null_quantile_threshold(raw_by_step[0]["C3"], pct),
                "sym_thresh": qhs.null_quantile_threshold(raw_by_step[0]["C4"], pct)}

    # 4) per-step Spearman + Recall@k.
    per_step = {}
    for s in usable:
        signed = sign * raw_by_step[s][rank_score]
        per_step[s] = {
            "spearman_rank": qhs.spearman_cheap_expensive(signed, tau_by_step[s]),
            "spearman_C1": qhs.spearman_cheap_expensive(raw_by_step[s]["C1"], tau_by_step[s]),
            "spearman_C2": qhs.spearman_cheap_expensive(raw_by_step[s]["C2"], tau_by_step[s]),
            "recall_pos": {k: qhs.recall_at_k(signed, tau_by_step[s], k, "pos") for k in (1, 3, 5)},
            "recall_neg": {k: qhs.recall_at_k(signed, tau_by_step[s], k, "neg") for k in (1, 3, 5)},
        }

    # 5) cross-run 5k recovery (clean_base@5k vs alt_from0_random@5k).
    cross_run = None
    if 5000 in usable and ALT_FROM0_CKPT.exists():
        alt_scores, alt_A = cheap_scores_for_ckpt(ALT_FROM0_CKPT, M, batch_size, seeds[0], dev)
        alt_jp = expensive_json_path(scan_dir, "alt5000", seeds[0])
        cross_run = {
            "clean_base_5k_recall_pos": per_step[5000]["recall_pos"],
            "alt_from0_5k_selected": _selected_summary(alt_scores, sign, rank_score, thresholds),
            "note": "alt_from0_random has only ckpt_step5000 (spec §5); expensive "
                    "tau for alt available only if its scan JSON exists.",
        }
        if alt_jp is not None:
            alt_tau = qhs.load_expensive_tau(str(alt_jp), L=alt_A.shape[1], H=alt_A.shape[2])
            alt_signed = sign * alt_scores[rank_score]
            cross_run["alt_from0_5k_recall_pos"] = {
                k: qhs.recall_at_k(alt_signed, alt_tau, k, "pos") for k in (1, 3, 5)}

    # 6) cost ratio (use the latest usable step's cached A_lh).
    cost = cost_ratio(A_cache[usable[-1]], M, batch_size)

    report = {
        "config": {"M": M, "batch_size": batch_size, "seeds": seeds,
                   "scan_dir": str(scan_dir), "rank_score": rank_score, "sign": sign,
                   "steps_usable": usable},
        "pooled_spearman_abs": rho_abs,
        "thresholds_from_step0_null": thresholds,
        "per_step": per_step,
        "cross_run_5k": cross_run,
        "cost_ratio": cost,
        "missing_expensive_json": missing,
    }
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=1, default=_jsonable)
    _print_summary(report)
    return report


def _selected_summary(scores, sign, rank_score, thresholds):
    pct = 95 if 95 in thresholds else (90 if 90 in thresholds else None)
    if pct is None:
        sel = qhs.select_heads(scores, sign=sign, rank_score=rank_score,
                               dead_thresh=0.0, sym_thresh=0.0)
    else:
        sel = qhs.select_heads(scores, sign=sign, rank_score=rank_score,
                               dead_thresh=thresholds[pct]["dead_thresh"],
                               sym_thresh=thresholds[pct]["sym_thresh"])
    return {"best_positive": sel["best_positive"], "pool": sel["pool"]}


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _print_summary(r):
    print(f"rank_score={r['config']['rank_score']} sign={r['config']['sign']:+.0f} "
          f"pooled|rho| C1={r['pooled_spearman_abs']['C1']:.3f} "
          f"C2={r['pooled_spearman_abs']['C2']:.3f}")
    for s, d in sorted(r["per_step"].items()):
        print(f"  step{s:>6}: rho_rank={d['spearman_rank']:.3f}  "
              f"recall+@1/3/5={tuple(d['recall_pos'].values())}  "
              f"recall-@1/3/5={tuple(d['recall_neg'].values())}")
    c = r["cost_ratio"]
    print(f"cost: expensive={c['t_expensive']:.3f}s cheap={c['t_cheap']:.3f}s "
          f"ratio={c['ratio']:.1f}x")
    if r["missing_expensive_json"]:
        print(f"WARNING missing expensive JSON for: {r['missing_expensive_json']}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scan-dir", default=str(_HERE / "batch_readout" / "logs" / "per_head_scan"))
    p.add_argument("--M", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default=str(_HERE / "batch_readout" / "logs" /
                                        "quick_selector_validation.json"))
    args = p.parse_args()
    run(args.scan_dir, args.M, args.batch_size, args.seeds, args.device, args.out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Static-import smoke (no GPU, no data needed)**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -c "import sys; sys.path.insert(0,'block_lo_arm_order_network'); import validate_quick_selector as v; print('imports OK', v.CLEAN_BASE_STEPS)"`
Expected: prints `imports OK [0, 1000, 5000, ...]` with no import error. (Confirms all reused symbols resolve.)

- [ ] **Step 3: Prerequisite check**

Run: `cd /home/admin/lyuyuhuan/order_lyu && ls block_lo_arm_order_network/batch_readout/logs/per_head_scan/ 2>/dev/null; ls block_lo_arm_order_network/probe_results/clean_base_random_perm/*.pt`
Expected: clean_base ckpts listed. If `per_head_scan/` is empty/absent, the expensive ladder has not been run yet — that is the stability-spec deliverable (spec §2 prerequisite). The driver will error with a clear message until those JSONs exist; proceed to Step 4 only with at least one matching JSON.

- [ ] **Step 4: One-ckpt GPU smoke**

Run (small M to be fast; point `--scan-dir` at whatever expensive JSONs exist, and restrict to one step+seed by temporarily passing `--seeds 0`):
`cd /home/admin/lyuyuhuan/order_lyu && python block_lo_arm_order_network/validate_quick_selector.py --M 20 --batch-size 32 --seeds 0 --device cuda:0 --scan-dir block_lo_arm_order_network/batch_readout/logs/per_head_scan --out /tmp/qsel_smoke.json`
Expected: runs to completion, prints the per-step summary + `cost: ... ratio=Nx`, writes `/tmp/qsel_smoke.json`. (If no GPU is free, defer to when one is — the math is already unit-covered by Tasks 1–5.)

- [ ] **Step 5: Commit**

```bash
cd /home/admin/lyuyuhuan/order_lyu
git add block_lo_arm_order_network/validate_quick_selector.py
git commit -m "feat(quick-head-selector): Task 6 — validation driver (Spearman/Recall/cross-run/cost)"
```

---

### Task 7: Full validation run + findings doc

**Files:**
- Create: `block_lo_arm_order_network/batch_readout/logs/quick_selector_validation.json` (produced)
- Create: `docs/superpowers/specs/2026-05-30-quick-head-selector-findings.md` (or append to spec)

This task only runs once the expensive ladder (clean_base 9 ckpt × 5 seed, + ideally alt_from0_random@5k scan) is on disk.

- [ ] **Step 1: Run the full validation**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python block_lo_arm_order_network/validate_quick_selector.py --M 100 --batch-size 32 --seeds 0 1 2 3 4 --device cuda:0 --scan-dir block_lo_arm_order_network/batch_readout/logs/per_head_scan --out block_lo_arm_order_network/batch_readout/logs/quick_selector_validation.json`
Expected: completes; prints pooled |rho| for C1/C2, per-step rho + recall, cost ratio.

- [ ] **Step 2: Write the findings doc**

Create `docs/superpowers/specs/2026-05-30-quick-head-selector-findings.md` capturing (against spec §5.1 desired criteria — descriptive, NOT hard gates):
- chosen `rank_score` (C1 vs C2) and its sign; pooled Spearman;
- per-step Spearman curve and Recall@1/3/5 (pos & neg), noting where signal emerges (≥20k expected high);
- step0/1k null behavior (cheap score should not fabricate a strong head);
- cross-run 5k: did the selector recover each run's own winner (clean_base vs alt_from0_random);
- threshold sensitivity (95th vs 90th);
- measured cost ratio;
- verdict in spec language: does top-3/top-5 reliably cover the expensive winner (minimum useful outcome)? If not, report as a clean negative (cheap proxy insufficient to replace CDL).

- [ ] **Step 3: Commit**

```bash
cd /home/admin/lyuyuhuan/order_lyu
git add block_lo_arm_order_network/batch_readout/logs/quick_selector_validation.json docs/superpowers/specs/2026-05-30-quick-head-selector-findings.md
git commit -m "docs(quick-head-selector): Task 7 — full validation results + findings"
```

---

## Self-Review

**Spec coverage:**
- §3 C1–C4 cheap scores → Task 1. ✓
- §3.1 sign calibration (raw + calibrated, no preset sign) → Task 2 + driver step 2. ✓
- §3.2 threshold from step0 null, default 95th / fallback 90th, sensitivity → Task 5 (`null_quantile_threshold`) + driver step 3 + Task 7 findings. ✓
- §4 selection rule, best+/best-/pool, never |score| → Task 3. ✓
- §5 validation: Spearman → Task 4 + driver; Recall@1/3/5 pos&neg → Task 4 + driver; cross-run 5k → driver step 5; null step0/1k → driver (step0 thresholds + per-step on step0/1k) + Task 7; cost ratio → driver `cost_ratio`. ✓
- §5.1 desired criteria (not hard gate) → Task 7 reports descriptively. ✓
- §6.1 signatures `cheap_head_scores`, `select_heads` → match Tasks 1, 3 (select_heads adds explicit `sign` arg from calibration — consistent with §3.1, an intentional refinement of the spec stub `rank_score="C1"`). ✓
- §2 red lines (attention-only; L2R diagnostic-only; seeded extraction reused) → driver reuses `extract_per_head_and_heavy_A` (seeded) and never feeds L2R into anything but metrics. ✓
- §2 prerequisite (expensive JSONs) → Task 6 step 3 + Task 7 gating. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code; commands have expected output. ✓

**Type consistency:** `head_cheap_scores`→dict{C1..C4 float}; `cheap_head_scores`→dict{C1..C4 (L,H)}; `calibrate_sign`→(sign,array); `select_heads`→dict{best_positive,(l,h,sign); best_negative list; pool list}; `spearman_cheap_expensive`→float; `recall_at_k`→bool; `load_expensive_tau`→(L,H); `null_quantile_threshold`→float. Driver uses exactly these. `_batch_mean_B`/`_orders_from_graphs`/`extract_per_head_and_heavy_A` signatures match `per_head_order_scan.py`. ✓
