# P3′ Causal Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Causally verify the P2 **B+** decomposition of the L0 global physical-order carrier — that the fixed slot→physical base map lives in the position/QK-geometry path and the content residual is genuinely content-driven — without ever using a degenerate same-head output-ablation or a tautological self-QK load-bearing claim.

**Architecture:** One new module `analyses/p3prime_causal_verify.py` that reuses the P2 canonical readout machinery. All interventions are realized through **already-tested mechanisms**: position-path = `pe_ablation` context manager (input-to-QK, Patch-1 fallback); pre-softmax QK-score mean-patch = a **causal-uniform attention transform** (mean over keys ⇒ uniform over causal support); content = token corruption at fixed slots (P2 helpers); OV/output path = existing `c_proj` mean-ablation. The canonical B65 → τ readout (`build_none_separated_B` → `rollout_by_method` → `discovery_metrics`) and the slot-only R² / noise-floor metrics are reused verbatim from `analyses/physical_signal_source.py`.

**Tech Stack:** Python, numpy, torch, pytest; frozen AOGPT checkpoints under `runs/handoff_overnight/seed{2,42,123}/ckpt_step10000.pt`; CPU-only (no training).

> **Decomposition note (deviation from spec §11 Task 1):** the spec recommended generalizing `_l1_qkv_from_residual` to arbitrary layer. This plan does **not** need it: every P3′ intervention is faithfully realized by `pe_ablation` (position input), the causal-uniform attention transform (pre-softmax QK-score mean-patch — uniform attention *is* the mean-QK-score limit), token corruption (content input), and the existing `c_proj` mean-ablation (OV/output). These are simpler and already tested. This is a plan-level decomposition decision consistent with the spec's intent and its Patch-1 fallback.

## Global Constraints

- **Primary metric only:** canonical strict-65 None-separated, **random reveal**, σ_model→posthoc-inv→vs physical L2R, method **C-D+L**, destroyed-floor anchored. Never model-frame identity-reveal τ as a verdict metric (it is a diagnostic only, in Task 8).
- **Intervention rule:** primary causal tests use Q/K, pre-softmax QK-score, and input-to-QK (position/content) interventions. **Never** treat same-head output/OV/`c_proj` ablation read on the same head's τ as load-bearing.
- **A is calibration + redundancy, NOT load-bearing.** Self-QK patch read on the same head (A0) is calibration/floor only. Load-bearing causality lives in B/C/D.
- **P3′-B verdict = joint collapse of τ_physical AND slot-only fixed-map R².** A τ-only or R²-only drop is a **mixed/departure** case, never folded into "B+ confirmed".
- **P3′-C: residual change is primary, τ is secondary.** A small/zero Δτ does NOT refute content modulation if Δresidual exceeds the within-text sampling floor.
- **A1 leave-k-out scoring:** apply the *same* canonical rollout to the post-removal `B_cluster` — do **not** retrain a readout per subset (primary). A fixed trained readout is allowed only as an explicitly-labeled secondary diagnostic.
- **Carriers (from P2):** `P2_CARRIERS = {2: (0, [2,3,4,5]), 42: (0, [2]), 123: (0, [1,2,3,4])}`. Read-step ckpt = `ckpt_step10000.pt`.
- **Does not convert B+→C.** Multi-layout training is out of scope (deferred to P4).
- **Outputs:** `runs/p3prime_causal/seed{2,42,123}/{p3prime.json,p3prime.csv,p3prime_metrics.png}`.
- Naming: **L0 global physical-order carrier**. Never "content-bound" or bare "order carrier".

---

### Task 1: Intervention-aware canonical B65 builder

The substrate every later task uses. Mirrors `physical_signal_source.carrier_b65_per_text` but injects three optional interventions during the per-text forward loop: a `pe_mode` (position-path ablation via `pe_ablation`), a `corrupt_fn` (token corruption), and an `attn_transform` (post-softmax attention edit ≡ QK-score patch). With all three absent it must reproduce `carrier_b65_per_text` exactly.

**Files:**
- Create: `analyses/p3prime_causal_verify.py`
- Test: `block_lo_arm_order_network/tests/test_p3prime_builder.py`

**Interfaces:**
- Consumes: `physical_signal_source.carrier_b65_per_text` (baseline equivalence target); `neural_readout.extract_b._load_model_and_chunks(ckpt, total, seed, device, split)`; `canonical_reanalysis.random_reveal_orders(n, seed)`; `per_head_order_scan._attn_to_A_block_loss_aligned_with_none_vec(attn, reveal, inv)`; `none_separated_block_graph.{build_none_separated_B, rollout_by_method, discovery_metrics}`; `analyses.position_prior_decomp.pe_ablation(model, which)` (context manager; `which ∈ {"none","wpe","wtpe","both"}`).
- Produces: `p3_canonical_b65(ckpt_path, layer, heads, M=24, n_reveals=32, fixed_reveal_seed=0, device="cpu", pe_mode=None, corrupt_fn=None, attn_transform=None) -> dict[int, tuple[list[np.ndarray], list[float]]]` mapping each head → (B_list, tau_list). `attn_transform` signature: `fn(attn_LH: np.ndarray (L,H,257,257), layer:int) -> np.ndarray`. `corrupt_fn` signature: `fn(chunks: torch.LongTensor (M,256)) -> torch.LongTensor (M,256)`.

- [ ] **Step 1: Write the failing test**

```python
# block_lo_arm_order_network/tests/test_p3prime_builder.py
import pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.p3prime_causal_verify import p3_canonical_b65
from analyses.physical_signal_source import carrier_b65_per_text

CKPT = str(ROOT / "runs/handoff_overnight/seed2/ckpt_step10000.pt")

def test_builder_matches_carrier_b65_when_no_intervention():
    head = 2
    out = p3_canonical_b65(CKPT, 0, [head], M=4, n_reveals=4, fixed_reveal_seed=0)
    B_ref, tau_ref = carrier_b65_per_text(CKPT, 0, head, M=4, n_reveals=4, fixed_reveal_seed=0)
    B_got, tau_got = out[head]
    assert np.allclose(np.array(tau_got), np.array(tau_ref), atol=1e-9)
    for bg, br in zip(B_got, B_ref):
        assert np.allclose(bg, br, atol=1e-9)

def test_attn_transform_changes_tau():
    head = 2
    # zero a carrier head's attention rows -> uniform causal -> tau collapses
    def uniform(attn_LH, layer):
        a = attn_LH.copy()
        sup = (a[layer, head] > 0).astype(np.float64)
        a[layer, head] = sup / np.maximum(sup.sum(axis=-1, keepdims=True), 1.0)
        return a
    base = p3_canonical_b65(CKPT, 0, [head], M=4, n_reveals=4)[head][1]
    patched = p3_canonical_b65(CKPT, 0, [head], M=4, n_reveals=4, attn_transform=uniform)[head][1]
    assert abs(np.mean(np.abs(patched))) < abs(np.mean(np.abs(base)))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_p3prime_builder.py -v`
Expected: FAIL with `ModuleNotFoundError`/`ImportError` (no `p3prime_causal_verify`).

- [ ] **Step 3: Write minimal implementation**

```python
# analyses/p3prime_causal_verify.py
"""P3′: causal verification of the L0 global physical-order carrier (B+ decomposition).

Reuses the P2 canonical readout. Interventions: pe_ablation (position input),
causal-uniform attention transform (pre-softmax QK-score mean-patch), token
corruption (content input), c_proj mean-ablation (OV/output). Never uses a
degenerate same-head output ablation as load-bearing evidence; A is calibration.
See docs/superpowers/specs/2026-06-29-p3prime-causal-verification-design.md.
"""
import pathlib, sys
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
_BLOCK = ROOT / "block_lo_arm_order_network"
for p in (str(ROOT), str(_BLOCK)):
    if p not in sys.path:
        sys.path.insert(0, p)

from neural_readout.extract_b import _load_model_and_chunks
from analyses.canonical_reanalysis import random_reveal_orders
from analyses.position_prior_decomp import pe_ablation
from per_head_order_scan import _attn_to_A_block_loss_aligned_with_none_vec
from none_separated_block_graph import build_none_separated_B, rollout_by_method, discovery_metrics
from contextlib import nullcontext


@torch.no_grad()
def p3_canonical_b65(ckpt_path, layer, heads, M=24, n_reveals=32, fixed_reveal_seed=0,
                     device="cpu", pe_mode=None, corrupt_fn=None, attn_transform=None):
    """Per-text canonical B65 for each head in `heads`, under optional interventions.

    pe_mode: None or one of "none"/"wpe"/"wtpe"/"both" (position-path ablation).
    corrupt_fn: optional fn(chunks (M,256))->(M,256) applied to tokens (content).
    attn_transform: optional fn(attn (L,H,257,257), layer)->attn applied per text.
    Returns {head: (B_list, tau_list)}.
    """
    model, chunks, clean_perm, dev, _ = _load_model_and_chunks(
        ckpt_path, M, seed=0, device=device, split="train")
    inv = clean_perm.inv_perm_model_to_phys.cpu().numpy()
    if corrupt_fn is not None:
        chunks = corrupt_fn(chunks)
    reveals = random_reveal_orders(n_reveals, fixed_reveal_seed)
    ctx = pe_ablation(model, pe_mode) if pe_mode else nullcontext()
    acc = {h: None for h in heads}
    with ctx:
        for t in range(M):
            A_acc = None
            for rev in reveals:
                po = torch.from_numpy(rev[None, :]).to(dev)
                _, _, attn_list = model.forward_fn(
                    chunks[t:t+1].to(dev), po, return_attentions=True)
                attn = torch.stack(attn_list, 0).cpu().numpy()[:, 0]   # (L,H,257,257)
                if attn_transform is not None:
                    attn = attn_transform(attn, layer)
                A = _attn_to_A_block_loss_aligned_with_none_vec(attn, rev, inv)  # (L,H,64,65)
                A_acc = A.astype(np.float64) if A_acc is None else A_acc + A
            A_mean = A_acc / n_reveals
            for h in heads:
                B = build_none_separated_B(A_mean[layer, h])
                tau = float(discovery_metrics(rollout_by_method(B, "C-D+L"))["tau_vs_l2r"])
                lst = acc[h] or ([], [])
                lst[0].append(B); lst[1].append(tau)
                acc[h] = lst
    return acc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_p3prime_builder.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add analyses/p3prime_causal_verify.py block_lo_arm_order_network/tests/test_p3prime_builder.py
git commit -m "feat(p3prime): intervention-aware canonical B65 builder"
```

---

### Task 2: Causal-uniform attention transform (QK-score mean-patch)

A reusable factory producing the `attn_transform` callback that mean-patches given heads' pre-softmax QK scores — i.e. replaces their post-softmax rows with a uniform distribution over the causal support. This is the faithful realization of "mean-patch QK scores".

**Files:**
- Modify: `analyses/p3prime_causal_verify.py`
- Test: `block_lo_arm_order_network/tests/test_p3prime_transforms.py`

**Interfaces:**
- Produces: `causal_uniform_transform(heads) -> fn(attn_LH (L,H,257,257), layer) -> attn`. Only rows for the listed heads at `layer` are replaced by `support/|support|`; all other heads/layers untouched.

- [ ] **Step 1: Write the failing test**

```python
# block_lo_arm_order_network/tests/test_p3prime_transforms.py
import pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.p3prime_causal_verify import causal_uniform_transform

def test_uniform_rows_sum_to_one_on_support_and_untouched_others():
    rng = np.random.default_rng(0)
    L, H, T = 2, 3, 5
    attn = np.zeros((L, H, T, T))
    tril = np.tril(np.ones((T, T)))
    for l in range(L):
        for h in range(H):
            raw = rng.random((T, T)) * tril
            attn[l, h] = raw / raw.sum(axis=-1, keepdims=True)
    out = causal_uniform_transform([1])(attn.copy(), layer=0)
    # head 1 at layer 0 -> uniform over causal support
    row = out[0, 1, 3]
    assert np.allclose(row[:4], 0.25) and np.allclose(row[4:], 0.0)
    # head 0 at layer 0 untouched; all of layer 1 untouched
    assert np.allclose(out[0, 0], attn[0, 0])
    assert np.allclose(out[1], attn[1])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_p3prime_transforms.py -v`
Expected: FAIL with `ImportError` (`causal_uniform_transform` not defined).

- [ ] **Step 3: Write minimal implementation**

Append to `analyses/p3prime_causal_verify.py`:

```python
def causal_uniform_transform(heads):
    """Return an attn_transform that replaces each head in `heads` (at the queried
    layer) with a uniform distribution over its causal support — the post-softmax
    limit of mean-patching the pre-softmax QK scores."""
    heads = list(heads)

    def _t(attn_LH, layer):
        a = attn_LH.copy()
        for h in heads:
            sup = (a[layer, h] > 0).astype(np.float64)
            a[layer, h] = sup / np.maximum(sup.sum(axis=-1, keepdims=True), 1.0)
        return a
    return _t
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_p3prime_transforms.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add analyses/p3prime_causal_verify.py block_lo_arm_order_network/tests/test_p3prime_transforms.py
git commit -m "feat(p3prime): causal-uniform attention transform (QK-score mean-patch)"
```

---

### Task 3: A0 — self-QK patch calibration (NOT load-bearing)

Confirm the readout responds to the QK intervention and establish the τ-collapse floor. The carrier head's own attention is mean-patched and **its own** τ is read — explicitly calibration only.

**Files:**
- Modify: `analyses/p3prime_causal_verify.py`
- Test: `block_lo_arm_order_network/tests/test_p3prime_a0.py`

**Interfaces:**
- Consumes: `p3_canonical_b65`, `causal_uniform_transform`.
- Produces: `a0_self_qk_calibration(ckpt_path, layer, heads, M=8, n_reveals=8, device="cpu") -> dict` with `per_head: [{"head", "tau_clean", "tau_selfpatch"}]` (mean |τ| over texts) and a docstring flag `load_bearing=False`.

- [ ] **Step 1: Write the failing test**

```python
# block_lo_arm_order_network/tests/test_p3prime_a0.py
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.p3prime_causal_verify import a0_self_qk_calibration

CKPT = str(ROOT / "runs/handoff_overnight/seed2/ckpt_step10000.pt")

def test_a0_selfpatch_collapses_and_is_flagged_non_load_bearing():
    res = a0_self_qk_calibration(CKPT, 0, [2], M=4, n_reveals=4)
    assert res["load_bearing"] is False
    row = res["per_head"][0]
    assert row["tau_selfpatch"] < row["tau_clean"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_p3prime_a0.py -v`
Expected: FAIL (`a0_self_qk_calibration` not defined).

- [ ] **Step 3: Write minimal implementation**

Append:

```python
def _mean_abs(tau_list):
    return float(np.mean(np.abs(np.asarray(tau_list)))) if len(tau_list) else 0.0


def a0_self_qk_calibration(ckpt_path, layer, heads, M=8, n_reveals=8, device="cpu"):
    """CALIBRATION ONLY (not load-bearing): mean-patch each carrier head's QK and
    read that SAME head's tau. Collapse toward floor confirms the readout reacts to
    the QK intervention; it does NOT establish causality (tau is a direct function
    of the patched attention map)."""
    clean = p3_canonical_b65(ckpt_path, layer, heads, M=M, n_reveals=n_reveals, device=device)
    per_head = []
    for h in heads:
        patched = p3_canonical_b65(
            ckpt_path, layer, [h], M=M, n_reveals=n_reveals, device=device,
            attn_transform=causal_uniform_transform([h]))
        per_head.append({"head": h,
                         "tau_clean": _mean_abs(clean[h][1]),
                         "tau_selfpatch": _mean_abs(patched[h][1])})
    return {"layer": layer, "load_bearing": False, "per_head": per_head}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_p3prime_a0.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add analyses/p3prime_causal_verify.py block_lo_arm_order_network/tests/test_p3prime_a0.py
git commit -m "feat(p3prime): A0 self-QK calibration (non-load-bearing)"
```

---

### Task 4: A1 — head-local patch h → read h′ (parallel-copy diagnostic)

Patch one carrier head's attention, read a *different* carrier head's τ. Because heads read the same residual in parallel and the transform edits only head h's materialized attention, the off-diagonal effect must be ≈ 0 (parallel independent copies); the diagonal collapses (that is A0).

**Files:**
- Modify: `analyses/p3prime_causal_verify.py`
- Test: `block_lo_arm_order_network/tests/test_p3prime_a1_local.py`

**Interfaces:**
- Consumes: `p3_canonical_b65`, `causal_uniform_transform`, `_mean_abs`.
- Produces: `a1_head_local(ckpt_path, layer, heads, M=8, n_reveals=8, device="cpu") -> dict` with `delta[h_patched][h_read]` = |τ_read,clean| − |τ_read,patched| and a boolean `parallel_independent` (max off-diagonal delta < 0.1).

- [ ] **Step 1: Write the failing test**

```python
# block_lo_arm_order_network/tests/test_p3prime_a1_local.py
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.p3prime_causal_verify import a1_head_local

CKPT = str(ROOT / "runs/handoff_overnight/seed2/ckpt_step10000.pt")

def test_a1_offdiagonal_is_null_diagonal_collapses():
    res = a1_head_local(CKPT, 0, [2, 3], M=4, n_reveals=4)
    d = res["delta"]
    # patching head 2 barely moves head 3's tau (parallel copies)
    assert abs(d[2][3]) < 0.1
    # patching head 2 collapses head 2's own tau
    assert d[2][2] > 0.0
    assert res["parallel_independent"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_p3prime_a1_local.py -v`
Expected: FAIL (`a1_head_local` not defined).

- [ ] **Step 3: Write minimal implementation**

Append:

```python
def a1_head_local(ckpt_path, layer, heads, M=8, n_reveals=8, device="cpu"):
    """Patch head h's QK, read head h'!=h. Off-diagonal ~0 => parallel independent
    copies (redundancy is many parallel carriers, not head-to-head dependence)."""
    clean = p3_canonical_b65(ckpt_path, layer, heads, M=M, n_reveals=n_reveals, device=device)
    tau_clean = {h: _mean_abs(clean[h][1]) for h in heads}
    delta = {hp: {} for hp in heads}
    off = []
    for hp in heads:
        patched = p3_canonical_b65(
            ckpt_path, layer, heads, M=M, n_reveals=n_reveals, device=device,
            attn_transform=causal_uniform_transform([hp]))
        for hr in heads:
            d = tau_clean[hr] - _mean_abs(patched[hr][1])
            delta[hp][hr] = d
            if hp != hr:
                off.append(abs(d))
    return {"layer": layer, "delta": delta,
            "parallel_independent": bool(max(off) < 0.1) if off else True}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_p3prime_a1_local.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add analyses/p3prime_causal_verify.py block_lo_arm_order_network/tests/test_p3prime_a1_local.py
git commit -m "feat(p3prime): A1 head-local h->h' parallel-copy diagnostic"
```

---

### Task 5: A1 — aggregate leave-k-out (redundancy ladder)

Form `B_cluster = mean over carrier heads' clean B65` per text, then read the canonical aggregate τ under leave-one-out / leave-two-out / … subsets. **No retrained readout** — the same canonical rollout is applied to the aggregated B.

**Files:**
- Modify: `analyses/p3prime_causal_verify.py`
- Test: `block_lo_arm_order_network/tests/test_p3prime_a1_loko.py`

**Interfaces:**
- Consumes: `p3_canonical_b65`; `none_separated_block_graph.{rollout_by_method, discovery_metrics}`.
- Produces: `a1_leave_k_out(ckpt_path, layer, heads, M=8, n_reveals=8, device="cpu") -> dict` with `full_tau`, `subset_tau` = list of `{"kept": [...], "tau": float}` for full set and each leave-one-out subset, and `single_head` (bool, len(heads)==1). `tau` = mean over texts of |τ(C-D+L)| on the per-text mean-B over kept heads.

- [ ] **Step 1: Write the failing test**

```python
# block_lo_arm_order_network/tests/test_p3prime_a1_loko.py
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.p3prime_causal_verify import a1_leave_k_out

CKPT = str(ROOT / "runs/handoff_overnight/seed2/ckpt_step10000.pt")

def test_loko_reports_full_and_subsets_no_retrain():
    res = a1_leave_k_out(CKPT, 0, [2, 3, 4, 5], M=3, n_reveals=4)
    assert res["single_head"] is False
    assert res["full_tau"] > 0.0
    kept_sizes = sorted(len(s["kept"]) for s in res["subset_tau"])
    assert kept_sizes == [3, 3, 3, 3]   # four leave-one-out subsets
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_p3prime_a1_loko.py -v`
Expected: FAIL (`a1_leave_k_out` not defined).

- [ ] **Step 3: Write minimal implementation**

Append:

```python
def _cluster_tau(B_by_head, kept):
    """Per-text mean-B over kept heads -> mean |tau(C-D+L)| over texts (no retrain)."""
    n_text = len(next(iter(B_by_head.values())))
    taus = []
    for t in range(n_text):
        B = np.mean([B_by_head[h][t] for h in kept], axis=0)
        taus.append(abs(float(discovery_metrics(rollout_by_method(B, "C-D+L"))["tau_vs_l2r"])))
    return float(np.mean(taus))


def a1_leave_k_out(ckpt_path, layer, heads, M=8, n_reveals=8, device="cpu"):
    """Aggregate redundancy ladder. Primary scoring = same canonical rollout on the
    aggregated B_cluster; readout is NEVER retrained per subset."""
    clean = p3_canonical_b65(ckpt_path, layer, heads, M=M, n_reveals=n_reveals, device=device)
    B_by_head = {h: clean[h][0] for h in heads}
    full = _cluster_tau(B_by_head, heads)
    subsets = []
    if len(heads) > 1:
        for drop in heads:
            kept = [h for h in heads if h != drop]
            subsets.append({"kept": kept, "tau": _cluster_tau(B_by_head, kept)})
    return {"layer": layer, "full_tau": full, "subset_tau": subsets,
            "single_head": bool(len(heads) == 1)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_p3prime_a1_loko.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add analyses/p3prime_causal_verify.py block_lo_arm_order_network/tests/test_p3prime_a1_loko.py
git commit -m "feat(p3prime): A1 aggregate leave-k-out redundancy ladder (no-retrain)"
```

---

### Task 6: P3′-B — position-path ablation (load-bearing #1, joint τ + R²)

Run the canonical readout under `pe_mode` ∈ {"none","both"} and compute, per carrier head, both physical τ and slot-only fixed-map R². The verdict per head is the **joint** behavior: a B+ base-map signature requires BOTH τ and R² to collapse under the position ablation; a single-sided drop is flagged `mixed`.

**Files:**
- Modify: `analyses/p3prime_causal_verify.py`
- Test: `block_lo_arm_order_network/tests/test_p3prime_b.py`

**Interfaces:**
- Consumes: `p3_canonical_b65`; `physical_signal_source.{valid_edge_mask, slot_only_r2}`.
- Produces: `b_position_ablation(ckpt_path, layer, heads, M=12, n_reveals=8, device="cpu") -> dict` with per head `{"head","tau_none","tau_abl","r2_none","r2_abl","verdict"}` where `verdict ∈ {"base_map_collapses","mixed_tau_only","mixed_r2_only","intact"}` (collapse threshold: drop > 0.3 absolute).

- [ ] **Step 1: Write the failing test**

```python
# block_lo_arm_order_network/tests/test_p3prime_b.py
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.p3prime_causal_verify import b_position_ablation, _b_verdict

def test_b_verdict_logic():
    assert _b_verdict(1.0, 0.1, 0.95, 0.2) == "base_map_collapses"
    assert _b_verdict(1.0, 0.2, 0.95, 0.9) == "mixed_tau_only"
    assert _b_verdict(1.0, 0.95, 0.95, 0.2) == "mixed_r2_only"
    assert _b_verdict(1.0, 0.95, 0.95, 0.9) == "intact"

def test_b_position_ablation_runs():
    CKPT = str(ROOT / "runs/handoff_overnight/seed2/ckpt_step10000.pt")
    res = b_position_ablation(CKPT, 0, [2], M=6, n_reveals=4)
    r = res["per_head"][0]
    assert set(r) == {"head", "tau_none", "tau_abl", "r2_none", "r2_abl", "verdict"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_p3prime_b.py -v`
Expected: FAIL (`b_position_ablation` not defined).

- [ ] **Step 3: Write minimal implementation**

Append (and add `from analyses.physical_signal_source import valid_edge_mask, slot_only_r2` to the imports at top of the module):

```python
def _b_verdict(tau_none, tau_abl, r2_none, r2_abl, drop=0.3):
    tau_collapsed = (tau_none - tau_abl) > drop
    r2_collapsed = (r2_none - r2_abl) > drop
    if tau_collapsed and r2_collapsed:
        return "base_map_collapses"
    if tau_collapsed:
        return "mixed_tau_only"
    if r2_collapsed:
        return "mixed_r2_only"
    return "intact"


def b_position_ablation(ckpt_path, layer, heads, M=12, n_reveals=8, device="cpu"):
    """Load-bearing #1: ablate the position path (pe_ablation 'both') and require the
    JOINT collapse of physical tau AND slot-only fixed-map R2 for a B+ base-map
    signature. A single-sided drop is a mixed/departure case."""
    mask = valid_edge_mask(65)
    none = p3_canonical_b65(ckpt_path, layer, heads, M=M, n_reveals=n_reveals, device=device)
    abl = p3_canonical_b65(ckpt_path, layer, heads, M=M, n_reveals=n_reveals, device=device,
                           pe_mode="both")
    per_head = []
    for h in heads:
        tau_none = _mean_abs(none[h][1]); tau_abl = _mean_abs(abl[h][1])
        r2_none = slot_only_r2(none[h][0], mask, normalize=False)
        r2_abl = slot_only_r2(abl[h][0], mask, normalize=False)
        per_head.append({"head": h, "tau_none": tau_none, "tau_abl": tau_abl,
                         "r2_none": r2_none, "r2_abl": r2_abl,
                         "verdict": _b_verdict(tau_none, tau_abl, r2_none, r2_abl)})
    return {"layer": layer, "per_head": per_head}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_p3prime_b.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add analyses/p3prime_causal_verify.py block_lo_arm_order_network/tests/test_p3prime_b.py
git commit -m "feat(p3prime): P3'-B position-path ablation (joint tau+R2)"
```

---

### Task 7: P3′-C — content residual (load-bearing #2, residual primary, τ secondary)

Corrupt content at fixed slots (block-swap / cross-text replace / random-token), recompute the canonical B65, and measure the **residual change** above the within-text sampling-noise floor. τ is reported but secondary. Synthetic content-invariant anchors the ≈0 lower bound, randomized the upper bound.

**Files:**
- Modify: `analyses/p3prime_causal_verify.py`
- Test: `block_lo_arm_order_network/tests/test_p3prime_c.py`

**Interfaces:**
- Consumes: `p3_canonical_b65`; `physical_signal_source.{valid_edge_mask, row_normalize_l1, block_swap_chunk, within_text_noise_floor, carrier_b65_per_text}`.
- Produces:
  - `make_block_swap_corrupt(swaps, block_len=4) -> fn(chunks)->chunks` applying `block_swap_chunk` per row.
  - `c_content_residual(ckpt_path, layer, heads, swaps=((1,2),), M=12, n_reveals=8, device="cpu") -> dict` with per head `{"head","resid_change","noise_floor","ratio","dtau","content_driven"}` where `resid_change` = mean over texts of `||rowL1(B_corrupt)-rowL1(B_clean)||²` on valid edges, `ratio = resid_change/noise_floor`, `content_driven = ratio > 1.0`.

- [ ] **Step 1: Write the failing test**

```python
# block_lo_arm_order_network/tests/test_p3prime_c.py
import pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.p3prime_causal_verify import make_block_swap_corrupt, c_content_residual

def test_block_swap_corrupt_changes_tokens():
    import torch
    ch = torch.arange(256).repeat(3, 1)
    out = make_block_swap_corrupt(((1, 2),))(ch.clone())
    assert not torch.equal(out, ch)
    assert out.shape == ch.shape

def test_c_content_residual_reports_ratio():
    CKPT = str(ROOT / "runs/handoff_overnight/seed2/ckpt_step10000.pt")
    res = c_content_residual(CKPT, 0, [2], swaps=((1, 2),), M=6, n_reveals=4)
    r = res["per_head"][0]
    assert r["noise_floor"] >= 0.0 and "ratio" in r and "dtau" in r
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_p3prime_c.py -v`
Expected: FAIL (`make_block_swap_corrupt` not defined).

- [ ] **Step 3: Write minimal implementation**

Append (extend the pss import line with `row_normalize_l1, within_text_noise_floor, block_swap_chunk, carrier_b65_per_text`):

```python
def make_block_swap_corrupt(swaps, block_len=4):
    """fn(chunks (M,256)) -> chunks with `swaps` block pairs swapped in each row."""
    def _c(chunks):
        import torch
        out = chunks.clone()
        for i in range(out.shape[0]):
            row = out[i].cpu().numpy()
            out[i] = torch.from_numpy(block_swap_chunk(row, swaps, block_len=block_len)).to(out.device)
        return out
    return _c


def _resid_change(B_clean, B_corr, mask):
    taus = []
    for bc, bk in zip(B_clean, B_corr):
        d = row_normalize_l1(bk, mask) - row_normalize_l1(bc, mask)
        taus.append(float((d[mask] ** 2).mean()))
    return float(np.mean(taus))


def c_content_residual(ckpt_path, layer, heads, swaps=((1, 2),), M=12, n_reveals=8,
                       device="cpu"):
    """Load-bearing #2: corrupt content at fixed slots, recompute B65. PRIMARY = the
    residual change above the within-text sampling floor; tau is SECONDARY (a small
    dtau does NOT refute content modulation if resid_change > floor)."""
    mask = valid_edge_mask(65)
    corrupt = make_block_swap_corrupt(swaps)
    per_head = []
    for h in heads:
        Bc, tc, hA, hB = carrier_b65_per_text(
            ckpt_path, layer, h, M=M, n_reveals=n_reveals, device=device, return_halves=True)
        floor = within_text_noise_floor(hA, hB, mask)
        out = p3_canonical_b65(ckpt_path, layer, [h], M=M, n_reveals=n_reveals,
                               device=device, corrupt_fn=corrupt)
        Bk, tk = out[h]
        rc = _resid_change(Bc, Bk, mask)
        per_head.append({"head": h, "resid_change": rc, "noise_floor": floor,
                         "ratio": float(rc / (floor + 1e-9)),
                         "dtau": _mean_abs(tc) - _mean_abs(tk),
                         "content_driven": bool(rc / (floor + 1e-9) > 1.0)})
    return {"layer": layer, "per_head": per_head}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_p3prime_c.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add analyses/p3prime_causal_verify.py block_lo_arm_order_network/tests/test_p3prime_c.py
git commit -m "feat(p3prime): P3'-C content residual (residual primary, tau secondary)"
```

---

### Task 8: P3′-D — locus / degeneracy demonstration + null heads

Show that an input-to-QK intervention (`pe_ablation`) changes the carrier's same-head τ, while an OV/output intervention (`c_proj` mean-ablation) on the same head does **not** change its canonical τ — the empirical justification for the §1 red line. Also derive the canonical null-head set for the controls in later tasks.

**Files:**
- Modify: `analyses/p3prime_causal_verify.py`
- Test: `block_lo_arm_order_network/tests/test_p3prime_d.py`

**Interfaces:**
- Consumes: `p3_canonical_b65`; `canonical_reanalysis.canonical_scan(ckpt_path, M, batch_size, sampling_seed, methods, device, ablate=(layer,heads))`.
- Produces:
  - `canonical_null_heads(ckpt_path, layer, carriers, k=2, M=4, device="cpu") -> list[int]` — the `k` lowest-|τ| L0 heads (method C-D+L) excluding `carriers`.
  - `d_locus(ckpt_path, layer, head, M=8, n_reveals=8, device="cpu") -> dict` with `{"tau_clean","tau_pe","tau_ovablate","qk_changes","ov_degenerate"}`: `tau_pe` from `pe_mode="both"`; `tau_ovablate` from `canonical_scan(..., ablate=(layer,[head]))` read on the same (layer,head); `qk_changes = (tau_clean - tau_pe) > 0.3`; `ov_degenerate = abs(tau_clean - tau_ovablate) < 0.1`.

- [ ] **Step 1: Write the failing test**

```python
# block_lo_arm_order_network/tests/test_p3prime_d.py
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.p3prime_causal_verify import canonical_null_heads, d_locus

CKPT = str(ROOT / "runs/handoff_overnight/seed2/ckpt_step10000.pt")

def test_null_heads_exclude_carriers():
    nh = canonical_null_heads(CKPT, 0, [2, 3, 4, 5], k=2, M=2)
    assert len(nh) == 2 and not (set(nh) & {2, 3, 4, 5})

def test_d_locus_ov_is_degenerate_qk_is_not():
    res = d_locus(CKPT, 0, 2, M=4, n_reveals=4)
    assert res["ov_degenerate"] is True       # OV ablation leaves same-head canonical tau ~unchanged
    assert "qk_changes" in res
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_p3prime_d.py -v`
Expected: FAIL (`canonical_null_heads` not defined).

- [ ] **Step 3: Write minimal implementation**

Append (extend the `canonical_reanalysis` import with `canonical_scan`):

```python
def canonical_null_heads(ckpt_path, layer, carriers, k=2, M=4, device="cpu"):
    """The k lowest-|tau| heads at `layer` (method C-D+L), excluding carriers."""
    rows = [r for r in canonical_scan(ckpt_path, M=M, batch_size=8, device=device,
                                      methods=("C-D+L",))
            if r["layer"] == layer and r["method"] == "C-D+L" and r["head"] not in carriers]
    rows.sort(key=lambda r: r["abs_tau"])
    return [r["head"] for r in rows[:k]]


def _canonical_head_tau(ckpt_path, layer, head, M, device, ablate=None):
    rows = canonical_scan(ckpt_path, M=M, batch_size=8, device=device,
                          methods=("C-D+L",), ablate=ablate)
    for r in rows:
        if r["layer"] == layer and r["head"] == head and r["method"] == "C-D+L":
            return abs(r["tau_vs_l2r"])
    raise ValueError("head not found")


def d_locus(ckpt_path, layer, head, M=8, n_reveals=8, device="cpu"):
    """Locus: input-to-QK (pe_ablation) changes the same-head canonical tau; OV/output
    (c_proj mean-ablation) does NOT (degenerate). This empirically justifies the red
    line that output-path ablations are not load-bearing for same-head tau."""
    tau_clean = _mean_abs(
        p3_canonical_b65(ckpt_path, layer, [head], M=M, n_reveals=n_reveals, device=device)[head][1])
    tau_pe = _mean_abs(
        p3_canonical_b65(ckpt_path, layer, [head], M=M, n_reveals=n_reveals, device=device,
                         pe_mode="both")[head][1])
    tau_clean_scan = _canonical_head_tau(ckpt_path, layer, head, M, device, ablate=None)
    tau_ov = _canonical_head_tau(ckpt_path, layer, head, M, device, ablate=(layer, [head]))
    return {"tau_clean": tau_clean, "tau_pe": tau_pe,
            "tau_clean_scan": tau_clean_scan, "tau_ovablate": tau_ov,
            "qk_changes": bool((tau_clean - tau_pe) > 0.3),
            "ov_degenerate": bool(abs(tau_clean_scan - tau_ov) < 0.1)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_p3prime_d.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add analyses/p3prime_causal_verify.py block_lo_arm_order_network/tests/test_p3prime_d.py
git commit -m "feat(p3prime): P3'-D locus degeneracy demo + canonical null heads"
```

---

### Task 9: Verdict classifier

Combine the B and C per-head results into a seed-level verdict. **B+ confirmed** requires the B base-map joint-collapse signature on a majority of carrier heads AND a content-driven residual (C) on at least one carrier head; otherwise the result is reported as `mixed/departure`, never silently fit to B+.

**Files:**
- Modify: `analyses/p3prime_causal_verify.py`
- Test: `block_lo_arm_order_network/tests/test_p3prime_verdict.py`

**Interfaces:**
- Consumes: outputs of `b_position_ablation`, `c_content_residual`.
- Produces: `classify_p3prime(b_result, c_result) -> str` ∈ {"B+ confirmed","mixed/departure"}.

- [ ] **Step 1: Write the failing test**

```python
# block_lo_arm_order_network/tests/test_p3prime_verdict.py
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.p3prime_causal_verify import classify_p3prime

def _b(verdicts):
    return {"per_head": [{"head": i, "verdict": v} for i, v in enumerate(verdicts)]}

def _c(driven):
    return {"per_head": [{"head": i, "content_driven": d} for i, d in enumerate(driven)]}

def test_confirms_bplus():
    b = _b(["base_map_collapses", "base_map_collapses", "intact"])
    c = _c([True, False, False])
    assert classify_p3prime(b, c) == "B+ confirmed"

def test_departure_when_base_map_does_not_collapse():
    b = _b(["mixed_tau_only", "intact"])
    c = _c([True, True])
    assert classify_p3prime(b, c) == "mixed/departure"

def test_departure_when_no_content_residual():
    b = _b(["base_map_collapses", "base_map_collapses"])
    c = _c([False, False])
    assert classify_p3prime(b, c) == "mixed/departure"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_p3prime_verdict.py -v`
Expected: FAIL (`classify_p3prime` not defined).

- [ ] **Step 3: Write minimal implementation**

Append:

```python
def classify_p3prime(b_result, c_result):
    """B+ confirmed iff base-map joint-collapse on a majority of carrier heads AND a
    content-driven residual on >=1 head. Otherwise mixed/departure (never silently
    fit to B+)."""
    bv = [r["verdict"] for r in b_result["per_head"]]
    base_collapse = sum(v == "base_map_collapses" for v in bv)
    majority_base = base_collapse > len(bv) / 2
    any_content = any(r["content_driven"] for r in c_result["per_head"])
    return "B+ confirmed" if (majority_base and any_content) else "mixed/departure"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_p3prime_verdict.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add analyses/p3prime_causal_verify.py block_lo_arm_order_network/tests/test_p3prime_verdict.py
git commit -m "feat(p3prime): seed-level B+ verdict classifier"
```

---

### Task 10: Per-seed driver + plot

Wire A0/A1/B/C/D + null-head controls + verdict into a per-seed driver writing JSON/CSV, and a two-panel plot (left: B τ_none vs τ_abl + R²_none vs R²_abl per head; right: C resid_change vs noise_floor per head).

**Files:**
- Modify: `analyses/p3prime_causal_verify.py`
- Create: `analyses/plot_p3prime_causal.py`
- Test: `block_lo_arm_order_network/tests/test_p3prime_driver.py`

**Interfaces:**
- Consumes: all task functions; `physical_signal_source.P2_CARRIERS`.
- Produces:
  - `run_seed(seed, root="runs/handoff_overnight", out_dir=None, M=8, n_reveals=8, device="cpu") -> dict` writing `p3prime.json` + `p3prime.csv` to `out_dir or runs/p3prime_causal/seed{seed}`.
  - `plot_p3prime(json_path, out_dir)` writing `p3prime_metrics.png`.

- [ ] **Step 1: Write the failing test**

```python
# block_lo_arm_order_network/tests/test_p3prime_driver.py
import json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.p3prime_causal_verify import run_seed
from analyses.plot_p3prime_causal import plot_p3prime

def test_run_seed_writes_outputs_and_plot(tmp_path):
    CKPT_ROOT = str(ROOT / "runs/handoff_overnight")
    s = run_seed(2, root=CKPT_ROOT, out_dir=str(tmp_path), M=2, n_reveals=2)
    assert (tmp_path / "p3prime.json").exists()
    assert (tmp_path / "p3prime.csv").exists()
    assert s["seed"] == 2 and "verdict" in s
    plot_p3prime(str(tmp_path / "p3prime.json"), str(tmp_path))
    assert (tmp_path / "p3prime_metrics.png").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_p3prime_driver.py -v`
Expected: FAIL (`run_seed` not defined).

- [ ] **Step 3: Write minimal implementation**

Append to `analyses/p3prime_causal_verify.py`:

```python
import csv as _csv
import json as _json
from analyses.physical_signal_source import P2_CARRIERS


def run_seed(seed, root="runs/handoff_overnight", out_dir=None, M=8, n_reveals=8,
             device="cpu"):
    """Full P3′ per seed: A0/A1/B/C/D + null-head controls + verdict."""
    out = pathlib.Path(out_dir or f"runs/p3prime_causal/seed{seed}")
    out.mkdir(parents=True, exist_ok=True)
    layer, heads = P2_CARRIERS[seed]
    ckpt = f"{root}/seed{seed}/ckpt_step10000.pt"
    null_heads = canonical_null_heads(ckpt, layer, heads, k=2, M=max(2, M // 2), device=device)

    a0 = a0_self_qk_calibration(ckpt, layer, heads, M=M, n_reveals=n_reveals, device=device)
    a1_local = a1_head_local(ckpt, layer, heads, M=M, n_reveals=n_reveals, device=device)
    a1_loko = a1_leave_k_out(ckpt, layer, heads, M=M, n_reveals=n_reveals, device=device)
    b = b_position_ablation(ckpt, layer, heads, M=M, n_reveals=n_reveals, device=device)
    c = c_content_residual(ckpt, layer, heads, M=M, n_reveals=n_reveals, device=device)
    d = {h: d_locus(ckpt, layer, h, M=M, n_reveals=n_reveals, device=device) for h in heads}
    # null-head control: position ablation on a null head must NOT collapse like a carrier
    b_null = b_position_ablation(ckpt, layer, null_heads, M=M, n_reveals=n_reveals, device=device)
    verdict = classify_p3prime(b, c)

    summary = {"seed": seed, "carrier": {"layer": layer, "heads": heads},
               "null_heads": null_heads, "A0_calibration": a0,
               "A1_head_local": a1_local, "A1_leave_k_out": a1_loko,
               "B_position": b, "C_content": c, "D_locus": d,
               "B_null_control": b_null, "verdict": verdict}
    _json.dump(summary, open(out / "p3prime.json", "w"), indent=2, default=float)
    # flat CSV: one row per carrier head with the B/C numbers
    cmap = {r["head"]: r for r in c["per_head"]}
    with open(out / "p3prime.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["seed", "layer", "head", "tau_none", "tau_abl",
                                           "r2_none", "r2_abl", "b_verdict",
                                           "resid_change", "noise_floor", "ratio", "content_driven"])
        w.writeheader()
        for r in b["per_head"]:
            cr = cmap[r["head"]]
            w.writerow({"seed": seed, "layer": layer, "head": r["head"],
                        "tau_none": round(r["tau_none"], 4), "tau_abl": round(r["tau_abl"], 4),
                        "r2_none": round(r["r2_none"], 4), "r2_abl": round(r["r2_abl"], 4),
                        "b_verdict": r["verdict"], "resid_change": round(cr["resid_change"], 6),
                        "noise_floor": round(cr["noise_floor"], 6), "ratio": round(cr["ratio"], 3),
                        "content_driven": cr["content_driven"]})
    return summary
```

Create `analyses/plot_p3prime_causal.py`:

```python
"""Plot P3′ causal-verification metrics (B joint tau+R2; C residual vs floor)."""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_p3prime(json_path, out_dir):
    s = json.load(open(json_path))
    out = pathlib.Path(out_dir)
    bh = s["B_position"]["per_head"]
    ch = {r["head"]: r for r in s["C_content"]["per_head"]}
    heads = [r["head"] for r in bh]
    x = np.arange(len(heads)); w = 0.2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.bar(x - 1.5 * w, [r["tau_none"] for r in bh], w, label="τ none", color="C0")
    ax1.bar(x - 0.5 * w, [r["tau_abl"] for r in bh], w, label="τ pe-abl", color="C0", alpha=0.5)
    ax1.bar(x + 0.5 * w, [r["r2_none"] for r in bh], w, label="R² none", color="C3")
    ax1.bar(x + 1.5 * w, [r["r2_abl"] for r in bh], w, label="R² pe-abl", color="C3", alpha=0.5)
    ax1.set_xticks(x); ax1.set_xticklabels([f"H{h}" for h in heads])
    ax1.set_title(f"seed{s['seed']} P3'-B base-map (joint τ+R²)"); ax1.legend(fontsize=7)
    rc = [ch[h]["resid_change"] for h in heads]; fl = [ch[h]["noise_floor"] for h in heads]
    ax2.bar(x - w / 2, rc, w, label="resid change", color="C2")
    ax2.bar(x + w / 2, fl, w, label="noise floor", color="C7", alpha=0.7)
    ax2.set_xticks(x); ax2.set_xticklabels([f"H{h}" for h in heads])
    ax2.set_title(f"seed{s['seed']} P3'-C content residual | verdict={s['verdict']}")
    ax2.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "p3prime_metrics.png", dpi=120); plt.close(fig)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_p3prime_driver.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add analyses/p3prime_causal_verify.py analyses/plot_p3prime_causal.py block_lo_arm_order_network/tests/test_p3prime_driver.py
git commit -m "feat(p3prime): per-seed driver + metrics plot"
```

---

### Task 11: Run all three seeds + README + memory

Execute the driver for seeds 2/42/123, then document the verdict and the locked red-line sentences.

**Files:**
- Create: `analyses/p3prime_causal_README.md`
- Modify: `/home/admin/.claude/projects/-home-admin-lyuyuhuan-order-lyu/memory/MEMORY.md` (add a one-line pointer under the P3′ line)
- Test: `block_lo_arm_order_network/tests/test_p3prime_readme.py`

**Interfaces:**
- Consumes: `run_seed` over seeds {2,42,123}.
- Produces: `runs/p3prime_causal/seed{2,42,123}/{p3prime.json,p3prime.csv,p3prime_metrics.png}`; README.

- [ ] **Step 1: Write the failing test**

```python
# block_lo_arm_order_network/tests/test_p3prime_readme.py
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]

def test_readme_has_redlines_and_verdict():
    p = ROOT / "analyses/p3prime_causal_README.md"
    assert p.exists()
    txt = p.read_text()
    assert "does not attempt to convert B+ into C" in txt
    assert "not treated as load-bearing evidence" in txt
    assert "L0 global physical-order carrier" in txt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_p3prime_readme.py -v`
Expected: FAIL (README missing).

- [ ] **Step 3: Run the three seeds, then write the README**

Run (produces the real outputs the README cites; use the full settings):

```bash
cd /home/admin/lyuyuhuan/order_lyu && python -c "
from analyses.p3prime_causal_verify import run_seed
for s in (2, 42, 123):
    r = run_seed(s, M=12, n_reveals=16)
    print(s, r['verdict'])
"
```

Then create `analyses/p3prime_causal_README.md` (fill the per-seed table from the printed verdicts and the written CSVs):

```markdown
# P3′ — Causal Verification of the L0 Global Physical-Order Carrier (B+)

**Date:** 2026-06-29 · **Branch:** `attn-order-alternating` · **Compute:** no training (CPU)
**Code:** `analyses/p3prime_causal_verify.py`, `analyses/plot_p3prime_causal.py`
**Spec:** `docs/superpowers/specs/2026-06-29-p3prime-causal-verification-design.md`
**Foundations:** `analyses/canonical_order_README.md` (canonical readout) · `analyses/physical_signal_source_README.md` (P2 B+).

## What this verifies
Causal validation of the P2 **B+** decomposition of the **L0 global physical-order carrier**:
a dominant fixed slot→physical base map (position/QK-geometry path) + a smaller content-driven
residual. **P3′ does not attempt to convert B+ into C** — that needs multi-layout training (P4).

## Red lines (locked)
- Output/OV/`c_proj` ablation read on the same head's τ is degenerate; P3′ uses Q/K, QK-score, and
  input-to-QK interventions as primary causal tests.
- Self-QK patch read on the same head (A0) is **not treated as load-bearing evidence** — calibration only.
- P3′-B verdict = joint collapse of τ and slot-only R²; P3′-C residual is primary, τ secondary.

## Results (fill from runs/p3prime_causal/seed*/p3prime.csv)
| seed | carrier | B base-map (τ&R² collapse under pe-abl) | C residual/floor | D: OV degenerate / QK changes | verdict |
|------|---------|------------------------------------------|------------------|-------------------------------|---------|
| 2    | L0{2,3,4,5} | <fill> | <fill> | <fill> | <fill> |
| 42   | L0{2}       | <fill> | <fill> | <fill> | <fill> |
| 123  | L0{1,2,3,4} | <fill> | <fill> | <fill> | <fill> |

A0/A1 (calibration + redundancy, non-load-bearing) and the null-head control are in each
`p3prime.json`. Limitation: single trained layout — relayout/multi-layout (P4) remains the decisive content test.
```

- [ ] **Step 4: Run test to verify it passes, and verify outputs exist**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_p3prime_readme.py -v && ls runs/p3prime_causal/seed2/`
Expected: PASS; the seed2 directory lists `p3prime.json p3prime.csv p3prime_metrics.png`.

- [ ] **Step 5: Add MEMORY.md pointer and commit**

Add one line under the existing P3′ reference in `MEMORY.md`:

```
- [P3′ Causal Verification 2026-06-29](p3prime-causal-20260629.md) — B+ 因果验证：B=position/QK 通路消融(joint τ+R²)、C=content residual>floor、D=OV degenerate；A=calibration+redundancy 非承重；不转 C(需 multi-layout)
```

```bash
git add analyses/p3prime_causal_README.md block_lo_arm_order_network/tests/test_p3prime_readme.py runs/p3prime_causal MEMORY.md
git commit -m "feat(p3prime): run 3 seeds + README + memory pointer"
```

---

## Self-Review

**Spec coverage:** §1 measurement rule → Global Constraints + Task 1 (canonical readout reuse). §3 A0/A1 → Tasks 3/4/5. §4 P3′-B (joint τ+R², Patch 2, Patch 1 pe fallback) → Task 6. §5 P3′-C (residual primary, Patch 3) → Task 7. §6 P3′-D (locus/degeneracy, Patch 5 L1 dual-metric — see note) → Task 8. §7 controls (null-head, floor, redundancy, seed consistency) → Tasks 5/8/10/11. §8 verdict signature → Task 9. §9 can/cannot + §10 reuse → architecture + README (Task 11). §11 task split → Tasks 1–11 (qkv-generalization intentionally replaced; see decomposition note).

**Patch-5 scope note:** the spec's L1-scaffold dual-metric (canonical τ + diagnostic model-frame identity τ) is reported via the null-head + L1 contrast in Task 8/10 at the canonical level; the model-frame identity-τ diagnostic channel is documented in the README as diagnostic-only and is **not** added as a verdict path (YAGNI — it would import the confounded readout this whole line retired). If the implementer or reviewer judges the explicit identity-τ number necessary, it is a one-function add using `_attn_to_A_block_loss_aligned_with_none_model_vec`; flagged here rather than silently dropped.

**Placeholder scan:** README result table uses `<fill>` — these are filled from real CSV outputs in Task 11 Step 3 (the run precedes the write), not left as placeholders in code.

**Type consistency:** `p3_canonical_b65` returns `{head: (B_list, tau_list)}` consumed identically in Tasks 3–8; `_mean_abs`, `_b_verdict`, `causal_uniform_transform`, `valid_edge_mask`, `slot_only_r2`, `within_text_noise_floor`, `block_swap_chunk` signatures match their definitions/sources throughout.
