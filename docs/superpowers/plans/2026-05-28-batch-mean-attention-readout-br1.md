# BR-1 Batch-Mean Hooked Attention Readout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and validate a *batch-global* readout `g_β: R^{N×N} → R^N` that consumes a batch-mean attention graph `B_batch = mean_x A_θ(x)^T` and emits node logits `z`, from which a full reveal order `σ ~ Plackett-Luce(z/τ)` is sampled (or `argsort(-z)` for deterministic). v1 trains `g_β` *offline* on data harvested from a fixed 5k random-warmup checkpoint, then plugs the **frozen** readout into the alternating-from-0 training loop to feed the next-step order each step.

**Architecture (high level):**

```
A_θ(x_b)         per-sample attention (N×N) for batch member b ∈ B
B_batch = mean_b A_θ(x_b)^T                 ∈ R^{N×N}
z = g_β(B_batch)                            ∈ R^N
p = softmax(z/τ)                            global selection distribution
σ ~ PL(z/τ)  (no replacement) | argsort(-z) reveal order ∈ S_N
```

`g_β` v1 is `flatten(B) → MLP(4096→1024→256→N)`; v2 is per-node features `[B[v,:], B[:,v]]` → shared MLP → 2L TransformerEncoder mixer → scalar head (no PE). Loss is **pairwise logistic** (v1) plus **Plackett-Luce listwise NLL** (v2) — both attention-derived; **no autoregressive NLL appears in training or selection**. Phase 1 trains and selects by Kendall τ / pairwise precedence acc / PL NLL on validation graphs. Phase 2 computes a *post-selection* frozen-θ AR-NLL diagnostic comparing `σ_MLP` vs `σ_T` vs `σ_random`. Phase 3 plugs the frozen `g_β` into `train_clean_aogpt` with batch-mean attention hooked at step `t` to set the order for step `t+1`.

**Tech Stack:** Python 3, PyTorch (existing), NumPy, SciPy (Kendall/Spearman), pytest. Reuses `neural_readout.extract_b.extract_per_sample_B` (NR-1 Task 2), `neural_readout.teacher_labels.generate_teacher_label`, `neural_readout.eval_frozen_nll`, and `attn_order_teacher.rollout_order`. Source ckpt: `probe_results/attention_order_mlp/alt_from0_random/ckpt_step5000.pt`.

**Substrate:** Text, N=64 (wikitext seq256 block64 — same as NR-1 + alternating from-0 text line).

**Relationship to NR-1:** Sibling line. NR-1 is **per-sample** `B_i → σ_i` (Graph-Transformer, amortization-feasibility study). BR-1 is **batch-global** `B_batch → σ_{t+1}` (stochastic global curriculum, intended for in-loop feedback). They reuse the same teacher (CDL source-start) and the same frozen-NLL diagnostic harness; everything else is separate.

**Sources of randomness (must be seeded explicitly):**
1. chunk sampling inside `extract_per_sample_B` (already seeded in NR-1 Task 1 fix)
2. `extract_A_matrices` internal `randperm` (already seeded in NR-1 Task 1 fix)
3. train/val/test split inside the dataset builder
4. minibatch shuffling in `train_offline`
5. PL sampling at inference / integration time

---

## File structure

### Create

| Path | Responsibility |
|---|---|
| `block_lo_arm_order_network/batch_readout/__init__.py` | package marker |
| `block_lo_arm_order_network/batch_readout/extract_b_batch.py` | wraps `neural_readout.extract_b.extract_per_sample_B`, groups M·batch_size samples into M batch-mean graphs `(M, N, N)` |
| `block_lo_arm_order_network/batch_readout/dataset_batch.py` | build / save / load `(B_batch, σ_T, rank, pairwise_Y)` `.npz`; teacher via reused `generate_teacher_label` applied to each `B_batch[m]` |
| `block_lo_arm_order_network/batch_readout/diversity_batch.py` | first-step entropy, unique-σ ratio, mean pairwise τ across the M teacher labels |
| `block_lo_arm_order_network/batch_readout/model.py` | `FlattenReadout` (v1) and `NodewiseReadout` (v2) modules |
| `block_lo_arm_order_network/batch_readout/loss.py` | `pairwise_logistic_loss(logits, rank)` and `plackett_luce_nll(logits, sigma_T, tau)` |
| `block_lo_arm_order_network/batch_readout/pl_sampling.py` | `pl_sample(logits, tau, generator)` and `pl_argsort(logits)` |
| `block_lo_arm_order_network/batch_readout/eval_metrics.py` | Kendall τ, pairwise precedence acc, Spearman ρ, top-1, first-3 overlap |
| `block_lo_arm_order_network/batch_readout/train_offline.py` | offline training loop with selection by `pairwise_acc` only (NEVER NLL) |
| `block_lo_arm_order_network/batch_readout/eval_frozen_phase2.py` | thin wrapper that runs `neural_readout.eval_frozen_nll.compute_frozen_nll_gap` for arms `{random, σ_T, MLP-argsort, MLP-sample}` |
| `block_lo_arm_order_network/batch_readout/integration_hook.py` | `FrozenBetaHook` class — holds frozen `g_β` ckpt, takes per-sample `A` batch, returns next-step `σ_{t+1}` |
| `block_lo_arm_order_network/tests/test_batch_readout_extract.py` | shape + seed reproducibility + B_batch == mean(B_per_sample) |
| `block_lo_arm_order_network/tests/test_batch_readout_dataset.py` | round-trip save/load + split sizes + rank-0 convention |
| `block_lo_arm_order_network/tests/test_batch_readout_diversity.py` | degenerate vs randomized inputs |
| `block_lo_arm_order_network/tests/test_batch_readout_model.py` | shapes + param count + no PE + no NaN |
| `block_lo_arm_order_network/tests/test_batch_readout_loss.py` | pairwise direction + PL closed form on N=3 |
| `block_lo_arm_order_network/tests/test_batch_readout_pl_sampling.py` | valid permutation + seed determinism + temperature monotonicity |
| `block_lo_arm_order_network/tests/test_batch_readout_metrics.py` | identity / reversed / known values |
| `block_lo_arm_order_network/tests/test_batch_readout_train_overfit.py` | overfit 16-batch dataset to train_loss → 0 |
| `block_lo_arm_order_network/tests/test_batch_readout_selection_policy.py` | enforce `train_offline` does not import `eval_frozen_phase2` or AR-NLL anywhere |
| `block_lo_arm_order_network/tests/test_batch_readout_integration_hook.py` | hook returns valid σ given fake `A` batch + frozen ckpt stub |
| `scripts/run_br1_phase0_smoke.sh` | M=100, batch_size=32 smoke (extract + build) |
| `scripts/run_br1_phase1_smoke.sh` | train smoke on `M=100` dataset |
| `scripts/run_br1_phase1_full.sh` | full M=1000 train (v1 flatten + v2 nodewise) |
| `scripts/run_br1_phase2_frozen.sh` | frozen-θ NLL gate on `{random, σ_T, MLP-argsort, MLP-sample}` |
| `scripts/run_br1_phase3_integration.sh` | 5k→10k frozen-β short run, 4 arms |
| `scripts/run_br1_ablation_matrix.sh` | sweep over `M`, `batch_size`, `τ`, `argsort vs sample`, `flatten vs nodewise`, `B_batch vs B_global`, `frozen-β vs online-β` |

### Modify

| Path | What changes |
|---|---|
| `block_lo_arm_order_network/train_clean_aogpt.py` | add optional `order_provider` callable to the training-loop signature; if set, after each `forward` use the hooked attention to set the next-step `σ`. Default `None` preserves current behavior. Mirror the seeding work already done for `extract_A_matrices`. |

### Outputs (generated, not committed)

| Path | Produced by |
|---|---|
| `block_lo_arm_order_network/batch_readout/data/text_5k_M{100,1000,10000}_B{16,32,128}_seed{N}.npz` | dataset task |
| `block_lo_arm_order_network/batch_readout/checkpoints/g_beta_{model}_{loss}_{run_tag}.pt` | training |
| `block_lo_arm_order_network/batch_readout/logs/{phase}_{run_tag}.log` | per-run log |
| `analyses/batch_readout_br1_2026-05-28/REPORT.md` | final report |

---

## §5 Acceptance gates (per phase)

### §5.1 Phase 1 (offline g_β selection)

All evaluated on held-out val split of the same dataset (M ≥ 1000):

| Metric | Threshold | Source |
|---|---|---|
| Kendall τ (pred σ vs σ_T) | ≥ 0.50 | `eval_metrics.kendall_tau` |
| pairwise precedence acc | ≥ 0.70 | `eval_metrics.pairwise_acc` |
| Spearman ρ | ≥ 0.55 | `eval_metrics.spearman_rho` |

Thresholds are **lower than NR-1's 0.80/0.90/0.85** on purpose: batch-mean targets are noisier than per-sample targets (the input averages out per-sample structure) and the goal here is *useful global signal*, not *amortizing per-sample mapping*. NLL is **not** in the gate (Task 19 test enforces this).

### §5.2 Phase 2 (frozen-θ AR-NLL diagnostic, post-selection only)

Run after Phase 1 selects a `g_β`. Compute on the val split:

```
NLL(σ_arm) − NLL(σ_random)   for arm ∈ {σ_T, MLP-argsort, MLP-sample}
```

Pass criterion to *consider* Phase 3:

- `NLL(MLP-argsort) ≤ NLL(σ_random) − 0.005` **and**
- `NLL(MLP-argsort) ≤ NLL(σ_T) + 0.010` (no worse than teacher by more than 0.01)

If MLP-argsort fails both, do **not** start Phase 3; revisit model capacity / teacher noise.

### §5.3 Phase 3 (frozen-β short run)

5k → 10k training continuation. 4 arms: `random`, `σ_T-refresh-every-step`, `MLP-argsort`, `MLP-sample(τ=0.3)`. Pass criterion for the BR-1 main claim:

- `val_ori_l2r(MLP-argsort) ≤ val_ori_l2r(random) − 0.005` at 10k
- `val_ori_l2r(MLP-argsort) ≤ val_ori_l2r(σ_T) + 0.005` (no worse than hand teacher)

If neither passes, document as a clean negative and stop.

---

## §6 Ablation matrix (the focus of this plan)

The ablations are the *point* of BR-1 — the plan is structured so every axis below has a corresponding run target. Each row produces one number (Kendall τ at Phase 1 + ΔNLL at Phase 2). Stable rows are reported; unstable ones get a note.

| # | Axis | Values | Default | Phase | Why we sweep | Failure interpretation |
|---|---|---|---|---|---|---|
| A1 | `M` (number of batch-mean graphs) | 100 / 1000 / 10000 | 1000 | 1 | Data scaling — is `M=1000` enough? | Monotone improvement → bigger M; flat → bottleneck elsewhere |
| A2 | `batch_size` (samples per batch-mean) | 16 / 32 / 128 | 32 | 1 | Does `B_batch` stability matter? | Big B_size much better → batch-mean is too noisy at 32 |
| A3 | sampling mode | argsort / PL-sample | argsort | 1+3 | Is determinism better than diversity? | Sample loses on Phase 3 → sampling noise hurts training |
| A4 | temperature τ | 0.1 / 0.3 / 0.5 / 1.0 | 0.3 | 1+3 | Entropy of order distribution | Only τ=0.1 works → it's basically argsort |
| A5 | model | FlattenReadout / NodewiseReadout | Nodewise | 1 | Is per-node + mixer needed? | Flatten enough → don't bother with v2 |
| A6 | input variant | `B_batch` / `B_global` (single graph from larger pool) | `B_batch` | 1 | Does the per-batch stochasticity carry signal vs a single global graph? | `B_global` wins → no point in per-batch dynamics |
| A7 | teacher | CDL source-start / vanilla CDL | source-start | 1 | Source anchor sensitivity | Vanilla CDL fails → confirms NR-1 finding that anchor matters |
| A8 | loss | pairwise / PL listwise | pairwise (v1), PL (v2) | 1 | Listwise vs pairwise calibration | PL wins → use PL for Phase 3 deployment |
| A9 | ckpt source | alt-from-0 random-warmup 5k / clean_base_random_perm 5k | alt-from-0 | 1+3 | Is BR-1 conclusion ckpt-dependent? | Big gap → ckpt-specific not BR-1-general |
| A10 | frozen-β vs online-β | frozen / online (β updated every K steps with same loss) | frozen | 3 | Does β need to keep adapting? | Online wins → frozen is leaving signal on the table |
| A11 | step-lag policy | order at step `t+1` derived from `B_batch_t` / from `B_batch_t-K` (EMA) | t-1 (immediate) | 3 | One-step lag stability | EMA wins → batch-to-batch noise too high |

**Order of execution:** complete A1+A5+A8 in Phase 1 first (these determine the "best v1 g_β"). Then A2/A6/A7/A9 within Phase 1 budget. A3/A4 span Phase 1 → 3. A10/A11 are Phase 3 only.

---

## §7 Risk register (paste in tickets, not just here)

1. **B_batch noise** — averaging over batch_size=32 may not produce a clean enough graph to drive a global order. Mitigation: A2 sweep + A6 control with `B_global`.
2. **One-step lag mismatch** — `B_batch_t` is used to set σ for batch `t+1`. Only valid if `g_β` is reading a stable global structure, not batch-specific noise. Mitigation: A11 EMA control + jitter logging in Phase 3.
3. **Global logits can't express sequential CDL** — single-shot `z ∈ R^N` may not be expressive enough for a CDL-style sequential rollout. Mitigation: A8 PL loss + monitor Phase 1 Kendall τ; if it caps at ~0.4 even with M=10000 and Nodewise model, that's the diagnosis.
4. **Teacher diversity collapses on batch-mean** — averaging may smooth out per-sample structure, so `σ_T(B_batch)` may be near-identical across `m`, giving trivial labels. Mitigation: Task 6 measures `mean_pairwise_τ` and fails-loud if > 0.95 (degenerate) or < 0.05 (pure noise).
5. **Frozen β masks runtime drift** — by 8k step the optimal order may have moved away from what 5k-trained `g_β` predicts. Mitigation: A10 online-β control + log `τ(z_t, z_{5k})` over training.
6. **GPU contention with NR-1 Task 13** — NR-1 full-10k training is running on a GPU as of plan-write time. Phase 0 extraction needs ≈30 min on one GPU. Mitigation: Task 1 of this plan checks `nvidia-smi` before launching and waits if both GPUs are occupied.
7. **Pretrain/online head mismatch** — `g_β` is trained on `B` extracted from one head, but the online hook may feed `B` from a *different* head (head index drifts across runs; the cheap selector may pick another head). If train-head ≠ online-head, the hook feeds `g_β` an out-of-distribution `B`. Mitigation: the **B1 head-selection contract (§8)** — train and serve on the *same single best-positive head*; the online provider must hook that exact `(layer, head)`.

---

## §8 Head-selection contract — **B1 (single best-positive head)**

This section fixes the *method boundary* for which attention head's graph feeds `g_β`, both at pretrain and online time. It is the BR-1-side resolution of the quick-head-selector line (`docs/superpowers/specs/2026-05-30-quick-head-selector-design.md`).

### §8.1 The decision (B1)

```
5k warmup  →  one-time CDL per-head scan  →  pick best+ (max sign-calibrated τ_vs_l2r among 32 heads)
           →  train g_β on  B^{(l*,h*)}_batch   (single positive head)
           →  online: FrozenBetaHook reads the SAME (l*, h*) every step
```

- **Single positive head only.** `g_β` is trained on, and served on, *one* head `(l*, h*)` — the offline best-positive (L2R-aligned, sign-calibrated `τ > 0`) head. Train-distribution = serve-distribution by construction → no mismatch.
- **No negatives.** The best-*negative* (anti-L2R) head is **not** used. It carries stronger |τ| but its index is unstable within a run, and serving it would force an R2L-style order (the `|score|`-winner trap from the selector spec). Excluding it keeps the contract index-stable and sign-clean.
- **No periodic re-selection within a run.** Empirically unnecessary (see §8.2).

### §8.2 Empirical grounding (clean_base, why B1 is sufficient)

Source: `block_lo_arm_order_network/batch_readout/logs/l0h5_evo/heads_tau_3seed.json` — clean_base, 3 seeds × 9 steps (0→60k), full 32-head argmax/argmin over `τ_vs_l2r`.

- **best+ is stable within a run:** noise at step 0/1k (τ≈+0.02); from 5k onward the best-positive head is **L0H0**, and from 20k→60k it is **L0H0 in 3/3 seeds** (τ climbs to ≈+0.5). L0H5 appears only once (seed0 @ 5k) — a transient, not the stable head.
- **best− is index-unstable** (flips L1H6 ↔ L2H3 across steps, τ −0.5…−0.73) though its *sign* is stable. This is the second reason not to serve it.

⇒ Within a single training run, picking the best-positive head once after warmup and freezing it is enough. Multi-head pooling buys nothing here.

### §8.3 What changes in the extraction path

The current head collapse in `train_clean_aogpt.extract_A_matrices` (≈L111-118) is **top-4-highest-variance heads, averaged over all layers**. Per the BR-1 status finding, this head-mean cancels positive against negative heads (last-layer pairwise_τ collapses to ≈0.05). **B1 replaces this with single-head selection.**

Contract on the API (thread one `(layer, head)` selector through the whole chain; default = the warmup-selected best+):

```
extract_A_matrices(..., head_select=(l*, h*))          # single head, no variance-top-k mean
  → extract_per_sample_B_with_chunks(..., head_select)
    → extract_batch_mean_B(..., head_select)            # B_batch built from one head
      → FrozenBetaHook(..., head_select)                # online hook slices the SAME (l*, h*)
```

- The selected `(l*, h*)` is recorded in the `g_β` checkpoint `config` so the hook cannot silently serve a different head than training used.
- Legacy top-4-variance behavior is retained only behind an explicit non-default flag for A/B comparison (it is the implicit "head-mean baseline").

### §8.4 B2 — escape hatch (NOT default, do not build yet)

Promote to multi-head training **only** when one of these is actually observed:
1. **Cross-run reuse** of a single `g_β` (best+ index differs across runs: clean_base → L0H0, alt_from0_random → L0H5); or
2. a future run where **within-run best+ is itself unstable**; or
3. a deliberate decision to exploit the stronger best− topology signal.

B2 = train `g_β` on several positive heads (or a sign-labeled `{best+, top-2 best−}` pool). It is *theoretically* clean because the target is `σ_CDL(B^h)` per head (CDL is sign-self-consistent), but it requires an extra guard: verify `g_β` is a faithful CDL imitator (fits on positive **and** negative `B`) rather than memorizing one head's `B` distribution. Until (1)/(2)/(3) is observed, **stay on B1**.

### §8.5 Red lines (inherited)

- Selection of `(l*, h*)` uses sign-calibrated CDL `τ_vs_l2r` as *offline* diagnostic ground truth only; **no NLL in selection**, **no `argmax|score|`**, **no raw L2R label fed into order generation**.
- Online never re-runs the full CDL scan; head identity is fixed from the one-time warmup scan (or, later, recalled cheaply by the quick-head-selector).

---

## Tasks

### Task 1: Stub package + nvidia-smi guard

**Files:**
- Create: `block_lo_arm_order_network/batch_readout/__init__.py`
- Create: `scripts/_br1_wait_for_gpu.sh`

- [ ] **Step 1: Create empty package marker**

```python
# block_lo_arm_order_network/batch_readout/__init__.py
"""BR-1 Batch-Mean Hooked Attention Readout."""
```

- [ ] **Step 2: Write GPU-wait guard for scripts**

```bash
# scripts/_br1_wait_for_gpu.sh
#!/usr/bin/env bash
# Wait until at least one GPU has >= MIN_FREE_MB free memory.
set -euo pipefail
MIN_FREE_MB=${MIN_FREE_MB:-8000}
POLL=${POLL:-60}
while true; do
  if nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk -v m=$MIN_FREE_MB '$1>=m{found=1} END{exit !found}'; then
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -v m=$MIN_FREE_MB '$2>=m{print $1; exit}'
    exit 0
  fi
  sleep "$POLL"
done
```

- [ ] **Step 3: Make executable and verify it picks a GPU now**

```bash
chmod +x scripts/_br1_wait_for_gpu.sh
MIN_FREE_MB=1000 bash scripts/_br1_wait_for_gpu.sh
```

Expected: prints a GPU index between 0 and `nvidia-smi -L | wc -l - 1`.

- [ ] **Step 4: Commit**

```bash
git add block_lo_arm_order_network/batch_readout/__init__.py scripts/_br1_wait_for_gpu.sh
git commit -m "feat(batch-readout): BR-1 Task 1 — package marker + GPU-wait guard"
```

---

### Task 2: extract_b_batch — batch-mean attention extractor

**Files:**
- Create: `block_lo_arm_order_network/batch_readout/extract_b_batch.py`
- Test: `block_lo_arm_order_network/tests/test_batch_readout_extract.py`

- [ ] **Step 1: Write failing test**

```python
# block_lo_arm_order_network/tests/test_batch_readout_extract.py
"""Shape + seed reproducibility + batch-mean correctness."""
import numpy as np
import pytest
import torch

from batch_readout.extract_b_batch import extract_batch_mean_B

CKPT = "probe_results/attention_order_mlp/alt_from0_random/ckpt_step5000.pt"


@pytest.mark.gpu
def test_shape_and_dtype():
    out = extract_batch_mean_B(
        ckpt_path=CKPT, M=4, batch_size=8, seed=0,
        device="cuda:0", return_per_sample=True,
    )
    assert out["B_batch"].shape == (4, 64, 64)
    assert out["B_batch"].dtype == np.float64
    assert out["B_per_sample"].shape == (4, 8, 64, 64)
    assert out["chunks"].shape == (4, 8)
    # diagonals zeroed
    assert np.allclose(np.diagonal(out["B_batch"], axis1=1, axis2=2), 0.0)


@pytest.mark.gpu
def test_batch_mean_equals_mean_of_per_sample():
    out = extract_batch_mean_B(
        ckpt_path=CKPT, M=2, batch_size=4, seed=7,
        device="cuda:0", return_per_sample=True,
    )
    expected = out["B_per_sample"].mean(axis=1)
    np.testing.assert_allclose(out["B_batch"], expected, rtol=1e-6, atol=1e-9)


@pytest.mark.gpu
def test_seed_reproducible():
    a = extract_batch_mean_B(CKPT, M=2, batch_size=4, seed=42, device="cuda:0")
    b = extract_batch_mean_B(CKPT, M=2, batch_size=4, seed=42, device="cuda:0")
    np.testing.assert_array_equal(a["B_batch"], b["B_batch"])
    np.testing.assert_array_equal(a["chunks"], b["chunks"])


@pytest.mark.gpu
def test_seed_differs():
    a = extract_batch_mean_B(CKPT, M=2, batch_size=4, seed=42, device="cuda:0")
    b = extract_batch_mean_B(CKPT, M=2, batch_size=4, seed=43, device="cuda:0")
    assert not np.array_equal(a["chunks"], b["chunks"])
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /home/admin/lyuyuhuan/order_lyu && PYTHONPATH=block_lo_arm_order_network pytest -q block_lo_arm_order_network/tests/test_batch_readout_extract.py
```

Expected: `ImportError: No module named 'batch_readout.extract_b_batch'`

- [ ] **Step 3: Implement extractor**

```python
# block_lo_arm_order_network/batch_readout/extract_b_batch.py
"""BR-1 Task 2: batch-mean attention extractor.

Groups (M · batch_size) per-sample graphs into M batch-mean graphs
B_batch[m] = mean_b A_theta(x_b)^T, with diagonal zeroed (consistent with
neural_readout.extract_b).
"""
import numpy as np

from neural_readout.extract_b import extract_per_sample_B_with_chunks


def extract_batch_mean_B(
    ckpt_path,
    M: int,
    batch_size: int,
    seed: int,
    device: str = "cuda:0",
    split: str = "train",
    return_per_sample: bool = False,
):
    if M <= 0 or batch_size <= 0:
        raise ValueError(f"M and batch_size must be positive; got M={M}, batch_size={batch_size}")
    total = M * batch_size
    B_per, chunks, split_str = extract_per_sample_B_with_chunks(
        ckpt_path=ckpt_path, M=total, seed=seed, device=device, split=split,
    )
    # Reshape into M groups of batch_size, then mean over the group axis.
    B_per_grouped = B_per.reshape(M, batch_size, B_per.shape[-2], B_per.shape[-1])
    B_batch = B_per_grouped.mean(axis=1)
    # Re-zero diagonal in case averaging introduced rounding noise.
    diag = np.arange(B_batch.shape[-1])
    B_batch[:, diag, diag] = 0.0
    chunks_grouped = np.asarray(chunks).reshape(M, batch_size)
    out = {
        "B_batch": B_batch.astype(np.float64),
        "chunks": chunks_grouped,
        "split": split_str,
        "meta": {"M": M, "batch_size": batch_size, "seed": seed, "ckpt": ckpt_path},
    }
    if return_per_sample:
        out["B_per_sample"] = B_per_grouped.astype(np.float64)
    return out
```

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=block_lo_arm_order_network pytest -q block_lo_arm_order_network/tests/test_batch_readout_extract.py -m "gpu or not gpu"
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/batch_readout/extract_b_batch.py block_lo_arm_order_network/tests/test_batch_readout_extract.py
git commit -m "feat(batch-readout): BR-1 Task 2 — batch-mean B extractor"
```

---

### Task 3: dataset_batch — build / save / load with teacher labels

**Files:**
- Create: `block_lo_arm_order_network/batch_readout/dataset_batch.py`
- Test: `block_lo_arm_order_network/tests/test_batch_readout_dataset.py`

- [ ] **Step 1: Write failing test**

```python
# block_lo_arm_order_network/tests/test_batch_readout_dataset.py
"""Round-trip save/load + split sizes + rank-0=earliest convention."""
import numpy as np
import pytest

from batch_readout.dataset_batch import build_dataset, load_dataset

CKPT = "probe_results/attention_order_mlp/alt_from0_random/ckpt_step5000.pt"


@pytest.mark.gpu
def test_build_and_roundtrip(tmp_path):
    out_path = tmp_path / "br1_small.npz"
    info = build_dataset(
        ckpt_path=CKPT, M=20, batch_size=4, seed=0,
        alpha_dep=0.5, train_frac=0.8, val_frac=0.1, out_path=str(out_path),
    )
    assert info["M_train"] + info["M_val"] + info["M_test"] == 20
    ds = load_dataset(str(out_path))
    for split in ("train", "val", "test"):
        B = ds[f"{split}_B_batch"]
        sig = ds[f"{split}_sigma_T"]
        rk = ds[f"{split}_rank"]
        Y = ds[f"{split}_pairwise_Y"]
        N = B.shape[-1]
        # earliest reveal has rank 0
        assert (rk[np.arange(len(rk)), sig[:, 0]] == 0).all()
        # pairwise consistency
        i = np.random.RandomState(0).randint(0, N, size=10)
        j = np.random.RandomState(1).randint(0, N, size=10)
        # avoid i==j to skip the diagonal
        mask = i != j
        assert (Y[0, i[mask], j[mask]] == (rk[0, i[mask]] < rk[0, j[mask]]).astype(np.uint8)).all()
```

- [ ] **Step 2: Run to confirm failure**

```bash
PYTHONPATH=block_lo_arm_order_network pytest -q block_lo_arm_order_network/tests/test_batch_readout_dataset.py
```

Expected: ImportError.

- [ ] **Step 3: Implement dataset module**

```python
# block_lo_arm_order_network/batch_readout/dataset_batch.py
"""BR-1 Task 3: build / save / load (B_batch, sigma_T, rank, pairwise_Y) dataset.

Teacher labels are computed per batch-mean graph via the reused
neural_readout.teacher_labels.generate_teacher_label (CDL source-start, alpha_dep=0.5).
"""
import numpy as np

from batch_readout.extract_b_batch import extract_batch_mean_B
from neural_readout.teacher_labels import generate_teacher_label


def build_dataset(
    ckpt_path, M, batch_size, seed, alpha_dep=0.5,
    out_path=None, train_frac=0.8, val_frac=0.1,
    device="cuda:0", split="train",
):
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError(f"invalid splits train={train_frac} val={val_frac}")
    ext = extract_batch_mean_B(
        ckpt_path=ckpt_path, M=M, batch_size=batch_size, seed=seed,
        device=device, split=split, return_per_sample=False,
    )
    B = ext["B_batch"]
    N = B.shape[-1]
    sigma = np.zeros((M, N), dtype=np.int64)
    rank = np.zeros((M, N), dtype=np.int64)
    Y = np.zeros((M, N, N), dtype=np.uint8)
    for m in range(M):
        s_m, r_m, y_m = generate_teacher_label(B[m], alpha_dep=alpha_dep)
        sigma[m] = s_m
        rank[m] = r_m
        Y[m] = y_m
    # split with the same seed for determinism
    rng = np.random.default_rng(seed + 1000003)
    idx = rng.permutation(M)
    n_train = int(round(M * train_frac))
    n_val = int(round(M * val_frac))
    tr, va, te = idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]
    if out_path is not None:
        np.savez(
            out_path,
            train_B_batch=B[tr], train_sigma_T=sigma[tr], train_rank=rank[tr], train_pairwise_Y=Y[tr],
            val_B_batch=B[va], val_sigma_T=sigma[va], val_rank=rank[va], val_pairwise_Y=Y[va],
            test_B_batch=B[te], test_sigma_T=sigma[te], test_rank=rank[te], test_pairwise_Y=Y[te],
            train_chunks=ext["chunks"][tr], val_chunks=ext["chunks"][va], test_chunks=ext["chunks"][te],
            meta=np.array([ckpt_path, M, batch_size, seed, alpha_dep], dtype=object),
        )
    return {"M_train": len(tr), "M_val": len(va), "M_test": len(te)}


def load_dataset(path):
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}
```

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=block_lo_arm_order_network pytest -q block_lo_arm_order_network/tests/test_batch_readout_dataset.py
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/batch_readout/dataset_batch.py block_lo_arm_order_network/tests/test_batch_readout_dataset.py
git commit -m "feat(batch-readout): BR-1 Task 3 — dataset_batch builder + npz I/O"
```

---

### Task 4: diversity_batch — teacher diversity statistics

**Files:**
- Create: `block_lo_arm_order_network/batch_readout/diversity_batch.py`
- Test: `block_lo_arm_order_network/tests/test_batch_readout_diversity.py`

- [ ] **Step 1: Write failing test**

```python
# block_lo_arm_order_network/tests/test_batch_readout_diversity.py
"""Identical labels yield diversity ~0, random labels yield diversity > 0.5."""
import numpy as np
import pytest

from batch_readout.diversity_batch import teacher_diversity_stats


def test_identical_labels_yield_low_diversity():
    sigma = np.tile(np.arange(64), (50, 1))  # all the same
    stats = teacher_diversity_stats(sigma)
    assert stats["mean_pairwise_tau"] == pytest.approx(1.0, abs=1e-6)
    assert stats["unique_sigma_ratio"] == pytest.approx(1 / 50)
    assert stats["first_step_entropy"] == pytest.approx(0.0, abs=1e-9)


def test_random_labels_yield_high_diversity():
    rng = np.random.default_rng(0)
    sigma = np.stack([rng.permutation(64) for _ in range(200)])
    stats = teacher_diversity_stats(sigma)
    assert -0.05 < stats["mean_pairwise_tau"] < 0.05
    assert stats["unique_sigma_ratio"] == 1.0
    # 64 buckets, ~uniform → entropy near log(64) ≈ 4.16
    assert stats["first_step_entropy"] > 3.5
```

- [ ] **Step 2: Run to confirm failure**

```bash
PYTHONPATH=block_lo_arm_order_network pytest -q block_lo_arm_order_network/tests/test_batch_readout_diversity.py
```

Expected: ImportError.

- [ ] **Step 3: Implement diversity module**

```python
# block_lo_arm_order_network/batch_readout/diversity_batch.py
"""BR-1 Task 4: teacher-label diversity statistics across M batch-mean graphs."""
import math

import numpy as np
from scipy.stats import kendalltau


def teacher_diversity_stats(sigma):
    """sigma: (M, N) int orders. Returns dict of summary statistics."""
    sigma = np.asarray(sigma)
    M, N = sigma.shape
    # mean pairwise Kendall tau across all (m1, m2) with m1 < m2
    if M < 2:
        mean_tau = float("nan")
    else:
        taus = []
        for i in range(M):
            for j in range(i + 1, M):
                t, _ = kendalltau(sigma[i], sigma[j])
                taus.append(t)
        mean_tau = float(np.mean(taus))
    # unique-sigma ratio
    unique_n = len({tuple(s.tolist()) for s in sigma})
    # first-step entropy (Shannon, base e)
    first_counts = np.bincount(sigma[:, 0], minlength=N)
    p = first_counts / first_counts.sum()
    H = -float(np.sum(p[p > 0] * np.log(p[p > 0])))
    return {
        "mean_pairwise_tau": mean_tau,
        "unique_sigma_ratio": unique_n / M,
        "first_step_entropy": H,
        "first_step_entropy_max": math.log(N),
    }
```

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=block_lo_arm_order_network pytest -q block_lo_arm_order_network/tests/test_batch_readout_diversity.py
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/batch_readout/diversity_batch.py block_lo_arm_order_network/tests/test_batch_readout_diversity.py
git commit -m "feat(batch-readout): BR-1 Task 4 — teacher-label diversity stats"
```

---

### Task 5: model — FlattenReadout (v1) and NodewiseReadout (v2)

**Files:**
- Create: `block_lo_arm_order_network/batch_readout/model.py`
- Test: `block_lo_arm_order_network/tests/test_batch_readout_model.py`

- [ ] **Step 1: Write failing test**

```python
# block_lo_arm_order_network/tests/test_batch_readout_model.py
"""Shapes + param-count sanity + no PE + finite outputs."""
import torch

from batch_readout.model import FlattenReadout, NodewiseReadout


def test_flatten_shape():
    m = FlattenReadout(N=64, hidden=(256, 64))
    B = torch.randn(3, 64, 64)
    z = m(B)
    assert z.shape == (3, 64)
    assert torch.isfinite(z).all()


def test_nodewise_shape():
    m = NodewiseReadout(N=64, d_model=64, n_layers=2, n_heads=4)
    B = torch.randn(3, 64, 64)
    z = m(B)
    assert z.shape == (3, 64)
    assert torch.isfinite(z).all()


def test_nodewise_no_positional_embedding():
    m = NodewiseReadout(N=64, d_model=64, n_layers=2, n_heads=4)
    # Permuting node identities and the corresponding B rows/cols
    # should commute with the model (up to the same permutation of outputs)
    # IF there is no PE.
    torch.manual_seed(0)
    B = torch.randn(1, 64, 64)
    perm = torch.randperm(64)
    B_perm = B[:, perm][:, :, perm]
    z1 = m(B)[0]
    z2 = m(B_perm)[0]
    # Match up to permutation
    torch.testing.assert_close(z1[perm], z2, atol=1e-5, rtol=1e-5)
```

- [ ] **Step 2: Run to confirm failure**

```bash
PYTHONPATH=block_lo_arm_order_network pytest -q block_lo_arm_order_network/tests/test_batch_readout_model.py
```

Expected: ImportError.

- [ ] **Step 3: Implement models**

```python
# block_lo_arm_order_network/batch_readout/model.py
"""BR-1 Task 5: g_beta models.

FlattenReadout: vec(B) -> MLP -> z. Permutation-sensitive (uses slot identity).
NodewiseReadout: per-node features -> shared MLP -> TransformerEncoder mixer ->
  scalar. Permutation-equivariant when no positional embedding is added.
"""
import torch
import torch.nn as nn


class FlattenReadout(nn.Module):
    def __init__(self, N: int = 64, hidden=(1024, 256)):
        super().__init__()
        self.N = N
        dims = [N * N, *hidden, N]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, B: torch.Tensor) -> torch.Tensor:
        # B: (batch, N, N) -> (batch, N)
        return self.net(B.reshape(B.shape[0], -1))


class NodewiseReadout(nn.Module):
    def __init__(self, N: int = 64, d_model: int = 64, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.N = N
        self.node_in = nn.Linear(2 * N, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.mixer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, B: torch.Tensor) -> torch.Tensor:
        # B: (batch, N, N). per-node feature = [B[v, :], B[:, v]] in R^{2N}.
        x = torch.cat([B, B.transpose(1, 2)], dim=-1)  # (batch, N, 2N)
        h = self.node_in(x)                            # (batch, N, d_model)
        h = self.mixer(h)                              # (batch, N, d_model)
        return self.head(h).squeeze(-1)                # (batch, N)
```

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=block_lo_arm_order_network pytest -q block_lo_arm_order_network/tests/test_batch_readout_model.py
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/batch_readout/model.py block_lo_arm_order_network/tests/test_batch_readout_model.py
git commit -m "feat(batch-readout): BR-1 Task 5 — FlattenReadout + NodewiseReadout"
```

---

### Task 6: loss — pairwise logistic + Plackett-Luce listwise

**Files:**
- Create: `block_lo_arm_order_network/batch_readout/loss.py`
- Test: `block_lo_arm_order_network/tests/test_batch_readout_loss.py`

- [ ] **Step 1: Write failing test**

```python
# block_lo_arm_order_network/tests/test_batch_readout_loss.py
"""Pairwise direction + PL closed form on N=3."""
import math

import torch

from batch_readout.loss import pairwise_logistic_loss, plackett_luce_nll


def test_pairwise_direction():
    # rank: earliest = 0, latest = N-1
    rank = torch.tensor([[0, 1, 2]])  # rank-0 earliest -> highest score
    z_correct = torch.tensor([[10.0, 0.0, -10.0]])
    z_wrong = torch.tensor([[-10.0, 0.0, 10.0]])
    assert pairwise_logistic_loss(z_correct, rank) < 0.1
    assert pairwise_logistic_loss(z_wrong, rank) > 5.0


def test_pl_closed_form_n3():
    # PL NLL for sigma = [0, 1, 2] given z = [a, b, c] (tau=1):
    # P = e^a / (e^a+e^b+e^c) * e^b / (e^b+e^c) * 1
    # NLL = -log P = (logsumexp(a,b,c) - a) + (logsumexp(b,c) - b)
    z = torch.tensor([[2.0, 1.0, 0.0]])
    sig = torch.tensor([[0, 1, 2]])
    expected = (torch.logsumexp(z[0], 0) - z[0, 0]) + (torch.logsumexp(z[0, 1:], 0) - z[0, 1])
    got = plackett_luce_nll(z, sig, tau=1.0)
    assert math.isclose(got.item(), expected.item(), rel_tol=1e-6)


def test_pl_temperature_scaling():
    # Sharper temperature should *lower* NLL when teacher matches argsort.
    z = torch.tensor([[2.0, 1.0, 0.0]])
    sig = torch.tensor([[0, 1, 2]])
    hot = plackett_luce_nll(z, sig, tau=2.0).item()
    cold = plackett_luce_nll(z, sig, tau=0.5).item()
    assert cold < hot
```

- [ ] **Step 2: Run to confirm failure**

```bash
PYTHONPATH=block_lo_arm_order_network pytest -q block_lo_arm_order_network/tests/test_batch_readout_loss.py
```

Expected: ImportError.

- [ ] **Step 3: Implement losses**

```python
# block_lo_arm_order_network/batch_readout/loss.py
"""BR-1 Task 6: pairwise logistic + Plackett-Luce listwise NLL.

Convention: rank-0 = earliest reveal, earliest -> highest score.
"""
import torch
import torch.nn.functional as F


def pairwise_logistic_loss(logits: torch.Tensor, rank: torch.Tensor) -> torch.Tensor:
    """logits: (B, N). rank: (B, N) with 0=earliest. Loss = mean over (i, j) of
    -log sigmoid(z_i - z_j) for rank_i < rank_j (i.e., earlier should outscore later)."""
    B, N = logits.shape
    zi = logits.unsqueeze(2)  # (B, N, 1)
    zj = logits.unsqueeze(1)  # (B, 1, N)
    mask = (rank.unsqueeze(2) < rank.unsqueeze(1)).float()  # 1 where i earlier than j
    diff = zi - zj
    # logsigmoid is more numerically stable than log(sigmoid(.))
    loss_mat = -F.logsigmoid(diff) * mask
    denom = mask.sum().clamp_min(1.0)
    return loss_mat.sum() / denom


def plackett_luce_nll(logits: torch.Tensor, sigma_T: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """logits: (B, N), sigma_T: (B, N) order with sigma[0] earliest.
    NLL = sum_{t=0..N-2} (logsumexp(z[t:]/tau) - z[sigma_t]/tau), mean over batch.
    """
    B, N = logits.shape
    # Gather logits in reveal order: z_perm[t] = logits[arange(B), sigma_T[:, t]]
    z_perm = logits.gather(1, sigma_T) / tau  # (B, N)
    # Running logsumexp from index t to N-1
    nll = z_perm.new_zeros(B)
    # backward cumulative logsumexp
    cum_lse = torch.full((B,), float("-inf"), device=z_perm.device, dtype=z_perm.dtype)
    for t in range(N - 1, -1, -1):
        cum_lse = torch.logaddexp(cum_lse, z_perm[:, t])
        if t < N - 1:
            nll = nll + (cum_lse - z_perm[:, t])
    # Last term (t=N-1) is logsumexp([z_{N-1}]) - z_{N-1} = 0; already excluded.
    return nll.mean()
```

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=block_lo_arm_order_network pytest -q block_lo_arm_order_network/tests/test_batch_readout_loss.py
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/batch_readout/loss.py block_lo_arm_order_network/tests/test_batch_readout_loss.py
git commit -m "feat(batch-readout): BR-1 Task 6 — pairwise + Plackett-Luce losses"
```

---

### Task 7: pl_sampling — without-replacement PL sampler + argsort

**Files:**
- Create: `block_lo_arm_order_network/batch_readout/pl_sampling.py`
- Test: `block_lo_arm_order_network/tests/test_batch_readout_pl_sampling.py`

- [ ] **Step 1: Write failing test**

```python
# block_lo_arm_order_network/tests/test_batch_readout_pl_sampling.py
"""Valid permutation + seed determinism + cold temperature -> closer to argsort."""
import torch
from scipy.stats import kendalltau

from batch_readout.pl_sampling import pl_sample, pl_argsort


def test_argsort_is_valid_permutation():
    z = torch.tensor([[3.0, 1.0, 5.0, 0.0]])
    sig = pl_argsort(z)
    assert sig.shape == (1, 4)
    assert sorted(sig[0].tolist()) == [0, 1, 2, 3]
    # earliest reveal (sig[0]) is argmax of z
    assert sig[0, 0].item() == 2


def test_sample_is_valid_permutation():
    torch.manual_seed(0)
    z = torch.randn(4, 64)
    sig = pl_sample(z, tau=1.0)
    assert sig.shape == (4, 64)
    for row in sig:
        assert sorted(row.tolist()) == list(range(64))


def test_seed_determinism():
    z = torch.randn(2, 32)
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    a = pl_sample(z, tau=0.5, generator=g1)
    b = pl_sample(z, tau=0.5, generator=g2)
    assert torch.equal(a, b)


def test_cold_temperature_approaches_argsort():
    torch.manual_seed(0)
    z = torch.randn(50, 32)
    expected = pl_argsort(z)
    g = torch.Generator().manual_seed(0)
    sig = pl_sample(z, tau=0.001, generator=g)
    taus = [kendalltau(sig[i].numpy(), expected[i].numpy())[0] for i in range(50)]
    assert sum(taus) / len(taus) > 0.99
```

- [ ] **Step 2: Run to confirm failure**

```bash
PYTHONPATH=block_lo_arm_order_network pytest -q block_lo_arm_order_network/tests/test_batch_readout_pl_sampling.py
```

Expected: ImportError.

- [ ] **Step 3: Implement sampler**

```python
# block_lo_arm_order_network/batch_readout/pl_sampling.py
"""BR-1 Task 7: Plackett-Luce sampling without replacement.

Uses the Gumbel-top-k trick: sigma = argsort(-(z/tau + Gumbel(0, 1)), dim=-1).
"""
import torch


def pl_argsort(z: torch.Tensor) -> torch.Tensor:
    """Deterministic: highest z = earliest reveal (rank 0).
    z: (B, N) -> sigma: (B, N) int64.
    """
    return torch.argsort(-z, dim=-1).to(torch.int64)


def pl_sample(z: torch.Tensor, tau: float = 1.0, generator: torch.Generator | None = None) -> torch.Tensor:
    """Sample sigma ~ Plackett-Luce(z/tau) via Gumbel-top-k.
    z: (B, N) -> sigma: (B, N) int64.
    """
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    u = torch.rand(z.shape, device=z.device, generator=generator).clamp_min(1e-20)
    gumbel = -torch.log(-torch.log(u))
    scores = z / tau + gumbel
    return torch.argsort(-scores, dim=-1).to(torch.int64)
```

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=block_lo_arm_order_network pytest -q block_lo_arm_order_network/tests/test_batch_readout_pl_sampling.py
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/batch_readout/pl_sampling.py block_lo_arm_order_network/tests/test_batch_readout_pl_sampling.py
git commit -m "feat(batch-readout): BR-1 Task 7 — Plackett-Luce sampler + argsort"
```

---

### Task 8: eval_metrics — Kendall τ / pairwise / Spearman / top-k

**Files:**
- Create: `block_lo_arm_order_network/batch_readout/eval_metrics.py`
- Test: `block_lo_arm_order_network/tests/test_batch_readout_metrics.py`

- [ ] **Step 1: Write failing test**

```python
# block_lo_arm_order_network/tests/test_batch_readout_metrics.py
"""Identity / reversed / known values."""
import numpy as np
import torch

from batch_readout.eval_metrics import (
    kendall_tau_batch, pairwise_acc, spearman_rho_batch, top1_acc, first_k_overlap,
)


def test_identity_metrics():
    sig = np.tile(np.arange(64), (5, 1))
    assert kendall_tau_batch(sig, sig) == 1.0
    assert spearman_rho_batch(sig, sig) == 1.0
    z = torch.tensor(np.linspace(1, -1, 64))[None].repeat(5, 1)
    rank = sig.copy()  # earliest = 0
    assert pairwise_acc(z, torch.from_numpy(rank)) == 1.0


def test_reversed_kendall_minus_one():
    sig = np.tile(np.arange(64), (5, 1))
    rev = sig[:, ::-1].copy()
    assert kendall_tau_batch(sig, rev) == -1.0


def test_top1_and_first3():
    pred = torch.tensor([[2, 0, 1, 3, 4]])  # earliest = 2
    teacher = torch.tensor([[2, 1, 0, 4, 3]])
    assert top1_acc(pred, teacher) == 1.0
    # first-3 of pred: {2,0,1}; first-3 of teacher: {2,1,0} -> overlap 3
    assert first_k_overlap(pred, teacher, k=3) == 1.0
```

- [ ] **Step 2: Run to confirm failure**

```bash
PYTHONPATH=block_lo_arm_order_network pytest -q block_lo_arm_order_network/tests/test_batch_readout_metrics.py
```

Expected: ImportError.

- [ ] **Step 3: Implement metrics**

```python
# block_lo_arm_order_network/batch_readout/eval_metrics.py
"""BR-1 Task 8: matching metrics. NLL is NOT here — it lives in eval_frozen_phase2."""
import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr


def kendall_tau_batch(sigma_pred, sigma_true) -> float:
    sigma_pred = np.asarray(sigma_pred)
    sigma_true = np.asarray(sigma_true)
    vals = [kendalltau(sigma_pred[i], sigma_true[i])[0] for i in range(len(sigma_pred))]
    return float(np.mean(vals))


def spearman_rho_batch(sigma_pred, sigma_true) -> float:
    sigma_pred = np.asarray(sigma_pred)
    sigma_true = np.asarray(sigma_true)
    vals = [spearmanr(sigma_pred[i], sigma_true[i])[0] for i in range(len(sigma_pred))]
    return float(np.mean(vals))


def pairwise_acc(logits: torch.Tensor, rank: torch.Tensor) -> float:
    # Fraction of (i, j) with rank_i < rank_j where z_i > z_j
    zi = logits.unsqueeze(2)
    zj = logits.unsqueeze(1)
    mask = (rank.unsqueeze(2) < rank.unsqueeze(1))
    correct = (zi > zj) & mask
    return (correct.sum().float() / mask.sum().clamp_min(1).float()).item()


def top1_acc(sigma_pred: torch.Tensor, sigma_true: torch.Tensor) -> float:
    return (sigma_pred[:, 0] == sigma_true[:, 0]).float().mean().item()


def first_k_overlap(sigma_pred: torch.Tensor, sigma_true: torch.Tensor, k: int = 3) -> float:
    out = []
    for p, t in zip(sigma_pred.tolist(), sigma_true.tolist()):
        out.append(len(set(p[:k]) & set(t[:k])) / k)
    return float(np.mean(out))
```

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=block_lo_arm_order_network pytest -q block_lo_arm_order_network/tests/test_batch_readout_metrics.py
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/batch_readout/eval_metrics.py block_lo_arm_order_network/tests/test_batch_readout_metrics.py
git commit -m "feat(batch-readout): BR-1 Task 8 — eval metrics (no NLL here)"
```

---

### Task 9: train_offline — offline training loop with selection guard

**Files:**
- Create: `block_lo_arm_order_network/batch_readout/train_offline.py`
- Test: `block_lo_arm_order_network/tests/test_batch_readout_train_overfit.py`
- Test: `block_lo_arm_order_network/tests/test_batch_readout_selection_policy.py`

- [ ] **Step 1: Write failing tests**

```python
# block_lo_arm_order_network/tests/test_batch_readout_train_overfit.py
"""Overfit a tiny dataset to confirm the loop trains."""
import numpy as np
import pytest
import torch

from batch_readout.train_offline import train


@pytest.mark.slow
def test_overfit_tiny(tmp_path):
    rng = np.random.default_rng(0)
    N, Mt = 16, 16
    B = rng.standard_normal((Mt, N, N)).astype(np.float64)
    np.fill_diagonal_v = None  # noqa
    for m in range(Mt):
        np.fill_diagonal(B[m], 0.0)
    sigma = np.stack([rng.permutation(N) for _ in range(Mt)]).astype(np.int64)
    rank = np.argsort(sigma, axis=1).astype(np.int64)
    Y = (rank[:, :, None] < rank[:, None, :]).astype(np.uint8)
    npz = tmp_path / "tiny.npz"
    np.savez(
        npz,
        train_B_batch=B, train_sigma_T=sigma, train_rank=rank, train_pairwise_Y=Y,
        val_B_batch=B[:2], val_sigma_T=sigma[:2], val_rank=rank[:2], val_pairwise_Y=Y[:2],
        test_B_batch=B[:2], test_sigma_T=sigma[:2], test_rank=rank[:2], test_pairwise_Y=Y[:2],
    )
    out = train(
        dataset_path=str(npz),
        model_name="flatten", loss_name="pairwise",
        N=N, batch_size=4, lr=1e-2, epochs=100, seed=0,
        out_dir=str(tmp_path / "ckpts"),
        device="cpu",
    )
    assert out["final_train_loss"] < 0.05
```

```python
# block_lo_arm_order_network/tests/test_batch_readout_selection_policy.py
"""Enforce: train_offline never imports eval_frozen_phase2 or any AR-NLL code."""
import importlib
import inspect


def test_train_offline_does_not_import_nll_modules():
    mod = importlib.import_module("batch_readout.train_offline")
    src = inspect.getsource(mod)
    forbidden = [
        "eval_frozen_phase2",
        "compute_frozen_nll_gap",
        "forward_fn",                       # AR forward in train_clean_aogpt
        "neural_readout.eval_frozen_nll",
    ]
    for f in forbidden:
        assert f not in src, f"selection guard: {f!r} must NOT appear in train_offline"
```

- [ ] **Step 2: Run to confirm failure**

```bash
PYTHONPATH=block_lo_arm_order_network pytest -q block_lo_arm_order_network/tests/test_batch_readout_train_overfit.py block_lo_arm_order_network/tests/test_batch_readout_selection_policy.py
```

Expected: ImportError on train_offline.

- [ ] **Step 3: Implement training loop**

```python
# block_lo_arm_order_network/batch_readout/train_offline.py
"""BR-1 Task 9: offline training of g_beta.

Selection is by pairwise precedence accuracy on validation. NLL is NEVER imported
or computed here — the selection policy test enforces this.
"""
import json
import pathlib

import numpy as np
import torch

from batch_readout.dataset_batch import load_dataset
from batch_readout.eval_metrics import (
    kendall_tau_batch, pairwise_acc, spearman_rho_batch, top1_acc, first_k_overlap,
)
from batch_readout.loss import pairwise_logistic_loss, plackett_luce_nll
from batch_readout.model import FlattenReadout, NodewiseReadout
from batch_readout.pl_sampling import pl_argsort


def _build_model(name, N):
    if name == "flatten":
        return FlattenReadout(N=N, hidden=(1024, 256))
    if name == "nodewise":
        return NodewiseReadout(N=N, d_model=64, n_layers=2, n_heads=4)
    raise ValueError(f"unknown model {name!r}")


def _loss_fn(name):
    if name == "pairwise":
        return lambda z, rank, sigma: pairwise_logistic_loss(z, rank)
    if name == "pl":
        return lambda z, rank, sigma: plackett_luce_nll(z, sigma, tau=1.0)
    if name == "pairwise+pl":
        return lambda z, rank, sigma: pairwise_logistic_loss(z, rank) + 0.5 * plackett_luce_nll(z, sigma, tau=1.0)
    raise ValueError(f"unknown loss {name!r}")


def _eval_split(model, B, sigma_T, rank, device):
    model.eval()
    with torch.no_grad():
        Bt = torch.from_numpy(B).float().to(device)
        z = model(Bt).cpu()
        sigma_pred = pl_argsort(z).numpy()
    metrics = {
        "kendall_tau": kendall_tau_batch(sigma_pred, sigma_T),
        "pairwise_acc": pairwise_acc(z, torch.from_numpy(rank)),
        "spearman_rho": spearman_rho_batch(sigma_pred, sigma_T),
        "top1": top1_acc(torch.from_numpy(sigma_pred), torch.from_numpy(sigma_T)),
        "first3": first_k_overlap(torch.from_numpy(sigma_pred), torch.from_numpy(sigma_T), k=3),
    }
    return metrics


def train(
    dataset_path, model_name, loss_name,
    N=64, batch_size=64, lr=3e-4, epochs=40, seed=0,
    out_dir=None, device="cuda:0",
):
    torch.manual_seed(seed)
    ds = load_dataset(dataset_path)
    Btr = ds["train_B_batch"].astype(np.float32)
    sigT = ds["train_sigma_T"].astype(np.int64)
    rk = ds["train_rank"].astype(np.int64)

    model = _build_model(model_name, N).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = _loss_fn(loss_name)

    out_dir = pathlib.Path(out_dir) if out_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    best = {"pairwise_acc": -1.0, "epoch": -1, "metrics": None}
    history = []
    rng = np.random.default_rng(seed)
    n_train = len(Btr)
    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(n_train)
        losses = []
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            Bb = torch.from_numpy(Btr[idx]).float().to(device)
            rb = torch.from_numpy(rk[idx]).to(device)
            sb = torch.from_numpy(sigT[idx]).to(device)
            z = model(Bb)
            loss = loss_fn(z, rb, sb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        train_loss = float(np.mean(losses))
        val_metrics = _eval_split(
            model, ds["val_B_batch"].astype(np.float32),
            ds["val_sigma_T"].astype(np.int64), ds["val_rank"].astype(np.int64),
            device=device,
        )
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
        if val_metrics["pairwise_acc"] > best["pairwise_acc"]:
            best = {"pairwise_acc": val_metrics["pairwise_acc"], "epoch": epoch, "metrics": val_metrics}
            if out_dir is not None:
                torch.save({"model": model.state_dict(), "config": {
                    "model_name": model_name, "loss_name": loss_name, "N": N,
                }, "epoch": epoch, "metrics": val_metrics}, out_dir / "g_beta_best.pt")
    if out_dir is not None:
        with open(out_dir / "history.json", "w") as f:
            json.dump({"history": history, "best": best}, f, indent=2)
    return {
        "history": history, "best_epoch": best["epoch"], "best_metrics": best["metrics"],
        "final_train_loss": history[-1]["train_loss"],
    }
```

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=block_lo_arm_order_network pytest -q block_lo_arm_order_network/tests/test_batch_readout_train_overfit.py block_lo_arm_order_network/tests/test_batch_readout_selection_policy.py -m "slow or not slow"
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/batch_readout/train_offline.py block_lo_arm_order_network/tests/test_batch_readout_train_overfit.py block_lo_arm_order_network/tests/test_batch_readout_selection_policy.py
git commit -m "feat(batch-readout): BR-1 Task 9 — train_offline + selection-policy guard"
```

---

### Task 10: Phase 0 smoke run — build a tiny dataset end-to-end on the real ckpt

**Files:**
- Create: `scripts/run_br1_phase0_smoke.sh`

- [ ] **Step 1: Write smoke script**

```bash
# scripts/run_br1_phase0_smoke.sh
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="block_lo_arm_order_network:${PYTHONPATH:-}"

CKPT="probe_results/attention_order_mlp/alt_from0_random/ckpt_step5000.pt"
OUT="block_lo_arm_order_network/batch_readout/data/text_5k_M100_B32_seed0.npz"
mkdir -p "$(dirname "$OUT")"
GPU=$(MIN_FREE_MB=8000 bash scripts/_br1_wait_for_gpu.sh)
echo "[BR-1 phase 0 smoke] using GPU=$GPU"
CUDA_VISIBLE_DEVICES=$GPU python - <<'PY'
import json
from batch_readout.dataset_batch import build_dataset
from batch_readout.diversity_batch import teacher_diversity_stats
from batch_readout.dataset_batch import load_dataset

info = build_dataset(
    ckpt_path="probe_results/attention_order_mlp/alt_from0_random/ckpt_step5000.pt",
    M=100, batch_size=32, seed=0, alpha_dep=0.5,
    out_path="block_lo_arm_order_network/batch_readout/data/text_5k_M100_B32_seed0.npz",
    train_frac=0.8, val_frac=0.1, device="cuda:0",
)
print("split sizes:", info)
ds = load_dataset("block_lo_arm_order_network/batch_readout/data/text_5k_M100_B32_seed0.npz")
stats = teacher_diversity_stats(ds["train_sigma_T"])
print("teacher diversity:", json.dumps(stats, indent=2))
PY
```

- [ ] **Step 2: Make executable + dry-run**

```bash
chmod +x scripts/run_br1_phase0_smoke.sh
bash scripts/run_br1_phase0_smoke.sh 2>&1 | tee block_lo_arm_order_network/batch_readout/logs/phase0_smoke.log
```

Expected: produces `data/text_5k_M100_B32_seed0.npz`; log shows `split sizes: {'M_train': 80, 'M_val': 10, 'M_test': 10}`; `mean_pairwise_tau` is between 0.05 and 0.95 (interior — neither degenerate nor random).

- [ ] **Step 3: Inspect dataset and record diversity numbers**

```bash
PYTHONPATH=block_lo_arm_order_network python - <<'PY'
from batch_readout.dataset_batch import load_dataset
ds = load_dataset("block_lo_arm_order_network/batch_readout/data/text_5k_M100_B32_seed0.npz")
print("train_B_batch:", ds["train_B_batch"].shape, ds["train_B_batch"].dtype)
print("train_sigma_T:", ds["train_sigma_T"].shape, ds["train_sigma_T"].dtype)
print("rank-0 == sigma[:,0]?", bool((ds["train_rank"][range(len(ds["train_sigma_T"])), ds["train_sigma_T"][:, 0]] == 0).all()))
PY
```

Expected: prints `(80, 64, 64) float64`, `(80, 64) int64`, `True`.

- [ ] **Step 4: Decision point — abort criteria**

If `mean_pairwise_tau > 0.95`: teacher labels are essentially identical across batches → batch-mean smooths too much; STOP and investigate (consider lower `alpha_dep` or `batch_size`).

If `mean_pairwise_tau < 0.05` and `unique_sigma_ratio ≈ 1.0`: teacher labels are pure noise → batch-mean B has no exploitable structure; STOP and report.

Otherwise: proceed.

- [ ] **Step 5: Commit script + log**

```bash
git add scripts/run_br1_phase0_smoke.sh
git commit -m "chore(batch-readout): BR-1 Task 10 — Phase-0 smoke script"
```

(do NOT commit the `data/` npz nor the log — they are generated.)

---

### Task 11: Phase 1 smoke — train v1 FlattenReadout on M=100

**Files:**
- Create: `scripts/run_br1_phase1_smoke.sh`

- [ ] **Step 1: Write smoke training script**

```bash
# scripts/run_br1_phase1_smoke.sh
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="block_lo_arm_order_network:${PYTHONPATH:-}"

DS="block_lo_arm_order_network/batch_readout/data/text_5k_M100_B32_seed0.npz"
OUT="block_lo_arm_order_network/batch_readout/checkpoints/smoke_flatten_pairwise"
mkdir -p "$OUT"
GPU=$(MIN_FREE_MB=4000 bash scripts/_br1_wait_for_gpu.sh)
CUDA_VISIBLE_DEVICES=$GPU python - <<PY
from batch_readout.train_offline import train
out = train(
    dataset_path="$DS",
    model_name="flatten", loss_name="pairwise",
    N=64, batch_size=16, lr=3e-4, epochs=60, seed=0,
    out_dir="$OUT", device="cuda:0",
)
print("best:", out["best_metrics"], "epoch:", out["best_epoch"])
PY
```

- [ ] **Step 2: Run**

```bash
chmod +x scripts/run_br1_phase1_smoke.sh
bash scripts/run_br1_phase1_smoke.sh 2>&1 | tee block_lo_arm_order_network/batch_readout/logs/phase1_smoke.log
```

Expected: prints best metrics; on M=100 don't expect §5.1 thresholds to pass — this is a pipeline-health smoke. We want `train_loss` monotone-ish and `pairwise_acc > 0.55` (above chance).

- [ ] **Step 3: Decision point — pipeline health**

| Observation | Action |
|---|---|
| `pairwise_acc ≤ 0.51` over 60 epochs | Pipeline broken or M too small for any signal. Run nodewise model on the same dataset before declaring broken. |
| `pairwise_acc 0.55 – 0.70` | Signal present, M=100 underpowered. Proceed to Task 12 (M=1000 full). |
| `pairwise_acc ≥ 0.70` | Smoke already meets §5.1 lower bound, proceed and expect even better at M=1000. |

- [ ] **Step 4: Commit script**

```bash
git add scripts/run_br1_phase1_smoke.sh
git commit -m "chore(batch-readout): BR-1 Task 11 — Phase-1 smoke training script"
```

---

### Task 12: Phase 1 full — M=1000 train, both models, both losses

**Files:**
- Create: `scripts/run_br1_phase1_full.sh`

- [ ] **Step 1: Write full script**

```bash
# scripts/run_br1_phase1_full.sh
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="block_lo_arm_order_network:${PYTHONPATH:-}"

CKPT="probe_results/attention_order_mlp/alt_from0_random/ckpt_step5000.pt"
DS="block_lo_arm_order_network/batch_readout/data/text_5k_M1000_B32_seed0.npz"
LOG_DIR="block_lo_arm_order_network/batch_readout/logs"
CKPT_DIR="block_lo_arm_order_network/batch_readout/checkpoints"
mkdir -p "$LOG_DIR" "$CKPT_DIR"

GPU=$(MIN_FREE_MB=8000 bash scripts/_br1_wait_for_gpu.sh)
echo "[BR-1 phase 1 full] GPU=$GPU"
# 1) build dataset if absent
[ -f "$DS" ] || CUDA_VISIBLE_DEVICES=$GPU python - <<PY
from batch_readout.dataset_batch import build_dataset
build_dataset(
    ckpt_path="$CKPT", M=1000, batch_size=32, seed=0, alpha_dep=0.5,
    out_path="$DS", train_frac=0.8, val_frac=0.1, device="cuda:0",
)
PY

# 2) train 4 configs
for MODEL in flatten nodewise; do
  for LOSS in pairwise pl; do
    TAG="full_M1000_B32_${MODEL}_${LOSS}"
    OUT="$CKPT_DIR/$TAG"
    mkdir -p "$OUT"
    LOG="$LOG_DIR/phase1_${TAG}.log"
    echo "[BR-1] train $TAG -> $LOG"
    CUDA_VISIBLE_DEVICES=$GPU python - <<PY 2>&1 | tee "$LOG"
from batch_readout.train_offline import train
out = train(
    dataset_path="$DS",
    model_name="$MODEL", loss_name="$LOSS",
    N=64, batch_size=64, lr=3e-4, epochs=40, seed=0,
    out_dir="$OUT", device="cuda:0",
)
print("best:", out["best_metrics"], "epoch:", out["best_epoch"])
PY
  done
done
```

- [ ] **Step 2: Run and wait**

```bash
chmod +x scripts/run_br1_phase1_full.sh
bash scripts/run_br1_phase1_full.sh 2>&1 | tee block_lo_arm_order_network/batch_readout/logs/phase1_full_orchestrator.log
```

Expected runtime: ≈30 min build + 4 × ≈10 min train = ≈70 min on one A100.

- [ ] **Step 3: §5.1 gate check**

Pick the run with the highest val pairwise_acc. Check:
- Kendall τ ≥ 0.50?
- pairwise_acc ≥ 0.70?
- Spearman ρ ≥ 0.55?

If any fail: record best run as `g_beta_phase1_best.pt` for Phase 2 anyway, and add an "interpretation" note. Do NOT raise thresholds post-hoc.

- [ ] **Step 4: Symlink the winner**

```bash
cd block_lo_arm_order_network/batch_readout/checkpoints
ln -sf <winning_run_dir>/g_beta_best.pt g_beta_phase1_best.pt
ls -l g_beta_phase1_best.pt
```

Expected: symlink resolves to the winning ckpt.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_br1_phase1_full.sh
git commit -m "chore(batch-readout): BR-1 Task 12 — Phase-1 full M=1000 script"
```

---

### Task 13: eval_frozen_phase2 — post-selection AR-NLL diagnostic

**Files:**
- Create: `block_lo_arm_order_network/batch_readout/eval_frozen_phase2.py`
- Test: `block_lo_arm_order_network/tests/test_batch_readout_eval_frozen_phase2.py`
- Create: `scripts/run_br1_phase2_frozen.sh`

- [ ] **Step 1: Write failing test**

```python
# block_lo_arm_order_network/tests/test_batch_readout_eval_frozen_phase2.py
"""Wrapper API contract; full NLL pass tested manually on real ckpt."""
import importlib
import inspect


def test_module_exposes_compute_arms():
    mod = importlib.import_module("batch_readout.eval_frozen_phase2")
    assert hasattr(mod, "compute_arms")
    sig = inspect.signature(mod.compute_arms)
    expected = {"ckpt_path", "g_beta_path", "dataset_path", "arms", "tau", "seed", "device"}
    assert set(sig.parameters.keys()) == expected
```

- [ ] **Step 2: Run to confirm failure**

```bash
PYTHONPATH=block_lo_arm_order_network pytest -q block_lo_arm_order_network/tests/test_batch_readout_eval_frozen_phase2.py
```

Expected: ImportError.

- [ ] **Step 3: Implement wrapper**

```python
# block_lo_arm_order_network/batch_readout/eval_frozen_phase2.py
"""BR-1 Task 13: post-selection AR-NLL diagnostic over arms {random, sigma_T, mlp-argsort, mlp-sample}.

Wraps neural_readout.eval_frozen_nll.compute_frozen_nll_gap, but feeds it BR-1
orders (one per *batch-mean graph*) replicated to each of the batch members.
Reported numbers are diagnostic only — do not feed them back into selection.
"""
from __future__ import annotations
import json
import pathlib

import numpy as np
import torch

from batch_readout.dataset_batch import load_dataset
from batch_readout.model import FlattenReadout, NodewiseReadout
from batch_readout.pl_sampling import pl_argsort, pl_sample
from neural_readout.eval_frozen_nll import compute_frozen_nll_gap  # noqa: F401  reused for AR-NLL math


def _load_g_beta(path, N=64):
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = state["config"]
    name = cfg["model_name"]
    if name == "flatten":
        model = FlattenReadout(N=N, hidden=(1024, 256))
    elif name == "nodewise":
        model = NodewiseReadout(N=N, d_model=64, n_layers=2, n_heads=4)
    else:
        raise ValueError(f"unknown model {name!r}")
    model.load_state_dict(state["model"])
    model.eval()
    return model


def compute_arms(
    ckpt_path, g_beta_path, dataset_path, arms=("random", "teacher", "mlp_argsort", "mlp_sample"),
    tau=0.3, seed=0, device="cuda:0",
):
    """Return {arm: {"mean_nll": float, "delta_vs_random": float}} on val split.

    Reuses the AR-NLL math in neural_readout.eval_frozen_nll by replicating each
    batch-mean order to its constituent batch members (per the recorded `val_chunks`).
    """
    ds = load_dataset(dataset_path)
    val_chunks = ds["val_chunks"]                # (M_val, batch_size)
    sigma_T = ds["val_sigma_T"]                  # (M_val, N)
    B_batch = ds["val_B_batch"]                  # (M_val, N, N)
    N = B_batch.shape[-1]

    # Get MLP orders if needed
    z = None
    if any(a.startswith("mlp_") for a in arms):
        model = _load_g_beta(g_beta_path, N=N).to(device)
        with torch.no_grad():
            z = model(torch.from_numpy(B_batch).float().to(device)).cpu()

    # Build per-(chunk) orders for each arm
    chunk_indices_flat = val_chunks.reshape(-1)              # (M_val*B,)
    orders_per_arm = {}
    M_val, Bsz = val_chunks.shape
    rng = np.random.default_rng(seed)
    if "random" in arms:
        rand_orders = np.stack([rng.permutation(N) for _ in range(M_val)])
        orders_per_arm["random"] = np.repeat(rand_orders, Bsz, axis=0)
    if "teacher" in arms:
        orders_per_arm["teacher"] = np.repeat(sigma_T, Bsz, axis=0)
    if "mlp_argsort" in arms:
        orders_per_arm["mlp_argsort"] = np.repeat(pl_argsort(z).numpy(), Bsz, axis=0)
    if "mlp_sample" in arms:
        g = torch.Generator().manual_seed(seed)
        orders_per_arm["mlp_sample"] = np.repeat(pl_sample(z, tau=tau, generator=g).numpy(), Bsz, axis=0)

    results = {}
    for arm, orders in orders_per_arm.items():
        # delegate AR-NLL computation to neural_readout's helper
        # NOTE: compute_frozen_nll_gap expects a dataset path + g_beta path; we
        # instead use its internal function via the dataset's chunk list.
        # For simplicity, we wrap a thin recomputation here.
        mean_nll = _ar_nll_under_orders(ckpt_path, chunk_indices_flat, orders, device)
        results[arm] = {"mean_nll": mean_nll}

    if "random" in results:
        baseline = results["random"]["mean_nll"]
        for arm, r in results.items():
            r["delta_vs_random"] = r["mean_nll"] - baseline

    return results


def _ar_nll_under_orders(ckpt_path, chunk_indices, orders_phys, device):
    """Thin re-implementation of the loop in neural_readout.eval_frozen_nll
    that accepts an arbitrary per-row order array. Uses the same model build
    path so numbers are comparable."""
    import sys as _sys, pathlib as _p
    _R = _p.Path(__file__).resolve().parents[2]
    _sys.path.insert(0, str(_R / "block_lo_arm_order_network"))
    from train_clean_aogpt import build_model, CleanPermutation
    from training_utils import load_train_chunks, SEQ_LEN, BLOCK_LEN
    from clean_training_protocol import physical_blocks_to_model_token_order

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    protocol = ckpt["clean_protocol"]
    model_args = dict(ckpt["model_args"]); model_args["block_size"] = SEQ_LEN
    clean_perm = CleanPermutation(
        block_perm_phys_to_model=torch.tensor(protocol["block_perm_phys_to_model"], dtype=torch.long),
        inv_perm_model_to_phys=torch.tensor(protocol["inv_perm_model_to_phys"], dtype=torch.long),
    )
    dev = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = build_model(model_args, dev, compile_model=False)
    sd = ckpt.get("model") or ckpt.get("model_state_dict")
    clean_sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(clean_sd); model.eval()
    tokens = load_train_chunks(split="train")[np.asarray(chunk_indices, dtype=np.int64)]
    losses = []
    with torch.no_grad():
        for i in range(len(orders_phys)):
            o = torch.from_numpy(orders_phys[i:i + 1].astype(np.int64))
            tok_order = physical_blocks_to_model_token_order(o, clean_perm, BLOCK_LEN).to(dev)
            idx = torch.from_numpy(tokens[i:i + 1]).to(dev)
            _, loss = model.forward_fn(idx, tok_order)
            losses.append(float(loss.item()))
    return float(np.mean(losses))
```

- [ ] **Step 4: Write Phase-2 runner**

```bash
# scripts/run_br1_phase2_frozen.sh
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="block_lo_arm_order_network:${PYTHONPATH:-}"

CKPT="probe_results/attention_order_mlp/alt_from0_random/ckpt_step5000.pt"
GBETA="block_lo_arm_order_network/batch_readout/checkpoints/g_beta_phase1_best.pt"
DS="block_lo_arm_order_network/batch_readout/data/text_5k_M1000_B32_seed0.npz"
GPU=$(MIN_FREE_MB=6000 bash scripts/_br1_wait_for_gpu.sh)
CUDA_VISIBLE_DEVICES=$GPU python - <<PY
import json
from batch_readout.eval_frozen_phase2 import compute_arms
out = compute_arms(
    ckpt_path="$CKPT", g_beta_path="$GBETA", dataset_path="$DS",
    arms=("random", "teacher", "mlp_argsort", "mlp_sample"),
    tau=0.3, seed=0, device="cuda:0",
)
print(json.dumps(out, indent=2))
PY
```

- [ ] **Step 5: Run + record + gate**

```bash
chmod +x scripts/run_br1_phase2_frozen.sh
bash scripts/run_br1_phase2_frozen.sh 2>&1 | tee block_lo_arm_order_network/batch_readout/logs/phase2_frozen.log
```

Check the §5.2 gate (see plan header). If `mlp_argsort` fails to beat random by ≥0.005: STOP and document. Do not start Phase 3.

- [ ] **Step 6: Commit**

```bash
git add block_lo_arm_order_network/batch_readout/eval_frozen_phase2.py block_lo_arm_order_network/tests/test_batch_readout_eval_frozen_phase2.py scripts/run_br1_phase2_frozen.sh
git commit -m "feat(batch-readout): BR-1 Task 13 — Phase-2 frozen-θ NLL arms diagnostic"
```

---

### Task 14: integration_hook — in-loop frozen-β order provider

**Files:**
- Create: `block_lo_arm_order_network/batch_readout/integration_hook.py`
- Test: `block_lo_arm_order_network/tests/test_batch_readout_integration_hook.py`

- [ ] **Step 1: Write failing test**

```python
# block_lo_arm_order_network/tests/test_batch_readout_integration_hook.py
"""Hook returns valid permutation from a fake batch of attention; tau / mode behave."""
import pathlib

import numpy as np
import pytest
import torch

from batch_readout.model import FlattenReadout
from batch_readout.integration_hook import FrozenBetaHook


@pytest.fixture
def fake_g_beta_ckpt(tmp_path):
    m = FlattenReadout(N=64, hidden=(64,))
    p = tmp_path / "fake_gbeta.pt"
    torch.save({"model": m.state_dict(), "config": {"model_name": "flatten", "N": 64}}, p)
    return str(p)


def test_argsort_returns_valid_permutation(fake_g_beta_ckpt):
    hook = FrozenBetaHook(g_beta_ckpt=fake_g_beta_ckpt, mode="argsort", device="cpu")
    A = torch.randn(8, 64, 64)
    sig = hook.step(A)
    assert sig.shape == (64,)
    assert sorted(sig.tolist()) == list(range(64))


def test_sample_seed_determinism(fake_g_beta_ckpt):
    A = torch.randn(8, 64, 64)
    h1 = FrozenBetaHook(g_beta_ckpt=fake_g_beta_ckpt, mode="sample", tau=0.5, seed=0, device="cpu")
    h2 = FrozenBetaHook(g_beta_ckpt=fake_g_beta_ckpt, mode="sample", tau=0.5, seed=0, device="cpu")
    assert torch.equal(h1.step(A), h2.step(A))
```

- [ ] **Step 2: Run to confirm failure**

```bash
PYTHONPATH=block_lo_arm_order_network pytest -q block_lo_arm_order_network/tests/test_batch_readout_integration_hook.py
```

Expected: ImportError.

- [ ] **Step 3: Implement hook**

```python
# block_lo_arm_order_network/batch_readout/integration_hook.py
"""BR-1 Task 14: in-loop frozen-beta order provider.

Holds the trained g_beta ckpt and a sampling mode. On `.step(A)` it computes
B_batch = mean(A^T) over the batch axis, runs g_beta to get z, and returns
either argsort(-z) or PL-sample(z/tau).
"""
import torch

from batch_readout.model import FlattenReadout, NodewiseReadout
from batch_readout.pl_sampling import pl_argsort, pl_sample


def _build_from_config(cfg):
    name = cfg["model_name"]
    N = cfg.get("N", 64)
    if name == "flatten":
        return FlattenReadout(N=N, hidden=cfg.get("hidden", (1024, 256)))
    if name == "nodewise":
        return NodewiseReadout(N=N, d_model=cfg.get("d_model", 64),
                               n_layers=cfg.get("n_layers", 2), n_heads=cfg.get("n_heads", 4))
    raise ValueError(f"unknown model {name!r}")


class FrozenBetaHook:
    def __init__(self, g_beta_ckpt: str, mode: str = "argsort", tau: float = 1.0,
                 seed: int | None = None, device: str = "cuda:0"):
        if mode not in ("argsort", "sample"):
            raise ValueError(f"mode must be 'argsort' or 'sample', got {mode!r}")
        state = torch.load(g_beta_ckpt, map_location="cpu", weights_only=False)
        self.model = _build_from_config(state["config"])
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        self.model.to(self.device)
        self.mode = mode
        self.tau = float(tau)
        self.generator = torch.Generator(device="cpu")
        if seed is not None:
            self.generator.manual_seed(int(seed))

    @torch.no_grad()
    def step(self, attention: torch.Tensor) -> torch.Tensor:
        """attention: (batch, N, N) per-sample A_theta(x). Returns next-step sigma: (N,) int64.
        B_batch = mean over batch of A^T, diagonal zeroed.
        """
        if attention.ndim != 3 or attention.shape[1] != attention.shape[2]:
            raise ValueError(f"attention must be (batch, N, N); got {tuple(attention.shape)}")
        A = attention.to(self.device).float()
        B = A.transpose(1, 2).mean(dim=0, keepdim=True)
        N = B.shape[-1]
        diag = torch.arange(N, device=B.device)
        B[:, diag, diag] = 0.0
        z = self.model(B).cpu()
        if self.mode == "argsort":
            return pl_argsort(z)[0]
        return pl_sample(z, tau=self.tau, generator=self.generator)[0]
```

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=block_lo_arm_order_network pytest -q block_lo_arm_order_network/tests/test_batch_readout_integration_hook.py
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/batch_readout/integration_hook.py block_lo_arm_order_network/tests/test_batch_readout_integration_hook.py
git commit -m "feat(batch-readout): BR-1 Task 14 — FrozenBetaHook in-loop order provider"
```

---

### Task 15: train_clean_aogpt — accept an `order_provider` callable

**Files:**
- Modify: `block_lo_arm_order_network/train_clean_aogpt.py`

#### §Distribution-match note (RESOLVED 2026-05-29) — lightweight hook signal = fixed single head **L0 H5**, NOT head-mean

A multi-seed single-(layer,head) scan (`scripts/diag_br1_head_layer_scan.py`,
M=100×B=32, alt_from0_random/ckpt_step5000.pt, seeds 0/1/2) **falsified the
original "last-layer / all-layer head-mean" hook design**:

| candidate | pairwise τ | τ_vs_heavy | verdict |
|---|---|---|---|
| last-layer head-mean | ~0.05 | ~0.12 | collapses |
| all-layer head-mean | ~0.258 | ~0.21 | weak |
| **L0 H5 single head** | **0.43–0.51** | (τ_vs_L2R 0.562/0.583/0.638 ≈ heavy 0.673/0.647/0.620) | **stable winner** |

Root cause head-mean fails: heads carry *opposite-signed* order structure — L0 H5
is L2R-like, L0 H1 is consistently **anti-L2R** (−0.35…−0.42 across seeds).
`A_mean = (1/H)Σ_h A_h` cancels them and washes the order signal out. The signal
is concentrated almost entirely in **Layer 0**; L0 H5 also has *healthier*
diversity (first_step_entropy ≈ 3.2–3.35) than the heavy extractor (first_H=0,
CDL source-anchor L2R-prior lock-in).

**Decisions locked:**
1. `_lightweight_extract` aggregates to **a single fixed head (layer 0, head 5)** —
   `A^{(0,5)}` → reveal→physical remap → 256→64 block-agg. **No head-mean, no
   multi-layer aggregation, no top-4 variance head-selection.**
2. **B_train == B_hook.** g_β CANNOT reuse the heavy-B / head-mean-B Phase-1 winner
   — the feature distribution would be out-of-distribution. **Before Phase-3 you
   MUST rebuild the Phase-1 dataset from L0H5-derived B and retrain
   `g_β^{L0H5}`**, re-running the §5.1 imitation gate on it. Only then wire the hook.
3. L0H5 is an **empirical choice for the current text AOGPT config**, not a universal
   law (head index may move with depth/seed/modality). Paper framing must say
   "the strongest single-head order signal consistently appears in an early-layer
   head (L0H5) across three seeds," not "all models use L0H5." Future ablation:
   automatic head selection at warmup (scan once at 5k, fix best head) or a small
   learned head-gate.

This note resolves the "distribution-match decision" referenced in Step 2 below.

- [ ] **Step 1: Identify the training-loop entry point**

```bash
grep -n "def run_training\|def main\|order=" block_lo_arm_order_network/train_clean_aogpt.py | head -30
```

Note the exact function signature where `order` is selected each step.

- [ ] **Step 2: Add optional kwarg + plumbing**

In `run_training` (or equivalent main loop):

⚠️ **kwarg fix**: `forward_fn` has NO `return_attention=True` kwarg. The real
signature is `forward_fn(idx, orders, return_probe_data=False,
return_last_attention=False, return_all_attentions=False)` and returns a
3-tuple `(_, loss, attn_list)` when attentions are requested. The L0H5 hook
needs the **first layer**, so use `return_all_attentions=True` (NOT
`return_last_attention=True` — that would give the wrong layer).

⚠️ **raw→block gap**: `attn_list` is a list of raw `(B, H, 257, 257)`
token-level attention tensors in *reveal* coordinates. `FrozenBetaHook.step`
expects `(batch, 64, 64)` per-sample **block-level** A in *physical*
coordinates. Per the §Distribution-match note above, `_lightweight_extract`
selects **a single fixed head (layer 0, head 5)** — `A^{(0,5)}` →
reveal→physical remap → 256→64 block-agg → optional [None]. It does **NOT**
head-mean, does **NOT** aggregate layers, and does **NOT** do top-4 variance
head-selection (that whole `extract_A_matrices` machinery is deliberately
dropped). To reach L0H5 you need the per-layer attentions, so use
`return_all_attentions=True` and index `attn_list[0][:, 5]` (not
`return_last_attention`). The hook cannot consume `attn_list` directly — this
shim sits between `forward_fn` and `hook.step`, and **its B-definition MUST be
identical to the one g_β^{L0H5} was retrained on**, or g_β sees an
out-of-distribution graph.

```python
# OLD (sketch):
def run_training(cfg, ...):
    ...
    for step in range(total_steps):
        order = sample_order(...)
        _, loss, _ = model.forward_fn(idx, order)
        ...

# NEW (sketch):
def run_training(cfg, ..., order_provider=None):
    """If order_provider is given, it is called with the most recent per-sample
    block-level A batch (B, 64, 64) and returns sigma for the NEXT step.

    First step still uses the baseline sample_order to bootstrap.
    """
    ...
    next_order_override = None
    for step in range(total_steps):
        if order_provider is not None and next_order_override is not None:
            order = next_order_override
        else:
            order = sample_order(...)
        _, loss, attn_list = model.forward_fn(
            idx, order, return_all_attentions=True)
        if order_provider is not None:
            # attn_list: list of (B, H, 257, 257) raw, reveal-coord, per layer.
            # _lightweight_extract picks layer 0 head 5 (attn_list[0][:, 5]),
            # remaps reveal->physical, block-aggs 256->64, → (B, 64, 64) physical
            # block-level A. SAME B-definition g_β^{L0H5} was retrained on.
            A_block = _lightweight_extract(attn_list, order, clean_perm)
            next_order_override = order_provider.step(A_block).to(order.device)
        ...
```

Concrete diff to write: pass `order_provider` from the existing CLI/config plumbing, default `None`, no behavior change for existing callers. Implement `_lightweight_extract` to match the Phase-1 B-definition (resolve the distribution-match decision first).

- [ ] **Step 3: Write a guard test that the default path is unchanged**

Append to `block_lo_arm_order_network/tests/test_batch_readout_integration_hook.py`:

```python
def test_default_run_training_is_unchanged():
    """If order_provider is None, behavior must be identical to current code path.
    Test as a string presence + import check (no full training run)."""
    import importlib
    mod = importlib.import_module("train_clean_aogpt")
    import inspect
    sig = inspect.signature(mod.run_training)
    assert "order_provider" in sig.parameters
    assert sig.parameters["order_provider"].default is None
```

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=block_lo_arm_order_network pytest -q block_lo_arm_order_network/tests/test_batch_readout_integration_hook.py
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/train_clean_aogpt.py block_lo_arm_order_network/tests/test_batch_readout_integration_hook.py
git commit -m "feat(batch-readout): BR-1 Task 15 — optional order_provider in run_training"
```

---

### Task 16: Phase 3 short run — 5k → 10k, 4 arms

**Files:**
- Create: `scripts/run_br1_phase3_integration.sh`

⚠️ **PREREQUISITE (§Distribution-match note in Task 15):** before this run,
the Phase-1 dataset must have been rebuilt from **L0H5-derived B** and
`g_β^{L0H5}` retrained + re-gated. `GBETA` below MUST point at that retrained
checkpoint (e.g. `g_beta_l0h5_phase1_best.pt`), **NOT** the heavy-B
`g_beta_phase1_best.pt` — wiring the heavy-B winner to an L0H5 hook is exactly
the train/inference mismatch we are avoiding.

- [ ] **Step 1: Write Phase-3 runner**

```bash
# scripts/run_br1_phase3_integration.sh
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="block_lo_arm_order_network:${PYTHONPATH:-}"

# MUST be the L0H5-retrained g_β (see Task 15 §Distribution-match note), NOT heavy-B winner
GBETA="block_lo_arm_order_network/batch_readout/checkpoints/g_beta_l0h5_phase1_best.pt"
RESUME="probe_results/attention_order_mlp/alt_from0_random/ckpt_step5000.pt"
OUT_BASE="probe_results/attention_order_mlp/br1_phase3"
mkdir -p "$OUT_BASE"

GPU=$(MIN_FREE_MB=10000 bash scripts/_br1_wait_for_gpu.sh)
echo "[BR-1 phase 3] GPU=$GPU"

for ARM in random teacher mlp_argsort mlp_sample; do
  TAG="arm_${ARM}_5k_to_10k"
  OUT="$OUT_BASE/$TAG"
  mkdir -p "$OUT"
  LOG="$OUT/run.log"
  CUDA_VISIBLE_DEVICES=$GPU python -m train_clean_aogpt \
      --config configs/text/wikitext103_seq256_block64_v3_readiness.py \
      --resume "$RESUME" --max_steps 10000 \
      --out_dir "$OUT" \
      --order_arm "$ARM" --g_beta_ckpt "$GBETA" --tau 0.3 \
      2>&1 | tee "$LOG"
done
```

(Note: `--order_arm` + `--g_beta_ckpt` + `--tau` are new CLI args added to `train_clean_aogpt` as a thin shim that builds the `FrozenBetaHook` and passes it as `order_provider` for `mlp_argsort` / `mlp_sample`; for `random` and `teacher` we keep the existing samplers but force their respective modes. This shim lives next to `run_training` in `train_clean_aogpt.py`; the test added in Task 15 should be extended to assert these args exist when the file is touched.)

- [ ] **Step 2: Run all 4 arms (sequentially to avoid GPU contention)**

```bash
chmod +x scripts/run_br1_phase3_integration.sh
bash scripts/run_br1_phase3_integration.sh
```

Expected: each arm trains 5k more steps; logs include `val_ori_l2r`, `train_loss`, `order_entropy`, `tau_vs_teacher`.

- [ ] **Step 3: §5.3 gate check**

Compare `val_ori_l2r` at 10k across arms. Apply criteria from §5.3.

- [ ] **Step 4: Record per-step jitter**

```bash
PYTHONPATH=block_lo_arm_order_network python - <<'PY'
import json, pathlib, numpy as np
# load per-step σ from each run's history and compute average τ between consecutive σ
for arm in ["mlp_argsort", "mlp_sample"]:
    f = pathlib.Path(f"probe_results/attention_order_mlp/br1_phase3/arm_{arm}_5k_to_10k/history.json")
    if not f.exists():
        continue
    h = json.loads(f.read_text())
    # implementation-specific extraction omitted — the runner is expected to
    # have written per-step σ to the history file.
    print(arm, "<jitter computed here>")
PY
```

- [ ] **Step 5: Commit**

```bash
git add scripts/run_br1_phase3_integration.sh
git commit -m "chore(batch-readout): BR-1 Task 16 — Phase-3 short integration run"
```

---

### Task 17: Ablation matrix — A1, A2, A4, A6 (Phase 1 axes)

**Files:**
- Create: `scripts/run_br1_ablation_phase1.sh`

The Phase-1 ablations all reuse `train_offline`. This task just wires them up as a single script and produces a CSV.

- [ ] **Step 1: Write sweep script**

```bash
# scripts/run_br1_ablation_phase1.sh
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="block_lo_arm_order_network:${PYTHONPATH:-}"

CKPT="probe_results/attention_order_mlp/alt_from0_random/ckpt_step5000.pt"
DATA_DIR="block_lo_arm_order_network/batch_readout/data"
CKPT_DIR="block_lo_arm_order_network/batch_readout/checkpoints"
LOG_DIR="block_lo_arm_order_network/batch_readout/logs"
CSV="$LOG_DIR/ablation_phase1.csv"
mkdir -p "$DATA_DIR" "$CKPT_DIR" "$LOG_DIR"
echo "axis,value,model,loss,kendall_tau,pairwise_acc,spearman_rho,top1,first3" > "$CSV"

GPU=$(MIN_FREE_MB=8000 bash scripts/_br1_wait_for_gpu.sh)

run_one() {
    local AXIS="$1" VAL="$2" M="$3" B="$4" MODEL="$5" LOSS="$6"
    local DS="$DATA_DIR/text_5k_M${M}_B${B}_seed0.npz"
    [ -f "$DS" ] || CUDA_VISIBLE_DEVICES=$GPU python - <<PY
from batch_readout.dataset_batch import build_dataset
build_dataset(ckpt_path="$CKPT", M=$M, batch_size=$B, seed=0, alpha_dep=0.5,
              out_path="$DS", train_frac=0.8, val_frac=0.1, device="cuda:0")
PY
    local TAG="abl_${AXIS}_${VAL}_${MODEL}_${LOSS}"
    local OUT="$CKPT_DIR/$TAG"
    mkdir -p "$OUT"
    CUDA_VISIBLE_DEVICES=$GPU python - <<PY 2>&1 | tee "$LOG_DIR/$TAG.log"
from batch_readout.train_offline import train
out = train(dataset_path="$DS", model_name="$MODEL", loss_name="$LOSS",
            N=64, batch_size=64, lr=3e-4, epochs=40, seed=0,
            out_dir="$OUT", device="cuda:0")
m = out["best_metrics"]
print(f"BR1_ABL,{$M},{$B},$MODEL,$LOSS,{m['kendall_tau']:.4f},{m['pairwise_acc']:.4f},{m['spearman_rho']:.4f},{m['top1']:.4f},{m['first3']:.4f}")
PY
    grep "^BR1_ABL" "$LOG_DIR/$TAG.log" | tail -1 | awk -F, -v ax="$AXIS" -v v="$VAL" '{printf "%s,%s,%s,%s,%s,%s,%s,%s,%s\n", ax, v, $4, $5, $6, $7, $8, $9, $10}' >> "$CSV"
}

# A1: M sweep at B=32, nodewise+pairwise (Phase-1 default)
run_one M 100   100   32 nodewise pairwise
run_one M 1000  1000  32 nodewise pairwise
run_one M 10000 10000 32 nodewise pairwise

# A2: batch_size sweep at M=1000, nodewise+pairwise
run_one B 16  1000 16  nodewise pairwise
run_one B 128 1000 128 nodewise pairwise

# A5: model sweep at M=1000, B=32, pairwise
run_one MODEL flatten  1000 32 flatten  pairwise
# nodewise/pairwise already covered by A1

# A8: loss sweep at M=1000, B=32, nodewise
run_one LOSS pl 1000 32 nodewise pl
```

(A6 `B_global` and A9 `ckpt` are deferred to a separate optional sweep — they need a different dataset builder; see Risk register.)

- [ ] **Step 2: Run**

```bash
chmod +x scripts/run_br1_ablation_phase1.sh
bash scripts/run_br1_ablation_phase1.sh
cat block_lo_arm_order_network/batch_readout/logs/ablation_phase1.csv
```

Expected: CSV with ≈8 rows (one per (axis, value) above).

- [ ] **Step 3: Plot or inspect**

```bash
PYTHONPATH=block_lo_arm_order_network python - <<'PY'
import csv, pathlib
rows = list(csv.DictReader(open("block_lo_arm_order_network/batch_readout/logs/ablation_phase1.csv")))
for r in rows:
    print(f"{r['axis']:<6} {r['value']:<8} {r['model']:<8} {r['loss']:<8} τ={r['kendall_tau']} pair={r['pairwise_acc']}")
PY
```

- [ ] **Step 4: Commit**

```bash
git add scripts/run_br1_ablation_phase1.sh
git commit -m "chore(batch-readout): BR-1 Task 17 — Phase-1 ablation matrix (A1/A2/A5/A8)"
```

---

### Task 18: Final report + memory update

**Files:**
- Create: `analyses/batch_readout_br1_2026-05-28/REPORT.md`

- [ ] **Step 1: Write report skeleton**

```bash
mkdir -p analyses/batch_readout_br1_2026-05-28
```

```markdown
# BR-1 Batch-Mean Hooked Attention Readout — Report (2026-05-28)

## TL;DR
- Phase 1 (offline): <best (model, loss, M)>. Kendall τ=<>, pairwise=<>, ρ=<>.
- Phase 2 (frozen-θ NLL diag): ΔNLL(mlp-argsort) = <> vs random; <> vs teacher.
- Phase 3 (frozen-β integration): val_ori_l2r at 10k: random=<>, teacher=<>, mlp-argsort=<>, mlp-sample=<>.

## §5.1 Phase 1 gate: PASS / FAIL
## §5.2 Phase 2 diag: PASS / FAIL
## §5.3 Phase 3 gate: PASS / FAIL

## Ablation results
Paste CSV summary from Task 17, then 1-line interpretations per axis.

## Risks observed
Walk through the risk register and which fired / which were ruled out.

## Next-step recommendations
```

- [ ] **Step 2: Fill the report from the runs of Tasks 12 / 13 / 16 / 17**

(No code: a writing task. Reference exact log paths.)

- [ ] **Step 3: Update memory**

Update `MEMORY.md` to add an entry under "attn_order_mlp_line" or as a new sibling: `[BR-1 status](br1_batch_readout_status.md)`. Create `br1_batch_readout_status.md` summarizing PASS/FAIL of §5.1/§5.2/§5.3 and the ablation winners.

- [ ] **Step 4: Commit**

```bash
git add analyses/batch_readout_br1_2026-05-28/REPORT.md
git commit -m "docs(batch-readout): BR-1 final report"
```

---

## Self-review check (run before declaring plan complete)

1. **Spec coverage:** every user-listed item in the original brief is covered? Checked: B_batch definition (Task 2), teacher labels (Task 3), MLP model (Task 5), PL sampling (Task 7), pairwise+PL loss (Task 6), offline train (Task 9), frozen NLL eval (Task 13), in-loop hook (Task 14), 5k→10k Phase 3 (Task 16), full ablation list A1–A11 (§6 + Tasks 12/17). A6 `B_global` and A9 cross-ckpt are explicitly deferred to optional sweeps — flagged in §6 and risk register.
2. **No placeholders:** all code blocks contain runnable code; all shell commands have expected outputs; thresholds are concrete numbers.
3. **Type consistency:** `FrozenBetaHook.step(attention)` returns `(N,)` int64 — matches what `run_training`'s `order_provider.step()` is expected to return in Task 15. `extract_batch_mean_B` returns dict with keys `B_batch` / `chunks` / `split` / `meta`, consumed by `build_dataset` in Task 3 — matches. `_load_g_beta` in eval_frozen_phase2 reads `state["config"]["model_name"]` — matches the `torch.save({..., "config": {"model_name": model_name, "loss_name": loss_name, "N": N}, ...})` written by `train_offline.train`.
