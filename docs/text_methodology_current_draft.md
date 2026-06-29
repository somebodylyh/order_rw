# Text Methodology: Current Draft

**Status: INTERNAL WHITEPAPER — NOT FINAL.**
Numbers marked 🔒 Locked / 🔵 Running / 🟡 Provisional.
Do not cite as definitive until all 🔵 → 🔒.

**Date**: 2026-06-10 (this draft)
**Scope**: Text modality only (Wikitext-103, block-permuted order-agnostic training).
**Related**: `docs/auto-order-head-method-retrospective.md` (historical narrative),
`docs/superpowers/specs/2026-06-01-auto-order-head-end-to-end-training-design.md` (design spec).

---

## §1 Problem

We study whether **random-order AO-GPT** internally discovers useful block reveal orders
**without being given physical block positions**.

- Model: AOGPT, 4 layers × 8 heads × 384 dim, block_size=256, block_len=4, n_blocks=64.
- Training: order-agnostic — each sequence's 64 blocks are permuted by a fixed clean permutation π
  that hides physical indices. Position embeddings are assigned by model slot, not physical position.
- Question: Does the attention graph encode a recoverable order signal?
  If so, can we extract it and feed it back to accelerate training?

---

## §2 Clean-Permuted Protocol 🔒 Locked

### 2.1 Fixed clean permutation

A single random permutation π is drawn once (permute_seed = dataset seed)
and applied to **every training and evaluation sequence**.

```
physical block index p ∈ {0..63}
model block index    m = π(p)
```

- π is fixed across all sequences and all training steps.
- Original physical index **never enters the forward pass**.
- Position embeddings are assigned by model slot m, not by p.

### 2.2 Coordinate remap

Attention matrices are extracted in **model coordinates** (m-space).
All offline analysis (CDL, g_β, evaluation vs ori-L2R) remaps to physical coordinates
via `inv_perm[m] = p`.

```python
# Extraction: A_model[m_i, m_j] → remap via inv_perm
# B_physical[p_i, p_j] = A_model[inv_perm[p_i], inv_perm[p_j]]
```

🔒 **This is locked.** The coordinate audit (`coordinate_check.json`) is bit-identical
across all runs. No physical index leaks into training.

### 2.3 Data loading: continuous stream 🔒 Locked

- **Before Jun-06**: fixed 430k-token chunk, ~18 effective epochs → overfitting (val loss U-shape).
- **After Jun-06**: `--data-source continuous` — random-window stream, train ≈ val, no overfitting.
- This fixed a ~0.15 nats confound in val loss. All results from Jun-08 onward use continuous stream.

---

## §3 Emergent Attention Order 🔒 Locked

### 3.1 CDL tau diagnostic: three-model comparison

CDL greedy rollout τ vs ori-L2R, computed on **offline** attention graphs
(model frozen, no training signal):

| Model | best head τ | top-3 mean τ | top-5 mean τ |
|-------|------------|-------------|-------------|
| shuffled-L2R (wrong fixed order) | +0.04 | 0.00 | −0.038 |
| random-order (ours) | +0.49 | +0.458 | +0.300 |
| ori-L2R (standard AR) | +0.373 | +0.302 | +0.258 |

🔒 **This is locked.** Source: `analyses/attention_diagnostic_20260609/`.

**Interpretation**:
- **random-order model recovers physical order in attention** (τ up to +0.49) —
  stronger than the ori-L2R model's own attention signal. This is the core emergent phenomenon.
- **shuffled-L2R model does NOT** (τ ≈ 0) — the wrong fixed order is not encoded.
  This proves order recovery is non-trivial: attention doesn't just encode "whatever order it was trained on."

### 3.2 Training evolution

CDL τ_vs_L2R increases with training steps (non-monotonic but trend is upward).
At 60k steps, top heads reach τ → +1.0.

🔒 **Observation locked.** The non-monotonicity is from randperm variance in extraction
(see `cdl_evolution_clean_base_20260527` memory); single-extraction τ should not be used
for fine-grained decisions.

---

## §4 CDL Scaffold 🟡 Provisional

### 4.1 What CDL is

CDL (Context-Dependent Lamination) is a greedy graph-rollout algorithm that converts
a block-level attention graph B (64×64, B = A^T) into a block ordering σ.

It uses three state-dependent terms:
- **C** (context): hidden-state dependent similarity
- **D** (dependency): attention-flow asymmetry
- **L** (locality): local attention concentration

and one static term:
- **U_t** (uniform): position-index based prior

### 4.2 CDL teacher family

CDL is a **teacher family**, not a single fixed algorithm. The default configuration
uses all three state-dependent terms with equal weights (C−D+L, 1:1:1).
Different orderings (text L2R vs image locality) may benefit from different term weightings;
this is a direction for future work (Stage 3+).

| CDL variant | τ vs ori-L2R (text, seed1 10k L0H7) |
|------------|--------------------------------------|
| full C−D+L | +0.49 |
| C−D only | +0.42 |
| −D only | +0.06 |
| C only | −0.22 |
| L only | −0.38 |

🟡 **Provisional.** The ablation was run on one checkpoint (10k, seed1, L0H7).
Results vary across checkpoints, heads, and modalities. On image, L is important
(unlike text where it's negative). **The full C−D+L (1:1:1) is the default baseline.**
Do not hard-code "−D only" or any single-term variant as the canonical CDL configuration.

### 4.3 CDL's role

CDL is a **scaffold / pseudo-teacher**, NOT the final deployed controller:
- It provides supervised targets for g_β pretraining.
- It never enters the training loop (no CDL in hook).
- Its value is in converting raw B → trainable σ_T targets.

---

## §5 g_β Controller 🔒 Core sanity locked, 🟡 Head selection

### 5.1 Architecture

NodewiseReadout: per-row MLP on B (64×64) → scalar rank score per block → argsort → σ.

```
g_β(B) = argsort(MLP(B[0]), MLP(B[1]), ..., MLP(B[63]))
```

Trained with pairwise ranking loss (Kendall τ optimization) against CDL teacher σ_T.

### 5.2 g_β is NOT a constant L2R prior 🔒 Locked

| Input B source | g_β output τ vs ori-L2R |
|---------------|------------------------|
| L0H7 (trained) | 0.93 |
| L1H1 (CDL-weak, τ_CDL=0.10) | 0.83 |
| random Gaussian | 0.04 |
| uniform / zero | 0.08 |
| L1H1 entry-shuffled | 0.03 |
| L1H1 row+col shuffled | 0.05 |

🔒 **This is locked.** g_β outputs L2R only when the input B carries real graph structure.
Unstructured or shuffled B → τ ≈ 0. This rules out "g_β just memorized the constant L2R order."

### 5.3 Phase 1.5 generalization gate 🔒 Locked

| Metric | seed2 L0H2 (M=2000) |
|--------|---------------------|
| g_β τ (full) | 0.849 |
| L2R-prior τ (baseline) | 0.76 |
| non-L2R subset τ | 0.64 |
| Phase 1.5 verdict | **PASS** |

🔒 **Locked.** g_β outperforms the constant-L2R baseline and reads B-dependent signal
on the non-L2R subset (teacher σ ≠ identity). The earlier smoke result (M=80, g_β < prior)
was an underpower artifact.

### 5.4 Head selection protocol 🟡 Provisional

**Current method (label-free audition)**:

1. Small-train checkpoint (10k steps, random-order continuous).
2. Per-head CDL scan (100-step sweep).
3. Top-4 heads → 500-step audition → select best by forward advantage.
4. g_β pretrained on selected head → Phase 1.5 gate.
5. Frozen_beta hook training with g_β(same head).

🟡 **Provisional.** The audition procedure is the intended final protocol,
but we have only run it on seed1's 10k checkpoint so far.
Cross-seed and cross-step audition not yet verified.

**Row-concentration is EXPLICITLY NOT the selection method.**
Row-concentration (negentropy of B rows) was tested and found to be anti-correlated
with τ for certain head families (e.g., seed1 L0H7: high row_conc but τ = −0.70).
It is reported only as a **failed heuristic baseline**.

**Oracle τ vs L2R is a post-hoc diagnostic only** — never used for selection.

### 5.5 Deployment head ≠ teacher head

| Head | CDL τ | g_β τ | Property |
|------|-------|-------|----------|
| L0H7 (seed1) | 0.49 | 0.93 | CDL-strong, g_β-strong → good teacher |
| L1H1 (seed1) | 0.10 | 0.83 | CDL-weak, g_β-strong → spontaneous L2R convergence |
| L0H2 (seed2) | — | 0.95 | B-dependent, non-L2R n=4-8 → canonical deployment head |

🔒 Pattern locked. The head with best CDL τ is not necessarily the best deployment head.
g_β can read order signal from heads where CDL greedy rollout fails.

---

## §6 Frozen Hook 🔵 Running

### 6.1 Protocol

```
Every K steps:
  Extract A_t (single head, block-level, B0 convention for none-token)
    → B_t = A_t^T
    → σ_{t+1} = g_β(B_t)  [argsort, τ=1.0]
    → feed σ_{t+1} as block order to AOGPT for next K steps

α schedule:
  α: 0 → 1, linear warmup over 5000 steps, ramp from resume step
  When α < 1: interpolate between random order and g_β order
  When α = 1: pure g_β order
```

### 6.2 Refresh overhead

K = 10 steps. Overhead ≈ 9% (extraction + g_β forward every 10 steps).
🔒 **Locked.** Verified at K=10; K=1 had 107% overhead (impractical).

### 6.3 Seed1 L1H1 (spontaneous L2R-convergent head) 🔒 Locked

| Metric | Value |
|--------|-------|
| g_β τ | 0.999 |
| Teacher diversity τ | 1.000 |
| non-L2R n | 0 |
| Nature | Spontaneous convergence to L2R |

🔒 **Observation locked.** This head's B structure spontaneously aligns to L2R.
This is an emergent property of order-agnostic training — the model recovers physical
order in attention without ever seeing it. g_β faithfully reads this signal.

🟡 **Claim caveat.** τ=0.999 looks indistinguishable from "g_β outputs constant L2R."
To rule this out, we require the full sanity chain:
1. ✅ unstructured B → g_β τ ≈ 0 (not constant prior)
2. ✅ shuffled-L2R model → τ ≈ 0 (wrong order not encoded)
3. ✅ fixed clean π → no coordinate leakage
4. Training with this head achieves ori_l2r = 3.312 at 60k
   (but see §7 for baseline comparison caveats).

### 6.4 Seed2 L0H2 (canonical, B-dependent head) 🔵 Running

| Metric | Value |
|--------|-------|
| g_β τ | 0.946 |
| Teacher diversity τ | 0.938 |
| non-L2R n | 4-8 |
| Nature | B-dependent readout with L2R-adjacent signal |

🔵 **Running.** Training at ~59k/60k steps. Latest eval: ori_l2r ≈ 3.332.

### 6.5 Seed2 L0H4 (marginal) 🔒 Locked

| Metric | Value |
|--------|-------|
| final ori_l2r | 3.430 |
| baseline ori_l2r at 50k | 3.445 |
| Δ vs baseline | −0.015 |

🔒 **Locked.** Marginal improvement. This head was selected by earlier heuristic
(pre-audition) and is not the canonical choice.

---

## §7 Evaluation 🟡 Provisional (numbers not yet locked)

### 7.1 Step-to-threshold (primary metric)

🔵 **Running — seed42 multi-start not yet available.**

Thresholds: loss levels that the random baseline reaches.
Hook method evaluated by how many FEWER steps (from the common 10k resume point)
it takes to reach each threshold.

**Seed2 data** (random baseline reaches 3.47 at step 42000, 3.45 at step 47000):

| Method | head | start | step to 3.50 | step to 3.47 | step to 3.45 | final ori_l2r |
|--------|------|-------|-------------|-------------|-------------|--------------|
| frozen_beta | L0H2 | 10k→60k | 21,000 | 24,500 | 27,000 | ~3.332 |
| frozen_beta | L0H4 | 10k→60k | 35,000 | 38,000 | 40,500 | 3.430 |
| random baseline | — | 0→50k | 37,000 | 42,000 | 47,000 | 3.445 |

🟡 **These are raw data points, not the final table.**
Seed42 multi-start (from10k/from20k/from40k) is running now.
Seed2 from20k/from40k not yet run.

### 7.2 Step savings (from seed2, provisional)

Saving = (baseline_steps − hook_steps) / (baseline_steps − 10000)
Both counted from the common 10k resume point.

| Threshold | baseline reaches at | hook (L0H2) reaches at | saving |
|-----------|-------------------|----------------------|--------|
| 3.50 | 37,000 | 21,000 | 59% |
| 3.47 | 42,000 | 24,500 | 55% |
| 3.45 | 47,000 | 27,000 | 54% |

🟡 **Provisional.** These are from a single seed (seed2), single head (L0H2),
single start point (10k). Multi-seed and multi-start robustness not yet demonstrated.

### 7.3 Wall-clock saving 🔒 Locked (V3 seed2)

From `analyses/wall_clock_saving.py` / `wall_clock_overlay.png`:

| Method | wall time to 3.47 | saving |
|--------|-------------------|--------|
| random baseline (continuous) | baseline | — |
| frozen_beta L0H2 (V3) | earlier | ~58–76% (two computation methods) |

🔒 **Locked for this run.** But single-seed, single-start only.

### 7.4 Baseline comparison protocol 🟡 Must standardize

Current issue: seed1 baseline only goes to 10k steps. Seed1 frozen_beta runs to 60k.
Direct Δ comparison (3.312 vs 3.76) is misleading because:
- 3.76 is at step 10k (early training, high loss)
- 3.312 is at step 60k (converged)
- The random baseline would also have improved from 10k → 60k

🟡 **Protocol to standardize:**
- All comparisons must use **step-to-threshold** (not final Δ).
- Baseline = **random continuation from the same resume checkpoint**,
  trained for the same number of additional steps.
- If baseline curve only goes to N steps, either extend it or restrict comparison
  to the overlapping step range.

### 7.5 Direct CDL hook ablation 🔴 Not yet run

To answer "why g_β and not just CDL?":

Proposed experiment: Replace g_β with direct CDL greedy rollout in the hook,
same head, same refresh interval. Compare step-to-threshold.

🟡 **Framing**: CDL hook tests whether the fixed scaffold alone explains the gain;
g_β is the learnable controller distilled from this scaffold.

### 7.6 Seed robustness matrix 🔵 Running

| seed | head | from10k | from20k | from40k |
|------|------|---------|---------|---------|
| seed1 | L1H1 | ✅ 3.312 | ⬜ | ⬜ |
| seed2 | L0H2 | 🔵 running (~3.33) | ⬜ | ⬜ |
| seed42 | L0H2 | 🔵 running | 🔵 running | 🔵 running |

🔵 **Running on GPU0**: g_β seed42 → frozen_beta seed2 from10k → seed42 from20k → seed42 from40k.

🟡 **Gap**: seed2 from20k/from40k and seed1 from20k/from40k not yet scheduled.

---

## §8 What Is Solid vs What Is Not

### 🔒 Locked (solid evidence)

| # | Claim | Evidence |
|---|-------|----------|
| L1 | Fixed clean permutation protocol | coordinate_check.json, no physical index in forward |
| L2 | Continuous data stream eliminates overfitting | train ≈ val, ~0.15 nats confound removed |
| L3 | Random-order attention recovers physical order | CDL τ up to +0.49 vs shuffled-L2R τ ≈ 0 |
| L4 | g_β is not a constant L2R prior | unstructured/shuffled B → τ ≈ 0 |
| L5 | Phase 1.5 generalization gate PASS | g_β τ = 0.849 > L2R-prior 0.76, non-L2R subset τ = 0.64 |
| L6 | g_β can read heads where CDL fails | L1H1 CDL τ=0.10 → g_β τ=0.83 |
| L7 | Frozen hook training converges | All runs complete 60k steps without divergence |
| L8 | Refresh overhead acceptable | K=10 → ~9% overhead |
| L9 | Shuffled-L2R control τ ≈ 0 | Wrong fixed order not encoded → recovery is non-trivial |

### 🔵 Running (evidence being collected)

| # | Claim | Status |
|---|-------|--------|
| R1 | Seed42 multi-start robustness | Pipeline running on GPU0 |
| R2 | Seed2 from20k/from40k | Not yet scheduled |
| R3 | Seed1 from20k/from40k | Not yet scheduled |

### 🟡 Provisional (needs more data or careful phrasing)

| # | Claim | Risk |
|---|-------|------|
| P1 | Head selection via label-free audition | Only run on one seed/step so far |
| P2 | CDL ablation: "−D only is load-bearing" | Different scripts give different rankings |
| P3 | Step savings of 54–59% | Single seed, single start; multi-seed robustness needed |
| P4 | L1H1 τ=0.999 = "spontaneous recovery" | Must be accompanied by full sanity chain (§5.2) |
| P5 | g_β hook beats random baseline on step-to-threshold | Direct random-continuation-from-same-ckpt not yet run |

### 🔴 Not yet done

| # | Experiment | Priority |
|---|-----------|----------|
| N1 | Direct CDL hook ablation | High — answers "why g_β" |
| N2 | Full seed robustness matrix (3 seeds × 3 starts) | High — main table |
| N3 | Step savings with overhead-adjusted wall clock | Medium |
| N4 | Cross-checkpoint audition (not just 10k) | Medium |

---

## §9 Writing Guidelines for Paper

### What to claim

1. **Order-agnostic training induces spontaneous physical-order recovery in attention**
   (L3 + L9, locked).

2. **g_β is a learned graph-to-order readout, not a constant prior**
   (L4 + L5, locked).

3. **Frozen g_β hook accelerates convergence toward ori-L2R loss**
   (L7 + L8, locked; P3 needs multi-seed).

### What NOT to claim (yet)

1. ❌ "Row-concentration selects the best head" — it doesn't; use audition.
2. ❌ "−D only is the unique load-bearing CDL term" — ablation results vary by setting.
3. ❌ "Seed1 L1H1 saves 45% steps" without the full sanity chain.
4. ❌ "Method beats L2R training" — we don't beat ori-L2R; we recover it faster than random.
5. ❌ Final headline numbers — wait for multi-seed table.

### Recommended framing

> Under order-agnostic training, the attention graph spontaneously encodes
> the physical block order. We extract this signal via a label-free head audition,
> distill it into a learned graph-to-order readout (g_β), and deploy it as a
> frozen controller in the training loop. The controller accelerates convergence
> toward the original-order loss, achieving the same loss level in fewer steps
> than a random-order continuation baseline.

---

## Appendix A: Run Registry

| Run ID | Seed | Head | Start | End | Final ori_l2r | Status |
|--------|------|------|-------|-----|--------------|--------|
| frozen_beta_seed1_from10k | 1 | L0H7 | 10k | 30k | 3.597 (@30k) | 🔒 Done (short) |
| frozen_beta_seed1_from10k_l1h1 | 1 | L1H1 | 10k | 60k | 3.312 | 🔒 Done |
| frozen_beta_seed2_from10k | 2 | L0H4 | 10k | 60k | 3.430 | 🔒 Done |
| frozen_beta_seed2_from10k_l0h2 | 2 | L0H2 | 10k | 60k | ~3.332 | 🔵 Running |
| frozen_beta_seed42_from10k_l0h2 | 42 | L0H2 | 10k | 60k | — | 🔵 Queued |
| frozen_beta_seed42_from20k_l0h2 | 42 | L0H2 | 20k | 60k | — | 🔵 Queued |
| frozen_beta_seed42_from40k_l0h2 | 42 | L0H2 | 40k | 60k | — | 🔵 Queued |

## Appendix B: Key File Locations

| Artifact | Path |
|----------|------|
| Attention diagnostic (3-model) | `analyses/attention_diagnostic_20260609/` |
| CDL teacher ablation | `analyses/cdl_teacher_ablation.py` |
| Wall clock analysis | `analyses/wall_clock_saving.py` |
| Phase 1.5 gate report | `batch_readout/logs/phase33_gbeta_seed2_from10k_l0h2/*/full/phase15_report.json` |
| Frozen hook integration | `batch_readout/hook_order_provider.py` |
| Training entry point | `train_clean_aogpt.py --run-kind frozen_beta` |
| Continuous stream loader | `batch_readout/cdl_order_provider.py` (data source) |
| Head selector (audition) | `block_lo_arm_order_network/per_head_order_scan.py` |
