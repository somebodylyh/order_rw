# 06 — Paper-Ready Summary

> 2–3 page summary for paper writing. Chinese with English key terms.
> Tone: paper planning, not chat log.

---

## 1. One-Sentence Summary

**Random-order trained autoregressive models spontaneously encode data-intrinsic sequential structure in their attention graphs, and a lightweight learned readout of these graphs accelerates training by 35–42%.**

---

## 2. Research Question

Do attention graphs from order-agnostic (random-block-order) trained autoregressive models contain readable signals about the data's intrinsic sequential structure, and can these signals be deployed to accelerate training?

---

## 3. Method Overview

```
AO-GPT (random block order training)
        ↓
Extract per-head attention graph B (64×64, M samples)
        ↓
CDL Teacher: greedy C−D decoding → pseudo-order labels
        ↓
Train g_β: lightweight MLP readout B → order weights
        ↓
Frozen Hook: g_β produces order for AO-GPT forward pass
        (no CDL at deployment time; K-step periodic refresh)
```

**Key properties**:
- g_β is trained offline once per (seed, head) pair
- At deployment, only a forward pass through g_β is needed — no iterative decoding
- g_β@10k transfers to later checkpoints within the same seed

---

## 4. Clean-Permutation Protocol

- Train AO-GPT with **random block permutation** (fixed per training run, different from L2R)
- Physical block indices NEVER enter the model forward pass
- B matrices are re-mapped from model coordinates to physical coordinates **only for offline analysis**
- eval metric: `val_ori_l2r_block` — model evaluated with L2R-ordered blocks (same parameters, different block permutation)

---

## 5. CDL Teacher

CDL (Column-Difference-Local) greedy decoding of the attention graph B produces a block traversal order. For a head with strong L2R alignment (τ≈+1.0), CDL output ≈ L2R order.

CDL teacher is used ONLY for:
- Training g_β (producing pseudo-order labels)
- Offline head selection diagnostics

CDL is NOT deployed in the training loop (g_β replaces it).

Component ablation (from `memory/cdl_teacher_ablation_results.md`): −D (column difference) is the load-bearing term; C (column sum) and L (local) are redundant for text.

---

## 6. g_β Learned Readout

g_β is a lightweight MLP that maps B → order weights. Trained with CDL teacher supervision.

**Input sanity** (Table A): g_β output τ=0.97 for real B, ≈0 for Gaussian/shuffled/destroyed B. Cannot be reduced to "memorizing L2R prior."

**g_β is seed-dependent and head-dependent.** Different seeds/heads require different g_β. The pipeline selects the best head per seed using cheap diagnostic signals (row-concentration, S_forward).

---

## 7. Frozen Hook

At deployment: g_β produces block order weights → AO-GPT forward pass uses these weights. g_β is refreshed every K steps (K=10 in current experiments).

The model continues training with α ramp (0→1 over warmup) to gradually shift from random order to g_β-guided order.

---

## 8. Main Results

### 8.1 Training Acceleration

| Metric | Value | Source |
|--------|:--:|------|
| Step Saving @3.47 (frozen_β, from10k) | **41.7%** (seed2), **39.3%** (seed42) | Table D |
| Step Saving @3.47 (CDL teacher, from10k) | **47.6%** | Table D |
| Recovery @50k (frozen_β, from10k seed2) | **86.0%** | Table C |
| Recovery @50k (frozen_β, from10k seed42) | **106.1%** (beats L2R ref) | Table C |
| Catch-up gap @50k (seed2) | within **0.020** of ori-L2R reference | Table E |

Conservative claim: **35–42% step saving** across 2 seeds, from10k+from20k. Seed42 fixed-step Recovery has been verified with matched baseline group audit (random=3.466, L2R=3.341, gap=0.125).

### 8.2 Mechanism

| B source | g_β τ | Verdict |
|----------|:--:|------|
| Real B | 0.97 | reads structure |
| Gaussian/shuffled/destroyed | ≈0 | no signal |
| Non-L2R subset | +0.11 Δ | B-dependent |

### 8.2 Attention Emergence

**Per-head CDL τ = ±1.000** for multiple individually-selected heads across all training granularities (32/64/128 groups). Signal strongest in layers 0–2. Selected order-bearing heads are sparse and seed-dependent — head audition is a necessary design component, not an afterthought.

Small model (47M): many order-bearing heads (28/32 with |τ|>0.9). Large model (317M): sparse order-bearing heads (6/256 with |τ|>0.9, best τ=+0.955).

Historical note: aggregate/CDL-rollout diagnostics give lower but positive numbers (e.g., τ≈0.49), but the method pipeline uses per-head selected graphs B^{l,h} as input, so per-head |τ|≈1.0 is the primary signal evidence.

---

## 9. Robustness

| Ablation | Result |
|------|------|
| 32/64/128 block aggregation | τ > 0.96 |
| 32/64/128 training granularity | per-head τ = ±1.0 |
| 317M model (16L/16H/1024d) | |τ|>0.9 heads exist |

---

## 10. Limitations

1. **g_β is seed/head-dependent.** Not a universal controller. Pipeline selects best head per seed.
2. **ori-L2R is reference, not upper bound.** Some methods beat it (CDL teacher Recovery > 100%). The absolute ceiling is unknown.
3. **Label-free head selection** uses cheap diagnostics (row-concentration) but has not been end-to-end validated.
4. **Image side** has diagnostic evidence only (D_manh locality, CDL teacher ablation). No frozen hook training on image models.
5. **317M** is diagnostic only (τ>0.9 confirmed, no hook training).
6. **Per-sample permutation** not tested — current protocol uses fixed permutation per batch.

---

## 11. Next Experiments (if requested)

| Priority | Experiment | Cost | Risk |
|:--:|------|:--:|:--:|
| P0 | Cross-model B sanity (g_β on shuffled-L2R B) | Low | Low |
| P1 | Third seed | Medium | Low |
| P2 | Label-free audition end-to-end | Medium | Medium |
| P3 | 317M frozen hook | High | High |
| — | Image teacher redesign | High | High |
