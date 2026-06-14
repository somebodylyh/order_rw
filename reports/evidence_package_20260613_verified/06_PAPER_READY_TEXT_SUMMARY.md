# 06 — Paper-Ready Text-Side Summary

> Based on verified evidence package (2026-06-14).
> All numbers from primary sources (eval_curve.tsv, head_scan JSON, config.json).

---

## 1. Problem

Training any-order autoregressive models (AO-GPT) requires choosing an input token order. Random order is the simplest choice but ignores data structure. We investigate whether AO-GPT attention graphs, even under random-order training, spontaneously encode the data's intrinsic sequential structure — and whether this signal can be distilled into a training accelerator without retraining the base model.

---

## 2. Method

**Three-stage pipeline**:

1. **Per-head signal extraction**: From a random-order-trained AO-GPT checkpoint, extract B^{l,h} (attention graph) for each (layer, head) pair. A cheap diagnostic (row-concentration) selects the most order-bearing head without using oracle labels.

2. **g_β distilled readout**: A lightweight MLP (g_β) is trained to map B^{l,h} → CDL teacher order (greedy C−D+L decoding on B). g_β is trained once at step 10k and frozen thereafter.

3. **Frozen hook training**: At a later step (10k / 20k / 40k), the frozen g_β is plugged into the AO-GPT training loop as an order controller. Input block order at each step is determined by g_β(B^{l,h}) via CDL greedy decoding. The base model continues normal training — only the block permutation is changed.

**Clean permutation protocol**: A single random block permutation is generated before training and held fixed for all samples. No per-sample variation. This isolates the effect of order quality from per-sample adaptation.

**Model**: 4-layer, 8-head, d=384 AO-GPT (47M params). Wikitext-103, block_size=256, 64 blocks × 4 tokens.

---

## 3. Per-Head Order-Bearing Signal

**Finding**: Under random-order training (α=0, continuous data loading), 9 out of 32 attention heads develop strong alignment with physical L2R order (|Kendall τ| > 0.9) by step 10k.

- Best head: L0H0 τ=+1.000 (all 100 eval samples produce identical L2R order)
- Signal is **sparse** — median |τ|=0.290; 22 positive, 10 negative heads
- Heavy τ (weighted average) = 0.876, dominated by a few strong heads — not a reliable primary metric
- Negative heads exist (L1H6 τ=−0.938, L2H3 τ=−0.938) — encode reverse-L2R

**Implication**: The model spontaneously discovers L2R-like structure without being told. But only a minority of heads do so — head selection is essential.

---

## 4. g_β Reads B Structure, Not a Fixed Prior

**Sanity validation**: Feed g_β with different B inputs and measure output order's τ vs L2R:

| B input | τ vs L2R | Interpretation |
|------|:--:|------|
| Real B (random-order model) | 0.97 | Reads structure |
| Gaussian B (matched μ,σ) | ≈0 | No signal from noise |
| Entry-shuffled B | ≈0 | Structure-dependent |
| Row+col shuffled B | ≈0 | Structure-dependent |
| Zero B | 1.0 | Tie-breaking artifact (margin=0) |
| Non-L2R subset vs prior | +0.11 Δ | B-dependent beyond L2R |

**Implication**: If g_β had memorized a constant L2R, all inputs would give τ≈1.0. Destroyed B gives τ≈0. Zero B τ=1.0 is a CDL tie-breaking artifact (all edges equal → picks first = L2R). g_β genuinely reads graph structure.

---

## 5. Frozen Hook Training Results

### Recovery @50k (Fixed-step quality metric)

| Seed | Head | from10k Recovery | from20k | from40k |
|------|------|:--:|:--:|:--:|
| seed-123 | L0H2 | **86.0%** | **83.0%** | 58.7% |
| seed-42 | L0H4 | **106.1%** | **101.5%** | 66.9% |

Recovery > 100% means the method's order gives lower loss than pure L2R at the same step.

### Step Saving @3.47 (Speed metric)

| Method | Step to 3.47 | Saving |
|------|:--:|:--:|
| frozen_β seed-123 from10k L0H2 | 24500 | **41.7%** |
| frozen_β seed-42 from10k L0H4 | 25500 | **39.3%** |
| frozen_β seed-123 from20k L0H2 | 27000 | **35.7%** |
| frozen_β seed-42 from20k L0H4 | 28000 | **33.3%** |

**Conservative range**: **33–42% step saving** across 2 seeds, from10k+from20k.

### CDL Teacher (supplementary)

CDL teacher bypasses g_β — directly uses CDL greedy on B. Achieves 22000 steps to 3.47 (47.6% saving) and 115.6% cross-seed Recovery. ⚠️ CDL teacher uses a different model seed (seed=2) than the frozen_β group (seed=123) — cross-seed comparison.

---

## 6. Robustness

### Not a 64-block artifact
- Same model with attention aggregated at 32/64/128 blocks: heavy τ > 0.96 in all cases
- Models trained with 32/64/128 shuffle granularity: per-head |τ|≈1.0 in all cases

### Not a 47M-scale artifact
- 317M model (16L/16H/1024d): 6/256 heads with |τ|>0.9 at step 5k. Best L0H10 τ=+0.955.
- Signal exists at scale, but is sparser (2.3% of heads vs 28% in small model)
- ⚠️ No frozen hook training run at 317M — diagnostic only

---

## 7. Limitations

1. **Seed-dependence**: g_β and selected head are seed-specific. No demonstrated cross-seed transfer.
2. **g_β training needs teacher**: Current g_β uses CDL teacher supervision. End-to-end label-free pipeline not shown.
3. **from40k weak**: Methods resuming from 40k show marginal benefit (model near convergence).
4. **Text only**: Image side has diagnostic results but no frozen hook training.
5. **Fixed permutation**: Per-sample order adaptation not explored.
6. **Single data source**: Wikitext-103 only. No cross-dataset validation.

---

## 8. Next Experiments (requires decision)

1. **Third seed** for statistical completeness (cheapest)
2. **317M frozen hook** for scale demonstration (expensive, risky)
3. **Seed-123 CDL teacher** for matched-group comparison with frozen_β
4. **End-to-end label-free pipeline** (audition head → unsupervised g_β → hook)
5. **Image frozen hook** extension
6. **Cross-dataset** validation
