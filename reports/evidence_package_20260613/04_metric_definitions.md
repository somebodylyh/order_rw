# 04 — Metric Definitions

> All definitions apply to text-side AO-GPT / Block-LO-ARM / frozen g_β experiments.
> Wikitext-103, 4L/8H/384d, block_size=256, block_order_block_len=4, num_blocks=64.

---

## 1. val_ori_l2r_block

**Definition**: Validation loss when blocks are ordered in original (physical) L2R order.
The model receives blocks ordered by their physical position in the document, i.e., block 0, block 1, ..., block 63.

**Column name in eval_curve.tsv**: `val_ori_l2r_block`

**This is the PRIMARY METRIC used in all main tables (Recovery, Step Saving, Catch-up).**

**Rationale**: This metric measures how well the model performs under the ideal L2R block ordering, decoupled from the training-time order. Same model parameters, same data, only block permutation differs.

**Aliases found**: `val_ori_l2r` (used interchangeably in some logs — same value as `val_ori_l2r_block` in all verified cases). If discrepancy found, `val_ori_l2r_block` takes precedence.

---

## 2. Fixed-step Recovery

```
Recovery_t = (L_random,t − L_method,t) / (L_random,t − L_L2R,t) × 100%
```

Where:
- `L_random,t`: val_ori_l2r_block of the random-order baseline at global step t
- `L_method,t`: val_ori_l2r_block of the method (frozen_β / CDL teacher) at global step t
- `L_L2R,t`: val_ori_l2r_block of the ori-L2R reference at global step t

**Interpretation**:
- 0% = random baseline (no recovery)
- 100% = ori-L2R reference (full recovery to reference level)
- \>100% = method achieves LOWER loss than ori-L2R reference at same step (method's order is better than pure L2R for this model)

**Critical constraints**:
- All three values MUST be at the same global step t
- All three MUST be from the same seed/baseline group
- Current standard: t = 50000 (step 50k)
- DO NOT mix method@60k with baseline@50k

**Recovery is a quality metric, NOT a speed metric.**

---

## 3. Step Saving

```
Saving(L^*) = (T_random(L^*) − T_method(L^*)) / T_random(L^*) × 100%
```

Where:
- L^* is a target val_ori_l2r_block threshold (currently L^* = 3.47)
- T_random(L^*) is the first training step where random baseline reaches val_ori_l2r_block ≤ L^*
- T_method(L^*) is the first training step where method reaches val_ori_l2r_block ≤ L^*

**Interpretation**: Percentage of training steps saved to reach the same L2R-evaluated loss level.

**Critical constraints**:
- If method starts at a step where val_ori_l2r_block is already ≤ L^*, label as "N/A (already below threshold)"
- If baseline never reaches L^*, label as "N/A (baseline never reaches)"
- Step values are from eval checkpoints (every 1000 steps). Linear interpolation between checkpoints is acceptable if noted.
- Current threshold L^* = 3.47 was chosen as a level all seed2 methods reach.

**Step Saving is a speed metric.**

---

## 4. Kendall τ (τ_vs_L2R)

**Definition**: Kendall rank correlation between the block order produced by a head's B matrix (via CDL greedy decoding) and the physical L2R order [0, 1, 2, ..., 63].

τ ∈ [−1, +1]:
- +1: head's CDL order matches L2R exactly
- −1: head's CDL order is exactly reversed L2R
- 0: no correlation

**Computed by**: `per_head_order_scan.py` → `extract_per_head_and_heavy_A()` → CDL greedy → Kendall τ vs [0..63]

**Critical notes**:
- τ is computed per-head (32 heads for 4L/8H), not aggregated unless stated
- "heavy τ" = weighted average across all 32 heads → **DO NOT USE as primary metric** (positive and negative heads cancel)
- "per-head τ" = τ for a single (layer, head) → **this is the primary head-level metric**
- |τ| > 0.9 is considered "strong signal"
- τ = 1.0 for zero B is a **tie-breaking artifact** (all edges equal → CDL picks first available = L2R), NOT evidence for L2R prior

**Related metrics**:
- `mean_pairwise_tau`: average pairwise τ between σ from different samples (M=100). Measures cross-sample consistency of a single head's CDL output.
- `first_step_entropy`: entropy of the first CDL step across samples. H=0 means all samples pick the same first block.
- `unique_sigma_ratio`: fraction of M samples producing unique σ. High = diverse, low = all converge to same order.

---

## 5. g_β τ_vs_L2R (readout metric)

**Definition**: Kendall τ between the order output by the g_β learned readout (given B as input) and the physical L2R order.

This is DIFFERENT from the CDL τ_vs_L2R:
- CDL τ: measures what CDL greedy decoding produces from raw B
- g_β τ: measures what the learned g_β network outputs given B as input

**Used in**: Mechanism sanity table (Table A). Tests whether g_β reads B structure or outputs a constant L2R.

---

## 6. Other eval modes

| Column in eval_curve.tsv | Meaning | Used in main tables? |
|------|------|:--:|
| `val_ori_l2r_block` | Model evaluated with L2R block order | ✅ PRIMARY |
| `val_train_objective` | Model evaluated with its training-time block order | ❌ |
| `val_model_order` | Model evaluated with its own produced order | ❌ |
| `val_unstructured_order` | Model evaluated with unstructured (bag-of-blocks) | ❌ |
| `val_rw_order` | Model evaluated with graph-RW produced order | ❌ |
| `val_beta_order` | Model evaluated with frozen_β output order | ❌ |
| `val_cdl_order` | Model evaluated with CDL teacher output order | ❌ |
| `val_ar_l2r` | Autoregressive L2R evaluation | ❌ |

---

## 7. Mapping / disambiguation

| If you see | It means | Check |
|------|------|------|
| `val_ori_l2r` | Same as `val_ori_l2r_block` | Verify in raw TSV header |
| "recovery" without step | Assume @50k unless stated | Verify source |
| "step saving" without threshold | Assume @3.47 unless stated | Verify source |
| "τ" without qualifier | Per-head CDL τ_vs_L2R unless stated | Verify source |
| "heavy τ" | Weighted average across all heads | DO NOT use alone |
| "random τ=0.49" | Likely from old diagnostic (CDL shuffle control) | Verify source |
| "CDL τ≈1.0" | Per-head τ for selected best head | Verify which head |

---

## 8. ori-L2R: reference, NOT upper bound

ori-L2R is a **reference point**, not a proven upper bound. The model could in principle discover orders better than L2R for its own loss landscape.

Evidence: CDL teacher from10k achieves Recovery > 100% at step 50k (meaning its produced order gives lower loss than pure L2R order for this model).

In all tables and claims: say "ori-L2R reference", never "ori-L2R upper bound".
