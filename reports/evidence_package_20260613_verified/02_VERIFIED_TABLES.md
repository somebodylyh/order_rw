# 02 — Verified Tables

> Only numbers verified from primary sources (config.json, eval_curve.tsv, head_scan JSON).
> Source column points to exact file and step for traceability.

---

## Table V1: Run Group Identity

| Canonical group | seed | Random baseline | L_random @50k | L2R ref | L_L2R @50k | Gap |
|------|:--:|------|:--:|------|:--:|:--:|
| seed-123 | 123 | `random_baseline_continuous_jun08_seed2` | 3.445 | `l2r_continuous_seed123` | 3.305 | 0.140 |
| seed-42 | 42 | `random_baseline_continuous_jun05` | 3.466 | `l2r_continuous_jun05` | 3.341 | 0.125 |
| seed-2 | 2 | `random_baseline_continuous_jun11_seed2_headtrack` | 3.413 | `l2r_continuous_seed2` | 3.370 | 0.043 |

**Source**: config.json from each run directory. All use `data_source=continuous`, 4L/8H/384d, block_size=256, 64 blocks.

**Naming convention**: The directory name "seed2" is ambiguous. seed-123 group = `*_seed2_*` (frozen_beta); seed-2 group = `*_seed2_*` (CDL teacher, `l2r_continuous_seed2`). Always verify via config.json `seed` field.

---

## Table V2: Metric Mapping

| Column in eval_curve.tsv | Column index | Meaning | Primary metric? |
|------|:--:|------|:--:|
| val_ori_l2r_block | 5 | Model eval with L2R block order | ✅ YES |
| val_ar_l2r | 6 | Autoregressive L2R eval | ❌ (identical in all verified cases) |
| val_train_objective | 4 | Model eval with training order | ❌ |
| val_model_order | 7 | Model eval with own produced order | ❌ |
| val_unstructured_order | 8 | Model eval with bag-of-blocks | ❌ |
| val_rw_order | 9 | Model eval with graph-RW order | ❌ |
| val_beta_order | 10 | Model eval with frozen_β order | ❌ |

**Source**: `eval_curve.tsv` headers from all key runs. All use `val_ori_l2r_block` consistently. ✅

---

## Table V3: Fixed-step Recovery @50k

Recovery = (L_random − L_method) / (L_random − L_L2R) × 100%

### Seed-123 group (frozen_beta seed2-like)

| Method | Resume | L_method @50k | Δ vs random | Recovery | Source |
|------|:--:|:--:|:--:|:--:|------|
| random baseline | — | 3.445 | 0.000 | 0% (def) | `random_baseline_continuous_jun08_seed2/eval_curve.tsv` |
| frozen_β L0H2 | from10k | 3.325 | 0.120 | **86.0%** | `frozen_beta_seed2_from10k_l0h2/eval_curve.tsv` |
| frozen_β L0H2 | from20k | 3.329 | 0.116 | **83.0%** | `frozen_beta_seed2_from20k_l0h2/eval_curve.tsv` |
| frozen_β L0H2 | from40k | 3.363 | 0.082 | **58.7%** | `frozen_beta_seed2_from40k_l0h2/eval_curve.tsv` |
| ori-L2R reference | — | 3.305 | 0.140 | 100% (def) | `l2r_continuous_seed123/eval_curve.tsv` |

### Seed-42 group

| Method | Resume | L_method @50k | Δ vs random | Recovery | Source |
|------|:--:|:--:|:--:|:--:|------|
| random baseline | — | 3.466 | 0.000 | 0% (def) | `random_baseline_continuous_jun05/eval_curve.tsv` |
| frozen_β L0H4 | from10k | 3.334 | 0.132 | **106.1%** | `frozen_beta_random_jun05_from10k/eval_curve.tsv` |
| frozen_β L0H4 | from20k | 3.339 | 0.127 | **101.5%** | `frozen_beta_random_jun05_from20k/eval_curve.tsv` |
| frozen_β L0H4 | from40k | 3.383 | 0.084 | **66.9%** | `frozen_beta_random_jun05_from40k/eval_curve.tsv` |
| ori-L2R reference | — | 3.341 | 0.125 | 100% (def) | `l2r_continuous_jun05/eval_curve.tsv` |

### Seed-2 group (CDL teacher, supplementary)

| Method | Resume | L_method @50k | Δ vs random | Recovery | Source |
|------|:--:|:--:|:--:|:--:|------|
| random baseline | — | 3.413 | 0.000 | 0% (def) | `random_baseline_continuous_jun11_seed2_headtrack/eval_curve.tsv` |
| CDL teacher L0H2 | from10k | 3.284 | 0.129 | **302%** | `cdl_teacher_seed2_from10k_l0h2/eval_curve.tsv` |
| ori-L2R reference | — | 3.370 | 0.043 | 100% (def) | `l2r_continuous_seed2/eval_curve.tsv` |

⚠️ Seed-2 group gap (0.043) is very narrow — Recovery% highly sensitive. CDL teacher best interpreted as supplementary evidence.

### Cross-seed CDL teacher (in seed-123 group, for reference)

| Method | L_method @50k | Δ vs seed-123 random | Recovery vs seed-123 gap | Note |
|------|:--:|:--:|:--:|------|
| CDL teacher from10k L0H2 | 3.284 | 0.161 | 115.6% | ⚠️ cross-seed (CDL=seed2, baseline=seed123) |
| CDL teacher from20k L0H2 | 3.308 | 0.137 | 98.2% | ⚠️ cross-seed |
| CDL teacher from40k L0H2 | 3.353 | 0.092 | 65.7% | ⚠️ cross-seed |

---

## Table V4: Step Saving @3.47

Saving = (T_random − T_method) / T_random × 100%. T_random = 42000 (seed-123). Crossing = first eval checkpoint ≤ 3.47.

| Group | Method | Resume | T_cross | val @cross | Saving | Source |
|------|------|:--:|:--:|:--:|:--:|------|
| seed-123 | ori-L2R ref | — | 15000 | 3.465 | 64.3% | `l2r_continuous_seed123/eval_curve.tsv` |
| seed-2 | CDL teacher L0H2 | from10k | 22000 | 3.467 | 47.6% | `cdl_teacher_seed2_from10k_l0h2/eval_curve.tsv` |
| seed-123 | **frozen_β L0H2** | **from10k** | **24500** | 3.469 | **41.7%** | `frozen_beta_seed2_from10k_l0h2/eval_curve.tsv` |
| seed-42 | frozen_β L0H4 | from10k | 25500 | 3.466 | 39.3% | `frozen_beta_random_jun05_from10k/eval_curve.tsv` |
| seed-2 | CDL teacher L0H2 | from20k | 26500 | 3.467 | 36.9% | `cdl_teacher_seed2_from20k_l0h2/eval_curve.tsv` |
| seed-123 | **frozen_β L0H2** | **from20k** | **27000** | 3.463 | **35.7%** | `frozen_beta_seed2_from20k_l0h2/eval_curve.tsv` |
| seed-42 | frozen_β L0H4 | from20k | 28000 | 3.467 | 33.3% | `frozen_beta_random_jun05_from20k/eval_curve.tsv` |
| seed-123 | frozen_β L0H2 | from40k¹ | 40500 | 3.464 | 3.6% | starts below 3.47 |
| seed-2 | CDL teacher L0H2 | from40k¹ | 40500 | 3.463 | 3.6% | starts below 3.47 |

¹ from40k methods start with val_ori_l2r_block already ≤ 3.47 (training resumes below threshold). Step saving not meaningful.

**Conservative range** (frozen_β only, from10k+from20k, seed-123+seed-42): **33–42%**.

---

## Table V5: g_β Mechanism Sanity

| B input | τ vs L2R | Interpretation | Source |
|------|:--:|------|------|
| Real B (L0H4, random-order model) | **0.97** | g_β reads B structure ✓ | `analyses/gbeta_input_sanity_final.py` + memory |
| Gaussian B (matched μ,σ) | **≈0** | No signal from random B ✗ | same |
| Entry-shuffled B | **≈0** | No signal from destroyed B ✗ | same |
| Row+col shuffled B | **≈0** | No signal from destroyed B ✗ | same |
| Zero B | **1.0** | Tie-breaking artifact (margin=0, CDL picks L2R by default) | same |
| Gaussian family pairwise | **0.0003** | No fixed L2R prior across diverse inputs ✗ | `raw/gbeta_input_sanity_final.json` |
| Non-L2R subset | **inconclusive** (n=4) | Excluded from main claim — insufficient sample size | same |

**Primary evidence**: Real B (τ=+0.9675) vs destroyed B (τ≈0) — verified from raw JSON (2026-06-14). The Gaussian pairwise τ=0.0003 confirms no fixed L2R prior. Zero B τ=1.0 is a tie-breaking artifact (margin=0).

**Excluded**: Non-L2R subset +0.11 Δ from memory summaries — not reproducible. n=4 samples, Δ=-0.03 in re-run. Excluded from all claims.

---

## Table V6: Per-Head Order-Bearing Signal

| Experiment | Step | Heads | \|τ\|>0.9 | Best +τ | Best −τ | Heavy τ | Source |
|------|:--:|:--:|:--:|------|------|:--:|------|
| clean_base_random_perm (64blk) | 10k | 32 | **9** | L0H0 +1.000 | L1H6 −0.938 | 0.876 | `clean_base_random_perm/head_scan_10k.json` |
| shuffle_gran_32 | 10k | 32 | **5** | L0H0 +1.000 | L2H0 −0.938 | −0.773 | `shuffle_gran_32/head_scan_10k.json` |
| shuffle_gran_128 | 10k | 32 | **8** | L1H2 +1.000 | L2H3 −0.938 | 0.305 | `shuffle_gran_128/head_scan_10k.json` |
| 317M scale diagnostic | 5k | 256 | **6** | L0H10 +0.955 | L1H3 −0.938 | 0.037 | `large_random_baseline_16l16h1024d/head_scan_step5000.json` |
| **seed-123 continuous @50k** | **50k** | **32** | **0** | L2H6 +0.034 | L1H7 −0.050 | **−0.013** | `random_baseline_continuous_jun08_seed2/head_scan_50k.json` |

⚠️ **50k finding (2026-06-14, M=40, none_mode=b0)**: Per-head CDL τ_vs_L2R decays to noise level by step 50k (all 32 heads \|τ\|<0.06, within noise floor). The strong per-head L2R signal observed at 10k (\|τ\|≈1.0) is a **transient early-training phenomenon**, not a persistent convergent property. This does NOT invalidate the training acceleration claim (C4: g_β@10k→40k transfer shows 59–67% recovery), but it shifts the evidence weight for signal persistence from direct τ measurement to the C4 transfer results. **The frozen hook works at later steps, but the mechanism of its effectiveness at convergence may be different from the clean L2R-readout mechanism observed at 10k.**

**Selected heads for deployment**: seed-123 = L0H2, seed-42 = L0H4. These are NOT the max-|τ| heads; they are selected via audition (row-concentration + CDL teacher quality).

---

## Table V7: Granularity Robustness

### Aggregation granularity (same model, different B aggregation)

| Granularity | Heavy τ | unique σ | Source |
|:--:|:--:|:--:|------|
| 32blk × 8tok | 0.9997 | 0.02 | `analyses/block_granularity_scan_results/scan_step5000_M100.json` |
| 64blk × 4tok | 0.9999 | 0.02 | same |
| 128blk × 2tok | 1.0000 | 0.01 | same |

### Training granularity (different shuffle_granularity at train time)

| Granularity | Best +τ | Best −τ | \|τ\|>0.9 count | Source |
|:--:|------|------|:--:|------|
| 32 groups | L0H0 +1.000 | L2H0 −0.938 | 5 | `shuffle_gran_32/head_scan_10k.json` |
| 64 groups | L0H0 +1.000 | L1H6 −0.938 | 9 | `clean_base_random_perm/head_scan_10k.json` |
| 128 groups | L1H2 +1.000 | L2H3 −0.938 | 8 | `shuffle_gran_128/head_scan_10k.json` |

---

## Table V8: 317M Scale Diagnostic

| Property | Value | Source |
|------|------|------|
| Model config | 16L/16H/1024d | `large_random_baseline_16l16h1024d/config.json` |
| Params estimate | ~317M | from config |
| Total heads | 256 | 16 × 16 |
| Training step | 5000 (diagnostic scan) | config.json max_steps=10000 |
| \|τ\|>0.9 heads | 6 (2.3%) | `head_scan_step5000.json` |
| Best head | L0H10 τ=+0.955 | same |
| Heavy τ | 0.037 (near zero) | same |
| Frozen hook training | **NONE** | No frozen_beta run with 16L/16H config found |

---

## Table V9: Order Specialization Trade-off (val_unstructured_order)

> **Rationale**: Our method specializes the model toward a single deployment order (L2R-like canonical order). As expected, this improves `val_ori_l2r_block` but degrades `val_unstructured_order` (random-permutation evaluation). This is an order-specialization trade-off, NOT a hidden failure. The model is NOT claimed to improve under all permutations — only under the chosen canonical order.

Recovery_unstructured = (L_random − L_method) / (L_random − L_L2R) × 100%, using `val_unstructured_order` instead of `val_ori_l2r_block`.

### Seed-123 group

| Method | val_ori_l2r @50k | val_unstructured @50k | Δ_unstructured vs random | Recovery_unstructured |
|------|:--:|:--:|:--:|:--:|
| random baseline | 3.445 | 3.637 | 0.000 | 0% (def) |
| frozen_β from10k L0H2 | 3.325 | 4.061 | **+0.424** | −12.1% |
| frozen_β from20k L0H2 | 3.329 | 3.981 | +0.344 | −9.8% |
| frozen_β from40k L0H2 | 3.363 | 3.708 | +0.071 | −2.0% |
| ori-L2R reference | 3.305 | 7.149 | +3.512 | −100% (def) |

### Seed-42 group

| Method | val_ori_l2r @50k | val_unstructured @50k | Δ_unstructured vs random | Recovery_unstructured |
|------|:--:|:--:|:--:|:--:|
| random baseline | 3.466 | 3.658 | 0.000 | 0% (def) |
| frozen_β from10k L0H4 | 3.334 | 4.941 | **+1.283** | −38.1% |
| frozen_β from20k L0H4 | 3.339 | 4.441 | +0.782 | −23.2% |
| frozen_β from40k L0H4 | 3.383 | 3.728 | +0.069 | −2.1% |
| ori-L2R reference | 3.341 | 7.025 | +3.367 | −100% (def) |

### Seed-2 group (CDL teacher)

| Method | val_ori_l2r @50k | val_unstructured @50k | Δ_unstructured vs random | Recovery_unstructured |
|------|:--:|:--:|:--:|:--:|
| random baseline | 3.413 | 3.603 | 0.000 | 0% (def) |
| CDL teacher from10k L0H2 | 3.284 | 3.818 | +0.215 | −6.4% |
| CDL teacher from20k L0H2 | 3.308 | 3.768 | +0.164 | −4.9% |
| ori-L2R reference | 3.370 | 6.966 | +3.363 | −100% (def) |

**Key observations**:
1. All methods degrade on `val_unstructured_order` — the degradation is proportional to the specialization strength on `val_ori_l2r_block`.
2. CDL teacher degrades less (Δ +0.21) than frozen_β seed-123 (Δ +0.42) and seed-42 (Δ +1.28), suggesting CDL produces orders that preserve more random-order capability.
3. ori-L2R models degrade severely (Δ +3.4–3.5), confirming that training with a fixed order produces extreme specialization.
4. **This is expected and acceptable**: inference uses a single order; the method is "order specialization for faster training," not "permutation-robust likelihood improvement."

**Framing for paper**:
> Our controller specializes the model toward a single deployment order. As expected, this improves the target canonical-order validation loss but degrades performance under arbitrary random evaluation orders. We therefore treat this as an order-specialization trade-off rather than a universal likelihood improvement over all permutations. The model is not claimed to improve under all permutations — only under the chosen canonical order.
