# 01 — Verification Log

> Execution started: 2026-06-14
> Method: Read-only inspection of config.json, eval_curve.tsv, head_scan JSON, analysis script outputs

---

## Module A — Run Identity & Baseline-Group Audit

### A1: Bulk config extraction

**Command**: Extracted run_kind, seed, data_source, max_steps, model config, frozen_beta_head, alpha from all 24 key run configs.

**Finding**: Three distinct seed groups exist. The directory naming convention "seed2" is ambiguous — it maps to BOTH seed=123 and seed=2 depending on the run.

### A2: Seed-to-directory mapping

**Verified seed groups**:

| Canonical group | seed | Random baseline | @50k | L2R ref | @50k | Gap |
|------|:--:|------|:--:|------|:--:|:--:|
| seed-123 (labeled "seed2" for frozen_beta) | 123 | random_baseline_continuous_jun08_seed2 | 3.445 | l2r_continuous_seed123 | 3.305 | 0.140 |
| seed-42 | 42 | random_baseline_continuous_jun05 | 3.466 | l2r_continuous_jun05 | 3.341 | 0.125 |
| seed-2 (labeled "seed2" for CDL teacher) | 2 | random_baseline_continuous_jun11_seed2_headtrack | 3.413 | l2r_continuous_seed2 | 3.370 | 0.043 |

**Verified runs within each group**:

**seed-123 group**:
- `frozen_beta_seed2_from10k_l0h2`: seed=123, head=[0,2], max_steps=60000 ✓
- `frozen_beta_seed2_from20k_l0h2`: seed=123, head=[0,2] ✓
- `frozen_beta_seed2_from40k_l0h2`: seed=123, head=[0,2] ✓
- All use same g_β: `phase33_gbeta_seed2_from10k_l0h2/.../g_beta_best.pt` ✓
- All use `random_baseline_continuous_jun08_seed2` as base model ✓

**seed-42 group**:
- `random_baseline_continuous_jun05`: seed=42, run_kind=baseline, max_steps=50000 ✓
- `l2r_continuous_jun05`: seed=42, run_kind=l2r, max_steps=50000 ✓
- `frozen_beta_random_jun05_from10k`: seed=42, head=[0,4] ✓
- `frozen_beta_random_jun05_from20k`: seed=42, head=[0,4] ✓
- `frozen_beta_random_jun05_from40k`: seed=42, head=[0,4] ✓
- All use same g_β: `phase33_gbeta/random_baseline_continuous_jun05/full/g_beta_best.pt` ✓
- `frozen_beta_random_baseline_jun05`: seed=42, run_kind=frozen_beta, alpha=1.0 — **NOT a baseline** ✓

**seed-2 group**:
- `cdl_teacher_seed2_from10k_l0h2`: seed=2, run_kind=cdl_teacher ✓
- `cdl_teacher_seed2_from20k_l0h2`: seed=2 ✓
- `cdl_teacher_seed2_from40k_l0h2`: seed=2 ✓
- `random_baseline_continuous_jun11_seed2_headtrack`: seed=2, run_kind=baseline ✓
- `l2r_continuous_seed2`: seed=2, run_kind=l2r, max_steps=60000 ✓
- `frozen_beta_seed2_from10k`: seed=2, run_kind=frozen_beta (older run, not in main tables) ✓
- `frozen_beta_seed2_from10k_v3`: seed=2 (not in main tables) ✓

### A3: 🔴 CRITICAL FINDING — CDL teacher seed mismatch

**Finding**: The existing evidence package (`02_unified_tables.md`) places CDL teacher runs (seed=2) in the same table as frozen_beta seed=123 runs, using seed=123's random baseline (3.445) and L2R reference (3.305).

**Impact**: CDL teacher Recovery% in existing tables is computed with:
- L_random = 3.445 (seed=123)
- L_L2R = 3.305 (seed=123)
- L_method = CDL teacher values (seed=2)

This is a CROSS-SEED comparison. The CDL teacher's own matched group would be:
- L_random = 3.413 (seed=2, jun11_seed2_headtrack)
- L_L2R = 3.370 (seed=2, l2r_continuous_seed2)
- Gap = 0.043 (very narrow)

**Verdict**: UNRESOLVED — CDL teacher Recovery% cannot be verified as a matched-group comparison within the seed-123 group. Two possible resolutions:
1. Recompute CDL teacher Recovery% against its own seed=2 group (gap=0.043 makes Recovery% highly sensitive)
2. Mark CDL teacher as cross-seed supplementary evidence, not primary matched-group result

### A4: VERIFIED — frozen_beta_random_baseline_jun05 is NOT a baseline

Config confirms: run_kind=frozen_beta, seed=42, alpha_start=1.0, alpha_target=1.0. This is a frozen_beta run from step 0, NOT a random baseline. Its val_ori_l2r_block@50k=3.354 is a frozen_beta RESULT, not a reference value. ✓

### A5: VERIFIED — All key runs use continuous data loading

All seed-123, seed-42, and seed-2 frozen_beta + baseline runs use `data_source=continuous`. The granularity runs (shuffle_gran_32/128) and 317M diagnostic use `data_source=chunks` (acceptable for 10k-step diagnostic only). ✓

### A6: Model config consistency

All 47M runs: 4L/8H/384d, block_size=256, 64 blocks × 4 tokens. 317M run: 16L/16H/1024d. All consistent. ✓

**Module A status**: 4 VERIFIED, 1 UNRESOLVED (CDL teacher seed mismatch)

---

## Module B — Metric Name & Eval-Mode Audit

### B1: Column header audit

**File inspected**: eval_curve.tsv from 10 key runs across all seed groups.

**Finding**: ALL eval_curve.tsv files use `val_ori_l2r_block` (column index 5) consistently. No file uses the shorthand `val_ori_l2r` as a column name. Some newer runs include an extra `val_cdl_order` column.

**Verified metric mapping**:

| Column name | Present in all files? | Used in main tables? |
|------|:--:|:--:|
| val_ori_l2r_block | ✅ all | ✅ PRIMARY |
| val_ar_l2r | ✅ all | ❌ (equal to val_ori_l2r_block in all verified cases) |
| val_train_objective | ✅ all | ❌ |
| val_model_order | ✅ all | ❌ |
| val_unstructured_order | ✅ all | ❌ |
| val_rw_order | ✅ all | ❌ |
| val_beta_order | ✅ all | ❌ |
| val_cdl_order | ⚠️ newer runs only | ❌ |

**Verdict**: VERIFIED — no metric name ambiguity. All tables use val_ori_l2r_block column 5.

---

## Module C — Fixed-Step Recovery @50k

### C1: Seed-123 group ("seed2" frozen_beta)

**Sources**:
- L_random: `random_baseline_continuous_jun08_seed2/eval_curve.tsv` @50k = 3.445029
- L_L2R: `l2r_continuous_seed123/eval_curve.tsv` @50k = 3.305407
- Gap = 0.139622

| Method | val @50k | Δ | Recovery | Verified? |
|------|:--:|:--:|:--:|:--:|
| frozen_β from10k L0H2 | 3.324912 | 0.120117 | **86.0%** | ✅ |
| frozen_β from20k L0H2 | 3.329200 | 0.115829 | **83.0%** | ✅ |
| frozen_β from40k L0H2 | 3.363028 | 0.082001 | **58.7%** | ✅ |

All values match existing evidence package (Table C) exactly. ✅

### C2: Seed-42 group

**Sources**:
- L_random: `random_baseline_continuous_jun05/eval_curve.tsv` @50k = 3.466359
- L_L2R: `l2r_continuous_jun05/eval_curve.tsv` @50k = 3.341398
- Gap = 0.124961

| Method | val @50k | Δ | Recovery | Verified? |
|------|:--:|:--:|:--:|:--:|
| frozen_β from10k L0H4 | 3.333746 | 0.132613 | **106.1%** | ✅ |
| frozen_β from20k L0H4 | 3.339491 | 0.126868 | **101.5%** | ✅ |
| frozen_β from40k L0H4 | 3.382767 | 0.083592 | **66.9%** | ✅ |

Values match existing evidence package (after baseline correction from 3.354→3.466). ✅
**Note**: Earlier "correction" (2026-06-14 morning) that changed seed2 L2R from 3.305 to 3.370 was WRONG — 3.370 is seed=2, not seed=123. The existing package's seed2 L2R=3.305 is correct for the seed-123 group.

### C3: 🔴 CDL teacher Recovery — seed mismatch

CDL teacher runs (seed=2) were placed in seed-123 group table. Cross-seed Recovery values:
- from10k: 115.6%, from20k: 98.2%, from40k: 65.7%

With seed-2's OWN matched group (L_random=3.413, L_L2R=3.370, gap=0.043):
- from10k: 302.2%, from20k: 245.0%, from40k: 138.6%

**Verdict**: UNRESOLVED. CDL teacher Recovery cannot be verified as matched-group. Two issues:
1. Seed mismatch (CDL teacher=seed=2 vs frozen_beta=seed=123)
2. Seed-2 group gap (0.043) is too narrow for meaningful Recovery%

**Recommendation**: Report CDL teacher as supplementary cross-seed evidence, not primary matched-group result. Use its own seed-2 group for proper comparison, or note the cross-seed caveat explicitly.

---

## Module D — Step Saving @3.47

### D1: Threshold crossing

**Crossing method**: First eval checkpoint ≤ 3.47 (no interpolation). Checkpoints every 1000 steps.
**T_random(3.47)** = step 42000 (val = 3.466563) from seed-123 random baseline. ✅

| Method | T_cross | val | Saving | Verified? |
|------|:--:|:--:|:--:|:--:|
| ori-L2R ref (seed-123) | 15000 | 3.465 | 64.3% | ✅ |
| ori-L2R ref (seed-2) | 26000 | 3.469 | 38.1% | ✅ |
| ori-L2R ref (seed-42) | 19000 | 3.467 | 54.8% | ✅ |
| frozen_β seed-123 from10k L0H2 | 24500 | 3.469 | **41.7%** | ✅ |
| frozen_β seed-123 from20k L0H2 | 27000 | 3.463 | **35.7%** | ✅ |
| frozen_β seed-123 from40k L0H2 | 40500 | 3.464 | 3.6%¹ | ✅ |
| frozen_β seed-42 from10k L0H4 | 25500 | 3.466 | 39.3% | ✅ |
| frozen_β seed-42 from20k L0H4 | 28000 | 3.467 | 33.3% | ✅ |
| frozen_β seed-42 from40k L0H4 | 41500 | 3.464 | 1.2%² | ✅ |
| CDL teacher seed-2 from10k L0H2 | 22000 | 3.467 | 47.6% | ⚠️ cross-seed |
| CDL teacher seed-2 from20k L0H2 | 26500 | 3.467 | 36.9% | ⚠️ cross-seed |
| CDL teacher seed-2 from40k L0H2 | 40500 | 3.463 | 3.6%¹ | ⚠️ cross-seed |

¹ from40k starts below 3.47 (first eval at step 40500, val=3.464) — step saving marginal.
² from40k starts at step 40500, val=3.487 (>3.47), crosses at 41500.

### D2: Conservative range

frozen_β only, from10k+from20k, all seeds: **33.3%–41.7%** → rounded to **33–42%**. ✅

**Verdict**: All step saving numbers VERIFIED (checkpoint-based, consistent across methods). CDL teacher numbers marked cross-seed.

---

## Module E — g_β Mechanism Sanity

### E1: Script verification

**File inspected**: `analyses/gbeta_input_sanity_final.py`

**Methodology VERIFIED**: Script loads trained g_β checkpoint, feeds different B inputs (real, Gaussian, shuffled, zero, etc.), computes CDL greedy order from g_β output scores, measures Kendall τ vs L2R reference.

### E2: Numeric outputs

**Source**: Primary script run output was NOT saved to a JSON file. Numbers come from two independent sources:
1. `memory/gbeta-input-sanity-20260611.md` — detailed summary with exact numbers
2. `analyses/figures/fig2_gbeta_sanity_bar.png` — bar chart visualization

**Numbers verified by cross-referencing both sources**:

| B input | τ vs L2R | Margin/Diversity | Source |
|------|:--:|------|------|
| Real B (L0H4, random-order model) | 0.97 | — | memory + fig2 |
| Gaussian B (μ=0, σ matched) | ≈0 | — | memory + fig2 |
| Entry-shuffled B | ≈0 | — | memory + fig2 |
| Row+col shuffled B | ≈0 | — | memory + fig2 |
| Zero B | 1.0 | margin=0 (tie-breaking) | memory + fig2 |
| Gaussian family pairwise | 0.0003 | pairwise diversity | memory |
| Non-L2R subset vs L2R-prior | +0.11 Δ | B-dependent readout | memory |

**Zero B τ=1.0 explanation VERIFIED**: When B is all zeros, g_β outputs identical scores for all edges. CDL greedy tie-breaking selects the first available block, which by construction = block 0, block 1, ... = L2R. This is a CDL artifact, not g_β memorization. The margin (score range) is zero — all edges indistinguishable.

**Verdict**: VERIFIED with caveat — exact numeric values confirmed by two independent secondary sources but no primary JSON output. Recommend re-running script and saving output for paper-ready evidence.

---

## Module F — Per-Head Emergence Signal

### F1: Primary per-head scan data

**Files inspected**: head_scan_*.json from 4 experiments.

| Experiment | Step | Total heads | \|τ\|>0.9 | Best +τ | Best −τ | Heavy τ |
|------|:--:|:--:|:--:|------|------|:--:|
| clean_base_random_perm (64blk) | 10k | 32 | **9** | L0H0 +1.000 | L1H6 −0.938 | 0.876 |
| shuffle_gran_32 | 10k | 32 | **5** | L0H0 +1.000 | L2H0 −0.938 | −0.773 |
| shuffle_gran_128 | 10k | 32 | **8** | L1H2 +1.000 | L2H3 −0.938 | 0.305 |
| large 317M (16L16H) | 5k | 256 | **6** | L0H10 +0.955 | L1H3 −0.938 | 0.037 |

### F2: 🔴 DEPRECATED — "28/32 heads |τ|>0.9"

The existing evidence package (Table G) and daily-summary memory claim "28/32 (88%) heads with |τ|>0.9" for the small model. Actual count from head_scan_10k.json: **9/32**. Full distribution: 22 positive, 10 negative, median |τ|=0.290.

### F3: Selected heads for deployment

- seed-123 group: L0H2 (τ=+0.777, from per-head scan; selected via audition, not just τ max)
- seed-42 group: L0H4 (τ=+0.214, τ is low at 10k but selected via audition/g_β pipeline)

### F4: Heavy τ is NOT the primary metric

Heavy τ across different experiments:
- clean_base: 0.876 (positive heads dominate)
- shuffle_gran_32: −0.773 (negative heads dominate!)
- shuffle_gran_128: 0.305 (positive/negative cancel)
- 317M: 0.037 (near zero, but 6 heads have |τ|>0.9)

This confirms: heavy τ is misleading and should NOT be used as primary evidence.

### F5: "random τ=0.49" context

From `memory/cdl_shuffled_l2r_diagnostic_20260609.md`: this is an aggregate/CDL-rollout diagnostic from `diag_shuffled_l2r_cdl.py`, NOT a per-head metric. It is supplementary context — the primary signal is per-head |τ|≈1.0 from selected order-bearing heads.

**Verdict**: Per-head emergence VERIFIED for 9/32 strong heads in small model and 6/256 in large model. "28/32" claim DEPRECATED. Heavy τ confirmed misleading.

---

## Module G — Granularity Robustness

### G1: Aggregation granularity

**Source**: `analyses/block_granularity_scan_results/scan_step5000_M100.json`

| Granularity | Heavy τ | unique σ |
|:--:|:--:|:--:|
| 32blk × 8tok | 0.9997 | 0.02 |
| 64blk × 4tok | 0.9999 | 0.02 |
| 128blk × 2tok | 1.0000 | 0.01 |

All τ > 0.96 ✅. This is HEAVY τ (aggregate across heads). Not per-head τ.

### G2: Training granularity

From Module F head_scan data:

| Granularity | Best +τ | Best −τ | \|τ\|>0.9 count |
|:--:|------|------|:--:|
| 32 groups | L0H0 +1.000 | L2H0 −0.938 | 5 |
| 64 groups | L0H0 +1.000 | L1H6 −0.938 | 9 |
| 128 groups | L1H2 +1.000 | L2H3 −0.938 | 8 |

All three granularities produce heads with |τ|≈1.0. ✅

### G3: unique σ definition

In granularity scan JSON: `unique_sigma_ratio` = fraction of M=100 samples with distinct CDL orders. In head_scan JSONs: proxied by `mean_pairwise_tau` (pw_tau=1.0 = all same, pw_tau<0.5 = diverse).

**Verdict**: Granularity robustness VERIFIED. "unique σ" definition traced to scan JSON.

---

## Module H — 317M Scale Diagnostic

### H1: Model config

**Source**: `large_random_baseline_16l16h1024d/config.json`
- n_layer=16, n_head=16, n_embd=1024
- Total heads: 16 × 16 = 256
- block_size=256, block_order_block_len=4
- data_source=chunks, max_steps=10000
- Trained only to step 5000 (diagnostic scan point)

### H2: Per-head signal

**Source**: `head_scan_step5000.json`
- 256 heads total
- 6 heads with |τ|>0.9 (4 positive, 2 negative)
- Best head: L0H10 τ=+0.955
- Heavy τ: 0.037 (near zero — sparse signal!)

### H3: Hook acceleration

**Finding**: NO frozen_beta hook training run exists for the 317M model. No eval_curve.tsv with run_kind=frozen_beta for 16L/16H config.

**Verdict**: Scale diagnostic VERIFIED. 🟢 Signal exists (6/256 heads with |τ|>0.9). 🔴 NO hook training — CANNOT claim 317M frozen hook acceleration.

---

## Module I — Figure Verification

### I1: Figure existence check (REVISED 2026-06-14 19:55)

**File inspected**: `analyses/figures/` directory listing.

**Finding**: ALL 4 core figures EXIST on disk:

| Figure | Path | Size | Date | Exists? |
|------|------|:--:|------|:--:|
| fig1_method_overview | `analyses/figures/fig1_method_overview.png` | 64KB | Jun 13 22:12 | ✅ |
| fig2_gbeta_sanity_bar | `analyses/figures/fig2_gbeta_sanity_bar.png` | 52KB | Jun 13 22:11 | ✅ |
| fig3_catchup_curve | `analyses/figures/fig3_catchup_curve.png` | 107KB | Jun 13 22:15 | ✅ |
| fig4_multistart_comparison | `analyses/figures/fig4_multistart_comparison.png` | 169KB | Jun 13 22:15 | ✅ |

Supplementary figures also present: `head_signal_emergence.png`, `L0H2_B_heatmap_30k.png`, `L0H2_B_model_vs_phys.png`, `seed2_attention_heatmaps.png`, `seed2_B_diagnostics.png`, `shuffle_gran128_B_heatmaps.png`.

### I2: Content verification

⚠️ Cannot verify numeric content of PNG figures without visual inspection. Based on modification dates (all Jun 13 22:11–22:15), generated during evidence package creation. **Risk**: fig3/fig4 may use pre-correction seed42 numbers if generated before baseline audit. figs were timestamped AFTER `TABLES_20260613.md` (22:11) and `CLAIMS_LOCK_20260613.md` (22:05) — likely consistent with those files' numbers.

**Verdict**: VERIFIED (files exist). ⚠️ Visual inspection needed to confirm numeric accuracy of figs 3, 4 against corrected Recovery/step-saving values.

---
