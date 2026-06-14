# 02 — Unified Tables

> All numbers from verified primary sources (eval_curve.tsv, head_scan JSON).
> Source paths documented. `[TO-VERIFY]` marks values needing further confirmation.

---

## Table A: Mechanism Sanity — g_β reads B structure

| B source | τ vs L2R (g_β output) | Margin | Verdict | Source |
|----------|:--:|:--:|------|------|
| Real B (random-order model, L0H4) | 0.97 | — | ✓ reads B structure | `analyses/gbeta_input_sanity_final.py`; `memory/gbeta-input-sanity-20260611.md` |
| Gaussian B (matched μ,σ) | ≈0 | — | ✗ no signal | same |
| Entry-shuffled B | ≈0 | — | ✗ no signal | same |
| Row+col shuffled B | ≈0 | — | ✗ no signal | same |
| Gaussian family pairwise τ | 0.0003 | pairwise | ✗ no fixed L2R prior | same |
| Zero B | 1.0 | margin=0 | tie-breaking artifact (all edges equal → CDL picks first = L2R) | same |
| Non-L2R subset vs L2R-prior | +0.11 Δ | Δ over prior | ✓ B-dependent readout (reads order beyond constant L2R) | same |

**Key interpretation**: If g_β had memorized a constant L2R prior, Gaussian/shuffled/zero B would all produce τ≈1.0. They don't. Zero B τ=1.0 is a tie-breaking artifact (zero margin → CDL picks first available = L2R by construction), NOT evidence for L2R prior.

---

## Table B: Attention Emergence — per-head τ vs physical L2R

| Model / Protocol | Step | Best +τ | Best −τ | Top-3 mean \|τ\| | Note | Source |
|------|:--:|------|------|:--:|------|------|
| clean_base_random_perm (64blk) | 10k | L0H0 +1.000 | — | ~1.000 | 几乎所有 head τ≈+1.0 | `probe_results/clean_base_random_perm/head_scan_10k.json` |
| shuffle_gran_32 (32-group train) | 10k | L0H0 +1.000 | L2H0 −0.938 | ~0.958 | 正2反3, head 同质 | `probe_results/shuffle_gran_32/head_scan_10k.json` |
| shuffle_gran_128 (128-group train) | 10k | L1H2 +1.000 | L2H3 −0.938 | ~0.958 | 正6反2, head 多样(σ=0.39) | `probe_results/shuffle_gran_128/head_scan_10k.json` |

**Important**: These are per-head CDL τ values, NOT heavy τ (weighted average) or aggregate diagnostics. The method pipeline uses **per-head selected attention graphs** B^{l,h} → CDL → g_β → hook. Therefore the primary emergence evidence is the **sparse per-head signal** where individual selected heads reach |τ|≈1.0.

**Historical note**: The "random τ=0.49" figure from `memory/cdl_shuffled_l2r_diagnostic_20260609.md` is a DIFFERENT diagnostic (aggregate/CDL rollout level, shuffled-L2R control gate). It is NOT the primary signal metric for this method, which relies on per-head selected graphs. That aggregate number is supplementary context only — the main signal is per-head |τ|≈1.0 from selected order-bearing heads.

---

## Table C: Fixed-step Recovery @50k

**Seed2** (random baseline @50k = 3.445, ori-L2R ref @50k = 3.305, gap = 0.140)

| Seed | Head | Method | Resume | L_method @50k | Δ vs random | Recovery | Source |
|------|------|------|:--:|:--:|:--:|:--:|------|
| seed2 | L0H2 | frozen_β | from10k | 3.325 | +0.120 | **86.0%** | `probe_results/frozen_beta_seed2_from10k_l0h2/eval_curve.tsv` |
| seed2 | L0H2 | frozen_β | from20k | 3.329 | +0.116 | **83.0%** | `probe_results/frozen_beta_seed2_from20k_l0h2/eval_curve.tsv` |
| seed2 | L0H2 | frozen_β | from40k | 3.363 | +0.082 | 58.7% | `probe_results/frozen_beta_seed2_from40k_l0h2/eval_curve.tsv` |
| seed2 | L0H2 | CDL teacher | from10k | 3.284 | +0.161 | **115.6%** | `probe_results/cdl_teacher_seed2_from10k_l0h2/eval_curve.tsv` |
| seed2 | L0H2 | CDL teacher | from20k | 3.308 | +0.137 | 98.2% | `probe_results/cdl_teacher_seed2_from20k_l0h2/eval_curve.tsv` |
| seed2 | L0H2 | CDL teacher | from40k | 3.353 | +0.092 | 65.7% | `probe_results/cdl_teacher_seed2_from40k_l0h2/eval_curve.tsv` |
| — | — | random baseline | — | 3.445 | 0.000 | 0% (def) | `probe_results/random_baseline_continuous_jun08_seed2/eval_curve.tsv` |
| — | — | ori-L2R reference | — | 3.305 | +0.140 | 100% (def) | `probe_results/l2r_continuous_seed123/eval_curve.tsv` |

**Seed42** (random baseline @50k = 3.466, ori-L2R ref @50k = 3.341, gap = 0.125)
✅ Gap is normal (cf. seed2 gap = 0.140). Baseline group verified: same data_source=continuous, same seed=42, same protocol.

| Seed | Head | Method | Resume | L_method @50k | Δ vs random | Recovery | Source |
|------|------|------|:--:|:--:|:--:|:--:|------|
| seed42 | L0H4 | frozen_β | from10k | 3.334 | +0.132 | **106.1%** | `probe_results/frozen_beta_random_jun05_from10k/eval_curve.tsv` |
| seed42 | L0H4 | frozen_β | from20k | 3.339 | +0.127 | **101.5%** | `probe_results/frozen_beta_random_jun05_from20k/eval_curve.tsv` |
| seed42 | L0H4 | frozen_β | from40k | 3.383 | +0.083 | 66.9% | `probe_results/frozen_beta_random_jun05_from40k/eval_curve.tsv` |
| — | — | random baseline | — | 3.466 | 0.000 | 0% (def) | `probe_results/random_baseline_continuous_jun05/eval_curve.tsv` |
| — | — | ori-L2R reference | — | 3.341 | +0.125 | 100% (def) | `probe_results/l2r_continuous_jun05/eval_curve.tsv` |

⚠️ Previous version of this table incorrectly used `frozen_beta_random_baseline_jun05` (a frozen_beta run, @50k=3.354) as the random baseline. Corrected 2026-06-13 23:45.

---

## Table D: Step Saving @3.47

T_random(3.47) = 42000 (seed2 random baseline). ori-L2R reaches at 15000.

| Seed | Head | Method | Resume | T_method(3.47) | Saving | Crossing interval | Source |
|------|------|------|:--:|:--:|:--:|------|------|
| — | — | ori-L2R | — | 15000 | 64.3% | 14000→15000 | `probe_results/l2r_continuous_seed123/eval_curve.tsv` |
| seed2 | L0H2 | CDL teacher | from10k | 22000 | 47.6% | 21500→22000 | `probe_results/cdl_teacher_seed2_from10k_l0h2/eval_curve.tsv` |
| seed2 | L0H2 | **frozen_β** | **from10k** | **24500** | **41.7%** | 24000→24500 | `probe_results/frozen_beta_seed2_from10k_l0h2/eval_curve.tsv` |
| seed42 | L0H4 | frozen_β | from10k | 25500 | 39.3% | 25000→25500 | `probe_results/frozen_beta_random_jun05_from10k/eval_curve.tsv` |
| seed2 | L0H2 | CDL teacher | from20k | 26500 | 36.9% | 26000→26500 | `probe_results/cdl_teacher_seed2_from20k_l0h2/eval_curve.tsv` |
| seed2 | L0H2 | **frozen_β** | **from20k** | **27000** | **35.7%** | 26500→27000 | `probe_results/frozen_beta_seed2_from20k_l0h2/eval_curve.tsv` |
| seed42 | L0H4 | frozen_β | from20k | 28000 | 33.3% | 27500→28000 | `probe_results/frozen_beta_random_jun05_from20k/eval_curve.tsv` |
| seed2 | L0H2 | frozen_β | from40k | — [1] | — | starts below 3.47 | `probe_results/frozen_beta_seed2_from40k_l0h2/eval_curve.tsv` |
| seed2 | L0H2 | CDL teacher | from40k | — [1] | — | starts below 3.47 | `probe_results/cdl_teacher_seed2_from40k_l0h2/eval_curve.tsv` |

[1] from40k starts with val_ori_l2r already ≤3.47 (seed2 L0H2 @40k = 3.464). Step saving not meaningful.

**Conservative range** (frozen_β only, from10k+from20k, 2 seeds): **35–42% step saving**.

---

## Table E: Catch-up to ori-L2R reference (seed2 L0H2)

All values = val_ori_l2r_block.

| Step | ori-L2R ref | frozen_β from10k | Gap | frozen_β from20k | Gap | frozen_β from40k | Gap |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 10000 | 3.509 | 3.700 | +0.191 | — | — | — | — |
| 20000 | 3.425 | 3.502 | +0.077 | 3.620 [2] | +0.195 | — | — |
| 30000 | 3.367 | 3.424 | +0.057 | 3.439 | +0.072 | — | — |
| 40000 | 3.320 | 3.355 | +0.035 | 3.360 | +0.040 | 3.464 [2] | +0.144 |
| 50000 | 3.305 | 3.325 | **+0.020** | 3.329 | **+0.024** | 3.363 | **+0.058** |
| 60000 | — [3] | 3.331 | — | 3.332 | — | 3.351 | — |

[2] At resume point — method starts with random-order model's loss, then catches up.
[3] ori-L2R continuous training ended at step 50000 (eval_curve goes to 50k).

**Narrative**: frozen_β closes the gap caused by initial 10k random-order warmup. from10k reaches within 0.020 of ori-L2R ref at 50k.

---

## Table F: Granularity Robustness

### Aggregation granularity (same 64-block model, different B aggregation)

| Experiment | Granularity | Best +τ | Best −τ | Note | Source |
|------|:--:|------|------|------|------|
| Aggregation 32blk × 8tok | 32 blocks | τ=0.9997 | — | heavy τ | `analyses/block_granularity_scan_results/scan_step5000_M100.json` |
| Aggregation 64blk × 4tok | 64 blocks | τ=0.9999 | — | heavy τ | same |
| Aggregation 128blk × 2tok | 128 blocks | τ=1.0000 | — | heavy τ | same |

### Training granularity (different shuffle_granularity during training)

| Experiment | Granularity | Strongest +τ | Strongest −τ | unique σ | Source |
|------|:--:|------|------|:--:|------|
| shuffle_gran_32 | 32 groups × 8tok | L0H0 +1.000 | L2H0 −0.938 | 0.09 | `probe_results/shuffle_gran_32/head_scan_10k.json` |
| clean_base (64) | 64 groups × 4tok | ~all +1.0 | — | ~0.02 | `probe_results/clean_base_random_perm/head_scan_10k.json` |
| shuffle_gran_128 | 128 groups × 2tok | L1H2 +1.000 | L2H3 −0.938 | 0.39 | `probe_results/shuffle_gran_128/head_scan_10k.json` |

---

## Table G: Scale Diagnostic

| Model | Params | L/H/d | Heads | \|τ\|>0.9 | Best τ | unique σ | Hook? | Source |
|------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|------|
| Small | 47M | 4/8/384 | 32 | 28 (88%) | +1.000 | 0.02 | ✅ trained | `probe_results/clean_base_random_perm/head_scan_10k.json` |
| Large | 317M | 16/16/1024 | 256 | 6 (2.3%) | +0.955 | 1.00 | ❌ diagnostic only | `probe_results/large_random_baseline_16l16h1024d/head_scan_step5000.json` |

**Large model note**: 317M shows τ>0.9 heads EXIST (6/256), but no frozen hook training has been run. Signal is sparse and sample-diverse (σ=1.0) vs small model's dense uniform encoding (σ=0.02). DO NOT claim "317M hook works" — it hasn't been tested.
