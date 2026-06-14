# 07 — Traceability Index

> Every verified number → primary source file → field/column → step.
> Paths relative to `/home/admin/lyuyuhuan/order_lyu/block_lo_arm_order_network/probe_results/` unless noted.

---

## Recovery @50k

| Number | Value | Source file | Field | Step |
|------|:--:|------|------|:--:|
| L_random (seed-123) | 3.445 | `random_baseline_continuous_jun08_seed2/eval_curve.tsv` | val_ori_l2r_block | 50000 |
| L_L2R (seed-123) | 3.305 | `l2r_continuous_seed123/eval_curve.tsv` | val_ori_l2r_block | 50000 |
| frozen_β seed-123 from10k | 3.325 | `frozen_beta_seed2_from10k_l0h2/eval_curve.tsv` | val_ori_l2r_block | 50000 |
| frozen_β seed-123 from20k | 3.329 | `frozen_beta_seed2_from20k_l0h2/eval_curve.tsv` | val_ori_l2r_block | 50000 |
| frozen_β seed-123 from40k | 3.363 | `frozen_beta_seed2_from40k_l0h2/eval_curve.tsv` | val_ori_l2r_block | 50000 |
| L_random (seed-42) | 3.466 | `random_baseline_continuous_jun05/eval_curve.tsv` | val_ori_l2r_block | 50000 |
| L_L2R (seed-42) | 3.341 | `l2r_continuous_jun05/eval_curve.tsv` | val_ori_l2r_block | 50000 |
| frozen_β seed-42 from10k | 3.334 | `frozen_beta_random_jun05_from10k/eval_curve.tsv` | val_ori_l2r_block | 50000 |
| frozen_β seed-42 from20k | 3.339 | `frozen_beta_random_jun05_from20k/eval_curve.tsv` | val_ori_l2r_block | 50000 |
| frozen_β seed-42 from40k | 3.383 | `frozen_beta_random_jun05_from40k/eval_curve.tsv` | val_ori_l2r_block | 50000 |
| CDL teacher from10k | 3.284 | `cdl_teacher_seed2_from10k_l0h2/eval_curve.tsv` | val_ori_l2r_block | 50000 |
| CDL teacher from20k | 3.308 | `cdl_teacher_seed2_from20k_l0h2/eval_curve.tsv` | val_ori_l2r_block | 50000 |
| CDL teacher from40k | 3.353 | `cdl_teacher_seed2_from40k_l0h2/eval_curve.tsv` | val_ori_l2r_block | 50000 |

## Step Saving @3.47

| Number | Value | Source file | Method |
|------|:--:|------|------|
| T_random(3.47) | 42000 | `random_baseline_continuous_jun08_seed2/eval_curve.tsv` | first ≤ 3.47 |
| T_L2R(3.47) seed-123 | 15000 | `l2r_continuous_seed123/eval_curve.tsv` | first ≤ 3.47 |
| T_frozenβ from10k | 24500 | `frozen_beta_seed2_from10k_l0h2/eval_curve.tsv` | first ≤ 3.47 |
| T_frozenβ from20k | 27000 | `frozen_beta_seed2_from20k_l0h2/eval_curve.tsv` | first ≤ 3.47 |
| T_frozenβ from10k (seed42) | 25500 | `frozen_beta_random_jun05_from10k/eval_curve.tsv` | first ≤ 3.47 |
| T_frozenβ from20k (seed42) | 28000 | `frozen_beta_random_jun05_from20k/eval_curve.tsv` | first ≤ 3.47 |
| T_CDL from10k | 22000 | `cdl_teacher_seed2_from10k_l0h2/eval_curve.tsv` | first ≤ 3.47 |

## Per-Head τ

| Number | Value | Source file | Field |
|------|:--:|------|------|
| L0H0 τ (clean_base 10k) | +1.000 | `clean_base_random_perm/head_scan_10k.json` | per_head[0].tau_vs_l2r |
| clean_base \|τ\|>0.9 count | 9 | same | count |
| clean_base heavy τ | 0.876 | same | heavy_baseline.tau_vs_l2r |
| shuffle_gran_32 best +τ | L0H0 +1.000 | `shuffle_gran_32/head_scan_10k.json` | per_head |
| shuffle_gran_128 best +τ | L1H2 +1.000 | `shuffle_gran_128/head_scan_10k.json` | per_head |
| 317M best τ | L0H10 +0.955 | `large_random_baseline_16l16h1024d/head_scan_step5000.json` | per_head |
| 317M \|τ\|>0.9 count | 6 | same | count |

## Granularity

| Number | Value | Source file | Field |
|------|:--:|------|------|
| 32blk heavy τ | 0.9997 | `analyses/block_granularity_scan_results/scan_step5000_M100.json` | per_granularity |
| 64blk heavy τ | 0.9999 | same | per_granularity |
| 128blk heavy τ | 1.0000 | same | per_granularity |

## Run Identity

| Run | seed | run_kind | Source file |
|------|:--:|------|------|
| random_baseline_continuous_jun08_seed2 | 123 | baseline | `config.json` args.seed, args.run_kind |
| random_baseline_continuous_jun05 | 42 | baseline | same |
| frozen_beta_seed2_from10k_l0h2 | 123 | frozen_beta | same |
| frozen_beta_random_jun05_from10k | 42 | frozen_beta | same |
| cdl_teacher_seed2_from10k_l0h2 | 2 | cdl_teacher | same |
| l2r_continuous_seed123 | 123 | l2r | same |
| l2r_continuous_jun05 | 42 | l2r | same |
| l2r_continuous_seed2 | 2 | l2r | same |
| frozen_beta_random_baseline_jun05 | 42 | **frozen_beta** | same (NOT baseline) |

## g_β Sanity

| Number | Value | Source |
|------|:--:|------|
| Real B τ | 0.97 | `analyses/gbeta_input_sanity_final.py` + memory |
| Gaussian B τ | ≈0 | same |
| Zero B τ (tie-breaking) | 1.0 | same |
| Gaussian pairwise τ | 0.0003 | same |
| Non-L2R Δ | +0.11 | same |

⚠️ No primary JSON output saved. Numbers from script methodology + memory summaries.

---

## Primary Source Directory Map

```
block_lo_arm_order_network/probe_results/
├── random_baseline_continuous_jun05/          ← seed42 random baseline
├── random_baseline_continuous_jun08_seed2/     ← seed-123 random baseline
├── random_baseline_continuous_jun11_seed2_headtrack/  ← seed2 random baseline
├── l2r_continuous_jun05/                       ← seed42 L2R ref
├── l2r_continuous_seed123/                     ← seed-123 L2R ref
├── l2r_continuous_seed2/                       ← seed2 L2R ref
├── frozen_beta_random_jun05_from10k/           ← seed42 frozen_β from10k L0H4
├── frozen_beta_random_jun05_from20k/           ← seed42 frozen_β from20k
├── frozen_beta_random_jun05_from40k/           ← seed42 frozen_β from40k
├── frozen_beta_seed2_from10k_l0h2/             ← seed-123 frozen_β from10k L0H2
├── frozen_beta_seed2_from20k_l0h2/             ← seed-123 frozen_β from20k
├── frozen_beta_seed2_from40k_l0h2/             ← seed-123 frozen_β from40k
├── cdl_teacher_seed2_from10k_l0h2/             ← seed2 CDL teacher from10k
├── cdl_teacher_seed2_from20k_l0h2/             ← seed2 CDL teacher from20k
├── cdl_teacher_seed2_from40k_l0h2/             ← seed2 CDL teacher from40k
├── clean_base_random_perm/                     ← chunk-based 64blk (head scan source)
├── shuffle_gran_32/                            ← 32-group training granularity
├── shuffle_gran_128/                           ← 128-group training granularity
├── large_random_baseline_16l16h1024d/          ← 317M diagnostic
└── frozen_beta_random_baseline_jun05/          ← 🔴 NOT a baseline (frozen_beta)
```
