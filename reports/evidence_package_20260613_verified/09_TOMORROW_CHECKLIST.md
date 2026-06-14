# 09 — Tomorrow Morning Checklist (2026-06-15)

> Execution rules for CC. Read-only verification of overnight runs.
> **Golden rule**: same metric, same seed group, same global step. Never mix steps.

---

## Step 1: Verify overnight runs completed

Check that these directories have eval_curve.tsv with data at step 60000:

```
block_lo_arm_order_network/probe_results/
├── cdl_teacher_seed123_from10k_l0h2/eval_curve.tsv
├── cdl_teacher_seed123_from20k_l0h2/eval_curve.tsv
├── cdl_teacher_seed123_from40k_l0h2/eval_curve.tsv
├── random_baseline_continuous_jun08_seed2_ext60k/eval_curve.tsv
├── l2r_continuous_seed123_ext60k/eval_curve.tsv
```

For each: `awk -F'\t' '/60000\t/{print $5}' <dir>/eval_curve.tsv`

## Step 2: Identity audit (Module A for new runs)

For each new run, check config.json:
- `seed=123`
- `run_kind` matches (baseline / l2r / cdl_teacher)
- `data_source=continuous`
- `model`: 4L/8H/384d
- `cdl_teacher_head=[0,2]` (for CDL runs)

## Step 3: Table A — 50k matched comparison

| Method | val_ori_l2r_block @50k | Source |
|------|:--:|------|
| random baseline (seed-123) | 3.445 | `random_baseline_continuous_jun08_seed2/eval_curve.tsv` |
| L2R reference (seed-123) | 3.305 | `l2r_continuous_seed123/eval_curve.tsv` |
| frozen_β from10k L0H2 | 3.325 | `frozen_beta_seed2_from10k_l0h2/eval_curve.tsv` |
| frozen_β from20k L0H2 | 3.329 | `frozen_beta_seed2_from20k_l0h2/eval_curve.tsv` |
| frozen_β from40k L0H2 | 3.363 | `frozen_beta_seed2_from40k_l0h2/eval_curve.tsv` |
| CDL teacher from10k L0H2 | **READ** | `cdl_teacher_seed123_from10k_l0h2/eval_curve.tsv` |
| CDL teacher from20k L0H2 | **READ** | `cdl_teacher_seed123_from20k_l0h2/eval_curve.tsv` |
| CDL teacher from40k L0H2 | **READ** | `cdl_teacher_seed123_from40k_l0h2/eval_curve.tsv` |

Compute: Recovery_50k = (3.445 - L_method) / (3.445 - 3.305) × 100%

## Step 4: Table B — 60k matched comparison

**Prerequisite check**: Do frozen_β runs have val_ori_l2r_block at step 60000?

```
awk -F'\t' '/60000\t/{print $5}' frozen_beta_seed2_from10k_l0h2/eval_curve.tsv
```

If YES → include frozen_β in 60k table. If NO → 60k table is CDL teacher only.

| Method | val @60k | Source |
|------|:--:|------|
| random baseline (seed-123) | **READ** | `random_baseline_continuous_jun08_seed2_ext60k/eval_curve.tsv` |
| L2R reference (seed-123) | **READ** | `l2r_continuous_seed123_ext60k/eval_curve.tsv` |
| CDL teacher from10k L0H2 | **READ** | `cdl_teacher_seed123_from10k_l0h2/eval_curve.tsv` |

Compute: Recovery_60k using matched 60k values only.

**Rule**: If any method lacks 60k data, leave that cell empty. Never substitute 50k for 60k.

## Step 5: Step saving @3.47 update

Check if extended random baseline changes T_random(3.47). The original random baseline already crossed 3.47 at step 42000. The extension from 50k→60k starts below 3.47 — no change expected.

Update T_random if needed. Recompute all step savings with consistent methodology (first eval checkpoint ≤ 3.47).

## Step 6: Interpret CDL teacher vs frozen_β

Three possible outcomes — all are fine:

| Outcome | CDL teacher vs frozen_β | Framing |
|------|------|------|
| CDL >> frozen_β | CDL much stronger | "g_β distills most of a stronger teacher policy" |
| CDL ≈ frozen_β | Similar | "Lightweight readout preserves teacher benefit" |
| CDL < frozen_β | CDL weaker | "Direct teacher rollout not automatically optimal online" |

CDL teacher is an **ablation/teacher-policy comparison**, not the main deployed method. The main claim remains:

> Frozen g_β accelerates training across two matched seed groups.

## Step 7: Update verified package

After verification:
1. Add 60k numbers to `02_VERIFIED_TABLES.md`
2. Close U1 in `04_UNRESOLVED_ITEMS.md`
3. Update `05_FINAL_CLAIMS_LOCK.md` if CDL teacher warrants main-table inclusion
4. Regenerate `TABLES_20260613.md` in project root with both 50k and 60k

## Step 8: Do NOT

- ❌ Compare method@50k to baseline@60k
- ❌ Use CDL teacher seed=2 (old) numbers — only seed-123 (new)
- ❌ Claim "CDL teacher is better than g_β" without noting it's a heavier policy
- ❌ Start new experiments
- ❌ Change the main claim unless CDL teacher result forces it
