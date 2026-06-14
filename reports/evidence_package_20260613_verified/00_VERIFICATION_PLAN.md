# 00 — Verification Plan

> Target: Text-side AO-GPT / frozen g_β evidence package
> Method: Read-only audit of existing logs / JSON / config / eval_curve files
> Output: verified / deprecated / unresolved classification for every key number
> Start: 2026-06-14

---

## Module A — Run Identity & Baseline-Group Audit

**Goal**: Confirm run_kind, seed, model config, data loader, eval protocol for every run in the evidence chain. Prevent "frozen_beta mislabeled as random baseline" errors.

**Groups to audit**:
1. seed2/seed123 group — random baseline + L2R ref + frozen_β + CDL teacher
2. seed42 group — random baseline + L2R ref + frozen_β
3. seed1 supporting group
4. shuffled-L2R control
5. ori-L2R reference (all seeds)
6. 317M diagnostic group
7. Granularity groups (32/64/128)

**Check per run**: config.json → run_kind, seed, data_source, max_steps, model_args, eval metric columns

**Key question**: Does `l2r_continuous_seed123` match seed2's random baseline (`random_baseline_continuous_jun08_seed2`, seed=123)? Or should seed2 use `l2r_continuous_seed2`?

---

## Module B — Metric Name & Eval-Mode Audit

**Goal**: Confirm all main tables use val_ori_l2r_block. Map all column names. Flag any discrepancies.

**Check**: eval_curve.tsv headers from all key runs. Verify val_ori_l2r ≡ val_ori_l2r_block.

---

## Module C — Fixed-Step Recovery @50k Verification

**Goal**: Recompute every Recovery% number from raw eval_curve data. Verify L_random, L_method, L_L2R are matched (same seed, same step 50k, same metric).

**Seed2 group**: L_random from random_baseline_continuous_jun08_seed2, L_L2R from matched L2R ref (TBD)
**Seed42 group**: L_random from random_baseline_continuous_jun05, L_L2R from l2r_continuous_seed42 or l2r_continuous_jun05

---

## Module D — Step Saving @3.47 Verification

**Goal**: Recompute threshold crossing for every method. Confirm T_random(3.47). Check whether from40k methods start below threshold.

---

## Module E — g_β Mechanism Sanity

**Goal**: Extract g_β τ numbers from primary script output (not just memory summaries).

**Source**: `analyses/gbeta_input_sanity_final.py` output or associated JSON.

---

## Module F — Per-Head Emergence Signal

**Goal**: Verify per-head τ scans from raw JSON. Confirm selected heads (seed2 L0H2, seed42 L0H4).

---

## Module G — Granularity Robustness

**Goal**: Verify aggregation τ (>0.96) and training granularity τ (±1.0).

---

## Module H — 317M Scale Diagnostic

**Goal**: Confirm model config, head count, best τ. Explicitly confirm NO hook training exists.

---

## Module I — Figure Verification

**Goal**: Check if core figures use verified numbers. Flag any using deprecated values.

---

## Execution Protocol

1. Each module step logs: file inspected, command run, finding, conclusion, verified/unresolved.
2. Cross-reference: config.json + eval_curve.tsv + head_scan JSON must agree.
3. Conflicts → unresolved, not guessed.
4. Deprecated numbers catalogued with correction and source.
