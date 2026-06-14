# 05 — Conflicts & TO-VERIFY Items

---

## 🔴 CONFLICT 1: Two different ori-L2R references

| Reference | File | ori_l2r @50k |
|------|------|:--:|
| l2r_continuous_seed123 | `probe_results/l2r_continuous_seed123/eval_curve.tsv` | **3.305** |
| l2r_continuous_jun05 | `probe_results/l2r_continuous_jun05/eval_curve.tsv` | **3.341** |

**Resolution**: These are different seeds (seed123 vs seed42) with different training runs. Both are valid ori-L2R references for their respective seed groups. 3.305 ≠ 3.341 is a ~0.036 difference — within normal seed variance.

**Action**: Use seed123 (3.305) for seed2 group tables. Use jun05 (3.341) for seed42 group tables. Always specify WHICH ori-L2R reference in table footnotes.

---

## ✅ RESOLVED: Seed42 baseline group mismatch (was CONFLICT 2)

**Root cause**: `frozen_beta_random_baseline_jun05` was incorrectly identified as a "random baseline." Inspection of its config.json reveals `run_kind: frozen_beta` — it is a frozen_beta run, NOT a random baseline. Its val_ori_l2r@50k=3.354 is the frozen_beta RESULT, not a baseline.

**Correct matched baseline group for seed42** (all verified from config.json):
| Role | Run | @50k | Config verified |
|------|------|:--:|:--:|
| Random baseline | `random_baseline_continuous_jun05` | **3.466** | seed=42, run_kind=baseline, data_source=continuous |
| ori-L2R reference | `l2r_continuous_jun05` | **3.341** | seed=42, data_source=continuous |
| Gap | — | **0.125** | normal (cf. seed2 gap=0.140) ✅ |
| Frozen_β runs resume from | `random_baseline_continuous_jun05/ckpt_step{10k,20k,40k}.pt` | — | verified from frozen_beta config.json resume_ckpt field |

**Corrected seed42 Recovery @50k**:
- from10k: (3.466−3.334)/0.125 = **106.1%** (beats reference)
- from20k: (3.466−3.339)/0.125 = **101.5%**
- from40k: (3.466−3.383)/0.125 = **66.9%**

**All 6 audit checks PASS**: same data_source, seed, eval metric, step, model config, protocol. ✅
Previously "narrow gap 0.013" was an artifact of using the wrong baseline value. Correct gap is 0.125 — consistent with seed2 (0.140).

---

## 🟡 CONFLICT 3: T_random(3.47) precision

| Source | Value |
|------|------|
| First eval ≤ 3.47 | 42000 (val=3.4666) |
| Last eval > 3.47 | 41000 (val=3.4706) |
| Interpolated crossing | ~41149 |

**Resolution**: Use 42000 as T_random(3.47) (first eval checkpoint ≤ target). Note ±1000 step precision. All step savings should cite this. If interpolation is preferred, change all methods to use interpolated steps consistently.

---

## ✅ RESOLVED: "random τ=0.49" vs "per-head τ≈1.0" — not a conflict (was CONFLICT 4)

- `memory/cdl_shuffled_l2r_diagnostic_20260609.md` mentions "random baseline τ=0.49"
- `per_head_order_scan.py` outputs show per-head τ≈1.0 for clean_base_random_perm

**Resolution**: The method pipeline uses **per-head selected attention graphs** B^{l,h} → CDL → g_β → hook. Therefore the primary signal evidence is the per-head scan where individual selected heads reach |τ|≈1.0. The τ=0.49 is a DIFFERENT diagnostic (aggregate/CDL-rollout level, from `diag_shuffled_l2r_cdl.py`) and is supplementary context only — it is not the primary signal source for this method. This is not a conflict; it's two different measurements at different levels of aggregation.

**Correct framing**: "Random-order AO-GPT contains sparse order-bearing heads with |τ|≈1.0. Aggregate diagnostics give lower but still positive alignment. The method relies on per-head selected signal, not aggregate."

---

## 🟡 ITEM 5: unique σ definition needs clarification

`unique_sigma_ratio` appears in head_scan JSONs. Likely = fraction of M=100 samples producing distinct CDL σ orders.

**Action**: Verify definition in `per_head_order_scan.py` source. If confirmed: add to metric definitions. If different: correct tables.

---

## 🟡 ITEM 6: Head selection — label-free vs oracle?

The current pipeline uses `g_β` trained with CDL teacher supervision. The "label-free audition" (row-concentration, S_forward) is used for head SELECTION, not for g_β training.

**Action**: Clarify in claims: "Head selected via cheap diagnostic signals; g_β trained with CDL teacher supervision. End-to-end label-free pipeline (head selection + g_β training + hook) has NOT been demonstrated."

---

## 🟡 ITEM 7: val_ori_l2r vs val_ori_l2r_block

In all verified eval_curve.tsv files, `val_ori_l2r_block` is the column name. Some memory files reference `val_ori_l2r`. In all checked cases, these refer to the same value.

**Action**: Establish `val_ori_l2r_block` as canonical name. Note in metric definitions that `val_ori_l2r` is an alias.

---

## 🟡 ITEM 8: seed42 Recovery% needs recomputation

Current TABLE C (TABLES_20260613.md) uses:
- seed42 random = 3.354 (correct)
- ori-L2R ref = 3.305 (WRONG for seed42 — should be 3.341)

Corrected Recovery:
- from10k: (3.354 − 3.334) / (3.354 − 3.341) = 0.020/0.013 **>100%** [NOTE: gap too narrow for meaningful Recovery]
- from20k: (3.354 − 3.339) / 0.013 **>100%**
- from40k: (3.354 − 3.383) / 0.013 = negative

**Conclusion**: Seed42 Recovery% is unreliable due to narrow gap. Step Saving is the more robust cross-seed metric for seed42.

---

## TO-VERIFY Checklist

- [ ] Confirm `unique_sigma_ratio` definition in per_head_order_scan.py
- [ ] Verify which seed42 ori-L2R reference was used for each table
- [ ] Check if any Recovery numbers mix baseline@50k with method@60k
- [ ] Verify CDL teacher from10k Recovery 115.6% is at step 50k (not 60k) — CONFIRMED: @50k = 3.284, @60k = 3.288
- [ ] Check if frozen_beta_seed2_from10k_v3 (L0H7, ori_l2r@50k=3.300) should be included in main table
- [ ] Verify g_β sanity numbers (0.97, ~0, 0.0003) from primary script output, not just memory summary
- [ ] Check if `frozen_beta_random_baseline_jun05` is truly equivalent to a "random baseline" or has frozen_beta overhead
- [ ] Validate that all "from40k" results start with val_ori_l2r already below 3.47, making step saving N/A
