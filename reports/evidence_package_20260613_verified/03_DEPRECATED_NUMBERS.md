# 03 — Deprecated Numbers & Claims

> Numbers/claims found in existing evidence package, memory files, or earlier tables that are superseded by verified values.
> Each entry: what was wrong, why, correct replacement, where it appeared.

---

## D1: seed42 random baseline = 3.354 (as a "random baseline")

| Field | Value |
|------|------|
| **Deprecated value** | seed42 random baseline @50k = 3.354 |
| **Why deprecated** | `frozen_beta_random_baseline_jun05` has `run_kind=frozen_beta`, NOT `run_kind=baseline`. Its alpha=1.0 means it was a frozen_beta run from step 0, not a clean random baseline. |
| **Correct value** | seed42 random baseline @50k = **3.466** from `random_baseline_continuous_jun05` (seed=42, run_kind=baseline, alpha=0) |
| **Impact** | All seed42 Recovery% and gap calculations using 3.354 are wrong |
| **Appeared in** | `TABLES_20260613.md` (pre-correction); `reports/evidence_package_20260613/02_unified_tables.md` Table C (pre-correction); `reports/evidence_package_20260613/05_conflicts_and_todo.md` ITEM 8 |
| **Action** | ✅ Corrected in `02_VERIFIED_TABLES.md` and `TABLES_20260613.md` (2026-06-14) |

---

## D2: "28/32 heads with \|τ\|>0.9" for small model

| Field | Value |
|------|------|
| **Deprecated claim** | "28/32 (88%) heads with \|τ\|>0.9" for small model (47M) |
| **Why deprecated** | Actual count from `clean_base_random_perm/head_scan_10k.json`: **9/32** heads with \|τ\|>0.9. Median \|τ\|=0.290. |
| **Correct value** | 9/32 heads with \|τ\|>0.9; 22 positive, 10 negative |
| **Impact** | Overstates signal density. Actual signal is sparse (9/32 strong heads), not dense. |
| **Appeared in** | `reports/evidence_package_20260613/02_unified_tables.md` Table G; `memory/daily-summary-20260613.md` |
| **Action** | ✅ Corrected in Table V6. Frame as "sparse order-bearing heads" not "dense uniform encoding." |

---

## D3: seed42 "narrow gap 0.013" / "Recovery unreliable"

| Field | Value |
|------|------|
| **Deprecated claim** | "Seed42 Recovery% is unreliable due to narrow gap 0.013" |
| **Why deprecated** | The "narrow gap" was computed using the wrong baseline (3.354) and wrong L2R ref. With correct baseline (3.466) and L2R (3.341), gap = 0.125 — normal and comparable to seed-123 (0.140). |
| **Correct gap** | 0.125 (seed42) vs 0.140 (seed-123) |
| **Impact** | Seed42 recovery was incorrectly dismissed as unreliable. Correct Recovery values are 106%/101%/67%. |
| **Appeared in** | `reports/evidence_package_20260613/05_conflicts_and_todo.md` ITEM 8; `TABLES_20260613.md` (pre-correction, seed42 note) |
| **Action** | ✅ Corrected. Seed42 Recovery is verified and robust. |

---

## D4: seed2 L2R reference = 3.370 (mixed-seed error)

| Field | Value |
|------|------|
| **Deprecated value** | seed2 L2R ref @50k = 3.370 (from `l2r_continuous_seed2`, seed=2) paired with random baseline seed=123 |
| **Why deprecated** | Seed mismatch: `random_baseline_continuous_jun08_seed2` has seed=123, `l2r_continuous_seed2` has seed=2. Mixed-seed comparison. |
| **Correct value** | seed-123 group L2R ref = **3.305** from `l2r_continuous_seed123` (seed=123, matches random baseline) |
| **Impact** | Brief window (2026-06-14 morning correction) where TABLES used 3.370, producing wrong Recovery% (e.g., 160% instead of 86%) |
| **Appeared in** | `TABLES_20260613.md` (2026-06-14 morning correction — partially wrong) |
| **Action** | ✅ Re-corrected. seed-123 group uses `l2r_continuous_seed123` (3.305). seed-2 group uses `l2r_continuous_seed2` (3.370) for CDL teacher only. |

---

## D5: ori-L2R as "upper bound"

| Field | Value |
|------|------|
| **Deprecated framing** | "ori-L2R is the upper bound for what order optimization can achieve" |
| **Why deprecated** | Multiple methods (frozen_β seed42 from10k @106%, CDL teacher) achieve Recovery > 100%, meaning their produced orders give lower loss than pure L2R. |
| **Correct framing** | "ori-L2R is a **reference point**, not an upper bound." |
| **Appeared in** | Implicit in Recovery% interpretation |
| **Action** | ✅ All verified tables label L2R as "reference". |

---

## D6: "random τ=0.49" as primary per-head signal

| Field | Value |
|------|------|
| **Deprecated framing** | Using aggregate/CDL-rollout τ≈0.49 (from shuffled-L2R diagnostic) as the primary attention emergence signal |
| **Why deprecated** | This is an aggregate diagnostic from `diag_shuffled_l2r_cdl.py`, NOT a per-head metric. The method pipeline uses **per-head selected B^{l,h}**, where |τ|≈1.0 for selected heads. Aggregate 0.49 is supplementary context. |
| **Correct framing** | "Primary signal source is selected per-head B^{l,h} with |τ|≈1.0. Aggregate τ≈0.49 is auxiliary." |
| **Appeared in** | `memory/cdl_shuffled_l2r_diagnostic_20260609.md` |
| **Action** | ✅ Framing corrected in Claims Lock. |

---

## D7: CDL teacher as matched-group comparison with frozen_β

| Field | Value |
|------|------|
| **Deprecated framing** | CDL teacher and frozen_β in the same seed group table without noting seed difference |
| **Why deprecated** | CDL teacher uses seed=2; frozen_beta seed-123 uses seed=123. They are different seeds with different model trajectories. |
| **Correct framing** | "CDL teacher provides supplementary cross-seed evidence. Direct seed-matched comparison not available (no seed=123 CDL teacher or seed=2 frozen_beta L0H2)." |
| **Appeared in** | `reports/evidence_package_20260613/02_unified_tables.md` Table C |
| **Action** | ✅ CDL teacher separated into own seed-2 group in `02_VERIFIED_TABLES.md`. |

---

## D8: Figures 1, 3, 4 as "paper-ready"

| Field | Value |
|------|------|
| **Deprecated claim** | fig1, fig3, fig4 are "paper/report ready" |
| **Why deprecated** | Files do not exist on disk (`analyses/figures/fig1_method_overview.png`, `fig3_catchup_curve.png`, `fig4_multistart_comparison.png` not found). |
| **Correct status** | "Needs generation." fig2 and shuffle_gran128 heatmaps exist. |
| **Appeared in** | `reports/evidence_package_20260613/figures_index.md` |
| **Action** | Generate figures from verified data. |

---

## D9: Any implication of 317M hook acceleration

| Field | Value |
|------|------|
| **Deprecated framing** | Any statement implying frozen hook training was done on 317M model |
| **Why deprecated** | No frozen_beta run with 16L/16H config exists. Only diagnostic scan at step 5000. |
| **Correct framing** | "317M model shows sparse order-bearing heads (6/256) in diagnostic scan. Frozen hook acceleration NOT tested at this scale." |
| **Appeared in** | Potential over-interpretation risk |
| **Action** | ✅ Explicitly stated in Table V8 and Claims Lock. |

---

## D10: "One g_β transfers across seeds/heads"

| Field | Value |
|------|------|
| **Deprecated framing** | Any implication that a single g_β works across seeds or heads |
| **Why deprecated** | Current evidence: g_β is trained per-seed, per-head. seed-123 uses L0H2; seed-42 uses L0H4. No cross-seed/head g_β validation exists. |
| **Correct framing** | "g_β is seed-dependent and head-dependent. Pipeline works with seed-dependently selected heads." |
| **Appeared in** | Risk of over-interpretation |
| **Action** | ✅ Explicitly in "CANNOT CLAIM" section. |
