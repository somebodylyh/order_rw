# 08 — Next Steps After Verification

> Recommended actions to resolve remaining issues and prepare for paper writing.
> Ordered by priority / cost.

---

## Before Paper Writing (P0)

### 1. Fix TABLES_20260613.md in project root
- **Issue**: 2026-06-14 morning correction partially wrong — changed seed2 L2R from 3.305 to 3.370 (seed mismatch)
- **Fix**: Change seed2 L2R ref BACK to 3.305 (`l2r_continuous_seed123`, seed=123). Keep seed42 correction (3.354→3.466). Use separate seed-2 group for CDL teacher.
- **Cost**: 5 min edit

### 2. Generate missing figures (U5)
- **Issue**: fig1, fig3, fig4 referenced but not on disk
- **Fix**: Run plotting scripts using verified numbers from `02_VERIFIED_TABLES.md`
- **Cost**: ~30 min scripting

### 3. Save g_β sanity primary output (U2)
- **Issue**: No JSON output from `gbeta_input_sanity_final.py`
- **Fix**: Re-run script, save output to `analyses/gbeta_input_sanity_final_output.json`
- **Cost**: ~5 min GPU

---

## Before Advisor Update (P1)

### 4. Fix deprecated "28/32" number in memory and existing package (D2)
- **Issue**: `daily-summary-20260613.md` and existing `02_unified_tables.md` claim 28/32 heads with |τ|>0.9
- **Fix**: Update to 9/32. Frame as "sparse" not "dense."
- **Cost**: Edit 2 files

### 5. Resolve CDL teacher seed mismatch (U1)
- **Option A**: Generate seed=123 CDL teacher from `random_baseline_continuous_jun08_seed2` ckpt
- **Option B**: Keep as cross-seed supplementary, add explicit caveat
- **Cost**: Option A = ~1h GPU for CDL extraction; Option B = edit documentation

### 6. Compute per-head τ for shuffled-L2R model (U6)
- **Issue**: Shuffled-L2R per-head scan data not available
- **Fix**: Run `per_head_order_scan.py` on `shuffled_l2r_continuous_jun05` ckpt
- **Cost**: ~15 min GPU

---

## Before Paper Submission (P2)

### 7. Third seed for statistical completeness
- **Issue**: Currently 2 matched seed groups (seed-123, seed-42)
- **Fix**: Train seed=1 random baseline + L2R + frozen_beta from10k
- **Cost**: ~2 days GPU

### 8. End-to-end label-free pipeline (U3)
- **Issue**: g_β training still uses CDL teacher supervision
- **Fix**: Implement unsupervised g_β training or accept as limitation
- **Cost**: Unknown — research risk

### 9. 317M frozen hook
- **Issue**: No hook training at scale
- **Fix**: Train g_β on 317M model, run frozen_beta
- **Cost**: ~3 days GPU, high risk

---

## Immediate TODOs (today / tonight)

- [x] Re-correct `TABLES_20260613.md`: seed2 L2R ref → 3.305 (seed-123), add separate seed-2 group
- [x] Re-run g_β sanity script and save output JSON → `raw/gbeta_input_sanity_final.json`
- [x] Launch matched CDL teacher seed-123 (from10k/20k/40k) on GPU1, ETA ~8h
- [ ] Update `memory/daily-summary-20260613.md`: fix 28/32 → 9/32
- [ ] Update `reports/evidence_package_20260613/02_unified_tables.md`: fix 28/32 → 9/32
- [ ] Update CLAIMS_LOCK_20260613.md in project root to match 05_FINAL_CLAIMS_LOCK.md
- [ ] Tomorrow: verify CDL teacher seed-123 results, recompute matched Recovery%

---

## Decision Items for Boss

1. **Paper scope**: Text-focused or multimodal?
2. **Compute**: Third seed, 317M hook, or both?
3. **CDL teacher**: Generate seed-matched runs or accept cross-seed?
4. **Label-free**: Invest in end-to-end or leave as limitation?
