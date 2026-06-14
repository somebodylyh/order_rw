# 01 — Claims Lock

> Status: 2026-06-13. Cross-verified against primary sources.
> ori-L2R = reference, NOT upper bound.
> Per-head τ, NOT heavy τ.

---

## CAN CLAIM

### 1. g_β reads structured B, not a constant L2R prior

**Supporting evidence** (Table A):
- Real B (random-order model): g_β τ_vs_L2R = **0.97**
- Gaussian B, entry-shuffled B, row+col shuffled B: all **≈0**
- Gaussian family pairwise τ: **0.0003**
- Zero B τ=1.0 explained as tie-breaking artifact (zero margin → CDL picks first = L2R)
- Non-L2R subset: **+0.11 Δ** above L2R-prior baseline

**Source**: `analyses/gbeta_input_sanity_final.py`; `memory/gbeta-input-sanity-20260611.md`

**Allowed wording**: "g_β reads structured attention graphs rather than a fixed L2R prior." "g_β output depends on the specific B matrix structure, not on a memorized constant."

**Forbidden wording**: "g_β learns the L2R order." "g_β memorizes L2R." "g_β is a prior."

### 2. Random-order AO-GPT contains sparse, order-bearing attention heads

**Supporting evidence** (Table B):
- clean_base_random_perm (64-block): individual per-head CDL τ = **+1.000** (L0H0, L0H6, L1H1...)
- Signal persists across training granularities (32-group, 128-group): per-head τ = **±1.000**
- Shuffled-L2R control: per-head τ falls to ~0 (wrong physical order not recovered)
- These heads are NOT fixed by index — they are seed-dependent and must be selected/auditioned
- In small model (47M): many order-bearing heads (28/32 with |τ|>0.9)
- In large model (317M): sparse order-bearing heads (6/256 with |τ|>0.9)

**Source**: `probe_results/clean_base_random_perm/head_scan_10k.json`; `probe_results/shuffle_gran_*/head_scan_10k.json`; `probe_results/large_random_baseline_16l16h1024d/head_scan_step5000.json`

**Method alignment**: The method pipeline uses **per-head selected attention graphs** B^{l,h} → CDL → g_β → hook. Therefore the primary emergence evidence is the sparse per-head signal, not aggregate diagnostics. A historical aggregate/CDL-rollout number τ≈0.49 (from `diag_shuffled_l2r_cdl.py`) is supplementary context, not the primary signal metric.

**Allowed wording**: "Random-order AO-GPT contains sparse order-bearing attention heads whose induced graph orders align strongly with physical L2R." "These heads are not fixed by index and must be selected/auditioned."

**Forbidden wording**: "Random-order attention overall has τ=0.49." (aggregate metric, not per-head selected signal). "The signal is universally present in all heads." "Shuffled-L2R encodes no order at all." (We only know it doesn't encode physical L2R.)

### 3. Frozen g_β accelerates training across 2 seeds and multiple resume points

**Supporting evidence** (Tables C, D, E):
- Seed2 L0H2 from10k: Recovery 86.0%, Step Saving **41.7%**
- Seed2 L0H2 from20k: Recovery 83.0%, Step Saving **35.7%**
- Seed42 L0H4 from10k: Recovery **106.1%** (beats ori-L2R ref), Step Saving **39.3%**
- Seed42 L0H4 from20k: Recovery **101.5%**, Step Saving **33.3%**
- Seed1 L1H1 from10k: val_ori_l2r@50k = **3.305** (matches ori-L2R reference)
- Seed2 v3 L0H7 from10k: val_ori_l2r@50k = **3.300** (beats reference)

**Seed42 baseline group verified** (2026-06-13 23:45): random baseline = `random_baseline_continuous_jun05` (@50k=3.466), ori-L2R ref = `l2r_continuous_jun05` (@50k=3.341), gap=0.125 (normal, cf. seed2 gap=0.140). Previous version incorrectly used `frozen_beta_random_baseline_jun05` (a frozen_beta run, not a random baseline) as the reference — corrected.

**Conservative range**: Step Saving **35–42%** across 2 seeds, from10k+from20k.

**Source**: eval_curve.tsv files in `probe_results/frozen_beta_*`, `probe_results/random_baseline_continuous_jun05/`, `probe_results/l2r_continuous_jun05/`

**Allowed wording**: "Frozen g_β accelerates training across two seeds and multiple resume points." "Conservative step saving of 35–42%."

**Forbidden wording**: "g_β works on any seed." "Universally accelerates training." "One controller fits all."

### 4. g_β@10k transfers to later checkpoints within same seed/head

**Supporting evidence**: All frozen_β multi-start runs (from10k, from20k, from40k) use the same g_β trained at step 10k. The readout remains effective when plugged into step 20k/40k models.

**Source**: Config files show same `frozen_beta_ckpt` for all resume points within a seed.

**Allowed wording**: "g_β@10k transfers to later checkpoints within the same seed and head." "Attention graph structure is stable enough for cross-step readout."

**Forbidden wording**: "g_β transfers across seeds." "g_β transfers across heads." "One g_β works everywhere."

### 5. Phenomenon survives granularity and scale diagnostics

**Supporting evidence** (Tables F, G):
- Aggregation granularity (32/64/128 block): τ > 0.96 in all cases
- Training granularity (32/64/128 group): per-head τ = ±1.0 in all cases
- 317M large model (16L/16H/1024d): |τ|>0.9 heads exist (6/256, best τ=+0.955)

**Source**: `analyses/block_granularity_scan_results/`; `probe_results/shuffle_gran_*/head_scan_10k.json`; `probe_results/large_random_baseline_16l16h1024d/head_scan_step5000.json`

**Allowed wording**: "Not a 64-block artifact. Not a 47M-scale artifact."

**Forbidden wording**: "317M frozen hook works." (Diagnostic only, no hook training.)

---

## CANNOT CLAIM

1. **One g_β transfers across seeds or heads.** Current evidence: g_β is seed-dependent and head-dependent.
   - Correct: "Pipeline works with seed-dependent selected heads."

2. **ori-L2R is an upper bound.** CDL teacher from10k achieves Recovery > 100% (beats L2R reference).
   - Correct: "ori-L2R is a reference point."

3. **Shuffled-L2R encodes no order at all.** Only shown that physical L2R signal is not recovered.
   - Correct: "Shuffled-L2R control does not recover physical order."

4. **Fully label-free head selection is end-to-end verified.** Audition selects head without oracle τ, but not end-to-end validated.
   - Correct: "Head selected via cheap diagnostic signals; oracle τ used for post-hoc validation."

5. **Image side is solved.** Image has diagnostic results (D_manh, CDL teacher ablation) but no frozen hook training.

6. **317M frozen hook acceleration is established.** Only diagnostic scan exists. No hook training run.

7. **Per-sample permutation / document-level unscrambling.** Current protocol uses fixed block permutation for all samples within a batch. No evidence for per-sample order adaptation.

---

## NEED BOSS DECISION

1. **Paper scope**: Text-focused paper (image as limitation/discussion) OR multimodal paper?
2. **Compute allocation**: Third seed (statistical completeness) OR 317M frozen hook (scale demonstration)?
3. **Method positioning**: "Attention-derived order controller for faster AO-GPT training" OR "Emergent data-structure signal in attention graphs"?
4. **Label-free audition**: Finalize before paper writing OR leave as future work?
5. **Image teacher redesign**: In scope for this paper OR deferred?
