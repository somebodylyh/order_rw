# 05 — Final Claims Lock (Verified)

> Based on verification pass 2026-06-14.
> All claims cross-referenced against primary sources (config.json, eval_curve.tsv, head_scan JSON).

---

## CAN CLAIM

### 1. Random-order AO-GPT develops sparse order-bearing attention heads

**Verified evidence**:
- 9/32 heads with |τ|>0.9 in clean_base_random_perm @10k (Table V6)
- Best head: L0H0 τ=+1.000, pw_tau=1.000 (all 100 samples produce identical L2R order)
- Signal is sparse, not dense — median |τ|=0.290, 22 positive / 10 negative heads
- Heavy τ (0.876) is dominated by a few strong positive heads — do NOT use as primary metric
- Shuffled-L2R control: physical τ falls to ≈0 (Table V6 note — per-head data pending U6)

**Source**: `block_lo_arm_order_network/probe_results/clean_base_random_perm/head_scan_10k.json`

**Allowed**: "Sparse order-bearing attention heads emerge under random-order training." "Individual heads reach |τ|=1.0 against physical L2R at early training."

**Forbidden**: "All heads encode L2R." "28/32 heads have |τ|>0.9." "Dense uniform encoding." "τ=0.49 as primary signal." "Signal persists unchanged to convergence."

⚠️ **Transient qualification (2026-06-14)**: Per-head CDL τ_vs_L2R decays to noise level by step 50k (all |τ|<0.06, heavy τ=−0.013; M=40, none_mode=b0, seed-123 continuous). The strong per-head signal at 10k is an early-training phenomenon. Training acceleration (Claim 3) is not invalidated — g_β@10k→40k transfer (59–67% recovery) provides indirect mechanism-persistence evidence. Paper should qualify: "Early-training order-bearing heads (≤10k) provide the signal source for g_β training; the resulting controller remains effective at convergence even as raw per-head τ attenuates."

---

### 2. g_β reads structured selected-head B^{l,h}, not a fixed L2R prior

**Verified evidence**:
- Real B → g_β τ=+0.9675 (strong readout) (Table V5)
- Gaussian / shuffled / destroyed B → τ≈0 (no signal) (Table V5)
- Gaussian family pairwise τ=0.0003 (no fixed prior across diverse inputs) (Table V5)
- Zero B τ=1.0 explained: margin=0 → CDL tie-breaking picks L2R by construction (Table V5)
- Raw JSON saved: `reports/evidence_package_20260613_verified/raw/gbeta_input_sanity_final.json`

⚠️ Non-L2R subset analysis is **excluded from the main claim**. The n=4 sample size is insufficient for statistical conclusion. The destroyed/randomized B controls alone suffice to establish the claim.

**Source**: `analyses/run_gbeta_sanity_save_json.py` → `raw/gbeta_input_sanity_final.json` (2026-06-14)

**Allowed**: "g_β reads structured attention graphs rather than a fixed L2R prior." "g_β output depends on specific B structure."

**Forbidden**: "g_β memorizes L2R." "g_β is a constant prior."

---

### 3. Frozen g_β accelerates training across two matched seed groups and multiple resume points

**Verified evidence**:
- seed-123 L0H2: Recovery from10k=86.0%, from20k=83.0%, from40k=58.7% (Table V3)
- seed-42 L0H4: Recovery from10k=106.1%, from20k=101.5%, from40k=66.9% (Table V3)
- Step saving: frozen_β from10k 39–42%, from20k 33–36% across both seeds (Table V4)
- Conservative range (frozen_β, from10k+from20k, both seeds): 33–42% step saving
- Frozen_β from10k beats L2R reference in step saving (41.7% vs L2R's 64.3% — but note L2R starts from step 0, not a fair speed comparison)

**Source**: `eval_curve.tsv` from `probe_results/frozen_beta_seed2_from*_l0h2/` and `probe_results/frozen_beta_random_jun05_from*/`

**Allowed**: "Frozen g_β accelerates training across two seeds and multiple resume points." "Conservative step saving of 33–42%." "Recovery >100% means the method's order beats L2R for the model."

**Forbidden**: "Universally accelerates training." "One controller fits all seeds."

---

### 4. g_β@10k transfers to later checkpoints within the same seed/head

**Verified evidence**:
- All seed-123 frozen_β runs (from10k, from20k, from40k) use the same g_β checkpoint from step 10k (config.json verified)
- All seed-42 frozen_β runs use the same g_β checkpoint from step 10k (config.json verified)
- Recovery remains positive at from20k and from40k (Table V3)

**Source**: `config.json` `frozen_beta_ckpt` field in each frozen_beta run directory.

**Allowed**: "g_β@10k transfers to later checkpoints within the same seed and head." "Attention graph structure is stable enough for cross-step readout."

**Forbidden**: "g_β transfers across seeds." "g_β transfers across heads."

---

### 5. Signal survives granularity and 317M scale diagnostics

**Verified evidence**:
- Aggregation granularity (32/64/128 block): heavy τ > 0.96 all three (Table V7)
- Training granularity (32/64/128 group): per-head |τ|≈1.0 in all three (Table V6)
- 317M (16L/16H/1024d): 6/256 heads with |τ|>0.9, best L0H10 τ=+0.955 (Table V8)

**Source**: `block_granularity_scan_results/scan_step5000_M100.json`; `head_scan_*.json` from shuffle_gran_* and large model

**Allowed**: "Not a 64-block artifact. Not a 47M-scale artifact." "Signal persists under granularity and scale variation."

**Forbidden**: "317M frozen hook works." (NO hook training exists — Table V8)

---

## CANNOT CLAIM

1. **One g_β transfers across seeds or heads.** g_β is seed-dependent (seed-123 uses L0H2, seed-42 uses L0H4). No cross-seed/head validation.
   - **Correct**: "Pipeline works with seed-dependently selected heads."

2. **ori-L2R is an upper bound.** Frozen_β seed42 from10k achieves 106.1% Recovery (beats L2R). CDL teacher also beats L2R.
   - **Correct**: "ori-L2R is a reference point."

3. **Shuffled-L2R encodes no order at all.** Only shown that physical L2R is not recovered. Could encode the shuffled order.
   - **Correct**: "Shuffled-L2R control does not recover physical order." (U6)

4. **Fully label-free head selection is end-to-end verified.** Head SELECTION uses cheap signals; g_β TRAINING uses CDL teacher. Full chain not demonstrated.
   - **Correct**: "Head selected via cheap diagnostic signals; g_β trained with CDL teacher supervision." (U3)

5. **Image side is solved.** Image has diagnostic results but no frozen hook training.

6. **317M frozen hook acceleration is established.** Only diagnostic scan exists. No hook training. (Table V8)

7. **Per-sample permutation / document-level unscrambling.** Protocol uses fixed block permutation. No evidence for per-sample adaptation.

8. **Permutation-robust likelihood improvement.** The method specializes the model toward a single deployment order. Under random evaluation orders (`val_unstructured_order`), frozen_β degrades relative to random baseline (Δ +0.42 seed-123, Δ +1.28 seed-42). This is expected order-specialization trade-off, NOT a hidden failure.
   - **Correct**: "We improve canonical-order training efficiency. The model is not claimed to improve under all permutations." (Table V9)
   - **Paper**: Report val_unstructured degradation in limitation/analysis section.

---

## NEED DECISION

1. **Paper scope**: Text-focused (image as limitation) OR multimodal extension?
2. **Compute allocation**: Third seed (statistical completeness) OR 317M hook (scale demonstration)?
3. **CDL teacher handling**: ✅ IN PROGRESS — `cdl_teacher_seed123_from10k_l0h2` running (31k/60k @22:12, ETA ~6.8h). Check 2026-06-15 morning.
4. **Label-free audition**: Finalize end-to-end before paper OR leave as limitation?
5. **Figure regeneration**: Generate fig1/fig3/fig4 from verified data (U5).
6. **50k head scan follow-up**: M=40 scan done (signal decayed). Larger M (200+) for better noise floor if reviewer questions M=40 sensitivity.
