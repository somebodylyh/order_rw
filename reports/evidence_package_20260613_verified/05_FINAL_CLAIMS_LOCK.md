# 05 — Final Claims Lock (Verified)

> Based on verification pass 2026-06-14.
> All claims cross-referenced against primary sources (config.json, eval_curve.tsv, head_scan JSON).

---

## CAN CLAIM

### 1. Random-order AO-GPT develops sparse order-bearing attention heads, and the signal survives B1/predictor-aligned extraction

**Verified evidence**:
- B0 clean-base @10k: 9/32 heads with |τ|>0.9; best head L0H0 τ=+1.000, pw_tau=1.000.
- B1/predictor clean-base ladder: L0H0 τ=+1.000 at 10k, 50k, and 60k; |τ|>0.9 count is 7/32 at 10k and 6/32 at 50k/60k.
- B1/predictor latest continuous run (seed=124): final 60k best head L0H4 τ=+0.957589, mean_pairwise_tau=0.930060; L0H4 is top in 5768/6001 tracked steps.
- 317M B0 scale diagnostic: 6/256 heads with |τ|>0.9, best L0H10 τ=+0.955.
- Signal remains sparse, not dense. Strong-head counts should be reported per protocol rather than merged.
- Shuffled-L2R control remains B0 legacy evidence; B1 shuffled-L2R control not found.

**Source**:
- B0 clean-base: `block_lo_arm_order_network/probe_results/clean_base_random_perm/head_scan_10k.json`
- B1/predictor clean-base ladder: `block_lo_arm_order_network/batch_readout/logs/per_head_scan_b1/ckpt10000_seed*.json`, `ckpt50000_seed*.json`, `ckpt60000_seed*.json`
- B1/predictor continuous seed=124: `block_lo_arm_order_network/probe_results/random_baseline_b1_headscan_seed124/head_signal_all.tsv`
- B0 317M: `block_lo_arm_order_network/probe_results/large_random_baseline_16l16h1024d/head_scan_step5000.json`

**Allowed**: "Sparse order-bearing attention heads emerge under random-order training." "The order-bearing signal survives collaborator-aligned B1/predictor-frame diagnostics." "B1 diagnostics show strong order-bearing heads persist to late checkpoints in the clean-base ladder and also appear in a continuous 60k run."

**Forbidden**: "All heads encode L2R." "28/32 heads have |τ|>0.9." "Dense uniform encoding." "The B1 diagnostic proves B1 frozen-hook acceleration." "The seed-123 continuous B0 50k attenuation is contradicted by the clean-base B1 ladder." These are different protocols/runs.

⚠️ **Protocol qualification (2026-06-15)**: The old 50k attenuation result is B0 on the seed-123 continuous run (`random_baseline_continuous_jun08_seed2/head_scan_50k.json`). The new late-checkpoint persistence result is B1/predictor on the clean-base ladder, plus B1/predictor tracking on a separate continuous seed=124 run. Therefore the safest statement is: "B1 diagnostics show that strong order-bearing heads persist to late checkpoints in the clean-base ladder and also appear in a continuous 60k run." Do not claim exact matched persistence for seed-123 continuous L0H2 under B1 unless that scan is rerun.

---

### 2. Legacy B0 g_β reads structured selected-head B^{l,h}, not a fixed L2R prior

**Verified evidence**:
- Real B under the legacy B0 controller path → g_β τ=+0.9675.
- Gaussian random B → τ=-0.0034; entry-shuffled B → τ=+0.0165; row/col shuffled B → τ=+0.0127.
- Gaussian family pairwise τ=0.0003 (no fixed prior across diverse inputs).
- Zero B τ=1.0 explained: margin=0 → CDL tie-breaking picks L2R by construction.
- Raw JSON saved: `reports/evidence_package_20260613_verified/raw/gbeta_input_sanity_final.json`
- B1/predictor diagnostics independently confirm strong order-bearing structure is not specific to B0 extraction, but no B1 g_β sanity run was found.

⚠️ Non-L2R subset analysis is **excluded from the main claim**. The n=4 sample size is insufficient for statistical conclusion. The destroyed/randomized B controls alone suffice to establish the claim.

**Source**: `analyses/run_gbeta_sanity_save_json.py` → `raw/gbeta_input_sanity_final.json` (2026-06-14)

**Allowed**: "Under the legacy B0 controller path, g_β reads structured attention graphs rather than a fixed L2R prior." "B1/predictor diagnostics independently show the upstream attention-order signal is not specific to B0 extraction."

**Forbidden**: "g_β memorizes L2R." "g_β is a constant prior." "B1 proves g_β reads B1 graphs." No B1 g_β sanity result was found.

---

### 3. Legacy B0 frozen g_β accelerates training across two matched seed groups and multiple resume points

**Verified evidence**:
- seed-123 L0H2: Recovery from10k=86.0%, from20k=83.0%, from40k=58.7% (Table V3)
- seed-42 L0H4: Recovery from10k=106.1%, from20k=101.5%, from40k=66.9% (Table V3)
- Step saving: frozen_β from10k 39–42%, from20k 33–36% across both seeds (Table V4)
- Conservative range (frozen_β, from10k+from20k, both seeds): 33–42% step saving
- Frozen_β from10k beats L2R reference in step saving (41.7% vs L2R's 64.3% — but note L2R starts from step 0, not a fair speed comparison)
- These acceleration results use the legacy B0 selected-head/controller extraction path.

**Source**: `eval_curve.tsv` from `probe_results/frozen_beta_seed2_from*_l0h2/` and `probe_results/frozen_beta_random_jun05_from*/`

**Allowed**: "Legacy B0 frozen g_β accelerates training across two seeds and multiple resume points." "Conservative step saving of 33–42%." "Recovery >100% means the method's order beats L2R for the model."

**Forbidden**: "Universally accelerates training." "One controller fits all seeds." "The acceleration runs already use B1." "B1-controller acceleration is proven." A B1 hook rerun would be needed for that claim.

---

### 4. g_β@10k transfers to later checkpoints within the same seed/head under the legacy B0 hook path

**Verified evidence**:
- All seed-123 frozen_β runs (from10k, from20k, from40k) use the same g_β checkpoint from step 10k (config.json verified)
- All seed-42 frozen_β runs use the same g_β checkpoint from step 10k (config.json verified)
- Recovery remains positive at from20k and from40k (Table V3)
- The transfer evidence is from the legacy B0 hook path, not from B1/predictor hook runs.

**Source**: `config.json` `frozen_beta_ckpt` field in each frozen_beta run directory.

**Allowed**: "Under the legacy B0 hook path, g_β@10k transfers to later checkpoints within the same seed and head." "Attention graph structure is stable enough for cross-step readout under the tested controller path."

**Forbidden**: "g_β transfers across seeds." "g_β transfers across heads." "B1 g_β@10k transfer is proven."

---

### 5. The signal is robust across granularity and scale diagnostics, but 317M and B1 hook acceleration remain untested

**Verified evidence**:
- B0 aggregation granularity (32/64/128 block): heavy τ > 0.96 all three (Table V7)
- B0 training granularity (32/64/128 group): per-head |τ|≈1.0 in all three (Table V6)
- B0 317M (16L/16H/1024d): 6/256 heads with |τ|>0.9, best L0H10 τ=+0.955 (Table V8)
- B1/predictor small-model diagnostics strengthen the mechanism story, but no B1 317M or B1 hook acceleration result was found.

**Source**: `block_granularity_scan_results/scan_step5000_M100.json`; `head_scan_*.json` from shuffle_gran_* and large model

**Allowed**: "Not a 64-block artifact under B0 granularity diagnostics." "Not a 47M-scale artifact under the B0 317M diagnostic." "B1/predictor diagnostics strengthen the mechanism story in the small model."

**Forbidden**: "317M frozen hook works." "317M B1 scan exists." "B1 hook acceleration works." No such runs were found.

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

9. **B1 replaces all B0 results.** B1 currently strengthens attention-signal diagnostics. Existing g_β sanity and frozen-hook acceleration remain legacy B0 controller-path results.
   - **Correct**: "B1 strengthens the mechanism story, but does not replace the B0 acceleration evidence."

10. **Existing B1 files use `none_mode=b1`.** Current B1 result files use `none_mode=predictor`; the code also contains a separate `none_mode=b1` physical-remap variant, but no result file using that enum was found.
   - **Correct**: "Existing B1 result files use the B1/predictor-aligned diagnostic convention."

---

## NEED DECISION

1. **Paper scope**: Text-focused (image as limitation) OR multimodal extension?
2. **Compute allocation**: Third seed (statistical completeness) OR 317M hook (scale demonstration)?
3. **CDL teacher handling**: ✅ IN PROGRESS — `cdl_teacher_seed123_from10k_l0h2` running (31k/60k @22:12, ETA ~6.8h). Check 2026-06-15 morning.
4. **Label-free audition**: Finalize end-to-end before paper OR leave as limitation?
5. **Figure regeneration**: Generate fig1/fig3/fig4 from verified data (U5).
6. **50k head scan follow-up**: M=40 scan done (signal decayed). Larger M (200+) for better noise floor if reviewer questions M=40 sensitivity.
7. **B1 naming cleanup**: decide whether "B1" means `none_mode=predictor` or the code enum `none_mode=b1`; until then write "B1/predictor-aligned diagnostic."
8. **Optional B1 hook rerun**: not required for existing B0 acceleration claims; needed only if claiming B1-controller acceleration.
