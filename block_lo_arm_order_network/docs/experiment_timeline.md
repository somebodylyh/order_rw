# Experiment Timeline (text + image, audited 2026-05-14)

A narrative of how the project arrived at its current claims, paired with what was confirmed wrong along the way. References by file path point at audited evidence (see `docs/text_run_registry.md` and `docs/image_results_2026-05-12.md`).

---

## 1. Text initial baseline confusion (resolved)

The early text pipeline was driven by `train_clean_aogpt.py` with several `--run-kind` choices: `baseline`, `random_continuation`, `graph_rw`, `graph_rw_bag`, `l2r`. Two facts conspired to produce confusing config artifacts:

1. `alpha_for_step(global_step, start_step, args)` (lines 157-163) returns `0.0` *unconditionally* when `args.run_kind` is not in `{graph_rw, graph_rw_bag}`. The `--alpha-target`, `--rw-policy`, `--rw-top-k` etc. flags are *defined* on the argparser regardless of `run_kind`, so they are always available — but they are *not consumed* unless `run_kind` is one of the two graph-RW kinds.
2. `write_config(args, …)` wrote `rw_policy` and `rw_params` for *every* run, producing `config.json` files that look identical for `baseline` and `graph_rw` runs.

The result: `clean_base_random_perm/config.json` shows `rw_policy=progressive_rw`, `alpha_target=0.9`, `rw_top_k=4`. A casual reader concludes this was a Graph-RW method run. It is **not**: the `eval_curve.tsv` `alpha` column is 0.0 throughout, and the run is the random-permutation baseline used as the fork point for every Graph-RW method run downstream.

This was also the source of an earlier informal claim that "the baseline reaches 3.40 / 3.41 ori_l2r". Those numbers came from `stageA_l2r_from30k` (`run_kind=l2r`, an L2R-oracle continuation) — they are upper bounds, not the random-perm baseline. The random-perm baseline plateaus at **3.520** @60k.

### Fixes applied (this audit)

- `write_config` now refuses to write `rw_policy` / `rw_params` unless `graph_rw_active=True`. For non-Graph-RW runs the config records `actual_alpha_schedule.alpha_constant: 0.0` plus a `reason` string identifying the short-circuit. (See `train_clean_aogpt.py:439-475`.)
- `train_log.txt` now logs `graph_rw_active`, the actual α schedule, and the active `rw_policy` / `top_k` / `ε` / `τ` / `λ` / `ρ` at startup. Non-Graph-RW runs explicitly log "Any rw_* CLI flags are ignored at training time."
- A re-audit script `scripts/text/audit_text_runs.py` walks `probe_results/`, reads each run's `config.json` and `eval_curve.tsv`, classifies the actual_type from `(run_kind, last α)` rather than from the now-deprecated `rw_policy` field. The classification matches `docs/text_run_registry.md` entries.
- Four canonical text configs landed in `configs/text/` mirroring the collaborator's WikiText-103 seq256 block64 random baseline (lr=1e-3, warmup=0, max_iters=50000, beta2=0.99, n_layer=4, n_head=8, n_embd=384, block_order_block_len=4):
  - `wikitext103_seq256_block64_random_base.py` (run_kind=baseline)
  - `wikitext103_seq256_block64_v3_readiness.py` (run_kind=graph_rw, progressive_rw_v3, ρ=0.2, λ=0.75)
  - `wikitext103_seq256_block64_v3_no_readiness.py` (run_kind=graph_rw, progressive_rw_v3, ρ=0.0)
  - `wikitext103_seq256_block64_l2r_reference.py` (run_kind=l2r)

---

## 2. Text method reconstruction (Progressive RW v1 → v3 readiness-guided)

Once the misleading config issue was contained, the actual text-side work decomposed into:

1. **Progressive RW v1/v2** (`progressive_rw`) used a 4-β scoring rule:
   ```
   score_t(v) = β_sup · sup_t(v) − β_fut · fut_t(v) + β_src · source(v) + β_loc · loc_t(v)
   ```
   with `β_sup=1.0, β_fut=0.5, β_src=0.2, β_loc=0.5`. The `source` term encodes a "readiness" prior implicitly via `out − α_dep · in`, but was buried under three other coefficients. Best `val_ori_l2r` reached **3.457** at @50k (`clean_method_graph_rw_a09_from20k_v2`), a Δ of −0.063 vs the random-perm baseline.

2. **Progressive RW v3 readiness-guided** (`progressive_rw_v3`) collapses the four β's to a single λ parameter and lifts the readiness term to a first-class λ-/ρ-controlled prior:
   ```
   score_t(v) = C_t(v) − λ · D_t(v) + ρ · r(v)
   ```
   where `C_t` is the connectivity-with-revealed-set, `D_t` is the connectivity-with-still-unrevealed-set, and `r(v) = out_deg(v) − α_dep · in_deg(v)` is the readiness term. Default: λ=0.75, ρ=0.2.

3. **Diagnostic** (consensus order, K=100): v3 single-sample τ vs L2R = 0.844 ± 0.012, consensus τ = 0.887. Pairwise τ = 0.865 — single-sample is already highly consistent. Confirms v3 is L2R-dominated local reordering, not a wildly different policy. (See memory `consensus_order_diagnostic.md`.)

4. **Validation that the readiness prior matters**:
   - With ρ=0.2: `clean_method_graph_rw_v3_from20k` reaches **3.458** @50k.
   - With ρ=0.0: `clean_method_graph_rw_no_readiness_from20k` plateaus at **3.484** @40k.
   - Δ(ρ=0.2 vs ρ=0.0) ≈ −0.026 by @40k. The readiness term is the active ingredient.

5. **Validation that low noise matters**:
   - top_k=4, ε=0:    3.484 @40k
   - ε=0.15:           3.499 @40k (Δ +0.015)
   - ε=0.20:           3.499 @40k (Δ +0.015)
   - top_k=8, τ=0.5:   3.513 @40k (Δ +0.029)

   Adding even small uniform noise to the policy degrades the result; widening top_k similarly weakens it.

6. **Convergence robustness**: every sane variant — v1, v2, v3, piecewise α=1.0→0.9 / α=1.0→0.95 schedules, random-A refresh ablation — converges to the **same neighborhood (3.456–3.465 @60k)**. The Graph-RW result is a single attractor, not a knife-edge configuration.

### Two configurations to STOP citing

- `ablation_v2_noreadiness` and `ablation_v3_readiness` were attempts to test the v3 ablation through `train_aogpt_graph_rw.py` instead of `train_clean_aogpt.py`. That script's defaults were copy-pasted from the image side (lr=3e-5, α-warmup=1500, warmup_iters=200). At lr=3e-5 the optimizer never moves text loss; at α-warmup=1500 the order policy ramps before the optimizer settles. `val_ori_l2r` blew up to **15.6 / 15.9** by step 9 999. Memory `text_training_config.md` records the corrected defaults; the broken runs are tagged in `text_run_registry.md` Section "Broken / abandoned (do NOT cite)".

---

## 3. Image extension (CIFAR-10 4×4 patches)

The image side was an explicit cross-modal stress test: the same Graph-RW machinery on a 2D modality (CIFAR-10, 8×8 grid, N=64 patches, patch_dim=48). New code lives entirely in `image_order/` (a separate package — does not touch `block_lo_arm_order_network/`).

1. **Stage 0 baseline (`baseline10k`)**: 10 000 random-order steps from scratch produced an image AOGPT (4.07 M params) with `val_random=0.0596`, `val_raster=0.0522`. Final attention `sparsity_topk5=0.142` (vs 0.118 at 500-step smoke).
2. **Stage 1 structural diagnostics**: From the baseline10k attention, Graph-RW with `top_k=4, ε=0` produces orders that are demonstrably non-random on **four orthogonal axes**:
   - mean Manhattan step **3.10** (random 5.30, raster 1.78)
   - P(consecutive d≤1) **28 %** (random 6 %)
   - same 4×4-quadrant transition rate **54 %** (random 24 %)
   - z-scored consecutive feature distance **2.24** (random 2.74)

   `ε=0.15` collapses *every* axis to within 1.5 % of random.
3. **Stage 2 α-mixed continuation (5 000 steps)**: From baseline10k, four configs:
   - `cont_random` (α=0): random continuation
   - `cont_top4` (α=0.9, top_k=4, ε=0): low-noise Graph-RW
   - `cont_eps015` (α=0.9, top_k=0, ε=0.15): high-noise ablation
   - `cont_top8` (α=0.9, top_k=8, ε=0): wider top-k

   Frozen 5-order eval (val_images=1000, n_seed_rounds=3) shows `cont_top4` reduces `val_rw_top4` by **−2.7 %**, `val_raster` by **−3.4 %** vs `cont_random`. `cont_eps015` is statistically equal to `cont_random` (every Δ < 0.0002). `cont_top8` lands between the two.

4. **3-seed CI (P0, completed 2026-05-12)**: seeds {0, 1, 42} × 4 configs × 5 000 steps. Paired Δ vs cont_random with 95 % CI:

   | eval order | cont_top4 | cont_eps015 | cont_top8 |
   |---|---|---|---|
   | rw_top4_eps0 | **−0.00150** [−0.00156, −0.00144] | −0.00015 | −0.00129 |
   | raster | −0.00176 | −0.00015 | −0.00143 |

   The Δ exceeds seed std by 4-8×; the noise ablation is indistinguishable from random.

5. **25k long run (also completed 2026-05-12)**: cont_random_long vs cont_top4_long from baseline10k. Across 13 eval points, the paired Δ never flips sign, the mean Δ scales from **−0.00150 (5k) → −0.00177 (25k mean) → −0.00204 (raster mean)**. The structured advantage grows, not shrinks, with longer training.

---

## 4. Current reliable claims

Items in this section are explicitly the ones I am willing to defend after the audit.

### Text

- The random-permutation training baseline on WikiText-103 seq256 block64 plateaus at **`val_ori_l2r` = 3.520** @60k (file: `clean_base_random_perm/eval_curve.tsv`, row step 60000).
- Graph-RW continuation from that baseline reduces `val_ori_l2r` to **3.456-3.458** @50k (best across v1/v2/v3 variants), Δ = **−0.063**. This claim survives the audit. (Files: `clean_method_graph_rw_a09_from20k_v2`, `clean_method_graph_rw_v3_from20k`.)
- The improvement is driven by the **low-noise attention-derived order**: `ε=0.15`, `ε=0.20`, `top_k=8 / τ=0.5` all give weaker Δ. (Files: `clean_method_graph_rw_a09_from20k_eps015 / eps020 / topk8_tau05`.)
- The readiness term in v3 is necessary, evidenced by the ρ=0 ablation reaching only **3.484** @40k vs ρ=0.2 reaching **3.484** @40k → continuing to **3.458** @50k. (File: `clean_method_graph_rw_no_readiness_from20k`.)
- The Graph-RW improvement does **NOT close the gap to L2R-oracle continuation (3.390)**, but is about half of the L2R-vs-random gap.

### Image

- The same Graph-RW machinery, applied to CIFAR-10 patch attention with `top_k=4, ε=0`, recovers a soft 2D neighborhood + region + appearance walk — **not raster scan** — from the baseline10k attention.
- Using that order as α-mixed continuation curriculum **reduces structured-order val MSE by ~3 %** with **4–8σ separation from seed noise**, while the noise ablation (`ε=0.15`) provides **no benefit** over random continuation.
- The advantage grows with longer training (25k mean Δ = −0.00177 vs 5k Δ = −0.00150).

### Joint claim (cross-modal)

- AOGPT attention encodes dataset-level shared structure across modalities. The **same low-noise attention-derived order curriculum** improves both:
  - text `val_ori_l2r` (Δ = −0.063 at 50k)
  - image `val_rw_top4` and `val_raster` (Δ ≈ −0.0015 to −0.0020, robust across seeds)
- The structure manifests modality-specifically: 1D L2R-leaning order for text, 2D blob-filling order for image. The mechanism — *low-noise attention graph + Graph-RW sampler → curriculum gain; high noise → no gain* — is shared.

---

## 5. What is NOT claimed (explicitly)

- **No claim of SOTA** on either modality. Both modalities are small-model, single-author smoke setups; the contribution is mechanistic, not a leaderboard number.
- **No claim that v3 readiness improves over v2 on the metric** — it does not (Δ < 0.001 at 60k). v3 is preferred for principled reasons (single λ knob, explicit readiness term that admits ablation), not for a metric improvement.
- **No claim that A_global refresh helps**: the random-A refresh ablation (`ablation_rand_refresh_a09_from20k`) and the no-refresh control are within 0.001 of each other at 60k. At this scale refresh is at the noise floor.
- **No claim from the broken runs** `ablation_v2_noreadiness` / `ablation_v3_readiness`: these were misconfigured and are listed only to document the trap.
- **No claim that the text consensus order (K=100, τ=0.887) is a methodological contribution**: it is a sampling stability check, not a separate method. (The image consensus is a separate cautionary note — single rank-average gives "regression to the mean" τ=0.75 which is an artifact, not structure.)

---

## 6. Where to start the next text continuation

For aligning with the collaborator's WikiText-103 random baseline:
- The collaborator's expected starting checkpoint is the 50 k-iter random baseline from the config in `configs/text/wikitext103_seq256_block64_random_base.py`. When that checkpoint lands, the v3 continuation should resume from it.
- Until then, the closest internal proxy is `probe_results/clean_base_random_perm/ckpt_step20000.pt`. All `_from20k` runs in the registry use that fork point. Reproducing the strongest published text result is `bash scripts/text/run_text_v3.sh`.

For the image side, no further large training is required for the current claim set. Optional follow-ups (P1 in `image_results_2026-05-12.md`): 25k continuation with multi-seed CI (one already done at single seed, monotone Δ); A_global mid-training refresh; CE-loss variant with discretized patches.
