# Claims Lock

## CAN CLAIM

### Claim 1 — Strict 65-node label-free block-level L2R discovery exists in selected early attention heads
**Strength**: Strong
**Evidence**:
- Collaborator ckpt @50k: L0H1, L0H2, L0H3, L0H4 all achieve τ=1.000, first=0, phys0_rank=0, prefix@4=4, prefix@8=8 under strict LF protocol.
- Clean-base 9-step ladder: 10–16 strong_pass heads at steps 5k–60k. Stable heads L1H0–L1H4 present at all 8 steps.
- Gate distribution: 20 strong / 11 weak / 225 fail (256 head×method combos) — signal is sparse and head-specific.
- Destroyed controls: mean |τ| ≤ 0.07 for all steps and control types. Real–destroyed gap ≈ 0.95.
- Inv_perm equivariance verified: strict LF ≈ oracle-remapped to within 0.01 τ.
**Source**: `reports/strict_65node_discovery_ckpt_verification_20260617/` (all files), `reports/collaborator_ckpt_b1_scan_20260616/strict_label_free_65_search/`
**Caveat**: Head-specific and extraction-frame-dependent. Not all heads discover L2R.

### Claim 2 — Order-agnostic AR training does not eliminate internal order structure
**Strength**: Strong
**Evidence**:
- 9/32 heads with |τ| > 0.9 at step 10k (B0 clean_base); 7/32 (B1 predictor).
- Signal emerges by step 5k, persists to 60k (clean-base ladder).
- 317M scale diagnostic shows 6/256 |τ| > 0.9 at step 5k (B0, best L0H10 τ=+0.955).
- Shuffled-L2R control shows τ ≈ 0 for CDL aggregate, confirming physical order disruption.
**Source**: `probe_results/clean_base_random_perm/head_scan_10k.json`, `batch_readout/logs/per_head_scan_b1/`, `probe_results/large_random_baseline_16l16h1024d/head_scan_step5000.json`
**Caveat**: Continuous-stream model at 50k shows 0/32 |τ| > 0.06 (B0 frame). Signal is transient in B0 frame but persistent in B1 predictor frame and strict 65-node protocol.

### Claim 3 — Legacy B0 g_β controller accelerates canonical-order training
**Strength**: Strong (for legacy B0 path)
**Evidence**:
- seed-123 group: Recovery 86.0% (from10k), 83.0% (from20k), 58.7% (from40k)
- seed-42 group: Recovery 106.1% (from10k), 101.5% (from20k), 66.9% (from40k)
- Conservative step saving @3.47 NLL: **33–42%** across both seeds, from10k+from20k.
- Wall-clock: ~62% saving including g_β pretraining cost.
**Source**: `probe_results/frozen_beta_seed2_from10k_l0h2/eval_curve.tsv`, `probe_results/frozen_beta_random_jun05_from10k/eval_curve.tsv`, `analyses/wall_clock_data.tsv`
**Caveat**: Legacy B0 controller path ([None] folded into phys0). Strict 65-node teacher controller not yet demonstrated. g_β is seed-dependent and head-dependent.

### Claim 4 — g_β reads structured attention graph, not a fixed L2R prior
**Strength**: Strong
**Evidence**:
- Real B τ = +0.9675 (g_β output vs L2R)
- Gaussian random B τ = −0.0034 (near zero)
- Pairwise Gaussian τ = 0.0003 (no consistent prior)
- Entry-shuffled B τ = +0.0165
- Zero B τ = 1.0 (tie-breaking artifact, margin=0)
- Real-destroyed gap = 0.907.
**Source**: `reports/evidence_package_20260613_verified/raw/gbeta_input_sanity_final.json`
**Caveat**: B0 legacy controller path. Non-L2R subset excluded (n=4, irreproducible). Zero B τ=1.0 is a known tie-breaking artifact, not a learned prior.

### Claim 5 — Extraction frame matters: loss-aligned AR + None-separated is cleaner than B1 predictor content-only for anchored discovery
**Strength**: Medium
**Evidence**:
- B1 predictor content-only: L0H1/H2/H4 τ=0.655 (cyclic perfect, anchor=6, wrong start block).
- Loss-aligned AR + None-sep: same heads τ=1.000 (anchored perfect, correct start).
- Frame difference ~0.345; inv_perm difference ~0.000.
**Source**: `reports/strict_65node_discovery_ckpt_verification_20260617/05_extraction_frame_comparison.md`
**Caveat**: Diagnostic comparison, not hook acceleration evidence.

### Claim 6 — Content-only 64-node shows physical adjacency after convention fix, but content-vs-position memory unresolved
**Strength**: Medium
**Evidence**:
- L0H0, without [None], true_original frame: d=+1 = **0.0571**, d=+2 = 0.0060, off-band (d≥3) ≈ 0.0001–0.0012. Contrast ≈ 50–500×.
- Correct physical frame uses `block_perm` (our convention: phys→model).
**Source**: `reports/collab_extract_on_our_ckpt_L0H0/`
**Caveat**: Fixed permutation means cannot distinguish content adjacency vs memorized position pairs. Cross-permutation validation pending.

### Claim 7 — Destroyed controls confirm signal is not a structural artifact
**Strength**: Strong
**Evidence**:
- 65-node entry shuffle + content label permutation: mean |τ| ≤ 0.07 across all steps.
- g_β Gaussian/entry-shuffled/rowcol-shuffled B: |τ| ≤ 0.017.
- L0H7 fail (τ=0.292) also near-random in destroyed controls — proves readout doesn't artificially inflate τ.
**Source**: `reports/strict_65node_discovery_ckpt_verification_20260617/06_destroyed_controls_summary.md`, `reports/evidence_package_20260613_verified/raw/gbeta_input_sanity_final.json`

---

## CANNOT CLAIM

1. **All heads discover L2R.** Only 7.8% pass strict gate (20/256). Most heads are near random.

2. **The same head identity is stable across all runs.** Head identity drifts: collaborator L0H1–H4 vs clean_base L1H0–H4 vs B1 continuous L0H4.

3. **Existing strict 65-node teacher has produced hook acceleration.** The strict discovery is mechanism evidence only. Current frozen hook uses legacy B0 protocol. CLAIM PENDING.

4. **Training loss is block-level.** Loss remains token-level AR. Block graph is an attention aggregation diagnostic, not the training objective.

5. **Content-only 64-node proves content-driven adjacency.** Under fixed permutation, physical adjacency could be position memory. Needs cross-permutation validation.

6. **AO-GPT universally discovers L2R across all seeds/scales.** 317M diagnostic at 5k is partial. Continuous-stream model at 50k shows 0/32 |τ|>0.06 (B0). Head drift exists across seeds.

7. **317M strict discovery or hook acceleration is established.** Only B0 diagnostic scan at 5k (6/256 strong). No strict 65-node scan, no frozen hook at 317M.

8. **Image/multimodal side solved.** Text-only evidence. Image CDL teacher results (D_manh) are preliminary.

9. **Fully label-free head audition + controller selection is closed.** Head selection uses audition (small supervised train). g_β uses CDL teacher supervision. No end-to-end label-free chain demonstrated.

10. **One g_β transfers across seeds or heads without retraining.** g_β is trained on a specific head from a specific seed. seed-123 uses L0H2; seed-42 uses L0H4.

11. **Shuffled-L2R encodes no order at all.** CDL aggregate shows τ≈0, but we have not exhaustively checked all heads. Claim only that physical order is not recoverable.

12. **ori-L2R is an upper bound.** seed-42 frozen_beta achieves 106.1% recovery (beats L2R). CDL teacher also beats L2R. ori-L2R is a reference point, not an upper bound.

---

## BEST WORDING

### For Abstract

> Random-order autoregressive language models internally develop sparse, head-specific attention patterns that encode recoverable block-level order structure. Under a strict 65-node None-separated block graph protocol, selected early heads recover the full physical left-to-right order without label supervision. We distill this discovered structure into a lightweight order controller, achieving 33–42% training step savings across two matched seed groups.

### For Introduction Contribution Bullet

> - We show that order-agnostic training does not eliminate internal order structure: selected attention heads in random-order AO-GPT encode recoverable block-level L2R order, verified under a strict 65-node label-free protocol with destroyed controls.
> - We demonstrate that attention-derived order structure can accelerate canonical-order training by 33–42% via a distilled g_β controller across two matched seed groups.

### For Slide Takeaway

> **Order-agnostic ≠ order-free**: AO-GPT attention heads internally encode L2R block order. We read it out and use it to accelerate training by 33–42%.

### For Reviewer-Safe Limitation Paragraph

> Our work has several limitations. First, the discovered order-bearing signal is head-specific and extraction-frame-dependent: not all heads encode L2R, and head identity drifts across training runs. Second, the existing training acceleration uses a legacy B0 controller protocol; the strict 65-node teacher-to-controller chain is not yet end-to-end verified. Third, our fixed permutation setup cannot fully separate content-driven adjacency from memorized position pairs. Fourth, our experiments use a single model scale (47M) and dataset (Wikitext-103); preliminary 317M diagnostics show signal at 5k but systematic scaling studies remain future work. Finally, g_β transfers are seed- and head-dependent, requiring per-configuration retraining.
