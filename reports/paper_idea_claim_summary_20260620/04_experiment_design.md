# Experiment Design

## E1 — Baseline / Legacy Acceleration Verification

**Objective**: Verify that legacy B0 frozen g_β accelerates canonical-order training.

**Protocol**:
- Train AO-GPT with random block permutations (continuous streaming)
- At 10k, extract B from selected head (L0H2 for seed-123, L0H4 for seed-42)
- Train CDL teacher → distill g_β
- Resume from 10k/20k/40k with frozen g_β hook (B0 protocol)
- Train to 60k, evaluate val_ori_l2r_block every 500 steps

**Metrics**:
- Recovery@50k = (L_random − L_method) / (L_random − L_L2R) × 100%
- Step saving@3.47: T_random − T_method / T_random × 100%
- val_ori_l2r_block: NLL under L2R block order

**Key runs** (all under `block_lo_arm_order_network/probe_results/`):
- `random_baseline_continuous_jun08_seed2/eval_curve.tsv` — seed-123 random baseline (3.445 @50k)
- `random_baseline_continuous_jun05/eval_curve.tsv` — seed-42 random baseline (3.466 @50k)
- `l2r_continuous_seed123/eval_curve.tsv` — seed-123 L2R ref (3.305 @50k)
- `l2r_continuous_seed42/eval_curve.tsv` — seed-42 L2R ref (3.341 @50k)
- `frozen_beta_seed2_from10k_l0h2/eval_curve.tsv` — seed-123 frozen_beta from10k
- `frozen_beta_random_jun05_from10k/eval_curve.tsv` — seed-42 frozen_beta from10k

---

## E2 — g_β Sanity

**Objective**: Prove g_β is not a fixed L2R prior, but reads real B structure.

**Protocol**:
- Train g_β on real selected-head B (B0, L0H2)
- Test g_β on: real B, Gaussian random B, zero B, entry-shuffled B, row+col shuffled B
- Generate 100 independent Gaussian samples for pairwise diversity test

**Key results** (source: `reports/evidence_package_20260613_verified/raw/gbeta_input_sanity_final.json`):
| Input | τ vs L2R |
|-------|---------|
| Real B | +0.9675 |
| Gaussian B | −0.0034 |
| Zero B | +1.0 (tie-breaking artifact, margin=0) |
| Entry-shuffled | +0.0165 |
| Row+col shuffled | +0.0127 |
| Pairwise Gaussian | 0.0003 |

**Verdict**: 4/4 checks PASS. g_β reads B structure, not fixed L2R prior.

---

## E3 — Attention Extraction Audit

**Objective**: Confirm signal is not a B0 artifact. Clarify B1/predictor shift and None handling.

**Sub-experiments**:
- **E3a**: B1 predictor signal survival (clean-base ladder, continuous seed=124)
- **E3b**: Loss-aligned AR frame vs B1 predictor comparison
- **E3c**: Content-only 64-node vs None-separated 65-node
- **E3d**: Permutation convention audit (block_perm direction)

**Key findings**:
- B1 signal survives but is weaker than 65-node for anchored discovery
- B1 content-only L0H1/H2/H4: τ=0.655 (cyclic perfect, wrong start) vs 65-node: τ=1.000
- [None] as independent BOS provides the anchor for correct start
- `block_perm` convention: phys→model in our repo
- Source: `reports/b1_attention_extraction_summary_20260615/`, `reports/strict_65node_discovery_ckpt_verification_20260617/`

---

## E4 — Strict 65-Node Label-Free Discovery

**Objective**: Prove selected heads can recover physical L2R from independent None start.

**Protocol**:
- Build B65 in model-frame coordinates
- Run CDL rollout (L-only or C-D+L) from None
- Posthoc: inv_perm translate σ_model → σ_phys, score vs L2R
- Gate: τ=1.000 AND first=0 AND phys0_rank=0 AND prefix@4=4 AND prefix@8=8

**Sub-experiments**:
- **E4a**: Collaborator ckpt @50k full sweep (256 head×method, M=20)
- **E4b**: Clean-base 9-step ladder (0/1k/5k/10k/20k/30k/40k/50k/60k, M=8)
- **E4c**: Destroyed controls (entry shuffle, content label perm)
- **E4d**: Oracle-remapped equivalence check
- **E4e**: Extraction frame comparison (B1 predictor vs loss-aligned AR)

**Key results**: See 05_current_results_by_claim.md, Claims 1, 5, 7.

---

## E5 — Stability / Robustness

**E5a — Head drift across runs**:
- Clean_base strong heads: L1H0–L1H4 (most stable)
- Collaborator strong heads: L0H1–L0H4
- The phenomenon is robust; specific head indices drift

**E5b — Extraction-frame dependence**:
- B1 predictor content-only vs loss-aligned AR + None-sep
- Gap ~0.345 for anchored discovery

**E5c — Training step stability**:
- 0/1k: zero strong_pass (signal requires training)
- 5k–60k: 10–16 strong heads per step (persistent)
- Best head shifts: L1H2→L1H4→L0H1→L0H0 across steps

**E5d — Scale diagnostic**:
- 317M model (16L/16H/1024d): 6/256 |τ|>0.9 at 5k (B0)
- No strict 65-node scan at 317M

---

## E6 — Pending: Strict-Teacher Controller Closure

**Objective**: Close the mechanism-to-acceleration chain.

**Plan** (see 09_recommended_next_experiments.md):
1. Select strong heads via non-oracle audition or fixed discovered set
2. Generate strict 65-node teacher orders
3. Train g_β on strict 65-node teacher
4. Sanity check: real vs destroyed B
5. Run short hook from 10k or 20k
6. Compare to legacy B0 hook and random baseline

**Status**: PENDING. This is the #1 priority.
