# Risks and Missing Experiments

## Green / Resolved

| Item | Resolution |
|------|-----------|
| seed42 baseline mismatch | Fixed: seed42 random baseline = 3.466 (NOT 3.354). Source: `random_baseline_continuous_jun05/eval_curve.tsv` |
| g_β sanity JSON saved | Verified values in `raw/gbeta_input_sanity_final.json` |
| None fold / anchor leak | Addressed by strict 65-node None-separated protocol |
| inv_perm / block_perm convention | Clarified: nanogpt vs block_lo_arm_order_network conventions documented |
| Destroyed controls near random | Verified across all 9 steps, both control types. |τ| ≤ 0.07 |
| L0H7 fail | Proves discovery is not trivial — serves as negative control |
| CDL teacher seed mismatch | Resolved: separate seed-2 group documented |
| Permutation equivariance (strict LF ≈ oracle-remapped) | Verified to within 0.01 τ |

---

## Yellow / Manageable

| Item | Risk | Mitigation |
|------|------|-----------|
| **Head-specific and head drift** | Strong-head indices differ across runs | Frame as "sparse order-bearing heads exist" not "head X always works." Document drift honestly. |
| **Extraction-frame-dependent** | B1 vs 65-node gap of 0.345 | 65-node is the preferred protocol. Frame comparison becomes a finding, not a flaw. |
| **Content-vs-position memory** | Fixed permutation cannot separate | Cross-permutation validation pending (P0). Currently caveated in all claims. |
| **Strict-teacher controller not yet hooked** | Mechanism-to-acceleration chain not closed | Frame paper as mechanism + preliminary controller. P0 closure experiment designed. |
| **Label-free head audition** | Current audition uses small supervised train | P1 experiment. Claim boundary: "not yet fully label-free." |
| **val_unstructured trade-off** | Order specialization degrades arbitrary-order performance | Document as trade-off. Not a blocker for acceleration claim. |
| **Limited seeds (2 matched groups)** | Statistical completeness | P1: add third seed. Current 2 seeds are sufficient for mechanism claim. |
| **M=8/20 in strict discovery** | Small sample size for gate statistics | P1: increase M to 200+. Current sample sufficient for strong-pass detection (τ=1.000 is saturated). |
| **50k per-head signal decay in B0 continuous** | 0/32 |τ|>0.9 at 50k | B1 predictor and 65-node show persistent signal. B0 frame is known to lose signal at convergence. |

---

## Red / Paper Blocker

| Item | Blocker For | Resolution |
|------|-----------|-----------|
| **Strict 65-node teacher → g_β → hook chain** | Full method paper | **P0 priority.** Without this, paper must be framed as mechanism discovery + legacy controller (B0), not end-to-end strict-teacher controller. |
| **Cross-permutation content-vs-position** | Claim about content-driven adjacency | **P0 priority.** Without this, content-only 64-node results can only be reported with major caveat about position memory. |
| **317M scale verification** | Scale generalization claim | Currently only B0 diagnostic at 5k. Need at minimum strict 65-node scan at 317M to claim scale robustness. |

### Paper Scope Decision

If the paper is framed as **mechanism discovery + preliminary controller evidence**:
- Red items are NOT blockers
- Current evidence is sufficient
- Must honestly state that strict-teacher controller is future work

If the paper is framed as **complete method (discovery → acceleration)**:
- P0 items MUST be completed first
- Without them, the paper has a gaping hole in the story

**Recommendation**: Frame as mechanism discovery paper with preliminary controller results for first submission. Close P0 items before submitting to top venues.
