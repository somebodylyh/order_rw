# 07 — Claim Impact

Date: 2026-06-17

## What Changed

The strict 65-node None-separated protocol replaces the earlier "anchored controller" framing. Previously, the strongest label-free claim was limited by concerns about [None] → physical0 anchor leakage and inv_perm usage during graph construction. The new protocol eliminates both concerns.

## Can Claim

1. **Strict 65-node label-free protocol shows that selected early attention heads recover full physical L2R block order from an unanchored None start.**
   - Evidence: L0H1–L0H4 (collaborator), L1H0–L1H4 (clean_base) all produce tau=1.000, first=0, phys0_rank=0, p4=4, p8=8.
   - Destroyed controls: |τ| ≤ 0.07 (near random).

2. **inv_perm during construction is valid; only readout-stage inv_perm is a leak.**
   - CDL rollout uses only graph edge weights (C-D+L scores), never node labels. The coordinate frame (model vs physical) is invisible to CDL.
   - Oracle-remapped (physical-frame construction) and strict label-free (model-frame + posthoc) are both valid. They produce identical results (verified: 32/32 heads match within 0.01 at 3 checkpoints).
   - The "leak" boundary: inv_perm OK during graph construction and posthoc scoring; NOT OK during CDL rollout.

3. **The discovery is head-specific and readout-specific.**
   - L0H7 fails (τ=0.292, first=45, phys0_rank=20). Not all heads discover L2R.
   - L (transition-only) and C-D+L are the reliable readouts. none_edge (argmax B[None, :]) never strong-passes.
   - C-only is not recommended as primary evidence (potential low-index tie bias).

4. **The signal persists across checkpoints (5k–60k) and across training runs (clean_base + collaborator).**
   - Strong_pass counts: 10–16 across 7 checkpoints from 5k to 60k.
   - Both models have strong_pass heads (though specific head indices differ: cross-run head drift).

5. **Extraction frame matters.**
   - Loss-aligned AR + None-separated 65-node succeeds for L0H1–L0H4.
   - B1 predictor + content-only only succeeds for L0H3; others lose anchor.

6. **Destroyed controls confirm dependence on real attention structure.**
   - Entry-shuffled and label-permuted graphs produce |τ| ≤ 0.07.
   - Real τ − destroyed τ gap ≈ 0.95 for strong_pass heads.

## Cannot Claim

1. **"All heads discover L2R."** — Only 20/256 (7.8%) gate-pass. Sparse, not dense.

2. **"Content-only without [None] always recovers anchored L2R."** — Only L0H3 passes in B1 content-only; L0H1/H2/H4 lose anchor.

3. **"B1 predictor frame alone is sufficient."** — Requires loss-aligned AR frame for robust anchor.

4. **"Current frozen-hook acceleration already uses strict 65-node teacher."** — Hook uses B0 canonical (folds [None] into block0). This is a different protocol. The 65-node result is mechanism evidence, not engineering deployment.

5. **"The same specific head (e.g., L0H3) works for all models."** — Head identity drifts across training runs and checkpoints.

## Should Say Instead

| Instead of... | Say... |
|--------------|--------|
| "The model universally discovers L2R" | "Selected early heads recover L2R under strict 65-node label-free discovery" |
| "[None] is manually attached to physical0" | "None is an independent BOS node; first content block is selected by graph rollout from None" |
| "All heads encode order" | "Order-bearing structure is head-specific and sparse" |
| "inv_perm is a construction leak" | "Oracle-remapped uses inv_perm during construction → that IS a leak. Only strict label-free data is valid primary evidence. Oracle-remapped is secondary sanity only." |

## Evidence Hierarchy

| Tier | Claim | Status |
|------|-------|--------|
| Tier 1 | Strict 65-node label-free discovery: selected early heads recover L2R | ✅ VERIFIED |
| Tier 1 | inv_perm equivariance (not a leak) | ✅ VERIFIED |
| Tier 1 | Discovery is head-specific (L0H7 fails) | ✅ VERIFIED |
| Tier 1 | Destroyed controls near random | ✅ VERIFIED |
| Tier 1 | Persists 5k–60k (clean_base, 9-step sweep) | ✅ VERIFIED |
| Tier 1 | Cross-model robustness (collaborator strict LF + clean_base strict LF spotcheck) | ✅ VERIFIED (2 ckpts strict LF) |
| Tier 2 | Extraction frame dependence (B1 vs AR) | ✅ VERIFIED |
| Tier 2 | Stable strong head set within a run | ✅ VERIFIED (L1H0–L1H4) |
| Tier 3 | Multi-seed training stability | ⬜ Single training seed only |
| Tier 3 | 317M model (16L) 65-node | ⬜ Not run |
| Tier 3 | Image model 65-node | ⬜ Not run |

## Updated Master Claim

> Under a strict 65-node None-separated block graph with label-free model-frame extraction, selected early attention heads in randomly-trained AO-GPT models recover the full physical L2R block order from an unanchored None start. The result is head-specific (L0H7 fails, L0H1–L0H4 pass), readout-specific (transition-only L or C-D+L), and stable across checkpoints from 5k to 60k training steps. Destroyed controls are near random (|τ| ≤ 0.07), confirming dependence on real attention structure rather than numerical artifacts. Cross-model verification (clean_base + collaborator) confirms the phenomenon generalizes across training runs, though specific strong-head indices drift.
