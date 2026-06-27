# Pillar ③ Causal Handoff Path-Patching — Results

**Date:** 2026-06-27 · **Branch:** `attn-order-alternating`
**Checkpoints:** `runs/handoff_overnight/seed{2,42,123}/ckpt_step10000.pt` (step-10000, any-order/random-reveal, 4L/8H/d384)
**Config:** `n_batches=4 × bs_mean=16`, τ method `C-D+L`, readouts τ-only (CPU)
**Outputs:** `runs/handoff_pathpatch/seed{2,42,123}/{stage1_table.csv,stage2_table.csv,summary.json,stage1_heatmap.png}`

## Verdict

**The order-signal *handoff* hypothesis (L0 source → redundant L1 carrier → downstream
order) is NOT causally supported.** The converged carrier is a **seed-/layer-self-contained,
redundantly distributed** structure, not one inherited from an earlier layer via an L0→L1
edge. This is consistent with Pillars ①②: carriers are seed-locked, emerge through an early
(step 1k–2k) reorganization, and show no clean temporal layer-to-layer handoff.

A secondary, *positive* finding holds: where a strong L1 carrier exists (seed2, seed123),
the redundancy signature `single ≤ LOO ≤ full-set` is monotone — the carrier is **distributed
across ~4 heads**, so single-head ablation under-reads it.

## Per-seed results

| seed | L1 strong carrier | 2a: Δτ L1-dst under full L0-ablation | 1b: L1 multiplicity (before→after L0-abl) | 2b path_fraction global (vs null-path) | redundancy ladder |
|------|-------------------|--------------------------------------|-------------------------------------------|----------------------------------------|-------------------|
| 2    | {0,3,5,7}         | **−0.020** (robust)                  | 3→3 (no collapse)                          | 0.184 **< null 0.316**                 | single .019 ≤ LOO .029 ≤ full .040 ✓ |
| 42   | — (none; weak {2,6}) | −0.024                            | 0→0 (no strong)                            | 0.755 > null 0.377 *(weak tier, tiny denominators — degenerate)* | .019 ≤ .019 ≤ .032 ✓ |
| 123  | {0,5,6,7}         | **−0.018** (robust)                  | 3→3 (no collapse)                          | 0.321 **< null 0.426**                 | single .012 ≤ LOO .020 ≤ full .020 ✓ |

(`n_strong_before=3` for the 4-head sets: at bs_mean=16 the borderline H0 averages just
under 0.95; the other three sit at 1.0.)

## Decision criteria (spec §"Decision criteria"), per seed

1. **L1 carrier-set ablation beyond null (Stage 1a):** ✗ — downstream (L2) Δτ is tiny
   (~0.02–0.04) and at the same floor as same-layer null-head ablation. The redundancy
   *ordering* holds (monotone) but the *magnitude* is not above the null.
2. **L0 weak-source ablation collapses L1 (Stage 1b):** ✗ — L1 multiplicity 3→3 and
   mean-τ Δ ≈ −0.02; seed123's null ablation even collapses one more head than the real
   source. L1 does **not** depend on L0.
3. **L0→L1 path-restricted QK above null-path (Stage 2):** ✗ for the strong-carrier seeds
   (path < null-path in both seed2 and seed123). seed42's high value is on weak-tier heads
   with near-zero denominators and is not interpreted as support.
4. **seed42 contrast:** seed42 has no strong L1 set (as predicted), but its path_fraction is
   *higher*, not lower — a degeneracy of dividing tiny path/full deltas on weak heads, not a
   handoff. The strong-carrier seeds are the informative ones, and they are negative.

→ **3 of 4 criteria negative; criterion 4 degenerate.** No causal handoff.

## Interpretation (spec interpretation table)

The seed2/seed123 pattern is **"single-head ablation null, set ablation slightly stronger but
near floor"** → the L1 carrier is **redundant/distributed**, but its causal footprint on
downstream order under any-order training is small, and it is **not fed by L0**. The order
structure is an emergent, layer-local redundant code (matching the line's framing: attention
self-aligns to intrinsic structure as *emergence*, not behavioral necessity), rather than a
multi-layer handoff circuit.

## Caveats

- Effects are small (Δτ ~0.02–0.04) — expected, since any-order training makes the objective
  order-insensitive and the order signal lives in attention, not loss. The **null-path /
  null-head controls** are therefore the operative comparison, and the strong-carrier seeds
  are below them.
- τ has real sampling variance; reported as mean over 4 batches of 16. Tighter CIs (more
  batches) would sharpen criterion-1's floor comparison but would not change the direction
  of 2a/1b (robust L1) or the path<null ordering.
- Pillars ④ (freeze/graft) and ⑤ (representation content) remain future specs; this null
  result reframes them — there is no handoff edge to graft, so ④ would instead test whether
  the layer-local redundant carrier re-emerges independently per seed.
