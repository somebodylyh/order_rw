# Task 1.2 — Round-2 multi-seed CI / sign-count + spot-check

**Date:** 2026-05-21. Inputs: `probe_results_image_large/grw_e3ctrlsmall_round2_multiseed_minimal/per_seed.tsv` (5 seeds: 42, 123, 314, 777, 2024; 4 arms). Aggregator: `scripts/aggregate_multiseed_minimal.py`. Schema: `cross = mean(val_random,raster,hilbert,Bcov_balanced,rw_top4_eps0,rw_eps015,rw_topk8)`; `structured = mean(val_raster,hilbert,Bcov_balanced,rw_top4_eps0,rw_topk8)`; `noisy = mean(val_random,val_rw_eps015)`.

## Spot-check (Amendment 2.I)
**Tier 1 (aggregation faithfulness):** re-ran the aggregator on disk eval_curve.tsv files → 80/80 numbers match committed `per_seed.tsv` exactly (tol 1e-4). PASS.

**Tier 2 (training code-path audit):** re-ran `cont_random` for seed=314 from the committed `block_lo_arm_order_network/train_imagelarge_round2.py` (5000 steps, identical args). Step-5000 eval row was **bit-identical** to the committed seed=314 cont_random row. PASS.

Conclusion: the externally-run Round-2 numbers came from the committed trainer; aggregation is faithful. **Spot-check PASS → proceed to CI.**

## Paired Δ / 95% CI / sign-count (n=5 seeds)

| comparison | metric | mean Δ | std | 95% CI | seeds Bcov wins |
|---|---|--:|--:|--:|---:|
| Bcov vs random | cross | −0.0029 | 0.0007 | ±0.0009 | **5/5** |
| Bcov vs random | structured | −0.0062 | 0.0009 | ±0.0011 | **5/5** |
| Bcov vs random | noisy | +0.0052 | 0.0008 | ±0.0010 | 0/5 |
| Bcov vs distance_only_coverage | cross | **−0.0021** | 0.0002 | ±0.0002 | **5/5** |
| Bcov vs distance_only_coverage | structured | **−0.0012** | 0.0002 | ±0.0002 | **5/5** |
| Bcov vs distance_only_coverage | noisy | −0.0044 | 0.0003 | ±0.0004 | 5/5 |
| Bcov vs shuffled_Bcov_balanced | cross | **−0.0049** | 0.0006 | ±0.0007 | **5/5** |
| Bcov vs shuffled_Bcov_balanced | structured | **−0.0075** | 0.0006 | ±0.0007 | **5/5** |
| Bcov vs shuffled_Bcov_balanced | noisy | +0.0015 | 0.0008 | ±0.0010 | 0/5 |

## Locked premise-K gate (Amendment 3.3)
PASS if at least one of `Bcov vs distance_only_coverage` OR `Bcov vs shuffled_Bcov` on (cross | structured) has **mean Δ < 0 AND |Δ| > 95% CI AND sign-count ≥ 4/5**.

**Result: 4/4 of those target rows pass all three sub-criteria.**
- Bcov vs distance_only_coverage / cross: Δ=−0.0021, |Δ|>±0.0002, 5/5 wins.
- Bcov vs distance_only_coverage / structured: Δ=−0.0012, |Δ|>±0.0002, 5/5 wins.
- Bcov vs shuffled_Bcov_balanced / cross: Δ=−0.0049, |Δ|>±0.0007, 5/5 wins.
- Bcov vs shuffled_Bcov_balanced / structured: Δ=−0.0075, |Δ|>±0.0007, 5/5 wins.

## Verdict
**Premise K SUPPORTED:** real-B edge-following (Bcov) yields task-level gains over both the geometry-only (distance_only_coverage) and the structure-destroying (shuffled_Bcov_balanced) controls, on cross and structured validation, across 5/5 seeds and beyond noise.

The noisy-axis trade-off is honest and visible (vs random: +0.0052, 0/5 wins) — Bcov is mildly specialized away from random-order evaluation, which is consistent with all training-on-structured-orders runs. This is a trade-off, not a failure.

## Honest caveats (write into the paper)
- |Δ| is small in absolute terms (≤ 0.008 nat) — must be reported with both std and CI; do not over-state.
- vs random comparison shows mixed pattern (noisy +0.005); paper should state task gain is against **geometry-only and destroyed-structure** controls, not against random per se.
- Conclusion is at THIS scale (l4h8e256, 5000-step continuation). Scaling behavior is Phase-3 / future work.

## Status (do NOT auto-launch)
- Premise K supported. Task 1.2 closed.
- **Phase 3 task-loss CEM oracle remains HELD pending explicit go-ahead** (Amendment 2.J launch conditions: also need the audit checklist to remain clean and any new ckpt-graph pairing protections for E3-large).
- Conditional PASS clause from Amendment 3.3 should now be active: "Multi-seed Round-2 results support this claim; we further corroborate it with task-loss CEM, which actively selects high real-B edge-following over geometric locality."
