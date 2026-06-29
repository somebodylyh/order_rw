# Uniform Label-Free Discovery v1 — Summary

**Date:** 2026-06-29 · **Branch:** `attn-order-alternating`
**Status:** cross-seed validated at step10k

## Protocol

| Stage | Uses physical label? | Method |
|---|---|---|
| Stage A — head scoring | No | split-half rollout stability + reveal-invariance + pairwise confidence + cycle consistency |
| Stage A — head selection | No | top-1 by label-free composite score |
| Stage C — readout training | No | single-head cluster-mean B → FlattenReadout, pairwise BCE |
| Final evaluation | **Yes** | posthoc inv_perm → τ_vs_physical |

## Results

| Seed | Selected head | Label-free score | τ_vs_physical | val_pairwise_acc |
|---|---|---|---|---|
| 2 | L0H3 | S=3.82 (split=1.0, reveal=1.0, conf=0.31, cycle=0.76) | **0.965** | 0.891 |
| 42 | L1H3 | S=3.85 (split=1.0, reveal=1.0, conf=0.22, cycle=0.81) | **1.000** | 0.998 |
| 123 | L1H7 | S=3.78 (split=1.0, reveal=1.0, conf=0.25, cycle=0.76) | **1.000** | 0.998 |

**Uniform label-free discovery v1 succeeds across all three seeds at step10k.**
Using only label-free graph stability scores, the protocol selects one head per seed
and trains a small FlattenReadout, achieving τ_vs_physical ≥ 0.97 on all seeds.

## Baselines

| Baseline | Seed=2 | Seed=42 | Seed=123 | Label-free? | Meaning |
|---|---|---|---|---|---|
| **Ours** | L0H3 τ=0.97 | L1H3 τ=1.00 | L1H7 τ=1.00 | Yes | proposed selector |
| Raw-rollout oracle | L0H0 τ=0.98 | L0H2 τ=0.99 | L0H1 τ=−0.02 | No | strongest head by canonical C-D+L rollout τ |
| Random head | L3H2 τ=−0.21 | L0H6 τ=0.22 | L0H5 τ=−1.00 | No | one random head |
| All-head mean (32) | τ=−0.01 | τ=−0.08 | τ=0.23 | No (no labels, but bad aggregate) | naive mean over all 32 heads |

**Note on the raw-rollout oracle:** This is NOT a readout oracle / upper bound.
It selects the head with strongest τ_canon from direct C-D+L rollout, which does not
guarantee readout success. Seed=123 shows this clearly: L0H1 has τ_canon=1.0 but
readout τ=−0.02. The true readout upper bound (train all 32 heads, select best by
physical τ posthoc) is pending.

## Key findings

1. **Cross-seed stable.** Same uniform protocol, three seeds, three different selected
   heads, all τ ≥ 0.97. No hard-coded layer or head index.

2. **Raw-rollout oracle is not a readout oracle.** The head with strongest canonical C-D+L
   rollout τ is not necessarily learnable by FlattenReadout (seed123 L0H1: τ_canon=1.0,
   readout τ=−0.02). The label-free selector instead discovers *readout-compatible*
   physical-order heads.

3. **Two carrier types.** The canonical-rollout physical carrier (studied in P1/P2,
   typically L0) and the learnable readout carrier (identified by this selector, L1
   in seed=42/123) can differ. Both carry physical-order information but via different
   graph structures.

4. **Aggregation is protocol, not cheating.** Single-sample / single-reveal scores are
   too noisy. Batch-mean aggregation is part of the label-free protocol.

5. **Small-M selection better.** M≈20 preserves split-half discrimination; M≥40
   saturates (too many heads score split=1.0). Selection M and training M are
   independent knobs.

6. **None node is mechanism.** None/BOS mass in the 65-node graph is part of the
   signal path and should NOT be penalized.

7. **All-head mean fails.** Positive and negative carriers cancel when naively averaged
   (τ≈0 for all seeds). Selection or clustering is required.

## Limitations

- Validated at step10k only; step sweep (5k/2k/20k) pending
- Top-1 selection; multi-head cluster approach not yet validated
- True readout oracle upper bound pending
- Destroyed-gap baseline pending
- No sample-size (M) sweep for training

## Next steps

1. True readout oracle upper bound (train all 32 heads, select best posthoc)
2. Step sweep: 5k, 2k, 20k
3. Cluster selection (top-4 → cluster → best cluster)
4. M-train sweep: 100, 200, 500
