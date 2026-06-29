# Key Results

## A. Strict 65-node collaborator @50k

Source: `reports/collaborator_ckpt_b1_scan_20260616/strict_label_free_65_search/all_head_methods_strict_label_free_65.tsv`

| Head | Method | tau | first | phys0_rank | prefix@4 | prefix@8 | destroyed |tau| | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---|
| L0H1 | L-transition-only (`L`) | 1.000 | 0 | 0 | 4 | 8 | 0.0498 | strong_pass |
| L0H2 | L-transition-only (`L`) | 1.000 | 0 | 0 | 4 | 8 | 0.0531 | strong_pass |
| L0H3 | L-transition-only (`L`) | 1.000 | 0 | 0 | 4 | 8 | 0.0501 | strong_pass |
| L0H4 | L-transition-only (`L`) | 1.000 | 0 | 0 | 4 | 8 | 0.0554 | strong_pass |
| L0H7 | C-D+L | 0.292 | 45 | 20 | 0 | 0 | 0.0825 | fail |

Interpretation:

- L0H1-L0H4 recover full L2R from independent None/BOS under strict 65-node LF.
- L0H7 fail shows this is head-specific, not a universal head property or trivial tie-break.

## B. Gate distribution

Source: `reports/strict_65node_discovery_ckpt_verification_20260617/02_existing_result_summary.md`

| Protocol | strong_pass | weak_pass | fail | total |
|---|---:|---:|---:|---:|
| Strict label-free | 20 | 11 | 225 | 256 |
| Oracle-remapped | 20 | 11 | 225 | 256 |

Interpretation:

- Exact match supports permutation equivariance.
- This does not mean `inv_perm` is used inside rollout; it means equivalent graph labelings recover equivalent orders.

## C. Clean-base 9-step ladder

Sources:

- `reports/strict_65node_discovery_ckpt_verification_20260617/strict_65node_ckpt_sweep.tsv`
- `reports/strict_65node_discovery_ckpt_verification_20260617/04_ckpt_sweep_results.md`

| Step | strong_total | Best head/method | best tau | destroyed mean |tau| |
|---:|---:|---|---:|---:|
| 0 | 0 | L0H2 C-D+L | 0.191 | 0.063 |
| 1000 | 0 | L2H4 none_edge | 0.142 | 0.069 |
| 5000 | 10 | L1H2 L | 1.000 | 0.050 |
| 10000 | 10 | L1H2 L | 1.000 | 0.050 |
| 20000 | 16 | L1H4 L | 1.000 | 0.052 |
| 30000 | 15 | L0H1 L | 1.000 | 0.053 |
| 40000 | 15 | L1H2 L | 1.000 | 0.055 |
| 50000 | 14 | L0H0 L | 1.000 | 0.057 |
| 60000 | 16 | L0H0 L | 1.000 | 0.054 |

Summary:

- 0/1k: 0 strong heads.
- 5k-60k: 10-16 strong rows per checkpoint.
- Stable heads include L1H0-L1H4; L0H0 emerges later.
- Phenomenon is stable; exact head identity drifts.

## D. Extraction frame comparison

Sources:

- `reports/strict_65node_discovery_ckpt_verification_20260617/05_extraction_frame_comparison.md`
- `reports/collaborator_ckpt_b1_scan_20260616/no_inv_label_free_readout_posthoc_M20_seed0.json`

| Head | B1 predictor + content-only | loss-aligned AR + None-separated |
|---|---:|---:|
| L0H1 | 0.655 | 1.000 |
| L0H2 | 0.655 | 1.000 |
| L0H3 | 1.000 | 1.000 |
| L0H4 | 0.655 | 1.000 |

Interpretation:

B1 predictor content-only can recover the order axis, often as a perfect cyclic order, but loses anchor for L0H1/L0H2/L0H4. Loss-aligned AR + None-separated keeps independent None/BOS and recovers the correct start.

## E. Destroyed controls

Sources:

- `reports/strict_65node_discovery_ckpt_verification_20260617/06_destroyed_controls_summary.md`
- `reports/strict_65node_discovery_ckpt_verification_20260617/destroyed_controls_by_step.tsv`

Summary:

- Collaborator strong heads have destroyed mean |tau| around 0.05-0.067 while real tau is 1.000.
- Real - destroyed gap is approximately 0.95 for strong heads.
- Clean-base ladder destroyed means are about 0.05-0.069 across steps.
- TSV spotcheck aggregates are near random but not all below a hard 0.07 threshold; the safe claim is "near random", not a fragile exact cutoff.

