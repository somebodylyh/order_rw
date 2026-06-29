# Old/New Random Checkpoint Matched Audit

Run date: 2026-06-26

## Protocol

- Old checkpoint group: `block_lo_arm_order_network/probe_results/random_baseline_continuous_jun08_seed2`
- New checkpoint group: `block_lo_arm_order_network/probe_results/overnight_20260625_random_baseline`
- Steps: 5000, 10000, 20000
- Samples: `M=500`, `bs_mean=4`, total 2000 fixed train-stream windows per checkpoint
- Probe orders: fixed by `seed + sample_index`, with `seed=0`
- Extraction: L0 all heads, strict65, node 0 is None/BOS, content nodes 1..64
- B convention: `B[source,target]` from `build_none_separated_B(A[target,source])`
- Physical scoring: `sigma_phys = inv_perm_model_to_phys[sigma_model]`
- Tau references: physical L2R `[0..63]` and reversed physical L2R `[63..0]`
- Quality score in this run uses fast `n_destroy_replicas=0`, so it combines rollout margin and head agreement; destroyed-gap is intentionally skipped for runtime. Core tau, B/B.T, model/physical, and consensus comparisons are unaffected.

## Consensus Physical-Frame B

| ckpt_group | step | consensus_tau_l2r | consensus_tau_rev_l2r | r_quality_abs_tau | near_diag_mass |
|---|---:|---:|---:|---:|---:|
| old | 5000 | -0.1235 | +0.1235 | -0.1939 | 0.5166 |
| old | 10000 | +0.2432 | -0.2432 | +0.0891 | 0.5650 |
| old | 20000 | +0.4757 | -0.4757 | +0.2966 | 0.6108 |
| new | 5000 | -0.0196 | +0.0196 | -0.0080 | 0.3937 |
| new | 10000 | -0.1124 | +0.1124 | +0.0526 | 0.4113 |
| new | 20000 | -0.0158 | +0.0158 | +0.1175 | 0.4140 |

## Top Absolute Per-Head Physical-Frame B

| ckpt_group | step | top_head | tau_l2r | abs_tau_gt_0.5 | quality |
|---|---:|---:|---:|---:|---:|
| old | 5000 | H0 | -0.3135 | 0.538 | -1.613 |
| old | 10000 | H1 | +0.5259 | 0.546 | +0.899 |
| old | 20000 | H7 | +0.6200 | 0.652 | +1.127 |
| new | 5000 | H6 | +0.0771 | 0.022 | -0.445 |
| new | 10000 | H5 | -0.3243 | 0.316 | +0.252 |
| new | 20000 | H5 | -0.2669 | 0.290 | +0.596 |

## B vs B.T Physical Consensus

| ckpt_group | step | B_tau_l2r | BT_tau_l2r |
|---|---:|---:|---:|
| old | 5000 | -0.1235 | -0.0279 |
| old | 10000 | +0.2432 | +0.1005 |
| old | 20000 | +0.4757 | -0.0126 |
| new | 5000 | -0.0196 | +0.0426 |
| new | 10000 | -0.1124 | +0.0019 |
| new | 20000 | -0.0158 | -0.0149 |

## Interpretation

The same diagnostic protocol gives substantially different old/new signals at matched steps. Old 10k and 20k show positive physical-L2R consensus, strengthening by 20k. New 5k/10k/20k are near-zero to weakly negative in consensus, with only a moderate negative single-head signal at new 10k/20k H5.

This supports the "checkpoint trajectory differs" explanation for the current mismatch. It does not support a simple B vs B.T direction flip: B.T does not recover the old positive 20k signal in the new checkpoints, and old 20k B.T collapses toward zero rather than becoming the sign-opposite counterpart.

Model-frame and physical-frame consensus rows are numerically identical after the explicit `inv_perm` remap for scoring, which is expected under this fixed layout. The physical B maps nevertheless show different near-diagonal mass: old increases from 0.5166 to 0.6108, while new stays around 0.39-0.41.

Conclusion: do not infer "g_beta from10k will not work" from the new per-head L2R scan alone. The safer statement is that early and mid-stage attention/CDL signals are checkpoint-trajectory dependent here, and per-head L2R alignment should be reported separately from full CDL teacher-match and g_beta teacher-match.
