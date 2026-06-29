# Model-Frame Head Drift Diagnostic

Output directory: `reports/gbeta_feedback_final_summary/model_frame_head_drift/`

This diagnostic was run on CPU with `M=4`, `batch_size=2`, `fwd_batch=2`, method `C-D+L`, tracked head `L0H2`. Treat it as a targeted failure-analysis pass, not a final high-power robustness estimate.

## Tracked L0H2

| label | step | tau_model_vs_semantic_path | rank_sem | tau_model_vs_identity | rank_id |
|---|---:|---:|---:|---:|---:|
| random20 | 20000 | 0.0645 | 11 | -0.1548 | 30 |
| random40 | 40000 | 0.0645 | 10 | -0.1548 | 31 |
| random60 | 60000 | -0.1052 | 28 | 0.0030 | 15 |
| gbeta_good60 | 60000 | -0.0397 | 23 | 0.1141 | 6 |
| gbeta_bad60 | 60000 | 0.0407 | 13 | 0.1746 | 3 |
| cdl_good60 | 60000 | 0.0119 | 12 | -0.0804 | 28 |
| cdl_bad60 | 60000 | -0.1052 | 28 | 0.0030 | 17 |

Tracked L0H2 is not a stable explanation by itself. The good g_beta run does not preserve high L0H2 semantic tau, while the bad g_beta run has higher L0H2 identity alignment. This argues against a simple story where "higher L0H2 tau means better final loss."

## Best Semantic Head Per Checkpoint

| label | best head | tau_model_vs_semantic_path | tau_model_vs_identity |
|---|---|---:|---:|
| random20 | L2H5 | 0.2183 | 0.1617 |
| random40 | L2H5 | 0.1478 | 0.0694 |
| random60 | L2H2 | 0.1468 | 0.0149 |
| gbeta_good60 | L1H6 | 0.1399 | -0.2163 |
| gbeta_bad60 | L2H3 | 0.2302 | -0.0962 |
| cdl_good60 | L2H3 | 0.2183 | -0.1438 |
| cdl_bad60 | L3H3 | 0.1379 | 0.0714 |

The bad g_beta run still has a strong best semantic readout (`L2H3`, tau `0.2302`). Therefore the failure case is not explained by a global disappearance of model-frame order signal. A more likely failure mode is that the frozen controller order is not useful for the training objective/eval protocol despite readable attention structure remaining elsewhere.

## Interpretation

- Good from20k frozen g_beta improves final `val_ori_l2r_block`, but this diagnostic does not show a simple strengthening of tracked L0H2 semantic tau.
- Bad from40k seed124 frozen g_beta fails final `val_ori_l2r_block`, yet retains strong best-head semantic signal. The issue is likely controller usefulness, selected-head mismatch, or protocol/seed sensitivity rather than absence of order signal.
- CDL good and bad runs show similar behavior: readable heads exist, but final loss depends on whether the injected order functions as a useful curriculum/controller.

## Controller Order Diagnostic

Output: `reports/gbeta_feedback_final_summary/controller_order_diag/controller_order.tsv`

This directly replays each frozen g_beta provider on its final 60k checkpoint and scores the emitted controller order.

| run | head | tau_controller_model_vs_semantic_path | tau_controller_model_vs_identity | tau_controller_phys_vs_l2r |
|---|---|---:|---:|---:|
| frozen_beta_b1_seed2_from20000 | L0H2 | 0.0099 | 0.0764 | -0.1786 |
| frozen_beta_b1_seed2_from40000 | L0H2 | 0.2312 | -0.0595 | 0.0506 |
| frozen_beta_from20k_fixseed | L0H2 | -0.0317 | -0.0208 | 0.0933 |
| frozen_beta_from20k_seed124 | L0H2 | 0.0823 | -0.2381 | 0.0010 |
| frozen_beta_from40k_seed124 | L0H0 | -0.0159 | -0.2609 | 0.1667 |

The bad from40k seed124 run uses `L0H0` and its emitted controller order is strongly anti-identity in model frame (`tau=-0.2609`). This should be read together with the small-train audition result: from40k seed124 selected `L0H0,rev`, while `L0H2,fwd` was nearly tied. The completed frozen run appears to use the `L0H0` g_beta path without an explicit reverse orientation, so the failure is most plausibly an audition-to-deployment/orientation mismatch plus long-horizon sensitivity. It is not evidence that small-train audition never selected the arm.

The good from20k runs also do not show a naive "high semantic tau controller" story. Their final-loss gains likely come from the controller acting as a useful curriculum/order bias under the training protocol, not from simply reproducing physical L2R or identity order.
