# gBeta Feedback Final Summary

## Key Results

- Random baseline final `val_ori_l2r_block`: 3.499628.
- Best final `val_ori_l2r_block`: `frozen g_beta B1 seed2 from20k` = 3.332232 (delta vs random -0.167396).
- Best frozen g_beta arm: `frozen g_beta B1 seed2 from20k` = 3.332232 (delta vs random -0.167396).
- Best CDL teacher arm: `CDL teacher from20k seed124` = 3.338534 (delta vs random -0.161094).
- Failure/instability case: `frozen g_beta from40k seed124` = 4.208990. This was not an arbitrary arm: the small-train audition selected `L0H0,rev` at from40k, with `L0H2,fwd` nearly tied. The completed frozen run appears to deploy `L0H0` without an explicit reverse orientation, so treat this as an audition-to-deployment/orientation mismatch plus long-horizon instability risk, not simply "small-train selected a bad head."
- Reverse control present: `CDL seed123 from20k L0H2 reverse` = 4.541146.
- Model-frame head-drift diagnostic: the bad g_beta run still has a strong best semantic readout (`L2H3`, tau `0.2302`), so the failure is not simply "order signal disappeared." It points toward controller usefulness / selected-head mismatch / protocol sensitivity.
- Direct controller-order diagnostic: the bad from40k seed124 run uses `L0H0` and emits an anti-identity model-frame controller order (`tau_model_vs_identity=-0.2609`). This matches the suspicion that the production run did not preserve the `rev` orientation selected by small-train audition.

## Claim Boundary

- This package summarizes existing completed runs; it does not add new training.
- Head-gated g_beta remains smoke-only and is not included in the main evidence claim.
- Treat seed/protocol inheritance carefully; compare matched groups before claiming robustness.
- The from40k seed124 frozen g_beta failure should be diagnosed before making broad stability claims. The immediate control is to preserve the audition-selected orientation (`L0H0,rev`, now expressible via `--frozen-beta-rev`) or run the near-tied `L0H2,fwd` from40k candidate under the same W&B-logged protocol.

## Files

- `summary_table.tsv`: final-row metrics for each run.
- `summary.json`: machine-readable copy of the table.
- `val_ori_l2r_block.png`, `val_train_objective.png`, `val_model_order.png`, `val_unstructured_order.png`: curve overlays.
- `head_drift_summary.md`: targeted model-frame diagnostic for good vs bad feedback runs.
- `model_frame_head_drift/`: raw diagnostic TSV/JSON.
- `controller_order_diag/controller_order.tsv`: direct replay of each frozen g_beta controller order on final checkpoints.
