# Current Claims Lock

## CAN CLAIM

1. Strict 65-node label-free block-level L2R discovery exists in selected early heads.
2. The discovery is head-specific and extraction-frame-dependent.
3. The result is not an `inv_perm` construction leak; strict LF and oracle-remapped match by permutation equivariance.
4. Destroyed controls collapse to near random, supporting real graph structure rather than tie-breaking or constant-prior explanations.
5. The discovery appears stably from 5k to 60k in the clean-base sweep, although head identity drifts.
6. Legacy frozen `g_beta` controller acceleration remains verified across two matched seed groups.
7. B1/predictor diagnostics show that the order-bearing attention signal survives collaborator-aligned extraction.

## CANNOT CLAIM

1. All heads discover L2R.
2. Head identity is fixed across runs or checkpoints.
3. B1 predictor content-only alone is sufficient for anchored discovery.
4. Current frozen hook acceleration already uses strict 65-node teacher.
5. The same `g_beta` transfers across seeds or heads.
6. Fully label-free head selection is end-to-end verified for controller training.
7. 317M strict 65-node discovery / hook acceleration is established unless explicitly verified.
8. Image or multimodal extension is solved.
9. The training loss itself is block-level; it is still token-level AR, while the discovery graph is block-level aggregation.
10. The old seed42 random baseline was 3.354; the verified random baseline is 3.466.
11. Strong heads are dense, such as 28/32; verified evidence supports sparse selected heads.

## SHOULD SAY INSTEAD

- Say: "The token-level AR model's attention can be aggregated into a strict 65-node block graph."
- Do not say: "The model is trained with true block-level loss."
- Say: "Selected early heads discover L2R under strict 65-node protocol."
- Do not say: "The whole model universally discovers L2R."
- Say: "Legacy acceleration motivates strict-teacher controller rerun."
- Do not say: "Strict 65-node teacher already caused the acceleration."
- Say: "B1/predictor diagnostics strengthen the mechanism story."
- Do not say: "B1/predictor diagnostics prove B1 hook acceleration."

## Source Lock

- Training acceleration: `reports/evidence_package_20260613_verified/02_VERIFIED_TABLES.md`.
- B1 diagnostic: `reports/b1_attention_extraction_summary_20260615/03_b1_signal_summary.md`.
- Strict 65-node mechanism: `reports/strict_65node_discovery_ckpt_verification_20260617/02_existing_result_summary.md`.
- Strict collaborator rows: `reports/collaborator_ckpt_b1_scan_20260616/strict_label_free_65_search/all_head_methods_strict_label_free_65.tsv`.

