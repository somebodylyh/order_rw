# Next Steps And Decisions

## Must Do Before Paper Draft

1. Incorporate B1/predictor diagnostic into the mechanism section.
   - Use B1 as signal-robustness evidence.
   - Keep current acceleration tables labeled B0 legacy hook path.

2. Report val_unstructured trade-off.
   - Frame as canonical-order specialization.
   - Source: `reports/evidence_package_20260613_verified/02_VERIFIED_TABLES.md` Table V9.

3. Freeze final claim lock.
   - Use `08_claims_and_boundaries.md` as meeting version.
   - Do not merge B0 and B1 numbers in one unqualified table.

4. Visually check or regenerate figures if they use deprecated numbers.
   - Watch for old `28/32` strong-head count.
   - Watch for wrong seed42 random baseline `3.354`; verified random baseline is `3.466`.
   - Watch for B0/B1 mixed labels.

5. Decide CDL teacher placement after matched run completes.
   - If no verified matched result: keep CDL teacher as supplementary.
   - If verified: add as teacher upper/supplementary comparison, not as `g_beta` deployment result.

## High-Value Optional Experiments

1. B1 hook smoke rerun.
   - Purpose: upgrade from "B1 signal robustness" to "B1 controller path may work."
   - Recommended if paper main method is rewritten around B1.

2. B1 `g_beta` sanity.
   - Purpose: prove `g_beta` reads B1 graphs, not only B0 graphs.
   - Lower cost than full B1 hook.

3. Label-free audition closure.
   - Purpose: reduce oracle/head-selection critique.
   - Could be smaller than full multi-seed training if framed as diagnostic.

4. Third seed.
   - Purpose: statistical completeness for training acceleration.
   - Strong paper value if compute is available.

5. Direct matched CDL comparison.
   - Purpose: clarify teacher vs learned controller.
   - Only useful after seed-matched run is verified.

6. Threshold sensitivity for step saving.
   - Purpose: show 33-42% is not cherry-picked at 3.47.
   - Can often be done from existing eval curves.

## Larger Scope Experiments

1. 317M hook.
   - High value for scale, high compute risk.

2. B1 317M scan.
   - Cheaper than 317M hook; upgrades scale diagnostic to B1.

3. No-PE control.
   - Addresses PE leakage critique but may open a new branch.

4. Image teacher redesign.
   - Keep separate unless text paper scope expands.

5. Multimodal extension.
   - Future-work scale, not needed for current text-side boss update.

## Decisions For Boss

| Decision | Options | Recommendation | Why |
|---|---|---|---|
| Paper scope | Text-side now; wait for scale; wait for image/multimodal | Text-side now, image/multimodal as future work | Current text evidence has a coherent mechanism + training chain; expansion risks delaying core paper |
| B1 migration depth | Diagnostic only; B1 `g_beta` sanity; B1 hook smoke; full B1 hook rerun | B1 `g_beta` sanity + small B1 hook smoke if resources allow | Best trade-off between protocol consistency and compute |
| Compute priority | Third seed; 317M hook; B1 hook; label-free audition | B1 smoke or third seed before 317M hook | Reduces central claim risk more cheaply |
| CDL teacher | Main comparator; supplementary; wait for matched result | Supplementary unless matched seed123 result is verified | Current verified CDL evidence has seed mismatch |
| Figure work | Use current figures; regenerate all; only verify core four | Verify core four and regenerate only if labels/numbers are stale | Avoid unnecessary figure churn |
| Claim wording | Strong B1 unified claim; staged B0/B1 claim | Staged claim for Wednesday | Accurate under current evidence and compatible with ongoing B1 replacement |

