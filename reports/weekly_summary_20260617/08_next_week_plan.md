# Next Week Plan

## P0 - Close mechanism-to-controller loop

- Train / distill `g_beta` using strict 65-node teacher.
- Run a short hook smoke test with the strict teacher.
- Compare strict-teacher hook against the legacy B0 hook.
- Verify destroyed-B sanity under the strict teacher.
- Decide whether the paper's main method is "strict 65-node teacher" or "mechanism diagnostic plus legacy controller acceleration."

## P1 - Make head audition clean

- Define a label-free head audition score.
- Show that the audition score selects strong-pass heads.
- Avoid oracle tau head selection in the controller path.
- Keep selected-head reporting sparse and protocol-specific.

## P1 - Additional robustness

- Add more checkpoints if any required intervals are missing.
- Add more seeds for strict 65-node discovery.
- Run 317M strict 65-node diagnostic if scale claim is needed.
- Run B1 strict 65-node across runs if B1 becomes the primary diagnostic convention.

## P2 - Paper packaging

- Update figures around strict 65-node graph, None-separated rollout, and destroyed controls.
- Update method section to separate token-level AR loss from block-level attention aggregation.
- Split mechanism results from controller acceleration results.
- Write limitations explicitly: token-level AR loss, block-level graph aggregation, head specificity, extraction-frame dependence, legacy acceleration boundary.

