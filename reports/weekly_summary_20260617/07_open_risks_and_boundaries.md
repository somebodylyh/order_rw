# Open Risks and Boundaries

## Green / Resolved

- seed42 baseline mismatch resolved: random baseline @50k is `3.466`, not `3.354`.
- `g_beta` sanity raw JSON saved and traceable.
- `inv_perm` boundary clarified by stage: graph construction/posthoc scoring allowed; rollout decisions not allowed.
- None-as-phys0 leak avoided in strict protocol by separating None as node0.
- Destroyed controls are near random relative to real strong-head tau=1.000.
- strict LF and oracle-remapped full sweeps match, supporting permutation equivariance rather than leakage.

## Yellow / Manageable

- Head-specific result: selected heads work; many heads fail.
- Head drift: clean-base strong heads are not the same as collaborator strong heads.
- Extraction-frame dependence: B1 content-only can recover cyclic axis but lose anchor.
- strict discovery is not yet connected to frozen hook acceleration.
- `val_unstructured_order` trade-off remains a limitation: the method specializes toward canonical order and degrades under arbitrary random evaluation orders.
- Controller head selection / audition still needs a clean label-free protocol.
- 317M strict 65-node protocol is not verified.
- B1 hook acceleration is not verified.

## Red / Next Blocker

No fatal blocker for a Wednesday mechanism update.

Main paper blocker: connect strict 65-node teacher to `g_beta` / hook acceleration if the paper wants one unified method-result chain rather than separate mechanism and legacy-controller evidence streams.

