# Direct Policy Input Audit

**Date:** 2026-06-25

## Protocol Table

| Method | L0 all-head | Model-frame strict65 | Probe mean | Head fusion | Learned | Sequential | Directly aligned with new g_beta |
|---|---:|---:|---:|---|---:|---:|---:|
| new/frozen g_beta | yes | yes | 4 probes | learned nonlinear gate/scorer | yes | no | reference |
| initial_cdl_one_shot | yes | yes | 4 probes | mean heads | no | no | yes |
| source_mass | yes | yes | 4 probes | mean heads | no | no | yes |
| readiness | yes | yes | 4 probes | mean heads | no | no | yes |
| historical sequential CDL | no, selected head | no, B1/physical-frame | separate extraction | selected head | no | yes | no |

All aligned direct methods use:

- matrix convention: `B[source,target]`
- content nodes: strict65 nodes `1..64`; node `0` is None
- diagonal handling: self edge explicitly excluded
- order direction: larger score is revealed earlier
- no `clean_perm`, `inv_perm`, `block_perm`, head ID, teacher weight, or tau in scoring

## Score Audit

| Policy | Score |
|---|---|
| `initial_cdl_one_shot` | `2 * B[0,i] - mean_{u != i} B[u,i]` |
| `source_mass` | `sum_{u != i} B[i,u]` |
| `readiness` | `sum_{u != i} B[i,u] - lambda_dep * sum_{u != i} B[u,i]` |

`initial_cdl_one_shot` is the old CDL score reduced only at the strict65
initial state. It is not equivalent to the final order from sequential CDL.
`source_mass` and `readiness` are alternative graph heuristics, not CDL scores.

## Refresh Audit

The model-frame frozen `g_beta` wrapper previously ignored
`--frozen-beta-refresh`; it recomputed probes on every provider call even when
the log reported refresh 10. The wrapper now implements the same refresh cache
as the direct provider, and the training CLI passes the configured value.

Consequences:

- Future runs with refresh 10 are protocol-aligned.
- A frozen `g_beta` process launched before this code change remains an
  every-call-refresh run and is not wall-clock aligned with direct refresh-10
  runs.
- Either rerun frozen `g_beta` after the fix, or run direct policies with
  `--direct-policy-refresh 1` for comparison to that historical process.

## Training Alignment

The supplied launcher uses the same:

- step-20000 resume checkpoint
- continuous train/validation streams
- seed 123
- optimizer and LR schedule
- batch size and gradient accumulation
- alpha schedule
- evaluation interval and validation windows
- four probe forwards and refresh interval 10

The only policy-specific variables are the direct score formula and
`lambda_dep` for readiness.

