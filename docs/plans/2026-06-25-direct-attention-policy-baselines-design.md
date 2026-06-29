# Direct Attention Policy Baselines Design

**Date:** 2026-06-25

## Goal

Add three non-learned reveal-order baselines that consume the same attention
input as the new model-frame `g_beta`:

- L0 all-head attention
- model-frame strict65 graphs
- multiple random probes averaged before scoring
- mean fusion over heads
- no physical-coordinate fields in the policy path

The baselines are `initial_cdl_one_shot`, `source_mass`, and `readiness`.
Historical single-head B1 sequential CDL remains a separate expensive teacher
reference.

## Approaches Considered

1. Reuse the old `CdlOrderProvider`.
   Rejected because it uses a selected head, B1/physical-frame extraction, and
   sequential rollout. Its input distribution does not match the new `g_beta`.

2. Duplicate the model-frame extraction in a new provider.
   Rejected because two nominally identical extraction paths could drift.

3. Share the model-frame strict65 probe extraction between frozen `g_beta` and
   direct policies.
   Selected. A small helper in `frozen_gbeta_hook.py` will produce the
   probe-averaged strict65 tensor. The learned and direct providers will consume
   that exact tensor.

## Architecture

`batch_readout/direct_order_provider.py` will contain pure score functions and a
training-loop-compatible provider. The score API accepts
`B[batch, head, 65, 65]` under the convention `B[source, target]` and returns
per-head scores plus mean-head scores.

The training CLI will expose:

```text
--run-kind direct_policy
--direct-policy initial_cdl_one_shot|source_mass|readiness
--direct-policy-lambda-dep 1.0
--direct-policy-refresh 10
--batch-mean-probes 4
```

The provider returns a model-frame block order. Existing training-loop code
performs the final model-to-physical translation only for applying the order to
the permuted training data. No physical metadata enters score computation.

## Score Definitions

For content submatrix `M = B[:, :, 1:, 1:]`, with diagonal excluded:

```text
initial_cdl_one_shot:
  2 * B[:, :, 0, 1:] - column_sum(M) / 63

source_mass:
  row_sum(M)

readiness:
  row_sum(M) - lambda_dep * column_sum(M)
```

Scores are averaged over heads. Larger score means earlier reveal.

The first formula is only the strict65 initial-state reduction of old CDL:

```text
q_0(i) = C_0(i) - D_0(i) + L_0(i)
       = 2 B[0, i] - mean_{u in U excluding i} B[u, i]
```

It is not equivalent to the full sequential CDL rollout.

## Testing

Pure score tests cover shape, descending order direction, robust diagonal
exclusion, row/column orientation, `lambda_dep`, and mean-head fusion. Provider
tests cover shared extraction use, refresh caching, model-frame output, and
absence of physical-coordinate parameters. CLI/config tests cover policy
selection and audit metadata.

## Audit Output

Startup logs and `config.json` will record:

| Field | Value |
|---|---|
| matrix convention | `B[source,target]` |
| frame | `model-frame strict65` |
| layer/head | `L0 all-head` |
| probe aggregation | mean over `batch_mean_probes` |
| head fusion | mean over heads |
| diagonal | excluded; strict65 diagonal expected zero |
| order direction | larger score earlier |

