# ListMLE CDL Distillation Design

**Date:** 2026-06-25

## Goal

Distill the full sequential CDL permutation directly from strict-65 attention
graphs into the existing `g_beta` scorer.

The student receives:

```text
B_raw: [batch, heads, 65, 65]
```

and emits:

```text
student_logits: [batch, 64]
```

Larger logits mean earlier reveal. Inference remains:

```text
predicted_order = argsort(student_logits, descending=True)
```

Training must not use `argsort`, physical coordinates, or a constructed
`64 x 64` pairwise target when `loss_type=listmle`.

## Teacher Target

The teacher is the full sequential CDL order:

```text
teacher_order: [batch, 64]
```

Each row is a permutation of block indices `0..63`, ordered from earliest to
latest reveal. For the existing multi-head teacher pipeline, the hard target is
the weighted consensus order derived from the label-free per-head CDL ranks and
teacher weights:

```text
mean_rank[i] = sum_h teacher_weight[h] * teacher_rank[h, i]
teacher_order = argsort(mean_rank)
```

The dataset should store this hard consensus order explicitly so training does
not reconstruct it from the soft pairwise matrix.

## Primary Objective: ListMLE

For teacher permutation `pi` and student logits `s`, use the
Plackett-Luce/ListMLE negative log likelihood:

```text
L = -sum_t log(
    exp(s[pi[t]]) /
    sum_{j=t}^{N-1} exp(s[pi[j]])
)
```

The vectorized implementation is:

```python
ordered_logits = student_logits.gather(1, teacher_order)
suffix_lse = torch.logcumsumexp(
    ordered_logits.flip(dims=[1]), dim=1
).flip(dims=[1])
per_position_loss = suffix_lse - ordered_logits
loss = per_position_loss.sum(dim=1).mean()
```

The first version uses uniform position weights. The final position contributes
zero by construction.

## Baselines

Training supports three mutually exclusive primary objectives:

### `listmle`

Direct full-permutation distillation. This is the new primary method.

### `pairwise_bce`

The current soft pairwise BCE objective. It remains the strong baseline and
uses the existing dynamic soft pairwise teacher.

### `rank_kl`

Convert the hard teacher rank into a categorical priority distribution:

```text
p_teacher[i] = softmax(-rank[i] / tau)
q_student[i] = softmax(student_logits[i])
L = KL(p_teacher || q_student)
```

This is an auxiliary comparison for priority-distribution distillation, not the
preferred objective for full-order fidelity. Initial temperatures are
`tau in {2, 4, 8, 16}`.

## Existing Regularizers

The head-gate entropy floor remains available for every loss type.

The existing per-head pairwise auxiliary loss is enabled only for
`pairwise_bce` in the first comparison. Applying it to ListMLE would make the
main experiment a mixed objective and prevent a clean comparison.

No prefix weighting and no `ListMLE + pairwise` objective are included in the
first implementation.

## Configuration

Add the following training configuration:

```text
loss_type = listmle | pairwise_bce | rank_kl
rank_kl_temperature = 4.0
```

Default the new comparison launcher to `listmle`. Preserve
`pairwise_bce` behavior and checkpoint compatibility.

Every checkpoint and metrics record must include the loss type and relevant
temperature.

## Evaluation

Evaluate on held-out validation and test graphs against the hard full CDL
consensus order.

Required metrics:

- Kendall tau versus full CDL
- pairwise accuracy versus full CDL
- Prefix@8 set overlap
- Prefix@16 set overlap
- primary validation loss
- training and validation loss curves
- training wall time
- peak CUDA memory when CUDA is used

Retain the existing destroyed-graph and gate diagnostics where applicable.

The current pairwise-BCE reference is approximately:

```text
Kendall tau:       0.984
pairwise accuracy: 0.997
Prefix@8:          0.973
```

ListMLE is successful if it approaches these held-out metrics while using a
direct permutation target and a simpler teacher representation.

## Complexity

For `N=64`:

- ListMLE target storage: `O(N)`
- ListMLE loss computation: `O(N)`
- Rank-KL target storage and computation: `O(N)`
- Pairwise target storage and loss computation: `O(N^2)` with 2,016 unique
  pairs

The student network forward pass is unchanged and is expected to dominate
training. The complexity comparison must therefore report measured wall time
and memory rather than relying only on asymptotic cost.

## Validation and Error Handling

`listmle_loss` must reject:

- non-2D student logits
- shape mismatch between logits and teacher order
- non-integer teacher order
- out-of-range indices
- rows that are not permutations
- unsupported reduction modes
- malformed position weights

Dataset loading must continue rejecting physical-coordinate fields.

## Tests

Unit tests must cover:

- correctly ordered logits have lower loss than reversed logits
- batched `[B, 64]` inputs
- finite backward gradients
- permutation validation
- invalid shapes, indices, and duplicate entries
- optional position weights, although weighting is not used in first runs
- Rank-KL target normalization and temperature validation
- loss configuration parsing
- unchanged `pairwise_bce` behavior

Integration tests must verify that a short CPU training run works for each loss
type and writes the selected objective into its config and checkpoint.

## Experiment Sequence

1. Train uniform ListMLE.
2. Rerun or reuse the protocol-matched Pairwise BCE baseline.
3. Train Rank-KL at temperatures `2, 4, 8, 16`.
4. Compare held-out fidelity, wall time, memory, and downstream AO-GPT NLL.
5. Only if ListMLE loses meaningful full-order accuracy, test
   `ListMLE + 0.05 * Pairwise BCE`.
6. Only if prefix fidelity is specifically deficient, test exponential
   position weights as a separate ablation.

