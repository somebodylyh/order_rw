# Label-Free L0 Multi-Head g_beta Pretrain Design

## Scope

This design covers only offline pretraining and evaluation of a multi-head
`g_beta`. It does not connect the controller to the AO-GPT training loop, run
reinforcement learning, or enable alpha co-training.

The experiment asks one question:

> Can a shared, map-conditioned controller read a CDL-derived order from raw L0
> all-head attention maps without using head identity, physical order labels, or
> a hard-coded inverse permutation?

## Fixed experiment inputs

- Source checkpoint:
  `block_lo_arm_order_network/probe_results/random_baseline_continuous_jun08_seed2/ckpt_step20000.pt`
- Layer: L0 only.
- Heads: all eight L0 heads.
- Dataset size: 2,000 batch-mean graphs.
- Samples per batch-mean graph: 16.
- Split: 80% train, 10% validation, 10% test.
- Coordinate frame: model frame throughout extraction, teacher construction,
  training, and model selection.
- Graph shape: strict-65, with node 0 as None/BOS and nodes 1 through 64 as
  model-frame content blocks.

Physical coordinates, `inverse_block_perm`, physical L2R/raster order, and
physical distance are excluded from extraction, teacher construction, model
input, loss, and checkpoint selection. They may be used only for post-hoc
interpretation.

The existing 251-snapshot raw head-map trajectory is not training data. It
contains four repeated validation samples, uses 64-by-64 B1 maps, and was
collected along a feedback-trained trajectory. It is reserved for temporal
generalization analysis.

## Data extraction

For every sampled sequence:

1. Run the checkpoint under a deterministic random model-frame probe order.
2. Extract all eight L0 attention heads.
3. Build a loss-aligned strict-65 graph per head without applying `inv_perm`.
4. Average 16 per-sample graphs to obtain one batch-mean graph.

The primary stored tensor is:

```text
B_raw: [2000, 8, 65, 65]
```

The extraction stage saves raw maps and provenance only. It does not generate
physical labels.

Before the full extraction, run a smoke extraction with 16 batch-mean graphs
and validate graph shape, finite values, zero diagonal, None-node placement,
and deterministic reproduction.

## CDL rollout and dynamic label-free teacher

For each graph and each of the eight heads, run greedy None-fixed CDL over the
64 content nodes. Save the content order and its inverse rank vector.

Each head receives three per-sample quality values.

### Standardized rollout margin

At every CDL step:

1. Compute utilities over the remaining candidates.
2. Z-score utilities within that step.
3. Take the largest standardized utility minus the second largest.

The head's margin score is the mean over rollout steps with at least two
candidates.

### Structure-preserving destroyed gap

The naive v0 uses one destroyed replica per graph and head.

The destroy operation:

- independently shuffles every content-to-content row;
- separately shuffles the None-to-content row;
- preserves the content-to-None column under the strict-65 definition;
- keeps the diagonal at zero.

It preserves row value distributions, sharpness, sparsity, and None structure
while destroying block identity and order topology.

The destroyed gap is:

```text
real standardized rollout margin - destroyed standardized rollout margin
```

The final experiment may increase to three destroyed replicas without changing
the method.

### Multi-head agreement

Convert each head order to an inverse rank vector:

```python
rank[order] = arange(64)
```

Agreement for one head is its mean Kendall tau against the other seven rank
vectors. Kendall tau must not be computed directly on the order lists.

### Dynamic teacher weights

For each sample, independently z-score margin, destroyed gap, and agreement
across its eight heads. Sum the three normalized values:

```text
quality = z_head(margin) + z_head(destroyed_gap) + z_head(agreement)
```

Convert quality to dynamic teacher weights:

```text
teacher_weight =
    0.95 * softmax(quality / 1.0)
    + 0.05 / 8
```

No hard Top-K selection is used in the primary method. All heads contribute,
with sample-dependent soft weights. A Top-K teacher may be evaluated later as
an ablation.

For each pair of content blocks `(i, j)`, construct the soft target:

```text
teacher_pairwise[i, j] =
    sum_h teacher_weight[h] * indicator(rank_h[i] < rank_h[j])
```

## Input normalization

Strict-65 block graphs are not assumed to have unit row sums. Derive four
channels:

```text
B_raw  = original strict-65 map
B_prob = row-normalized B_raw
B_logz = whole-map z-score of log(B_raw + 1e-8), per sample and head
B_rowz = row-wise z-score of B_raw
```

The block scorer receives all four channels. The gate does not receive
`B_raw`, reducing absolute-scale shortcuts.

## Student architecture

The primary model has no head identity embedding. Both encoders share
parameters across all eight heads.

### Shared block scorer

For each content block and each head, concatenate the row and column from all
four channels:

```text
4 channels * (65 row values + 65 column values) = 520 features
```

Use a small shared MLP:

```text
520 -> 256 -> 64 -> 1
```

This produces:

```text
scores_per_head: [batch, 8, 64]
```

No handcrafted entropy, locality, physical-position, or Top-K scalar features
are included in v0.

### Shared dynamic gate

For each head, derive a compact fixed pooling summary from `B_prob`, `B_logz`,
and `B_rowz`. Use simple mean, standard deviation, maximum, and minimum pooling
per channel. Feed the pooled vector through a shared MLP to produce one gate
logit per head.

Apply softmax across the eight heads:

```text
student_alpha: [batch, 8]
```

The final block score is the weighted sum of per-head scores. The gate has no
head ID and no direct supervision from teacher weights.

## Training objective

The naive v0 computes all 2,016 unordered content-block pairs rather than
sampling pairs.

The loss contains:

1. Final soft pairwise BCE between final score differences and the soft teacher.
2. Per-head auxiliary soft pairwise BCE with weight 0.05.
3. A small gate entropy-floor penalty with initial weight 0.001.

During training, randomly mask one head per sample before gate softmax. There
is no complex dropout schedule in v0.

Student alpha is not trained to reproduce teacher weights. Alignment between
them is an analysis result, not a training target.

## Model selection and evaluation

Select the checkpoint using validation teacher-pairwise accuracy. Do not use
physical metrics for selection.

Required offline metrics:

- pairwise accuracy against the consensus teacher;
- Kendall tau against a hard order derived from consensus mean rank;
- prefix overlap at 4 and 8 against the consensus teacher;
- teacher-weight and student-alpha entropy;
- per-sample alpha variation;
- correlation, KL, and top-head agreement between student alpha and teacher
  weights, for analysis only.

Required sanity checks:

- structure-preserving destroyed input;
- remove the top-alpha head and renormalize;
- compare against a uniform-head mean baseline;
- compare against a single-head student baseline.

Post-hoc physical interpretation may report physical L2R/raster alignment,
model order versus inverse permutation, identity alignment, and image spatial
quality. These metrics never affect training or checkpoint selection.

The old head-track trajectory is used only after pretraining to measure
temporal generalization across steps.

## Success criteria

The minimal successful result requires:

1. High validation/test imitation of the dynamic CDL consensus teacher.
2. A material performance drop under structure-preserving destroyed input.
3. Student alpha varies across samples rather than remaining constant.
4. Removing the top-alpha head degrades performance without collapsing it to
   chance.

The first version does not require medium/failing heads alone to recover the
teacher.

## Explicit non-goals

- No AO-GPT frozen hook in this implementation stage.
- No alpha-only co-training.
- No RL, REINFORCE, finite-difference search, SoftSort, or Gumbel-Sinkhorn.
- No cross-layer gate.
- No head identity embedding in the primary model.
- No physical coordinate input.
- No direct inverse-permutation teacher.
- No complicated hand-engineered gate feature set.
