# ListMLE CDL Distillation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add direct full-CDL-permutation distillation with ListMLE, preserve Pairwise BCE as a protocol-matched baseline, and add Rank-KL as a priority-distribution ablation.

**Architecture:** Keep the existing `L0DynamicGBeta` model unchanged: it maps L0 all-head strict-65 graphs to 64 reveal-priority logits. Extend the dataset with an explicit hard consensus CDL order, dispatch the training objective by `loss_type`, and evaluate every objective against the same held-out hard CDL order. Preserve all label-free boundaries and existing pairwise checkpoints.

**Tech Stack:** Python 3.8, PyTorch, NumPy, SciPy, pytest.

---

### Task 1: Add Hard Consensus Orders to the Dataset Schema

**Files:**
- Modify: `analyses/build_l0_dynamic_gbeta_dataset.py:70-95`
- Modify: `analyses/build_l0_dynamic_gbeta_dataset.py:224-264`
- Modify: `block_lo_arm_order_network/tests/test_build_l0_dynamic_gbeta_dataset.py:80-110`

**Step 1: Write the failing dataset tests**

Extend the synthetic teacher fixture with:

```python
teacher["consensus_order"] = np.tile(
    np.arange(64, dtype=np.int64), (M, 1),
)
```

Add assertions:

```python
assert z["teacher_consensus_order"].shape == (20, 64)
for row in z["teacher_consensus_order"]:
    np.testing.assert_array_equal(np.sort(row), np.arange(64))
```

Add a unit test for the consensus helper:

```python
def test_weighted_consensus_order_sorts_weighted_mean_rank():
    ranks = np.array([
        [0, 1, 2, 3],
        [3, 2, 1, 0],
    ])
    weights = np.array([0.75, 0.25])
    order = weighted_consensus_order(ranks, weights)
    np.testing.assert_array_equal(order, np.array([0, 1, 2, 3]))
```

**Step 2: Run tests to verify they fail**

Run:

```bash
python3 -m pytest \
  block_lo_arm_order_network/tests/test_build_l0_dynamic_gbeta_dataset.py \
  -q
```

Expected: FAIL because `weighted_consensus_order` and
`teacher_consensus_order` do not exist.

**Step 3: Implement the consensus helper and saved field**

Add:

```python
def weighted_consensus_order(
    ranks: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    ranks = np.asarray(ranks, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    if ranks.ndim != 2:
        raise ValueError(f"ranks must be [H, N], got {ranks.shape}")
    if weights.shape != (ranks.shape[0],):
        raise ValueError(
            f"weights must be [{ranks.shape[0]}], got {weights.shape}"
        )
    mean_rank = (ranks.astype(np.float64) * weights[:, None]).sum(axis=0)
    return np.argsort(mean_rank, kind="stable").astype(np.int64)
```

Allocate `teacher_consensus_order = np.zeros((M, 64), dtype=np.int64)`,
populate it after `build_dynamic_teacher`, and include
`"consensus_order"` in the teacher dictionary.

Extend `save_dataset`'s allowed teacher fields with `"consensus_order"`.

**Step 4: Run tests to verify they pass**

Run the command from Step 2.

Expected: PASS.

**Step 5: Commit**

```bash
git add analyses/build_l0_dynamic_gbeta_dataset.py \
  block_lo_arm_order_network/tests/test_build_l0_dynamic_gbeta_dataset.py
git commit -m "feat: save hard CDL consensus orders"
```

### Task 2: Add a Migration Path for Existing Datasets

**Files:**
- Create: `analyses/add_consensus_order_to_gbeta_dataset.py`
- Create: `block_lo_arm_order_network/tests/test_add_consensus_order_to_gbeta_dataset.py`

**Step 1: Write failing migration tests**

Create a small `.npz` with `teacher_ranks`, `teacher_weights`, and all existing
fields. Test that migration:

```python
add_consensus_order(src, dst)
with np.load(dst) as z:
    assert z["teacher_consensus_order"].shape == (M, 64)
    assert set(z.files) == set(original_fields) | {"teacher_consensus_order"}
```

Also test refusal when ranks or weights are absent and in-place migration via a
temporary output plus atomic replace.

**Step 2: Run tests to verify they fail**

```bash
python3 -m pytest \
  block_lo_arm_order_network/tests/test_add_consensus_order_to_gbeta_dataset.py \
  -q
```

Expected: FAIL because the migration module does not exist.

**Step 3: Implement the migration command**

Use structured NumPy loading and saving:

```python
def add_consensus_order(src_path: str, dst_path: str) -> str:
    with np.load(src_path, allow_pickle=True) as z:
        payload = {key: z[key].copy() for key in z.files}
    ranks = payload["teacher_ranks"]
    weights = payload["teacher_weights"]
    payload["teacher_consensus_order"] = np.stack([
        weighted_consensus_order(ranks[m], weights[m])
        for m in range(ranks.shape[0])
    ])
    np.savez_compressed(dst_path, **payload)
    return dst_path
```

Add CLI flags `--src`, `--dst`, and `--in-place`. Do not recompute CDL.

**Step 4: Run migration tests**

Run the command from Step 2.

Expected: PASS.

**Step 5: Commit**

```bash
git add analyses/add_consensus_order_to_gbeta_dataset.py \
  block_lo_arm_order_network/tests/test_add_consensus_order_to_gbeta_dataset.py
git commit -m "feat: migrate gbeta datasets to hard consensus orders"
```

### Task 3: Implement ListMLE and Rank-KL Losses

**Files:**
- Create: `block_lo_arm_order_network/batch_readout/order_distillation_losses.py`
- Create: `block_lo_arm_order_network/tests/test_order_distillation_losses.py`

**Step 1: Write failing ListMLE tests**

Cover:

```python
def test_listmle_prefers_teacher_order():
    order = torch.tensor([[0, 1, 2, 3]])
    perfect = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    reversed_logits = perfect.flip(1)
    assert listmle_loss(perfect, order) < listmle_loss(reversed_logits, order)

def test_listmle_backward_is_finite():
    logits = torch.randn(3, 64, requires_grad=True)
    order = torch.stack([torch.randperm(64) for _ in range(3)])
    loss = listmle_loss(logits, order)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
```

Add validation tests for:

- logits not `[B, N]`
- shape mismatch
- floating-point teacher order
- negative or out-of-range indices
- duplicate indices
- malformed position weights
- invalid reduction

**Step 2: Write failing Rank-KL tests**

Cover target normalization, perfect-logit preference, finite backward, and
temperature rejection:

```python
target = rank_kl_target(order, temperature=4.0)
torch.testing.assert_close(target.sum(dim=1), torch.ones(B))
```

**Step 3: Run tests to verify they fail**

```bash
python3 -m pytest \
  block_lo_arm_order_network/tests/test_order_distillation_losses.py \
  -q
```

Expected: FAIL because the module does not exist.

**Step 4: Implement minimal validated losses**

Implement:

```python
def validate_teacher_order(student_logits, teacher_order): ...

def listmle_loss(
    student_logits,
    teacher_order,
    position_weights=None,
    reduction="mean",
): ...

def order_to_rank_tensor(teacher_order): ...

def rank_kl_target(teacher_order, temperature): ...

def rank_kl_loss(student_logits, teacher_order, temperature=4.0): ...
```

Use `F.kl_div(F.log_softmax(student_logits, dim=1), target,
reduction="batchmean")`.

Do not add combined loss or nonuniform weights to training configuration.

**Step 5: Run tests and compile**

```bash
python3 -m pytest \
  block_lo_arm_order_network/tests/test_order_distillation_losses.py \
  -q
python3 -m py_compile \
  block_lo_arm_order_network/batch_readout/order_distillation_losses.py
```

Expected: PASS and no compile output.

**Step 6: Commit**

```bash
git add block_lo_arm_order_network/batch_readout/order_distillation_losses.py \
  block_lo_arm_order_network/tests/test_order_distillation_losses.py
git commit -m "feat: add listwise CDL distillation losses"
```

### Task 4: Extend Dataset Loading Without Breaking Pairwise Checkpoints

**Files:**
- Modify: `block_lo_arm_order_network/batch_readout/train_l0_dynamic_gbeta.py:31-77`
- Modify: `block_lo_arm_order_network/tests/test_train_l0_dynamic_gbeta.py:27-132`

**Step 1: Write failing loader tests**

Update the synthetic dataset to save `consensus_order`. Assert:

```python
assert data["teacher_consensus_order"].shape == (32, 64)
assert data["teacher_consensus_order"].dtype == torch.int64
```

Add a legacy-loader test:

```python
data = load_pretrain_dataset(path_without_consensus, device="cpu",
                             require_consensus_order=False)
assert data["teacher_consensus_order"] is None
```

Add a strict test:

```python
with pytest.raises(KeyError, match="teacher_consensus_order"):
    load_pretrain_dataset(path_without_consensus, require_consensus_order=True)
```

**Step 2: Run tests to verify they fail**

```bash
python3 -m pytest \
  block_lo_arm_order_network/tests/test_train_l0_dynamic_gbeta.py \
  -k "load_pretrain_dataset" -q
```

Expected: FAIL because the loader has no consensus-order support.

**Step 3: Implement backward-compatible loading**

Add `require_consensus_order: bool = False`. Load the field when present and
return `None` otherwise. Continue requiring `teacher_pairwise` only when the
selected training objective needs it; do not silently synthesize pairwise
targets for ListMLE.

**Step 4: Run loader tests**

Run the command from Step 2.

Expected: PASS.

**Step 5: Commit**

```bash
git add block_lo_arm_order_network/batch_readout/train_l0_dynamic_gbeta.py \
  block_lo_arm_order_network/tests/test_train_l0_dynamic_gbeta.py
git commit -m "feat: load hard CDL order targets"
```

### Task 5: Dispatch the Training Objective by Loss Type

**Files:**
- Modify: `block_lo_arm_order_network/batch_readout/train_l0_dynamic_gbeta.py:82-146`
- Modify: `block_lo_arm_order_network/batch_readout/train_l0_dynamic_gbeta.py:153-350`
- Modify: `block_lo_arm_order_network/tests/test_train_l0_dynamic_gbeta.py:135-377`

**Step 1: Write failing objective-dispatch tests**

Parameterize tiny training:

```python
@pytest.mark.parametrize("loss_type", [
    "listmle", "pairwise_bce", "rank_kl",
])
def test_tiny_training_supports_each_loss_type(...):
    result = train(..., loss_type=loss_type, epochs=2)
    assert np.isfinite(result["best_val_loss_final"])
```

Assert config and checkpoint fields:

```python
assert cfg["loss_type"] == loss_type
assert cfg["rank_kl_temperature"] == 4.0
assert cfg["selection_metric"] == "val_loss_final"
```

Add an invalid loss-type test and a test that ListMLE has `loss_aux == 0.0`.

**Step 2: Run tests to verify they fail**

```bash
python3 -m pytest \
  block_lo_arm_order_network/tests/test_train_l0_dynamic_gbeta.py \
  -k "loss_type or tiny_training_supports" -q
```

Expected: FAIL because `train` and `_run_epoch` do not accept `loss_type`.

**Step 3: Refactor batch contents and objective computation**

Create:

```python
LOSS_TYPES = ("listmle", "pairwise_bce", "rank_kl")

def compute_primary_loss(
    scores,
    scores_per_head,
    teacher_order,
    teacher_pairwise,
    loss_type,
    rank_kl_temperature,
):
    ...
```

Rules:

- `listmle`: `listmle_loss(scores, teacher_order)`, no per-head pairwise aux.
- `pairwise_bce`: preserve current soft pairwise BCE and per-head auxiliary.
- `rank_kl`: `rank_kl_loss(scores, teacher_order, temperature)`, no pairwise
  auxiliary.
- gate entropy regularization remains active for all objectives.

DataLoaders should carry `B_raw`, hard order, and pairwise target. If a target
is unavailable and required by the selected objective, fail before training.

Keep metric keys stable:

```text
loss
loss_final
loss_aux
loss_ent
pairwise_acc
gate_entropy_mean
```

For every objective, compute `pairwise_acc` from the hard teacher order using
score comparisons. Keep the old soft-target pairwise accuracy under a separate
key such as `soft_pairwise_acc` only for `pairwise_bce`.

**Step 4: Update checkpoint selection and config**

Checkpoint selection remains the lowest validation primary loss. Save:

```python
"loss_type": loss_type,
"rank_kl_temperature": rank_kl_temperature,
"selection_metric": "val_loss_final",
```

**Step 5: Run focused and full training tests**

```bash
python3 -m pytest \
  block_lo_arm_order_network/tests/test_train_l0_dynamic_gbeta.py \
  -q
```

Expected: PASS.

**Step 6: Commit**

```bash
git add block_lo_arm_order_network/batch_readout/train_l0_dynamic_gbeta.py \
  block_lo_arm_order_network/tests/test_train_l0_dynamic_gbeta.py
git commit -m "feat: train gbeta with selectable order losses"
```

### Task 6: Add Objective-Agnostic Order Metrics

**Files:**
- Create: `block_lo_arm_order_network/batch_readout/order_distillation_metrics.py`
- Create: `block_lo_arm_order_network/tests/test_order_distillation_metrics.py`

**Step 1: Write failing metric tests**

Cover exact and reversed orders:

```python
perfect = order_metrics(scores_for(order), order)
assert perfect["kendall_tau"] == pytest.approx(1.0)
assert perfect["pairwise_acc"] == pytest.approx(1.0)
assert perfect["prefix8"] == pytest.approx(1.0)
assert perfect["prefix16"] == pytest.approx(1.0)
```

Verify batch averaging and shape validation.

**Step 2: Run tests to verify they fail**

```bash
python3 -m pytest \
  block_lo_arm_order_network/tests/test_order_distillation_metrics.py \
  -q
```

Expected: FAIL because the metrics module does not exist.

**Step 3: Implement metrics**

Implement tensor-native pairwise accuracy and NumPy/SciPy Kendall tau:

```python
def predicted_order(scores): ...
def hard_pairwise_accuracy(scores, teacher_order): ...
def prefix_overlap(scores, teacher_order, k): ...
def kendall_tau_batch(scores, teacher_order): ...
def order_metrics(scores, teacher_order): ...
```

Return normalized Prefix@K fractions, not raw counts.

**Step 4: Run metric tests**

Run the command from Step 2.

Expected: PASS.

**Step 5: Commit**

```bash
git add block_lo_arm_order_network/batch_readout/order_distillation_metrics.py \
  block_lo_arm_order_network/tests/test_order_distillation_metrics.py
git commit -m "feat: add CDL order fidelity metrics"
```

### Task 7: Update Evaluation for All Three Objectives

**Files:**
- Modify: `analyses/eval_l0_dynamic_gbeta.py:31-78`
- Modify: `analyses/eval_l0_dynamic_gbeta.py:100-164`
- Modify: `analyses/eval_l0_dynamic_gbeta.py:170-300`
- Modify: `analyses/characterize_l0_dynamic_gbeta.py:182-259`
- Create: `block_lo_arm_order_network/tests/test_eval_l0_dynamic_gbeta_losses.py`

**Step 1: Write failing evaluation tests**

For synthetic checkpoints of each loss type, assert summary output contains:

```text
primary_loss
kendall_tau
pairwise_acc
prefix8
prefix16
gate_entropy_mean
```

Assert evaluation reads the checkpoint's `loss_type` and uses the matching
primary loss. Add a test that old pairwise checkpoints default to
`pairwise_bce`.

**Step 2: Run tests to verify they fail**

```bash
python3 -m pytest \
  block_lo_arm_order_network/tests/test_eval_l0_dynamic_gbeta_losses.py \
  -q
```

Expected: FAIL because evaluation is hard-coded to pairwise BCE.

**Step 3: Implement objective-aware evaluation**

Load both `teacher_consensus_order` and `teacher_pairwise`. Use the shared loss
dispatcher and order metrics. Apply identical order metrics to normal,
destroyed, remove-top-alpha, and uniform-alpha scores.

In `characterize_l0_dynamic_gbeta.py`, consume the saved consensus order
directly instead of reconstructing it per sample. Add normalized Prefix@16.

**Step 4: Run focused evaluation tests**

Run the command from Step 2.

Expected: PASS.

**Step 5: Run relevant existing tests**

```bash
python3 -m pytest \
  block_lo_arm_order_network/tests/test_eval_l0_dynamic_gbeta.py \
  block_lo_arm_order_network/tests/test_train_l0_dynamic_gbeta.py \
  -q
```

Expected: PASS.

**Step 6: Commit**

```bash
git add analyses/eval_l0_dynamic_gbeta.py \
  analyses/characterize_l0_dynamic_gbeta.py \
  block_lo_arm_order_network/tests/test_eval_l0_dynamic_gbeta_losses.py
git commit -m "feat: evaluate gbeta order losses consistently"
```

### Task 8: Add Protocol-Matched Experiment Launcher and Summary

**Files:**
- Create: `scripts/run_gbeta_loss_comparison_20260625.sh`
- Create: `analyses/summarize_gbeta_loss_comparison.py`
- Create: `block_lo_arm_order_network/tests/test_summarize_gbeta_loss_comparison.py`

**Step 1: Write failing summary tests**

Create synthetic run directories with config and evaluation JSON. Assert the
summary table contains one row per objective and columns:

```text
loss_type
temperature
val_primary_loss
test_tau
test_pairwise_acc
test_prefix8
test_prefix16
train_seconds
peak_cuda_memory_mb
```

**Step 2: Run tests to verify they fail**

```bash
python3 -m pytest \
  block_lo_arm_order_network/tests/test_summarize_gbeta_loss_comparison.py \
  -q
```

Expected: FAIL because the summarizer does not exist.

**Step 3: Implement the launcher**

The launcher must:

1. Migrate the existing M=2000 dataset once if the consensus field is absent.
2. Run `listmle`.
3. Run protocol-matched `pairwise_bce`.
4. Run `rank_kl` at temperatures `2`, `4`, `8`, and `16`.
5. Use the same seed, architecture, data split, epochs, batch size, optimizer,
   learning rate, and gate regularization.
6. Write separate output directories and logs.

Do not include position weighting or combined loss.

**Step 4: Implement the summarizer**

Write both Markdown and JSON summaries. Mark the current acceptance references:

```text
tau ~= 0.984
pairwise accuracy ~= 0.997
Prefix@8 ~= 0.973
```

Do not automatically declare a winner from training loss because losses are on
different scales.

**Step 5: Run tests and shell validation**

```bash
python3 -m pytest \
  block_lo_arm_order_network/tests/test_summarize_gbeta_loss_comparison.py \
  -q
bash -n scripts/run_gbeta_loss_comparison_20260625.sh
```

Expected: PASS.

**Step 6: Commit**

```bash
git add scripts/run_gbeta_loss_comparison_20260625.sh \
  analyses/summarize_gbeta_loss_comparison.py \
  block_lo_arm_order_network/tests/test_summarize_gbeta_loss_comparison.py
git commit -m "exp: compare gbeta order distillation losses"
```

### Task 9: Full Verification and Smoke Run

**Files:**
- Modify only if verification finds defects.

**Step 1: Run the focused test suite**

```bash
python3 -m pytest \
  block_lo_arm_order_network/tests/test_order_distillation_losses.py \
  block_lo_arm_order_network/tests/test_order_distillation_metrics.py \
  block_lo_arm_order_network/tests/test_build_l0_dynamic_gbeta_dataset.py \
  block_lo_arm_order_network/tests/test_add_consensus_order_to_gbeta_dataset.py \
  block_lo_arm_order_network/tests/test_train_l0_dynamic_gbeta.py \
  block_lo_arm_order_network/tests/test_eval_l0_dynamic_gbeta_losses.py \
  block_lo_arm_order_network/tests/test_summarize_gbeta_loss_comparison.py \
  -q
```

Expected: PASS.

**Step 2: Compile modified Python modules**

```bash
python3 -m py_compile \
  analyses/build_l0_dynamic_gbeta_dataset.py \
  analyses/add_consensus_order_to_gbeta_dataset.py \
  analyses/eval_l0_dynamic_gbeta.py \
  analyses/characterize_l0_dynamic_gbeta.py \
  analyses/summarize_gbeta_loss_comparison.py \
  block_lo_arm_order_network/batch_readout/order_distillation_losses.py \
  block_lo_arm_order_network/batch_readout/order_distillation_metrics.py \
  block_lo_arm_order_network/batch_readout/train_l0_dynamic_gbeta.py
```

Expected: no output.

**Step 3: Run a CPU smoke comparison**

Use a temporary synthetic dataset and two epochs per objective. Verify every
run writes:

```text
config.json
g_beta_best.pt
g_beta_last.pt
metrics.jsonl
summary.json
```

**Step 4: Migrate the real dataset to a new file**

```bash
python3 analyses/add_consensus_order_to_gbeta_dataset.py \
  --src block_lo_arm_order_network/batch_readout/checkpoints/l0_dynamic_gbeta_ds_step20k_M2000.npz \
  --dst block_lo_arm_order_network/batch_readout/checkpoints/l0_dynamic_gbeta_ds_step20k_M2000_order.npz
```

Expected: new dataset with the original fields plus
`teacher_consensus_order`; source file unchanged.

**Step 5: Run one-batch real-data smoke per objective**

Run ListMLE, Pairwise BCE, and Rank-KL with a tiny epoch or batch limit. Confirm
finite losses, gradients, metrics, and checkpoint writes before launching the
full comparison.

**Step 6: Review the diff and commit verification fixes**

```bash
git status --short
git diff --check
```

Commit only fixes directly related to this implementation.

