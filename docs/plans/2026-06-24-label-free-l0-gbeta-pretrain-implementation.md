# Label-Free L0 Multi-Head g_beta Pretrain Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and evaluate a naive offline pretraining pipeline that learns a dynamic, head-ID-free `g_beta` from L0 all-head model-frame strict-65 attention maps and a label-free dynamic CDL consensus teacher.

**Architecture:** Extract 2,000 batch-mean L0 graphs from the random-baseline 20k checkpoint, construct per-sample soft pairwise CDL teachers from standardized margin, destroyed-gap, and rank agreement, then train a shared block scorer plus shared dynamic gate. Keep extraction, teacher construction, training, and checkpoint selection free of physical coordinates and permutation labels.

**Tech Stack:** Python, PyTorch, NumPy, SciPy, pytest, existing AO-GPT attention extraction and CDL utilities.

---

### Task 1: Add model-frame strict-65 all-head extraction

**Files:**
- Create: `block_lo_arm_order_network/batch_readout/l0_strict65.py`
- Test: `tests/test_l0_strict65.py`

**Step 1: Write the failing tests**

Test the conversion independently of AO-GPT:

```python
import numpy as np

from batch_readout.l0_strict65 import (
    build_model_frame_strict65,
    batch_mean_heads,
)


def test_build_model_frame_strict65_shape_and_none_node():
    rng = np.random.default_rng(0)
    attn = rng.random((2, 8, 257, 257), dtype=np.float32)
    probe_orders = np.stack([rng.permutation(256) for _ in range(2)])

    out = build_model_frame_strict65(attn, probe_orders)

    assert out.shape == (2, 8, 65, 65)
    assert np.isfinite(out).all()
    assert np.allclose(np.diagonal(out, axis1=-2, axis2=-1), 0.0)
    assert np.any(out[..., 0, 1:] != 0.0)
    assert np.all(out[..., 1:, 0] == 0.0)


def test_model_frame_extraction_ignores_physical_permutation():
    rng = np.random.default_rng(1)
    attn = rng.random((1, 8, 257, 257), dtype=np.float32)
    probe_orders = np.stack([rng.permutation(256)])

    first = build_model_frame_strict65(attn, probe_orders)
    second = build_model_frame_strict65(attn, probe_orders)

    np.testing.assert_allclose(first, second)


def test_batch_mean_heads_groups_samples():
    x = np.arange(4 * 8 * 65 * 65, dtype=np.float32).reshape(4, 8, 65, 65)
    out = batch_mean_heads(x, batch_mean_size=2)
    assert out.shape == (2, 8, 65, 65)
    np.testing.assert_allclose(out[0], x[:2].mean(axis=0))
```

**Step 2: Run the tests and verify failure**

Run:

```bash
python3 -m pytest tests/test_l0_strict65.py -v
```

Expected: FAIL because `batch_readout.l0_strict65` does not exist.

**Step 3: Implement the minimal extraction module**

Use the existing loss-aligned extractor with an identity model-frame mapping,
then build the None-separated graph:

```python
from __future__ import annotations

import numpy as np

from none_separated_block_graph import build_none_separated_B
from per_head_order_scan import _attn_to_A_block_loss_aligned_with_none_vec
from training_utils import N


def build_model_frame_strict65(
    attn_l0: np.ndarray,
    probe_orders: np.ndarray,
) -> np.ndarray:
    """Convert L0 token attention to model-frame strict-65 B graphs.

    attn_l0: (sample, head, 257, 257)
    probe_orders: (sample, 256), model-token reveal orders
    returns: (sample, head, 65, 65)
    """
    attn_l0 = np.asarray(attn_l0)
    probe_orders = np.asarray(probe_orders, dtype=np.int64)
    if attn_l0.ndim != 4:
        raise ValueError(f"attn_l0 must be 4D, got {attn_l0.shape}")
    if probe_orders.shape != (attn_l0.shape[0], 256):
        raise ValueError(
            f"probe_orders must be {(attn_l0.shape[0], 256)}, got {probe_orders.shape}"
        )

    identity = np.arange(N, dtype=np.int64)
    output = np.empty((attn_l0.shape[0], attn_l0.shape[1], N + 1, N + 1),
                      dtype=np.float32)
    for sample in range(attn_l0.shape[0]):
        A = _attn_to_A_block_loss_aligned_with_none_vec(
            attn_l0[sample],
            probe_orders[sample],
            identity,
        )
        for head in range(attn_l0.shape[1]):
            output[sample, head] = build_none_separated_B(A[head])
    return output


def batch_mean_heads(B: np.ndarray, batch_mean_size: int) -> np.ndarray:
    B = np.asarray(B, dtype=np.float32)
    if batch_mean_size <= 0 or B.shape[0] % batch_mean_size:
        raise ValueError("sample count must be divisible by batch_mean_size")
    groups = B.shape[0] // batch_mean_size
    return B.reshape(groups, batch_mean_size, *B.shape[1:]).mean(axis=1)
```

Do not accept `clean_perm`, `block_perm`, or `inv_perm` arguments in this
module.

**Step 4: Run tests**

Run:

```bash
python3 -m pytest tests/test_l0_strict65.py -v
```

Expected: 3 passed.

**Step 5: Commit**

```bash
git add block_lo_arm_order_network/batch_readout/l0_strict65.py tests/test_l0_strict65.py
git commit -m "feat: add model-frame strict65 L0 extraction"
```

### Task 2: Implement label-free CDL teacher utilities

**Files:**
- Create: `block_lo_arm_order_network/batch_readout/label_free_cdl_teacher.py`
- Test: `tests/test_label_free_cdl_teacher.py`

**Step 1: Write failing tests**

Cover rank conversion, standardized margin, agreement, destruction invariants,
and soft teacher consistency:

```python
import numpy as np

from batch_readout.label_free_cdl_teacher import (
    destroy_strict65,
    order_to_rank,
    rank_agreement,
    soft_teacher_from_heads,
    standardized_rollout_margin,
)


def test_order_to_rank_is_inverse_permutation():
    order = np.array([2, 0, 3, 1])
    np.testing.assert_array_equal(order_to_rank(order), [1, 3, 0, 2])


def test_rank_agreement_uses_precedence_not_order_values():
    same = rank_agreement(
        np.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.int64)
    )
    np.testing.assert_allclose(same, [1.0, 1.0])


def test_standardized_margin_is_scale_invariant():
    B = np.zeros((5, 5), dtype=np.float64)
    B[0, 1:] = [4, 3, 2, 1]
    B[1, 2:] = [3, 2, 1]
    assert standardized_rollout_margin(B) == pytest.approx(
        standardized_rollout_margin(B * 100.0)
    )


def test_destroy_preserves_rows_and_none_column():
    rng = np.random.default_rng(3)
    B = rng.random((65, 65))
    np.fill_diagonal(B, 0.0)
    B[1:, 0] = 0.0
    out = destroy_strict65(B, seed=9)

    np.testing.assert_array_equal(out[1:, 0], B[1:, 0])
    np.testing.assert_array_equal(np.diag(out), np.zeros(65))
    for row in range(1, 65):
        np.testing.assert_allclose(
            np.sort(out[row, 1:][np.arange(64) != row - 1]),
            np.sort(B[row, 1:][np.arange(64) != row - 1]),
        )


def test_soft_teacher_is_antisymmetric():
    orders = np.array([
        [0, 1, 2, 3],
        [0, 2, 1, 3],
    ])
    quality = np.array([1.0, 0.0])
    result = soft_teacher_from_heads(orders, quality, temperature=1.0, smoothing=0.05)
    Y = result["pairwise"]
    np.testing.assert_allclose(Y + Y.T, np.ones_like(Y), atol=1e-6)
    np.testing.assert_allclose(np.diag(Y), 0.5)
```

Add the missing `pytest` import.

**Step 2: Run tests and verify failure**

```bash
python3 -m pytest tests/test_label_free_cdl_teacher.py -v
```

Expected: FAIL because the module does not exist.

**Step 3: Implement minimal utilities**

Required public functions:

```python
def order_to_rank(order: np.ndarray) -> np.ndarray
def rollout_with_standardized_margins(B65: np.ndarray) -> tuple[np.ndarray, np.ndarray]
def standardized_rollout_margin(B65: np.ndarray) -> float
def destroy_strict65(B65: np.ndarray, seed: int) -> np.ndarray
def rank_agreement(orders: np.ndarray) -> np.ndarray
def headwise_zscore(values: np.ndarray, eps: float = 1e-6) -> np.ndarray
def soft_teacher_from_heads(
    orders: np.ndarray,
    quality: np.ndarray,
    temperature: float = 1.0,
    smoothing: float = 0.05,
) -> dict
def build_dynamic_teacher(B_heads: np.ndarray, destroy_seed: int) -> dict
```

Implementation rules:

- Rollout starts with selected node 0 and candidates 1 through 64.
- Return content block IDs 0 through 63.
- At each step, call existing `teacher_scores`.
- Z-score candidate utilities before computing top1-minus-top2.
- `destroy_strict65` must preserve the content-to-None column and zero diagonal.
- Agreement converts every order to rank before calling SciPy `kendalltau`.
- Per-sample quality is:

```python
quality = (
    headwise_zscore(margin)
    + headwise_zscore(destroyed_gap)
    + headwise_zscore(agreement)
)
```

- Teacher weights are:

```python
weights = (1.0 - smoothing) * softmax(quality / temperature) + smoothing / H
```

- Pairwise diagonal is exactly 0.5.

**Step 4: Run focused tests**

```bash
python3 -m pytest tests/test_label_free_cdl_teacher.py -v
```

Expected: all tests pass.

**Step 5: Run existing CDL tests**

```bash
python3 -m pytest block_lo_arm_order_network/test_attn_order.py \
  block_lo_arm_order_network/tests/test_none_separated_block_graph.py -v
```

Expected: all tests pass.

**Step 6: Commit**

```bash
git add block_lo_arm_order_network/batch_readout/label_free_cdl_teacher.py \
  tests/test_label_free_cdl_teacher.py
git commit -m "feat: add dynamic label-free CDL teacher"
```

### Task 3: Build the offline L0 pretrain dataset

**Files:**
- Create: `analyses/build_l0_dynamic_gbeta_dataset.py`
- Test: `tests/test_build_l0_dynamic_gbeta_dataset.py`

**Step 1: Write failing serialization tests**

Use synthetic graphs and avoid loading AO-GPT in unit tests:

```python
import numpy as np

from build_l0_dynamic_gbeta_dataset import (
    deterministic_split,
    save_dataset,
)


def test_deterministic_split_is_disjoint():
    split = deterministic_split(20, seed=7)
    joined = np.concatenate([split["train"], split["val"], split["test"]])
    assert len(np.unique(joined)) == 20
    assert len(split["train"]) == 16
    assert len(split["val"]) == 2
    assert len(split["test"]) == 2


def test_saved_dataset_contains_auditable_teacher_fields(tmp_path):
    B = np.zeros((4, 8, 65, 65), dtype=np.float32)
    teacher = {
        "orders": np.zeros((4, 8, 64), dtype=np.int64),
        "ranks": np.zeros((4, 8, 64), dtype=np.int64),
        "margin": np.zeros((4, 8), dtype=np.float32),
        "destroyed_gap": np.zeros((4, 8), dtype=np.float32),
        "agreement": np.zeros((4, 8), dtype=np.float32),
        "quality": np.zeros((4, 8), dtype=np.float32),
        "weights": np.full((4, 8), 1 / 8, dtype=np.float32),
        "pairwise": np.full((4, 64, 64), 0.5, dtype=np.float32),
    }
    path = tmp_path / "ds.npz"
    save_dataset(path, B, teacher, deterministic_split(4, seed=0), {"source": "test"})
    with np.load(path, allow_pickle=True) as z:
        assert z["B_raw"].shape == (4, 8, 65, 65)
        assert z["teacher_pairwise"].shape == (4, 64, 64)
        assert "inv_perm" not in z.files
        assert "block_perm" not in z.files
```

**Step 2: Run tests and verify failure**

```bash
python3 -m pytest tests/test_build_l0_dynamic_gbeta_dataset.py -v
```

Expected: FAIL because the builder module does not exist.

**Step 3: Implement dataset builder helpers**

The CLI must expose:

```text
--ckpt
--M
--batch-mean-size
--seed
--device
--split train
--forward-batch
--destroy-replicas
--teacher-temperature
--teacher-smoothing
--out
```

Defaults:

```text
M=2000
batch_mean_size=16
seed=2
layer=0 fixed
destroy_replicas=1
teacher_temperature=1.0
teacher_smoothing=0.05
```

Reuse `_load_model_and_chunks` only for checkpoint/model/data loading. Do not
copy teacher logic into this script.

Extract in bounded forward batches:

```python
_, _, attn_list = model.forward_fn(idx_batch, probe_orders, return_attentions=True)
attn_l0 = attn_list[0].cpu().numpy()
B_samples = build_model_frame_strict65(attn_l0, probe_orders.cpu().numpy())
```

Generate deterministic random model-frame probe orders. Do not use
`clean_perm` after loading the checkpoint.

Save these fields:

```text
B_raw
head_orders
head_ranks
quality_margin
quality_destroyed_gap
quality_agreement
quality_total
teacher_weights
teacher_pairwise
train_idx
val_idx
test_idx
meta_json
```

Do not save `block_perm` or `inv_perm`.

**Step 4: Run unit tests**

```bash
python3 -m pytest tests/test_build_l0_dynamic_gbeta_dataset.py -v
```

Expected: all tests pass.

**Step 5: Run a small CPU/single-GPU smoke extraction**

```bash
python3 analyses/build_l0_dynamic_gbeta_dataset.py \
  --ckpt block_lo_arm_order_network/probe_results/random_baseline_continuous_jun08_seed2/ckpt_step20000.pt \
  --M 16 \
  --batch-mean-size 2 \
  --forward-batch 2 \
  --destroy-replicas 1 \
  --device cuda:0 \
  --out /tmp/l0_dynamic_gbeta_smoke.npz
```

Expected:

```text
B_raw shape: (16, 8, 65, 65)
teacher_pairwise shape: (16, 64, 64)
finite: true
saved: /tmp/l0_dynamic_gbeta_smoke.npz
```

If CUDA is unavailable, use `--device cpu --M 2 --batch-mean-size 1`.

**Step 6: Add an audit command**

```bash
python3 -c "import numpy as np; z=np.load('/tmp/l0_dynamic_gbeta_smoke.npz'); print(z.files); print(z['B_raw'].shape); print(z['teacher_weights'][0]); print(z['teacher_pairwise'].min(), z['teacher_pairwise'].max())"
```

Expected: no permutation fields; weights sum to 1; pairwise targets lie in
`[0, 1]`.

**Step 7: Commit**

```bash
git add analyses/build_l0_dynamic_gbeta_dataset.py \
  tests/test_build_l0_dynamic_gbeta_dataset.py
git commit -m "feat: build L0 dynamic gbeta dataset"
```

### Task 4: Implement normalization and the naive dynamic g_beta

**Files:**
- Create: `block_lo_arm_order_network/batch_readout/l0_dynamic_gbeta.py`
- Test: `tests/test_l0_dynamic_gbeta.py`

**Step 1: Write failing model tests**

```python
import torch

from batch_readout.l0_dynamic_gbeta import (
    L0DynamicGBeta,
    normalize_strict65,
)


def test_normalize_strict65_shapes_and_finiteness():
    B = torch.rand(3, 8, 65, 65)
    channels = normalize_strict65(B)
    assert channels.shape == (3, 8, 4, 65, 65)
    assert torch.isfinite(channels).all()
    assert torch.allclose(channels[:, :, 1].sum(dim=-1)[B.sum(dim=-1) > 0],
                          torch.ones_like(B.sum(dim=-1)[B.sum(dim=-1) > 0]),
                          atol=1e-5)


def test_model_outputs_dynamic_alpha_without_head_ids():
    torch.manual_seed(0)
    model = L0DynamicGBeta()
    B = torch.rand(4, 8, 65, 65)
    scores, aux = model(B)
    assert scores.shape == (4, 64)
    assert aux["scores_per_head"].shape == (4, 8, 64)
    assert aux["alpha"].shape == (4, 8)
    torch.testing.assert_close(aux["alpha"].sum(dim=1), torch.ones(4))
    assert aux["alpha"].std(dim=0).mean() > 0


def test_head_permutation_equivariance():
    torch.manual_seed(1)
    model = L0DynamicGBeta()
    model.eval()
    B = torch.rand(2, 8, 65, 65)
    permutation = torch.tensor([3, 0, 7, 1, 5, 2, 6, 4])
    inverse = torch.argsort(permutation)
    score_a, aux_a = model(B, apply_head_dropout=False)
    score_b, aux_b = model(B[:, permutation], apply_head_dropout=False)
    torch.testing.assert_close(score_a, score_b, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(aux_a["alpha"], aux_b["alpha"][:, inverse],
                               atol=1e-5, rtol=1e-5)
```

The permutation-equivariance test is the guard against hidden head identity.

**Step 2: Run tests and verify failure**

```bash
python3 -m pytest tests/test_l0_dynamic_gbeta.py -v
```

Expected: FAIL because the model module does not exist.

**Step 3: Implement normalization**

Return channels ordered as raw, probability, log-z, and row-z:

```python
def normalize_strict65(B, eps=1e-8):
    raw = B.float()
    row_sum = raw.sum(dim=-1, keepdim=True)
    prob = raw / row_sum.clamp_min(eps)

    log_raw = torch.log(raw.clamp_min(eps))
    log_mean = log_raw.mean(dim=(-2, -1), keepdim=True)
    log_std = log_raw.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    logz = (log_raw - log_mean) / log_std

    row_mean = raw.mean(dim=-1, keepdim=True)
    row_std = raw.std(dim=-1, keepdim=True).clamp_min(1e-6)
    rowz = (raw - row_mean) / row_std
    return torch.stack([raw, prob, logz, rowz], dim=2)
```

Keep zero rows finite.

**Step 4: Implement the model**

Required structure:

```python
class L0DynamicGBeta(nn.Module):
    def __init__(self, heads=8, nodes=65, hidden=(256, 64), gate_hidden=32):
        ...

    def forward(self, B_raw, apply_head_dropout=None):
        channels = normalize_strict65(B_raw)
        block_features = build_row_col_features(channels)
        scores_h = self.block_scorer(block_features).squeeze(-1)

        gate_channels = channels[:, :, 1:]  # no raw channel
        gate_summary = torch.cat([
            gate_channels.mean(dim=(-2, -1)),
            gate_channels.std(dim=(-2, -1)),
            gate_channels.amax(dim=(-2, -1)),
            gate_channels.amin(dim=(-2, -1)),
        ], dim=-1)
        gate_logits = self.gate_encoder(gate_summary).squeeze(-1)
        ...
```

For content block `i`, use node index `i + 1`. Concatenate the complete row
and column from all four channels, producing 520 features.

Head dropout:

- active only while training unless explicitly overridden;
- mask one independently sampled head per sample;
- replace its gate logit with `-inf`;
- do not alter `scores_per_head`.

Return:

```python
scores, {
    "scores_per_head": scores_h,
    "gate_logits": gate_logits,
    "alpha": alpha,
    "dropped_head": dropped_head,
}
```

**Step 5: Run model tests**

```bash
python3 -m pytest tests/test_l0_dynamic_gbeta.py -v
```

Expected: all tests pass.

**Step 6: Commit**

```bash
git add block_lo_arm_order_network/batch_readout/l0_dynamic_gbeta.py \
  tests/test_l0_dynamic_gbeta.py
git commit -m "feat: add dynamic L0 gbeta model"
```

### Task 5: Implement soft pairwise losses and metrics

**Files:**
- Create: `block_lo_arm_order_network/batch_readout/soft_pairwise.py`
- Test: `tests/test_soft_pairwise.py`

**Step 1: Write failing tests**

```python
import torch

from batch_readout.soft_pairwise import (
    gate_entropy_floor,
    soft_pairwise_accuracy,
    soft_pairwise_bce,
)


def test_soft_pairwise_bce_prefers_correct_scores():
    target = torch.tensor([[
        [0.5, 1.0, 1.0],
        [0.0, 0.5, 1.0],
        [0.0, 0.0, 0.5],
    ]])
    correct = torch.tensor([[3.0, 2.0, 1.0]])
    wrong = -correct
    assert soft_pairwise_bce(correct, target) < soft_pairwise_bce(wrong, target)


def test_soft_pairwise_accuracy_ignores_diagonal():
    scores = torch.tensor([[3.0, 2.0, 1.0]])
    target = torch.tensor([[
        [0.5, 1.0, 1.0],
        [0.0, 0.5, 1.0],
        [0.0, 0.0, 0.5],
    ]])
    assert soft_pairwise_accuracy(scores, target) == 1.0


def test_entropy_floor_penalizes_collapsed_gate():
    uniform = torch.full((2, 8), 1 / 8)
    collapsed = torch.nn.functional.one_hot(torch.tensor([0, 1]), 8).float()
    assert gate_entropy_floor(collapsed, min_entropy=1.5) > \
           gate_entropy_floor(uniform, min_entropy=1.5)
```

**Step 2: Run tests and verify failure**

```bash
python3 -m pytest tests/test_soft_pairwise.py -v
```

Expected: FAIL because the module does not exist.

**Step 3: Implement the loss**

Use only the strict upper triangle:

```python
def soft_pairwise_bce(scores, target):
    diff = scores.unsqueeze(-1) - scores.unsqueeze(-2)
    n = scores.shape[-1]
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=scores.device), diagonal=1)
    return F.binary_cross_entropy_with_logits(diff[:, mask], target[:, mask])
```

`soft_pairwise_accuracy` compares the sign of score differences with whether
the soft target is greater than 0.5. Exclude exact 0.5 ties from the metric.

`gate_entropy_floor` returns:

```python
relu(min_entropy - entropy(alpha)).mean()
```

**Step 4: Run tests**

```bash
python3 -m pytest tests/test_soft_pairwise.py -v
```

Expected: all tests pass.

**Step 5: Commit**

```bash
git add block_lo_arm_order_network/batch_readout/soft_pairwise.py \
  tests/test_soft_pairwise.py
git commit -m "feat: add soft pairwise training losses"
```

### Task 6: Implement offline training and checkpoint selection

**Files:**
- Create: `block_lo_arm_order_network/batch_readout/train_l0_dynamic_gbeta.py`
- Test: `tests/test_train_l0_dynamic_gbeta.py`

**Step 1: Write a tiny end-to-end failing test**

Create a synthetic dataset where one head contains an obvious ordering signal:

```python
def test_tiny_training_improves_validation_pairwise_accuracy(tmp_path):
    dataset = make_synthetic_dataset(samples=32, seed=0)
    path = tmp_path / "synthetic.npz"
    np.savez(path, **dataset)
    result = train(
        dataset_path=str(path),
        out_dir=str(tmp_path / "run"),
        epochs=4,
        batch_size=8,
        device="cpu",
        seed=0,
    )
    assert result["best_val_pairwise_acc"] > 0.7
    assert (tmp_path / "run" / "g_beta_best.pt").exists()
    assert (tmp_path / "run" / "history.json").exists()
```

Keep the synthetic generator in the test file.

**Step 2: Run test and verify failure**

```bash
python3 -m pytest tests/test_train_l0_dynamic_gbeta.py -v
```

Expected: FAIL because the trainer does not exist.

**Step 3: Implement the trainer**

CLI arguments:

```text
--dataset
--out-dir
--epochs 40
--batch-size 32
--lr 3e-4
--weight-decay 1e-2
--lambda-aux 0.05
--lambda-entropy 0.001
--min-gate-entropy 1.5
--seed 0
--device cuda:0
```

Per batch:

```python
scores, aux = model(B)
loss_final = soft_pairwise_bce(scores, teacher_pairwise)
loss_aux = mean(
    soft_pairwise_bce(aux["scores_per_head"][:, h], teacher_pairwise)
    for h in range(8)
)
loss_ent = gate_entropy_floor(aux["alpha"], min_gate_entropy)
loss = loss_final + lambda_aux * loss_aux + lambda_entropy * loss_ent
```

Select the checkpoint only by validation teacher-pairwise accuracy.

Save:

```text
g_beta_best.pt
history.json
config.json
```

Checkpoint config must include:

```text
model_name=l0_dynamic_gbeta_v0
heads=8
nodes=65
head_identity=false
lambda_aux
lambda_entropy
min_gate_entropy
dataset_path
dataset_meta
```

**Step 4: Run the tiny training test**

```bash
python3 -m pytest tests/test_train_l0_dynamic_gbeta.py -v
```

Expected: pass.

**Step 5: Run all new unit tests**

```bash
python3 -m pytest \
  tests/test_l0_strict65.py \
  tests/test_label_free_cdl_teacher.py \
  tests/test_build_l0_dynamic_gbeta_dataset.py \
  tests/test_l0_dynamic_gbeta.py \
  tests/test_soft_pairwise.py \
  tests/test_train_l0_dynamic_gbeta.py -v
```

Expected: all tests pass.

**Step 6: Commit**

```bash
git add block_lo_arm_order_network/batch_readout/train_l0_dynamic_gbeta.py \
  tests/test_train_l0_dynamic_gbeta.py
git commit -m "feat: train dynamic L0 gbeta offline"
```

### Task 7: Implement evaluation and required sanity checks

**Files:**
- Create: `analyses/eval_l0_dynamic_gbeta.py`
- Test: `tests/test_eval_l0_dynamic_gbeta.py`

**Step 1: Write failing ablation tests**

```python
def test_remove_top_alpha_masks_and_renormalizes():
    alpha = torch.tensor([[0.1, 0.6, 0.3]])
    masked = remove_top_alpha(alpha)
    torch.testing.assert_close(masked, torch.tensor([[0.25, 0.0, 0.75]]))


def test_destroyed_eval_uses_same_structure_preserving_destroy():
    B = synthetic_strict65_batch()
    destroyed = destroy_batch(B, seed=4)
    assert destroyed.shape == B.shape
    torch.testing.assert_close(destroyed[..., 1:, 0], B[..., 1:, 0])
```

**Step 2: Run tests and verify failure**

```bash
python3 -m pytest tests/test_eval_l0_dynamic_gbeta.py -v
```

Expected: FAIL because evaluation helpers do not exist.

**Step 3: Implement evaluation**

Evaluate the test split under:

```text
normal
destroyed
remove_top_alpha
uniform_alpha
single_head_best_by_validation
```

The single-head baseline may select a head using validation teacher-pairwise
accuracy. It must not use physical metrics.

Report:

```text
pairwise_acc
kendall_tau_vs_consensus
prefix4_overlap
prefix8_overlap
teacher_weight_entropy
student_alpha_entropy
alpha_sample_std
alpha_teacher_pearson
alpha_teacher_kl
alpha_teacher_top_head_agreement
```

Save:

```text
summary.json
per_sample.tsv
alpha_summary.tsv
```

Do not import or load `block_perm` or `inv_perm` in this script.

**Step 4: Run evaluation tests**

```bash
python3 -m pytest tests/test_eval_l0_dynamic_gbeta.py -v
```

Expected: all tests pass.

**Step 5: Commit**

```bash
git add analyses/eval_l0_dynamic_gbeta.py tests/test_eval_l0_dynamic_gbeta.py
git commit -m "feat: evaluate dynamic L0 gbeta sanity checks"
```

### Task 8: Add smoke and full-run launch scripts

**Files:**
- Create: `scripts/run_l0_dynamic_gbeta_pretrain_smoke.sh`
- Create: `scripts/run_l0_dynamic_gbeta_pretrain_full.sh`
- Test: `tests/test_l0_dynamic_gbeta_scripts.py`

**Step 1: Write script-content tests**

Assert the full script:

- uses the exact random-baseline 20k checkpoint;
- sets `M=2000`;
- sets batch-mean size 16;
- does not contain `inv_perm`, `block_perm`, `physical`, `frozen_beta`, or
  `train_clean_aogpt.py`;
- runs builder, trainer, then evaluator.

**Step 2: Run tests and verify failure**

```bash
python3 -m pytest tests/test_l0_dynamic_gbeta_scripts.py -v
```

Expected: FAIL because scripts do not exist.

**Step 3: Implement scripts**

Smoke:

```text
M=16
batch_mean_size=2
epochs=2
device configurable
output under /tmp or a smoke-named probe_results directory
```

Full:

```text
M=2000
batch_mean_size=16
epochs=40
W&B not required for offline v0
timestamped output under block_lo_arm_order_network/batch_readout/logs/
```

Both scripts use `set -euo pipefail` and log exact commands and output paths.

**Step 4: Validate scripts**

```bash
bash -n scripts/run_l0_dynamic_gbeta_pretrain_smoke.sh
bash -n scripts/run_l0_dynamic_gbeta_pretrain_full.sh
python3 -m pytest tests/test_l0_dynamic_gbeta_scripts.py -v
```

Expected: syntax checks pass and tests pass.

**Step 5: Run smoke end to end**

```bash
GPU=cuda:0 bash scripts/run_l0_dynamic_gbeta_pretrain_smoke.sh
```

Expected:

- dataset generated;
- trainer writes `g_beta_best.pt`;
- evaluator writes `summary.json`;
- no AO-GPT continuation training starts.

**Step 6: Commit**

```bash
git add scripts/run_l0_dynamic_gbeta_pretrain_smoke.sh \
  scripts/run_l0_dynamic_gbeta_pretrain_full.sh \
  tests/test_l0_dynamic_gbeta_scripts.py
git commit -m "feat: add L0 gbeta pretrain launch scripts"
```

### Task 9: Final verification before the full extraction

**Files:**
- Modify only if verification reveals a defect.

**Step 1: Run the complete focused test suite**

```bash
python3 -m pytest \
  tests/test_l0_strict65.py \
  tests/test_label_free_cdl_teacher.py \
  tests/test_build_l0_dynamic_gbeta_dataset.py \
  tests/test_l0_dynamic_gbeta.py \
  tests/test_soft_pairwise.py \
  tests/test_train_l0_dynamic_gbeta.py \
  tests/test_eval_l0_dynamic_gbeta.py \
  tests/test_l0_dynamic_gbeta_scripts.py -v
```

Expected: all tests pass.

**Step 2: Run relevant existing regression tests**

```bash
python3 -m pytest \
  block_lo_arm_order_network/test_attn_order.py \
  block_lo_arm_order_network/tests/test_none_separated_block_graph.py \
  block_lo_arm_order_network/tests/test_batch_readout_integration_hook.py \
  block_lo_arm_order_network/tests/test_hook_order_provider.py -v
```

Expected: all tests pass.

**Step 3: Audit the smoke dataset**

Verify:

```text
B_raw shape is (16, 8, 65, 65)
diagonal is zero
content-to-None column matches strict-65 definition
all teacher weights sum to one
pairwise targets are antisymmetric around 0.5
no permutation arrays are stored
```

**Step 4: Audit the model**

Verify:

```text
head permutation equivariance passes
alpha varies across samples
destroyed input changes scores and alpha
remove-top-alpha evaluation renormalizes correctly
checkpoint selection uses only teacher-pairwise validation accuracy
```

**Step 5: Run the full offline experiment**

```bash
GPU=cuda:0 bash scripts/run_l0_dynamic_gbeta_pretrain_full.sh
```

Do not launch frozen-hook or alpha co-training automatically after completion.

**Step 6: Record results**

Add a short result note under the run directory containing:

```text
source checkpoint
dataset hash
git commit
best epoch
test pairwise accuracy
test Kendall tau
destroyed delta
remove-top-head delta
alpha sample variation
```

**Step 7: Final commit if result-report plumbing was added**

```bash
git add <only files changed for result-report plumbing>
git commit -m "chore: finalize L0 gbeta pretrain verification"
```
