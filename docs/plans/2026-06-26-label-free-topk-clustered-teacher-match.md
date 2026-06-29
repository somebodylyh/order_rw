# Label-Free Top-k Clustered Teacher-Match Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a teacher-match-only audit showing whether label-free top-k carrier candidates can produce a clean clustered pairwise teacher without using physical L2R labels.

**Architecture:** Reuse saved strict65 `A_with_none_lh_mean.npy` tensors and existing label-free selector outputs. Add a clustering/teacher builder that turns selected candidate rollouts into naive and cluster-aware soft pairwise teachers, then train a tiny offline pairwise readout against those teachers. Oracle tau/L2R fields are used only in post-hoc reports.

**Tech Stack:** Python 3, NumPy, SciPy Kendall tau, PyTorch, existing `none_separated_block_graph.py`, `attn_order_teacher.py`, `batch_readout/soft_pairwise.py`, and `batch_readout/head_gated_gbeta.py`.

---

### Task 1: Candidate Selection Manifest

**Files:**
- Create: `scripts/build_label_free_candidate_sets.py`
- Test: `block_lo_arm_order_network/tests/test_build_label_free_candidate_sets.py`

**Step 1: Write failing tests**

Create tests that build tiny selector rows and verify:

```python
def test_topk_structure_selection_does_not_read_oracle_fields(tmp_path):
    rows = [
        {"layer": 1, "head": 7, "method": "L", "structure_score": 2.0, "posthoc_tau_vs_l2r": -1.0},
        {"layer": 0, "head": 1, "method": "L", "structure_score": 1.0, "posthoc_tau_vs_l2r": 1.0},
    ]
    out = select_candidates(rows, top_k=1, mode="structure")
    assert out[0]["layer"] == 1
    assert out[0]["head"] == 7
```

Also test deterministic `random4`:

```python
def test_random_selection_is_seeded():
    rows = [{"layer": 0, "head": i, "method": "L", "structure_score": float(i)} for i in range(8)]
    assert select_candidates(rows, top_k=4, mode="random", seed=123) == select_candidates(rows, top_k=4, mode="random", seed=123)
```

**Step 2: Run failing tests**

Run:

```bash
python3 -m pytest block_lo_arm_order_network/tests/test_build_label_free_candidate_sets.py -q
```

Expected: import/function-not-found failure.

**Step 3: Implement candidate builder**

Implement:

```python
def select_candidates(rows, top_k: int, mode: str, seed: int = 0):
    if mode == "structure":
        eligible = [r for r in rows if np.isfinite(float(r["structure_score"]))]
        return sorted(eligible, key=lambda r: float(r["structure_score"]), reverse=True)[:top_k]
    if mode == "random":
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(rows), size=top_k, replace=False)
        return [rows[int(i)] for i in idx]
    raise ValueError(mode)
```

CLI:

```bash
python3 scripts/build_label_free_candidate_sets.py \
  --selector-json reports/label_free_selector_audit_20260626/new5k/label_free_selector_rows.json \
  --out-dir reports/topk_cluster_teacher_match_20260626/new5k \
  --sets top4:structure:4 top8:structure:8 random4:random:4
```

Write one JSON per set:

```text
candidate_sets/top4.json
candidate_sets/top8.json
candidate_sets/random4_seed0.json
```

Do not include oracle tau in the selection logic. It may remain in copied rows only for final post-hoc reporting.

**Step 4: Run tests**

Run:

```bash
python3 -m pytest block_lo_arm_order_network/tests/test_build_label_free_candidate_sets.py -q
```

Expected: pass.

---

### Task 2: Order Clustering And Teacher Construction

**Files:**
- Create: `scripts/build_topk_clustered_pairwise_teacher.py`
- Test: `block_lo_arm_order_network/tests/test_topk_clustered_pairwise_teacher.py`

**Step 1: Write failing tests**

Test reverse orders split into separate clusters:

```python
def test_reverse_orders_split_into_clusters():
    orders = np.array([
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [3, 2, 1, 0],
        [3, 2, 1, 0],
    ])
    clusters = cluster_orders_by_tau(orders, threshold=0.0)
    assert sorted(len(c) for c in clusters) == [2, 2]
```

Test soft pairwise teacher confidence:

```python
def test_pairwise_teacher_confidence_marks_consensus():
    orders = np.array([[0, 1, 2], [0, 1, 2]])
    Y, C = build_soft_pairwise_from_orders(orders, np.array([0.5, 0.5]))
    assert Y[0, 1] == 1.0
    assert C[0, 1] == 0.5
```

**Step 2: Run failing tests**

Run:

```bash
python3 -m pytest block_lo_arm_order_network/tests/test_topk_clustered_pairwise_teacher.py -q
```

Expected: import/function-not-found failure.

**Step 3: Implement clustering and teacher builder**

Implement functions:

```python
def candidate_order(A_lh, candidate):
    B = build_none_separated_B(A_lh[layer, head])
    return rollout_by_method(B, method)

def cluster_orders_by_tau(orders, threshold=0.0):
    # Greedy connected components over graph edge tau(order_i, order_j) >= threshold.

def build_soft_pairwise_from_orders(orders, weights):
    # Y[a,b] = sum_i w_i * 1[rank_i[a] < rank_i[b]]
    # confidence C[a,b] = abs(Y[a,b] - 0.5)
```

CLI:

```bash
python3 scripts/build_topk_clustered_pairwise_teacher.py \
  --a-npy reports/5k_signal_provenance_audit_20260626/old_diag_jun25_5k/A_with_none_lh_mean.npy \
  --candidate-json reports/topk_cluster_teacher_match_20260626/new5k/candidate_sets/top8.json \
  --out-dir reports/topk_cluster_teacher_match_20260626/new5k/top8 \
  --cluster-threshold 0.0 \
  --score-temperature 0.5
```

Outputs:

```text
teacher_naive.npz
teacher_cluster0.npz
teacher_cluster1.npz
cluster_report.json
summary.md
```

Each `.npz` contains:

```text
B_heads: (K, 65, 65)
orders: (K, 64)
weights: (K,)
Y_pair: (64, 64)
confidence: (64, 64)
candidate_labels: strings
```

**Step 4: Run tests**

Run:

```bash
python3 -m pytest block_lo_arm_order_network/tests/test_topk_clustered_pairwise_teacher.py -q
```

Expected: pass.

---

### Task 3: Teacher-Match Model

**Files:**
- Create: `scripts/train_cluster_teacher_match.py`
- Test: `block_lo_arm_order_network/tests/test_train_cluster_teacher_match.py`

**Step 1: Write failing tests**

Use a synthetic teacher where order is known and verify the model can overfit:

```python
def test_tiny_pairwise_model_overfits_synthetic_teacher(tmp_path):
    B_heads = np.random.default_rng(0).normal(size=(4, 65, 65)).astype("float32")
    Y = np.zeros((64, 64), dtype="float32")
    for i in range(64):
        for j in range(64):
            Y[i, j] = 0.5 if i == j else float(i < j)
    C = np.abs(Y - 0.5).astype("float32")
    result = train_teacher_match(B_heads, Y, C, epochs=20, hidden=64, seed=0)
    assert result["val_pairwise_acc"] > 0.9
```

**Step 2: Run failing tests**

Run:

```bash
python3 -m pytest block_lo_arm_order_network/tests/test_train_cluster_teacher_match.py -q
```

Expected: import/function-not-found failure.

**Step 3: Implement minimal teacher-match trainer**

Start with a simple deterministic model, not continuation hook:

```python
class ClusterTeacherReadout(nn.Module):
    def __init__(self, K, N=64, hidden=256):
        self.gate = nn.Sequential(nn.Linear(K * 65 * 65, hidden), nn.GELU(), nn.Linear(hidden, K))
        self.scorer = nn.Sequential(nn.Linear(65 * 65, hidden), nn.GELU(), nn.Linear(hidden, N))

    def forward(self, B_heads):
        # B_heads: (batch, K, 65, 65)
        alpha = softmax(gate(flatten_all_heads))
        per_head_scores = scorer(each_head)
        scores = sum_i alpha_i * per_head_scores_i
        return scores, {"alpha": alpha, "per_head_scores": per_head_scores}
```

Loss:

```python
diff = scores[:, :, None] - scores[:, None, :]
loss = BCEWithLogits(diff[mask], Y[mask], weight=confidence[mask])
```

Metrics:

- `val_bce`;
- `confidence_weighted_val_acc`;
- `confidence_coverage` for thresholds 0.05, 0.10, 0.20;
- `gate_mean`;
- `gate_entropy`.

CLI:

```bash
python3 scripts/train_cluster_teacher_match.py \
  --teacher-npz reports/topk_cluster_teacher_match_20260626/new5k/top8/teacher_cluster0.npz \
  --out-dir reports/topk_cluster_teacher_match_20260626/new5k/top8/cluster0_match \
  --epochs 200 --lr 3e-4 --seed 0
```

**Step 4: Run tests**

Run:

```bash
python3 -m pytest block_lo_arm_order_network/tests/test_train_cluster_teacher_match.py -q
```

Expected: pass.

---

### Task 4: Batch Runner For First-Stage Matrix

**Files:**
- Create: `scripts/run_topk_cluster_teacher_match_20260626.sh`
- Create: `scripts/summarize_topk_cluster_teacher_match.py`

**Step 1: Write runner**

The runner must build candidate sets, build teachers, and train teacher-match for:

```text
new5k: top4 naive, top4 clustered, top8 naive, top8 clustered, random4 clustered
new10k: top4 naive, top4 clustered, top8 naive, top8 clustered, random4 clustered
```

Use existing inputs:

```text
new5k A: reports/5k_signal_provenance_audit_20260626/old_diag_jun25_5k/A_with_none_lh_mean.npy
new5k selector: reports/label_free_selector_audit_20260626/new5k/label_free_selector_rows.json
new10k A: reports/10k_signal_carrier_layer_multiseed_20260626/seed123_new/A_with_none_lh_mean.npy
new10k selector: reports/label_free_selector_audit_20260626/new10k/label_free_selector_rows.json
```

**Step 2: Write summarizer**

Summary table columns:

```text
ckpt, candidate_set, teacher_type, cluster_id, selected_candidates,
cluster_composition, cluster_size, confidence_coverage_0.1,
val_bce, confidence_weighted_val_acc, gate_mean,
posthoc_tau_mean, posthoc_tau_min, reverse_contamination_rate
```

`posthoc_*` fields are computed only after training and marked post-hoc in the summary.

**Step 3: Run smoke matrix**

Run one small smoke:

```bash
bash scripts/run_topk_cluster_teacher_match_20260626.sh smoke
```

Expected:

- creates `reports/topk_cluster_teacher_match_20260626`;
- finishes in minutes;
- summary includes both new5k and new10k top4 clustered rows.

**Step 4: Run full first-stage matrix**

Run:

```bash
bash scripts/run_topk_cluster_teacher_match_20260626.sh full
```

Expected:

- all required combinations complete;
- no continuation training is launched.

---

### Task 5: Success-Gate Report

**Files:**
- Create: `reports/topk_cluster_teacher_match_20260626/summary.md`

**Step 1: Generate report**

Use `scripts/summarize_topk_cluster_teacher_match.py` to write:

- selected candidates per setting;
- cluster composition;
- confidence coverage;
- teacher-match metrics;
- top8 naive vs top8 clustered comparison;
- random4 clustered comparison;
- post-hoc tau reveal.

**Step 2: Apply continuation gate**

Continuation is allowed only if most of these hold:

```text
top4/top8 clustered val acc > random4 clustered
top8 clustered > top8 naive when reverse contamination exists
confidence coverage at threshold 0.1 >= 0.40
post-hoc selected cluster tau is high and positive
new5k or new10k passes stably
```

If top8 fails but top4 passes, use top4 as primary and keep top8 as robustness failure/diagnostic.

**Step 3: Verification**

Run:

```bash
python3 -m py_compile scripts/build_label_free_candidate_sets.py scripts/build_topk_clustered_pairwise_teacher.py scripts/train_cluster_teacher_match.py scripts/summarize_topk_cluster_teacher_match.py
python3 -m pytest block_lo_arm_order_network/tests/test_build_label_free_candidate_sets.py block_lo_arm_order_network/tests/test_topk_clustered_pairwise_teacher.py block_lo_arm_order_network/tests/test_train_cluster_teacher_match.py -q
```

Expected: all pass.

---

## Non-Goals For This Plan

- No same-start continuation.
- No hook integration.
- No test NLL based head selection.
- No use of tau/L2R/physical-first/prefix for selection.
- No oracle top4 in the primary matrix; oracle top4 can be added later as post-hoc upper bound.

## First-Stage Decision Rule

Proceed to continuation only after the teacher-match report shows that clustered top-k is learnable, has non-trivial confidence coverage, and beats random4. If the clustered teacher fails, debug selector/clustering/teacher construction before spending GPU on continuation.
