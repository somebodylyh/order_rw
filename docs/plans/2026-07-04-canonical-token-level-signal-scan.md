# Canonical Token-Level Signal Scan Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the fixed-L2R token scan with the canonical strict65 random-reveal global-mean protocol and rerun method-gbeta 50k.

**Architecture:** Reuse the canonical checkpoint/chunk loader and strict model-frame extractor. Accumulate block and token A maps over `M * batch_size` independently randomized reveals, roll out CDL once per head from each global mean, then map both outputs to physical coordinates before rank-based comparison.

**Tech Stack:** Python, NumPy, PyTorch, SciPy, pytest.

---

### Task 1: Lock down protocol helpers with failing tests

**Files:**
- Create: `tests/test_token_level_signal_scan.py`
- Modify: `analyses/token_level_signal_scan.py`

**Step 1: Write failing tests**

Add tests that import the analysis module and require:

```python
def test_random_reveal_orders_are_independent_and_deterministic(): ...
def test_order_kendall_tau_uses_item_ranks(): ...
def test_model_token_order_maps_to_physical_blocks_and_offsets(): ...
def test_coarsegrain_token_B_recovers_block_B_off_diagonal(): ...
```

The Kendall test must use a permutation counterexample where direct
`scipy.stats.kendalltau(order_a, order_b)` differs from rank-based tau.

**Step 2: Run tests to verify RED**

Run:

```bash
python3 -m pytest tests/test_token_level_signal_scan.py -q
```

Expected: failures because the canonical helper functions do not exist.

**Step 3: Implement minimal helpers**

Add pure functions for:

- `random_reveal_orders(total, seed, num_blocks=64, block_len=4)`;
- `order_to_rank(order)` and `order_kendall_tau(left, right)`;
- block/token model-to-physical maps from `inv_perm`;
- masked 4x4 token-B coarse-graining for protocol verification.

**Step 4: Run tests to verify GREEN**

Run the same pytest command and require all tests to pass.

**Step 5: Commit**

Commit only the helper tests and helper implementation.

### Task 2: Replace fixed-L2R extraction with canonical global-mean extraction

**Files:**
- Modify: `analyses/token_level_signal_scan.py`
- Modify: `tests/test_token_level_signal_scan.py`

**Step 1: Write failing protocol tests**

Add a small synthetic test proving:

- `total == M * batch_size`;
- each sample receives its own random reveal;
- accumulation divides by `total` once;
- `forward_batch` changes execution chunking but not semantic sample count.

**Step 2: Run the protocol tests to verify RED**

Expected: the old `run_scan` does not expose canonical semantics.

**Step 3: Implement the canonical data flow**

Refactor `run_scan` to:

1. load model, model-frame chunks, `clean_perm`, and device through
   `neural_readout.extract_b._load_model_and_chunks`;
2. set `total = M * batch_size`;
3. generate `total` independent random block reveals with `seed+i`;
4. forward in `forward_batch` chunks;
5. accumulate model-frame block `(L,H,64,65)` and token `(L,H,256,257)` A maps;
6. divide by `total`, build B, run `C-D+L`, and posthoc-map both orders;
7. compute signed rank-tau and physical-frame boundary metrics;
8. save protocol and frame metadata.

Remove the long fixed-probe speculation comment and the fixed-L2R probe path.

**Step 4: Run all focused tests**

```bash
python3 -m pytest tests/test_token_level_signal_scan.py -q
python3 -m py_compile analyses/token_level_signal_scan.py
```

Expected: all pass with no warnings or syntax errors.

**Step 5: Commit**

Commit the canonical extraction refactor and protocol tests.

### Task 3: Run the canonical method-gbeta audit

**Files:**
- Generate: `analyses/token_level_signal_scan.json`

**Step 1: Run the checkpoint scan**

```bash
PYTHONPATH=chenhe_rerun:block_lo_arm_order_network \
python3 analyses/token_level_signal_scan.py \
  --ckpt chenhe_rerun/out/rerun/method_gbeta_bm16g1500_50k/ckpt.pt \
  --M 40 --batch-size 4 --forward-batch 4 --seed 42 --device cpu
```

**Step 2: Verify the artifact**

Check metadata reports `total=160`, independent random reveals, model-frame
strict extraction, posthoc physical mapping, and inverse-rank Kendall tau.
Require finite metrics for all 32 heads and near-zero coarse-graining error.

**Step 3: Compare against the deprecated result**

Report the new strongest heads, signed means, counts, and whether the original
token-level claim survives under the canonical protocol. Do not reuse the old
fixed-L2R JSON numbers.

**Step 4: Run final verification**

```bash
python3 -m pytest tests/test_token_level_signal_scan.py -q
python3 -m py_compile analyses/token_level_signal_scan.py
```

Expected: all tests pass and the result JSON is parseable with 32 finite rows.
