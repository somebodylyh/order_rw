# Direct Attention Policy Baselines Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add three non-learned model-frame strict65 order policies that are directly comparable to the new frozen `g_beta`.

**Architecture:** Extract the probe-averaged L0 all-head strict65 tensor through one shared helper. Feed it either to frozen `g_beta` or to pure direct score functions, then return a model-frame block order to the existing training loop.

**Tech Stack:** Python, PyTorch, NumPy, pytest, AO-GPT training CLI.

---

### Task 1: Lock Down Direct Score Semantics

**Files:**
- Create: `block_lo_arm_order_network/tests/test_direct_order_provider.py`
- Create: `block_lo_arm_order_network/batch_readout/direct_order_provider.py`

**Step 1: Write failing score tests**

Add tests for:

- `(batch, heads, 65, 65) -> (batch, heads, 64), (batch, 64)`
- `initial_cdl_one_shot` column dependency and descending order
- diagonal exclusion even when diagonal is non-zero
- `source_mass` row direction
- `readiness = row - lambda_dep * column`
- arithmetic mean fusion over heads

**Step 2: Verify RED**

Run:

```bash
python3 -m pytest block_lo_arm_order_network/tests/test_direct_order_provider.py -q
```

Expected: import failure because `direct_order_provider.py` does not exist.

**Step 3: Implement minimal pure score functions**

Implement:

```python
initial_cdl_scores_strict65(B)
source_mass_scores_strict65(B)
readiness_scores_strict65(B, lambda_dep=1.0)
direct_policy_scores(B, policy, lambda_dep=1.0)
```

Validate rank and strict65 shape. Explicitly subtract the diagonal before row
and column sums. Return per-head and mean-head scores.

**Step 4: Verify GREEN**

Run the focused test file and expect all tests to pass.

### Task 2: Share the Probe-Averaged strict65 Extraction

**Files:**
- Modify: `block_lo_arm_order_network/batch_readout/frozen_gbeta_hook.py`
- Modify: `block_lo_arm_order_network/tests/test_frozen_gbeta_hook.py`

**Step 1: Write a failing extraction-helper test**

Use a fake AO-GPT model to assert that the helper:

- runs the requested probe count
- returns `(batch, heads, 65, 65)`
- is deterministic in `(seed, global_step)`
- has no physical-coordinate parameters

**Step 2: Verify RED**

Run the focused helper test and expect missing-symbol failure.

**Step 3: Extract the existing loop into a helper**

Add `extract_probe_averaged_model_frame_strict65(...)` and make
`FrozenGBetaModelFrameProvider` call it without changing score behavior.

**Step 4: Verify GREEN**

Run frozen-hook and strict65 tests.

### Task 3: Add the Direct Provider

**Files:**
- Modify: `block_lo_arm_order_network/batch_readout/direct_order_provider.py`
- Modify: `block_lo_arm_order_network/tests/test_direct_order_provider.py`

**Step 1: Write failing provider tests**

Test valid model-frame permutation, larger-score-first behavior, refresh cache,
and label-free constructor/source audit.

**Step 2: Verify RED**

Run the focused provider tests and expect missing-provider failure.

**Step 3: Implement `DirectModelFrameOrderProvider`**

Call the shared extraction helper, score the tensor, take descending `argsort`,
cache according to `refresh_every`, and return the first model-frame batch
order to match the current frozen `g_beta` block-provider contract.

**Step 4: Verify GREEN**

Run the focused provider tests.

### Task 4: Integrate the Training CLI and Audit Metadata

**Files:**
- Modify: `block_lo_arm_order_network/train_clean_aogpt.py`
- Modify: `block_lo_arm_order_network/tests/test_train_clean_wandb_logging.py`
- Create or modify: `block_lo_arm_order_network/tests/test_direct_policy_training_config.py`

**Step 1: Write failing parser and audit tests**

Assert:

- `--run-kind direct_policy` parses
- all three policy names parse
- default `lambda_dep` is `1.0`
- config audit contains the agreed input protocol

**Step 2: Verify RED**

Run the focused tests and expect parser-choice/config failures.

**Step 3: Implement one generic training branch**

Add `direct_policy` to alpha scheduling, evaluation, continuous-data
allow-list, provider initialization, and per-sample alpha mixing. Record
`val_direct_order` and policy audit metadata.

**Step 4: Verify GREEN**

Run focused training/config tests.

### Task 5: Add Formal Run Commands and Audit Table

**Files:**
- Create: `scripts/run_direct_policy_aligned_20260625.sh`
- Create: `reports/direct_policy_input_audit_20260625.md`

**Step 1: Add three aligned commands**

Match the frozen `g_beta` start checkpoint, schedule, data, probe count,
refresh interval, and evaluation settings. Vary only `--direct-policy` and
`--direct-policy-lambda-dep` where applicable.

**Step 2: Add the input-protocol audit table**

Compare new `g_beta`, the three direct policies, and historical sequential CDL.
Mark sequential CDL as non-aligned historical/expensive reference.

**Step 3: Syntax-check the shell script**

Run:

```bash
bash -n scripts/run_direct_policy_aligned_20260625.sh
```

Expected: exit code 0.

### Task 6: Full Verification

**Files:**
- Verify all modified files.

**Step 1: Run focused tests**

```bash
python3 -m pytest \
  block_lo_arm_order_network/tests/test_direct_order_provider.py \
  block_lo_arm_order_network/tests/test_frozen_gbeta_hook.py \
  block_lo_arm_order_network/tests/test_l0_strict65.py \
  block_lo_arm_order_network/tests/test_direct_policy_training_config.py -q
```

**Step 2: Run syntax checks**

```bash
python3 -m py_compile \
  block_lo_arm_order_network/batch_readout/direct_order_provider.py \
  block_lo_arm_order_network/batch_readout/frozen_gbeta_hook.py \
  block_lo_arm_order_network/train_clean_aogpt.py
bash -n scripts/run_direct_policy_aligned_20260625.sh
```

**Step 3: Inspect the final diff**

Confirm no changes to `cdl_order_provider.py`, no physical fields in direct
score/provider code, and no unrelated files modified.

