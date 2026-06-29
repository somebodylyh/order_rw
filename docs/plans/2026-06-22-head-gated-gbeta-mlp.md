# Head-Gated gBeta MLP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade g_beta from a manually selected single-head distillation controller into a head-gated MLP that receives all heads from one layer, learns which heads contain order signal under CDL/manual-teacher supervision, and can be safely plugged back into AO-GPT training.

**Architecture:** Treat g_beta as the distilled order MLP. Phase 1 freezes AO-GPT and trains g_beta variants offline against existing CDL/manual labels. Phase 2 plugs the best frozen g_beta into AO-GPT training. Phase 3 only then allows joint/end-to-end training, with explicit model-frame head-drift diagnostics to detect whether the input heads change under feedback.

**Tech Stack:** Python, PyTorch, existing `block_lo_arm_order_network` training/checkpoint code, existing strict-65/model-frame diagnostics, TSV/JSON reports, matplotlib plots.

---

## Non-Negotiable Coordinate Convention

- Primary order coordinate is **model frame**.
- Existing clean convention:
  - `inv_perm[model_block] = physical_block`
  - `block_perm[physical_block] = model_block`
  - semantic model path is `block_perm[np.arange(N)]`
- Main diagnostic metrics:
  - `tau_model_vs_semantic_path`
  - `tau_model_vs_identity`
- Physical-frame metrics are sanity checks only. Do not use posthoc physical L2R as the primary claim.

---

## Experiment Priorities

1. **Offline distillation sanity:** Can g_beta learn the hand-read/CDL order signal from head features?
2. **Automatic head selection:** Does a head-gated g_beta beat single-head and mean-all-head baselines?
3. **Frozen feedback:** Does the distilled g_beta improve AO-GPT training when plugged in as a fixed order controller?
4. **Head drift check:** Does feedback training change the input head's readable order signal?
5. **End-to-end training:** Only attempt after phases 1-4 are stable.

## Plan Review Additions

This plan intentionally upgrades the existing audition -> CDL -> g_beta -> frozen-beta pipeline without changing the outer pipeline structure. The following risks must be handled during implementation:

- Do not rely only on `manual_l0h2_cd` as teacher. L0H2 is a useful canonical head, but head identity drifts across runs. Always compare against at least one per-head/CDL-derived teacher.
- Do not start sparse top-k gating cold. Begin with soft attention over all heads (`k=H`) plus entropy pressure, then test hard top-k after the gate shows stable preference.
- If full-gradient/end-to-end g_beta is run, repeat the head-drift diagnostic afterward. Full gradient can reward-hack the input heads into becoming easy for the gate rather than improving genuine order signal.
- Add a selected-head sanity check: the head selected by sparse gate should perform similarly as a single-head g_beta. If sparse gate greatly exceeds every selected single head, report it as nontrivial head composition, not simple head selection.

---

### Task 1: Inventory Existing gBeta and CDL Code Paths

**Files:**
- Read: `block_lo_arm_order_network/`
- Read: `scripts/`
- Read: `analyses/diag_model_frame_feedback.py`
- Create: `reports/head_gated_gbeta_inventory.md`

**Step 1: Locate g_beta model definitions**

Run:

```bash
rg -n "class .*Beta|g_beta|frozen_beta|beta" block_lo_arm_order_network scripts analyses
```

Expected: find current g_beta model, training entry points, and frozen-feedback integration.

**Step 2: Locate CDL/manual label generation**

Run:

```bash
rg -n "cdl|teacher|rollout|C-D|C\\+L|strict" block_lo_arm_order_network scripts analyses
```

Expected: identify where teacher orders/scores are produced and consumed.

**Step 3: Write inventory report**

Create `reports/head_gated_gbeta_inventory.md` with:

- current g_beta input shape
- current g_beta target/label format
- current frozen-beta training entry
- current CDL teacher training entry
- exact files to modify in later tasks

**Step 4: Commit**

```bash
git add reports/head_gated_gbeta_inventory.md
git commit -m "docs: inventory gbeta and cdl code paths"
```

---

### Task 2: Build a Reusable Model-Frame Distillation Dataset

**Files:**
- Create: `analyses/build_gbeta_headset_dataset.py`
- Output: `block_lo_arm_order_network/batch_readout/datasets/headset_l{layer}_{teacher}_{run}.npz`

**Step 1: Define dataset schema**

Each `.npz` must contain:

```text
B_heads: float32 [num_samples, num_heads, N, N+1]
teacher_order_model: int64 [num_samples, N]
teacher_pairwise: float32 [num_samples, N, N] optional
block_perm_phys_to_model: int64 [N]
inv_perm_model_to_phys: int64 [N]
layer: int
source_ckpt: str
teacher_method: str
coordinate_frame: "model"
```

**Step 2: Extract all heads from one layer**

Use the same extraction logic as `analyses/diag_model_frame_feedback.py` and `scripts/search_strict_label_free_65.py`.

Important: graph construction must stay in model frame. Do not apply `inv_perm` before teacher creation.

**Step 3: Generate teacher labels**

Support at least:

```text
teacher=manual_l0h2_cd
teacher=per_head_cdl_rollout
teacher=minus_d_only
teacher=full_cdl
teacher=semantic_model_path_oracle
```

Use `semantic_model_path_oracle` only as a diagnostic upper-bound label, not as a paper claim.

Teacher notes:

- `manual_l0h2_cd` tests whether the MLP can mimic the current canonical manually selected head.
- `per_head_cdl_rollout` avoids teaching the gate to imitate only L0H2; it lets the label come from the best per-head CDL readout under the current checkpoint.
- `minus_d_only` tests the load-bearing ablation where the directional/destroyed component carries the signal.
- `full_cdl` tests the operational teacher used by the current CDL pipeline.

**Step 4: Smoke test**

Run:

```bash
python3 analyses/build_gbeta_headset_dataset.py \
  --ckpt block_lo_arm_order_network/probe_results/random_baseline_continuous_jun08_seed2/ckpt_step20000.pt \
  --layer 0 \
  --teacher manual_l0h2_cd \
  --M 2 \
  --batch-size 2 \
  --out /tmp/headset_smoke.npz
```

Expected:

```text
saved /tmp/headset_smoke.npz
B_heads shape = [4, H, 64, 65]
coordinate_frame = model
```

**Step 5: Commit**

```bash
git add analyses/build_gbeta_headset_dataset.py
git commit -m "feat: build model-frame headset distillation dataset"
```

---

### Task 3: Implement gBeta Variants

**Files:**
- Modify or create model file found in Task 1, likely under `block_lo_arm_order_network/`
- Create: `tests/test_head_gated_gbeta.py` if tests exist; otherwise create `analyses/smoke_head_gated_gbeta.py`

**Step 1: Implement three variants behind one interface**

Required variants:

```text
single_head_gbeta
mean_head_gbeta
sparse_head_gated_gbeta
```

Interface:

```python
scores, aux = model(B_heads)
```

Expected shapes:

```text
B_heads: [batch, H, N, N+1]
scores:  [batch, N, N] or [batch, N]
aux["gate_weights"]: [batch, H]
```

**Step 2: Implement sparse gate**

Do not start with hard top-k. Implement two modes:

```text
gate_mode=soft_all
gate_mode=topk
```

`soft_all`:

```python
weights = softmax(logits / temperature)
B_mix = sum_h weights_h * B_h
```

`topk`:

```python
logits = gate_encoder(head_features)
mask = topk(logits, k)
weights = softmax(masked_logits)
B_mix = sum_h weights_h * B_h
```

Default:

```text
gate_mode = soft_all for smoke/warmup
k = H for soft_all
k = 2 for topk after warmup
entropy_penalty = 0.01
```

**Step 3: Add mean-head baseline**

This is intentionally expected to be weak:

```python
B_mix = B_heads.mean(dim=1)
```

It is needed because it tests the hypothesis that averaging destroys directionality.

**Step 4: Smoke test**

Run:

```bash
python3 analyses/smoke_head_gated_gbeta.py
```

Expected:

```text
single_head_gbeta ok
mean_head_gbeta ok
sparse_head_gated_gbeta ok
gate_weights sum to 1
```

**Step 5: Commit**

```bash
git add block_lo_arm_order_network tests analyses/smoke_head_gated_gbeta.py
git commit -m "feat: add head-gated gbeta variants"
```

---

### Task 4: Offline Distillation Training

**Files:**
- Create: `scripts/train_head_gated_gbeta.py`
- Create: `scripts/run_head_gated_gbeta_distill_smoke.sh`
- Create: `scripts/run_head_gated_gbeta_distill_full.sh`
- Output: `block_lo_arm_order_network/batch_readout/logs/head_gated_gbeta_*`

**Step 1: Implement training objective**

Use one primary objective:

```text
pairwise order loss over teacher_order_model
```

Optional secondary objective:

```text
next-block CE along teacher order
```

Log:

```text
train_loss
val_loss
tau_model_vs_teacher
tau_model_vs_semantic_path
tau_model_vs_identity
gate_entropy
top_heads
```

**Step 2: Train smoke**

Run:

```bash
bash scripts/run_head_gated_gbeta_distill_smoke.sh
```

Expected:

```text
single_head_gbeta reaches non-random tau on smoke
sparse_head_gated_gbeta runs without NaN
mean_head_gbeta included as baseline
```

**Step 3: Train full variants**

Run:

```bash
bash scripts/run_head_gated_gbeta_distill_full.sh
```

Variants:

```text
single_head L0H2
mean_head layer0
sparse_head_gated layer0 soft_all
sparse_head_gated layer0 topk=1
sparse_head_gated layer0 topk=2
sparse_head_gated layer0 topk=4
```

Teacher variants:

```text
manual_l0h2_cd
per_head_cdl_rollout
minus_d_only
full_cdl
```

Run the full grid only after smoke. First full pass can restrict to:

```text
manual_l0h2_cd x {single_head, mean_head, soft_all, topk=2}
full_cdl x {single_head, mean_head, soft_all, topk=2}
minus_d_only x {single_head, soft_all, topk=2}
```

**Step 4: Commit**

```bash
git add scripts/train_head_gated_gbeta.py scripts/run_head_gated_gbeta_distill_*.sh
git commit -m "feat: train head-gated gbeta distillation"
```

---

### Task 5: Offline Evaluation and Selection

**Files:**
- Create: `analyses/eval_head_gated_gbeta.py`
- Create: `analyses/plot_head_gated_gbeta.py`
- Output: `reports/head_gated_gbeta_eval/`

**Step 1: Evaluate all variants**

Run:

```bash
python3 analyses/eval_head_gated_gbeta.py \
  --runs block_lo_arm_order_network/batch_readout/logs/head_gated_gbeta_* \
  --out-dir reports/head_gated_gbeta_eval
```

**Step 2: Required table columns**

```text
run
variant
layer
topk
teacher
tau_model_vs_teacher
tau_model_vs_semantic_path
tau_model_vs_identity
gate_entropy
top1_head
top2_heads
```

**Step 3: Selection rule**

Pick the variant that satisfies:

```text
high tau_model_vs_teacher
high tau_model_vs_semantic_path
low tau_model_vs_identity unless identity is intended
low-to-moderate gate entropy
stable selected heads across validation batches
```

**Step 4: Selected-head sanity check**

For each sparse-gated run, identify the top selected head(s), then evaluate:

```text
single_head_gbeta on selected head 1
single_head_gbeta on selected head 2
sparse_head_gated_gbeta
```

Interpretation:

```text
sparse ~= selected single head: gate is doing clean head selection
sparse >> all selected single heads: gate is doing nontrivial head composition
sparse < selected single head: gate is hurting the signal
```

**Step 5: Commit**

```bash
git add analyses/eval_head_gated_gbeta.py analyses/plot_head_gated_gbeta.py reports/head_gated_gbeta_eval
git commit -m "eval: compare head-gated gbeta variants"
```

---

### Task 6: Small-Train Audition Stage

**Files:**
- Create: `scripts/run_head_gated_audition.sh`
- Create: `analyses/eval_head_gated_audition.py`
- Output: `block_lo_arm_order_network/probe_results/head_gated_audition_*`

**Purpose:** Bridge offline τ ranking and full frozen feedback. The existing head selection pipeline was **performance-auditioned** — a candidate head ran a small CDL or gβ feedback train (e.g., 20k→22k or 20k→25k), and short-train val/loss/curve determined whether it was worth a full 60k run. Offline τ alone is insufficient: a head can be **readable** (high τ) but not **useful** as a training controller (no small-train improvement).

**Step 1: Run small-train feedback for each candidate**

Candidates come from Task 5 selection:

```text
top head(s) from sparse gate
single-head L0H2 (canonical reference)
mean-head (expected-weak baseline)
random continuation (training drift baseline)
```

Small-train config:

```text
start from 20k ckpt
train 2k-5k steps (20k→22k or 20k→25k)
controller: frozen (read-only, no gradient into AO-GPT)
log every 100 steps
```

**Step 2: Required metrics**

```text
val_ori_l2r_block
train_obj
eval_curve stability (no sudden collapse)
tau_model_vs_semantic_path at end of small train
delta vs random continuation from same ckpt
```

**Step 3: Decision rule**

```text
offline τ high + small-train win         → strongest candidate → full 60k
offline τ high + small-train no-improve  → readable but not useful as controller
offline τ medium + small-train stable win → usable controller despite modest τ
offline τ low + small-train lose         → drop
```

This explains why pure τ-max head selection failed historically: some heads are readable but don't function as optimization controllers under feedback training.

**Step 4: Commit**

```bash
git add scripts/run_head_gated_audition.sh analyses/eval_head_gated_audition.py
git commit -m "exp: small-train audition gate for head-gated gbeta candidates"
```

---

### Task 7: Frozen Feedback Integration

**Files:**
- Modify existing frozen-beta training entry found in Task 1
- Create: `scripts/queue_head_gated_gbeta_feedback.sh`
- Output: `block_lo_arm_order_network/probe_results/head_gated_gbeta_feedback_*`

**Step 1: Add controller loader**

The training script must accept:

```text
--gbeta-controller path/to/g_beta_best.pt
--gbeta-input-mode layer_heads
--gbeta-layer 0
--gbeta-topk 2
--gbeta-frozen True
```

**Step 2: Run smoke from 20k**

Run:

```bash
bash scripts/queue_head_gated_gbeta_feedback.sh smoke
```

Expected:

```text
loads checkpoint
loads g_beta
generates model-frame order
runs 100-500 training steps
eval_curve.tsv contains ori_l2r/model_order/unstructured/controller metrics
```

**Step 3: Run full feedback**

Run:

```bash
bash scripts/queue_head_gated_gbeta_feedback.sh full
```

Required arms:

```text
random continuation from same ckpt
single-head frozen g_beta
sparse head-gated frozen g_beta
mean-head frozen g_beta
```

**Step 4: Commit**

```bash
git add scripts/queue_head_gated_gbeta_feedback.sh block_lo_arm_order_network
git commit -m "feat: plug head-gated gbeta into frozen feedback training"
```

---

### Task 8: Head Drift Diagnostic

**Files:**
- Extend: `analyses/diag_model_frame_feedback.py`
- Create: `reports/head_drift_head_gated_gbeta/`

**Step 1: Scan fixed checkpoints**

Use the same ladder for all arms:

```text
20k start
30k
40k
50k
60k
```

**Step 2: Required metrics**

For selected/tracked heads:

```text
tau_model_vs_semantic_path
tau_model_vs_identity
rank_model_semantic_path
sigma_model_first16
```

For g_beta gate:

```text
top1_head
top2_heads
gate_entropy
gate_weight_by_head
```

**Step 3: Interpret with difference-in-differences**

Compute:

```text
delta_random = metric_60k_random - metric_20k_start
delta_feedback = metric_60k_feedback - metric_20k_start
feedback_effect = delta_feedback - delta_random
```

This answers: did MLP/controller participation alter the input head beyond normal training drift?

**Step 4: Commit**

```bash
git add analyses/diag_model_frame_feedback.py reports/head_drift_head_gated_gbeta
git commit -m "eval: add model-frame head drift diagnostics"
```

---

### Task 9: Optional End-to-End gBeta Training

**Files:**
- Modify training entry found in Task 1
- Create: `scripts/queue_head_gated_gbeta_joint.sh`

**Step 1: Add joint mode**

Arguments:

```text
--gbeta-frozen False
--gbeta-lr 1e-5
--gbeta-entropy-penalty 0.01
--gbeta-gate-topk 2
--gbeta-stopgrad-head-features True/False
```

**Step 2: Start with stop-gradient**

First joint run should update g_beta but not route gradients from order loss into AO-GPT head features.

**Step 3: Only then test full gradient**

Full gradient is high risk because it may rewrite the input heads and invalidate the controller target.

After any full-gradient run, immediately rerun Task 8 (head drift). The main failure mode is reward hacking: the model may reshape input heads to be easy for g_beta to select rather than strengthening genuine order signal. This must be measured with model-frame head drift, not only final validation loss.

**Step 4: Required negative controls**

```text
mean-head joint
reverse teacher joint
identity-order joint
```

**Step 5: Commit**

```bash
git add scripts/queue_head_gated_gbeta_joint.sh block_lo_arm_order_network
git commit -m "exp: add optional joint head-gated gbeta training"
```

---

## Success Criteria

Minimum publishable technical success:

```text
sparse head-gated g_beta >= single-head g_beta on teacher tau
sparse head-gated g_beta > mean-head g_beta
gate selects a small stable subset of heads
frozen feedback improves or matches current g_beta/CDL curves
head drift is measured in model frame
```

Strong success:

```text
sparse head-gated g_beta removes manual head selection
frozen feedback remains stable to 60k
reverse/identity controls fail as expected
gate analysis identifies order-informative heads consistently
```

Do not claim:

```text
runtime content-dependent ordering
teacher beats pure L2R semantically
end-to-end discovery
```

unless separate relocation/content-dependence tests support it.

---

## Recommended First Night Run

Do only phases 1-5 first:

```text
inventory
dataset smoke
model smoke
offline distillation full
offline eval plot
```

Do not start end-to-end training until sparse-gated g_beta clearly beats mean-head and matches or beats single-head offline.
