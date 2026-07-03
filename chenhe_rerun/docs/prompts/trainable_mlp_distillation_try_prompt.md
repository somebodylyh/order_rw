# Agent Prompt: Create a Trainable-MLP Distillation Try on the Current Mainline

## Role

You are working inside the `language` branch of `Yangxiaohehehe/nanogpt-learned-order`. Your task is to understand the current WikiText103 `seq256/permute/block64` learned-order mainline, then create the next controlled experiment for **trainable MLP distillation inside the backbone run**.

The goal is not to redesign the whole method. Keep the method simple and controlled.

---

## Current mainline

The current active mainline is the MLP distillation line under:

```text
Report/language/wikitext103/mlp/distillation/tries_by_stage/05_latest_shadow_schedule_seed_sweep/
```

Treat `try56`-`try60` as the current baseline family. Do **not** treat the older frozen try28/try33 insertion line as the active method, except as historical context.

Current strongest retained baseline:

```text
try60
seed = 2053
policy input attention = 1024 samples
shadow MSE item = 512 samples
teacher label orientation = linear_profile_loss
schedule = 10k-18k shadow train, 18k-32k MLP-policy anneal to 0.8, 32k-35k fixed-cache anneal to 1.0, 35k-50k fixed
final val = 3.3245
best val = 3.3128
final diagnostic original tau = 0.9950
teacher-vs-MLP score pair tau = 0.9960
teacher-vs-MLP score MSE = 0.0169
```

The current baseline trains a random-init score MLP during the shadow phase, then freezes MLP optimizer updates after the shadow window while the cached MLP order is refreshed and used by the policy schedule.

The new experiment should make **one main change**: keep the MLP parameters open after the shadow phase and continue updating them by teacher-score MSE during the policy-refresh phase. This is still distillation; it is not policy gradient and it is not backpropagating language loss through the order.

---

## High-level objective

Create a new try that keeps the current no-EMA teacher/student framework, but changes the frozen-after-shadow MLP baseline into a **trainable online-distillation MLP**:

```text
0k–10k:   Random warmup
10k–18k:  Random training + shadow-train MLP with MSE only
18k–32k:  Random → MLP anneal to policy_prob=0.8, while continuing to train MLP with MSE only
32k–35k:  stop MLP updates and order refresh, anneal the fixed cached MLP order to policy_prob=1.0
35k–50k:  use the final cached MLP order deterministically
```

The new try should answer one question:

> Does continuing to train the MLP by online teacher-score MSE during the 18k-32k policy-refresh phase improve over the latest frozen-after-shadow MLP baseline?

Do **not** add pairwise loss, Rayleigh/Laplacian regularization, policy gradient, sampling, logits EMA, order EMA, or attention EMA in this first version.

---

## Current research assumptions

### Teacher method

The current teacher is the no-prior Fiedler distribution teacher:

```text
current-frame L0 layer-mean without_none attention A
→ W = max(A, A.T), diagonal zeroed
→ graph Laplacian Fiedler vector
→ raw/reverse direction chosen by current-model linear_profile_loss
→ oriented continuous teacher priority vector z_T
```

Important boundaries:

- Use only current-frame attention and current-model loss/profile signals.
- Do not use `OriginalL2R`, original-frame tau, or validation PPL to recover the order, orient the teacher, select checkpoints, or tune the policy.
- `OriginalL2R` and original-frame tau may be logged only as diagnostics after the current-frame order already exists.
- The safe claim is recoverable local block-level language order structure, not full global L2R induction.

### MLP method

The MLP is a student policy, not a mechanistic replacement for the spectral teacher.

The current mainline uses a random-init score MLP trained inside the run by MSE to the teacher score. Do **not** default to loading an old offline MLP checkpoint. Use random init unless the user explicitly asks for a warm-start ablation, and if warm-starting is used, record it as a separate experimental factor.

The student should output one score/logit per block:

```text
s = MLP(A)
larger score = earlier reveal
order = argsort(s, descending=True)
```

For this new try, the MLP training loss is MSE only:

```text
loss_mlp = MSE(zscore(MLP(A)), zscore(z_T))
```

where `z_T` is the oriented Fiedler teacher priority vector produced from the same current-frame attention matrix.

---

## First actions

1. Confirm you are on the correct branch and do not destroy local work:

```bash
git status
git branch --show-current
```

2. Read the current language entry points:

```text
README.md
base_cn.md
docs/README.md
docs/prompts/prompt_language_block_current_task.md
Report/language/wikitext103/MAINLINE_AND_HISTORY.md
Report/language/wikitext103/mlp/distillation/README.md
Report/language/wikitext103/order_teacher_distribution/head_signal_stability/current_distribution_methodology_summary.md
```

3. Locate and read the latest MLP distillation tries:

```text
Report/language/wikitext103/mlp/distillation/tries_by_stage/05_latest_shadow_schedule_seed_sweep/README.md
Report/language/wikitext103/mlp/distillation/tries_by_stage/05_latest_shadow_schedule_seed_sweep/try_56/
Report/language/wikitext103/mlp/distillation/tries_by_stage/05_latest_shadow_schedule_seed_sweep/try_57/
Report/language/wikitext103/mlp/distillation/tries_by_stage/05_latest_shadow_schedule_seed_sweep/try_58/
Report/language/wikitext103/mlp/distillation/tries_by_stage/05_latest_shadow_schedule_seed_sweep/try_59/
Report/language/wikitext103/mlp/distillation/tries_by_stage/05_latest_shadow_schedule_seed_sweep/try_60/
```

Also search for nearby local tries if they exist:

```bash
find -L Report/language/wikitext103/mlp/distillation -maxdepth 5 -type f | sort | grep -E 'try_(5[6-9]|6[0-9])'
find config/WikiText103/seq256/permute/block64 -type f | sort | grep -E 'try5[6-9]|try6[0-9]|mlp'
```

The root-level `Report/language/wikitext103/mlp/distillation/try_XX` paths may be symlinks for compatibility. Prefer the classified `tries_by_stage/` paths when writing new documentation.

4. Identify the matched baseline config and result, not a checkpoint to load by default.

Use `try60` as the first matched baseline unless a newer completed local try has clearly superseded it. Record:

```text
baseline try id
config path
out_dir
input/order diagnostic path
teacher definition
input family
loss type
best val metric
final val metric
seed
policy input attention sample count
shadow MSE item sample count
whether larger-logit-earlier / argsort_desc is used
```

---

## New try design

Create the next available try number after the current completed local tries. Since `try_60` is already the current best baseline, the next normal number is likely `try_61`, but verify the local tree before choosing.

Suggested name:

```text
try_XX_trainable_mlp_mse_online_noema_shadow512_policy1024_train10k32k_stop32k_prob08_fixed35k
```

### Schedule

```text
0–10k:
    AO-GPT trains with Random order.
    MLP is random initialized but not used as policy.
    No MLP policy probability.

10–18k:
    AO-GPT still trains with Random order.
    Shadow-train the MLP using MSE to current Fiedler teacher.
    policy_prob = 0.

18–32k:
    AO-GPT uses Random → MLP probability annealing from 0.0 to 0.8.
    Continue to train MLP with MSE to current Fiedler teacher.
    Refresh cached MLP order from the updated MLP on the same current attention family.

32–35k:
    Freeze MLP parameters.
    Stop collecting/updating attention for MLP.
    Stop refreshing MLP order.
    Anneal the fixed cached MLP order probability from 0.8 to 1.0.

35–50k:
    Use the final cached MLP order as fixed order.
```

### No EMA

Set all EMA-like mechanisms off:

```python
attn_mlp_policy_ema_decay = 0.0
attn_mlp_policy_attention_ema_enabled = False
attn_mlp_policy_logits_ema_enabled = False
```

The 512/1024-sample attention aggregation described below is **not EMA**. It is a within-update batch average.

---

## Attention aggregation for MLP training

The user’s current training setting is:

```text
batch_size = 64
gradient_accumulation_steps = 2
```

Therefore one optimizer step sees:

```text
64 × 2 = 128 sequence samples
```

The current mainline separates two aggregation scales:

```text
shadow MSE training item: 512 samples
policy/order-refresh input attention: 1024 samples in try60, 512 samples in try58/try59
```

For the first trainable-MLP follow-up, match `try60` unless the user explicitly asks for a policy-attn512 ablation:

```text
10k-18k shadow MSE item = 512 samples
18k-32k online MSE/order-refresh item = 1024 samples
```

Sample-count translation:

```text
512 samples = 8 micro-batches of size 64 = 4 optimizer steps
1024 samples = 16 micro-batches of size 64 = 8 optimizer steps
```

Preferred implementation:

```text
Reuse training-forward attentions.
Accumulate L0 layer-mean without_none attention matrices until the current phase target sample count is reached.
Average them into one A_train.
Use A_train for teacher construction and MLP MSE update.
After the optimizer step, update cached MLP order from the updated MLP on A_train.
Clear the accumulator.
```

Pseudocode:

```python
target_samples = 512 if iter_num < 18000 else 1024

A_sum += A_mb.float() * num_samples_in_mb
sample_count += num_samples_in_mb

if sample_count >= target_samples:
    A_train = A_sum / sample_count
    A_train.fill_diagonal_(0.0)

    with torch.no_grad():
        z_T = build_oriented_fiedler_teacher(
            A=A_train,
            orientation='linear_profile_loss',
        )

    scores = mlp(A_train.detach())
    loss_mlp = mse(zscore(scores), zscore(z_T))

    optimizer_mlp.zero_grad(set_to_none=True)
    loss_mlp.backward()
    clip_grad_norm_(mlp.parameters(), 1.0)
    optimizer_mlp.step()

    with torch.no_grad():
        cached_order = argsort(mlp(A_train.detach()), descending=True)

    A_sum.zero_()
    sample_count = 0
```

If it is difficult to reuse training attentions safely, use the fallback implementation:

```text
Every MLP update, run the required number of no-grad probe micro-batches.
Average their attentions into A_train.
Use A_train for teacher, MLP update, and cached order.
```

The preferred option is cheaper because it reuses forward-pass attentions.

---

## MLP loss

Use MSE only:

```python
def zscore(x, eps=1e-6):
    return (x - x.mean()) / x.std(unbiased=False).clamp_min(eps)

loss_mlp = F.mse_loss(
    zscore(mlp_scores),
    zscore(teacher_priority),
)
```

Do not add in this try:

```text
pairwise BCE
rank loss
Rayleigh/Laplacian loss
policy gradient
sampling / Gumbel-top-k
logits EMA
order EMA
attention EMA
```

If the code already has optional pairwise or Rayleigh paths, keep their weights at zero.

---

## Policy usage

During 10k–18k:

```text
policy_prob = 0
AO-GPT uses Random order
MLP trains in shadow only
```

During 18k–32k:

```text
policy_prob = linear(step, 18k, 32k, 0, 0.8)
MLP optimizer remains active
MLP is still trained by teacher-score MSE only
with probability policy_prob: use cached MLP order
otherwise: use Random order
```

During 32k–35k:

```text
policy_prob = linear(step, 32k, 35k, 0.8, 1.0)
MLP optimizer is stopped
cached MLP order is no longer refreshed
with probability policy_prob: use the final cached MLP order
otherwise: use Random order
```

During 35k–50k:

```text
use fixed_order = final cached MLP order
no MLP parameter update
no MLP order refresh
```

If there is no cached MLP order yet, fallback to Random.

---

## Suggested config values

Adapt names to the existing codebase, but preserve this behavior:

```python
# base
max_iters = 50000
lr_decay_iters = 50000
init_from = 'scratch'  # unless there is a matched 10k base checkpoint protocol
resume_optimizer_state = False

# MLP policy
attn_mlp_policy_enabled = True
attn_mlp_policy_random_init = True
attn_mlp_policy_path = ''
attn_mlp_policy_freeze = False
attn_mlp_policy_train_loss = 'teacher_score_mse'  # or current supported MSE-only name

attn_mlp_policy_layer = 0
attn_mlp_policy_head = -1
attn_mlp_policy_export_type = 'without_none'
attn_mlp_policy_feature_mode = 'attention'
attn_mlp_policy_input_channels = 1
attn_mlp_policy_input_normalization = 'zscore'
attn_mlp_policy_order_mode = 'argsort_desc'

# schedule
attn_mlp_policy_start_iter = 18000
attn_mlp_policy_start_prob = 0.0
attn_mlp_policy_end_prob = 1.0
attn_mlp_policy_anneal_start_iter = 18000
attn_mlp_policy_anneal_end_iter = 35000
attn_mlp_policy_prob_schedule = 'piecewise'
attn_mlp_policy_prob_points = '0:0.0,9999:0.0,10000:0.0,17999:0.0,18000:0.0,32000:0.8,35000:1.0,50000:1.0'
attn_mlp_policy_attention_batches_per_update = 8  # 1024 policy-input samples when batch_size=64 and grad_acc=2
attn_mlp_policy_update_stop_iter = 32000

# no EMA
attn_mlp_policy_ema_decay = 0.0
attn_mlp_policy_attention_ema_enabled = False
attn_mlp_policy_logits_ema_enabled = False

# MLP optimizer
attn_mlp_policy_lr = 2e-4
attn_mlp_policy_weight_decay = 0.01
attn_mlp_policy_beta1 = 0.9
attn_mlp_policy_beta2 = 0.99
attn_mlp_policy_grad_clip = 1.0
attn_mlp_policy_lr_anneal_enabled = False

# aggregation
attn_mlp_policy_shadow_mse_start_iter = 10000
attn_mlp_policy_shadow_mse_stop_iter = 32000
attn_mlp_policy_shadow_mse_batches_per_item = 4  # 512-sample shadow items before 18k
attn_mlp_policy_shadow_mse_train_items_per_update = 2
attn_mlp_policy_shadow_mse_val_items_per_update = 0
attn_mlp_policy_shadow_mse_train_samples_per_step = 0
attn_mlp_policy_shadow_mse_val_samples_per_step = 0
attn_mlp_policy_shadow_mse_label_orientation = 'linear_profile_loss'
attn_mlp_policy_shadow_mse_orientation_x_mode = 'last_step'
```

If the existing code cannot use different sample counts for 10k-18k shadow MSE and 18k-32k online policy-phase MSE, add the minimal new config keys needed to separate them cleanly. Document the new keys and keep the default behavior backward compatible for existing configs.

---

## W&B and logging

Enable W&B for the full AO-GPT run:

```python
wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-tryXX-seedYYYY-trainable-mlp-mse-noema-shadow512-policy1024-train10k32k-stop32k-prob08-fixed35k'
```

Do not hard-code any W&B API key. Use the environment or existing login.

Log at minimum:

```text
mlp/train_mse
mlp/online_mse_18k32k
mlp/teacher_tau          # if available
mlp/score_std
mlp/margin_p10           # if easy
mlp/margin_median        # if easy
policy/alpha
policy/update_count
policy/samples_per_matrix
policy/cached_order_tau_to_prev
lm/train_loss
lm/val_loss
lm/best_val_loss
```

Also log final fixed order and save order history if existing infrastructure supports it.

Optional but useful diagnostics:

```text
teacher_order_original_tau        # diagnostic only
mlp_order_original_tau            # diagnostic only
mlp_teacher_tau
prefix_overlap@8/@16
local_pair_agreement@1/@2/@4
attention_diag_mass_k1/k2/k4
attention_row_entropy
```

Do not use diagnostic original-frame quantities for training or selection.

---

## Required files to create

For the new try, create:

```text
config/WikiText103/seq256/permute/block64/attn_mlp_tryXX_seedYYYY_trainable_mlp_mse_noema_shadow512_policy1024_train10k32k_stop32k_prob08_fixed35k.py
Report/language/wikitext103/mlp/distillation/tries_by_stage/06_trainable_online_mse_refinement/try_XX/experiment_design.md
Report/language/wikitext103/mlp/distillation/tries_by_stage/06_trainable_online_mse_refinement/try_XX/run_main_tryXX_seedYYYY_cudaN.sh
```

If the report tree still uses root compatibility links, create or update `Report/language/wikitext103/mlp/distillation/try_XX` as a symlink to the classified `tries_by_stage/06_trainable_online_mse_refinement/try_XX` directory. If running two seeds, create one config/run script per seed and a shared try directory.

Update indexes after creating the new try:

```text
Report/language/wikitext103/mlp/distillation/README.md
Report/language/wikitext103/MAINLINE_AND_HISTORY.md
docs/findings/findings_language.md
RUN_COMMANDS.md
docs/prompts/prompt_language_block_current_task.md
```

Do not overwrite `try56`-`try60` reports, configs, logs, or checkpoints.

---

## Acceptance criteria

The new try is ready to run when:

1. It starts from the current mainline protocol: random-init MLP, no offline checkpoint load, unless the user explicitly requested a warm-start ablation.
2. It uses current-frame L0 layer-mean `without_none` attention.
3. It trains the MLP with MSE only from 10k to 32k: 10k-18k shadow MSE plus 18k-32k online policy-phase MSE.
4. It uses no EMA of attention, logits, order, rank, or priority.
5. Each shadow MSE item is averaged from 512 samples, and the 18k-32k policy/order-refresh input matches the chosen baseline aggregation, defaulting to try60-style 1024 samples.
6. It anneals Random → MLP from 18k to 32k with MLP probability 0.0→0.8, then fixed cached MLP order from 32k to 35k with probability 0.8→1.0.
7. It freezes the final cached MLP order after 35k and trains fixed-order until 50k.
8. W&B is enabled without hard-coded credentials.
9. The experiment design records all config paths, output dirs, checkpoint paths if any are produced or resumed, seeds, aggregation counts, update windows, and interpretation boundaries.

---

## Interpretation after the run

Compare the new trainable-MLP run against the latest matched frozen-after-shadow MLP baseline, normally `try60` unless a newer completed local baseline supersedes it.

Primary downstream metric:

```text
best validation loss / final validation loss
```

Primary student-policy metric:

```text
MLP-teacher MSE and MLP-teacher order agreement
```

If the trainable-MLP run improves validation loss while preserving MLP-teacher agreement, interpret it as:

> Online MSE refinement helps the student adapt to the attention distribution induced during annealing.

If it does not improve, interpret it as:

> The latest frozen-after-shadow MLP is already sufficient; online MSE updates are unnecessary or introduce instability.

Do not claim that the MLP is mechanistically equivalent to the Fiedler teacher. It is a distilled policy.
