"""WikiText-103 seq256 L2R reference (run_kind=l2r).

Trained with L2R block order at every step. Used ONLY as upper-bound reference
for the Graph-RW method, not as a method run.

Two flavors:
  (a) From scratch       — clean_ori_l2r_from_scratch (best ori_l2r ≈ 3.467)
  (b) Continuation oracle — stageA_l2r_from30k       (best ori_l2r ≈ 3.390)

This config is set up for (a). To replicate (b), set resume_ckpt to a previous
L2R checkpoint and set max_iters accordingly.

NOTE: alpha_for_step in train_clean_aogpt.py returns 0 unconditionally when
run_kind=l2r — so any rw_* fields are inert. They are omitted here for clarity.
"""

# ---------------------- I/O ----------------------
out_dir = "out/reference/l2r/seq256/block64/wikitext103_l2r_from_scratch"
eval_interval = 250
eval_iters = 200
log_interval = 10

# ---------------------- W&B ----------------------
wandb_log = True
wandb_project = "AOGPT-order-block-64-final"
wandb_run_name = "seq256-l2r-from-scratch-60k"

# ---------------------- Data ----------------------
dataset = "wikitext103"
batch_size = 64
block_size = 256
gradient_accumulation_steps = 2
permute_data = False
permute_seed = 42

# ---------------------- Model (must match baseline) ----------------------
model_type = "aogpt"
train_stage = "standard"
aogpt_train_mode = "Random"
main_eval_mode = "Random"
generalization_eval_mode = ""
n_layer = 4
n_head = 8
n_embd = 384
dropout = 0
block_order_block_len = 4

# ---------------------- Optim (must match baseline) ----------------------
learning_rate = 1e-3
max_iters = 60000
lr_decay_iters = 60000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 0

# ---------------------- Run kind ----------------------
# L2R reference. alpha_for_step returns 0 in train_clean_aogpt.py for this kind.
run_kind = "l2r"
graph_rw_active = False
actual_alpha_used = 0.0
