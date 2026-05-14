"""WikiText-103 seq256 v3 readiness-guided Graph-RW continuation.

Continuation from the random-perm baseline checkpoint at step 20000.

Algorithm (`progressive_rw_v3`, see `directed_graph_policy.py`):
    score_t(v) = C_t(v) - lambda * D_t(v) + rho * r(v)
where r(v) is the readiness term derived from out-degree minus alpha_dep * in-degree
on B = A_global^T.

This config is the canonical text method run. It reproduces the strongest text-side
ori_l2r number (3.456-3.458 @50k vs random baseline 3.520).
"""

# ---------------------- I/O ----------------------
out_dir = "out/method/v3_readiness/seq256/block64/wikitext103_v3_from20k"
eval_interval = 250
eval_iters = 200
log_interval = 10

# ---------------------- W&B ----------------------
wandb_log = True
wandb_project = "AOGPT-order-block-64-final"
wandb_run_name = "seq256-v3-readiness-from20k-60k"

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

# ---------------------- Graph-RW v3 (these ARE consumed) ----------------------
run_kind = "graph_rw"
graph_rw_active = True

# Start checkpoint: collaborator's 50k baseline OR our internal clean_base@20k.
# Update this path when the collaborator's baseline lands.
resume_ckpt = "probe_results/clean_base_random_perm/ckpt_step20000.pt"

# Policy
graph_policy = "progressive_rw_v3"
rw_top_k = 4
rw_epsilon_uniform = 0.0
rw_tau_start = 0.10
rw_tau_step = 0.10

# Readiness term
rw_lambda = 0.75
rw_rho = 0.2
rw_alpha_dep = 0.5

# Mixing schedule
alpha_start = 0.0
alpha_target = 0.9
alpha_warmup_steps = 10000

# Refresh schedule (use existing default behavior; set explicitly here)
refresh_every = 1500   # set to 0 to disable; matches existing v3 run
ema_beta = 0.9
