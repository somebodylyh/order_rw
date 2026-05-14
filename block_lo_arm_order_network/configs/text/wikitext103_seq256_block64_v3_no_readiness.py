"""WikiText-103 seq256 v3 ablation with rho=0 (no readiness term).

Identical to wikitext103_seq256_block64_v3_readiness.py except rw_rho=0.0. This is
the canonical ablation showing that the readiness term is responsible for closing
the gap to v2/v3-readiness (rho=0 stalls around 3.484 @40k while rho=0.2 reaches
3.458 @50k).
"""

# ---------------------- I/O ----------------------
out_dir = "out/method/v3_no_readiness/seq256/block64/wikitext103_v3_no_readiness_from20k"
eval_interval = 250
eval_iters = 200
log_interval = 10

# ---------------------- W&B ----------------------
wandb_log = True
wandb_project = "AOGPT-order-block-64-final"
wandb_run_name = "seq256-v3-no-readiness-from20k"

# ---------------------- Data ----------------------
dataset = "wikitext103"
batch_size = 64
block_size = 256
gradient_accumulation_steps = 2
permute_data = False
permute_seed = 42

# ---------------------- Model ----------------------
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

# ---------------------- Optim ----------------------
learning_rate = 1e-3
max_iters = 60000
lr_decay_iters = 60000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 0

# ---------------------- Graph-RW v3 ablation ----------------------
run_kind = "graph_rw"
graph_rw_active = True

resume_ckpt = "probe_results/clean_base_random_perm/ckpt_step20000.pt"

graph_policy = "progressive_rw_v3"
rw_top_k = 4
rw_epsilon_uniform = 0.0
rw_tau_start = 0.10
rw_tau_step = 0.10

rw_lambda = 0.75
rw_rho = 0.0          # <-- only difference vs the readiness config
rw_alpha_dep = 0.5

alpha_start = 0.0
alpha_target = 0.9
alpha_warmup_steps = 10000

refresh_every = 1500
ema_beta = 0.9
