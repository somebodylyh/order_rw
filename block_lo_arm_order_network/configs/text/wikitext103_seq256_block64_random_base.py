"""WikiText-103 seq256 non-permuted RANDOM-order baseline (collaborator's reference config).

This config trains the AOGPT random-permutation baseline (run_kind=baseline). It does
NOT do Graph-RW. The `rw_*` fields below are explicitly NOT consumed at training time
(see train_clean_aogpt.py:alpha_for_step which short-circuits to alpha=0 for run_kind
not in {graph_rw, graph_rw_bag}).

Aligned to the collaborator's WikiText-103 seq256 block64 50k-iter random baseline.
"""

# ---------------------- I/O ----------------------
out_dir = "out/base/nonpermute/seq256/block64/out-wikitext103-seq256-random-b64-50000-iters"
eval_interval = 250
eval_iters = 200
log_interval = 10

# ---------------------- W&B ----------------------
wandb_log = True
wandb_project = "AOGPT-order-block-64-final"
wandb_run_name = "seq256-random-b64-nonpermute-50000-iters"

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
max_iters = 50000
lr_decay_iters = 50000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 0

# ---------------------- Run kind ----------------------
# IMPORTANT: this is a baseline (random-perm) run. The next line drives
# train_clean_aogpt.py: alpha_for_step returns 0.0 unconditionally when
# run_kind == "baseline", so any rw_* fields left around (none below) would be
# inert. We omit them here to keep the config honest.
run_kind = "baseline"
graph_rw_active = False
actual_alpha_used = 0.0
