# Based on chenhe baseline config: config/WikiText103/seq256/permute/block64/random.py
# Only method / order-policy (+ continuation + bookkeeping) fields changed.
#
# V3 frozen-gβ order policy: resume the backbone from the shared random parent P
# and continue to 50k with the frozen CDL-pretrained gβ selecting the reveal order.
#
# PRODUCTION NOTE: produce the parent with `--max_iters=10000` but KEEP
# `lr_decay_iters=50000` (do NOT shrink it) so the lr schedule is continuous across
# the parent->continuation boundary. CDL-pretrain gβ from that SAME parent.

out_dir = 'out/rerun/gbeta_frozen_warmup'
eval_interval = 250
eval_iters = 200
log_interval = 10

wandb_log = True
wandb_project = 'order-rerun-block64'
wandb_run_name = 'seq256-gbeta-frozen-b64-permute-from10k'

dataset = 'wikitext103'
batch_size = 64
block_size = 256
gradient_accumulation_steps = 2
permute_data = True
permute_seed = 42
permute_mode = 'block'

model_type = 'aogpt'
train_stage = 'standard'
aogpt_train_mode = 'GBetaFrozenOrder'
main_eval_mode = 'AR'
generalization_eval_mode = ''
n_layer = 4
n_head = 8
n_embd = 384
dropout = 0

block_order_block_len = 4

learning_rate = 1e-3
max_iters = 50000
lr_decay_iters = 50000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 0

# ── continuation from the shared random parent P ──
init_from = 'ckpt'
init_from_ckpt = 'out/rerun/parent_random_10k/ckpt.pt'
init_from_ckpt_mode = 'full_state'

# ── frozen gβ order policy ──
gbeta_ckpt = 'out/rerun/gbeta_from_parent10k/g_beta_best.pt'
gbeta_batch_mean_probes = 4
gbeta_refresh_every = 1
gbeta_probe_mode = 'eval'
