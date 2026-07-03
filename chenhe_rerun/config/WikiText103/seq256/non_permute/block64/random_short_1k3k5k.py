# WikiText103 seq256 non-permuted RANDOM config for short checkpoint diagnostics.
# Matches block64 random baseline, but trains only to 5k and saves 1k/3k/5k.

out_dir = 'out/base/nonpermute/seq256/block64/out-wikitext103-seq256-random-b64-nonpermute-short-1k3k5k'
eval_interval = 250
eval_iters = 200
log_interval = 10

wandb_log = False
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-random-b64-nonpermute-short-1k3k5k'

dataset = 'wikitext103'
batch_size = 64
block_size = 256
gradient_accumulation_steps = 2
permute_data = False
permute_seed = 42

model_type = 'aogpt'
train_stage = 'standard'
aogpt_train_mode = 'Random'
main_eval_mode = 'Random'
generalization_eval_mode = ''
n_layer = 4
n_head = 8
n_embd = 384
dropout = 0

block_order_block_len = 4

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 0

save_iter_checkpoints = True
save_iter_checkpoint_steps = '1000,3000,5000'
save_iter_checkpoint_keep = 0
