# WikiText103 seq128 permuted RANDOM config for the token-level setup (block_len=1).

out_dir = 'out/base/permute/seq128/block1/out-wikitext103-seq128-random-b1-permute'
eval_interval = 250
eval_iters = 200
log_interval = 10

wandb_log = True
wandb_project = 'AOGPT-order-block-seq128'
wandb_run_name = 'seq128-random-b1-permute-base'

dataset = 'wikitext103'
batch_size = 128
block_size = 128
gradient_accumulation_steps = 2
permute_data = True
permute_seed = 42
permute_mode = 'block'

model_type = 'aogpt'
train_stage = 'standard'
aogpt_train_mode = 'Random'
main_eval_mode = 'Random'
generalization_eval_mode = ''
order_impl = 'token'
n_layer = 4
n_head = 8
n_embd = 384
dropout = 0

block_order_block_len = 1

learning_rate = 1e-3
max_iters = 40000
lr_decay_iters = 40000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 0
