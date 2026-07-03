# WikiText103 seq384 permuted RANDOM config for the 96-block setup (block_len=4).

out_dir = 'out/base/permute/seq384/block96/out-wikitext103-seq384-random-b96-permute-block-70000-iters'
eval_interval = 250
eval_iters = 200
eval_batch_size = 16
log_interval = 10

wandb_log = True
wandb_project = 'AOGPT-order-block-96-final'
wandb_run_name = 'seq384-random-b96-permute-70000-iters'

dataset = 'wikitext103'
batch_size = 32
block_size = 384
gradient_accumulation_steps = 2
permute_data = True
permute_seed = 42
permute_mode = 'block'

model_type = 'aogpt'
train_stage = 'standard'
aogpt_train_mode = 'Random'
main_eval_mode = 'Random'
generalization_eval_mode = ''
order_impl = 'block'
n_layer = 4
n_head = 8
n_embd = 384
dropout = 0
compile = False

block_order_block_len = 4

learning_rate = 1e-3
max_iters = 70000
lr_decay_iters = 70000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 0
