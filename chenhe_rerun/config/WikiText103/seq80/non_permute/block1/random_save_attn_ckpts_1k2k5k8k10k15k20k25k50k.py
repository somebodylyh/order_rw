# WikiText103 seq80 non-permuted RANDOM config for attention diagnostics.
# Same training setup as random.py, with milestone checkpoints kept for attn plots.

out_dir = 'out/base/nonpermute/seq80/block1/out-wikitext103-seq80-random-b1-nonpermute-save-attn-ckpts-50000-iters'
eval_interval = 250
eval_iters = 200
log_interval = 10

wandb_log = True
wandb_project = 'AOGPT-order-block-seq80-final'
wandb_run_name = 'seq80-random-b1-nonpermute-save-attn-ckpts-50000-iters'

dataset = 'wikitext103'
data_record_mode = 'stream'
batch_size = 256
block_size = 80
gradient_accumulation_steps = 2
permute_data = False
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
block_order_layout = 'contiguous'

learning_rate = 1e-3
max_iters = 50000
lr_decay_iters = 50000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 0

save_iter_checkpoints = True
save_iter_checkpoint_steps = '1000,2000,5000,8000,10000,15000,20000,25000,50000'
save_iter_checkpoint_keep = 0
