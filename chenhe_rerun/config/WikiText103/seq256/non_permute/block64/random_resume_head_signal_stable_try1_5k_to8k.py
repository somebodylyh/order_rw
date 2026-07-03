# Continue the seq256/block64 non-permuted Random checkpoint for head-signal
# stability diagnostics. This config is intentionally plain Random training:
# the orientation rule is evaluated by external analysis scripts, not used as a
# training target.

out_dir = 'out/base/nonpermute/seq256/block64/out-wikitext103-seq256-random-b64-head-signal-stable-try1-resume5k-to8k'
eval_interval = 500
eval_iters = 100
log_interval = 50

wandb_log = False
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-random-b64-head-signal-stable-try1-resume5k-to8k'

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

init_from = 'resume'
resume_optimizer_state = True
learning_rate = 1e-3
max_iters = 8000
lr_decay_iters = 50000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 0

always_save_checkpoint = True
save_iter_checkpoints = True
save_iter_checkpoint_steps = '5500,6000,7000,8000'
save_iter_checkpoint_keep = 0

compile = False
eval_generate_step_loss_log = False
eval_kendall_distance_log = False
