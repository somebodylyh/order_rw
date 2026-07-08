# SCALE Stage-2a: 12L/768/8h random-order AOGPT warmup on OWT, to 10k (ckpt.pt = warmup).
# head-select + gβ read this early/generic backbone. n_head=8 (assert_layout HEADS=8).
out_dir = 'out/rerun_owt/scale12L768_baseline'
eval_interval = 1000
eval_iters = 100
log_interval = 50
always_save_checkpoint = True

wandb_log = True
wandb_project = 'amor-order'
wandb_run_name = 'scale12L768_baseline'

dataset = 'openwebtext'
data_record_mode = 'stream'
batch_size = 64
block_size = 256
gradient_accumulation_steps = 2
permute_data = True
permute_seed = 42
permute_mode = 'block'

model_type = 'aogpt'
train_stage = 'standard'
aogpt_train_mode = 'Random'
main_eval_mode = 'AR'
generalization_eval_mode = ''

block_order_block_len = 4

learning_rate = 6e-4
warmup_iters = 0
max_iters = 30000
lr_decay_iters = 30000   # anchor decay to the full run length (not 5k) so the warmup snapshot isn't over-annealed
min_lr = 6e-5
beta1 = 0.9
beta2 = 0.95
weight_decay = 0.1
grad_clip = 1.0

init_from = 'ckpt'
init_from_ckpt = 'out/rerun_owt/scale12L768_warmup5k/ckpt.pt'
init_from_ckpt_mode = 'weights_only'
