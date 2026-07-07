# AMOR VQ-image deployment — random-order baseline.
# Same backbone/data/LR as the gβ arms; only the order policy differs (Random).

out_dir = '../out/rerun_vq/deploy_random'
eval_interval = 500
eval_iters = 100
log_interval = 50
wandb_log = False

dataset = 'imagenet64vq_patch2x2'
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

learning_rate = 1e-4
max_iters = 20000
lr_decay_iters = 20000
min_lr = 1e-5
beta2 = 0.99
warmup_iters = 200

init_from = 'ckpt'
init_from_ckpt = '../out/rerun_vq/vq_backbone_compat.pt'
init_from_ckpt_mode = 'weights_only'
