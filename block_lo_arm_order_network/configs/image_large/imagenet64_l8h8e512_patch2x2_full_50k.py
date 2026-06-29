"""ImageNet-64 VQ-f4 full 800k patch2x2 baseline — resume from 20k to 50k.

Continues patch2x2_full_baseline from ckpt.pt (20k) to 50k.
Saves checkpoints at 30k, 40k, 50k for frozen diagnostic.
"""

# ---------------------- I/O ----------------------
out_dir = "out/image_large/imagenet64_vq_f4_800k/seq256/patch2x2_full_baseline"
eval_interval = 250
eval_iters = 200
log_interval = 10
always_save_checkpoint = False
save_iter_checkpoints = True
save_iter_checkpoint_steps = "30000,40000,50000"
save_iter_checkpoint_keep = 0

# ---------------------- W&B ----------------------
wandb_log = True
wandb_project = "AOGPT-order-image64-vq-f4"
wandb_run_name = "imagenet64-vqf4-800k-seq256-patch2x2-full-b16-l8h8e512-50k"
wandb_run_id = ""

# ---------------------- Data ----------------------
dataset = "Imagenet64VQ_f4_800k_full_patch8x8"
data_record_mode = "fixed"

permute_data = True
permute_seed = 42
permute_mode = "block"

batch_size = 16
gradient_accumulation_steps = 16
block_size = 256

# ---------------------- Model ----------------------
init_from = "resume"
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0
bias = False

aogpt_train_mode = "Random"
main_eval_mode = "Random"
generalization_eval_mode = ""
order_impl = "block"
block_order_block_len = 4

# ---------------------- Optim ----------------------
learning_rate = 1e-3
max_iters = 50000
lr_decay_iters = 50000
min_lr = 1e-4
beta2 = 0.99
weight_decay = 1e-1
warmup_iters = 0
grad_clip = 1.0
decay_lr = True

# ---------------------- Eval ----------------------
eval_generate_step_loss_log = True
eval_generate_step_batches = 200
eval_generate_step_loss_filename = "generate_step_block_loss_latest.png"
eval_kendall_distance_log = True
eval_kendall_num_orders = 100

# ---------------------- Train stage ----------------------
train_stage = "standard"

# ---------------------- Decorative metadata ----------------------
image_size = 8
image_block_size = 2
block_order_layout = "image_2d"
position_encoding_mode = "absolute"
rope_theta = 10000.0

# ---------------------- Segment-curriculum (off) ----------------------
segment_guided_ratio = 0.0
segment_source_json = ""
segment_top_k_pairs = 64
segment_max_len = 4
segment_max_units_per_order = 2
segment_use_all_units = False
