"""Smoke config for ImageNet-64 patch2x2 baseline training (500 steps)."""

# ---------------------- I/O ----------------------
out_dir = "out/smoke/imagenet64_vq_f4_800k_patch2x2"
eval_interval = 100
eval_iters = 10
log_interval = 10

# ---------------------- W&B ----------------------
wandb_log = False
wandb_project = ""
wandb_run_name = ""
wandb_run_id = ""

# ---------------------- Data ----------------------
dataset = "Imagenet64VQ_f4_800k_patch2x2"
data_record_mode = "fixed"

permute_data = True
permute_seed = 42
permute_mode = "block"

batch_size = 16
gradient_accumulation_steps = 4   # effective batch = 64
block_size = 256

# ---------------------- Model ----------------------
init_from = "scratch"
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0
bias = False

aogpt_train_mode = "Random"
main_eval_mode = "Random"
generalization_eval_mode = ""
order_impl = "block"
block_order_block_len = 4          # 2×2 patch = 4 tokens

# ---------------------- Optim ----------------------
learning_rate = 1e-3
max_iters = 500
lr_decay_iters = 500
min_lr = 1e-4
beta2 = 0.99
weight_decay = 1e-1
warmup_iters = 10
grad_clip = 1.0
decay_lr = True

# ---------------------- Eval ----------------------
eval_generate_step_loss_log = False
eval_generate_step_batches = 0
eval_generate_step_loss_filename = ""
eval_kendall_distance_log = False
eval_kendall_num_orders = 0

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
