"""Partner-aligned ImageNet-64 VQ-f4 800k random-order baseline.

Mirrors ych/nanogpt-learned-order's training pipeline (uses ych's train.py +
AOGPT_block.py + configurator.py exec mechanism). The only thing this repo
contributes for the baseline is the tokenized data under
`data/Imagenet64VQ_f4_800k/` (see ../data/Imagenet64VQ_f4_800k/prepare.py).

Architecture parameters match ych's existing ImageNet-32 ckpt's `model_args`:
    n_layer=8, n_head=8, n_embd=512, vocab_size=8192, block_order_block_len=1,
    order_impl='block', position_encoding=absolute (1D wpe).

Data parameters match the ImageNet-64 setup:
    block_size = 256        (16x16 = 256 VQ tokens per image)
    permute_data = True     (matches ych's `permute_mode='block'` data shuffling)

Run inside ych's repo:
    cp data/Imagenet64VQ_f4_800k <ych_repo>/data/
    cd <ych_repo>
    python train.py /path/to/configs/image_partner/imagenet64_l8h8e512_random_base.py

The decorative fields below (`image_size`, `block_order_layout`, etc.) match
the ImageNet-32 ckpt's metadata so logging is consistent — but they are NOT
consumed by ych's current train.py / AOGPT_block.py code. Position encoding
is the standard 1D wpe(block_size+1, n_embd).
"""

# ---------------------- I/O ----------------------
out_dir = "out/base/permute/imagenet64_vq_f4_800k/seq256/block256/out-imagenet64-vqf4-800k-seq256-random-b256-permute-50000-iters"
eval_interval = 250
eval_iters = 200
log_interval = 10

# ---------------------- W&B ----------------------
wandb_log = True
wandb_project = "AOGPT-order-image-vq-f4"
wandb_run_name = "imagenet64-vqf4-800k-seq256-random-b256-permute-50000-iters"
wandb_run_id = ""

# ---------------------- Data ----------------------
dataset = "Imagenet64VQ_f4_800k"
data_record_mode = "fixed"

permute_data = True
permute_seed = 42
permute_mode = "block"

batch_size = 256
gradient_accumulation_steps = 1
block_size = 256                 # 16x16 VQ token grid per image

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
block_order_block_len = 1        # token-level

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

# ---------------------- Eval (matches ych ImageNet-32 ckpt config) ----------------------
eval_generate_step_loss_log = True
eval_generate_step_batches = 200
eval_generate_step_loss_filename = "generate_step_block_loss_latest.png"
eval_kendall_distance_log = True
eval_kendall_num_orders = 100

# ---------------------- Train stage ----------------------
train_stage = "standard"

# ---------------------- Decorative metadata (consumed only by wandb logging) ----------------------
image_size = 16                   # VQ token-grid side; the raw image is 64x64
image_block_size = 1
block_order_layout = "image_2d"
position_encoding_mode = "absolute"
rope_theta = 10000.0

# ---------------------- Segment-curriculum (off — ImageNet-64 baseline is pure random) ----------------------
segment_guided_ratio = 0.0
segment_source_json = ""
segment_top_k_pairs = 64
segment_max_len = 4
segment_max_units_per_order = 2
segment_use_all_units = False
