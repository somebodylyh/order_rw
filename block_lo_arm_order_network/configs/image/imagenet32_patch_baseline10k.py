"""E1: ImageNet32 continuous patch baseline (10 000 steps, random patch order).

Mirrors `cifar_patch_baseline10k.py` exactly — only the dataset changes. ImageNet32
is the bilinear downsample of the existing ImageNet64 800k subset
(/tmp/imagenet64_800k/), produced by `image_order/data/imagenet32/build_imagenet32.py`.
No new dataset source is introduced.

Model: ImageAOGPTConfig defaults (n_patches=64, patch_dim=48, n_embd=256, n_layer=4,
n_head=8, cond_dim=128, bias=True, dropout=0.0) — identical to E0 toy.

Purpose: isolate "dataset complexity" as the single changed axis between E0 (CIFAR
continuous patch + MSE + l4h8e256) and E1.

Launch (NOT executed by this config file — this is documentation):

    # One-time data prep (if not already built):
    python image_order/data/imagenet32/build_imagenet32.py

    # Smoke (500 steps):
    python -u image_order/train_image_random_imagenet32.py \
        --output-dir probe_results_image/baseline_imagenet32/smoke500 \
        --max-steps 500 --batch-size 64 --eval-interval 100 \
        --warmup-iters 50 --device cuda

    # Full 10k (matches cifar_patch_baseline10k.py):
    python -u image_order/train_image_random_imagenet32.py \
        --output-dir probe_results_image/baseline_imagenet32/baseline10k \
        --max-steps 10000 --batch-size 64 --lr 3e-4 --min-lr 3e-5 \
        --weight-decay 0.1 --beta1 0.9 --beta2 0.95 --grad-clip 1.0 \
        --warmup-iters 200 --log-interval 200 --eval-interval 500 \
        --max-eval-batches 16 --val-images 1000 --seed 42 --device cuda
"""

# ---------------------- I/O ----------------------
output_dir = "probe_results_image/baseline_imagenet32/baseline10k"

# ---------------------- Train ----------------------
max_steps = 10000
batch_size = 64
lr = 3e-4
min_lr = 3e-5
warmup_iters = 200
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# ---------------------- Eval ----------------------
log_interval = 200
eval_interval = 500
max_eval_batches = 16
val_images = 1000

# ---------------------- Misc ----------------------
seed = 42
device = "cuda"

# ---------------------- Model (defaults from ImageAOGPTConfig — match toy E0) ----------------------
# n_patches=64, patch_dim=48, n_embd=256, n_layer=4, n_head=8, cond_dim=128
# bias=True, dropout=0.0

# ---------------------- Run kind ----------------------
graph_rw_active = False
actual_alpha_used = 0.0

# ---------------------- Alignment metadata ----------------------
experiment_id = "E1"
dataset = "imagenet32_bilinear_from_imagenet64"
data_prep_script = "image_order/data/imagenet32/build_imagenet32.py"
data_path_train = "image_order/data/imagenet32/train.npy"
data_path_val = "image_order/data/imagenet32/val.npy"
train_script = "image_order/train_image_random_imagenet32.py"
aligns_with = "configs/image/cifar_patch_baseline10k.py (E0)"
single_changed_axis = "dataset only (CIFAR-10 → ImageNet32)"
