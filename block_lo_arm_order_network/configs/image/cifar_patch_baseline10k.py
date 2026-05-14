"""CIFAR-10 image-patch baseline (10 000 steps, random patch order).

Trains image_order/model_image_aogpt.py from scratch on CIFAR-10 patches with
purely random reveal orders. Produces the source checkpoint for Stage 2 / CI /
long-run experiments.

Aligned with image_order/train_image_random.py defaults; only overrides batch_size
and step counts to match the published baseline10k checkpoint.
"""

# ---------------------- I/O ----------------------
output_dir = "probe_results_image/baseline/baseline10k"

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
device = "cuda"  # falls back to cpu in the script if cuda unavailable

# ---------------------- Model (defaults from ImageAOGPTConfig) ----------------------
# n_patches=64, patch_dim=48, n_embd=256, n_layer=4, n_head=8, cond_dim=128
# bias=True, dropout=0.0
# Override via train_image_random.py CLI if you want to vary these.

# ---------------------- Run kind ----------------------
# Image baseline does NOT use Graph-RW (no continuation, no alpha-mixed loss).
graph_rw_active = False
actual_alpha_used = 0.0
