"""CIFAR-10 image-patch Stage-2 Graph-RW continuation (5 000 steps from baseline10k).

α-mixed continuation training using Graph-RW low-noise (top_k=4, ε=0) reveal orders
on the image patch AOGPT. From baseline10k checkpoint with B = A_global^T extracted
from that checkpoint.

Reproduces the cont_top4 row of the published Stage-2 / 3-seed CI table:
    val_rw_top4 paired Δ = −0.00150 (95% CI [−0.00156, −0.00144], 3-seed CI)
    val_raster paired Δ  = −0.00176

For other configs (cont_random / cont_eps015 / cont_top8) override `alpha`,
`rw_top_k`, `rw_epsilon` from the CLI.
"""

# ---------------------- I/O ----------------------
output_dir = "probe_results_image/graph_rw/cont_top4"
baseline_ckpt = "probe_results_image/baseline/baseline10k/ckpt_step10000.pt"
b_path = "probe_results_image/attention/baseline10k/B_global.npy"

# ---------------------- Train ----------------------
max_steps = 5000
batch_size = 64
lr = 3e-5
min_lr = 3e-6
warmup_iters = 100
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# ---------------------- α-mixed loss ----------------------
alpha = 0.9         # constant; raise alpha_warmup if you want a ramp
alpha_warmup = 0    # 0 disables the warmup → constant α throughout

# ---------------------- Graph-RW policy ----------------------
rw_top_k = 4
rw_epsilon = 0.0
rw_tau_start = 0.10
rw_tau_step = 0.10

# ---------------------- Eval ----------------------
log_interval = 100
eval_interval = 1000
max_eval_batches = 16
val_images = 1000

# ---------------------- Misc ----------------------
seed = 42
device = "cuda"
