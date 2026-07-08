# AMOR VQ-image deploy from 15k warmup — frozen (gβ frozen-deployed 15k->50k, no PG).
# gβ head-selected + trained on the 15k warmup ckpt (earlier/generic signal).
# model dims inherited from ckpt model_args (n_layer8/head8/embd512, vocab8192).

out_dir = '../out/rerun_vq/deploy15k_frozen'
eval_interval = 500
eval_iters = 100
log_interval = 50
wandb_log = True
wandb_project = 'amor-order'
wandb_run_name = 'vq15k_frozen'

dataset = 'imagenet64vq_patch2x2'
batch_size = 64
block_size = 256
gradient_accumulation_steps = 2
permute_data = True
permute_seed = 42
permute_mode = 'block'

model_type = 'aogpt'
train_stage = 'standard'
aogpt_train_mode = 'GBetaFrozenOrder'
main_eval_mode = 'AR'
generalization_eval_mode = ''
block_order_block_len = 4

learning_rate = 1e-4
max_iters = 50000
lr_decay_iters = 50000
min_lr = 1e-5
beta2 = 0.99
warmup_iters = 200

init_from = 'ckpt'
init_from_ckpt = '../out/rerun_vq/vq_backbone_15k_compat.pt'
init_from_ckpt_mode = 'weights_only'

# ── anchor-router gβ order policy (PG-only) ──
gbeta_ckpt = '../out/rerun_vq/gbeta_vq15k_bm16/g_beta_best.pt'
gbeta_parent_ckpt = '../probe_results_image/vq64_fixed_random_l8h8e512/ckpt_step15000.pt'
gbeta_anchor_size = 16
gbeta_probe_order_mode = 'deployment'
gbeta_batch_mean_probes = 4
gbeta_refresh_every = 1
gbeta_probe_mode = 'eval'
gbeta_trainable = False

gbeta_pg_start_iter = 20000   # gβ frozen-deployed 15k->20k, then PG 20k->50k
gbeta_pg_update_every = 100
gbeta_pg_k = 4
gbeta_pg_tau = 0.05
gbeta_pg_lr = 3e-5
gbeta_pg_adv_clip = 0.1
gbeta_teacher_refresh_every = 0
