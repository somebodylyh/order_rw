# AMOR VQ-image deployment — PG-only gβ arm (tau=0.05, teacher off).
# Continues the random VQ backbone (Imagenet64 VQ patch2x2) under gβ order.
# model dims (n_layer8/head8/embd512/vocab8192) come from the ckpt's model_args.

out_dir = '../out/rerun_vq/deploy_pgonly'
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
aogpt_train_mode = 'GBetaFrozenOrder'
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

# ── anchor-router gβ order policy (PG-only) ──
gbeta_ckpt = '../out/rerun_vq/gbeta_vq_bm16/g_beta_best.pt'
gbeta_parent_ckpt = '../probe_results_image/vq64_fixed_random_l8h8e512/ckpt_step30000.pt'
gbeta_anchor_size = 16
gbeta_probe_order_mode = 'deployment'
gbeta_batch_mean_probes = 4
gbeta_refresh_every = 1
gbeta_probe_mode = 'eval'
gbeta_trainable = True

gbeta_pg_start_iter = 1000
gbeta_pg_update_every = 100
gbeta_pg_k = 4
gbeta_pg_tau = 0.05
gbeta_pg_lr = 3e-5
gbeta_pg_adv_clip = 0.1
gbeta_teacher_refresh_every = 0
