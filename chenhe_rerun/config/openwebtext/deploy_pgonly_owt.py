# SCALE deploy — pgonly gβ (12L/768/8h, OWT). gβ frozen 10k->20k, PG 20k->60k.
# init from 10k warmup (chenhe-native ckpt, no compat needed). model dims from ckpt.
out_dir = 'out/rerun_owt/deploy_pgonly'
eval_interval = 1000
eval_iters = 100
log_interval = 50
wandb_log = True
wandb_project = 'amor-order'
wandb_run_name = 'scale12L768_pgonly_owt'

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
aogpt_train_mode = 'GBetaFrozenOrder'
main_eval_mode = 'AR'
generalization_eval_mode = ''
block_order_block_len = 4

learning_rate = 6e-4
max_iters = 60000
lr_decay_iters = 60000
min_lr = 6e-5
beta2 = 0.95
warmup_iters = 0

init_from = 'ckpt'
init_from_ckpt = 'out/rerun_owt/scale12L768_warmup10k/ckpt.pt'
init_from_ckpt_mode = 'weights_only'

gbeta_ckpt = 'out/rerun_owt/gbeta_owt_bm16/g_beta_best.pt'
gbeta_parent_ckpt = 'out/rerun_owt/scale12L768_warmup10k/ckpt.pt'
gbeta_anchor_size = 16
gbeta_probe_order_mode = 'deployment'
gbeta_batch_mean_probes = 4
gbeta_refresh_every = 1
gbeta_probe_mode = 'eval'
gbeta_trainable = True

gbeta_pg_start_iter = 20000
gbeta_pg_update_every = 100
gbeta_pg_k = 4
gbeta_pg_tau = 0.05
gbeta_pg_lr = 3e-5
gbeta_pg_adv_clip = 0.1
gbeta_teacher_refresh_every = 0
