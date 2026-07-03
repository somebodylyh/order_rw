# WikiText103 seq256 permuted block64 try_2 main continuation.
#
# Resume from the random permuted 50k checkpoint, attach the no-prior MLP
# policy trained in Report/MLP_loss_training/try_2/policy_train, update cached
# order for 5k steps, then freeze the cached order and adapt the main model to
# 80k.
#
# No OriginalL2R/tau/L2R/permutation oracle is used for training, rerank,
# early stop, or selection. Original-frame fields are diagnostics only.

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-try2-policy-resume-random50k-update55k-fixed80k-b64-permute-block'
eval_interval = 250
eval_iters = 200
log_interval = 10

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-try2-policy-resume-random50k-update55k-fixed80k-b64-permute'

dataset = 'wikitext103'
batch_size = 64
block_size = 256
gradient_accumulation_steps = 2
permute_data = True
permute_seed = 42
permute_mode = 'block'

model_type = 'aogpt'
train_stage = 'standard'
aogpt_train_mode = 'AttnMLPFrozenOrder'
main_eval_mode = 'Random'
generalization_eval_mode = ''
n_layer = 4
n_head = 8
n_embd = 384
dropout = 0
compile = False

block_order_block_len = 4

init_from = 'resume'
resume_optimizer_state = False

learning_rate = 2e-4
decay_lr = False
max_iters = 80000
lr_decay_iters = 80000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 0

attn_mlp_policy_enabled = True
attn_mlp_policy_path = 'Report/MLP_loss_training/try_2/policy_train/policy.pt'
attn_mlp_policy_freeze = True

attn_mlp_policy_layer = 2
attn_mlp_policy_head = -1
attn_mlp_policy_use_global = False
attn_mlp_policy_export_type = 'with_none'
attn_mlp_policy_feature_mode = 'try20'
attn_mlp_policy_feature_clip = 8.0
attn_mlp_policy_input_normalization = 'none'

attn_mlp_policy_start_iter = 50000
attn_mlp_policy_start_prob = 1.0
attn_mlp_policy_end_prob = 1.0
attn_mlp_policy_anneal_start_iter = 50000
attn_mlp_policy_anneal_end_iter = 50000
attn_mlp_policy_collect_warmup_attention = True

attn_mlp_policy_ema_decay = 0.95
attn_mlp_policy_update_every = 1
attn_mlp_policy_update_stop_iter = 55000
attn_mlp_policy_order_mode = 'argsort_desc'
attn_mlp_policy_per_sample = False
attn_mlp_policy_fallback = 'random'
attn_mlp_policy_log_interval = 100
attn_mlp_policy_save_orders = True
