# WikiText103 seq256 permuted block64 config for one-pass lagged
# Frozen Attention-MLP order policy.
#
# This keeps the random baseline architecture/budget, but switches the training
# order carrier after attn_mlp_policy_start_iter. Orders are current-frame block
# ids. Original-frame diagnostics are reporting-only.

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-attn-mlp-frozen-order-b64-permute-block-50000-iters'
eval_interval = 250
eval_iters = 200
log_interval = 10

wandb_log = False
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-attn-mlp-frozen-order-b64-permute-50000-iters'

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

block_order_block_len = 4

learning_rate = 1e-3
max_iters = 50000
lr_decay_iters = 50000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 0

# Frozen Attn-MLP policy. Replace this with the MLP checkpoint distilled from
# the attention-spectral recovery operator for this exact num_blocks=64 setting.
attn_mlp_policy_enabled = True
attn_mlp_policy_start_iter = 8000
attn_mlp_policy_path = 'checkpoints/attn_mlp/wikitext103_seq256_permute_block64/policy.pt'
attn_mlp_policy_freeze = True

attn_mlp_policy_layer = 0
attn_mlp_policy_head = 7
attn_mlp_policy_use_global = False
attn_mlp_policy_export_type = 'with_none'
attn_mlp_policy_input_normalization = 'none'

attn_mlp_policy_ema_decay = 0.95
attn_mlp_policy_update_every = 1
attn_mlp_policy_order_mode = 'argsort'
attn_mlp_policy_per_sample = False
attn_mlp_policy_fallback = 'random'
attn_mlp_policy_collect_warmup_attention = True

attn_mlp_policy_log_interval = 100
attn_mlp_policy_save_orders = True
