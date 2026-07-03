# WikiText103 seq256 permuted block64 ablation:
# start from a fresh random AO-GPT run, then attach the frozen try20 Attn-MLP
# order policy after a Random warmup. The MLP itself is frozen; only the main
# AO-GPT model is trained.
#
# Schedule:
# - iter < 8000: pure Random orders while collecting attention/loss features
# - 8000..20000: linearly increase MLP-order usage from 25% to 100%
# - iter >= 20000: use the frozen MLP cached order whenever available
#
# Original L2R / tau remain diagnostics only and are not used by the policy.

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-try20-attn-mlp-frozen-warmup8k-anneal20k-b64-permute-block-50000-iters'
eval_interval = 250
eval_iters = 200
log_interval = 10

wandb_log = False
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-try20-attn-mlp-frozen-warmup8k-anneal20k-b64-permute-50000-iters'

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

# Frozen try20 Attn-MLP. This checkpoint was trained with 3 input planes:
# robust attention z-score, symmetric attention, and current block-loss
# difference. Therefore `attn_mlp_policy_feature_mode = "try20"` is required.
attn_mlp_policy_enabled = True
attn_mlp_policy_path = 'Report/attn_mlp_loss_design_exploration/try20_all_heads_full_nll_dominant_formal/policy.pt'
attn_mlp_policy_freeze = True

attn_mlp_policy_layer = 2
attn_mlp_policy_head = -1
attn_mlp_policy_use_global = False
attn_mlp_policy_export_type = 'with_none'
attn_mlp_policy_feature_mode = 'try20'
attn_mlp_policy_feature_clip = 8.0
attn_mlp_policy_input_normalization = 'none'

attn_mlp_policy_start_iter = 8000
attn_mlp_policy_start_prob = 0.25
attn_mlp_policy_end_prob = 1.0
attn_mlp_policy_anneal_start_iter = 8000
attn_mlp_policy_anneal_end_iter = 20000
attn_mlp_policy_collect_warmup_attention = True

attn_mlp_policy_ema_decay = 0.95
attn_mlp_policy_update_every = 1
attn_mlp_policy_order_mode = 'argsort_desc'
attn_mlp_policy_per_sample = False
attn_mlp_policy_fallback = 'random'
attn_mlp_policy_log_interval = 100
attn_mlp_policy_save_orders = True
