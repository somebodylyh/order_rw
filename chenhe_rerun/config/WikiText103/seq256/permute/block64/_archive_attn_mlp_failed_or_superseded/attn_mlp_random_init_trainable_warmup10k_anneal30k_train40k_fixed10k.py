# WikiText103 seq256 permuted block64 ablation:
# random-init trainable Attn-MLP with a four-phase schedule.
#
# Schedule:
# - iter < 10000: pure Random orders; MLP is not trained
# - 10000..30000: train MLP online and anneal MLP-order usage from 0% to 100%
# - 30000..40000: keep training MLP while using MLP cached order at 100%
# - 40000..50000: freeze MLP/order updates and train the main model with the
#   fixed cached order from the end of the previous phase
#
# The MLP training signal is current-frame loss_diff pairwise preference only.
# Original L2R / tau remain diagnostics and are not used by this policy.

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-attn-mlp-random-init-trainable-warmup10k-anneal30k-train40k-fixed10k-b64-permute-block-50000-iters'
eval_interval = 250
eval_iters = 200
log_interval = 10

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-attn-mlp-random-init-trainable-warmup10k-anneal30k-train40k-fixed10k-b64-permute-50000-iters'

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

attn_mlp_policy_enabled = True
attn_mlp_policy_random_init = True
attn_mlp_policy_path = ''
attn_mlp_policy_freeze = False

attn_mlp_policy_layer = 2
attn_mlp_policy_head = -1
attn_mlp_policy_use_global = False
attn_mlp_policy_export_type = 'with_none'
attn_mlp_policy_feature_mode = 'try20'
attn_mlp_policy_feature_clip = 8.0
attn_mlp_policy_input_normalization = 'none'
attn_mlp_policy_input_channels = 3
attn_mlp_policy_hidden_dims = '1024,1024'
attn_mlp_policy_dropout = 0.0
attn_mlp_policy_activation = 'gelu'

attn_mlp_policy_start_iter = 10000
attn_mlp_policy_start_prob = 0.0
attn_mlp_policy_end_prob = 1.0
attn_mlp_policy_anneal_start_iter = 10000
attn_mlp_policy_anneal_end_iter = 30000
attn_mlp_policy_collect_warmup_attention = False

attn_mlp_policy_ema_decay = 0.95
attn_mlp_policy_update_every = 1
attn_mlp_policy_update_stop_iter = 40000
attn_mlp_policy_order_mode = 'argsort_desc'
attn_mlp_policy_per_sample = False
attn_mlp_policy_fallback = 'random'
attn_mlp_policy_log_interval = 100
attn_mlp_policy_save_orders = True

attn_mlp_policy_lr = 2e-5
attn_mlp_policy_weight_decay = 0.01
attn_mlp_policy_beta1 = 0.9
attn_mlp_policy_beta2 = 0.99
attn_mlp_policy_grad_clip = 1.0
attn_mlp_policy_train_loss = 'loss_diff_pairwise'
attn_mlp_policy_pair_tau = 1.0
attn_mlp_policy_pair_margin = 0.25
attn_mlp_policy_pair_max_weight = 4.0
attn_mlp_policy_logit_l2 = 1e-4
attn_mlp_policy_min_logit_std = 0.5
attn_mlp_policy_std_floor_weight = 0.01
