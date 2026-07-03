# WikiText103 seq256 non-permuted block64 config for the one-step lagged
# fixed-head online attention-spectral teacher policy.
#
# This is Version A: one specified layer/head is used as the policy signal.
# Every update step recovers a current-frame block order with the old
# norm + spectral-axis + direction-scoring operator, then uses it on the next
# optimizer step. Training anneals from pure random orders to the recovered
# fixed-head order, so there is no separate policy warmup phase.

out_dir = 'out/base/nonpermute/seq256/block64/out-wikitext103-seq256-online-spectral-fixed-head-b64-50000-iters'
eval_interval = 250
eval_iters = 200
log_interval = 10

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-online-spectral-fixed-head-b64-nonpermute-50000-iters'

dataset = 'wikitext103'
batch_size = 64
block_size = 256
gradient_accumulation_steps = 2
permute_data = False
permute_seed = 42

model_type = 'aogpt'
train_stage = 'standard'
aogpt_train_mode = 'OnlineSpectralFixedHeadOrder'
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

online_spectral_policy_enabled = True
online_spectral_policy_layer = 0
online_spectral_policy_head = 7
online_spectral_policy_export_type = 'with_none'
online_spectral_policy_update_every = 1
online_spectral_policy_probe_batches = 64
online_spectral_policy_probe_batch_size = 64
online_spectral_policy_probe_split = 'train'
online_spectral_policy_probe_include_train_step_attention = False
online_spectral_policy_use_ema = False
online_spectral_policy_ema_decay = 0.0

online_spectral_policy_anneal_start_iter = 0
online_spectral_policy_anneal_end_iter = 10000
online_spectral_policy_start_prob = 0.0
online_spectral_policy_end_prob = 1.0
online_spectral_policy_fallback = 'random'

online_spectral_policy_num_components = 4
online_spectral_policy_component_pairs = '1-2'
online_spectral_policy_num_angles = 16
online_spectral_policy_k_values = '8,10'
online_spectral_policy_group_methods = 'gap'
online_spectral_policy_threshold_percentile = 60.0
online_spectral_policy_transform = 'relu'
online_spectral_policy_temperature = 1.0
online_spectral_policy_direction_lambdas = '0,0.1,0.25'
online_spectral_policy_directed_score_weight = 0.25
online_spectral_policy_band_quality_weight = 0.05
online_spectral_policy_score_adjacency_sym = 'max'
online_spectral_policy_log_interval = 100
online_spectral_policy_save_orders = True
online_spectral_policy_order_history_enabled = True
online_spectral_policy_order_history_top_candidates = 8
online_spectral_policy_order_history_include_priority = True
