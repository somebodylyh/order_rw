# WikiText103 seq80 non-permuted token-level config for the online spectral
# fixed-head cached-order policy.
#
# This uses train.py's OnlineSpectralFixedHeadOrder path: the model trains with
# a one-step-lagged cached block/token order recovered from one fixed attention
# head. The cached order is periodically refreshed from no-grad train probes and
# annealed in from random fallback orders. Each refresh enumerates spectral
# candidates from that fixed head and caches the highest attention-spectral
# scoring order, matching the seq256/block64 fixed-head flow.

out_dir = 'out/base/nonpermute/seq80/block1/out-wikitext103-seq80-online-spectral-fixed-head-b1-nonpermute-50000-iters'
eval_interval = 250
eval_iters = 200
log_interval = 10

wandb_log = True
wandb_project = 'AOGPT-order-block-seq80-final'
wandb_run_name = 'seq80-online-spectral-fixed-head-b1-nonpermute-50000-iters'

dataset = 'wikitext103'
data_record_mode = 'stream'
batch_size = 256
block_size = 80
gradient_accumulation_steps = 2
permute_data = False
permute_seed = 42
permute_mode = 'block'

model_type = 'aogpt'
train_stage = 'standard'
aogpt_train_mode = 'OnlineSpectralFixedHeadOrder'
main_eval_mode = 'Random'
generalization_eval_mode = ''
order_impl = 'token'
n_layer = 4
n_head = 8
n_embd = 384
dropout = 0

block_order_block_len = 1
block_order_layout = 'contiguous'

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

# Match the seq256/block64 fixed-head online flow: refresh the cached order
# every optimizer step from 64 x 64 = 4096 no-grad train probe sequences.
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
