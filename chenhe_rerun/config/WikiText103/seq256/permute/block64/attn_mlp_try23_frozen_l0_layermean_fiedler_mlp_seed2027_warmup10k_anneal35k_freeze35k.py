# try_23: Seed ablation for frozen L0 layer-mean Fiedler-distilled Attn-MLP.
#
# Paired comparison for distribution try_24:
# - distribution baseline: seed=2027, permute_seed=42, L0 all-head mean,
#   without_none, pairwise_max_fiedler, update20, warmup10k, anneal35k, fixed50k.
# - this run: same seed and schedule, but uses the frozen try_21 MLP as the
#   order operator on L0 all-head mean attention.
#
# The MLP is frozen. This isolates seed sensitivity of the backbone-level MLP
# replacement for the latest graph-Laplacian/Fiedler distribution method.

_base_config = 'config/WikiText103/seq256/permute/block64/random.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

seed = 2027
permute_seed = 42

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-try23-frozen-l0-layermean-fiedler-mlp-seed2027-warmup10k-anneal35k-freeze35k-b64-permute-block-50000-iters'

init_from = 'scratch'
resume_optimizer_state = False
max_iters = 50000
lr_decay_iters = 50000
always_save_checkpoint = True
compile = False

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-try23-frozen-l0-layermean-fiedler-mlp-seed2027-warmup10k-anneal35k-freeze35k-b64-permute-50000-iters'

aogpt_train_mode = 'AttnMLPFrozenOrder'
main_eval_mode = 'Random'
generalization_eval_mode = ''

attn_mlp_policy_enabled = True
attn_mlp_policy_random_init = False
attn_mlp_policy_path = 'checkpoints/attn_mlp_distillation/try21_l0_layermean_fiedler_mlp_h2048_2048_1024/best_by_val_tau.pt'
attn_mlp_policy_freeze = True
attn_mlp_policy_train_loss = 'none'

# L0 all-head mean, matching distribution try_24 head_signal_probe_heads='0:mean'.
attn_mlp_policy_use_global = False
attn_mlp_policy_layer = 0
attn_mlp_policy_head = -1
attn_mlp_policy_export_type = 'without_none'
attn_mlp_policy_feature_mode = 'attention'
attn_mlp_policy_input_channels = 1
attn_mlp_policy_feature_clip = 0.0
attn_mlp_policy_input_normalization = 'zscore'

# try_21 distillation trained with higher logits meaning earlier blocks.
attn_mlp_policy_order_mode = 'argsort_desc'
attn_mlp_policy_per_sample = False
attn_mlp_policy_fallback = 'random'

# Match the latest distribution bridge schedule:
# - 0..10000: pure Random warmup.
# - 10000..35000: refresh MLP order every 20 steps; train anneals Random -> MLP.
# - 35000..50000: stop refreshing and use the final cached MLP order.
attn_mlp_policy_start_iter = 10000
attn_mlp_policy_start_prob = 0.0
attn_mlp_policy_end_prob = 1.0
attn_mlp_policy_anneal_start_iter = 10000
attn_mlp_policy_anneal_end_iter = 35000
attn_mlp_policy_collect_warmup_attention = False
attn_mlp_policy_update_every = 20
attn_mlp_policy_update_stop_iter = 35000
attn_mlp_policy_log_interval = 50
attn_mlp_policy_save_orders = True

# One smoothing path, analogous to the distribution priority EMA decay.
attn_mlp_policy_ema_decay = 0.95
attn_mlp_policy_logits_ema_enabled = False

attn_mlp_policy_log_input_attn = True
attn_mlp_policy_log_input_attn_interval = 1000
attn_mlp_policy_log_input_attn_prefix = 'try23_l0_layermean_fiedler_mlp_seed2027_input_attn'
attn_mlp_policy_log_input_attn_out_dir = 'Report/language/wikitext103/mlp/distillation/try_23/input_attn'
attn_mlp_policy_log_input_attn_save_latest = True
attn_mlp_policy_log_input_attn_cmap = 'coolwarm'
attn_mlp_policy_log_input_attn_vmax_percentile = 99.0

# Keep this run a pure frozen-distillation backbone test.
attn_mlp_policy_loss_stop_enabled = False
attn_mlp_policy_axis_profile_weight = 0.0
attn_mlp_policy_axis_profile_dir_weight = 0.0
attn_mlp_policy_sampled_orders_per_state = 0
attn_mlp_policy_random_baseline_orders = 0
attn_mlp_policy_pg_weight = 0.0
attn_mlp_policy_prefix_reward_weight = 0.0
attn_mlp_policy_move_pref_weight = 0.0
attn_mlp_policy_move_pref_pairs_per_state = 0
attn_mlp_policy_attn_pair_weight = 0.0
attn_mlp_policy_attn_close_weight = 0.0
attn_mlp_policy_head_profile_weight = 0.0
attn_mlp_policy_head_profile_every = 0

online_spectral_policy_enabled = False
online_spectral_enabled = False
online_attn_probe_enabled = False
head_signal_probe_enabled = False
head_direction_logger_enabled = False
head_asym_selector_enabled = False
head_signal_consistency_enabled = False
head_signal_position_regularizer_enabled = False

