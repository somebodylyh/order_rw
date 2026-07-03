# WikiText103 seq80 permuted block1 seq-teacher try_1.
#
# Step-1 teacher-only feasibility run. This intentionally does not create or
# train an Attn-MLP. The goal is to validate the seq80/token-level teacher
# behavior before introducing MLP distillation.
#
# Isolated seq layout:
# - config lives under seq80/permute/block1/seq_teacher/
# - outputs go to out/seq/...
# - reports go to Report/language/wikitext103/seq/...
#
# Base alignment:
# - seq80 token-level permuted random base
# - model size fixed at 4 layers, 8 heads, 384 emb
# - block_order_block_len=1, order_impl='token'
#
# Teacher method follows the strongest block Fiedler-teacher family:
# - current-frame attention only
# - all-layer all-head mean attention as the input matrix
# - with_none token-level export
# - W=max(A,A.T), diagonal zeroed
# - graph Laplacian Fiedler vector recovers one undirected axis
# - current-model linear_profile_loss chooses raw vs reverse direction
# - no priority/order EMA; accepted teacher order is cached directly
#
# Version-B schedule:
# - 0..10000: pure random, no teacher/probe updates
# - 10000..32000: update teacher every 20 steps; policy prob 0.0 -> 0.8
# - 32000..35000: stop teacher updates; cached order prob 0.8 -> 1.0
# - 35000..50000: deterministic fixed cached teacher order
#
# Original L2R/tau/PPL remain diagnostics only.

_base_config = 'config/WikiText103/seq80/permute/block1/random.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

seed = 2053
permute_seed = 42

out_dir = (
    'out/seq/permute/seq80/block1/'
    'out-wikitext103-seq80-seqteacher-try1-online-spectral-'
    'alllayers-allheads-withnone-noema-versionB-update20-warmup10k-'
    'stop32k-prob08-freeze35k-b1-permute'
)

init_from = 'scratch'
resume_optimizer_state = False
max_iters = 50000
lr_decay_iters = 50000
always_save_checkpoint = True
compile = False
eval_batch_size = 64

wandb_log = True
wandb_project = 'AOGPT-order-seq80-final'
wandb_run_name = (
    'seq80-seqteacher-try1-online-spectral-alllayers-allheads-withnone-'
    'noema-versionB-update20-warmup10k-stop32k-prob08-freeze35k'
)

aogpt_train_mode = 'OnlineSpectralFixedHeadOrder'
main_eval_mode = 'Random'
generalization_eval_mode = ''

# Disable the old online spectral updater; all teacher updates come from the
# head-signal bridge below.
online_spectral_policy_enabled = True
online_spectral_policy_probe_batches = 0
online_spectral_policy_probe_include_train_step_attention = False
online_spectral_policy_loss_rerank_enabled = False

teacher_start = 10000
teacher_stop = 32000
freeze_iter = 35000

online_spectral_policy_update_every = 20
online_spectral_policy_update_start_iter = teacher_start
online_spectral_policy_update_stop_iter = teacher_stop

online_spectral_policy_anneal_start_iter = teacher_start
online_spectral_policy_anneal_end_iter = freeze_iter
online_spectral_policy_start_prob = 0.0
online_spectral_policy_end_prob = 1.0
online_spectral_policy_prob_schedule = 'piecewise'
online_spectral_policy_prob_points = '0:0.0,9999:0.0,10000:0.0,32000:0.8,35000:1.0,50000:1.0'
online_spectral_policy_fallback = 'random'

# Teacher-only fixed-hard-order policy: no rank/priority EMA, no MLP.
online_spectral_policy_use_ema = False
online_spectral_policy_ema_decay = 0.0
online_spectral_policy_freeze_to_map_order_after_stop = False

online_spectral_policy_try19_bridge_enabled = True
online_spectral_policy_try19_bridge_mode = 'fixed_top1'
online_spectral_policy_try19_bridge_order_field = 'loss_profile_selected_order_current'
online_spectral_policy_try19_bridge_score_field = 'loss_profile_score_gap'
online_spectral_policy_try19_bridge_min_candidate_score = 1e-4
online_spectral_policy_try19_bridge_max_candidates = 1
online_spectral_policy_try19_bridge_start_iter = teacher_start
online_spectral_policy_try19_bridge_stop_iter = teacher_stop

online_spectral_policy_order_history_enabled = True
online_spectral_policy_order_history_top_candidates = 1
online_spectral_policy_order_history_include_priority = False

head_signal_probe_enabled = True
head_signal_probe_start_iter = teacher_start
head_signal_probe_stop_iter = teacher_stop
head_signal_probe_interval = 20

_report_root = (
    'Report/language/wikitext103/order_teacher_distribution/token/seq80/try_1'
)
head_signal_probe_out_dir = _report_root + '/online_training_probe'

# all:mean is the seq-level teacher input: mean over all layers and all heads.
head_signal_probe_heads = 'all:mean'
head_signal_probe_batches = 4
head_signal_probe_batch_size = 256
head_signal_probe_loss_batches = 4
head_signal_probe_loss_batch_size = 128
head_signal_probe_candidate_batch_size = 2
head_signal_probe_prefix_k = 16
head_signal_probe_prefix_weight = 0.7
head_signal_probe_full_weight = 0.3
head_signal_probe_export_type = 'with_none'

head_signal_probe_candidate_source = 'pairwise_max_fiedler'
head_signal_probe_top_m = 1
head_signal_probe_candidate_loss_profile_max_rank = 1
head_signal_probe_candidate_loss_profile_include_reverse = True

head_signal_probe_consensus_enabled = False
head_signal_probe_consensus_leave_one_out = False
head_signal_probe_deterministic = True
head_signal_probe_seed = 24681357
head_signal_probe_orientation_rule = 'linear_profile_candidate'
head_signal_probe_candidate_loss_profile_enabled = True
head_signal_probe_candidate_loss_profile_score = 'linear_profile'
head_signal_probe_candidate_loss_profile_exp_tau = 16.0
head_signal_probe_candidate_loss_profile_min_gap = 0.0
head_signal_probe_candidate_loss_profile_min_alignment = 0.0
head_signal_probe_candidate_loss_profile_low_confidence_fallback = 'none'
head_signal_probe_position_anchor_enabled = False
head_signal_probe_position_consensus_enabled = False

online_spectral_policy_log_input_attn = True
online_spectral_policy_log_input_attn_interval = 1000
online_spectral_policy_log_input_attn_prefix = (
    'seq80_seqteacher_try1_alllayers_allheads_withnone_noema_versionB_input_attn'
)
online_spectral_policy_log_input_attn_out_dir = _report_root + '/input_attn'
online_spectral_policy_log_input_attn_save_latest = True
online_spectral_policy_log_input_attn_cmap = 'coolwarm'
online_spectral_policy_log_input_attn_vmax_percentile = 99.0

attn_mlp_policy_enabled = False
online_spectral_enabled = False
head_direction_logger_enabled = False
head_asym_selector_enabled = False

assert bool(permute_data)
assert str(permute_mode) == 'block'
assert int(block_size) == 80
assert int(block_order_block_len) == 1
assert str(order_impl) == 'token'
assert int(n_layer) == 4
assert int(n_head) == 8
assert int(n_embd) == 384
assert int(batch_size) == 256
assert int(gradient_accumulation_steps) == 2
assert str(aogpt_train_mode) == 'OnlineSpectralFixedHeadOrder'
assert str(head_signal_probe_heads) == 'all:mean'
assert str(head_signal_probe_export_type) == 'with_none'
assert int(head_signal_probe_batches) * int(head_signal_probe_batch_size) == 1024
assert int(head_signal_probe_loss_batches) * int(head_signal_probe_loss_batch_size) == 512
assert not bool(online_spectral_policy_use_ema)
assert not bool(attn_mlp_policy_enabled)
