"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import itertools
import pickle
import sys
import json
import csv
from ast import literal_eval
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

from AOGPT import AOGPTConfig, AOGPT
from attn_mlp_order_policy import (
    FeatureAttentionOrderMLP,
    FlatAttentionOrderMLP,
    load_frozen_attn_mlp_policy,
    logits_to_order,
)
from online_spectral_order_policy import (
    FixedHeadSpectralPolicyConfig,
    affinity_from_adjacency,
    anti,
    candidates_to_priority_vector,
    grouped_order,
    parse_floats,
    parse_ints,
    parse_pairs,
    recover_fixed_head_spectral_candidates,
    robust_z,
    sample_order_from_priority,
    score_order,
    spectral_coordinates,
    sym,
)
from order_utils import (
    block_permutation_to_token_permutation,
    build_fixed_block_permutation,
    expand_block_orders_to_token_orders,
    kendall_tau_to_l2r_per_sample,
    sample_random_block_orders,
    token_losses_to_block_losses,
    invert_permutation,
)

os.environ.setdefault("WANDB_MODE", "online")
# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out/manual/default'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_batch_size = 0 # if >0, use this micro-batch size for estimate_loss instead of batch_size
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
save_iter_checkpoints = False
save_iter_checkpoint_dir = ''
save_iter_checkpoint_keep = 0
save_iter_checkpoint_steps = ''
init_from = 'scratch' # 'scratch' or 'resume' or 'ckpt' or 'gpt2*'
resume_optimizer_state = True # if False, resume model weights but reinitialize optimizer
# ── V3 frozen-gβ order policy (orderhead_v3); only used by aogpt_train_mode='GBetaFrozenOrder' ──
gbeta_ckpt = ''            # path to g_beta_best.pt
gbeta_batch_mean_probes = 4
gbeta_refresh_every = 1
gbeta_probe_mode = 'eval'  # backbone mode for attention probes (parity with Stage-1 producer)
init_from_ckpt = ''        # arbitrary parent ckpt for init_from='ckpt' continuation
init_from_ckpt_mode = 'full_state'  # 'full_state' (optimizer+scheduler) | 'weights_only'
# wandb logging
wandb_log = True 
wandb_project = 'ao-gpt-experiments' # 你的项目名称
wandb_run_name = 'mdm_random_order_run1' # 你的实验运行名称
wandb_run_id = ''
seed = 1337
# data
dataset = 'openwebtext'
data_record_mode = 'stream'
permute_data = False
permute_seed = 42
permute_mode = 'block'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# AOGPT-only settings
aogpt_train_mode = 'AR'
main_eval_mode = 'Random'
generalization_eval_mode = ''
order_impl = 'block'
block_order_layout = 'contiguous'
fixed_block_order = '' # comma/space separated block ids for aogpt_train_mode='FixedBlockOrder'
image_size = 0
image_block_size = 0
image_block_height = 0
image_block_width = 0
position_encoding_mode = 'absolute'
rope_theta = 10000.0
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
eval_generate_step_loss_log = True
eval_generate_step_batches = 200
eval_generate_step_loss_filename = 'generate_step_block_loss_latest.png'
eval_kendall_distance_log = True
eval_kendall_num_orders = 100
train_stage = 'standard'
block_order_block_len = 16
segment_guided_ratio = 0.0
segment_source_json = ''
segment_top_k_pairs = 64
segment_max_len = 4
segment_max_units_per_order = 2
segment_use_all_units = False
segment_lock_final_units = True
segment_schedule_boundaries = ''
segment_schedule_ratios = ''
segment_schedule_source_jsons = ''
# Online cheap recovery diagnostics. These are disabled by default and only
# consume current-frame/model-side signals from the active training policy.
online_pair_stats_enabled = False
online_pair_stats_out_dir = ''
online_pair_stats_write_every = 250
online_pair_stats_min_count = 1
online_pair_stats_top_k_export = 256
online_spectral_enabled = False
online_spectral_mode = 'cache_exact' # cache_exact or subspace
online_spectral_out_dir = ''
online_spectral_interval = 0
online_spectral_batch_size = 8
online_spectral_max_updates = 0
online_spectral_write_every = 1
online_spectral_export_type = 'with_none'
online_spectral_ema_decay = 0.95
online_spectral_subspace_rank = 8
online_spectral_subspace_steps_per_update = 1
online_spectral_subspace_seed = 12345
online_spectral_threshold_percentile = 60.0
online_spectral_transform = 'relu'
online_spectral_temperature = 1.0
attn_mlp_policy_enabled = False
attn_mlp_policy_start_iter = 8000
attn_mlp_policy_path = ''
attn_mlp_policy_freeze = True
attn_mlp_policy_layer = 0
attn_mlp_policy_head = 0
attn_mlp_policy_use_global = False
attn_mlp_policy_export_type = 'with_none'
attn_mlp_policy_feature_mode = 'attention'
attn_mlp_policy_feature_clip = 0.0
attn_mlp_policy_input_normalization = 'none'
attn_mlp_policy_attention_ema_enabled = False
attn_mlp_policy_ema_decay = 0.95
attn_mlp_policy_update_every = 1
attn_mlp_policy_update_stop_iter = 0 # 0 disables the stop; otherwise freeze cached order at this iter
attn_mlp_policy_order_mode = 'argsort'
attn_mlp_policy_per_sample = False
attn_mlp_policy_fallback = 'random'
attn_mlp_policy_collect_warmup_attention = False
attn_mlp_policy_start_prob = 1.0
attn_mlp_policy_end_prob = 1.0
attn_mlp_policy_anneal_start_iter = 8000
attn_mlp_policy_anneal_end_iter = 8000
attn_mlp_policy_prob_schedule = 'linear' # linear or piecewise
attn_mlp_policy_prob_points = '' # e.g. "20000:0.0,32000:0.8,35000:1.0"
attn_mlp_policy_log_interval = 100
attn_mlp_policy_save_orders = True
attn_mlp_policy_logits_ema_enabled = False
attn_mlp_policy_logits_ema_decay = 0.95
attn_mlp_policy_logits_ema_normalize = 'zscore' # sigmoid, zscore, center, none
attn_mlp_policy_logits_ema_start_iter = 0
attn_mlp_policy_log_input_attn = False
attn_mlp_policy_log_input_attn_interval = 0
attn_mlp_policy_log_input_attn_prefix = 'attn_mlp_input_attn'
attn_mlp_policy_log_input_attn_out_dir = ''
attn_mlp_policy_log_input_attn_save_latest = True
attn_mlp_policy_log_input_attn_cmap = 'coolwarm'
attn_mlp_policy_log_input_attn_vmax_percentile = 99.0
attn_mlp_policy_order_history_enabled = True
attn_mlp_policy_order_history_interval = 1
attn_mlp_policy_order_history_path = ''
attn_mlp_policy_order_history_include_scores = True
attn_mlp_policy_order_history_include_input_stats = True
attn_mlp_policy_teacher_diag_enabled = False
attn_mlp_policy_teacher_diag_interval = 1
attn_mlp_policy_teacher_diag_orientation = 'match_mlp' # match_mlp, raw, reverse, original_best
attn_mlp_policy_teacher_diag_include_scores = True
attn_mlp_policy_lazy_init_enabled = False
attn_mlp_policy_lazy_init_iter = 0
attn_mlp_policy_random_init = False
attn_mlp_policy_input_channels = 0 # 0 infers from feature_mode
attn_mlp_policy_hidden_dims = '1024,1024'
attn_mlp_policy_dropout = 0.0
attn_mlp_policy_activation = 'gelu'
attn_mlp_policy_lr = 2e-5
attn_mlp_policy_lr_anneal_enabled = False
attn_mlp_policy_lr_anneal_start_iter = 0
attn_mlp_policy_lr_anneal_end_iter = 0
attn_mlp_policy_lr_anneal_min_lr = 0.0
attn_mlp_policy_lr_anneal_style = 'cosine' # cosine, linear
attn_mlp_policy_loss_stop_enabled = False
attn_mlp_policy_loss_stop_metric = 'attn_mlp_train_loss'
attn_mlp_policy_loss_stop_start_iter = 0
attn_mlp_policy_loss_stop_patience = 5
attn_mlp_policy_loss_stop_min_delta = 0.0
attn_mlp_policy_loss_stop_check_every = 1
attn_mlp_policy_weight_decay = 0.01
attn_mlp_policy_beta1 = 0.9
attn_mlp_policy_beta2 = 0.99
attn_mlp_policy_grad_clip = 1.0
attn_mlp_policy_train_loss = 'sampled_nll_pg'
attn_mlp_policy_attention_batches_per_update = 1
attn_mlp_policy_shadow_mse_enabled = False
attn_mlp_policy_shadow_mse_start_iter = 8000
attn_mlp_policy_shadow_mse_stop_iter = 15000
attn_mlp_policy_shadow_mse_train_frac = 0.75
attn_mlp_policy_shadow_mse_batches_per_item = 1
attn_mlp_policy_shadow_mse_train_items_per_update = 0
attn_mlp_policy_shadow_mse_val_items_per_update = 0
attn_mlp_policy_shadow_mse_train_samples_per_step = 0
attn_mlp_policy_shadow_mse_val_samples_per_step = 0
attn_mlp_policy_shadow_mse_prefix_k = 16
attn_mlp_policy_shadow_mse_prefix_weight = 0.7
attn_mlp_policy_shadow_mse_full_weight = 0.3
attn_mlp_policy_shadow_mse_label_orientation = 'loss_profile' # loss_profile, linear_profile_loss, raw, reverse, original_best
attn_mlp_policy_shadow_mse_orientation_x_mode = 'all' # all, last_step
attn_mlp_policy_shadow_mse_log_path = ''
attn_mlp_policy_shadow_mse_eval_after_stop_enabled = False
attn_mlp_policy_shadow_mse_eval_after_stop_interval = 1
attn_mlp_policy_axis_profile_weight = 0.0
attn_mlp_policy_axis_profile_dir_weight = 0.0
attn_mlp_policy_axis_profile_dir_margin = 0.005
attn_mlp_policy_axis_profile_min_abs_q = 0.0
attn_mlp_policy_axis_profile_score = 'linear_profile'
attn_mlp_policy_directed_ribbon_flow_sign = -1.0
attn_mlp_policy_directed_ribbon_rank_tau = 1.0
attn_mlp_policy_directed_ribbon_margin = 1.0
attn_mlp_policy_directed_ribbon_band_width = 8.0
attn_mlp_policy_directed_ribbon_band_weight = 0.05
attn_mlp_policy_logit_l2 = 1e-4
attn_mlp_policy_min_logit_std = 0.5
attn_mlp_policy_std_floor_weight = 0.01
attn_mlp_policy_sampled_orders_per_state = 0
attn_mlp_policy_random_baseline_orders = 0
attn_mlp_policy_nll_states_per_update = 1
attn_mlp_policy_pg_weight = 1.0
attn_mlp_policy_prefix_reward_weight = 0.0
attn_mlp_policy_prefix_k = 8
attn_mlp_policy_reward_scale_floor = 0.001
attn_mlp_policy_advantage_clip = 5.0
attn_mlp_policy_move_pref_weight = 0.0
attn_mlp_policy_move_pref_pairs_per_state = 0
attn_mlp_policy_move_pref_window = 1
attn_mlp_policy_move_pref_tau = 1.0
attn_mlp_policy_move_pref_margin = 0.02
attn_mlp_policy_move_pref_max_weight = 8.0
attn_mlp_policy_move_pref_prefix_weight = 0.0
attn_mlp_policy_attn_pair_weight = 0.0
attn_mlp_policy_attn_pair_top_frac = 0.10
attn_mlp_policy_attn_pair_min_z = 0.0
attn_mlp_policy_attn_pair_max_weight = 8.0
attn_mlp_policy_attn_pair_tau = 2.0
attn_mlp_policy_attn_close_weight = 0.0
attn_mlp_policy_attn_close_tau = 4.0
attn_mlp_policy_attn_close_margin = 0.0
attn_mlp_policy_min_entropy = 0.0
attn_mlp_policy_max_entropy = 10.0
attn_mlp_policy_entropy_floor_weight = 0.0
attn_mlp_policy_entropy_ceiling_weight = 0.0
attn_mlp_policy_head_profile_weight = 0.0
attn_mlp_policy_head_profile_start_iter = 0
attn_mlp_policy_head_profile_every = 0
attn_mlp_policy_head_profile_tau = 2.0
attn_mlp_policy_head_profile_min_abs_q = 0.05
attn_mlp_policy_head_profile_max_weight = 4.0
attn_mlp_policy_head_profile_use_cached = True
attn_mlp_policy_head_profile_accept_gate_enabled = False
attn_mlp_policy_head_profile_accept_antialigned_only = True
attn_mlp_policy_head_profile_accept_alignment_threshold = 0.0
attn_mlp_policy_head_profile_accept_margin = 0.0
attn_mlp_policy_head_profile_best_memory_enabled = False
attn_mlp_policy_head_profile_best_memory_use_for_fixed = False
online_spectral_policy_enabled = False
online_spectral_policy_layer = 0
online_spectral_policy_head = 7
online_spectral_policy_export_type = 'with_none'
online_spectral_policy_update_every = 1
online_spectral_policy_update_start_iter = 0
online_spectral_policy_update_stop_iter = -1
online_spectral_policy_freeze_to_map_order_after_stop = False
online_spectral_policy_probe_batches = 0
online_spectral_policy_probe_batch_size = 64
online_spectral_policy_probe_split = 'train'
online_spectral_policy_probe_include_train_step_attention = False
online_spectral_policy_ema_decay = 0.0
online_spectral_policy_use_ema = False
online_spectral_policy_anneal_start_iter = 0
online_spectral_policy_anneal_end_iter = 8000
online_spectral_policy_start_prob = 0.0
online_spectral_policy_end_prob = 1.0
online_spectral_policy_prob_schedule = 'linear' # linear or piecewise
online_spectral_policy_prob_points = '' # e.g. "10000:0.0,32000:0.8,35000:1.0"
online_spectral_policy_fallback = 'random'
online_spectral_policy_log_interval = 100
online_spectral_policy_save_orders = True
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
online_spectral_policy_top_m = 32
online_spectral_policy_teacher_temperature = 1.0
online_spectral_policy_score_normalization = 'zscore'
online_spectral_policy_priority_ema_decay = 0.95
online_spectral_policy_continuous_priority_normalization = 'minmax'
online_spectral_policy_continuous_priority_eps = 1e-8
online_spectral_policy_distribution_sample_mode = 'gumbel' # gumbel or map
online_spectral_policy_sample_temperature = 0.7
online_spectral_policy_order_usage_mode = 'probability_anneal' # probability_anneal or temperature_sampling
online_spectral_policy_temperature_sampling_start_temperature = 2.0
online_spectral_policy_temperature_sampling_end_temperature = 0.1
online_spectral_policy_temperature_sampling_schedule = 'linear' # linear or cosine
online_spectral_policy_random_mix_prob = 0.2
online_spectral_policy_distribution_per_sample = True
online_spectral_policy_loss_rerank_enabled = False
online_spectral_policy_loss_rerank_top_k = 64
online_spectral_policy_loss_rerank_batches = 4
online_spectral_policy_loss_rerank_batch_size = 64
online_spectral_policy_loss_rerank_split = 'train'
online_spectral_policy_loss_rerank_candidate_batch_size = 1
online_spectral_policy_loss_rerank_prefix_k = 8
online_spectral_policy_loss_rerank_attention_weight = 1.0
online_spectral_policy_loss_rerank_prefix_weight = 0.25
online_spectral_policy_loss_rerank_full_weight = 0.15
online_spectral_policy_late_l2r_anneal_enabled = False
online_spectral_policy_late_l2r_anneal_start_iter = 8000
online_spectral_policy_late_l2r_anneal_end_iter = 50000
online_spectral_policy_late_l2r_start_prob = 0.0
online_spectral_policy_late_l2r_end_prob = 1.0
online_spectral_policy_late_l2r_per_sample = True
online_spectral_policy_order_history_enabled = False
online_spectral_policy_order_history_path = ''
online_spectral_policy_order_history_top_candidates = 0
online_spectral_policy_order_history_include_priority = True
online_spectral_policy_try19_bridge_enabled = False
online_spectral_policy_try19_bridge_mode = 'none' # fixed_top1, oriented_distribution, continuous_fiedler_minmax, or hybrid_direct_ema
online_spectral_policy_try19_bridge_order_field = 'loss_profile_consensus_order_current'
online_spectral_policy_try19_bridge_score_field = 'loss_profile_score_gap'
online_spectral_policy_try19_bridge_min_candidate_score = 1e-4
online_spectral_policy_try19_bridge_max_candidates = 32
online_spectral_policy_try19_bridge_start_iter = -1
online_spectral_policy_try19_bridge_stop_iter = -1
online_spectral_policy_hybrid_direct_prob_start = 0.0
online_spectral_policy_hybrid_direct_prob_end = 0.0
online_spectral_policy_hybrid_direct_prob_anneal_start_iter = 0
online_spectral_policy_hybrid_direct_prob_anneal_end_iter = 0
online_spectral_policy_hybrid_direct_prob_schedule = 'linear'
online_spectral_policy_hybrid_direct_prob_points = ''
online_spectral_policy_hybrid_per_sample = True
online_spectral_policy_hybrid_freeze_order = 'ema' # ema, hard, or hybrid
online_spectral_policy_hybrid_mix_hard_after_ema_stop = False
online_spectral_policy_hybrid_update_hard_after_ema_stop = False
online_spectral_policy_log_input_attn = False
online_spectral_policy_log_input_attn_interval = 0
online_spectral_policy_log_input_attn_prefix = 'online_spectral_input_attn'
online_spectral_policy_log_input_attn_out_dir = ''
online_spectral_policy_log_input_attn_save_latest = True
online_spectral_policy_log_input_attn_cmap = 'coolwarm'
online_spectral_policy_log_input_attn_vmax_percentile = 99.0
head_signal_probe_enabled = False
head_signal_probe_interval = 0
head_signal_probe_start_iter = 0
head_signal_probe_stop_iter = -1
head_signal_probe_out_dir = ''
head_signal_probe_heads = '1:6,2:7,3:4,0:7,1:5'
head_signal_probe_batches = 4
head_signal_probe_batch_size = 16
head_signal_probe_loss_batches = 4
head_signal_probe_loss_batch_size = 16
head_signal_probe_candidate_batch_size = 4
head_signal_probe_prefix_k = 16
head_signal_probe_prefix_weight = 0.7
head_signal_probe_full_weight = 0.3
head_signal_probe_export_type = 'with_none'
head_signal_probe_candidate_source = 'spectral' # spectral, direct_asym_eig, pairwise_max_fiedler
head_signal_probe_direct_asym_eig_mode = 'raw_right_largest_real_real'
head_signal_probe_fixed_angle_idx = 4
head_signal_probe_top_m = 32
head_signal_probe_num_components = 4
head_signal_probe_component_pairs = '1-2'
head_signal_probe_num_angles = 16
head_signal_probe_k_values = '8,10'
head_signal_probe_group_methods = 'gap'
head_signal_probe_threshold_percentile = 60.0
head_signal_probe_transform = 'relu'
head_signal_probe_temperature = 1.0
head_signal_probe_direction_lambdas = '0,0.1,0.25'
head_signal_probe_directed_score_weight = 0.25
head_signal_probe_band_quality_weight = 0.05
head_signal_probe_score_adjacency_sym = 'max'
head_signal_probe_consensus_enabled = True
head_signal_probe_consensus_leave_one_out = True
head_signal_probe_deterministic = False
head_signal_probe_seed = 24681357
head_signal_probe_orientation_rule = 'loss_consensus'
head_signal_probe_candidate_loss_profile_enabled = False
head_signal_probe_candidate_loss_profile_score = 'linear_profile'
head_signal_probe_candidate_loss_profile_max_rank = 6
head_signal_probe_candidate_loss_profile_include_reverse = True
head_signal_probe_candidate_loss_profile_exp_tau = 16.0
head_signal_probe_candidate_loss_profile_min_gap = 0.0
head_signal_probe_candidate_loss_profile_min_alignment = 0.0
head_signal_probe_candidate_loss_profile_low_confidence_fallback = 'none'
head_signal_probe_position_anchor_enabled = False
head_signal_probe_position_anchor_source = 'wtpe_pc1'
head_signal_probe_position_consensus_enabled = False
head_signal_probe_position_consensus_leave_one_out = True
all_head_attn_dataset_enabled = False
all_head_attn_dataset_start_iter = 10000
all_head_attn_dataset_stop_iter = 15000
all_head_attn_dataset_interval = 50
all_head_attn_dataset_batch_size = 16
all_head_attn_dataset_micro_batches_per_record = 1
all_head_attn_dataset_batches_per_step = 1
all_head_attn_dataset_export_type = 'without_none'
all_head_attn_dataset_order_mode = 'random' # random, ar, original_l2r
all_head_attn_dataset_out_dir = ''
all_head_attn_dataset_shard_size = 64
all_head_attn_dataset_dtype = 'float16' # float16, bfloat16, float32
all_head_attn_dataset_seed = 314159
all_head_attn_dataset_split = 'train'
all_head_attn_dataset_max_records = 0
all_head_pairwise_dataset_enabled = False
all_head_pairwise_dataset_start_iter = 10000
all_head_pairwise_dataset_stop_iter = 14992
all_head_pairwise_dataset_interval = 16
all_head_pairwise_dataset_attention_batch_size = 16
all_head_pairwise_dataset_attention_micro_batches_per_record = 16
all_head_pairwise_dataset_export_type = 'without_none'
all_head_pairwise_dataset_attention_order_mode = 'random' # random, ar, original_l2r
all_head_pairwise_dataset_out_dir = ''
all_head_pairwise_dataset_shard_size = 512
all_head_pairwise_dataset_dtype = 'float16' # float16, bfloat16, float32
all_head_pairwise_dataset_seed = 161803
all_head_pairwise_dataset_max_records = 0
all_head_pairwise_dataset_train_records = 250
all_head_pairwise_dataset_heads = 'all'
all_head_pairwise_dataset_candidate_source = 'direct_asym_eig'
all_head_pairwise_dataset_direct_asym_eig_mode = 'raw_right_largest_real_real'
all_head_pairwise_dataset_loss_batches = 4
all_head_pairwise_dataset_loss_batch_size = 16
all_head_pairwise_dataset_loss_candidate_batch_size = 4
all_head_pairwise_dataset_loss_prefix_k = 16
all_head_pairwise_dataset_loss_score = 'linear_profile'
all_head_pairwise_dataset_loss_exp_tau = 16.0
all_head_pairwise_dataset_store_pairwise_q = True
all_head_pairwise_dataset_resume_existing = False
head_direction_logger_enabled = False
head_direction_logger_interval = 0
head_direction_logger_start_iter = 0
head_direction_logger_stop_iter = -1
head_direction_logger_batch_size = 16
head_direction_logger_batches = 1
head_direction_logger_export_type = 'without_none'
head_direction_logger_frame = 'true_original'
head_direction_logger_wandb_prefix = 'head_direction'
head_direction_logger_log_grid = True
head_direction_logger_log_individual_maps = True
head_direction_logger_individual_interval = 0
head_direction_logger_cmap = 'viridis'
head_direction_logger_vmax_percentile = 99.5
head_direction_logger_save_latest = True
head_direction_logger_out_dir = ''
head_direction_logger_restore_rng = True
head_direction_logger_deterministic = False
head_direction_logger_seed = 24681357
head_asym_selector_enabled = False
head_asym_selector_target_iter = 10000
head_asym_selector_split = 'train'
head_asym_selector_batch_size = 32
head_asym_selector_batches = 4
head_asym_selector_export_type = 'without_none'
head_asym_selector_out_dir = ''
head_asym_selector_wandb_prefix = 'head_asym_selector'
head_asym_selector_log_wandb = True
head_asym_selector_save_matrices = True
head_asym_selector_save_all_head_maps = False
head_asym_selector_all_head_maps_dir = ''
head_asym_selector_all_head_maps_cmap = 'viridis'
head_asym_selector_all_head_maps_vmax_percentile = 99.5
head_asym_selector_selection_rule = 'asym_argmax' # asym_argmax, asym_topk_direct_asym_eig_loss, oracle_direct_asym_eig_original_tau
head_asym_selector_asym_top_k = 8
head_asym_selector_direct_asym_eig_mode = 'raw_right_largest_real_real'
head_asym_selector_loss_batches = 4
head_asym_selector_loss_batch_size = 16
head_asym_selector_loss_candidate_batch_size = 8
head_asym_selector_loss_prefix_k = 16
head_asym_selector_loss_score = 'linear_profile'
head_asym_selector_loss_exp_tau = 16.0
head_asym_selector_assign_to_attn_mlp_policy = False
head_asym_selector_init_attn_mlp_policy = False
head_asym_selector_restore_rng = True
head_asym_selector_deterministic = True
head_asym_selector_seed = 24681357
# -----------------------------------------------------------------------------

def _apply_config_overrides():
    for arg in sys.argv[1:]:
        if '=' not in arg:
            assert not arg.startswith('--')
            config_file = arg
            print(f"Overriding config with {config_file}:")
            with open(config_file, 'r', encoding='utf-8') as handle:
                config_text = handle.read()
            print(config_text)
            exec(config_text, globals())
        else:
            assert arg.startswith('--')
            key, val = arg.split('=', 1)
            key = key[2:]
            if key not in globals():
                raise ValueError(f"Unknown config key: {key}")
            try:
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                attempt = val
            if type(attempt) is not type(globals()[key]):
                raise TypeError(
                    f"Type mismatch for {key}: expected {type(globals()[key]).__name__}, "
                    f"got {type(attempt).__name__}"
                )
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
_apply_config_overrides()
config = {k: globals()[k] for k in config_keys} # will be useful for logging
config['main_eval_mode'] = main_eval_mode
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(int(seed) + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
effective_order_block_len = 1 if str(order_impl) == 'token' else int(block_order_block_len)
if block_size % effective_order_block_len != 0:
    raise ValueError(
        f"block_size={block_size} must be divisible by effective_order_block_len={effective_order_block_len}"
    )
num_blocks = block_size // effective_order_block_len

np.random.seed(permute_seed)
if permute_data:
    if permute_mode != 'block':
        raise ValueError(f"Unsupported permute_mode={permute_mode!r}. Only 'block' is supported.")
    fixed_block_perm = build_fixed_block_permutation(num_blocks, permute_seed)
    inverse_block_perm = invert_permutation(fixed_block_perm)
    fixed_token_perm = block_permutation_to_token_permutation(
        fixed_block_perm,
        block_len=effective_order_block_len,
        block_order_layout=block_order_layout,
        image_size=image_size,
        image_block_size=image_block_size,
        image_block_height=image_block_height,
        image_block_width=image_block_width,
    )
else:
    fixed_block_perm = None
    inverse_block_perm = None
    fixed_token_perm = None

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split, batch_size_override=None):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    local_batch_size = batch_size if batch_size_override is None else int(batch_size_override)
    if str(data_record_mode) == 'fixed':
        num_records = len(data) // block_size
        if num_records <= 0:
            raise ValueError(f"Dataset split {split!r} is shorter than one fixed block_size={block_size} record.")
        ix = torch.randint(num_records, (local_batch_size,))
        starts = (ix * block_size).tolist()
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in starts])
        y = x.clone()
    elif str(data_record_mode) == 'stream':
        ix = torch.randint(len(data) - block_size, (local_batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    else:
        raise ValueError(f"Unsupported data_record_mode={data_record_mode!r}. Expected 'stream' or 'fixed'.")
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    if permute_data:
        perm_idx = fixed_token_perm.to(device)
        x = x[:, perm_idx]
        y = y[:, perm_idx]
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9
resume_attn_mlp_policy_state = None
resume_online_spectral_policy_state = None

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout,
                  block_order_block_len=effective_order_block_len,
                  block_order_layout=block_order_layout,
                  image_size=image_size,
                  image_block_size=image_block_size,
                  image_block_height=image_block_height,
                  image_block_width=image_block_width,
                  order_impl=order_impl,
                  position_encoding_mode=position_encoding_mode,
                  rope_theta=rope_theta) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = AOGPTConfig(**model_args)
    model = AOGPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in [
        'n_layer',
        'n_head',
        'n_embd',
        'block_size',
        'bias',
        'vocab_size',
        'block_order_block_len',
        'order_impl',
    ]:
        model_args[k] = checkpoint_model_args[k]
    model_args['block_order_layout'] = checkpoint_model_args.get('block_order_layout', 'contiguous')
    model_args['image_size'] = checkpoint_model_args.get('image_size', 0)
    model_args['image_block_size'] = checkpoint_model_args.get('image_block_size', 0)
    model_args['image_block_height'] = checkpoint_model_args.get('image_block_height', 0)
    model_args['image_block_width'] = checkpoint_model_args.get('image_block_width', 0)
    model_args['position_encoding_mode'] = checkpoint_model_args.get('position_encoding_mode', 'absolute')
    model_args['rope_theta'] = checkpoint_model_args.get('rope_theta', 10000.0)
    # create the model
    gptconf = AOGPTConfig(**model_args)
    model = AOGPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    resume_attn_mlp_policy_state = checkpoint.get('attn_mlp_policy_state')
    resume_online_spectral_policy_state = checkpoint.get('online_spectral_policy_state')
elif init_from == 'ckpt':
    # continuation from an arbitrary parent checkpoint (shared-parent protocol):
    # load parent weights, CONTINUE the step counter, only variable = order policy.
    print(f"Initializing from parent ckpt {init_from_ckpt} (mode={init_from_ckpt_mode})")
    checkpoint = torch.load(init_from_ckpt, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size',
              'block_order_block_len', 'order_impl']:
        if k in checkpoint_model_args:
            model_args[k] = checkpoint_model_args[k]
    model_args['block_order_layout'] = checkpoint_model_args.get('block_order_layout', 'contiguous')
    model_args['image_size'] = checkpoint_model_args.get('image_size', 0)
    model_args['image_block_size'] = checkpoint_model_args.get('image_block_size', 0)
    model_args['image_block_height'] = checkpoint_model_args.get('image_block_height', 0)
    model_args['image_block_width'] = checkpoint_model_args.get('image_block_width', 0)
    model_args['position_encoding_mode'] = checkpoint_model_args.get('position_encoding_mode', 'absolute')
    model_args['rope_theta'] = checkpoint_model_args.get('rope_theta', 10000.0)
    gptconf = AOGPTConfig(**model_args)
    model = AOGPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']                 # continue the step counter
    best_val_loss = checkpoint.get('best_val_loss', 1e9)
    # weights_only continuation reinitialises the optimizer; full_state loads it below.
    resume_optimizer_state = (init_from_ckpt_mode == 'full_state')
# elif init_from.startswith('gpt2'):
#     print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
#     # initialize from OpenAI GPT-2 weights
#     override_args = dict(dropout=dropout)
#     model = GPT.from_pretrained(init_from, override_args)
#     # read off the created config params, so we can store them into checkpoint correctly
#     for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
#         model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

if train_stage != 'standard':
    raise ValueError(f"Unsupported train_stage={train_stage!r}. Only 'standard' is available.")


attn_mlp_policy = None
attn_mlp_policy_config = None
attn_mlp_policy_optimizer = None
attn_mlp_policy_A_ema = None
attn_mlp_policy_cached_order = None
attn_mlp_policy_last_logits = None
attn_mlp_policy_logits_ema = None
attn_mlp_policy_updates = 0
attn_mlp_policy_attention_updates = 0
attn_mlp_policy_fallback_count = 0
attn_mlp_policy_last_update_iter = -1
attn_mlp_policy_last_attention_iter = -1
attn_mlp_policy_input_attn_last_log_iter = -1
attn_mlp_policy_head_profile_last_iter = -1
attn_mlp_policy_head_profile_cached_q = None
attn_mlp_policy_head_profile_cached_order = None
attn_mlp_policy_head_profile_cached_stats = None
attn_mlp_policy_head_profile_best_q = None
attn_mlp_policy_head_profile_best_order = None
attn_mlp_policy_head_profile_best_iter = -1
attn_mlp_policy_latest_stats = None
attn_mlp_policy_last_train_stats = None
attn_mlp_policy_last_teacher_diag = None
attn_mlp_policy_current_lr = None
attn_mlp_policy_loss_stop_best = None
attn_mlp_policy_loss_stop_best_iter = -1
attn_mlp_policy_loss_stop_bad_steps = 0
attn_mlp_policy_loss_stop_checks = 0
attn_mlp_policy_loss_stop_triggered = False
attn_mlp_policy_loss_stop_iter = -1
attn_mlp_policy_loss_stop_last_metric = None
attn_mlp_policy_matrix_buffer = []
attn_mlp_policy_shadow_mse_items_current = None
attn_mlp_policy_shadow_mse_batch_buffer = []
attn_mlp_policy_shadow_mse_item_buffer = []


def _attn_mlp_policy_mode_enabled():
    return bool(attn_mlp_policy_enabled) or str(aogpt_train_mode) == "AttnMLPFrozenOrder"


def _parse_attn_mlp_hidden_dims(raw_value):
    if isinstance(raw_value, str):
        return [int(item.strip()) for item in raw_value.split(',') if item.strip()]
    if isinstance(raw_value, (list, tuple)):
        return [int(item) for item in raw_value]
    return [int(raw_value)]


def _attn_mlp_expected_input_channels():
    mode = str(attn_mlp_policy_feature_mode or "attention").lower()
    if mode in {"attention", "matrix", "single", "attention_direct", "direct_attention", "attention_only_direct"}:
        return 1
    if mode in {"attention_loss_direct", "attention_loss_no_sym", "try20_no_sym", "loss_planes_direct"}:
        return 2
    if mode in {"try20", "attention_loss", "attention_loss_features", "loss_planes"}:
        return 3
    raise ValueError(f"Unsupported attn_mlp_policy_feature_mode={attn_mlp_policy_feature_mode!r}")


def _build_random_attn_mlp_policy():
    expected_channels = _attn_mlp_expected_input_channels()
    input_channels = int(attn_mlp_policy_input_channels)
    if input_channels <= 0:
        input_channels = expected_channels
    if input_channels != expected_channels:
        raise ValueError(
            f"attn_mlp_policy_input_channels={input_channels} does not match "
            f"feature_mode={attn_mlp_policy_feature_mode!r} expected_channels={expected_channels}."
        )
    config_local = {
        "num_blocks": int(num_blocks),
        "input_channels": int(input_channels),
        "hidden_dims": _parse_attn_mlp_hidden_dims(attn_mlp_policy_hidden_dims),
        "dropout": float(attn_mlp_policy_dropout),
        "activation": str(attn_mlp_policy_activation),
        "input_normalization": str(attn_mlp_policy_input_normalization),
    }
    if input_channels == 1:
        model_config = dict(config_local)
        model_config.pop("input_channels", None)
        policy = FlatAttentionOrderMLP(**model_config)
    else:
        policy = FeatureAttentionOrderMLP(**config_local)
    return policy, config_local


def _configure_attn_mlp_policy_trainability():
    global attn_mlp_policy_optimizer, attn_mlp_policy_current_lr
    if attn_mlp_policy is None:
        return
    trainable = not bool(attn_mlp_policy_freeze)
    for param in attn_mlp_policy.parameters():
        param.requires_grad_(trainable)
    if trainable:
        attn_mlp_policy.train()
        attn_mlp_policy_current_lr = float(attn_mlp_policy_lr)
        attn_mlp_policy_optimizer = torch.optim.AdamW(
            attn_mlp_policy.parameters(),
            lr=float(attn_mlp_policy_current_lr),
            betas=(float(attn_mlp_policy_beta1), float(attn_mlp_policy_beta2)),
            weight_decay=float(attn_mlp_policy_weight_decay),
        )
    else:
        attn_mlp_policy.eval()
        attn_mlp_policy_optimizer = None
        attn_mlp_policy_current_lr = None


def _attn_mlp_policy_lr_for_iter(current_iter):
    base_lr = float(attn_mlp_policy_lr)
    if not bool(attn_mlp_policy_lr_anneal_enabled):
        return base_lr
    start = int(attn_mlp_policy_lr_anneal_start_iter)
    end = int(attn_mlp_policy_lr_anneal_end_iter)
    if start <= 0:
        start = int(attn_mlp_policy_start_iter)
    if end <= 0:
        end = int(attn_mlp_policy_update_stop_iter)
    if end <= start:
        return max(0.0, float(attn_mlp_policy_lr_anneal_min_lr))
    min_lr = max(0.0, float(attn_mlp_policy_lr_anneal_min_lr))
    if int(current_iter) <= start:
        return base_lr
    if int(current_iter) >= end:
        return min_lr
    ratio = (float(current_iter) - float(start)) / max(1.0, float(end - start))
    ratio = max(0.0, min(1.0, ratio))
    style = str(attn_mlp_policy_lr_anneal_style or "cosine").lower()
    if style in {"cosine", "cos"}:
        scale = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return min_lr + (base_lr - min_lr) * scale
    if style in {"linear", "lin"}:
        return base_lr + (min_lr - base_lr) * ratio
    raise ValueError(
        f"Unsupported attn_mlp_policy_lr_anneal_style={attn_mlp_policy_lr_anneal_style!r}"
    )


def _attn_mlp_set_optimizer_lr(current_iter):
    global attn_mlp_policy_current_lr
    if attn_mlp_policy_optimizer is None:
        attn_mlp_policy_current_lr = None
        return None
    lr_value = float(_attn_mlp_policy_lr_for_iter(current_iter))
    for group in attn_mlp_policy_optimizer.param_groups:
        group["lr"] = lr_value
    attn_mlp_policy_current_lr = lr_value
    return lr_value


def _attn_mlp_policy_updates_frozen_for_iter(current_iter):
    stop_iter = int(attn_mlp_policy_update_stop_iter)
    return stop_iter > 0 and int(current_iter) >= stop_iter


def _attn_mlp_policy_loss_stop_update(train_stats, current_iter):
    global attn_mlp_policy_loss_stop_best, attn_mlp_policy_loss_stop_best_iter
    global attn_mlp_policy_loss_stop_bad_steps, attn_mlp_policy_loss_stop_checks
    global attn_mlp_policy_loss_stop_triggered, attn_mlp_policy_loss_stop_iter
    global attn_mlp_policy_loss_stop_last_metric

    if not bool(attn_mlp_policy_loss_stop_enabled):
        return
    if bool(attn_mlp_policy_loss_stop_triggered):
        return
    if int(current_iter) < int(attn_mlp_policy_loss_stop_start_iter):
        return
    check_every = max(1, int(attn_mlp_policy_loss_stop_check_every))
    if int(attn_mlp_policy_updates) % check_every != 0:
        return
    metric_name = str(attn_mlp_policy_loss_stop_metric or "attn_mlp_train_loss")
    if not isinstance(train_stats, dict) or metric_name not in train_stats:
        return
    metric_value = float(train_stats[metric_name])
    if not math.isfinite(metric_value):
        return

    attn_mlp_policy_loss_stop_checks += 1
    attn_mlp_policy_loss_stop_last_metric = metric_value
    min_delta = max(0.0, float(attn_mlp_policy_loss_stop_min_delta))
    if attn_mlp_policy_loss_stop_best is None:
        attn_mlp_policy_loss_stop_best = metric_value
        attn_mlp_policy_loss_stop_best_iter = int(current_iter)
        attn_mlp_policy_loss_stop_bad_steps = 0
        return
    if metric_value < float(attn_mlp_policy_loss_stop_best) - min_delta:
        attn_mlp_policy_loss_stop_best = metric_value
        attn_mlp_policy_loss_stop_best_iter = int(current_iter)
        attn_mlp_policy_loss_stop_bad_steps = 0
        return

    attn_mlp_policy_loss_stop_bad_steps += 1
    if int(attn_mlp_policy_loss_stop_bad_steps) >= max(1, int(attn_mlp_policy_loss_stop_patience)):
        attn_mlp_policy_loss_stop_triggered = True
        attn_mlp_policy_loss_stop_iter = int(current_iter)
        if master_process:
            print(
                "attn-mlp loss-stop triggered "
                f"at iter {int(current_iter)}: metric={metric_name} "
                f"current={metric_value:.6f} best={float(attn_mlp_policy_loss_stop_best):.6f} "
                f"best_iter={int(attn_mlp_policy_loss_stop_best_iter)} "
                f"bad_steps={int(attn_mlp_policy_loss_stop_bad_steps)}"
            )


def _init_attn_mlp_policy():
    global attn_mlp_policy, attn_mlp_policy_config, attn_mlp_policy_optimizer
    global attn_mlp_policy_A_ema, attn_mlp_policy_cached_order, attn_mlp_policy_last_logits
    global attn_mlp_policy_logits_ema
    global attn_mlp_policy_updates, attn_mlp_policy_attention_updates, attn_mlp_policy_fallback_count
    global attn_mlp_policy_last_update_iter, attn_mlp_policy_last_attention_iter
    global attn_mlp_policy_input_attn_last_log_iter
    global attn_mlp_policy_head_profile_last_iter
    global attn_mlp_policy_head_profile_cached_q, attn_mlp_policy_head_profile_cached_order
    global attn_mlp_policy_head_profile_cached_stats
    global attn_mlp_policy_head_profile_best_q, attn_mlp_policy_head_profile_best_order
    global attn_mlp_policy_head_profile_best_iter
    global attn_mlp_policy_last_train_stats
    global attn_mlp_policy_last_teacher_diag
    global attn_mlp_policy_current_lr
    global attn_mlp_policy_loss_stop_best, attn_mlp_policy_loss_stop_best_iter
    global attn_mlp_policy_loss_stop_bad_steps, attn_mlp_policy_loss_stop_checks
    global attn_mlp_policy_loss_stop_triggered, attn_mlp_policy_loss_stop_iter
    global attn_mlp_policy_loss_stop_last_metric
    if not _attn_mlp_policy_mode_enabled():
        return
    if attn_mlp_policy is not None:
        return
    if str(aogpt_train_mode) != "AttnMLPFrozenOrder":
        raise ValueError(
            "attn_mlp_policy_enabled=True currently expects "
            "aogpt_train_mode='AttnMLPFrozenOrder'."
        )
    if bool(attn_mlp_policy_lazy_init_enabled) and not resume_attn_mlp_policy_state:
        lazy_iter = int(attn_mlp_policy_lazy_init_iter)
        if lazy_iter <= 0:
            lazy_iter = int(attn_mlp_policy_start_iter)
        if int(iter_num) < lazy_iter:
            if master_process:
                print(
                    "AttnMLPFrozenOrder lazy init pending: "
                    f"iter={int(iter_num)} lazy_init_iter={int(lazy_iter)} "
                    f"start_iter={int(attn_mlp_policy_start_iter)}"
                )
            return
    if bool(attn_mlp_policy_per_sample):
        raise ValueError("attn_mlp_policy_per_sample=True is not implemented in the first frozen global-order path.")
    if bool(attn_mlp_policy_random_init):
        attn_mlp_policy, attn_mlp_policy_config = _build_random_attn_mlp_policy()
        attn_mlp_policy.to(device)
    else:
        if not str(attn_mlp_policy_path).strip():
            raise ValueError(
                "AttnMLPFrozenOrder requires attn_mlp_policy_path unless "
                "attn_mlp_policy_random_init=True."
            )
        attn_mlp_policy, attn_mlp_policy_config = load_frozen_attn_mlp_policy(
            str(attn_mlp_policy_path),
            num_blocks=num_blocks,
            device=device,
            input_normalization=str(attn_mlp_policy_input_normalization),
        )
    _configure_attn_mlp_policy_trainability()
    if resume_attn_mlp_policy_state:
        state = resume_attn_mlp_policy_state
        if state.get("model_state_dict") is not None:
            attn_mlp_policy.load_state_dict(state["model_state_dict"], strict=True)
        elif init_from == 'resume' and not bool(attn_mlp_policy_freeze):
            raise ValueError(
                "Cannot resume a trainable AttnMLP policy without model_state_dict "
                "in checkpoint['attn_mlp_policy_state']."
            )
        if attn_mlp_policy_optimizer is not None and state.get("optimizer_state_dict") is not None:
            attn_mlp_policy_optimizer.load_state_dict(state["optimizer_state_dict"])
        if bool(attn_mlp_policy_attention_ema_enabled) and state.get("A_ema") is not None:
            attn_mlp_policy_A_ema = torch.as_tensor(state["A_ema"], dtype=torch.float32, device="cpu")
        if state.get("cached_order") is not None:
            attn_mlp_policy_cached_order = torch.as_tensor(state["cached_order"], dtype=torch.long, device="cpu")
        if state.get("last_logits") is not None:
            attn_mlp_policy_last_logits = torch.as_tensor(state["last_logits"], dtype=torch.float32, device="cpu")
        if state.get("logits_ema") is not None:
            attn_mlp_policy_logits_ema = torch.as_tensor(state["logits_ema"], dtype=torch.float32, device="cpu")
        attn_mlp_policy_updates = int(state.get("policy_updates", 0))
        attn_mlp_policy_attention_updates = int(state.get("attention_updates", 0))
        attn_mlp_policy_fallback_count = int(state.get("fallback_count", 0))
        attn_mlp_policy_last_update_iter = int(state.get("last_update_iter", -1))
        attn_mlp_policy_last_attention_iter = int(state.get("last_attention_iter", -1))
        attn_mlp_policy_input_attn_last_log_iter = int(state.get("input_attn_last_log_iter", -1))
        if state.get("current_lr") is not None:
            attn_mlp_policy_current_lr = float(state.get("current_lr"))
            if attn_mlp_policy_optimizer is not None:
                for group in attn_mlp_policy_optimizer.param_groups:
                    group["lr"] = float(attn_mlp_policy_current_lr)
        if state.get("loss_stop_best") is not None:
            attn_mlp_policy_loss_stop_best = float(state.get("loss_stop_best"))
        attn_mlp_policy_loss_stop_best_iter = int(state.get("loss_stop_best_iter", -1))
        attn_mlp_policy_loss_stop_bad_steps = int(state.get("loss_stop_bad_steps", 0))
        attn_mlp_policy_loss_stop_checks = int(state.get("loss_stop_checks", 0))
        attn_mlp_policy_loss_stop_triggered = bool(state.get("loss_stop_triggered", False))
        attn_mlp_policy_loss_stop_iter = int(state.get("loss_stop_iter", -1))
        if state.get("loss_stop_last_metric") is not None:
            attn_mlp_policy_loss_stop_last_metric = float(state.get("loss_stop_last_metric"))
        attn_mlp_policy_head_profile_last_iter = int(state.get("head_profile_last_iter", -1))
        if state.get("head_profile_cached_q") is not None:
            attn_mlp_policy_head_profile_cached_q = torch.as_tensor(
                state["head_profile_cached_q"],
                dtype=torch.float32,
                device="cpu",
            )
        if state.get("head_profile_cached_order") is not None:
            attn_mlp_policy_head_profile_cached_order = torch.as_tensor(
                state["head_profile_cached_order"],
                dtype=torch.long,
                device="cpu",
            )
        if state.get("head_profile_best_q") is not None:
            attn_mlp_policy_head_profile_best_q = torch.as_tensor(
                state["head_profile_best_q"],
                dtype=torch.float32,
                device="cpu",
            )
        if state.get("head_profile_best_order") is not None:
            attn_mlp_policy_head_profile_best_order = torch.as_tensor(
                state["head_profile_best_order"],
                dtype=torch.long,
                device="cpu",
            )
        attn_mlp_policy_head_profile_best_iter = int(state.get("head_profile_best_iter", -1))
        if isinstance(state.get("last_train_stats"), dict):
            attn_mlp_policy_last_train_stats = dict(state["last_train_stats"])
            attn_mlp_policy_head_profile_cached_stats = {
                key: value
                for key, value in attn_mlp_policy_last_train_stats.items()
                if str(key).startswith("attn_mlp_train_head_profile_")
            }
        if isinstance(state.get("last_teacher_diag"), dict):
            attn_mlp_policy_last_teacher_diag = dict(state["last_teacher_diag"])
    if master_process:
        restored_text = "restored cached state" if resume_attn_mlp_policy_state else "fresh cached state"
        print(
            "AttnMLPFrozenOrder enabled: "
            f"path={attn_mlp_policy_path}, start_iter={int(attn_mlp_policy_start_iter)}, "
            f"random_init={bool(attn_mlp_policy_random_init)}, trainable={not bool(attn_mlp_policy_freeze)}, "
            f"layer={int(attn_mlp_policy_layer)}, head={int(attn_mlp_policy_head)}, "
            f"use_global={bool(attn_mlp_policy_use_global)}, "
            f"feature_mode={str(attn_mlp_policy_feature_mode)}, "
            f"prob={float(attn_mlp_policy_start_prob):.3f}->{float(attn_mlp_policy_end_prob):.3f} "
            f"{int(attn_mlp_policy_anneal_start_iter)}..{int(attn_mlp_policy_anneal_end_iter)}, "
            f"prob_schedule={str(attn_mlp_policy_prob_schedule)}, "
            f"{restored_text}"
        )
        if str(attn_mlp_policy_prob_schedule).strip().lower() in {"piecewise", "points", "point", "schedule"}:
            print(f"AttnMLP prob points: {attn_mlp_policy_prob_points}")


def _attn_mlp_policy_active_for_iter(current_iter):
    return _attn_mlp_policy_mode_enabled() and int(current_iter) >= int(attn_mlp_policy_start_iter)


def _attn_mlp_policy_prob_for_iter(current_iter):
    def _clamp_prob(value):
        return max(0.0, min(1.0, float(value)))

    if not _attn_mlp_policy_active_for_iter(current_iter):
        return 0.0
    schedule = str(attn_mlp_policy_prob_schedule).strip().lower()
    if schedule in {"piecewise", "points", "point", "schedule"}:
        raw_points = attn_mlp_policy_prob_points
        parsed_points = []
        if isinstance(raw_points, str):
            for item in raw_points.split(","):
                item = item.strip()
                if not item:
                    continue
                if ":" not in item:
                    raise ValueError("attn_mlp_policy_prob_points entries must use iter:prob")
                iter_text, prob_text = item.split(":", 1)
                parsed_points.append((int(float(iter_text.strip())), _clamp_prob(prob_text.strip())))
        else:
            for item in raw_points:
                iter_value, prob_value = item
                parsed_points.append((int(iter_value), _clamp_prob(prob_value)))
        if not parsed_points:
            raise ValueError(
                "attn_mlp_policy_prob_schedule='piecewise' requires non-empty "
                "attn_mlp_policy_prob_points"
            )
        parsed_points = sorted(parsed_points, key=lambda item: int(item[0]))
        current = int(current_iter)
        if current <= int(parsed_points[0][0]):
            return _clamp_prob(parsed_points[0][1])
        for (left_iter, left_prob), (right_iter, right_prob) in zip(parsed_points[:-1], parsed_points[1:]):
            if current <= int(right_iter):
                if int(right_iter) <= int(left_iter):
                    return _clamp_prob(right_prob)
                ratio = (float(current) - float(left_iter)) / max(1.0, float(right_iter - left_iter))
                return _clamp_prob(float(left_prob) + ratio * (float(right_prob) - float(left_prob)))
        return _clamp_prob(parsed_points[-1][1])
    if schedule not in {"", "linear", "lin"}:
        raise ValueError(
            f"Unsupported attn_mlp_policy_prob_schedule={attn_mlp_policy_prob_schedule!r}; "
            "expected 'linear' or 'piecewise'."
        )
    start = int(attn_mlp_policy_anneal_start_iter)
    end = int(attn_mlp_policy_anneal_end_iter)
    start_prob = float(attn_mlp_policy_start_prob)
    end_prob = float(attn_mlp_policy_end_prob)
    if end <= start:
        return _clamp_prob(end_prob)
    if int(current_iter) <= start:
        return _clamp_prob(start_prob)
    if int(current_iter) >= end:
        return _clamp_prob(end_prob)
    ratio = (float(current_iter) - float(start)) / max(1.0, float(end - start))
    prob = start_prob + ratio * (end_prob - start_prob)
    return _clamp_prob(prob)


def _attn_mlp_policy_should_collect_attention(current_iter):
    if not _attn_mlp_policy_mode_enabled():
        return False
    if int(current_iter) < int(attn_mlp_policy_start_iter) and not bool(attn_mlp_policy_collect_warmup_attention):
        return False
    if (
        _attn_mlp_policy_shadow_mse_configured()
        and int(current_iter) < int(attn_mlp_policy_start_iter)
        and int(current_iter) < int(attn_mlp_policy_shadow_mse_start_iter)
    ):
        return False
    stop_iter = int(attn_mlp_policy_update_stop_iter)
    if stop_iter > 0 and int(current_iter) >= stop_iter:
        return False
    update_every = max(1, int(attn_mlp_policy_update_every))
    return int(current_iter) % update_every == 0


def _attn_mlp_policy_shadow_mse_loss_names():
    return {
        "teacher_score_mse",
        "shadow_teacher_score_mse",
        "shadow_fiedler_mse",
        "fiedler_score_mse",
    }


def _attn_mlp_policy_shadow_mse_configured():
    loss_name = str(attn_mlp_policy_train_loss or "none").strip().lower()
    return bool(attn_mlp_policy_shadow_mse_enabled) or loss_name in _attn_mlp_policy_shadow_mse_loss_names()


def _attn_mlp_policy_shadow_mse_grouped_items_enabled():
    return (
        int(attn_mlp_policy_shadow_mse_batches_per_item) > 1
        or int(attn_mlp_policy_shadow_mse_train_items_per_update) > 0
        or int(attn_mlp_policy_shadow_mse_val_items_per_update) > 0
    )


def _attn_mlp_policy_shadow_mse_fixed_sample_split_enabled():
    return (
        int(attn_mlp_policy_shadow_mse_train_samples_per_step) > 0
        or int(attn_mlp_policy_shadow_mse_val_samples_per_step) > 0
    )


def _attn_mlp_policy_shadow_mse_train_active(current_iter):
    if not _attn_mlp_policy_shadow_mse_configured():
        return False
    start_iter = int(attn_mlp_policy_shadow_mse_start_iter)
    stop_iter = int(attn_mlp_policy_shadow_mse_stop_iter)
    if int(current_iter) < start_iter:
        return False
    if stop_iter > 0 and int(current_iter) >= stop_iter:
        return False
    return True


def _attn_mlp_policy_shadow_mse_eval_after_stop_active(current_iter):
    if not _attn_mlp_policy_shadow_mse_configured():
        return False
    if not bool(attn_mlp_policy_shadow_mse_eval_after_stop_enabled):
        return False
    stop_iter = int(attn_mlp_policy_shadow_mse_stop_iter)
    if stop_iter <= 0 or int(current_iter) < stop_iter:
        return False
    update_stop = int(attn_mlp_policy_update_stop_iter)
    if update_stop > 0 and int(current_iter) >= update_stop:
        return False
    interval = max(1, int(attn_mlp_policy_shadow_mse_eval_after_stop_interval))
    return int(current_iter) % interval == 0


def _attn_mlp_cached_block_order(batch_size_local, device_local):
    if attn_mlp_policy_cached_order is None:
        return None
    order = attn_mlp_policy_cached_order.to(device=device_local, dtype=torch.long)
    return order.unsqueeze(0).expand(int(batch_size_local), -1)


def _attn_mlp_random_fallback_block_orders(batch_size_local, device_local):
    return sample_random_block_orders(
        batch_size=int(batch_size_local),
        num_blocks=num_blocks,
        device=device_local,
    )


def _attn_mlp_extract_attentions(outputs):
    for value in outputs[2:]:
        if isinstance(value, list) and value and torch.is_tensor(value[0]):
            return value
    return None


def _attn_mlp_attention_matrix_from_outputs(outputs, block_orders):
    attentions = _attn_mlp_extract_attentions(outputs)
    if not attentions:
        return None
    export_type = str(attn_mlp_policy_export_type)
    if bool(attn_mlp_policy_use_global):
        matrices = []
        for layer_attn in attentions:
            layer_heads = _aggregate_layerhead_attention_to_current_blocks(
                layer_attn.detach(),
                block_orders,
                export_type,
            )
            matrices.append(layer_heads.mean(dim=0))
        matrix = torch.stack(matrices, dim=0).mean(dim=0)
    else:
        layer_idx = int(attn_mlp_policy_layer)
        if layer_idx < 0:
            layer_idx = len(attentions) + layer_idx
        if layer_idx < 0 or layer_idx >= len(attentions):
            raise ValueError(
                f"attn_mlp_policy_layer={attn_mlp_policy_layer} is outside available layers 0..{len(attentions)-1}."
            )
        layer_heads = _aggregate_layerhead_attention_to_current_blocks(
            attentions[layer_idx].detach(),
            block_orders,
            export_type,
        )
        head_idx = int(attn_mlp_policy_head)
        if head_idx < 0:
            matrix = layer_heads.mean(dim=0)
        else:
            if head_idx >= int(layer_heads.size(0)):
                raise ValueError(
                    f"attn_mlp_policy_head={attn_mlp_policy_head} is outside available heads 0..{int(layer_heads.size(0))-1}."
                )
            matrix = layer_heads[head_idx]
    matrix = matrix.detach().float()
    matrix.fill_diagonal_(0.0)
    return matrix


def _attn_mlp_attention_sample_matrices_from_outputs(outputs, block_orders):
    attentions = _attn_mlp_extract_attentions(outputs)
    if not attentions or block_orders is None:
        return None
    export_type = str(attn_mlp_policy_export_type)
    if bool(attn_mlp_policy_use_global):
        matrices = []
        for layer_attn in attentions:
            layer_heads = _aggregate_layerhead_attention_to_current_blocks_per_sample(
                layer_attn.detach(),
                block_orders,
                export_type,
            )
            matrices.append(layer_heads.mean(dim=1))
        matrix = torch.stack(matrices, dim=0).mean(dim=0)
    else:
        layer_idx = int(attn_mlp_policy_layer)
        if layer_idx < 0:
            layer_idx = len(attentions) + layer_idx
        if layer_idx < 0 or layer_idx >= len(attentions):
            raise ValueError(
                f"attn_mlp_policy_layer={attn_mlp_policy_layer} is outside available layers 0..{len(attentions)-1}."
            )
        layer_heads = _aggregate_layerhead_attention_to_current_blocks_per_sample(
            attentions[layer_idx].detach(),
            block_orders,
            export_type,
        )
        head_idx = int(attn_mlp_policy_head)
        if head_idx < 0:
            matrix = layer_heads.mean(dim=1)
        else:
            if head_idx >= int(layer_heads.size(1)):
                raise ValueError(
                    f"attn_mlp_policy_head={attn_mlp_policy_head} is outside available heads 0..{int(layer_heads.size(1))-1}."
                )
            matrix = layer_heads[:, head_idx]
    matrix = matrix.detach().float()
    diag = torch.arange(int(matrix.size(-1)), device=matrix.device)
    matrix[:, diag, diag] = 0.0
    return matrix


def _attn_mlp_extract_token_losses(outputs):
    for value in outputs[2:]:
        if torch.is_tensor(value) and value.ndim >= 2:
            return value
    return None


def _attn_mlp_robust_z_offdiag(matrix, eps=1e-8):
    values = matrix.detach().float()
    n = int(values.size(0))
    eye = torch.eye(n, dtype=torch.bool, device=values.device)
    finite = torch.isfinite(values) & (~eye)
    out = torch.zeros_like(values, dtype=torch.float32)
    if not bool(finite.any()):
        return out
    vals = values[finite]
    median = torch.median(vals)
    q25 = torch.quantile(vals, 0.25)
    q75 = torch.quantile(vals, 0.75)
    scale = (q75 - q25) / 1.349
    if (not torch.isfinite(scale)) or float(scale.item()) < float(eps):
        scale = vals.std(unbiased=False)
    if (not torch.isfinite(scale)) or float(scale.item()) < float(eps):
        scale = vals.new_tensor(1.0)
    out[finite] = (vals - median) / scale
    out.fill_diagonal_(0.0)
    return out


def _attn_mlp_robust_z_vector(values, eps=1e-6):
    values = values.float()
    median = values.median(dim=-1, keepdim=True).values
    q25 = torch.quantile(values, 0.25, dim=-1, keepdim=True)
    q75 = torch.quantile(values, 0.75, dim=-1, keepdim=True)
    scale = ((q75 - q25) / 1.349).clamp_min(float(eps))
    return (values - median) / scale


def _attn_mlp_policy_input_from_outputs(outputs, block_orders):
    matrix = _attn_mlp_attention_matrix_from_outputs(outputs, block_orders)
    if matrix is None:
        return None
    mode = str(attn_mlp_policy_feature_mode or "attention").lower()
    if mode in {"attention", "matrix", "single"}:
        return matrix
    if mode in {"attention_direct", "direct_attention", "attention_only_direct"}:
        matrix = _attn_mlp_robust_z_offdiag(matrix)
        if float(attn_mlp_policy_feature_clip) > 0.0:
            clip = float(attn_mlp_policy_feature_clip)
            matrix = matrix.clamp(min=-clip, max=clip)
        return matrix
    two_channel_modes = {"attention_loss_direct", "attention_loss_no_sym", "try20_no_sym", "loss_planes_direct"}
    three_channel_modes = {"try20", "attention_loss", "attention_loss_features", "loss_planes"}
    if mode not in two_channel_modes and mode not in three_channel_modes:
        raise ValueError(f"Unsupported attn_mlp_policy_feature_mode={attn_mlp_policy_feature_mode!r}")
    token_losses = _attn_mlp_extract_token_losses(outputs)
    if token_losses is None:
        return None
    block_losses_reveal = token_losses_to_block_losses(
        token_losses.detach(),
        block_len=effective_order_block_len,
    ).float()
    current_losses = torch.empty_like(block_losses_reveal)
    current_losses.scatter_(1, block_orders.long(), block_losses_reveal)
    loss_vector = current_losses.mean(dim=0)
    a_z = _attn_mlp_robust_z_offdiag(matrix)
    sym = torch.maximum(a_z, a_z.t())
    loss_z = _attn_mlp_robust_z_vector(loss_vector.view(1, -1)).view(-1)
    loss_diff = loss_z[:, None] - loss_z[None, :]
    if mode in two_channel_modes:
        features = torch.stack([a_z, loss_diff], dim=0)
    else:
        features = torch.stack([a_z, sym, loss_diff], dim=0)
    if float(attn_mlp_policy_feature_clip) > 0.0:
        clip = float(attn_mlp_policy_feature_clip)
        features = features.clamp(min=-clip, max=clip)
    eye = torch.eye(int(features.size(-1)), dtype=torch.bool, device=features.device)
    features[:, eye] = 0.0
    return features


def _attn_mlp_plackett_luce_log_prob(logits, orders):
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    if orders.ndim == 1:
        orders = orders.unsqueeze(0)
    if logits.ndim != 2 or orders.ndim != 2:
        raise ValueError("Plackett-Luce log-prob expects logits/orders with shape [B, N].")
    ordered_logits = logits.float().gather(dim=1, index=orders.long())
    rev_lse = torch.flip(
        torch.logcumsumexp(torch.flip(ordered_logits, dims=[1]), dim=1),
        dims=[1],
    )
    return (ordered_logits - rev_lse).sum(dim=1)


def _attn_mlp_sample_pl_orders_and_logps(logits, num_samples):
    logits = logits.float()
    samples = []
    logps = []
    eps = 1e-6
    for _ in range(max(1, int(num_samples))):
        uniform = torch.rand_like(logits).clamp(eps, 1.0 - eps)
        gumbel = -torch.log(-torch.log(uniform))
        order = torch.argsort(logits + gumbel, dim=-1, descending=True)
        logp = _attn_mlp_plackett_luce_log_prob(logits, order)
        samples.append(order)
        logps.append(logp)
    return torch.stack(samples, dim=0).long(), torch.stack(logps, dim=0)


@torch.no_grad()
def _attn_mlp_full_and_prefix_loss_for_block_order(x, block_order, prefix_k=0):
    was_training = model.training
    model.eval()
    try:
        if block_order.ndim == 1:
            block_orders = block_order.to(device=x.device, dtype=torch.long).unsqueeze(0).expand(x.size(0), -1)
        else:
            block_orders = block_order.to(device=x.device, dtype=torch.long)
        with ctx:
            outputs = _forward_with_explicit_block_orders(
                x,
                block_orders,
                return_token_loss=True,
                return_logits=False,
            )
        token_losses = _attn_mlp_extract_token_losses(outputs)
        loss_value = outputs[1]
        if loss_value is None:
            if token_losses is None:
                raise RuntimeError("Expected scalar loss or token losses when evaluating AttnMLP order loss.")
            loss_value = token_losses.detach().float().mean()
        full_loss = float(loss_value.detach().float().mean().item())
        prefix_loss = full_loss
        if int(prefix_k) > 0:
            if token_losses is not None:
                block_losses = token_losses_to_block_losses(
                    token_losses.detach(),
                    block_len=effective_order_block_len,
                ).float()
                k = max(1, min(int(prefix_k), int(block_losses.size(1))))
                prefix_loss = float(block_losses[:, :k].mean().detach().float().item())
        return full_loss, prefix_loss
    finally:
        if was_training:
            model.train()


@torch.no_grad()
def _attn_mlp_block_loss_profile_for_block_order(x, block_order):
    was_training = model.training
    model.eval()
    try:
        if block_order.ndim == 1:
            block_orders = block_order.to(device=x.device, dtype=torch.long).unsqueeze(0).expand(x.size(0), -1)
        else:
            block_orders = block_order.to(device=x.device, dtype=torch.long)
        with ctx:
            outputs = _forward_with_explicit_block_orders(
                x,
                block_orders,
                return_token_loss=True,
                return_logits=False,
            )
        token_losses = _attn_mlp_extract_token_losses(outputs)
        if token_losses is None:
            raise RuntimeError("Expected token losses when evaluating AttnMLP order loss profile.")
        block_losses = token_losses_to_block_losses(
            token_losses.detach(),
            block_len=effective_order_block_len,
        ).float()
        return block_losses.mean(dim=0).detach().float()
    finally:
        if was_training:
            model.train()


def _attn_mlp_axis_profile_score_from_profile(profile):
    mode = str(attn_mlp_policy_axis_profile_score or "linear_profile").strip().lower()
    profile = profile.float()
    if mode in {"full", "full_loss", "mean"}:
        return profile.mean()
    if mode in {"prefix", "prefix_loss"}:
        k = max(1, min(int(attn_mlp_policy_prefix_k), int(profile.numel())))
        return profile[:k].mean()
    if mode in {"linear", "linear_profile", "linear_profile_loss"}:
        weights = torch.linspace(1.0, 0.1, steps=int(profile.numel()), device=profile.device)
        weights = weights / weights.sum().clamp_min(1e-8)
        return (weights * profile).sum()
    if mode in {"exp", "exp_profile", "exp_profile_loss"}:
        tau = max(float(head_signal_probe_candidate_loss_profile_exp_tau), 1e-6)
        idx = torch.arange(int(profile.numel()), dtype=torch.float32, device=profile.device)
        weights = torch.exp(-idx / tau)
        weights = weights / weights.sum().clamp_min(1e-8)
        return (weights * profile).sum()
    raise ValueError(f"Unsupported attn_mlp_policy_axis_profile_score={attn_mlp_policy_axis_profile_score!r}")


def _attn_mlp_zscore_1d(values):
    values = values.float()
    return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)


def _attn_mlp_soft_priority_from_logits(logits, tau):
    tau = max(float(tau), 1e-6)
    pair = (logits.float()[:, None] - logits.float()[None, :]) / tau
    return torch.sigmoid(pair).sum(dim=1)


def _attn_mlp_directed_ribbon_loss(policy_input):
    global attn_mlp_policy_last_train_stats
    if attn_mlp_policy is None or attn_mlp_policy_optimizer is None:
        return None
    if policy_input.ndim != 2:
        raise ValueError(
            "directed_ribbon expects a raw [N,N] attention matrix; "
            f"got shape={tuple(policy_input.shape)}."
        )
    matrix = policy_input.detach().float().to(device=device)
    if int(matrix.size(0)) != int(matrix.size(1)):
        raise ValueError(f"Expected square attention matrix, got shape={tuple(matrix.shape)}.")

    attn_mlp_policy.train()
    logits = attn_mlp_policy(matrix)
    n = int(logits.numel())
    eye = torch.eye(n, dtype=torch.bool, device=logits.device)

    flow = float(attn_mlp_policy_directed_ribbon_flow_sign) * (matrix - matrix.t())
    weights = torch.relu(flow).masked_fill(eye, 0.0)
    weight_sum_raw = weights.sum()
    if float(weight_sum_raw.detach().item()) <= 0.0:
        ribbon_loss = logits.new_tensor(0.0)
        side_loss = logits.new_tensor(float("nan"))
        band_loss = logits.new_tensor(float("nan"))
        active_edges = 0.0
        weight_mean = logits.new_tensor(0.0)
    else:
        weights = weights / weight_sum_raw.clamp_min(1e-8)
        priority = _attn_mlp_soft_priority_from_logits(
            logits,
            float(attn_mlp_policy_directed_ribbon_rank_tau),
        )
        # weights[i, j] means key block j should be earlier than query block i.
        delta = priority[None, :] - priority[:, None]
        side = F.softplus(float(attn_mlp_policy_directed_ribbon_margin) - delta)
        band = torch.relu(delta - float(attn_mlp_policy_directed_ribbon_band_width)).pow(2)
        side_loss = (weights * side).sum()
        band_loss = (weights * band).sum()
        ribbon_loss = side_loss + float(attn_mlp_policy_directed_ribbon_band_weight) * band_loss
        active_edges = float((weights > 0.0).float().sum().detach().item())
        weight_mean = weights[weights > 0.0].mean() if bool((weights > 0.0).any()) else logits.new_tensor(0.0)

    probs = torch.softmax(logits.float(), dim=-1)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum()
    logit_l2 = logits.float().pow(2).mean()
    logit_std = logits.float().std(unbiased=False)
    std_floor = torch.relu(logits.new_tensor(float(attn_mlp_policy_min_logit_std)) - logit_std).pow(2)
    entropy_floor = torch.relu(logits.new_tensor(float(attn_mlp_policy_min_entropy)) - entropy).pow(2)
    entropy_ceiling = torch.relu(entropy - logits.new_tensor(float(attn_mlp_policy_max_entropy))).pow(2)
    total_loss = (
        ribbon_loss
        + float(attn_mlp_policy_logit_l2) * logit_l2
        + float(attn_mlp_policy_std_floor_weight) * std_floor
        + float(attn_mlp_policy_entropy_floor_weight) * entropy_floor
        + float(attn_mlp_policy_entropy_ceiling_weight) * entropy_ceiling
    )

    attn_mlp_policy_optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    grad_norm = 0.0
    if float(attn_mlp_policy_grad_clip) > 0.0:
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
            attn_mlp_policy.parameters(),
            float(attn_mlp_policy_grad_clip),
        )
        grad_norm = float(grad_norm_tensor.item())
    attn_mlp_policy_optimizer.step()
    attn_mlp_policy.eval()

    with torch.no_grad():
        attn_mlp_policy_last_train_stats = {
            "attn_mlp_train_enabled": 1.0,
            "attn_mlp_train_loss": float(total_loss.detach().item()),
            "attn_mlp_train_directed_ribbon_loss": float(ribbon_loss.detach().item()),
            "attn_mlp_train_directed_ribbon_side_loss": float(side_loss.detach().item()),
            "attn_mlp_train_directed_ribbon_band_loss": float(band_loss.detach().item()),
            "attn_mlp_train_directed_ribbon_flow_sign": float(attn_mlp_policy_directed_ribbon_flow_sign),
            "attn_mlp_train_directed_ribbon_active_edges": float(active_edges),
            "attn_mlp_train_directed_ribbon_weight_mean": float(weight_mean.detach().item()),
            "attn_mlp_train_logit_l2": float(logit_l2.detach().item()),
            "attn_mlp_train_logit_std": float(logit_std.detach().item()),
            "attn_mlp_train_logit_entropy": float(entropy.detach().item()),
            "attn_mlp_train_std_floor": float(std_floor.detach().item()),
            "attn_mlp_train_entropy_floor": float(entropy_floor.detach().item()),
            "attn_mlp_train_entropy_ceiling": float(entropy_ceiling.detach().item()),
            "attn_mlp_train_grad_norm": float(grad_norm),
        }
    return attn_mlp_policy_last_train_stats


def _attn_mlp_axis_profile_simple_loss(policy_input, probe_batches):
    global attn_mlp_policy_last_train_stats
    if attn_mlp_policy is None or attn_mlp_policy_optimizer is None:
        return None
    if policy_input.ndim == 2:
        axis_matrix = policy_input.detach().float().to(device=device)
        model_input = axis_matrix
    elif policy_input.ndim == 3:
        axis_matrix = policy_input[0].detach().float().to(device=device)
        model_input = policy_input.detach().float().to(device=device)
    else:
        raise ValueError(
            "axis_profile_simple requires either a [N,N] attention matrix or "
            "a [C,N,N] feature tensor."
        )
    if int(axis_matrix.size(0)) != int(axis_matrix.size(1)):
        raise ValueError(f"Expected square axis matrix, got shape={tuple(axis_matrix.shape)}.")

    attn_mlp_policy.train()
    logits = attn_mlp_policy(model_input)
    n = int(logits.numel())
    pair_i, pair_j = torch.triu_indices(n, n, offset=1, device=logits.device)

    q_matrix = axis_matrix - axis_matrix.t()
    q_pair = q_matrix[pair_i, pair_j].float()
    d_pair = (logits[pair_i] - logits[pair_j]).float()
    valid = torch.isfinite(q_pair) & torch.isfinite(d_pair)
    min_abs_q = max(0.0, float(attn_mlp_policy_axis_profile_min_abs_q))
    if min_abs_q > 0.0:
        valid = valid & (q_pair.abs() >= min_abs_q)
    if bool(valid.any()):
        q_valid = q_pair[valid]
        d_valid = d_pair[valid]
        weights = q_valid.abs().clamp_min(1e-6)
        weights = weights / weights.sum().clamp_min(1e-8)
        q_mean = (weights * q_valid).sum()
        d_mean = (weights * d_valid).sum()
        q_centered = q_valid - q_mean
        d_centered = d_valid - d_mean
        cov = (weights * q_centered * d_centered).sum()
        q_var = (weights * q_centered.pow(2)).sum()
        d_var = (weights * d_centered.pow(2)).sum()
        axis_corr = cov / (q_var * d_var).sqrt().clamp_min(1e-8)
        axis_loss = 1.0 - axis_corr.abs()
        axis_pair_acc = ((torch.sign(d_valid.detach()) == torch.sign(axis_corr.detach() * q_valid)).float() * weights).sum()
        q_abs_mean = q_valid.abs().mean()
        q_abs_max = q_valid.abs().max()
        pair_count = float(q_valid.numel())
    else:
        axis_corr = logits.new_tensor(float("nan"))
        axis_loss = logits.new_tensor(0.0)
        axis_pair_acc = logits.new_tensor(float("nan"))
        q_abs_mean = logits.new_tensor(float("nan"))
        q_abs_max = logits.new_tensor(float("nan"))
        pair_count = 0.0

    map_order = logits_to_order(logits.detach(), mode=str(attn_mlp_policy_order_mode)).to(
        device=device,
        dtype=torch.long,
    )
    reverse_order = torch.flip(map_order, dims=[0])
    dir_targets = []
    dir_margins = []
    dir_abs_margins = []
    dir_map_scores = []
    dir_reverse_scores = []
    dir_forward_better = []
    dir_pair_accs = []
    margin_threshold = max(0.0, float(attn_mlp_policy_axis_profile_dir_margin))
    for x in list(probe_batches or [])[: max(1, int(attn_mlp_policy_nll_states_per_update))]:
        x = x.detach().to(device=device, dtype=torch.long)
        map_profile = _attn_mlp_block_loss_profile_for_block_order(x, map_order)
        reverse_profile = _attn_mlp_block_loss_profile_for_block_order(x, reverse_order)
        map_score = float(_attn_mlp_axis_profile_score_from_profile(map_profile).detach().cpu().item())
        reverse_score = float(_attn_mlp_axis_profile_score_from_profile(reverse_profile).detach().cpu().item())
        margin = reverse_score - map_score
        dir_margins.append(float(margin))
        dir_abs_margins.append(abs(float(margin)))
        dir_map_scores.append(float(map_score))
        dir_reverse_scores.append(float(reverse_score))
        dir_forward_better.append(float(margin > 0.0))
        if abs(float(margin)) < margin_threshold:
            continue
        target_order = map_order if margin > 0.0 else reverse_order
        rank = torch.empty(n, dtype=torch.float32, device=logits.device)
        rank[target_order] = torch.arange(n, dtype=torch.float32, device=logits.device)
        priority = 1.0 - rank / max(1.0, float(n - 1))
        dir_targets.append(priority)
        target_pair = (rank[pair_i] < rank[pair_j]).float()
        pred_pair = (logits[pair_i].detach() > logits[pair_j].detach()).float()
        dir_pair_accs.append(float((pred_pair == target_pair).float().mean().detach().cpu().item()))

    if dir_targets:
        target_priority = torch.stack(dir_targets, dim=0).mean(dim=0)
        dir_loss = F.mse_loss(_attn_mlp_zscore_1d(logits.float()), _attn_mlp_zscore_1d(target_priority))
        dir_pair_acc = float(np.mean(dir_pair_accs)) if dir_pair_accs else float("nan")
    else:
        dir_loss = logits.new_tensor(0.0)
        dir_pair_acc = float("nan")

    probs = torch.softmax(logits.float(), dim=-1)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum()
    logit_l2 = logits.float().pow(2).mean()
    logit_std = logits.float().std(unbiased=False)
    std_floor = torch.relu(logits.new_tensor(float(attn_mlp_policy_min_logit_std)) - logit_std).pow(2)
    entropy_floor = torch.relu(logits.new_tensor(float(attn_mlp_policy_min_entropy)) - entropy).pow(2)
    entropy_ceiling = torch.relu(entropy - logits.new_tensor(float(attn_mlp_policy_max_entropy))).pow(2)

    total_loss = (
        float(attn_mlp_policy_axis_profile_weight) * axis_loss
        + float(attn_mlp_policy_axis_profile_dir_weight) * dir_loss
        + float(attn_mlp_policy_logit_l2) * logit_l2
        + float(attn_mlp_policy_std_floor_weight) * std_floor
        + float(attn_mlp_policy_entropy_floor_weight) * entropy_floor
        + float(attn_mlp_policy_entropy_ceiling_weight) * entropy_ceiling
    )

    attn_mlp_policy_optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    grad_norm = 0.0
    if float(attn_mlp_policy_grad_clip) > 0.0:
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
            attn_mlp_policy.parameters(),
            float(attn_mlp_policy_grad_clip),
        )
        grad_norm = float(grad_norm_tensor.item())
    attn_mlp_policy_optimizer.step()
    attn_mlp_policy.eval()

    with torch.no_grad():
        attn_mlp_policy_last_train_stats = {
            "attn_mlp_train_enabled": 1.0,
            "attn_mlp_train_loss": float(total_loss.detach().item()),
            "attn_mlp_train_axis_profile_axis_loss": float(axis_loss.detach().item()),
            "attn_mlp_train_axis_profile_dir_loss": float(dir_loss.detach().item()),
            "attn_mlp_train_axis_profile_axis_weight": float(attn_mlp_policy_axis_profile_weight),
            "attn_mlp_train_axis_profile_dir_weight": float(attn_mlp_policy_axis_profile_dir_weight),
            "attn_mlp_train_axis_profile_corr": float(axis_corr.detach().item()),
            "attn_mlp_train_axis_profile_abs_corr": float(axis_corr.detach().abs().item()),
            "attn_mlp_train_axis_profile_pair_acc": float(axis_pair_acc.detach().item()),
            "attn_mlp_train_axis_profile_pair_count": float(pair_count),
            "attn_mlp_train_axis_profile_q_abs_mean": float(q_abs_mean.detach().item()),
            "attn_mlp_train_axis_profile_q_abs_max": float(q_abs_max.detach().item()),
            "attn_mlp_train_axis_profile_dir_margin_mean": float(np.mean(dir_margins)) if dir_margins else float("nan"),
            "attn_mlp_train_axis_profile_dir_margin_abs_mean": float(np.mean(dir_abs_margins)) if dir_abs_margins else float("nan"),
            "attn_mlp_train_axis_profile_dir_accept_rate": float(len(dir_targets)) / float(max(1, len(dir_margins))),
            "attn_mlp_train_axis_profile_dir_forward_better_frac": (
                float(np.mean(dir_forward_better)) if dir_forward_better else float("nan")
            ),
            "attn_mlp_train_axis_profile_dir_map_score": float(np.mean(dir_map_scores)) if dir_map_scores else float("nan"),
            "attn_mlp_train_axis_profile_dir_reverse_score": (
                float(np.mean(dir_reverse_scores)) if dir_reverse_scores else float("nan")
            ),
            "attn_mlp_train_axis_profile_dir_pair_acc": float(dir_pair_acc),
            "attn_mlp_train_logit_l2": float(logit_l2.detach().item()),
            "attn_mlp_train_logit_std": float(logit_std.detach().item()),
            "attn_mlp_train_logit_entropy": float(entropy.detach().item()),
            "attn_mlp_train_std_floor": float(std_floor.detach().item()),
            "attn_mlp_train_entropy_floor": float(entropy_floor.detach().item()),
            "attn_mlp_train_entropy_ceiling": float(entropy_ceiling.detach().item()),
            "attn_mlp_train_grad_norm": float(grad_norm),
        }
    return attn_mlp_policy_last_train_stats


def _attn_mlp_attention_pair_losses(logits, features):
    zero = logits.new_tensor(0.0)
    nan = logits.new_tensor(float("nan"))
    if features is None or features.ndim != 3 or int(features.size(1)) != int(features.size(2)):
        return zero, zero, {
            "attn_mlp_train_attn_pair_loss": float("nan"),
            "attn_mlp_train_attn_pair_acc": float("nan"),
            "attn_mlp_train_attn_pair_count": 0.0,
            "attn_mlp_train_attn_pair_weight_mean": 0.0,
            "attn_mlp_train_attn_close_loss": float("nan"),
            "attn_mlp_train_attn_close_dist_mean": float("nan"),
        }

    n = int(features.size(-1))
    if n <= 1:
        return zero, zero, {
            "attn_mlp_train_attn_pair_loss": float("nan"),
            "attn_mlp_train_attn_pair_acc": float("nan"),
            "attn_mlp_train_attn_pair_count": 0.0,
            "attn_mlp_train_attn_pair_weight_mean": 0.0,
            "attn_mlp_train_attn_close_loss": float("nan"),
            "attn_mlp_train_attn_close_dist_mean": float("nan"),
        }

    # Channel 0 is directed robust-z attention; high A[i, j] means block i
    # attends to block j. This is a current-frame internal signal, not an
    # original-order label.
    scores = features[0].detach().float().to(device=logits.device)
    eye = torch.eye(n, dtype=torch.bool, device=scores.device)
    valid = torch.isfinite(scores) & (~eye)
    values = scores[valid]
    if values.numel() == 0:
        return zero, zero, {
            "attn_mlp_train_attn_pair_loss": float("nan"),
            "attn_mlp_train_attn_pair_acc": float("nan"),
            "attn_mlp_train_attn_pair_count": 0.0,
            "attn_mlp_train_attn_pair_weight_mean": 0.0,
            "attn_mlp_train_attn_close_loss": float("nan"),
            "attn_mlp_train_attn_close_dist_mean": float("nan"),
        }

    top_frac = float(attn_mlp_policy_attn_pair_top_frac)
    if 0.0 < top_frac < 1.0:
        threshold = torch.quantile(values.float(), max(0.0, min(1.0, 1.0 - top_frac)))
    else:
        threshold = scores.new_tensor(float(attn_mlp_policy_attn_pair_min_z))
    raw_weights = (scores - threshold).clamp_min(0.0)
    raw_weights = raw_weights.masked_fill(~valid, 0.0)

    if float(raw_weights.sum().detach().item()) <= 0.0:
        k = max(1, int(round(float(values.numel()) * max(min(top_frac, 1.0), 0.0))))
        if k <= 0:
            k = max(1, min(int(values.numel()), n))
        k = min(k, int(values.numel()))
        flat_valid_idx = torch.nonzero(valid.reshape(-1), as_tuple=False).view(-1)
        top_idx = torch.topk(values.float(), k=k, largest=True).indices
        raw_weights = torch.zeros_like(scores)
        raw_weights.reshape(-1)[flat_valid_idx[top_idx]] = 1.0

    max_weight = max(float(attn_mlp_policy_attn_pair_max_weight), 0.0)
    if max_weight > 0.0:
        raw_weights = raw_weights.clamp_max(max_weight)
    rows, cols = torch.where(raw_weights > 0.0)
    if rows.numel() == 0:
        return zero, zero, {
            "attn_mlp_train_attn_pair_loss": float("nan"),
            "attn_mlp_train_attn_pair_acc": float("nan"),
            "attn_mlp_train_attn_pair_count": 0.0,
            "attn_mlp_train_attn_pair_weight_mean": 0.0,
            "attn_mlp_train_attn_close_loss": float("nan"),
            "attn_mlp_train_attn_close_dist_mean": float("nan"),
        }

    logits_f = logits.float()
    weights = raw_weights[rows, cols].float()
    pair_tau = max(float(attn_mlp_policy_attn_pair_tau), 1e-6)
    # Descending logits define earlier reveal positions. If i attends to j,
    # prefer j to have the larger logit and therefore appear earlier.
    pref_logits = (logits_f[cols] - logits_f[rows]) / pair_tau
    pair_targets = torch.ones_like(pref_logits)
    pair_losses = F.binary_cross_entropy_with_logits(pref_logits, pair_targets, reduction="none")
    pair_loss = (pair_losses * weights).sum() / weights.sum().clamp_min(1e-8)
    pair_acc = ((pref_logits.detach() > 0.0).float() * weights).sum() / weights.sum().clamp_min(1e-8)

    close_tau = max(float(attn_mlp_policy_attn_close_tau), 1e-6)
    rank_logits = (logits_f.unsqueeze(0) - logits_f.unsqueeze(1)) / close_tau
    soft_ranks = torch.sigmoid(rank_logits).sum(dim=1)
    rank_dist = (soft_ranks[rows] - soft_ranks[cols]).abs() / float(max(1, n - 1))
    close_margin = max(float(attn_mlp_policy_attn_close_margin), 0.0)
    close_penalty = torch.relu(rank_dist - close_margin).pow(2)
    close_loss = (close_penalty * weights).sum() / weights.sum().clamp_min(1e-8)
    close_dist_mean = (rank_dist.detach() * weights).sum() / weights.sum().clamp_min(1e-8)

    stats = {
        "attn_mlp_train_attn_pair_loss": float(pair_loss.detach().item()),
        "attn_mlp_train_attn_pair_acc": float(pair_acc.detach().item()),
        "attn_mlp_train_attn_pair_count": float(rows.numel()),
        "attn_mlp_train_attn_pair_weight_mean": float(weights.detach().mean().item()),
        "attn_mlp_train_attn_close_loss": float(close_loss.detach().item()),
        "attn_mlp_train_attn_close_dist_mean": float(close_dist_mean.detach().item()),
    }
    return pair_loss, close_loss, stats


def _attn_mlp_sync_attention_matrix(matrix):
    if ddp:
        dist.all_reduce(matrix, op=dist.ReduceOp.SUM)
        matrix = matrix / float(ddp_world_size)
    return matrix


def _attn_mlp_policy_matrix_for_update(step_matrices):
    global attn_mlp_policy_matrix_buffer
    if not step_matrices:
        return None
    step_matrix = torch.stack(step_matrices, dim=0).mean(dim=0).detach().float()
    batches_per_update = max(1, int(attn_mlp_policy_attention_batches_per_update))
    if batches_per_update <= 1:
        return step_matrix

    step_device = step_matrix.device
    attn_mlp_policy_matrix_buffer.append(step_matrix.cpu())
    if len(attn_mlp_policy_matrix_buffer) < batches_per_update:
        return None

    selected = [attn_mlp_policy_matrix_buffer.pop(0) for _ in range(batches_per_update)]
    return torch.stack(selected, dim=0).mean(dim=0).to(device=step_device)


def _attn_mlp_head_profile_due(current_iter):
    if float(attn_mlp_policy_head_profile_weight) <= 0.0:
        return False
    if int(current_iter) < int(attn_mlp_policy_head_profile_start_iter):
        return False
    every = int(attn_mlp_policy_head_profile_every)
    if int(attn_mlp_policy_head_profile_last_iter) < 0:
        return True
    if every <= 0:
        return True
    return int(current_iter) - int(attn_mlp_policy_head_profile_last_iter) >= every


@torch.no_grad()
def _attn_mlp_compute_head_profile_consensus_target(current_iter):
    heads = _head_signal_probe_heads()
    if not heads:
        return None

    was_training = model.training
    model.eval()
    try:
        probe_batch_size = max(1, int(head_signal_probe_batch_size))
        matrix_sums = None
        total_samples = 0
        for _ in range(max(1, int(head_signal_probe_batches))):
            X_probe, _ = get_batch('train', batch_size_override=probe_batch_size)
            block_orders = sample_random_block_orders(
                batch_size=int(X_probe.size(0)),
                num_blocks=num_blocks,
                device=X_probe.device,
            )
            with ctx:
                outputs = _forward_with_explicit_block_orders(
                    X_probe,
                    block_orders,
                    return_attentions=True,
                    return_logits=False,
                )
            attentions = _attn_mlp_extract_attentions(outputs)
            if not attentions:
                continue
            layer_matrices = []
            for layer_attn in attentions:
                layer_matrices.append(
                    _aggregate_layerhead_attention_to_current_blocks(
                        layer_attn.detach(),
                        block_orders,
                        str(head_signal_probe_export_type),
                    ).detach().cpu()
                )
            matrices = torch.stack(layer_matrices, dim=0).to(dtype=torch.float64, device="cpu")
            if matrix_sums is None:
                matrix_sums = matrices * float(X_probe.size(0))
            else:
                matrix_sums += matrices * float(X_probe.size(0))
            total_samples += int(X_probe.size(0))

        if matrix_sums is None or total_samples <= 0:
            return None
        matrices = matrix_sums / float(total_samples)
        for layer_idx in range(matrices.size(0)):
            for head_idx in range(matrices.size(1)):
                matrices[layer_idx, head_idx].fill_diagonal_(0.0)

        recovery_config = FixedHeadSpectralPolicyConfig(
            num_components=int(head_signal_probe_num_components),
            component_pairs=str(head_signal_probe_component_pairs),
            num_angles=int(head_signal_probe_num_angles),
            k_values=str(head_signal_probe_k_values),
            group_methods=str(head_signal_probe_group_methods),
            threshold_percentile=float(head_signal_probe_threshold_percentile),
            transform=str(head_signal_probe_transform),
            temperature=float(head_signal_probe_temperature),
            direction_lambdas=str(head_signal_probe_direction_lambdas),
            directed_score_weight=float(head_signal_probe_directed_score_weight),
            band_quality_weight=float(head_signal_probe_band_quality_weight),
            score_adjacency_sym=str(head_signal_probe_score_adjacency_sym),
        )

        rows = []
        order_loss_requests = []
        max_rank = max(1, int(head_signal_probe_candidate_loss_profile_max_rank))
        for layer_idx, head_idx in heads:
            head_label = _head_signal_head_label(head_idx)
            row = {
                "layer": int(layer_idx),
                "head": int(head_idx),
                "head_label": str(head_label),
                "head_aggregation": "mean" if int(head_idx) == -1 else "single",
                "iter": int(current_iter),
            }
            try:
                candidates = _head_signal_recover_candidates(
                    _head_signal_probe_matrix_for_head(matrices, layer_idx, head_idx),
                    recovery_config,
                )
                candidate_orders = []
                candidate_seen = set()
                for candidate in candidates[:max_rank]:
                    candidate_order = [int(value) for value in candidate["order"]]
                    variants = (
                        [candidate_order, list(reversed(candidate_order))]
                        if bool(head_signal_probe_candidate_loss_profile_include_reverse)
                        else [candidate_order]
                    )
                    for order_variant in variants:
                        key = tuple(int(value) for value in order_variant)
                        if key in candidate_seen:
                            continue
                        candidate_seen.add(key)
                        candidate_orders.append(list(key))
                row["_head_signal_candidate_orders_current"] = candidate_orders
                order_loss_requests.extend(candidate_orders)
            except Exception as exc:
                row["error"] = str(exc)
            rows.append(row)

        loss_by_order = _head_signal_evaluate_order_losses(order_loss_requests)
        _head_signal_apply_candidate_loss_profile_consensus(rows, loss_by_order)

        q_matrices = []
        q_weights = []
        score_gaps = []
        alignments = []
        fallback_count = 0
        tau_values = []
        for row in rows:
            selected = row.get("consensus_order_current")
            if not selected:
                continue
            score_gap = float(row.get("loss_profile_score_gap", 0.0))
            align = float(row.get("loss_profile_consensus_alignment", 0.0))
            used_fallback = bool(row.get("loss_profile_used_fallback", False))
            weight = max(score_gap, float(head_signal_probe_candidate_loss_profile_min_gap), 1e-4)
            q_matrices.append(_head_signal_precedence_matrix(selected))
            q_weights.append(float(weight))
            score_gaps.append(float(score_gap))
            alignments.append(float(align))
            fallback_count += int(used_fallback)
            if "consensus_tau_diagnostic" in row:
                tau_values.append(float(row["consensus_tau_diagnostic"]))

        if not q_matrices:
            return None
        denom = float(sum(q_weights))
        q_matrix = sum(float(w) * matrix for w, matrix in zip(q_weights, q_matrices)) / float(max(denom, 1e-8))
        np.fill_diagonal(q_matrix, 0.0)
        target_order = _head_signal_order_from_precedence_matrix(q_matrix)
        target_original = _head_signal_order_to_original(target_order)
        target_tau = _head_signal_tau_to_l2r(target_original)
        q_abs = np.abs(q_matrix[np.triu_indices(int(num_blocks), k=1)])
        stats = {
            "attn_mlp_train_head_profile_refreshed": 1.0,
            "attn_mlp_train_head_profile_last_iter": float(current_iter),
            "attn_mlp_train_head_profile_heads": float(len(q_matrices)),
            "attn_mlp_train_head_profile_samples": float(total_samples),
            "attn_mlp_train_head_profile_score_gap_mean": float(np.mean(score_gaps)) if score_gaps else float("nan"),
            "attn_mlp_train_head_profile_alignment_mean": float(np.mean(alignments)) if alignments else float("nan"),
            "attn_mlp_train_head_profile_fallback_frac": float(fallback_count) / float(max(1, len(q_matrices))),
            "attn_mlp_train_head_profile_q_abs_mean": float(np.mean(q_abs)) if q_abs.size else float("nan"),
            "attn_mlp_train_head_profile_q_abs_max": float(np.max(q_abs)) if q_abs.size else float("nan"),
            "attn_mlp_train_head_profile_tau_diagnostic": float(target_tau),
            "attn_mlp_train_head_profile_abs_tau_diagnostic": abs(float(target_tau)),
            "attn_mlp_train_head_profile_head_tau_mean_diagnostic": (
                float(np.mean(tau_values)) if tau_values else float("nan")
            ),
        }
        if master_process:
            first16 = [int(v) for v in target_order[:16]]
            print(
                "attn-mlp head-profile target "
                f"at iter {int(current_iter)}: heads={len(q_matrices)} "
                f"q_abs_mean={stats['attn_mlp_train_head_profile_q_abs_mean']:.4f} "
                f"tau_diag={float(target_tau):+.4f} first16={first16}"
            )
        q_tensor = torch.tensor(q_matrix, dtype=torch.float32, device="cpu")
        order_tensor = torch.tensor(target_order, dtype=torch.long, device="cpu")
        return q_tensor, order_tensor, stats
    finally:
        if was_training:
            model.train()


@torch.no_grad()
def _attn_mlp_head_profile_accept_new_target(q_tensor, order_tensor, current_iter):
    global attn_mlp_policy_head_profile_best_q
    global attn_mlp_policy_head_profile_best_order
    global attn_mlp_policy_head_profile_best_iter

    stats = {
        "attn_mlp_train_head_profile_gate_enabled": float(
            bool(attn_mlp_policy_head_profile_accept_gate_enabled)
        ),
        "attn_mlp_train_head_profile_gate_checked": 0.0,
        "attn_mlp_train_head_profile_gate_accepted": 1.0,
        "attn_mlp_train_head_profile_gate_alignment": float("nan"),
        "attn_mlp_train_head_profile_gate_new_score": float("nan"),
        "attn_mlp_train_head_profile_gate_cached_score": float("nan"),
        "attn_mlp_train_head_profile_gate_best_score": float("nan"),
        "attn_mlp_train_head_profile_gate_selected_score": float("nan"),
        "attn_mlp_train_head_profile_gate_score_delta": float("nan"),
        "attn_mlp_train_head_profile_gate_margin": float(
            attn_mlp_policy_head_profile_accept_margin
        ),
        "attn_mlp_train_head_profile_gate_selected_source_id": 0.0,
        "attn_mlp_train_head_profile_gate_selected_is_new": 1.0,
        "attn_mlp_train_head_profile_gate_selected_is_cached": 0.0,
        "attn_mlp_train_head_profile_gate_selected_is_best": 0.0,
        "attn_mlp_train_head_profile_best_memory_enabled": float(
            bool(attn_mlp_policy_head_profile_best_memory_enabled)
        ),
        "attn_mlp_train_head_profile_best_memory_available": float(
            attn_mlp_policy_head_profile_best_order is not None
        ),
        "attn_mlp_train_head_profile_best_memory_updated": 0.0,
        "attn_mlp_train_head_profile_best_memory_iter": float(
            attn_mlp_policy_head_profile_best_iter
        ),
    }
    source_ids = {"new": 0.0, "cached": 1.0, "best": 2.0}

    def _as_order_list(tensor):
        return [int(v) for v in tensor.detach().cpu().view(-1).tolist()]

    def _order_is_valid(order):
        return len(order) == int(num_blocks) and sorted(order) == list(range(int(num_blocks)))

    def _q_from_order(order):
        return torch.tensor(
            _head_signal_precedence_matrix(order),
            dtype=torch.float32,
            device="cpu",
        )

    def _select_stats(source, score, new_score, cached_score, best_score):
        selected_source = str(source)
        stats.update(
            {
                "attn_mlp_train_head_profile_gate_accepted": float(selected_source == "new"),
                "attn_mlp_train_head_profile_gate_new_score": float(new_score),
                "attn_mlp_train_head_profile_gate_cached_score": float(cached_score),
                "attn_mlp_train_head_profile_gate_best_score": float(best_score),
                "attn_mlp_train_head_profile_gate_selected_score": float(score),
                "attn_mlp_train_head_profile_gate_selected_source_id": float(
                    source_ids.get(selected_source, -1.0)
                ),
                "attn_mlp_train_head_profile_gate_selected_is_new": float(selected_source == "new"),
                "attn_mlp_train_head_profile_gate_selected_is_cached": float(selected_source == "cached"),
                "attn_mlp_train_head_profile_gate_selected_is_best": float(selected_source == "best"),
            }
        )

    if not bool(attn_mlp_policy_head_profile_accept_gate_enabled):
        return True, stats, q_tensor.detach().cpu(), order_tensor.detach().cpu()
    if (
        attn_mlp_policy_head_profile_cached_q is None
        or attn_mlp_policy_head_profile_cached_order is None
    ):
        return True, stats, q_tensor.detach().cpu(), order_tensor.detach().cpu()

    new_q = q_tensor.detach().float().cpu().numpy()
    cached_q = attn_mlp_policy_head_profile_cached_q.detach().float().cpu().numpy()
    numerator = float((new_q * cached_q).sum())
    denom = float(np.sqrt(max(float((new_q * new_q).sum()), 0.0) * max(float((cached_q * cached_q).sum()), 0.0)))
    alignment = numerator / max(denom, 1e-8)
    stats["attn_mlp_train_head_profile_gate_alignment"] = float(alignment)

    alignment_threshold = float(attn_mlp_policy_head_profile_accept_alignment_threshold)
    if bool(attn_mlp_policy_head_profile_accept_antialigned_only) and alignment >= alignment_threshold:
        return True, stats, q_tensor.detach().cpu(), order_tensor.detach().cpu()

    candidates = []
    new_order = _as_order_list(order_tensor)
    cached_order = _as_order_list(attn_mlp_policy_head_profile_cached_order)
    if _order_is_valid(new_order):
        candidates.append(
            {
                "source": "new",
                "order": new_order,
                "q": q_tensor.detach().cpu(),
            }
        )
    if _order_is_valid(cached_order):
        candidates.append(
            {
                "source": "cached",
                "order": cached_order,
                "q": attn_mlp_policy_head_profile_cached_q.detach().cpu(),
            }
        )
    if bool(attn_mlp_policy_head_profile_best_memory_enabled):
        if attn_mlp_policy_head_profile_best_order is not None:
            best_order = _as_order_list(attn_mlp_policy_head_profile_best_order)
            if _order_is_valid(best_order):
                best_q = (
                    attn_mlp_policy_head_profile_best_q.detach().cpu()
                    if attn_mlp_policy_head_profile_best_q is not None
                    else _q_from_order(best_order)
                )
                candidates.append(
                    {
                        "source": "best",
                        "order": best_order,
                        "q": best_q,
                    }
                )
    loss_by_order = _head_signal_evaluate_order_losses([item["order"] for item in candidates])
    stats["attn_mlp_train_head_profile_gate_checked"] = 1.0
    score_name = str(head_signal_probe_candidate_loss_profile_score).strip().lower()
    scored = []
    for item in candidates:
        loss_item = loss_by_order.get(tuple(item["order"]))
        if loss_item is None:
            continue
        score = _head_signal_order_loss_profile_score(loss_item, score_name)
        scored.append(
            {
                "source": str(item["source"]),
                "order": item["order"],
                "q": item["q"],
                "score": float(score),
            }
        )
    if not scored:
        if master_process:
            print(
                "attn-mlp head-profile gate "
                f"at iter {int(current_iter)}: accept=True reason=missing_score "
                f"alignment={alignment:+.4f}"
            )
        return True, stats, q_tensor.detach().cpu(), order_tensor.detach().cpu()

    scored.sort(
        key=lambda item: (
            float(item["score"]),
            float(source_ids.get(str(item["source"]), 9.0)),
            json.dumps(item["order"], separators=(",", ":")),
        )
    )
    new_items = [item for item in scored if item["source"] == "new"]
    cached_items = [item for item in scored if item["source"] == "cached"]
    best_items = [item for item in scored if item["source"] == "best"]
    new_score = float(new_items[0]["score"]) if new_items else float("nan")
    cached_score = float(cached_items[0]["score"]) if cached_items else float("nan")
    best_score = float(best_items[0]["score"]) if best_items else float("nan")
    old_items = [item for item in scored if item["source"] != "new"]
    old_best = old_items[0] if old_items else None
    new_item = new_items[0] if new_items else None
    if new_item is None:
        selected = old_best if old_best is not None else scored[0]
    elif old_best is None:
        selected = new_item
    elif bool(attn_mlp_policy_head_profile_best_memory_enabled):
        margin = max(float(attn_mlp_policy_head_profile_accept_margin), 0.0)
        selected = new_item if float(new_item["score"]) <= float(old_best["score"]) - margin else old_best
    else:
        margin = max(float(attn_mlp_policy_head_profile_accept_margin), 0.0)
        selected = new_item if float(new_item["score"]) <= float(cached_score) - margin else (cached_items[0] if cached_items else old_best)

    if selected is None:
        selected = scored[0]
    selected_source = str(selected["source"])
    selected_score = float(selected["score"])
    score_delta = float(new_score) - float(cached_score)
    margin = max(float(attn_mlp_policy_head_profile_accept_margin), 0.0)
    _select_stats(selected_source, selected_score, new_score, cached_score, best_score)
    stats["attn_mlp_train_head_profile_gate_score_delta"] = float(score_delta)
    stats["attn_mlp_train_head_profile_gate_margin"] = float(margin)
    if master_process:
        print(
            "attn-mlp head-profile gate "
            f"at iter {int(current_iter)}: selected={selected_source} "
            f"alignment={alignment:+.4f} score_delta={score_delta:+.6f} "
            f"new={new_score:.6f} cached={cached_score:.6f} "
            f"best={best_score:.6f} selected_score={selected_score:.6f} "
            f"margin={margin:.6f}"
        )
    return (
        selected_source == "new",
        stats,
        selected["q"].detach().cpu(),
        torch.tensor(selected["order"], dtype=torch.long, device="cpu"),
    )


def _attn_mlp_get_head_profile_target(current_iter):
    global attn_mlp_policy_head_profile_last_iter
    global attn_mlp_policy_head_profile_cached_q
    global attn_mlp_policy_head_profile_cached_order
    global attn_mlp_policy_head_profile_cached_stats
    global attn_mlp_policy_head_profile_best_q
    global attn_mlp_policy_head_profile_best_order
    global attn_mlp_policy_head_profile_best_iter

    if float(attn_mlp_policy_head_profile_weight) <= 0.0:
        return None
    refreshed = False
    if _attn_mlp_head_profile_due(current_iter):
        payload = _attn_mlp_compute_head_profile_consensus_target(current_iter)
        if payload is not None:
            q_tensor, order_tensor, stats = payload
            accept_target, gate_stats, selected_q, selected_order = _attn_mlp_head_profile_accept_new_target(
                q_tensor,
                order_tensor,
                current_iter,
            )
            stats = dict(stats)
            stats.update(gate_stats)
            attn_mlp_policy_head_profile_last_iter = int(current_iter)
            if selected_q is not None and selected_order is not None:
                selected_q = selected_q.detach().cpu()
                selected_order = selected_order.detach().cpu()
                previous_cached = attn_mlp_policy_head_profile_cached_order
                cached_changed = True
                if previous_cached is not None:
                    cached_changed = not torch.equal(
                        selected_order.view(-1).to(dtype=torch.long),
                        previous_cached.detach().cpu().view(-1).to(dtype=torch.long),
                    )
                attn_mlp_policy_head_profile_cached_q = selected_q
                attn_mlp_policy_head_profile_cached_order = selected_order
                best_updated = False
                if bool(attn_mlp_policy_head_profile_best_memory_enabled):
                    best_changed = True
                    if attn_mlp_policy_head_profile_best_order is not None:
                        best_changed = not torch.equal(
                            selected_order.view(-1).to(dtype=torch.long),
                            attn_mlp_policy_head_profile_best_order.detach().cpu().view(-1).to(dtype=torch.long),
                        )
                    if best_changed or attn_mlp_policy_head_profile_best_q is None:
                        attn_mlp_policy_head_profile_best_q = selected_q.detach().cpu()
                        attn_mlp_policy_head_profile_best_order = selected_order.detach().cpu()
                        attn_mlp_policy_head_profile_best_iter = int(current_iter)
                        best_updated = True
                stats["attn_mlp_train_head_profile_best_memory_updated"] = float(best_updated)
                stats["attn_mlp_train_head_profile_best_memory_available"] = float(
                    attn_mlp_policy_head_profile_best_order is not None
                )
                stats["attn_mlp_train_head_profile_best_memory_iter"] = float(
                    attn_mlp_policy_head_profile_best_iter
                )
                attn_mlp_policy_head_profile_cached_stats = dict(stats)
                refreshed = bool(accept_target or cached_changed)
            else:
                cached_stats = dict(attn_mlp_policy_head_profile_cached_stats or {})
                cached_stats.update(gate_stats)
                cached_stats["attn_mlp_train_head_profile_refreshed"] = 0.0
                cached_stats["attn_mlp_train_head_profile_last_iter"] = float(current_iter)
                attn_mlp_policy_head_profile_cached_stats = cached_stats

    if attn_mlp_policy_head_profile_cached_q is None:
        return None
    if not bool(attn_mlp_policy_head_profile_use_cached) and not refreshed:
        return None

    stats = dict(attn_mlp_policy_head_profile_cached_stats or {})
    stats["attn_mlp_train_head_profile_refreshed"] = float(refreshed)
    stats["attn_mlp_train_head_profile_last_iter"] = float(attn_mlp_policy_head_profile_last_iter)
    return (
        attn_mlp_policy_head_profile_cached_q,
        attn_mlp_policy_head_profile_cached_order,
        stats,
    )


def _attn_mlp_head_profile_loss(logits):
    zero = logits.new_tensor(0.0)
    base_stats = {
        "attn_mlp_train_head_profile_loss": float("nan"),
        "attn_mlp_train_head_profile_acc": float("nan"),
        "attn_mlp_train_head_profile_pairs": 0.0,
        "attn_mlp_train_head_profile_weight": float(attn_mlp_policy_head_profile_weight),
        "attn_mlp_train_head_profile_pair_weight_mean": 0.0,
        "attn_mlp_train_head_profile_refreshed": 0.0,
        "attn_mlp_train_head_profile_last_iter": float(attn_mlp_policy_head_profile_last_iter),
        "attn_mlp_train_head_profile_heads": 0.0,
        "attn_mlp_train_head_profile_samples": 0.0,
        "attn_mlp_train_head_profile_score_gap_mean": float("nan"),
        "attn_mlp_train_head_profile_alignment_mean": float("nan"),
        "attn_mlp_train_head_profile_fallback_frac": float("nan"),
        "attn_mlp_train_head_profile_q_abs_mean": float("nan"),
        "attn_mlp_train_head_profile_q_abs_max": float("nan"),
        "attn_mlp_train_head_profile_tau_diagnostic": float("nan"),
        "attn_mlp_train_head_profile_abs_tau_diagnostic": float("nan"),
        "attn_mlp_train_head_profile_head_tau_mean_diagnostic": float("nan"),
        "attn_mlp_train_head_profile_gate_enabled": float(
            bool(attn_mlp_policy_head_profile_accept_gate_enabled)
        ),
        "attn_mlp_train_head_profile_gate_checked": 0.0,
        "attn_mlp_train_head_profile_gate_accepted": float("nan"),
        "attn_mlp_train_head_profile_gate_alignment": float("nan"),
        "attn_mlp_train_head_profile_gate_new_score": float("nan"),
        "attn_mlp_train_head_profile_gate_cached_score": float("nan"),
        "attn_mlp_train_head_profile_gate_best_score": float("nan"),
        "attn_mlp_train_head_profile_gate_selected_score": float("nan"),
        "attn_mlp_train_head_profile_gate_score_delta": float("nan"),
        "attn_mlp_train_head_profile_gate_margin": float(
            attn_mlp_policy_head_profile_accept_margin
        ),
        "attn_mlp_train_head_profile_gate_selected_source_id": float("nan"),
        "attn_mlp_train_head_profile_gate_selected_is_new": float("nan"),
        "attn_mlp_train_head_profile_gate_selected_is_cached": float("nan"),
        "attn_mlp_train_head_profile_gate_selected_is_best": float("nan"),
        "attn_mlp_train_head_profile_best_memory_enabled": float(
            bool(attn_mlp_policy_head_profile_best_memory_enabled)
        ),
        "attn_mlp_train_head_profile_best_memory_available": float(
            attn_mlp_policy_head_profile_best_order is not None
        ),
        "attn_mlp_train_head_profile_best_memory_updated": 0.0,
        "attn_mlp_train_head_profile_best_memory_iter": float(
            attn_mlp_policy_head_profile_best_iter
        ),
    }
    if float(attn_mlp_policy_head_profile_weight) <= 0.0:
        return zero, base_stats

    payload = _attn_mlp_get_head_profile_target(iter_num)
    if payload is None:
        return zero, base_stats
    q_tensor, _order_tensor, stats = payload
    q = q_tensor.to(device=logits.device, dtype=torch.float32)
    pair_i, pair_j = torch.triu_indices(int(num_blocks), int(num_blocks), offset=1, device=logits.device)
    q_pair = q[pair_i, pair_j]
    pair_weights = q_pair.abs()
    min_abs_q = max(float(attn_mlp_policy_head_profile_min_abs_q), 0.0)
    valid = torch.isfinite(q_pair) & (pair_weights >= min_abs_q)
    if not bool(valid.any()):
        merged_stats = dict(base_stats)
        merged_stats.update(stats)
        return zero, merged_stats
    pair_i = pair_i[valid]
    pair_j = pair_j[valid]
    q_pair = q_pair[valid]
    pair_weights = pair_weights[valid]
    max_weight = max(float(attn_mlp_policy_head_profile_max_weight), 0.0)
    if max_weight > 0.0:
        pair_weights = pair_weights.clamp_max(max_weight)
    targets = (q_pair > 0.0).float()
    pref_logits = (logits.float()[pair_i] - logits.float()[pair_j]) / max(
        float(attn_mlp_policy_head_profile_tau),
        1e-6,
    )
    pair_losses = F.binary_cross_entropy_with_logits(pref_logits, targets, reduction="none")
    pair_loss = (pair_losses * pair_weights).sum() / pair_weights.sum().clamp_min(1e-8)
    pair_acc = (
        ((pref_logits.detach() > 0.0).float() == targets).float() * pair_weights
    ).sum() / pair_weights.sum().clamp_min(1e-8)
    merged_stats = dict(base_stats)
    merged_stats.update(stats)
    merged_stats.update(
        {
            "attn_mlp_train_head_profile_loss": float(pair_loss.detach().item()),
            "attn_mlp_train_head_profile_acc": float(pair_acc.detach().item()),
            "attn_mlp_train_head_profile_pairs": float(pair_i.numel()),
            "attn_mlp_train_head_profile_weight": float(attn_mlp_policy_head_profile_weight),
            "attn_mlp_train_head_profile_pair_weight_mean": float(pair_weights.detach().mean().item()),
        }
    )
    return pair_loss, merged_stats


def _attn_mlp_train_policy_sampled_nll_pg(features, probe_batches, include_move_pref=False):
    if not probe_batches:
        return None
    num_states = max(1, int(attn_mlp_policy_nll_states_per_update))
    num_samples = max(1, int(attn_mlp_policy_sampled_orders_per_state))
    num_random = max(1, int(attn_mlp_policy_random_baseline_orders))
    prefix_k = int(attn_mlp_policy_prefix_k)
    move_pairs_per_state = max(0, int(attn_mlp_policy_move_pref_pairs_per_state))
    move_window = max(1, min(int(attn_mlp_policy_move_pref_window), max(1, num_blocks - 1)))

    attn_mlp_policy.train()
    logits = attn_mlp_policy(features)
    sampled_orders, logps = _attn_mlp_sample_pl_orders_and_logps(logits, num_samples)
    sampled_orders = sampled_orders.squeeze(1) if sampled_orders.ndim == 3 and sampled_orders.size(1) == 1 else sampled_orders
    logps = logps.squeeze(1) if logps.ndim == 2 and logps.size(1) == 1 else logps
    map_order = logits_to_order(logits.detach(), mode=str(attn_mlp_policy_order_mode)).detach().to(
        device=device,
        dtype=torch.long,
    )

    sample_losses = []
    sample_prefix_losses = []
    baseline_losses = []
    baseline_prefix_losses = []
    move_pref_logits = []
    move_pref_targets = []
    move_pref_weights = []
    move_deltas = []
    move_base_full_losses = []
    move_swap_full_losses = []
    move_base_scores = []
    move_best_scores = []
    state_count = 0
    for x in probe_batches[:num_states]:
        x = x.detach().to(device=device, dtype=torch.long)
        random_full = []
        random_prefix = []
        for _ in range(num_random):
            random_order = torch.randperm(num_blocks, device=device)
            full_loss, prefix_loss = _attn_mlp_full_and_prefix_loss_for_block_order(
                x,
                random_order,
                prefix_k=prefix_k,
            )
            random_full.append(full_loss)
            random_prefix.append(prefix_loss)
        baseline_losses.append(float(np.mean(random_full)))
        baseline_prefix_losses.append(float(np.mean(random_prefix)))

        per_state_sample = []
        per_state_prefix = []
        for sample_idx in range(num_samples):
            full_loss, prefix_loss = _attn_mlp_full_and_prefix_loss_for_block_order(
                x,
                sampled_orders[sample_idx],
                prefix_k=prefix_k,
            )
            per_state_sample.append(full_loss)
            per_state_prefix.append(prefix_loss)
        sample_losses.append(per_state_sample)
        sample_prefix_losses.append(per_state_prefix)

        need_base_order_loss = bool(include_move_pref) and move_pairs_per_state > 0
        base_full = None
        base_prefix = None
        base_score = None
        if need_base_order_loss:
            base_full, base_prefix = _attn_mlp_full_and_prefix_loss_for_block_order(
                x,
                map_order,
                prefix_k=prefix_k,
            )
            base_score = float(base_full) + float(attn_mlp_policy_move_pref_prefix_weight) * float(base_prefix)

        if bool(include_move_pref) and move_pairs_per_state > 0 and base_score is not None:
            candidate_positions = [
                (left, left + offset)
                for offset in range(1, move_window + 1)
                for left in range(0, num_blocks - offset)
            ]
            if candidate_positions:
                perm = torch.randperm(len(candidate_positions), device=device)
                selected = perm[: min(move_pairs_per_state, len(candidate_positions))].detach().cpu().tolist()
                for candidate_idx in selected:
                    left_pos, right_pos = candidate_positions[int(candidate_idx)]
                    left_block = int(map_order[left_pos].item())
                    right_block = int(map_order[right_pos].item())
                    swapped_order = map_order.clone()
                    swapped_order[left_pos] = map_order[right_pos]
                    swapped_order[right_pos] = map_order[left_pos]
                    swap_full, swap_prefix = _attn_mlp_full_and_prefix_loss_for_block_order(
                        x,
                        swapped_order,
                        prefix_k=prefix_k,
                    )
                    swap_score = float(swap_full) + float(attn_mlp_policy_move_pref_prefix_weight) * float(swap_prefix)
                    delta = float(swap_score) - float(base_score)
                    target = 1.0 if delta >= 0.0 else 0.0
                    margin = max(float(attn_mlp_policy_move_pref_margin), 1e-8)
                    max_weight = max(float(attn_mlp_policy_move_pref_max_weight), 0.0)
                    weight = min(max(abs(delta) / margin, 0.0), max_weight)
                    if weight <= 0.0:
                        continue
                    move_pref_logits.append((logits[left_block] - logits[right_block]) / max(float(attn_mlp_policy_move_pref_tau), 1e-6))
                    move_pref_targets.append(float(target))
                    move_pref_weights.append(float(weight))
                    move_deltas.append(float(delta))
                    move_base_full_losses.append(float(base_full))
                    move_swap_full_losses.append(float(swap_full))
                    move_base_scores.append(float(base_score))
                    move_best_scores.append(min(float(base_score), float(swap_score)))
        state_count += 1

    sample_losses_t = logits.new_tensor(sample_losses)
    sample_prefix_t = logits.new_tensor(sample_prefix_losses)
    baseline_t = logits.new_tensor(baseline_losses).view(-1, 1)
    baseline_prefix_t = logits.new_tensor(baseline_prefix_losses).view(-1, 1)
    full_advantages = baseline_t - sample_losses_t
    prefix_advantages = baseline_prefix_t - sample_prefix_t
    advantages = full_advantages + float(attn_mlp_policy_prefix_reward_weight) * prefix_advantages
    reward_scale = advantages.detach().std(unbiased=False).clamp_min(float(attn_mlp_policy_reward_scale_floor))
    adv_norm = (advantages / reward_scale).clamp(
        -float(attn_mlp_policy_advantage_clip),
        float(attn_mlp_policy_advantage_clip),
    )
    logps_t = logps.view(1, -1).expand(state_count, -1)
    loss_pg = -(adv_norm.detach() * logps_t).mean()
    if move_pref_logits:
        move_logits_t = torch.stack(move_pref_logits).float()
        move_targets_t = logits.new_tensor(move_pref_targets).float()
        move_weights_t = logits.new_tensor(move_pref_weights).float()
        move_losses = F.binary_cross_entropy_with_logits(move_logits_t, move_targets_t, reduction="none")
        move_loss = (move_losses * move_weights_t).sum() / move_weights_t.sum().clamp_min(1e-8)
        move_pred = (move_logits_t.detach() > 0.0).float()
        move_acc = ((move_pred == move_targets_t).float() * move_weights_t).sum() / move_weights_t.sum().clamp_min(1e-8)
        move_delta_mean = logits.new_tensor(move_deltas).float().mean()
        move_delta_abs_mean = logits.new_tensor(move_deltas).float().abs().mean()
        move_weight_mean = move_weights_t.mean()
        move_base_full_mean = logits.new_tensor(move_base_full_losses).float().mean()
        move_swap_full_mean = logits.new_tensor(move_swap_full_losses).float().mean()
        move_base_score_mean = logits.new_tensor(move_base_scores).float().mean()
        move_best_score_mean = logits.new_tensor(move_best_scores).float().mean()
        move_pair_count = float(len(move_pref_logits))
    else:
        move_loss = logits.new_tensor(0.0)
        move_acc = logits.new_tensor(float("nan"))
        move_delta_mean = logits.new_tensor(float("nan"))
        move_delta_abs_mean = logits.new_tensor(float("nan"))
        move_weight_mean = logits.new_tensor(0.0)
        move_base_full_mean = logits.new_tensor(float("nan"))
        move_swap_full_mean = logits.new_tensor(float("nan"))
        move_base_score_mean = logits.new_tensor(float("nan"))
        move_best_score_mean = logits.new_tensor(float("nan"))
        move_pair_count = 0.0

    probs = torch.softmax(logits.float(), dim=-1)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum()
    logit_l2 = logits.float().pow(2).mean()
    logit_std = logits.float().std(unbiased=False)
    std_floor = torch.relu(logits.new_tensor(float(attn_mlp_policy_min_logit_std)) - logit_std).pow(2)
    entropy_floor = torch.relu(logits.new_tensor(float(attn_mlp_policy_min_entropy)) - entropy).pow(2)
    entropy_ceiling = torch.relu(entropy - logits.new_tensor(float(attn_mlp_policy_max_entropy))).pow(2)
    attn_pair_loss, attn_close_loss, attn_pair_stats = _attn_mlp_attention_pair_losses(logits, features)
    head_profile_loss, head_profile_stats = _attn_mlp_head_profile_loss(logits)

    total_loss = (
        float(attn_mlp_policy_pg_weight) * loss_pg
        + float(attn_mlp_policy_move_pref_weight) * move_loss
        + float(attn_mlp_policy_attn_pair_weight) * attn_pair_loss
        + float(attn_mlp_policy_attn_close_weight) * attn_close_loss
        + float(attn_mlp_policy_head_profile_weight) * head_profile_loss
        + float(attn_mlp_policy_logit_l2) * logit_l2
        + float(attn_mlp_policy_std_floor_weight) * std_floor
        + float(attn_mlp_policy_entropy_floor_weight) * entropy_floor
        + float(attn_mlp_policy_entropy_ceiling_weight) * entropy_ceiling
    )
    attn_mlp_policy_optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    grad_norm = 0.0
    if float(attn_mlp_policy_grad_clip) > 0.0:
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
            attn_mlp_policy.parameters(),
            float(attn_mlp_policy_grad_clip),
        )
        grad_norm = float(grad_norm_tensor.item())
    attn_mlp_policy_optimizer.step()
    attn_mlp_policy.eval()

    with torch.no_grad():
        attn_mlp_policy_last_train_stats = {
            "attn_mlp_train_enabled": 1.0,
            "attn_mlp_train_loss": float(total_loss.detach().item()),
            "attn_mlp_train_pg_loss": float(loss_pg.detach().item()),
            "attn_mlp_train_move_loss": float(move_loss.detach().item()),
            "attn_mlp_train_move_acc": float(move_acc.detach().item()),
            "attn_mlp_train_move_pairs": float(move_pair_count),
            "attn_mlp_train_move_delta_mean": float(move_delta_mean.detach().item()),
            "attn_mlp_train_move_delta_abs_mean": float(move_delta_abs_mean.detach().item()),
            "attn_mlp_train_move_weight_mean": float(move_weight_mean.detach().item()),
            "attn_mlp_train_move_weight": float(attn_mlp_policy_move_pref_weight),
            "attn_mlp_train_move_base_full_loss": float(move_base_full_mean.detach().item()),
            "attn_mlp_train_move_swap_full_loss": float(move_swap_full_mean.detach().item()),
            "attn_mlp_train_move_base_score": float(move_base_score_mean.detach().item()),
            "attn_mlp_train_move_best_score": float(move_best_score_mean.detach().item()),
            "attn_mlp_train_attn_pair_weight": float(attn_mlp_policy_attn_pair_weight),
            "attn_mlp_train_attn_close_weight": float(attn_mlp_policy_attn_close_weight),
            **attn_pair_stats,
            **head_profile_stats,
            "attn_mlp_train_pair_loss": float("nan"),
            "attn_mlp_train_axis_loss": float("nan"),
            "attn_mlp_train_pair_acc": float("nan"),
            "attn_mlp_train_pair_weight_mean": 0.0,
            "attn_mlp_train_pair_weight": 0.0,
            "attn_mlp_train_axis_weight": 0.0,
            "attn_mlp_train_target_axis_std": float("nan"),
            "attn_mlp_train_logit_l2": float(logit_l2.detach().item()),
            "attn_mlp_train_logit_std": float(logit_std.detach().item()),
            "attn_mlp_train_logit_entropy": float(entropy.detach().item()),
            "attn_mlp_train_std_floor": float(std_floor.detach().item()),
            "attn_mlp_train_entropy_floor": float(entropy_floor.detach().item()),
            "attn_mlp_train_entropy_ceiling": float(entropy_ceiling.detach().item()),
            "attn_mlp_train_grad_norm": float(grad_norm),
            "attn_mlp_train_sample_minus_random_mean": float(
                (sample_losses_t - baseline_t).detach().mean().item()
            ),
            "attn_mlp_train_prefix_minus_random_mean": float(
                (sample_prefix_t - baseline_prefix_t).detach().mean().item()
            ),
            "attn_mlp_train_full_advantage_mean": float(full_advantages.detach().mean().item()),
            "attn_mlp_train_prefix_advantage_mean": float(prefix_advantages.detach().mean().item()),
            "attn_mlp_train_sample_full_loss": float(sample_losses_t.detach().mean().item()),
            "attn_mlp_train_random_full_loss": float(baseline_t.detach().mean().item()),
            "attn_mlp_train_sample_prefix_loss": float(sample_prefix_t.detach().mean().item()),
            "attn_mlp_train_random_prefix_loss": float(baseline_prefix_t.detach().mean().item()),
            "attn_mlp_train_reward_scale": float(reward_scale.detach().item()),
            "attn_mlp_train_pl_logp": float(logps.detach().mean().item()),
            "attn_mlp_train_nll_states": float(state_count),
            "attn_mlp_train_sampled_orders_per_state": float(num_samples),
            "attn_mlp_train_random_baseline_orders": float(num_random),
        }
    return attn_mlp_policy_last_train_stats


def _attn_mlp_train_policy_from_input(policy_input, probe_batches=None, current_iter=None):
    global attn_mlp_policy_last_train_stats
    if (
        attn_mlp_policy is None
        or attn_mlp_policy_optimizer is None
        or policy_input is None
        or bool(attn_mlp_policy_freeze)
    ):
        return None
    loss_name = str(attn_mlp_policy_train_loss or "none").lower()
    if loss_name in {"none", "off", ""}:
        attn_mlp_policy_last_train_stats = {"attn_mlp_train_enabled": 0.0}
        return attn_mlp_policy_last_train_stats
    shadow_mse_loss_names = _attn_mlp_policy_shadow_mse_loss_names()
    sampled_loss_names = {"sampled_nll_pg", "full_nll_pg", "nll_pg"}
    axis_profile_loss_names = {
        "axis_profile_simple",
        "attention_axis_profile",
        "direct_axis_profile",
    }
    directed_ribbon_loss_names = {
        "directed_ribbon",
        "attention_directed_ribbon",
        "asym_ribbon",
    }
    move_pref_loss_names = {"sampled_nll_pg_move_pref", "nll_pg_move_pref", "move_pref_pg"}
    if loss_name not in {
        *shadow_mse_loss_names,
        *axis_profile_loss_names,
        *directed_ribbon_loss_names,
        *sampled_loss_names,
        *move_pref_loss_names,
    }:
        raise ValueError(f"Unsupported attn_mlp_policy_train_loss={attn_mlp_policy_train_loss!r}")
    if loss_name in shadow_mse_loss_names:
        stats = _attn_mlp_train_policy_teacher_score_mse(
            policy_input.detach().float(),
            current_iter=current_iter,
        )
        if stats is None:
            attn_mlp_policy_last_train_stats = {"attn_mlp_train_enabled": 0.0}
            return attn_mlp_policy_last_train_stats
        attn_mlp_policy_last_train_stats = stats
        return attn_mlp_policy_last_train_stats
    if loss_name in directed_ribbon_loss_names:
        stats = _attn_mlp_directed_ribbon_loss(policy_input.detach().float())
        if stats is None:
            attn_mlp_policy_last_train_stats = {"attn_mlp_train_enabled": 0.0}
            return attn_mlp_policy_last_train_stats
        attn_mlp_policy_last_train_stats = stats
        return attn_mlp_policy_last_train_stats
    if loss_name in axis_profile_loss_names:
        stats = _attn_mlp_axis_profile_simple_loss(
            policy_input.detach().float(),
            list(probe_batches or []),
        )
        if stats is None:
            attn_mlp_policy_last_train_stats = {"attn_mlp_train_enabled": 0.0}
            return attn_mlp_policy_last_train_stats
        attn_mlp_policy_last_train_stats = stats
        return attn_mlp_policy_last_train_stats
    if policy_input.ndim != 3:
        raise ValueError(
            "Trainable AttnMLP losses require a feature tensor."
        )

    features = policy_input.detach().float().to(device=device)
    if loss_name in sampled_loss_names or loss_name in move_pref_loss_names:
        stats = _attn_mlp_train_policy_sampled_nll_pg(
            features,
            list(probe_batches or []),
            include_move_pref=(loss_name in move_pref_loss_names),
        )
        if stats is None:
            attn_mlp_policy_last_train_stats = {"attn_mlp_train_enabled": 0.0}
            return attn_mlp_policy_last_train_stats
        attn_mlp_policy_last_train_stats = stats
        return attn_mlp_policy_last_train_stats
    raise ValueError(f"Unsupported attn_mlp_policy_train_loss={attn_mlp_policy_train_loss!r}")


def _attn_mlp_is_valid_block_order(order):
    if order is None:
        return False
    order_cpu = order.detach().to(device="cpu", dtype=torch.long).view(-1)
    if int(order_cpu.numel()) != int(num_blocks):
        return False
    return sorted(order_cpu.tolist()) == list(range(num_blocks))


def _attn_mlp_normalize_order_logits(logits):
    values = logits.detach().float().cpu()
    mode = str(attn_mlp_policy_logits_ema_normalize).strip().lower()
    if mode in {"", "none", "raw"}:
        return values
    if mode in {"sigmoid", "score", "scores", "prob", "priority"}:
        return torch.sigmoid(values)
    if mode in {"center", "centered", "mean"}:
        return values - values.mean()
    if mode in {"zscore", "standard", "standardize"}:
        centered = values - values.mean()
        return centered / (centered.std(unbiased=False) + 1e-6)
    raise ValueError(
        "Unsupported attn_mlp_policy_logits_ema_normalize="
        f"{attn_mlp_policy_logits_ema_normalize!r}. Expected sigmoid, zscore, center, or none."
    )


def _attn_mlp_order_logits_for_cache(raw_logits, current_iter):
    global attn_mlp_policy_logits_ema
    if not bool(attn_mlp_policy_logits_ema_enabled):
        return raw_logits.detach().float()
    if int(current_iter) < int(attn_mlp_policy_logits_ema_start_iter):
        return raw_logits.detach().float()

    update = _attn_mlp_normalize_order_logits(raw_logits)
    if attn_mlp_policy_logits_ema is None or tuple(attn_mlp_policy_logits_ema.shape) != tuple(update.shape):
        attn_mlp_policy_logits_ema = update.detach().cpu()
    else:
        decay = min(0.9999, max(0.0, float(attn_mlp_policy_logits_ema_decay)))
        attn_mlp_policy_logits_ema = (
            attn_mlp_policy_logits_ema.detach().cpu() * decay
            + update.detach().cpu() * (1.0 - decay)
        )
    return attn_mlp_policy_logits_ema.to(device=raw_logits.device, dtype=torch.float32)


def _attn_mlp_json_safe(value):
    if isinstance(value, dict):
        return {str(k): _attn_mlp_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_attn_mlp_json_safe(v) for v in value]
    if torch.is_tensor(value):
        return _attn_mlp_json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return _attn_mlp_json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _attn_mlp_tensor_to_float_list(values):
    tensor = values.detach().float().cpu().view(-1)
    return [float(v) for v in tensor.tolist()]


def _attn_mlp_order_to_list(order):
    if order is None:
        return None
    if torch.is_tensor(order):
        values = order.detach().cpu().long().view(-1).tolist()
    else:
        values = list(order)
    return [int(v) for v in values]


def _attn_mlp_order_to_original(order_current):
    order_list = _attn_mlp_order_to_list(order_current)
    if order_list is None:
        return None
    if bool(permute_data) and fixed_block_perm is not None:
        mapper = fixed_block_perm.detach().cpu().long()
        return [int(mapper[int(v)].item()) for v in order_list]
    return list(order_list)


def _attn_mlp_original_order_metrics(order_current):
    order_original = _attn_mlp_order_to_original(order_current)
    if order_original is None:
        return None
    order_tensor = torch.tensor(order_original, dtype=torch.long).view(1, -1)
    tau = float(kendall_tau_to_l2r_per_sample(order_tensor).float().cpu()[0].item())
    total_pairs = int(num_blocks) * (int(num_blocks) - 1) / 2.0
    raw_distance = (1.0 - tau) * (total_pairs / 2.0)
    normalized_distance = raw_distance / total_pairs if total_pairs > 0 else raw_distance
    return {
        "order_original_diagnostic": order_original,
        "tau_original_diagnostic": float(tau),
        "kendall_distance_original": float(raw_distance),
        "normalized_kendall_distance_original": float(normalized_distance),
    }


def _attn_mlp_pairwise_order_stats(order_a, order_b):
    order_a_list = _attn_mlp_order_to_list(order_a)
    order_b_list = _attn_mlp_order_to_list(order_b)
    if order_a_list is None or order_b_list is None:
        return {"pair_acc": float("nan"), "pair_tau": float("nan"), "pair_count": 0.0}
    n = int(num_blocks)
    if len(order_a_list) != n or len(order_b_list) != n:
        return {"pair_acc": float("nan"), "pair_tau": float("nan"), "pair_count": 0.0}
    rank_a = torch.empty(n, dtype=torch.long)
    rank_b = torch.empty(n, dtype=torch.long)
    for pos, item in enumerate(order_a_list):
        rank_a[int(item)] = int(pos)
    for pos, item in enumerate(order_b_list):
        rank_b[int(item)] = int(pos)
    idx_i, idx_j = torch.triu_indices(n, n, offset=1)
    same = (rank_a[idx_i] < rank_a[idx_j]) == (rank_b[idx_i] < rank_b[idx_j])
    pair_count = int(same.numel())
    if pair_count <= 0:
        return {"pair_acc": float("nan"), "pair_tau": float("nan"), "pair_count": 0.0}
    pair_acc = float(same.float().mean().item())
    return {
        "pair_acc": pair_acc,
        "pair_tau": float(2.0 * pair_acc - 1.0),
        "pair_count": float(pair_count),
    }


def _attn_mlp_score_similarity_stats(scores_a, scores_b):
    a = torch.as_tensor(scores_a, dtype=torch.float32).detach().cpu().view(-1)
    b = torch.as_tensor(scores_b, dtype=torch.float32).detach().cpu().view(-1)
    if a.numel() != b.numel() or a.numel() <= 0:
        return {
            "mse": float("nan"),
            "mae": float("nan"),
            "pearson": float("nan"),
            "score_pair_acc": float("nan"),
            "score_pair_tau": float("nan"),
            "score_pair_count": 0.0,
        }
    finite = torch.isfinite(a) & torch.isfinite(b)
    if int(finite.sum().item()) <= 0:
        return {
            "mse": float("nan"),
            "mae": float("nan"),
            "pearson": float("nan"),
            "score_pair_acc": float("nan"),
            "score_pair_tau": float("nan"),
            "score_pair_count": 0.0,
        }
    a_valid = a[finite]
    b_valid = b[finite]
    diff = a_valid - b_valid
    mse = float((diff * diff).mean().item())
    mae = float(diff.abs().mean().item())
    a_centered = a_valid - a_valid.mean()
    b_centered = b_valid - b_valid.mean()
    denom = torch.sqrt((a_centered * a_centered).sum() * (b_centered * b_centered).sum())
    pearson = float((a_centered * b_centered).sum().item() / float(denom.item())) if float(denom.item()) > 1e-12 else float("nan")

    if int(finite.sum().item()) < 2:
        pair_acc = float("nan")
        pair_tau = float("nan")
        pair_count = 0
    else:
        a_pair = a[finite]
        b_pair = b[finite]
        n = int(a_pair.numel())
        idx_i, idx_j = torch.triu_indices(n, n, offset=1)
        delta_a = a_pair[idx_i] - a_pair[idx_j]
        delta_b = b_pair[idx_i] - b_pair[idx_j]
        comparable = (delta_a.abs() > 1e-12) & (delta_b.abs() > 1e-12)
        pair_count = int(comparable.sum().item())
        if pair_count <= 0:
            pair_acc = float("nan")
            pair_tau = float("nan")
        else:
            same = (delta_a[comparable] > 0.0) == (delta_b[comparable] > 0.0)
            pair_acc = float(same.float().mean().item())
            pair_tau = float(2.0 * pair_acc - 1.0)

    return {
        "mse": mse,
        "mae": mae,
        "pearson": pearson,
        "score_pair_acc": float(pair_acc),
        "score_pair_tau": float(pair_tau),
        "score_pair_count": float(pair_count),
    }


def _attn_mlp_teacher_fiedler_scores(matrix):
    values = matrix.detach().float().cpu().numpy() if torch.is_tensor(matrix) else np.asarray(matrix, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).copy()
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"teacher_diag pairwise_max_fiedler expects a square matrix, got shape={tuple(values.shape)}")
    if int(values.shape[0]) != int(num_blocks):
        raise ValueError(
            "teacher_diag pairwise_max_fiedler expected "
            f"{int(num_blocks)} blocks, got {int(values.shape[0])}"
        )
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(values, 0.0)
    affinity = np.maximum(values, values.T)
    affinity = np.nan_to_num(affinity, nan=0.0, posinf=0.0, neginf=0.0)
    affinity = np.maximum(affinity, 0.0)
    np.fill_diagonal(affinity, 0.0)
    degree = affinity.sum(axis=1)
    laplacian = np.diag(degree) - affinity
    eigvals, eigvecs = np.linalg.eigh(laplacian)
    if eigvals.size <= 1 or eigvecs.size <= 0:
        raise ValueError("teacher_diag pairwise_max_fiedler produced no nontrivial eigenvectors")
    vector = np.asarray(eigvecs[:, 1], dtype=np.float64)
    if not np.isfinite(vector).all():
        raise ValueError("teacher_diag pairwise_max_fiedler produced a non-finite Fiedler vector")
    vector_min = float(np.min(vector))
    vector_max = float(np.max(vector))
    vector_range = float(vector_max - vector_min)
    if vector_range <= 1e-12:
        raise ValueError("teacher_diag pairwise_max_fiedler produced a degenerate Fiedler vector")
    vector_norm = (vector - vector_min) / vector_range
    reverse_priority = torch.tensor(vector_norm, dtype=torch.float32)
    raw_priority = torch.tensor(1.0 - vector_norm, dtype=torch.float32)
    spectral_gap = (
        float(eigvals[2] - eigvals[1])
        if eigvals.size > 2
        else float(abs(eigvals[1]))
    )
    meta = {
        "candidate_source": "pairwise_max_fiedler",
        "input": "policy_input_attention",
        "affinity": "max(A,A.T)",
        "laplacian": "D-W",
        "eigen_selector": "second_smallest_laplacian",
        "score_scale": "minmax_fiedler_priority",
        "raw_priority": "1 - minmax(fiedler_vector)",
        "reverse_priority": "minmax(fiedler_vector)",
        "laplacian_eigval0": float(eigvals[0]),
        "laplacian_fiedler_eigval": float(eigvals[1]),
        "laplacian_next_eigval": float(eigvals[2]) if eigvals.size > 2 else float("nan"),
        "spectral_gap": float(spectral_gap),
        "affinity_sum": float(affinity.sum()),
        "affinity_density": float(np.mean(affinity > 0.0)),
        "vector_min": float(vector_min),
        "vector_max": float(vector_max),
        "vector_std": float(np.std(vector)),
    }
    return raw_priority, reverse_priority, meta


def _attn_mlp_shadow_mse_log_file():
    configured = str(attn_mlp_policy_shadow_mse_log_path).strip()
    if configured:
        return configured
    input_attn_dir = str(attn_mlp_policy_log_input_attn_out_dir).strip()
    if input_attn_dir:
        return os.path.join(input_attn_dir, "attn_mlp_shadow_mse_history.jsonl")
    return os.path.join(out_dir, "attn_mlp_shadow_mse_history.jsonl")


def _attn_mlp_shadow_mse_record_from_outputs(outputs, block_orders, batch_x):
    mode = str(attn_mlp_policy_feature_mode or "attention").lower()
    if mode not in {"attention", "matrix", "single"}:
        raise ValueError(
            "attn_mlp_policy_train_loss='teacher_score_mse' expects "
            "attn_mlp_policy_feature_mode='attention'/'matrix'/'single'."
        )
    sample_matrices = _attn_mlp_attention_sample_matrices_from_outputs(outputs, block_orders)
    if sample_matrices is None or sample_matrices.ndim != 3 or int(sample_matrices.size(0)) <= 0:
        return None
    batch_n = min(int(sample_matrices.size(0)), int(batch_x.size(0)))
    if batch_n <= 0:
        return None
    return {
        "sample_matrices": sample_matrices[:batch_n].detach().float().cpu(),
        "x": batch_x[:batch_n].detach().clone(),
    }


def _attn_mlp_shadow_mse_item_from_records(records):
    records = [record for record in records if isinstance(record, dict)]
    if not records:
        return None
    matrices = torch.cat([record["sample_matrices"] for record in records], dim=0)
    x_values = torch.cat([record["x"] for record in records], dim=0)
    batch_n = min(int(matrices.size(0)), int(x_values.size(0)))
    if batch_n <= 0:
        return None
    matrices = matrices[:batch_n]
    x_values = x_values[:batch_n]
    train_frac = min(0.99, max(0.01, float(attn_mlp_policy_shadow_mse_train_frac)))
    if batch_n >= 2:
        train_count = int(round(float(batch_n) * train_frac))
        train_count = max(1, min(batch_n - 1, train_count))
    else:
        train_count = 1
    val_count = batch_n - train_count
    train_matrix = matrices[:train_count].mean(dim=0)
    if val_count > 0:
        val_matrix = matrices[train_count:batch_n].mean(dim=0)
        val_x = x_values[train_count:batch_n]
    else:
        val_matrix = train_matrix.detach().clone()
        val_x = x_values[:train_count]
    return {
        "train_matrix": train_matrix.detach().float(),
        "val_matrix": val_matrix.detach().float(),
        "train_x": x_values[:train_count].detach().clone(),
        "val_x": val_x.detach().clone(),
        "sample_count": int(batch_n),
        "train_count": int(train_count),
        "val_count": int(val_count),
        "val_is_holdout": bool(val_count > 0),
    }


def _attn_mlp_shadow_mse_item_from_records_fixed_samples(records):
    records = [record for record in records if isinstance(record, dict)]
    if not records:
        return None
    matrices = torch.cat([record["sample_matrices"] for record in records], dim=0)
    x_values = torch.cat([record["x"] for record in records], dim=0)
    batch_n = min(int(matrices.size(0)), int(x_values.size(0)))
    train_count = max(1, int(attn_mlp_policy_shadow_mse_train_samples_per_step))
    val_count = max(0, int(attn_mlp_policy_shadow_mse_val_samples_per_step))
    needed = train_count + val_count
    if batch_n < max(1, needed):
        return None
    matrices = matrices[:needed]
    x_values = x_values[:needed]
    train_matrix = matrices[:train_count].mean(dim=0)
    if val_count > 0:
        val_matrix = matrices[train_count:needed].mean(dim=0)
        val_x = x_values[train_count:needed]
    else:
        val_matrix = train_matrix.detach().clone()
        val_x = x_values[:train_count]
    return {
        "train_matrix": train_matrix.detach().float(),
        "val_matrix": val_matrix.detach().float(),
        "train_x": x_values[:train_count].detach().clone(),
        "val_x": val_x.detach().clone(),
        "sample_count": int(needed),
        "train_count": int(train_count),
        "val_count": int(val_count),
        "val_is_holdout": bool(val_count > 0),
        "sample_split_mode": "fixed_per_step",
    }


def _attn_mlp_shadow_mse_matrix_item_from_records(records, label_records=None):
    records = [record for record in records if isinstance(record, dict)]
    if not records:
        return None
    matrices = torch.cat([record["sample_matrices"] for record in records], dim=0)
    x_values = torch.cat([record["x"] for record in records], dim=0)
    batch_n = min(int(matrices.size(0)), int(x_values.size(0)))
    if batch_n <= 0:
        return None
    matrices = matrices[:batch_n]
    x_values = x_values[:batch_n]
    item = {
        "matrix": matrices.mean(dim=0).detach().float(),
        "x": x_values.detach().clone(),
        "sample_count": int(batch_n),
        "source_batches": int(len(records)),
    }
    if label_records is not None:
        label_records = [record for record in label_records if isinstance(record, dict)]
        if label_records:
            label_x = torch.cat([record["x"] for record in label_records], dim=0)
            label_n = int(label_x.size(0))
            if label_n > 0:
                item["label_x"] = label_x[:label_n].detach().clone()
                item["label_sample_count"] = int(label_n)
                item["label_source_batches"] = int(len(label_records))
    return item


def _attn_mlp_shadow_mse_buffer_stats():
    orientation_x_mode = str(attn_mlp_policy_shadow_mse_orientation_x_mode).strip().lower()
    return {
        "attn_mlp_train_shadow_mse_batches_per_item": float(attn_mlp_policy_shadow_mse_batches_per_item),
        "attn_mlp_train_shadow_mse_train_items_per_update": float(attn_mlp_policy_shadow_mse_train_items_per_update),
        "attn_mlp_train_shadow_mse_val_items_per_update": float(attn_mlp_policy_shadow_mse_val_items_per_update),
        "attn_mlp_train_shadow_mse_train_samples_per_step": float(attn_mlp_policy_shadow_mse_train_samples_per_step),
        "attn_mlp_train_shadow_mse_val_samples_per_step": float(attn_mlp_policy_shadow_mse_val_samples_per_step),
        "attn_mlp_train_shadow_mse_orientation_x_mode_last_step": float(
            orientation_x_mode in {"last", "last_step", "last_batch", "last_update_step"}
        ),
        "attn_mlp_train_shadow_mse_batch_buffer": float(len(attn_mlp_policy_shadow_mse_batch_buffer)),
        "attn_mlp_train_shadow_mse_item_buffer": float(len(attn_mlp_policy_shadow_mse_item_buffer)),
    }


def _attn_mlp_shadow_mse_items_for_update(records, emit=True):
    global attn_mlp_policy_shadow_mse_batch_buffer, attn_mlp_policy_shadow_mse_item_buffer
    records = [record for record in records if isinstance(record, dict)]
    if _attn_mlp_policy_shadow_mse_fixed_sample_split_enabled():
        item = _attn_mlp_shadow_mse_item_from_records_fixed_samples(records)
        if item is not None:
            attn_mlp_policy_shadow_mse_item_buffer.append(item)
        if not bool(emit):
            return []
        selected = list(attn_mlp_policy_shadow_mse_item_buffer)
        attn_mlp_policy_shadow_mse_item_buffer = []
        return selected

    if not _attn_mlp_policy_shadow_mse_grouped_items_enabled():
        item = _attn_mlp_shadow_mse_item_from_records(records)
        if item is not None:
            attn_mlp_policy_shadow_mse_item_buffer.append(item)
        if not bool(emit):
            return []
        selected = list(attn_mlp_policy_shadow_mse_item_buffer)
        attn_mlp_policy_shadow_mse_item_buffer = []
        return selected

    if records:
        attn_mlp_policy_shadow_mse_batch_buffer.append(list(records))

    batches_per_item = max(1, int(attn_mlp_policy_shadow_mse_batches_per_item))
    orientation_x_mode = str(attn_mlp_policy_shadow_mse_orientation_x_mode).strip().lower()
    while len(attn_mlp_policy_shadow_mse_batch_buffer) >= batches_per_item:
        grouped_records = []
        last_batch_records = []
        for _ in range(batches_per_item):
            batch_records = attn_mlp_policy_shadow_mse_batch_buffer.pop(0)
            grouped_records.extend(batch_records)
            last_batch_records = batch_records
        label_records = None
        if orientation_x_mode in {"last", "last_step", "last_batch", "last_update_step"}:
            label_records = last_batch_records
        item = _attn_mlp_shadow_mse_matrix_item_from_records(
            grouped_records,
            label_records=label_records,
        )
        if item is not None:
            item["batches_per_item"] = int(batches_per_item)
            item["orientation_x_mode"] = orientation_x_mode or "all"
            attn_mlp_policy_shadow_mse_item_buffer.append(item)
    if not bool(emit):
        return []

    train_items = int(attn_mlp_policy_shadow_mse_train_items_per_update)
    val_items = int(attn_mlp_policy_shadow_mse_val_items_per_update)
    train_items = max(1, train_items)
    val_items = max(0, val_items)
    needed = train_items + val_items
    if needed <= 0 or len(attn_mlp_policy_shadow_mse_item_buffer) < needed:
        return []

    selected = [attn_mlp_policy_shadow_mse_item_buffer.pop(0) for _ in range(needed)]
    for item_idx, item in enumerate(selected):
        item["split"] = "train" if item_idx < train_items else "val"
    return selected


def _attn_mlp_shadow_mse_weighted_order_loss(full_loss, prefix_loss):
    return (
        float(attn_mlp_policy_shadow_mse_full_weight) * float(full_loss)
        + float(attn_mlp_policy_shadow_mse_prefix_weight) * float(prefix_loss)
    )


def _attn_mlp_shadow_mse_linear_profile_loss(profile):
    profile = profile.float()
    weights = torch.linspace(1.0, 0.1, steps=int(profile.numel()), device=profile.device)
    weights = weights / weights.sum().clamp_min(1e-8)
    return float((weights * profile).sum().detach().float().item())


def _attn_mlp_shadow_mse_target_for_matrix(matrix, x_subset):
    raw_scores, reverse_scores, teacher_meta = _attn_mlp_teacher_fiedler_scores(matrix)
    raw_order = logits_to_order(raw_scores, mode="argsort_desc").detach().cpu().long().view(-1)
    reverse_order = logits_to_order(reverse_scores, mode="argsort_desc").detach().cpu().long().view(-1)
    orientation = str(attn_mlp_policy_shadow_mse_label_orientation).strip().lower()
    selected_name = "raw"
    selected_by = "configured_raw"
    raw_full = raw_prefix = raw_score = float("nan")
    reverse_full = reverse_prefix = reverse_score = float("nan")
    raw_linear = reverse_linear = float("nan")
    if orientation in {"reverse", "rev", "teacher_reverse"}:
        selected_name = "reverse"
        selected_by = "configured_reverse"
    elif orientation in {"original_best", "best_original", "original_tau"}:
        raw_original = _attn_mlp_original_order_metrics(raw_order) or {}
        reverse_original = _attn_mlp_original_order_metrics(reverse_order) or {}
        raw_tau = float(raw_original.get("tau_original_diagnostic", float("-inf")))
        reverse_tau = float(reverse_original.get("tau_original_diagnostic", float("-inf")))
        selected_name = "raw" if raw_tau >= reverse_tau else "reverse"
        selected_by = "max_teacher_original_tau_diagnostic"
    elif orientation in {"", "loss", "loss_profile", "teacher_loss", "lm_loss"}:
        if x_subset is None or int(x_subset.size(0)) <= 0:
            selected_name = "raw"
            selected_by = "fallback_raw_no_subset"
        else:
            raw_full, raw_prefix = _attn_mlp_full_and_prefix_loss_for_block_order(
                x_subset,
                raw_order.to(device=x_subset.device),
                prefix_k=int(attn_mlp_policy_shadow_mse_prefix_k),
            )
            reverse_full, reverse_prefix = _attn_mlp_full_and_prefix_loss_for_block_order(
                x_subset,
                reverse_order.to(device=x_subset.device),
                prefix_k=int(attn_mlp_policy_shadow_mse_prefix_k),
            )
            raw_score = _attn_mlp_shadow_mse_weighted_order_loss(raw_full, raw_prefix)
            reverse_score = _attn_mlp_shadow_mse_weighted_order_loss(reverse_full, reverse_prefix)
            selected_name = "raw" if raw_score <= reverse_score else "reverse"
            selected_by = "min_weighted_lm_loss"
    elif orientation in {"linear", "linear_profile", "linear_profile_loss", "teacher_linear_profile"}:
        if x_subset is None or int(x_subset.size(0)) <= 0:
            selected_name = "raw"
            selected_by = "fallback_raw_no_subset"
        else:
            raw_profile = _attn_mlp_block_loss_profile_for_block_order(
                x_subset,
                raw_order.to(device=x_subset.device),
            )
            reverse_profile = _attn_mlp_block_loss_profile_for_block_order(
                x_subset,
                reverse_order.to(device=x_subset.device),
            )
            prefix_k = max(1, min(int(attn_mlp_policy_shadow_mse_prefix_k), int(raw_profile.numel())))
            raw_full = float(raw_profile.mean().detach().float().item())
            raw_prefix = float(raw_profile[:prefix_k].mean().detach().float().item())
            reverse_full = float(reverse_profile.mean().detach().float().item())
            reverse_prefix = float(reverse_profile[:prefix_k].mean().detach().float().item())
            raw_linear = _attn_mlp_shadow_mse_linear_profile_loss(raw_profile)
            reverse_linear = _attn_mlp_shadow_mse_linear_profile_loss(reverse_profile)
            raw_score = raw_linear
            reverse_score = reverse_linear
            selected_name = "raw" if raw_score <= reverse_score else "reverse"
            selected_by = "min_linear_profile_loss"
    elif orientation in {"raw", "teacher_raw"}:
        selected_name = "raw"
        selected_by = "configured_raw"
    else:
        selected_name = "raw"
        selected_by = f"fallback_raw_unknown_orientation:{orientation}"

    if selected_name == "raw":
        selected_scores = raw_scores
        selected_order = raw_order
    else:
        selected_scores = reverse_scores
        selected_order = reverse_order

    stats = {
        "orientation": str(attn_mlp_policy_shadow_mse_label_orientation),
        "selected_sign": selected_name,
        "selected_reverse": bool(selected_name == "reverse"),
        "selected_by": selected_by,
        "raw_full_loss": float(raw_full),
        "raw_prefix_loss": float(raw_prefix),
        "raw_linear_profile_loss": float(raw_linear),
        "raw_weighted_loss": float(raw_score),
        "reverse_full_loss": float(reverse_full),
        "reverse_prefix_loss": float(reverse_prefix),
        "reverse_linear_profile_loss": float(reverse_linear),
        "reverse_weighted_loss": float(reverse_score),
        "loss_score_name": "linear_profile_loss"
        if orientation in {"linear", "linear_profile", "linear_profile_loss", "teacher_linear_profile"}
        else "weighted_prefix_full_loss",
        "loss_gap_raw_minus_reverse": float(raw_score - reverse_score)
        if math.isfinite(raw_score) and math.isfinite(reverse_score)
        else float("nan"),
        "teacher_meta": teacher_meta,
    }
    stats.update(_attn_mlp_prefix_order_diag("teacher_raw", raw_order))
    stats.update(_attn_mlp_prefix_order_diag("teacher_reverse", reverse_order))
    stats.update(_attn_mlp_prefix_order_diag("teacher_selected", selected_order))
    return selected_scores.detach().float().cpu(), selected_order.detach().cpu().long(), stats


def _attn_mlp_shadow_mse_prediction_stats(pred_scores, target_scores, selected_order):
    score_stats = _attn_mlp_score_similarity_stats(pred_scores, target_scores)
    pred_order = logits_to_order(torch.as_tensor(pred_scores).float(), mode=str(attn_mlp_policy_order_mode))
    order_stats = _attn_mlp_pairwise_order_stats(pred_order, selected_order)
    return {
        "mse": float(score_stats["mse"]),
        "mae": float(score_stats["mae"]),
        "pearson": float(score_stats["pearson"]),
        "score_pair_acc": float(score_stats["score_pair_acc"]),
        "score_pair_tau": float(score_stats["score_pair_tau"]),
        "order_pair_acc": float(order_stats["pair_acc"]),
        "order_pair_tau": float(order_stats["pair_tau"]),
    }


def _attn_mlp_shadow_mse_mean(values):
    finite = [float(value) for value in values if isinstance(value, (int, float, np.generic)) and math.isfinite(float(value))]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def _attn_mlp_shadow_mse_write_history(current_iter, stats, details):
    if not master_process or not isinstance(stats, dict):
        return
    row = {
        "version": 1,
        "iter": int(current_iter),
        "wall_time": float(time.time()),
        "train_loss_name": str(attn_mlp_policy_train_loss),
        "shadow_mse_enabled": bool(_attn_mlp_policy_shadow_mse_configured()),
        "shadow_mse_train_active": bool(_attn_mlp_policy_shadow_mse_train_active(current_iter)),
        "shadow_mse_start_iter": int(attn_mlp_policy_shadow_mse_start_iter),
        "shadow_mse_stop_iter": int(attn_mlp_policy_shadow_mse_stop_iter),
        "shadow_mse_train_frac": float(attn_mlp_policy_shadow_mse_train_frac),
        "shadow_mse_batches_per_item": int(attn_mlp_policy_shadow_mse_batches_per_item),
        "shadow_mse_train_items_per_update": int(attn_mlp_policy_shadow_mse_train_items_per_update),
        "shadow_mse_val_items_per_update": int(attn_mlp_policy_shadow_mse_val_items_per_update),
        "shadow_mse_train_samples_per_step": int(attn_mlp_policy_shadow_mse_train_samples_per_step),
        "shadow_mse_val_samples_per_step": int(attn_mlp_policy_shadow_mse_val_samples_per_step),
        "label_orientation": str(attn_mlp_policy_shadow_mse_label_orientation),
        "orientation_x_mode": str(attn_mlp_policy_shadow_mse_orientation_x_mode),
        "prefix_k": int(attn_mlp_policy_shadow_mse_prefix_k),
        "prefix_weight": float(attn_mlp_policy_shadow_mse_prefix_weight),
        "full_weight": float(attn_mlp_policy_shadow_mse_full_weight),
        "stats": stats,
    }
    if bool(attn_mlp_policy_order_history_include_scores) and details:
        compact = []
        for detail in details[:2]:
            compact_detail = {
                "split": str(detail.get("split", "")),
                "target_selected_sign": str(detail.get("target_stats", {}).get("selected_sign", "")),
                "target_selected_reverse": bool(detail.get("target_stats", {}).get("selected_reverse", False)),
                "target_scores": _attn_mlp_tensor_to_float_list(detail.get("target_scores", [])),
                "pred_scores": _attn_mlp_tensor_to_float_list(detail.get("pred_scores", [])),
                "selected_order_current": _attn_mlp_order_to_list(detail.get("selected_order")),
                "target_stats": detail.get("target_stats", {}),
                "prediction_stats": detail.get("prediction_stats", {}),
                "sample_count": int(detail.get("sample_count", 0)),
                "label_sample_count": int(detail.get("label_sample_count", 0)),
                "orientation_x_mode": str(detail.get("orientation_x_mode", "")),
            }
            compact.append(compact_detail)
        row["details"] = compact
    path = _attn_mlp_shadow_mse_log_file()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(_attn_mlp_json_safe(row), ensure_ascii=True, sort_keys=True) + "\n")


def _attn_mlp_shadow_mse_finalize_stats(
    current_iter,
    prepared,
    total_loss_value=None,
    grad_norm=0.0,
    train_enabled=0.0,
    frozen_after_stop=False,
    eval_after_stop=False,
):
    details = []
    train_metrics = []
    val_metrics = []
    selected_reverse = []
    raw_weighted = []
    reverse_weighted = []
    loss_gaps = []
    train_counts = []
    val_counts = []
    logit_stds = []
    with torch.no_grad():
        attn_mlp_policy.eval()
        for item, train_target, train_order, train_target_stats in prepared:
            if "matrix" in item:
                split = str(item.get("split", "train"))
                pred_logits = attn_mlp_policy(item["matrix"].detach().float().to(device=device))
                pred = torch.sigmoid(pred_logits.float()).detach().cpu().view(-1)
                pred_stats = _attn_mlp_shadow_mse_prediction_stats(pred, train_target, train_order)
                details.append(
                    {
                        "split": split,
                        "target_scores": train_target,
                        "pred_scores": pred,
                        "selected_order": train_order,
                        "target_stats": train_target_stats,
                        "prediction_stats": pred_stats,
                        "sample_count": int(item.get("sample_count", 0)),
                        "label_sample_count": int(item.get("label_sample_count", item.get("sample_count", 0))),
                        "orientation_x_mode": str(item.get("orientation_x_mode", "")),
                    }
                )
                if split == "val":
                    val_metrics.append(pred_stats)
                    val_counts.append(float(item.get("sample_count", 0)))
                else:
                    train_metrics.append(pred_stats)
                    selected_reverse.append(float(train_target_stats.get("selected_reverse", False)))
                    raw_weighted.append(float(train_target_stats.get("raw_weighted_loss", float("nan"))))
                    reverse_weighted.append(float(train_target_stats.get("reverse_weighted_loss", float("nan"))))
                    loss_gaps.append(float(train_target_stats.get("loss_gap_raw_minus_reverse", float("nan"))))
                    train_counts.append(float(item.get("sample_count", 0)))
                    logit_stds.append(float(pred_logits.detach().float().std(unbiased=False).item()))
                continue

            train_pred_logits = attn_mlp_policy(item["train_matrix"].detach().float().to(device=device))
            train_pred = torch.sigmoid(train_pred_logits.float()).detach().cpu().view(-1)
            train_pred_stats = _attn_mlp_shadow_mse_prediction_stats(train_pred, train_target, train_order)
            train_metrics.append(train_pred_stats)
            details.append(
                {
                    "split": "train",
                    "target_scores": train_target,
                    "pred_scores": train_pred,
                    "selected_order": train_order,
                    "target_stats": train_target_stats,
                    "prediction_stats": train_pred_stats,
                }
            )
            selected_reverse.append(float(train_target_stats.get("selected_reverse", False)))
            raw_weighted.append(float(train_target_stats.get("raw_weighted_loss", float("nan"))))
            reverse_weighted.append(float(train_target_stats.get("reverse_weighted_loss", float("nan"))))
            loss_gaps.append(float(train_target_stats.get("loss_gap_raw_minus_reverse", float("nan"))))
            train_counts.append(float(item.get("train_count", 0)))
            val_counts.append(float(item.get("val_count", 0)))
            logit_stds.append(float(train_pred_logits.detach().float().std(unbiased=False).item()))

            try:
                val_target, val_order, val_target_stats = _attn_mlp_shadow_mse_target_for_matrix(
                    item["val_matrix"],
                    item["val_x"],
                )
                val_pred_logits = attn_mlp_policy(item["val_matrix"].detach().float().to(device=device))
                val_pred = torch.sigmoid(val_pred_logits.float()).detach().cpu().view(-1)
                val_pred_stats = _attn_mlp_shadow_mse_prediction_stats(val_pred, val_target, val_order)
                val_metrics.append(val_pred_stats)
                details.append(
                    {
                        "split": "val",
                        "target_scores": val_target,
                        "pred_scores": val_pred,
                        "selected_order": val_order,
                        "target_stats": val_target_stats,
                        "prediction_stats": val_pred_stats,
                    }
                )
            except Exception as exc:
                if master_process:
                    print(f"attn-mlp shadow MSE val target error at iter {current_iter}: {exc}")

    def _metric_mean(metric_list, key):
        return _attn_mlp_shadow_mse_mean([metrics.get(key, float("nan")) for metrics in metric_list])

    if total_loss_value is None:
        loss_value = _metric_mean(train_metrics, "mse")
    elif torch.is_tensor(total_loss_value):
        loss_value = float(total_loss_value.detach().float().item())
    else:
        loss_value = float(total_loss_value)

    stats = {
        "attn_mlp_train_enabled": float(train_enabled),
        "attn_mlp_train_loss": float(loss_value),
        "attn_mlp_train_shadow_mse_active": float(train_enabled),
        "attn_mlp_train_shadow_mse_eval_after_stop": float(eval_after_stop),
        "attn_mlp_train_shadow_mse_frozen": float(frozen_after_stop),
        "attn_mlp_train_shadow_mse_items": float(len(prepared)),
        "attn_mlp_train_shadow_mse_train_items": float(len(train_metrics)),
        "attn_mlp_train_shadow_mse_val_items": float(len(val_metrics)),
        "attn_mlp_train_shadow_mse_train_count": _attn_mlp_shadow_mse_mean(train_counts),
        "attn_mlp_train_shadow_mse_val_count": _attn_mlp_shadow_mse_mean(val_counts),
        "attn_mlp_train_shadow_mse_train_loss": _metric_mean(train_metrics, "mse"),
        "attn_mlp_train_shadow_mse_train_mae": _metric_mean(train_metrics, "mae"),
        "attn_mlp_train_shadow_mse_train_pearson": _metric_mean(train_metrics, "pearson"),
        "attn_mlp_train_shadow_mse_train_score_pair_tau": _metric_mean(train_metrics, "score_pair_tau"),
        "attn_mlp_train_shadow_mse_train_order_pair_tau": _metric_mean(train_metrics, "order_pair_tau"),
        "attn_mlp_train_shadow_mse_val_loss": _metric_mean(val_metrics, "mse"),
        "attn_mlp_train_shadow_mse_val_mae": _metric_mean(val_metrics, "mae"),
        "attn_mlp_train_shadow_mse_val_pearson": _metric_mean(val_metrics, "pearson"),
        "attn_mlp_train_shadow_mse_val_score_pair_tau": _metric_mean(val_metrics, "score_pair_tau"),
        "attn_mlp_train_shadow_mse_val_order_pair_tau": _metric_mean(val_metrics, "order_pair_tau"),
        "attn_mlp_train_shadow_mse_selected_reverse_frac": _attn_mlp_shadow_mse_mean(selected_reverse),
        "attn_mlp_train_shadow_mse_raw_weighted_loss": _attn_mlp_shadow_mse_mean(raw_weighted),
        "attn_mlp_train_shadow_mse_reverse_weighted_loss": _attn_mlp_shadow_mse_mean(reverse_weighted),
        "attn_mlp_train_shadow_mse_loss_gap_raw_minus_reverse": _attn_mlp_shadow_mse_mean(loss_gaps),
        "attn_mlp_train_logit_std": _attn_mlp_shadow_mse_mean(logit_stds),
        "attn_mlp_train_grad_norm": float(grad_norm),
    }
    stats.update(_attn_mlp_shadow_mse_buffer_stats())
    _attn_mlp_shadow_mse_write_history(current_iter, stats, details)
    return stats


def _attn_mlp_train_policy_teacher_score_mse_split_items(
    items,
    current_iter,
    train_active,
    eval_after_stop_active,
    frozen_after_stop,
):
    global attn_mlp_policy_last_train_stats
    prepared = []
    losses = []
    attn_mlp_policy.train()
    for item in items:
        if "matrix" not in item:
            continue
        try:
            target_x = item.get("label_x", item.get("x"))
            target_scores, selected_order, target_stats = _attn_mlp_shadow_mse_target_for_matrix(
                item["matrix"],
                target_x,
            )
        except Exception as exc:
            if master_process:
                print(f"attn-mlp shadow MSE split target error at iter {current_iter}: {exc}")
            continue
        prepared.append((item, target_scores, selected_order, target_stats))
        if train_active and str(item.get("split", "train")) != "val":
            train_matrix = item["matrix"].detach().float().to(device=device)
            target = target_scores.detach().float().to(device=device)
            logits = attn_mlp_policy(train_matrix)
            pred = torch.sigmoid(logits.float()).view(-1)
            losses.append(F.mse_loss(pred, target.view(-1)))

    if eval_after_stop_active and not train_active:
        if not prepared:
            stats = {
                "attn_mlp_train_enabled": 0.0,
                "attn_mlp_train_shadow_mse_active": 0.0,
                "attn_mlp_train_shadow_mse_eval_after_stop": 1.0,
                "attn_mlp_train_shadow_mse_frozen": float(frozen_after_stop),
                "attn_mlp_train_shadow_mse_error": 1.0,
                "attn_mlp_train_shadow_mse_error_code": "no_valid_eval_targets",
            }
            stats.update(_attn_mlp_shadow_mse_buffer_stats())
            attn_mlp_policy_last_train_stats = stats
            return attn_mlp_policy_last_train_stats
        attn_mlp_policy_last_train_stats = _attn_mlp_shadow_mse_finalize_stats(
            current_iter,
            prepared,
            total_loss_value=None,
            grad_norm=0.0,
            train_enabled=0.0,
            frozen_after_stop=True,
            eval_after_stop=True,
        )
        return attn_mlp_policy_last_train_stats

    if not losses:
        stats = {
            "attn_mlp_train_enabled": 0.0,
            "attn_mlp_train_shadow_mse_active": 1.0,
            "attn_mlp_train_shadow_mse_eval_after_stop": 0.0,
            "attn_mlp_train_shadow_mse_frozen": float(frozen_after_stop),
            "attn_mlp_train_shadow_mse_error": 1.0,
            "attn_mlp_train_shadow_mse_error_code": "no_valid_train_targets",
        }
        stats.update(_attn_mlp_shadow_mse_buffer_stats())
        attn_mlp_policy_last_train_stats = stats
        return attn_mlp_policy_last_train_stats

    total_loss = torch.stack(losses).mean()
    attn_mlp_policy_optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    grad_norm = 0.0
    if float(attn_mlp_policy_grad_clip) > 0.0:
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
            attn_mlp_policy.parameters(),
            float(attn_mlp_policy_grad_clip),
        )
        grad_norm = float(grad_norm_tensor.item())
    attn_mlp_policy_optimizer.step()
    attn_mlp_policy.eval()

    attn_mlp_policy_last_train_stats = _attn_mlp_shadow_mse_finalize_stats(
        current_iter,
        prepared,
        total_loss_value=total_loss,
        grad_norm=float(grad_norm),
        train_enabled=1.0,
        frozen_after_stop=False,
        eval_after_stop=False,
    )
    return attn_mlp_policy_last_train_stats


def _attn_mlp_train_policy_teacher_score_mse(policy_input, current_iter=None):
    global attn_mlp_policy_last_train_stats
    current_iter = int(iter_num if current_iter is None else current_iter)
    frozen_after_stop = (
        int(attn_mlp_policy_shadow_mse_stop_iter) > 0
        and current_iter >= int(attn_mlp_policy_shadow_mse_stop_iter)
    )
    train_active = _attn_mlp_policy_shadow_mse_train_active(current_iter)
    eval_after_stop_active = _attn_mlp_policy_shadow_mse_eval_after_stop_active(current_iter)
    if not train_active and not eval_after_stop_active:
        attn_mlp_policy_last_train_stats = {
            "attn_mlp_train_enabled": 0.0,
            "attn_mlp_train_shadow_mse_active": 0.0,
            "attn_mlp_train_shadow_mse_eval_after_stop": 0.0,
            "attn_mlp_train_shadow_mse_frozen": float(frozen_after_stop),
            "attn_mlp_train_shadow_mse_start_iter": float(attn_mlp_policy_shadow_mse_start_iter),
            "attn_mlp_train_shadow_mse_stop_iter": float(attn_mlp_policy_shadow_mse_stop_iter),
        }
        attn_mlp_policy_last_train_stats.update(_attn_mlp_shadow_mse_buffer_stats())
        return attn_mlp_policy_last_train_stats
    items = list(attn_mlp_policy_shadow_mse_items_current or [])
    if not items:
        if _attn_mlp_policy_shadow_mse_grouped_items_enabled():
            stats = {
                "attn_mlp_train_enabled": 0.0,
                "attn_mlp_train_shadow_mse_active": float(train_active),
                "attn_mlp_train_shadow_mse_eval_after_stop": float(eval_after_stop_active),
                "attn_mlp_train_shadow_mse_frozen": float(frozen_after_stop),
                "attn_mlp_train_shadow_mse_waiting_for_items": 1.0,
                "attn_mlp_train_shadow_mse_error": 0.0,
            }
            stats.update(_attn_mlp_shadow_mse_buffer_stats())
            attn_mlp_policy_last_train_stats = stats
            return attn_mlp_policy_last_train_stats
        attn_mlp_policy_last_train_stats = {
            "attn_mlp_train_enabled": 0.0,
            "attn_mlp_train_shadow_mse_active": float(train_active),
            "attn_mlp_train_shadow_mse_eval_after_stop": float(eval_after_stop_active),
            "attn_mlp_train_shadow_mse_frozen": float(frozen_after_stop),
            "attn_mlp_train_shadow_mse_error": 1.0,
            "attn_mlp_train_shadow_mse_error_code": "no_shadow_items",
        }
        return attn_mlp_policy_last_train_stats

    if any(isinstance(item, dict) and "matrix" in item for item in items):
        return _attn_mlp_train_policy_teacher_score_mse_split_items(
            items,
            current_iter=current_iter,
            train_active=train_active,
            eval_after_stop_active=eval_after_stop_active,
            frozen_after_stop=frozen_after_stop,
        )

    if eval_after_stop_active and not train_active:
        prepared = []
        for item in items:
            try:
                target_scores, selected_order, target_stats = _attn_mlp_shadow_mse_target_for_matrix(
                    item["train_matrix"],
                    item["train_x"],
                )
            except Exception as exc:
                if master_process:
                    print(f"attn-mlp shadow MSE eval target error at iter {current_iter}: {exc}")
                continue
            prepared.append((item, target_scores, selected_order, target_stats))
        if not prepared:
            attn_mlp_policy_last_train_stats = {
                "attn_mlp_train_enabled": 0.0,
                "attn_mlp_train_shadow_mse_active": 0.0,
                "attn_mlp_train_shadow_mse_eval_after_stop": 1.0,
                "attn_mlp_train_shadow_mse_frozen": float(frozen_after_stop),
                "attn_mlp_train_shadow_mse_error": 1.0,
                "attn_mlp_train_shadow_mse_error_code": "no_valid_eval_targets",
            }
            return attn_mlp_policy_last_train_stats
        attn_mlp_policy_last_train_stats = _attn_mlp_shadow_mse_finalize_stats(
            current_iter,
            prepared,
            total_loss_value=None,
            grad_norm=0.0,
            train_enabled=0.0,
            frozen_after_stop=True,
            eval_after_stop=True,
        )
        return attn_mlp_policy_last_train_stats

    attn_mlp_policy.train()
    losses = []
    prepared = []
    for item in items:
        try:
            target_scores, selected_order, target_stats = _attn_mlp_shadow_mse_target_for_matrix(
                item["train_matrix"],
                item["train_x"],
            )
        except Exception as exc:
            if master_process:
                print(f"attn-mlp shadow MSE target error at iter {current_iter}: {exc}")
            continue
        train_matrix = item["train_matrix"].detach().float().to(device=device)
        target = target_scores.detach().float().to(device=device)
        logits = attn_mlp_policy(train_matrix)
        pred = torch.sigmoid(logits.float()).view(-1)
        losses.append(F.mse_loss(pred, target.view(-1)))
        prepared.append((item, target_scores, selected_order, target_stats))
    if not losses:
        attn_mlp_policy_last_train_stats = {
            "attn_mlp_train_enabled": 0.0,
            "attn_mlp_train_shadow_mse_active": 1.0,
            "attn_mlp_train_shadow_mse_eval_after_stop": 0.0,
            "attn_mlp_train_shadow_mse_frozen": float(frozen_after_stop),
            "attn_mlp_train_shadow_mse_error": 1.0,
            "attn_mlp_train_shadow_mse_error_code": "no_valid_targets",
        }
        return attn_mlp_policy_last_train_stats

    total_loss = torch.stack(losses).mean()
    attn_mlp_policy_optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    grad_norm = 0.0
    if float(attn_mlp_policy_grad_clip) > 0.0:
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
            attn_mlp_policy.parameters(),
            float(attn_mlp_policy_grad_clip),
        )
        grad_norm = float(grad_norm_tensor.item())
    attn_mlp_policy_optimizer.step()
    attn_mlp_policy.eval()

    attn_mlp_policy_last_train_stats = _attn_mlp_shadow_mse_finalize_stats(
        current_iter,
        prepared,
        total_loss_value=total_loss,
        grad_norm=float(grad_norm),
        train_enabled=1.0,
        frozen_after_stop=False,
        eval_after_stop=False,
    )
    return attn_mlp_policy_last_train_stats


def _attn_mlp_prefix_order_diag(prefix, order_current):
    metrics = _attn_mlp_original_order_metrics(order_current) or {}
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _attn_mlp_policy_teacher_diag_due():
    if not bool(attn_mlp_policy_teacher_diag_enabled) or not master_process:
        return False
    interval = int(attn_mlp_policy_teacher_diag_interval)
    if interval <= 0:
        return False
    if int(attn_mlp_policy_updates) <= 0:
        return False
    return int(attn_mlp_policy_updates) % interval == 0


def _attn_mlp_policy_compute_teacher_diag(raw_logits, order_logits, cached_order, policy_input, current_iter):
    if not _attn_mlp_policy_teacher_diag_due():
        return None
    raw_logits_cpu = raw_logits.detach().float().cpu().view(-1)
    order_logits_cpu = order_logits.detach().float().cpu().view(-1)
    cached_order_cpu = cached_order.detach().cpu().long().view(-1)
    raw_order = logits_to_order(raw_logits_cpu, mode=str(attn_mlp_policy_order_mode)).detach().cpu().long().view(-1)
    order_logits_order = logits_to_order(order_logits_cpu, mode=str(attn_mlp_policy_order_mode)).detach().cpu().long().view(-1)
    mlp_scores = torch.sigmoid(raw_logits_cpu)
    try:
        teacher_raw_scores, teacher_reverse_scores, teacher_meta = _attn_mlp_teacher_fiedler_scores(policy_input)
    except Exception as exc:
        return {
            "enabled": True,
            "iter": int(current_iter),
            "policy_update": int(attn_mlp_policy_updates),
            "error": str(exc),
        }

    teacher_raw_order = logits_to_order(teacher_raw_scores, mode="argsort_desc").detach().cpu().long().view(-1)
    teacher_reverse_order = logits_to_order(teacher_reverse_scores, mode="argsort_desc").detach().cpu().long().view(-1)
    raw_score_stats = _attn_mlp_score_similarity_stats(mlp_scores, teacher_raw_scores)
    reverse_score_stats = _attn_mlp_score_similarity_stats(mlp_scores, teacher_reverse_scores)

    orientation = str(attn_mlp_policy_teacher_diag_orientation).strip().lower()
    if orientation in {"", "match", "match_mlp", "mse", "best_mse"}:
        selected_name = "raw" if raw_score_stats["mse"] <= reverse_score_stats["mse"] else "reverse"
        selected_by = "min_mse_to_sigmoid_raw_logits"
    elif orientation in {"raw", "teacher_raw"}:
        selected_name = "raw"
        selected_by = "configured_raw"
    elif orientation in {"reverse", "rev", "teacher_reverse"}:
        selected_name = "reverse"
        selected_by = "configured_reverse"
    elif orientation in {"original_best", "best_original", "original_tau"}:
        raw_original = _attn_mlp_original_order_metrics(teacher_raw_order) or {}
        reverse_original = _attn_mlp_original_order_metrics(teacher_reverse_order) or {}
        raw_tau = float(raw_original.get("tau_original_diagnostic", float("-inf")))
        reverse_tau = float(reverse_original.get("tau_original_diagnostic", float("-inf")))
        selected_name = "raw" if raw_tau >= reverse_tau else "reverse"
        selected_by = "max_teacher_original_tau_diagnostic"
    else:
        selected_name = "raw" if raw_score_stats["mse"] <= reverse_score_stats["mse"] else "reverse"
        selected_by = f"fallback_min_mse_for_unknown_orientation:{orientation}"

    if selected_name == "raw":
        selected_scores = teacher_raw_scores
        selected_order = teacher_raw_order
        selected_score_stats = raw_score_stats
    else:
        selected_scores = teacher_reverse_scores
        selected_order = teacher_reverse_order
        selected_score_stats = reverse_score_stats

    cached_order_stats = _attn_mlp_pairwise_order_stats(cached_order_cpu, selected_order)
    raw_order_stats = _attn_mlp_pairwise_order_stats(raw_order, selected_order)
    order_logits_order_stats = _attn_mlp_pairwise_order_stats(order_logits_order, selected_order)

    diag = {
        "enabled": True,
        "iter": int(current_iter),
        "policy_update": int(attn_mlp_policy_updates),
        "attention_update": int(attn_mlp_policy_attention_updates),
        "orientation": str(attn_mlp_policy_teacher_diag_orientation),
        "selected_sign": selected_name,
        "selected_reverse": bool(selected_name == "reverse"),
        "selected_by": selected_by,
        "mlp_score_source": "sigmoid(raw_logits)",
        "teacher_score_source": "pairwise_max_fiedler_minmax_priority",
        "mse_raw": float(raw_score_stats["mse"]),
        "mae_raw": float(raw_score_stats["mae"]),
        "pearson_raw": float(raw_score_stats["pearson"]),
        "score_pair_acc_raw": float(raw_score_stats["score_pair_acc"]),
        "score_pair_tau_raw": float(raw_score_stats["score_pair_tau"]),
        "mse_reverse": float(reverse_score_stats["mse"]),
        "mae_reverse": float(reverse_score_stats["mae"]),
        "pearson_reverse": float(reverse_score_stats["pearson"]),
        "score_pair_acc_reverse": float(reverse_score_stats["score_pair_acc"]),
        "score_pair_tau_reverse": float(reverse_score_stats["score_pair_tau"]),
        "mse_match": float(selected_score_stats["mse"]),
        "mae_match": float(selected_score_stats["mae"]),
        "pearson_match": float(selected_score_stats["pearson"]),
        "score_pair_acc_match": float(selected_score_stats["score_pair_acc"]),
        "score_pair_tau_match": float(selected_score_stats["score_pair_tau"]),
        "score_pair_count": float(selected_score_stats["score_pair_count"]),
        "cached_order_pair_acc_match": float(cached_order_stats["pair_acc"]),
        "cached_order_tau_match": float(cached_order_stats["pair_tau"]),
        "cached_order_pair_count": float(cached_order_stats["pair_count"]),
        "raw_order_pair_acc_match": float(raw_order_stats["pair_acc"]),
        "raw_order_tau_match": float(raw_order_stats["pair_tau"]),
        "order_logits_order_pair_acc_match": float(order_logits_order_stats["pair_acc"]),
        "order_logits_order_tau_match": float(order_logits_order_stats["pair_tau"]),
        "teacher_meta": teacher_meta,
    }
    diag.update(_attn_mlp_prefix_order_diag("mlp_cached", cached_order_cpu))
    diag.update(_attn_mlp_prefix_order_diag("mlp_raw", raw_order))
    diag.update(_attn_mlp_prefix_order_diag("teacher_raw", teacher_raw_order))
    diag.update(_attn_mlp_prefix_order_diag("teacher_reverse", teacher_reverse_order))
    diag.update(_attn_mlp_prefix_order_diag("teacher_match", selected_order))
    if bool(attn_mlp_policy_teacher_diag_include_scores):
        diag["mlp_raw_scores_sigmoid"] = _attn_mlp_tensor_to_float_list(mlp_scores)
        diag["mlp_order_scores"] = _attn_mlp_tensor_to_float_list(order_logits_cpu)
        diag["teacher_raw_scores"] = _attn_mlp_tensor_to_float_list(teacher_raw_scores)
        diag["teacher_reverse_scores"] = _attn_mlp_tensor_to_float_list(teacher_reverse_scores)
        diag["teacher_match_scores"] = _attn_mlp_tensor_to_float_list(selected_scores)
        diag["teacher_match_order_current"] = _attn_mlp_order_to_list(selected_order)
    return diag


def _attn_mlp_teacher_diag_stats_for_wandb(diag):
    if not isinstance(diag, dict) or diag.get("error"):
        if isinstance(diag, dict) and diag.get("error"):
            return {"attn_mlp_teacher_diag_error": 1.0}
        return {}
    stats = {}
    scalar_keys = {
        "policy_update",
        "attention_update",
        "selected_reverse",
        "mse_raw",
        "mae_raw",
        "pearson_raw",
        "score_pair_acc_raw",
        "score_pair_tau_raw",
        "mse_reverse",
        "mae_reverse",
        "pearson_reverse",
        "score_pair_acc_reverse",
        "score_pair_tau_reverse",
        "mse_match",
        "mae_match",
        "pearson_match",
        "score_pair_acc_match",
        "score_pair_tau_match",
        "score_pair_count",
        "cached_order_pair_acc_match",
        "cached_order_tau_match",
        "cached_order_pair_count",
        "raw_order_pair_acc_match",
        "raw_order_tau_match",
        "order_logits_order_pair_acc_match",
        "order_logits_order_tau_match",
        "mlp_cached_tau_original_diagnostic",
        "mlp_raw_tau_original_diagnostic",
        "teacher_raw_tau_original_diagnostic",
        "teacher_reverse_tau_original_diagnostic",
        "teacher_match_tau_original_diagnostic",
        "teacher_match_kendall_distance_original",
        "teacher_match_normalized_kendall_distance_original",
    }
    for key in scalar_keys:
        if key in diag:
            value = diag[key]
            if isinstance(value, bool):
                stats[f"attn_mlp_teacher_diag_{key}"] = float(value)
            elif isinstance(value, (int, float, np.generic)):
                stats[f"attn_mlp_teacher_diag_{key}"] = float(value)
    meta = diag.get("teacher_meta")
    if isinstance(meta, dict):
        for key in (
            "laplacian_eigval0",
            "laplacian_fiedler_eigval",
            "laplacian_next_eigval",
            "spectral_gap",
            "affinity_sum",
            "affinity_density",
            "vector_std",
        ):
            value = meta.get(key)
            if isinstance(value, (int, float, np.generic)):
                stats[f"attn_mlp_teacher_diag_{key}"] = float(value)
    return stats


def _attn_mlp_policy_order_history_file():
    configured = str(attn_mlp_policy_order_history_path).strip()
    if configured:
        return configured
    input_attn_dir = str(attn_mlp_policy_log_input_attn_out_dir).strip()
    if input_attn_dir:
        return os.path.join(input_attn_dir, "attn_mlp_policy_order_history.jsonl")
    return os.path.join(out_dir, "attn_mlp_policy_order_history.jsonl")


def _attn_mlp_policy_order_history_due():
    if not bool(attn_mlp_policy_order_history_enabled) or not master_process:
        return False
    interval = int(attn_mlp_policy_order_history_interval)
    if interval <= 0:
        return False
    if int(attn_mlp_policy_updates) <= 0:
        return False
    return int(attn_mlp_policy_updates) % interval == 0


def _attn_mlp_policy_order_history_order_payload(prefix, order_current):
    order_list = _attn_mlp_order_to_list(order_current)
    if order_list is None:
        return {}
    metrics = _attn_mlp_original_order_metrics(order_list) or {}
    payload = {f"{prefix}_order_current": order_list}
    for key, value in metrics.items():
        payload[f"{prefix}_{key}"] = value
    return payload


def _attn_mlp_policy_write_order_history(
    current_iter,
    raw_logits,
    order_logits,
    cached_order,
    policy_input,
    trained_this_update,
    teacher_diag=None,
):
    if not _attn_mlp_policy_order_history_due():
        return
    raw_logits_cpu = raw_logits.detach().float().cpu().view(-1)
    order_logits_cpu = order_logits.detach().float().cpu().view(-1)
    cached_order_cpu = cached_order.detach().cpu().long().view(-1)
    raw_order = logits_to_order(raw_logits.detach().float(), mode=str(attn_mlp_policy_order_mode)).detach().cpu().long()
    reverse_order = torch.flip(cached_order_cpu, dims=[0])
    raw_reverse_order = torch.flip(raw_order.view(-1), dims=[0])
    score_source = "raw_logits"
    if (
        bool(attn_mlp_policy_logits_ema_enabled)
        and int(current_iter) >= int(attn_mlp_policy_logits_ema_start_iter)
        and attn_mlp_policy_logits_ema is not None
    ):
        score_source = "logits_ema"

    row = {
        "version": 1,
        "iter": int(current_iter),
        "wall_time": float(time.time()),
        "policy_update": int(attn_mlp_policy_updates),
        "attention_update": int(attn_mlp_policy_attention_updates),
        "trained_this_update": bool(trained_this_update),
        "policy_updates_frozen": bool(_attn_mlp_policy_updates_frozen_for_iter(current_iter)),
        "start_iter": int(attn_mlp_policy_start_iter),
        "update_every": int(attn_mlp_policy_update_every),
        "update_stop_iter": int(attn_mlp_policy_update_stop_iter),
        "layer": int(attn_mlp_policy_layer),
        "head": int(attn_mlp_policy_head),
        "export_type": str(attn_mlp_policy_export_type),
        "feature_mode": str(attn_mlp_policy_feature_mode),
        "input_normalization": str(attn_mlp_policy_input_normalization),
        "order_mode": str(attn_mlp_policy_order_mode),
        "score_source_for_cached_order": score_source,
        "attention_ema_enabled": bool(attn_mlp_policy_attention_ema_enabled),
        "attention_ema_decay": float(attn_mlp_policy_ema_decay),
        "logits_ema_enabled": bool(attn_mlp_policy_logits_ema_enabled),
        "logits_ema_decay": float(attn_mlp_policy_logits_ema_decay),
        "logits_ema_normalize": str(attn_mlp_policy_logits_ema_normalize),
        "logits_ema_start_iter": int(attn_mlp_policy_logits_ema_start_iter),
        "permute_data": bool(permute_data),
        "permute_seed": int(permute_seed),
        "original_diagnostic_note": (
            "current-frame orders are the policy orders used by training; "
            "original diagnostics remap current block ids through fixed_block_perm and are not used by the policy."
        ),
    }
    row.update(_attn_mlp_policy_order_history_order_payload("cached", cached_order_cpu))
    row.update(_attn_mlp_policy_order_history_order_payload("raw", raw_order))
    row.update(_attn_mlp_policy_order_history_order_payload("cached_reverse", reverse_order))
    row.update(_attn_mlp_policy_order_history_order_payload("raw_reverse", raw_reverse_order))
    row["raw_order_matches_cached_order"] = bool(torch.equal(raw_order.view(-1), cached_order_cpu))

    if bool(attn_mlp_policy_order_history_include_scores):
        row["raw_scores"] = _attn_mlp_tensor_to_float_list(raw_logits_cpu)
        row["order_scores"] = _attn_mlp_tensor_to_float_list(order_logits_cpu)
        if attn_mlp_policy_logits_ema is not None:
            row["logits_ema_scores"] = _attn_mlp_tensor_to_float_list(attn_mlp_policy_logits_ema)
    if bool(attn_mlp_policy_order_history_include_input_stats):
        try:
            row["policy_input_stats"] = _attn_mlp_input_attn_stats(policy_input.detach().float())
        except Exception as exc:
            row["policy_input_stats_error"] = str(exc)
    if teacher_diag is not None:
        row["teacher_diag"] = teacher_diag

    path = _attn_mlp_policy_order_history_file()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(_attn_mlp_json_safe(row), ensure_ascii=True, sort_keys=True) + "\n")


def _attn_mlp_input_attn_log_dir():
    if str(attn_mlp_policy_log_input_attn_out_dir).strip():
        return str(attn_mlp_policy_log_input_attn_out_dir)
    return os.path.join(out_dir, "attn_mlp_input_attention")


def _attn_mlp_input_attn_vmax(matrix_np, percentile_override=None):
    values = np.asarray(matrix_np, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    abs_values = np.abs(finite)
    percentile = (
        float(attn_mlp_policy_log_input_attn_vmax_percentile)
        if percentile_override is None
        else float(percentile_override)
    )
    if percentile > 0.0 and percentile < 100.0:
        vmax = float(np.percentile(abs_values, percentile))
    else:
        vmax = float(np.max(abs_values))
    if not math.isfinite(vmax) or vmax <= 0.0:
        vmax = float(np.max(abs_values)) if abs_values.size else 1.0
    return max(vmax, 1e-6)


def _attn_mlp_input_attn_figure(
    matrix_np,
    current_iter,
    vmax,
    frame_name="current_l2r",
    title_context=None,
    cmap=None,
    title_prefix="Attn-MLP input attention",
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = str(frame_name or "current_l2r")
    is_original = frame in {"original", "original_l2r", "true_original"}
    frame_label = "original L2R" if is_original else "current L2R"
    if title_context is None:
        title_context = (
            f"L{int(attn_mlp_policy_layer)}H{int(attn_mlp_policy_head)} "
            f"{str(attn_mlp_policy_export_type)} {str(attn_mlp_policy_feature_mode)}"
        )
    fig, ax = plt.subplots(figsize=(5.2, 4.6), constrained_layout=True)
    image = ax.imshow(
        matrix_np,
        cmap=str(attn_mlp_policy_log_input_attn_cmap if cmap is None else cmap),
        vmin=-float(vmax),
        vmax=float(vmax),
        interpolation="nearest",
    )
    ax.set_title(
        f"{str(title_prefix)} ({frame_label}) | "
        f"iter={int(current_iter)} {str(title_context)}",
        fontsize=10,
    )
    axis_label = "original block" if is_original else "current block"
    ax.set_xlabel(f"key/{axis_label}")
    ax.set_ylabel(f"query/{axis_label}")
    fig.colorbar(image, ax=ax, shrink=0.82, pad=0.02)
    return fig


def _attn_mlp_input_attn_original_l2r_matrix(matrix_cpu):
    if not permute_data or fixed_block_perm is None:
        return None
    if matrix_cpu.ndim != 2 or matrix_cpu.size(0) != matrix_cpu.size(1):
        return None
    if int(matrix_cpu.size(0)) != int(num_blocks):
        return None
    mapper = fixed_block_perm.to(dtype=torch.long, device=matrix_cpu.device)
    original_to_current = invert_permutation(mapper)
    return matrix_cpu.index_select(0, original_to_current).index_select(1, original_to_current)


def _attn_mlp_input_attn_stats(matrix):
    values = matrix.detach().float()
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "abs_mean": float("nan"),
            "abs_max": float("nan"),
            "skew_abs_mean": float("nan"),
        }
    skew = values - values.t() if values.ndim == 2 and values.size(0) == values.size(1) else None
    skew_abs_mean = (
        float(skew.detach().float().abs().mean().item())
        if skew is not None
        else float("nan")
    )
    return {
        "mean": float(finite.mean().item()),
        "std": float(finite.std(unbiased=False).item()),
        "min": float(finite.min().item()),
        "max": float(finite.max().item()),
        "abs_mean": float(finite.abs().mean().item()),
        "abs_max": float(finite.abs().max().item()),
        "skew_abs_mean": float(skew_abs_mean),
    }


def _attn_mlp_log_input_attention_if_due(matrix, current_iter):
    global attn_mlp_policy_input_attn_last_log_iter
    if not bool(attn_mlp_policy_log_input_attn) or not master_process:
        return
    interval = int(attn_mlp_policy_log_input_attn_interval)
    current_iter = int(current_iter)
    if interval <= 0:
        return
    if int(attn_mlp_policy_input_attn_last_log_iter) >= 0:
        last_bucket = int(attn_mlp_policy_input_attn_last_log_iter) // interval
        current_bucket = int(current_iter) // interval
        if current_bucket <= last_bucket:
            return
    elif current_iter < interval:
        return
    wandb_module = globals().get("wandb", None) if bool(wandb_log) else None
    if wandb_module is None and not bool(attn_mlp_policy_log_input_attn_save_latest):
        return
    if matrix is None:
        return
    matrix_cpu = matrix.detach().float().cpu()
    if matrix_cpu.ndim == 3:
        matrix_cpu = matrix_cpu[0]
    if matrix_cpu.ndim != 2:
        return

    matrix_np = matrix_cpu.numpy().astype(np.float32, copy=False)
    vmax = _attn_mlp_input_attn_vmax(matrix_np)
    stats = _attn_mlp_input_attn_stats(matrix_cpu)
    prefix = str(attn_mlp_policy_log_input_attn_prefix).strip() or "attn_mlp_input_attn"
    figure = None
    original_figure = None
    try:
        figure = _attn_mlp_input_attn_figure(matrix_np, current_iter, vmax, frame_name="current_l2r")
        original_matrix_cpu = _attn_mlp_input_attn_original_l2r_matrix(matrix_cpu)
        original_matrix_np = None
        if original_matrix_cpu is not None:
            original_matrix_np = original_matrix_cpu.numpy().astype(np.float32, copy=False)
            original_figure = _attn_mlp_input_attn_figure(
                original_matrix_np,
                current_iter,
                vmax,
                frame_name="original_l2r",
            )
        if wandb_module is not None:
            payload = {
                "iter": int(current_iter),
                f"{prefix}/vmax": float(vmax),
                f"{prefix}/mean": float(stats["mean"]),
                f"{prefix}/std": float(stats["std"]),
                f"{prefix}/min": float(stats["min"]),
                f"{prefix}/max": float(stats["max"]),
                f"{prefix}/abs_mean": float(stats["abs_mean"]),
                f"{prefix}/abs_max": float(stats["abs_max"]),
                f"{prefix}/skew_abs_mean": float(stats["skew_abs_mean"]),
                f"{prefix}/heatmap": wandb_module.Image(
                    figure,
                    caption=(
                        f"iter={int(current_iter)} "
                        f"L{int(attn_mlp_policy_layer)}H{int(attn_mlp_policy_head)} "
                        f"{str(attn_mlp_policy_export_type)} {str(attn_mlp_policy_feature_mode)} "
                        "current_l2r"
                    ),
                ),
            }
            if original_figure is not None:
                payload[f"{prefix}/heatmap_original_l2r"] = wandb_module.Image(
                    original_figure,
                    caption=(
                        f"iter={int(current_iter)} "
                        f"L{int(attn_mlp_policy_layer)}H{int(attn_mlp_policy_head)} "
                        f"{str(attn_mlp_policy_export_type)} {str(attn_mlp_policy_feature_mode)} "
                        "original_l2r"
                    ),
                )
                payload[f"{prefix}/original_l2r_available"] = 1.0
            else:
                payload[f"{prefix}/original_l2r_available"] = 0.0
            wandb_module.log(payload)
        if bool(attn_mlp_policy_log_input_attn_save_latest):
            stats_dir = _attn_mlp_input_attn_log_dir()
            os.makedirs(stats_dir, exist_ok=True)
            np.save(os.path.join(stats_dir, "latest_input_attention.npy"), matrix_np)
            figure.savefig(
                os.path.join(stats_dir, "latest_input_attention.png"),
                dpi=180,
                bbox_inches="tight",
            )
            if original_matrix_np is not None and original_figure is not None:
                np.save(
                    os.path.join(stats_dir, "latest_input_attention_original_l2r.npy"),
                    original_matrix_np,
                )
                original_figure.savefig(
                    os.path.join(stats_dir, "latest_input_attention_original_l2r.png"),
                    dpi=180,
                    bbox_inches="tight",
                )
            with open(os.path.join(stats_dir, "latest_input_attention_stats.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "iter": int(current_iter),
                        "layer": int(attn_mlp_policy_layer),
                        "head": int(attn_mlp_policy_head),
                        "export_type": str(attn_mlp_policy_export_type),
                        "feature_mode": str(attn_mlp_policy_feature_mode),
                        "vmax": float(vmax),
                        "frame": "current_l2r",
                        "original_l2r_available": bool(original_matrix_np is not None),
                        "original_l2r_note": (
                            "Rows/cols are remapped with inverse_block_perm so index i is original block i. "
                            "This is diagnostic only; policy training still consumes current-frame attention."
                        ),
                        **stats,
                    },
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
        attn_mlp_policy_input_attn_last_log_iter = int(current_iter)
    finally:
        if figure is not None:
            import matplotlib.pyplot as plt
            plt.close(figure)
        if original_figure is not None:
            import matplotlib.pyplot as plt
            plt.close(original_figure)


def _online_spectral_input_attn_log_dir():
    if str(online_spectral_policy_log_input_attn_out_dir).strip():
        return str(online_spectral_policy_log_input_attn_out_dir)
    return os.path.join(out_dir, "online_spectral_input_attention")


def _online_spectral_log_input_attention_if_due(matrix, current_iter, layer_idx, head_idx, head_label, export_type):
    if not bool(online_spectral_policy_log_input_attn):
        return
    if not _online_spectral_policy_mode_enabled():
        return
    if not master_process:
        return
    interval = int(online_spectral_policy_log_input_attn_interval)
    if interval <= 0 or int(current_iter) % interval != 0:
        return
    if matrix is None:
        return

    matrix_cpu = matrix.detach().float().cpu()
    if matrix_cpu.ndim == 3:
        matrix_cpu = matrix_cpu[0]
    if matrix_cpu.ndim != 2:
        return

    layer_idx = int(layer_idx)
    head_idx = int(head_idx)
    head_label = str(head_label)
    head_tag = f"L{layer_idx}H{head_label}"
    matrix_np = matrix_cpu.numpy().astype(np.float32, copy=False)
    vmax = _attn_mlp_input_attn_vmax(
        matrix_np,
        percentile_override=float(online_spectral_policy_log_input_attn_vmax_percentile),
    )
    stats = _attn_mlp_input_attn_stats(matrix_cpu)
    original_matrix_cpu = _attn_mlp_input_attn_original_l2r_matrix(matrix_cpu)
    original_matrix_np = None
    if original_matrix_cpu is not None:
        original_matrix_np = original_matrix_cpu.numpy().astype(np.float32, copy=False)

    prefix_base = str(online_spectral_policy_log_input_attn_prefix).strip() or "online_spectral_input_attn"
    prefix = f"{prefix_base}/{head_tag}"
    title_context = (
        f"{head_tag} {str(export_type)} {str(head_signal_probe_candidate_source)} distribution_probe"
    )
    figure = None
    original_figure = None
    try:
        figure = _attn_mlp_input_attn_figure(
            matrix_np,
            current_iter,
            vmax,
            frame_name="current_l2r",
            title_context=title_context,
            cmap=str(online_spectral_policy_log_input_attn_cmap),
            title_prefix="Distribution input attention",
        )
        if original_matrix_np is not None:
            original_figure = _attn_mlp_input_attn_figure(
                original_matrix_np,
                current_iter,
                vmax,
                frame_name="original_l2r",
                title_context=title_context,
                cmap=str(online_spectral_policy_log_input_attn_cmap),
                title_prefix="Distribution input attention",
            )

        wandb_module = globals().get("wandb", None)
        if bool(wandb_log) and wandb_module is not None:
            payload = {
                "iter": int(current_iter),
                f"{prefix}/vmax": float(vmax),
                f"{prefix}/mean": float(stats["mean"]),
                f"{prefix}/std": float(stats["std"]),
                f"{prefix}/min": float(stats["min"]),
                f"{prefix}/max": float(stats["max"]),
                f"{prefix}/abs_mean": float(stats["abs_mean"]),
                f"{prefix}/abs_max": float(stats["abs_max"]),
                f"{prefix}/skew_abs_mean": float(stats["skew_abs_mean"]),
                f"{prefix}/layer": float(layer_idx),
                f"{prefix}/head": float(head_idx),
                f"{prefix}/head_is_mean": float(head_idx == -1),
                f"{prefix}/heatmap_current_l2r": wandb_module.Image(
                    figure,
                    caption=(
                        f"iter={int(current_iter)} {head_tag} "
                        f"{str(export_type)} current_l2r"
                    ),
                ),
                f"{prefix}/original_l2r_available": float(original_figure is not None),
            }
            if original_figure is not None:
                payload[f"{prefix}/heatmap_original_l2r"] = wandb_module.Image(
                    original_figure,
                    caption=(
                        f"iter={int(current_iter)} {head_tag} "
                        f"{str(export_type)} original_l2r"
                    ),
                )
            wandb_module.log(payload)

        if bool(online_spectral_policy_log_input_attn_save_latest):
            stats_dir = os.path.join(_online_spectral_input_attn_log_dir(), head_tag)
            os.makedirs(stats_dir, exist_ok=True)
            np.save(os.path.join(stats_dir, "latest_input_attention_current_l2r.npy"), matrix_np)
            figure.savefig(
                os.path.join(stats_dir, "latest_input_attention_current_l2r.png"),
                dpi=180,
                bbox_inches="tight",
            )
            if original_matrix_np is not None and original_figure is not None:
                np.save(
                    os.path.join(stats_dir, "latest_input_attention_original_l2r.npy"),
                    original_matrix_np,
                )
                original_figure.savefig(
                    os.path.join(stats_dir, "latest_input_attention_original_l2r.png"),
                    dpi=180,
                    bbox_inches="tight",
                )
            with open(os.path.join(stats_dir, "latest_input_attention_stats.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "iter": int(current_iter),
                        "layer": int(layer_idx),
                        "head": int(head_idx),
                        "head_label": str(head_label),
                        "head_aggregation": "mean" if int(head_idx) == -1 else "single",
                        "export_type": str(export_type),
                        "candidate_source": str(head_signal_probe_candidate_source),
                        "direct_asym_eig_mode": str(head_signal_probe_direct_asym_eig_mode),
                        "vmax": float(vmax),
                        "frame": "current_l2r",
                        "original_l2r_available": bool(original_matrix_np is not None),
                        "original_l2r_note": (
                            "Rows/cols are remapped with inverse_block_perm so index i is original block i. "
                            "This is diagnostic only; policy training still consumes current-frame attention."
                        ),
                        **stats,
                    },
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
    finally:
        if figure is not None:
            import matplotlib.pyplot as plt
            plt.close(figure)
        if original_figure is not None:
            import matplotlib.pyplot as plt
            plt.close(original_figure)


def _attn_mlp_update_policy_from_matrix(matrix, current_iter, probe_batches=None):
    global attn_mlp_policy_A_ema, attn_mlp_policy_cached_order, attn_mlp_policy_last_logits
    global attn_mlp_policy_updates, attn_mlp_policy_attention_updates
    global attn_mlp_policy_last_update_iter, attn_mlp_policy_last_attention_iter
    global attn_mlp_policy_latest_stats
    global attn_mlp_policy_last_train_stats
    global attn_mlp_policy_last_teacher_diag
    if attn_mlp_policy is None or matrix is None:
        return
    with torch.no_grad():
        matrix = _attn_mlp_sync_attention_matrix(matrix.detach().float())
        _attn_mlp_log_input_attention_if_due(matrix, current_iter)
        if bool(attn_mlp_policy_attention_ema_enabled):
            matrix_cpu = matrix.detach().float().cpu()
            if attn_mlp_policy_A_ema is None:
                attn_mlp_policy_A_ema = matrix_cpu
            else:
                decay = min(0.9999, max(0.0, float(attn_mlp_policy_ema_decay)))
                attn_mlp_policy_A_ema = attn_mlp_policy_A_ema * decay + matrix_cpu * (1.0 - decay)
            policy_input = attn_mlp_policy_A_ema
        else:
            policy_input = matrix
        attn_mlp_policy_attention_updates += 1
        attn_mlp_policy_last_attention_iter = int(current_iter)
    trained_this_update = False
    if bool(attn_mlp_policy_loss_stop_triggered):
        skipped_stats = dict(attn_mlp_policy_last_train_stats or {})
        skipped_stats.update(
            {
                "attn_mlp_train_enabled": 0.0,
                "attn_mlp_train_skipped_loss_stop": 1.0,
                "attn_mlp_train_lr": float(attn_mlp_policy_current_lr or 0.0),
            }
        )
        attn_mlp_policy_last_train_stats = skipped_stats
    else:
        _attn_mlp_set_optimizer_lr(current_iter)
        train_stats = _attn_mlp_train_policy_from_input(
            policy_input,
            probe_batches=probe_batches,
            current_iter=current_iter,
        )
        if isinstance(train_stats, dict):
            train_stats = dict(train_stats)
            train_stats["attn_mlp_train_skipped_loss_stop"] = 0.0
            train_stats["attn_mlp_train_lr"] = float(attn_mlp_policy_current_lr or 0.0)
            attn_mlp_policy_last_train_stats = train_stats
            trained_this_update = float(train_stats.get("attn_mlp_train_enabled", 0.0)) > 0.0
    if int(current_iter) + 1 < int(attn_mlp_policy_start_iter):
        _attn_mlp_refresh_latest_stats(policy_refreshed=False)
        return
    with torch.no_grad():
        attn_mlp_policy.eval()
        logits = attn_mlp_policy(policy_input.to(device=device))
        order_logits = _attn_mlp_order_logits_for_cache(logits, current_iter)
        order = logits_to_order(order_logits, mode=str(attn_mlp_policy_order_mode))
        if order.numel() != num_blocks or sorted(order.detach().cpu().tolist()) != list(range(num_blocks)):
            raise ValueError("Attn MLP produced an invalid block order.")
        attn_mlp_policy_last_logits = logits.detach().float().cpu()
        attn_mlp_policy_cached_order = order.detach().to(device="cpu", dtype=torch.long)
        attn_mlp_policy_updates += 1
        attn_mlp_policy_last_update_iter = int(current_iter)
        teacher_diag = _attn_mlp_policy_compute_teacher_diag(
            raw_logits=logits,
            order_logits=order_logits,
            cached_order=attn_mlp_policy_cached_order,
            policy_input=policy_input,
            current_iter=current_iter,
        )
        if teacher_diag is not None:
            attn_mlp_policy_last_teacher_diag = teacher_diag
        _attn_mlp_policy_write_order_history(
            current_iter,
            raw_logits=logits,
            order_logits=order_logits,
            cached_order=attn_mlp_policy_cached_order,
            policy_input=policy_input,
            trained_this_update=trained_this_update,
            teacher_diag=teacher_diag,
        )
    if trained_this_update:
        _attn_mlp_policy_loss_stop_update(attn_mlp_policy_last_train_stats, current_iter)
    _attn_mlp_refresh_latest_stats(policy_refreshed=True)
    log_every = int(attn_mlp_policy_log_interval)
    if master_process and log_every > 0 and int(attn_mlp_policy_updates) % log_every == 0:
        first16 = [int(v) for v in attn_mlp_policy_cached_order[:16].tolist()]
        print(
            "attn-mlp policy update "
            f"{int(attn_mlp_policy_updates)} at iter {int(current_iter)}: "
            f"cached_order_current_first16={first16}"
        )


def _attn_mlp_refresh_latest_stats(policy_refreshed=False):
    global attn_mlp_policy_latest_stats
    if not _attn_mlp_policy_mode_enabled():
        attn_mlp_policy_latest_stats = None
        return
    stats = {
        "attn_mlp_policy_active": float(_attn_mlp_policy_active_for_iter(iter_num)),
        "attn_mlp_policy_prob": float(_attn_mlp_policy_prob_for_iter(iter_num)),
        "attn_mlp_policy_updates": float(attn_mlp_policy_updates),
        "attn_mlp_policy_attention_updates": float(attn_mlp_policy_attention_updates),
        "attn_mlp_policy_attention_batches_per_update": float(attn_mlp_policy_attention_batches_per_update),
        "attn_mlp_policy_attention_matrix_buffer": float(len(attn_mlp_policy_matrix_buffer)),
        "attn_mlp_policy_fallback_count": float(attn_mlp_policy_fallback_count),
        "attn_mlp_policy_last_update_iter": float(attn_mlp_policy_last_update_iter),
        "attn_mlp_policy_last_attention_iter": float(attn_mlp_policy_last_attention_iter),
        "attn_mlp_policy_cached": float(attn_mlp_policy_cached_order is not None),
        "attn_mlp_policy_refreshed": float(bool(policy_refreshed)),
        "attn_mlp_policy_updates_frozen": float(_attn_mlp_policy_updates_frozen_for_iter(iter_num)),
        "attn_mlp_policy_current_lr": float(attn_mlp_policy_current_lr or 0.0),
        "attn_mlp_policy_loss_stop_enabled": float(bool(attn_mlp_policy_loss_stop_enabled)),
        "attn_mlp_policy_loss_stop_triggered": float(bool(attn_mlp_policy_loss_stop_triggered)),
        "attn_mlp_policy_loss_stop_iter": float(attn_mlp_policy_loss_stop_iter),
        "attn_mlp_policy_loss_stop_bad_steps": float(attn_mlp_policy_loss_stop_bad_steps),
        "attn_mlp_policy_loss_stop_checks": float(attn_mlp_policy_loss_stop_checks),
        "attn_mlp_policy_loss_stop_best": float(
            attn_mlp_policy_loss_stop_best if attn_mlp_policy_loss_stop_best is not None else float("nan")
        ),
        "attn_mlp_policy_loss_stop_best_iter": float(attn_mlp_policy_loss_stop_best_iter),
        "attn_mlp_policy_loss_stop_last_metric": float(
            attn_mlp_policy_loss_stop_last_metric
            if attn_mlp_policy_loss_stop_last_metric is not None
            else float("nan")
        ),
        "attn_mlp_policy_logits_ema_enabled": float(bool(attn_mlp_policy_logits_ema_enabled)),
        "attn_mlp_policy_logits_ema_available": float(attn_mlp_policy_logits_ema is not None),
        "attn_mlp_policy_logits_ema_decay": float(attn_mlp_policy_logits_ema_decay),
        "attn_mlp_attention_ema_enabled": float(bool(attn_mlp_policy_attention_ema_enabled)),
        "attn_mlp_policy_order_history_enabled": float(bool(attn_mlp_policy_order_history_enabled)),
        "attn_mlp_policy_order_history_interval": float(attn_mlp_policy_order_history_interval),
        "attn_mlp_policy_teacher_diag_enabled": float(bool(attn_mlp_policy_teacher_diag_enabled)),
        "attn_mlp_policy_teacher_diag_interval": float(attn_mlp_policy_teacher_diag_interval),
    }
    if bool(attn_mlp_policy_attention_ema_enabled) and attn_mlp_policy_A_ema is not None:
        finite = attn_mlp_policy_A_ema[torch.isfinite(attn_mlp_policy_A_ema)]
        if finite.numel() > 0:
            stats.update(
                {
                    "attn_mlp_A_ema_mean": float(finite.mean().item()),
                    "attn_mlp_A_ema_std": float(finite.std(unbiased=False).item()),
                    "attn_mlp_A_ema_min": float(finite.min().item()),
                    "attn_mlp_A_ema_max": float(finite.max().item()),
                }
            )
    if attn_mlp_policy_last_logits is not None:
        logits = attn_mlp_policy_last_logits.float()
        stats.update(
            {
                "attn_mlp_logits_mean": float(logits.mean().item()),
                "attn_mlp_logits_std": float(logits.std(unbiased=False).item()),
                "attn_mlp_logits_min": float(logits.min().item()),
                "attn_mlp_logits_max": float(logits.max().item()),
            }
        )
    if attn_mlp_policy_logits_ema is not None:
        logits_ema = attn_mlp_policy_logits_ema.float()
        stats.update(
            {
                "attn_mlp_logits_ema_mean": float(logits_ema.mean().item()),
                "attn_mlp_logits_ema_std": float(logits_ema.std(unbiased=False).item()),
                "attn_mlp_logits_ema_min": float(logits_ema.min().item()),
                "attn_mlp_logits_ema_max": float(logits_ema.max().item()),
            }
        )
    if attn_mlp_policy_last_train_stats is not None:
        stats.update(attn_mlp_policy_last_train_stats)
    if attn_mlp_policy_last_teacher_diag is not None:
        stats.update(_attn_mlp_teacher_diag_stats_for_wandb(attn_mlp_policy_last_teacher_diag))
    attn_mlp_policy_latest_stats = stats


def _attn_mlp_checkpoint_state():
    if not _attn_mlp_policy_mode_enabled():
        return None
    state = {
        "enabled": bool(_attn_mlp_policy_mode_enabled()),
        "path": str(attn_mlp_policy_path),
        "policy_config": attn_mlp_policy_config or {},
        "start_iter": int(attn_mlp_policy_start_iter),
        "layer": int(attn_mlp_policy_layer),
        "head": int(attn_mlp_policy_head),
        "use_global": bool(attn_mlp_policy_use_global),
        "export_type": str(attn_mlp_policy_export_type),
        "feature_mode": str(attn_mlp_policy_feature_mode),
        "feature_clip": float(attn_mlp_policy_feature_clip),
        "input_normalization": str(attn_mlp_policy_input_normalization),
        "attention_ema_enabled": bool(attn_mlp_policy_attention_ema_enabled),
        "attention_input_mode": (
            "attention_ema" if bool(attn_mlp_policy_attention_ema_enabled) else "current_matrix_no_ema"
        ),
        "ema_decay": float(attn_mlp_policy_ema_decay),
        "update_every": int(attn_mlp_policy_update_every),
        "update_stop_iter": int(attn_mlp_policy_update_stop_iter),
        "order_mode": str(attn_mlp_policy_order_mode),
        "collect_warmup_attention": bool(attn_mlp_policy_collect_warmup_attention),
        "start_prob": float(attn_mlp_policy_start_prob),
        "end_prob": float(attn_mlp_policy_end_prob),
        "anneal_start_iter": int(attn_mlp_policy_anneal_start_iter),
        "anneal_end_iter": int(attn_mlp_policy_anneal_end_iter),
        "prob_schedule": str(attn_mlp_policy_prob_schedule),
        "prob_points": str(attn_mlp_policy_prob_points),
        "logits_ema_enabled": bool(attn_mlp_policy_logits_ema_enabled),
        "logits_ema_decay": float(attn_mlp_policy_logits_ema_decay),
        "logits_ema_normalize": str(attn_mlp_policy_logits_ema_normalize),
        "logits_ema_start_iter": int(attn_mlp_policy_logits_ema_start_iter),
        "order_history_enabled": bool(attn_mlp_policy_order_history_enabled),
        "order_history_interval": int(attn_mlp_policy_order_history_interval),
        "order_history_path": str(attn_mlp_policy_order_history_path),
        "order_history_resolved_path": str(_attn_mlp_policy_order_history_file()),
        "order_history_include_scores": bool(attn_mlp_policy_order_history_include_scores),
        "order_history_include_input_stats": bool(attn_mlp_policy_order_history_include_input_stats),
        "teacher_diag_enabled": bool(attn_mlp_policy_teacher_diag_enabled),
        "teacher_diag_interval": int(attn_mlp_policy_teacher_diag_interval),
        "teacher_diag_orientation": str(attn_mlp_policy_teacher_diag_orientation),
        "teacher_diag_include_scores": bool(attn_mlp_policy_teacher_diag_include_scores),
        "lazy_init_enabled": bool(attn_mlp_policy_lazy_init_enabled),
        "lazy_init_iter": int(attn_mlp_policy_lazy_init_iter),
        "head_profile_accept_gate_enabled": bool(
            attn_mlp_policy_head_profile_accept_gate_enabled
        ),
        "head_profile_accept_antialigned_only": bool(
            attn_mlp_policy_head_profile_accept_antialigned_only
        ),
        "head_profile_accept_alignment_threshold": float(
            attn_mlp_policy_head_profile_accept_alignment_threshold
        ),
        "head_profile_accept_margin": float(attn_mlp_policy_head_profile_accept_margin),
        "head_profile_best_memory_enabled": bool(
            attn_mlp_policy_head_profile_best_memory_enabled
        ),
        "head_profile_best_memory_use_for_fixed": bool(
            attn_mlp_policy_head_profile_best_memory_use_for_fixed
        ),
        "random_init": bool(attn_mlp_policy_random_init),
        "freeze": bool(attn_mlp_policy_freeze),
        "input_channels": int(attn_mlp_policy_input_channels),
        "hidden_dims": str(attn_mlp_policy_hidden_dims),
        "dropout": float(attn_mlp_policy_dropout),
        "activation": str(attn_mlp_policy_activation),
        "lr": float(attn_mlp_policy_lr),
        "lr_anneal_enabled": bool(attn_mlp_policy_lr_anneal_enabled),
        "lr_anneal_start_iter": int(attn_mlp_policy_lr_anneal_start_iter),
        "lr_anneal_end_iter": int(attn_mlp_policy_lr_anneal_end_iter),
        "lr_anneal_min_lr": float(attn_mlp_policy_lr_anneal_min_lr),
        "lr_anneal_style": str(attn_mlp_policy_lr_anneal_style),
        "current_lr": float(attn_mlp_policy_current_lr or 0.0),
        "loss_stop_enabled": bool(attn_mlp_policy_loss_stop_enabled),
        "loss_stop_metric": str(attn_mlp_policy_loss_stop_metric),
        "loss_stop_start_iter": int(attn_mlp_policy_loss_stop_start_iter),
        "loss_stop_patience": int(attn_mlp_policy_loss_stop_patience),
        "loss_stop_min_delta": float(attn_mlp_policy_loss_stop_min_delta),
        "loss_stop_check_every": int(attn_mlp_policy_loss_stop_check_every),
        "loss_stop_best": (
            float(attn_mlp_policy_loss_stop_best)
            if attn_mlp_policy_loss_stop_best is not None
            else None
        ),
        "loss_stop_best_iter": int(attn_mlp_policy_loss_stop_best_iter),
        "loss_stop_bad_steps": int(attn_mlp_policy_loss_stop_bad_steps),
        "loss_stop_checks": int(attn_mlp_policy_loss_stop_checks),
        "loss_stop_triggered": bool(attn_mlp_policy_loss_stop_triggered),
        "loss_stop_iter": int(attn_mlp_policy_loss_stop_iter),
        "loss_stop_last_metric": (
            float(attn_mlp_policy_loss_stop_last_metric)
            if attn_mlp_policy_loss_stop_last_metric is not None
            else None
        ),
        "train_loss": str(attn_mlp_policy_train_loss),
        "axis_profile_weight": float(attn_mlp_policy_axis_profile_weight),
        "axis_profile_dir_weight": float(attn_mlp_policy_axis_profile_dir_weight),
        "axis_profile_dir_margin": float(attn_mlp_policy_axis_profile_dir_margin),
        "axis_profile_min_abs_q": float(attn_mlp_policy_axis_profile_min_abs_q),
        "axis_profile_score": str(attn_mlp_policy_axis_profile_score),
        "directed_ribbon_flow_sign": float(attn_mlp_policy_directed_ribbon_flow_sign),
        "directed_ribbon_rank_tau": float(attn_mlp_policy_directed_ribbon_rank_tau),
        "directed_ribbon_margin": float(attn_mlp_policy_directed_ribbon_margin),
        "directed_ribbon_band_width": float(attn_mlp_policy_directed_ribbon_band_width),
        "directed_ribbon_band_weight": float(attn_mlp_policy_directed_ribbon_band_weight),
        "logit_l2": float(attn_mlp_policy_logit_l2),
        "min_logit_std": float(attn_mlp_policy_min_logit_std),
        "std_floor_weight": float(attn_mlp_policy_std_floor_weight),
        "sampled_orders_per_state": int(attn_mlp_policy_sampled_orders_per_state),
        "random_baseline_orders": int(attn_mlp_policy_random_baseline_orders),
        "nll_states_per_update": int(attn_mlp_policy_nll_states_per_update),
        "pg_weight": float(attn_mlp_policy_pg_weight),
        "prefix_reward_weight": float(attn_mlp_policy_prefix_reward_weight),
        "prefix_k": int(attn_mlp_policy_prefix_k),
        "reward_scale_floor": float(attn_mlp_policy_reward_scale_floor),
        "advantage_clip": float(attn_mlp_policy_advantage_clip),
        "move_pref_weight": float(attn_mlp_policy_move_pref_weight),
        "move_pref_pairs_per_state": int(attn_mlp_policy_move_pref_pairs_per_state),
        "move_pref_window": int(attn_mlp_policy_move_pref_window),
        "move_pref_tau": float(attn_mlp_policy_move_pref_tau),
        "move_pref_margin": float(attn_mlp_policy_move_pref_margin),
        "move_pref_max_weight": float(attn_mlp_policy_move_pref_max_weight),
        "move_pref_prefix_weight": float(attn_mlp_policy_move_pref_prefix_weight),
        "attn_pair_weight": float(attn_mlp_policy_attn_pair_weight),
        "attn_pair_top_frac": float(attn_mlp_policy_attn_pair_top_frac),
        "attn_pair_min_z": float(attn_mlp_policy_attn_pair_min_z),
        "attn_pair_max_weight": float(attn_mlp_policy_attn_pair_max_weight),
        "attn_pair_tau": float(attn_mlp_policy_attn_pair_tau),
        "attn_close_weight": float(attn_mlp_policy_attn_close_weight),
        "attn_close_tau": float(attn_mlp_policy_attn_close_tau),
        "attn_close_margin": float(attn_mlp_policy_attn_close_margin),
        "min_entropy": float(attn_mlp_policy_min_entropy),
        "max_entropy": float(attn_mlp_policy_max_entropy),
        "entropy_floor_weight": float(attn_mlp_policy_entropy_floor_weight),
        "entropy_ceiling_weight": float(attn_mlp_policy_entropy_ceiling_weight),
        "head_profile_weight": float(attn_mlp_policy_head_profile_weight),
        "head_profile_start_iter": int(attn_mlp_policy_head_profile_start_iter),
        "head_profile_every": int(attn_mlp_policy_head_profile_every),
        "head_profile_tau": float(attn_mlp_policy_head_profile_tau),
        "head_profile_min_abs_q": float(attn_mlp_policy_head_profile_min_abs_q),
        "head_profile_max_weight": float(attn_mlp_policy_head_profile_max_weight),
        "head_profile_use_cached": bool(attn_mlp_policy_head_profile_use_cached),
        "policy_updates": int(attn_mlp_policy_updates),
        "attention_updates": int(attn_mlp_policy_attention_updates),
        "fallback_count": int(attn_mlp_policy_fallback_count),
        "last_update_iter": int(attn_mlp_policy_last_update_iter),
        "last_attention_iter": int(attn_mlp_policy_last_attention_iter),
        "input_attn_last_log_iter": int(attn_mlp_policy_input_attn_last_log_iter),
        "head_profile_last_iter": int(attn_mlp_policy_head_profile_last_iter),
        "head_profile_best_iter": int(attn_mlp_policy_head_profile_best_iter),
        "note": (
            "cached_order is in current-frame block ids/coordinates. "
            "AttnMLP consumes either the latest collected attention matrix directly "
            "or the optional attention EMA when attention_ema_enabled=True. "
            "Original-frame fields are diagnostics only and are not used by this policy."
        ),
    }
    if bool(attn_mlp_policy_attention_ema_enabled) and attn_mlp_policy_A_ema is not None:
        state["A_ema"] = attn_mlp_policy_A_ema.detach().cpu()
    if attn_mlp_policy_cached_order is not None:
        state["cached_order"] = attn_mlp_policy_cached_order.detach().cpu()
    if attn_mlp_policy_head_profile_cached_order is not None:
        state["head_profile_cached_order"] = attn_mlp_policy_head_profile_cached_order.detach().cpu()
    if attn_mlp_policy_head_profile_cached_q is not None:
        state["head_profile_cached_q"] = attn_mlp_policy_head_profile_cached_q.detach().cpu()
    if attn_mlp_policy_head_profile_best_order is not None:
        state["head_profile_best_order"] = attn_mlp_policy_head_profile_best_order.detach().cpu()
    if attn_mlp_policy_head_profile_best_q is not None:
        state["head_profile_best_q"] = attn_mlp_policy_head_profile_best_q.detach().cpu()
    if bool(attn_mlp_policy_save_orders) and attn_mlp_policy_cached_order is not None:
        order_current = [int(v) for v in attn_mlp_policy_cached_order.detach().cpu().tolist()]
        state["cached_order_current"] = order_current
        if permute_data and fixed_block_perm is not None:
            mapper = fixed_block_perm.to(dtype=torch.long, device=attn_mlp_policy_cached_order.device)
            state["cached_order_original"] = [
                int(v) for v in mapper[attn_mlp_policy_cached_order.long()].detach().cpu().tolist()
            ]
    if attn_mlp_policy_last_logits is not None:
        state["last_logits"] = attn_mlp_policy_last_logits.detach().cpu()
    if attn_mlp_policy_logits_ema is not None:
        state["logits_ema"] = attn_mlp_policy_logits_ema.detach().cpu()
    if attn_mlp_policy is not None and (bool(attn_mlp_policy_random_init) or not bool(attn_mlp_policy_freeze)):
        state["model_state_dict"] = {
            key: value.detach().cpu()
            for key, value in attn_mlp_policy.state_dict().items()
        }
    if attn_mlp_policy_optimizer is not None:
        state["optimizer_state_dict"] = attn_mlp_policy_optimizer.state_dict()
    if attn_mlp_policy_last_train_stats is not None:
        state["last_train_stats"] = dict(attn_mlp_policy_last_train_stats)
    if attn_mlp_policy_last_teacher_diag is not None:
        state["last_teacher_diag"] = _attn_mlp_json_safe(attn_mlp_policy_last_teacher_diag)
    return state


online_spectral_policy_config = None
online_spectral_policy_A_ema = None
online_spectral_policy_cached_order = None
online_spectral_policy_last_meta = None
online_spectral_policy_priority_ema = None
online_spectral_policy_last_priority_update = None
online_spectral_policy_map_order = None
online_spectral_policy_hybrid_order = None
online_spectral_policy_distribution_updates = 0
online_spectral_policy_last_candidate_weights = None
online_spectral_policy_last_distribution_meta = None
online_spectral_policy_map_order_changed_kendall = None
online_spectral_policy_updates = 0
online_spectral_policy_attention_updates = 0
online_spectral_policy_random_count = 0
online_spectral_policy_fallback_count = 0
online_spectral_policy_late_l2r_count = 0
online_spectral_policy_failed_updates = 0
online_spectral_policy_last_update_iter = -1
online_spectral_policy_last_attention_iter = -1
online_spectral_policy_last_attention_samples = 0
online_spectral_policy_latest_stats = None
online_spectral_policy_try19_bridge_updates = 0
online_spectral_policy_try19_bridge_last_meta = None


def _online_spectral_distribution_mode_enabled():
    return str(aogpt_train_mode) == "OnlineSpectralOrderDistribution"


def _online_spectral_policy_mode_enabled():
    return (
        bool(online_spectral_policy_enabled)
        or str(aogpt_train_mode) in {"OnlineSpectralFixedHeadOrder", "OnlineSpectralOrderDistribution"}
    )


def _kendall_tau_between_orders(order_a, order_b):
    values_a = [int(v) for v in torch.as_tensor(order_a, dtype=torch.long).view(-1).cpu().tolist()]
    values_b = [int(v) for v in torch.as_tensor(order_b, dtype=torch.long).view(-1).cpu().tolist()]
    if len(values_a) != len(values_b) or sorted(values_a) != sorted(values_b):
        raise ValueError("Kendall tau orders must contain the same items.")
    n = len(values_a)
    total_pairs = n * (n - 1) / 2.0
    if total_pairs <= 0:
        return 1.0
    rank_b = {value: idx for idx, value in enumerate(values_b)}
    projected = [rank_b[value] for value in values_a]
    inversions = 0
    for i in range(n):
        left = projected[i]
        for j in range(i + 1, n):
            if left > projected[j]:
                inversions += 1
    return float(1.0 - 2.0 * float(inversions) / float(total_pairs))


def _init_online_spectral_policy():
    global online_spectral_policy_config
    global online_spectral_policy_A_ema, online_spectral_policy_cached_order, online_spectral_policy_last_meta
    global online_spectral_policy_priority_ema, online_spectral_policy_last_priority_update
    global online_spectral_policy_map_order, online_spectral_policy_hybrid_order
    global online_spectral_policy_distribution_updates
    global online_spectral_policy_last_candidate_weights, online_spectral_policy_last_distribution_meta
    global online_spectral_policy_map_order_changed_kendall
    global online_spectral_policy_updates, online_spectral_policy_attention_updates
    global online_spectral_policy_random_count, online_spectral_policy_fallback_count
    global online_spectral_policy_late_l2r_count
    global online_spectral_policy_failed_updates
    global online_spectral_policy_last_update_iter, online_spectral_policy_last_attention_iter
    global online_spectral_policy_last_attention_samples
    global online_spectral_policy_try19_bridge_updates, online_spectral_policy_try19_bridge_last_meta
    if not _online_spectral_policy_mode_enabled():
        return
    if str(aogpt_train_mode) not in {"OnlineSpectralFixedHeadOrder", "OnlineSpectralOrderDistribution"}:
        raise ValueError(
            "online_spectral_policy_enabled=True currently expects "
            "aogpt_train_mode='OnlineSpectralFixedHeadOrder' or 'OnlineSpectralOrderDistribution'."
        )
    online_spectral_policy_config = FixedHeadSpectralPolicyConfig(
        num_components=int(online_spectral_policy_num_components),
        component_pairs=str(online_spectral_policy_component_pairs),
        num_angles=int(online_spectral_policy_num_angles),
        k_values=str(online_spectral_policy_k_values),
        group_methods=str(online_spectral_policy_group_methods),
        threshold_percentile=float(online_spectral_policy_threshold_percentile),
        transform=str(online_spectral_policy_transform),
        temperature=float(online_spectral_policy_temperature),
        direction_lambdas=str(online_spectral_policy_direction_lambdas),
        directed_score_weight=float(online_spectral_policy_directed_score_weight),
        band_quality_weight=float(online_spectral_policy_band_quality_weight),
        score_adjacency_sym=str(online_spectral_policy_score_adjacency_sym),
    )
    if resume_online_spectral_policy_state:
        state = resume_online_spectral_policy_state
        if state.get("A_ema") is not None:
            online_spectral_policy_A_ema = torch.as_tensor(state["A_ema"], dtype=torch.float32, device="cpu")
        if state.get("cached_order") is not None:
            online_spectral_policy_cached_order = torch.as_tensor(
                state["cached_order"],
                dtype=torch.long,
                device="cpu",
            )
        if state.get("priority_ema") is not None:
            online_spectral_policy_priority_ema = torch.as_tensor(
                state["priority_ema"],
                dtype=torch.float32,
                device="cpu",
            )
        if state.get("priority_update") is not None:
            online_spectral_policy_last_priority_update = torch.as_tensor(
                state["priority_update"],
                dtype=torch.float32,
                device="cpu",
            )
        if state.get("map_order") is not None:
            online_spectral_policy_map_order = torch.as_tensor(
                state["map_order"],
                dtype=torch.long,
                device="cpu",
            )
        if state.get("hybrid_order") is not None:
            online_spectral_policy_hybrid_order = torch.as_tensor(
                state["hybrid_order"],
                dtype=torch.long,
                device="cpu",
            )
        if state.get("last_candidate_weights") is not None:
            online_spectral_policy_last_candidate_weights = torch.as_tensor(
                state["last_candidate_weights"],
                dtype=torch.float32,
                device="cpu",
            )
        online_spectral_policy_last_distribution_meta = state.get("last_distribution_meta")
        online_spectral_policy_distribution_updates = int(state.get("distribution_updates", 0))
        if state.get("map_order_changed_kendall") is not None:
            online_spectral_policy_map_order_changed_kendall = float(state.get("map_order_changed_kendall"))
        online_spectral_policy_last_meta = state.get("last_meta")
        online_spectral_policy_updates = int(state.get("policy_updates", 0))
        online_spectral_policy_attention_updates = int(state.get("attention_updates", 0))
        online_spectral_policy_random_count = int(state.get("random_count", 0))
        online_spectral_policy_fallback_count = int(state.get("fallback_count", 0))
        online_spectral_policy_late_l2r_count = int(state.get("late_l2r_count", 0))
        online_spectral_policy_failed_updates = int(state.get("failed_updates", 0))
        online_spectral_policy_last_update_iter = int(state.get("last_update_iter", -1))
        online_spectral_policy_last_attention_iter = int(state.get("last_attention_iter", -1))
        online_spectral_policy_last_attention_samples = int(state.get("last_attention_samples", 0))
        online_spectral_policy_try19_bridge_updates = int(state.get("try19_bridge_updates", 0))
        online_spectral_policy_try19_bridge_last_meta = state.get("try19_bridge_last_meta")
        if online_spectral_policy_priority_ema is not None and online_spectral_policy_map_order is None:
            online_spectral_policy_map_order = torch.argsort(
                online_spectral_policy_priority_ema.float(),
                descending=True,
            ).to(dtype=torch.long, device="cpu")
    if master_process:
        restored_text = "restored cached state" if resume_online_spectral_policy_state else "fresh cached state"
        print(
            f"{str(aogpt_train_mode)} enabled: "
            f"layer={int(online_spectral_policy_layer)}, head={int(online_spectral_policy_head)}, "
            f"angles={int(online_spectral_policy_num_angles)}, "
            f"order_usage_mode={str(_online_spectral_policy_order_usage_mode())}, "
            f"distribution_sample_mode={str(online_spectral_policy_distribution_sample_mode)}, "
            f"anneal={float(online_spectral_policy_start_prob):.3f}->"
            f"{float(online_spectral_policy_end_prob):.3f} over "
            f"{int(online_spectral_policy_anneal_start_iter)}.."
            f"{int(online_spectral_policy_anneal_end_iter)}, "
            f"prob_schedule={str(online_spectral_policy_prob_schedule)}, "
            f"top_m={int(online_spectral_policy_top_m)}, {restored_text}"
        )
        if str(online_spectral_policy_prob_schedule).strip().lower() in {"piecewise", "points", "point", "schedule"}:
            print(f"online spectral prob points: {online_spectral_policy_prob_points}")
        if _online_spectral_policy_order_usage_mode() == "temperature_sampling":
            print(
                "online spectral temperature sampling enabled: "
                f"T={float(online_spectral_policy_temperature_sampling_start_temperature):.4f}->"
                f"{float(online_spectral_policy_temperature_sampling_end_temperature):.4f}, "
                f"schedule={str(online_spectral_policy_temperature_sampling_schedule)}, "
                f"sample_window={int(online_spectral_policy_anneal_start_iter)}.."
                f"{int(online_spectral_policy_anneal_end_iter)}"
            )
        if bool(online_spectral_policy_try19_bridge_enabled):
            print(
                "try19 bridge enabled: "
                f"mode={str(online_spectral_policy_try19_bridge_mode)}, "
                f"order_field={str(online_spectral_policy_try19_bridge_order_field)}, "
                f"score_field={str(online_spectral_policy_try19_bridge_score_field)}, "
                f"max_candidates={int(online_spectral_policy_try19_bridge_max_candidates)}"
            )
        if bool(online_spectral_policy_late_l2r_anneal_enabled):
            print(
                "online spectral late-L2R anneal enabled: "
                f"{float(online_spectral_policy_late_l2r_start_prob):.3f}->"
                f"{float(online_spectral_policy_late_l2r_end_prob):.3f} over "
                f"{int(online_spectral_policy_late_l2r_anneal_start_iter)}.."
                f"{int(online_spectral_policy_late_l2r_anneal_end_iter)}, "
                f"per_sample={bool(online_spectral_policy_late_l2r_per_sample)}"
            )


def _online_spectral_policy_prob_for_iter(current_iter):
    def _clamp_prob(value):
        return max(0.0, min(1.0, float(value)))

    schedule = str(online_spectral_policy_prob_schedule).strip().lower()
    if schedule in {"piecewise", "points", "point", "schedule"}:
        raw_points = online_spectral_policy_prob_points
        parsed_points = []
        if isinstance(raw_points, str):
            for item in raw_points.split(","):
                item = item.strip()
                if not item:
                    continue
                if ":" not in item:
                    raise ValueError("online_spectral_policy_prob_points entries must use iter:prob")
                iter_text, prob_text = item.split(":", 1)
                parsed_points.append((int(float(iter_text.strip())), _clamp_prob(prob_text.strip())))
        else:
            for item in raw_points:
                iter_value, prob_value = item
                parsed_points.append((int(iter_value), _clamp_prob(prob_value)))
        if not parsed_points:
            raise ValueError(
                "online_spectral_policy_prob_schedule='piecewise' requires non-empty "
                "online_spectral_policy_prob_points"
            )
        parsed_points = sorted(parsed_points, key=lambda item: int(item[0]))
        current = int(current_iter)
        if current <= int(parsed_points[0][0]):
            return _clamp_prob(parsed_points[0][1])
        for (left_iter, left_prob), (right_iter, right_prob) in zip(parsed_points[:-1], parsed_points[1:]):
            if current <= int(right_iter):
                if int(right_iter) <= int(left_iter):
                    return _clamp_prob(right_prob)
                ratio = (float(current) - float(left_iter)) / max(1.0, float(right_iter - left_iter))
                return _clamp_prob(float(left_prob) + ratio * (float(right_prob) - float(left_prob)))
        return _clamp_prob(parsed_points[-1][1])
    if schedule not in {"", "linear", "lin"}:
        raise ValueError(
            f"Unsupported online_spectral_policy_prob_schedule={online_spectral_policy_prob_schedule!r}; "
            "expected 'linear' or 'piecewise'."
        )

    start = int(online_spectral_policy_anneal_start_iter)
    end = int(online_spectral_policy_anneal_end_iter)
    p0 = float(online_spectral_policy_start_prob)
    p1 = float(online_spectral_policy_end_prob)
    if int(current_iter) <= start:
        return _clamp_prob(p0)
    if end <= start or int(current_iter) >= end:
        return _clamp_prob(p1)
    frac = (float(current_iter) - float(start)) / max(1.0, float(end - start))
    return _clamp_prob(p0 + frac * (p1 - p0))


def _online_spectral_policy_order_usage_mode():
    mode = str(online_spectral_policy_order_usage_mode).strip().lower()
    aliases = {
        "": "probability_anneal",
        "prob": "probability_anneal",
        "probability": "probability_anneal",
        "probability_annealing": "probability_anneal",
        "random_to_learned": "probability_anneal",
        "temp": "temperature_sampling",
        "temperature": "temperature_sampling",
        "temperature_sample": "temperature_sampling",
        "temp_sampling": "temperature_sampling",
    }
    return aliases.get(mode, mode)


def _online_spectral_policy_temperature_sampling_stage(current_iter):
    start = int(online_spectral_policy_anneal_start_iter)
    end = int(online_spectral_policy_anneal_end_iter)
    if int(current_iter) < start:
        return "random"
    if end <= start or int(current_iter) >= end:
        return "map"
    return "sample"


def _online_spectral_policy_temperature_sampling_stage_code(current_iter):
    stage = _online_spectral_policy_temperature_sampling_stage(current_iter)
    return {"random": 0.0, "sample": 1.0, "map": 2.0}.get(stage, float("nan"))


def _online_spectral_policy_temperature_for_iter(current_iter):
    start_temperature = max(float(online_spectral_policy_temperature_sampling_start_temperature), 1e-6)
    end_temperature = max(float(online_spectral_policy_temperature_sampling_end_temperature), 1e-6)
    start = int(online_spectral_policy_anneal_start_iter)
    end = int(online_spectral_policy_anneal_end_iter)
    if int(current_iter) <= start:
        return start_temperature
    if end <= start or int(current_iter) >= end:
        return end_temperature
    frac = (float(current_iter) - float(start)) / max(1.0, float(end - start))
    schedule = str(online_spectral_policy_temperature_sampling_schedule).strip().lower()
    if schedule in {"cosine", "cos"}:
        frac = 0.5 - 0.5 * math.cos(math.pi * frac)
    elif schedule not in {"linear", "lin"}:
        raise ValueError(
            "Unsupported online_spectral_policy_temperature_sampling_schedule="
            f"{online_spectral_policy_temperature_sampling_schedule!r}; expected 'linear' or 'cosine'."
        )
    return max(1e-6, start_temperature + frac * (end_temperature - start_temperature))


def _online_spectral_policy_late_l2r_prob_for_iter(current_iter):
    if not bool(online_spectral_policy_late_l2r_anneal_enabled):
        return 0.0
    start = int(online_spectral_policy_late_l2r_anneal_start_iter)
    end = int(online_spectral_policy_late_l2r_anneal_end_iter)
    p0 = float(online_spectral_policy_late_l2r_start_prob)
    p1 = float(online_spectral_policy_late_l2r_end_prob)
    if int(current_iter) <= start:
        return max(0.0, min(1.0, p0))
    if end <= start or int(current_iter) >= end:
        return max(0.0, min(1.0, p1))
    frac = (float(current_iter) - float(start)) / max(1.0, float(end - start))
    return max(0.0, min(1.0, p0 + frac * (p1 - p0)))


def _online_spectral_policy_updates_frozen(current_iter):
    stop_iter = int(online_spectral_policy_update_stop_iter)
    return stop_iter >= 0 and int(current_iter) >= stop_iter


def _online_spectral_policy_should_collect_attention(current_iter):
    if not _online_spectral_policy_mode_enabled():
        return False
    if int(current_iter) < int(online_spectral_policy_update_start_iter):
        return False
    if _online_spectral_policy_updates_frozen(current_iter):
        return False
    if (
        bool(online_spectral_policy_try19_bridge_enabled)
        and _online_spectral_try19_bridge_should_update(current_iter)
    ):
        return False
    update_every = max(1, int(online_spectral_policy_update_every))
    return int(current_iter) % update_every == 0


def _online_spectral_policy_should_collect_train_step_attention(current_iter):
    if not _online_spectral_policy_should_collect_attention(current_iter):
        return False
    if int(online_spectral_policy_probe_batches) <= 0:
        return True
    return bool(online_spectral_policy_probe_include_train_step_attention)


def _online_spectral_cached_block_order(batch_size_local, device_local):
    if online_spectral_policy_cached_order is None:
        return None
    order = online_spectral_policy_cached_order.to(device=device_local, dtype=torch.long)
    return order.unsqueeze(0).expand(int(batch_size_local), -1)


def _online_spectral_random_block_orders(batch_size_local, device_local):
    return sample_random_block_orders(
        batch_size=int(batch_size_local),
        num_blocks=num_blocks,
        device=device_local,
    )


def _online_spectral_ar_block_orders(batch_size_local, device_local):
    return torch.arange(num_blocks, device=device_local).unsqueeze(0).expand(int(batch_size_local), -1)


def _online_spectral_hybrid_mode_enabled():
    mode_name = str(online_spectral_policy_try19_bridge_mode).strip().lower()
    return mode_name in {
        "hybrid",
        "hybrid_direct_ema",
        "direct_ema",
        "fixed_distribution_mix",
        "hard_ema",
    }


def _online_spectral_policy_hybrid_direct_prob_for_iter(current_iter):
    def _clamp_prob(value):
        return max(0.0, min(1.0, float(value)))

    schedule = str(online_spectral_policy_hybrid_direct_prob_schedule).strip().lower()
    if schedule in {"piecewise", "points", "point", "schedule"}:
        raw_points = online_spectral_policy_hybrid_direct_prob_points
        parsed_points = []
        if isinstance(raw_points, str):
            for item in raw_points.split(","):
                item = item.strip()
                if not item:
                    continue
                if ":" not in item:
                    raise ValueError(
                        "online_spectral_policy_hybrid_direct_prob_points entries must use iter:prob"
                    )
                iter_text, prob_text = item.split(":", 1)
                parsed_points.append((int(float(iter_text.strip())), _clamp_prob(prob_text.strip())))
        else:
            for item in raw_points:
                iter_value, prob_value = item
                parsed_points.append((int(iter_value), _clamp_prob(prob_value)))
        if not parsed_points:
            raise ValueError(
                "online_spectral_policy_hybrid_direct_prob_schedule='piecewise' requires non-empty "
                "online_spectral_policy_hybrid_direct_prob_points"
            )
        parsed_points = sorted(parsed_points, key=lambda item: int(item[0]))
        current = int(current_iter)
        if current <= int(parsed_points[0][0]):
            return _clamp_prob(parsed_points[0][1])
        for (left_iter, left_prob), (right_iter, right_prob) in zip(parsed_points[:-1], parsed_points[1:]):
            if current <= int(right_iter):
                if int(right_iter) <= int(left_iter):
                    return _clamp_prob(right_prob)
                ratio = (float(current) - float(left_iter)) / max(1.0, float(right_iter - left_iter))
                return _clamp_prob(float(left_prob) + ratio * (float(right_prob) - float(left_prob)))
        return _clamp_prob(parsed_points[-1][1])

    start = int(online_spectral_policy_hybrid_direct_prob_anneal_start_iter)
    end = int(online_spectral_policy_hybrid_direct_prob_anneal_end_iter)
    start_prob = float(online_spectral_policy_hybrid_direct_prob_start)
    end_prob = float(online_spectral_policy_hybrid_direct_prob_end)
    if end <= start:
        return _clamp_prob(end_prob)
    if int(current_iter) <= start:
        return _clamp_prob(start_prob)
    if int(current_iter) >= end:
        return _clamp_prob(end_prob)
    ratio = (float(current_iter) - float(start)) / max(1.0, float(end - start))
    prob = start_prob + ratio * (end_prob - start_prob)
    return _clamp_prob(prob)


def _online_spectral_rank_score_from_order(order):
    order_tensor = torch.as_tensor(order, dtype=torch.long, device="cpu").view(-1)
    if int(order_tensor.numel()) != int(num_blocks):
        raise ValueError("Hybrid rank score expects one complete block order.")
    score = torch.empty(int(num_blocks), dtype=torch.float32, device="cpu")
    for rank, block_idx in enumerate(order_tensor.tolist()):
        score[int(block_idx)] = float(int(num_blocks) - int(rank))
    return score


def _online_spectral_blend_hard_ema_orders(hard_order, ema_order, direct_prob):
    hard_score = _online_spectral_rank_score_from_order(hard_order)
    ema_score = _online_spectral_rank_score_from_order(ema_order)
    prob = max(0.0, min(1.0, float(direct_prob)))
    blended = hard_score * prob + ema_score * (1.0 - prob)
    return torch.argsort(blended, descending=True).to(dtype=torch.long, device="cpu")


def _online_spectral_hybrid_frozen_order():
    source = str(online_spectral_policy_hybrid_freeze_order).strip().lower()
    if source in {"hard", "fixed", "direct"} and online_spectral_policy_cached_order is not None:
        return online_spectral_policy_cached_order.detach().cpu().to(dtype=torch.long), "Hard"
    if source in {"hybrid", "blend", "mixed"}:
        if online_spectral_policy_hybrid_order is not None:
            return online_spectral_policy_hybrid_order.detach().cpu().to(dtype=torch.long), "Hybrid"
        if online_spectral_policy_cached_order is not None and online_spectral_policy_map_order is not None:
            return _online_spectral_blend_hard_ema_orders(
                online_spectral_policy_cached_order,
                online_spectral_policy_map_order,
                _online_spectral_policy_hybrid_direct_prob_for_iter(iter_num),
            ), "Hybrid"
    if online_spectral_policy_map_order is not None:
        return online_spectral_policy_map_order.detach().cpu().to(dtype=torch.long), "EMA"
    if online_spectral_policy_cached_order is not None:
        return online_spectral_policy_cached_order.detach().cpu().to(dtype=torch.long), "HardFallback"
    return None, "Missing"


def _online_spectral_apply_late_l2r_anneal(block_orders, policy_name):
    global online_spectral_policy_late_l2r_count
    prob = _online_spectral_policy_late_l2r_prob_for_iter(iter_num)
    if prob <= 0.0:
        return block_orders, policy_name
    batch_size_local = int(block_orders.size(0))
    ar_orders = _online_spectral_ar_block_orders(batch_size_local, block_orders.device)
    selected_count = 0
    if prob >= 1.0:
        mixed_orders = ar_orders
        selected_count = batch_size_local
    elif bool(online_spectral_policy_late_l2r_per_sample):
        mask = torch.rand(batch_size_local, device=block_orders.device) < float(prob)
        selected_count = int(mask.sum().item())
        if selected_count <= 0:
            return block_orders, policy_name
        mixed_orders = torch.where(mask.unsqueeze(1), ar_orders, block_orders)
    else:
        if np.random.random() >= float(prob):
            return block_orders, policy_name
        mixed_orders = ar_orders
        selected_count = batch_size_local
    if torch.is_grad_enabled():
        online_spectral_policy_late_l2r_count += int(selected_count)
    suffix = "LateL2R" if selected_count == batch_size_local else "LateL2RMix"
    return mixed_orders, f"{policy_name}{suffix}"


def _online_spectral_policy_sample_block_orders(idx, return_units=False):
    global online_spectral_policy_random_count, online_spectral_policy_fallback_count
    prob = _online_spectral_policy_prob_for_iter(iter_num)
    use_policy = online_spectral_policy_cached_order is not None and np.random.random() < prob
    if use_policy:
        block_orders = _online_spectral_cached_block_order(idx.size(0), idx.device)
        block_orders, policy_name = _online_spectral_apply_late_l2r_anneal(
            block_orders,
            "OnlineSpectralFixedHeadOrder",
        )
        ordered_units = _singleton_units_for_block_orders(block_orders) if return_units else None
        return block_orders, ordered_units, policy_name

    if torch.is_grad_enabled():
        if online_spectral_policy_cached_order is None and prob > 0.0:
            online_spectral_policy_fallback_count += 1
        else:
            online_spectral_policy_random_count += 1
    fallback = str(online_spectral_policy_fallback)
    if fallback == "random":
        block_orders = _online_spectral_random_block_orders(idx.size(0), idx.device)
        ordered_units = _singleton_units_for_block_orders(block_orders) if return_units else None
        return block_orders, ordered_units, "OnlineSpectralAnnealRandom"
    if fallback == "ar":
        block_orders = _online_spectral_ar_block_orders(idx.size(0), idx.device)
        ordered_units = _singleton_units_for_block_orders(block_orders) if return_units else None
        return block_orders, ordered_units, "OnlineSpectralAnnealAR"
    raise ValueError(f"Unsupported online_spectral_policy_fallback={fallback!r}")


def _online_spectral_distribution_sample_block_orders(idx, return_units=False):
    global online_spectral_policy_random_count, online_spectral_policy_fallback_count
    batch_size_local = int(idx.size(0))
    prob = _online_spectral_policy_prob_for_iter(iter_num)
    order_usage_mode = _online_spectral_policy_order_usage_mode()
    if order_usage_mode not in {"probability_anneal", "temperature_sampling"}:
        raise ValueError(
            "Unsupported online_spectral_policy_order_usage_mode="
            f"{online_spectral_policy_order_usage_mode!r}; expected 'probability_anneal' or 'temperature_sampling'."
        )
    temperature_stage = _online_spectral_policy_temperature_sampling_stage(iter_num)
    if order_usage_mode == "temperature_sampling":
        use_policy = (
            online_spectral_policy_priority_ema is not None
            and temperature_stage in {"sample", "map"}
        )
    else:
        use_policy = online_spectral_policy_priority_ema is not None and np.random.random() < prob
    if use_policy:
        priority = online_spectral_policy_priority_ema.to(device=idx.device, dtype=torch.float32)
        if order_usage_mode == "temperature_sampling" and temperature_stage == "map":
            if online_spectral_policy_map_order is not None:
                map_order = online_spectral_policy_map_order.to(device=idx.device, dtype=torch.long)
            else:
                map_order = torch.argsort(priority, descending=True).to(dtype=torch.long)
            block_orders = map_order.unsqueeze(0).expand(batch_size_local, -1)
            block_orders, policy_name = _online_spectral_apply_late_l2r_anneal(
                block_orders,
                "OnlineSpectralOrderDistributionTemperatureMapOrder",
            )
            ordered_units = _singleton_units_for_block_orders(block_orders) if return_units else None
            return block_orders, ordered_units, policy_name
        if (
            _online_spectral_policy_updates_frozen(iter_num)
            and bool(online_spectral_policy_freeze_to_map_order_after_stop)
        ):
            if (
                _online_spectral_hybrid_mode_enabled()
                and bool(online_spectral_policy_hybrid_mix_hard_after_ema_stop)
                and online_spectral_policy_cached_order is not None
                and online_spectral_policy_map_order is not None
            ):
                hard_order = online_spectral_policy_cached_order.to(device=idx.device, dtype=torch.long)
                ema_order = online_spectral_policy_map_order.to(device=idx.device, dtype=torch.long)
                direct_prob = _online_spectral_policy_hybrid_direct_prob_for_iter(iter_num)
                if direct_prob >= 1.0:
                    block_orders = hard_order.unsqueeze(0).expand(batch_size_local, -1)
                    policy_name = "OnlineSpectralOrderDistributionFrozenEMAHardOrder"
                elif direct_prob <= 0.0:
                    block_orders = ema_order.unsqueeze(0).expand(batch_size_local, -1)
                    policy_name = "OnlineSpectralOrderDistributionFrozenEMAOrder"
                elif bool(online_spectral_policy_hybrid_per_sample):
                    hard_rows = hard_order.unsqueeze(0).expand(batch_size_local, -1)
                    ema_rows = ema_order.unsqueeze(0).expand(batch_size_local, -1)
                    mask = torch.rand(batch_size_local, device=idx.device) < float(direct_prob)
                    block_orders = torch.where(mask.unsqueeze(1), hard_rows, ema_rows)
                    policy_name = "OnlineSpectralOrderDistributionFrozenEMAHardMix"
                else:
                    if np.random.random() < float(direct_prob):
                        block_orders = hard_order.unsqueeze(0).expand(batch_size_local, -1)
                        policy_name = "OnlineSpectralOrderDistributionFrozenEMAHardOrder"
                    else:
                        block_orders = ema_order.unsqueeze(0).expand(batch_size_local, -1)
                        policy_name = "OnlineSpectralOrderDistributionFrozenEMAOrder"
                block_orders, policy_name = _online_spectral_apply_late_l2r_anneal(block_orders, policy_name)
                ordered_units = _singleton_units_for_block_orders(block_orders) if return_units else None
                return block_orders, ordered_units, policy_name
            if _online_spectral_hybrid_mode_enabled():
                frozen_order, frozen_source = _online_spectral_hybrid_frozen_order()
                if frozen_order is not None:
                    map_order = frozen_order.to(device=idx.device, dtype=torch.long)
                    block_orders = map_order.unsqueeze(0).expand(batch_size_local, -1)
                    block_orders, policy_name = _online_spectral_apply_late_l2r_anneal(
                        block_orders,
                        f"OnlineSpectralOrderDistributionFrozen{frozen_source}Order",
                    )
                    ordered_units = _singleton_units_for_block_orders(block_orders) if return_units else None
                    return block_orders, ordered_units, policy_name
            if online_spectral_policy_map_order is not None:
                map_order = online_spectral_policy_map_order.to(device=idx.device, dtype=torch.long)
            else:
                map_order = torch.argsort(priority, descending=True).to(dtype=torch.long)
            block_orders = map_order.unsqueeze(0).expand(batch_size_local, -1)
            block_orders, policy_name = _online_spectral_apply_late_l2r_anneal(
                block_orders,
                "OnlineSpectralOrderDistributionFrozenMapOrder",
            )
            ordered_units = _singleton_units_for_block_orders(block_orders) if return_units else None
            return block_orders, ordered_units, policy_name
        if (
            _online_spectral_hybrid_mode_enabled()
            and online_spectral_policy_cached_order is not None
            and online_spectral_policy_map_order is not None
        ):
            hard_order = online_spectral_policy_cached_order.to(device=idx.device, dtype=torch.long)
            ema_order = online_spectral_policy_map_order.to(device=idx.device, dtype=torch.long)
            direct_prob = _online_spectral_policy_hybrid_direct_prob_for_iter(iter_num)
            if direct_prob >= 1.0:
                block_orders = hard_order.unsqueeze(0).expand(batch_size_local, -1)
                policy_name = "OnlineSpectralOrderDistributionHybridHardOrder"
            elif direct_prob <= 0.0:
                block_orders = ema_order.unsqueeze(0).expand(batch_size_local, -1)
                policy_name = "OnlineSpectralOrderDistributionHybridEMAOrder"
            elif bool(online_spectral_policy_hybrid_per_sample):
                hard_rows = hard_order.unsqueeze(0).expand(batch_size_local, -1)
                ema_rows = ema_order.unsqueeze(0).expand(batch_size_local, -1)
                mask = torch.rand(batch_size_local, device=idx.device) < float(direct_prob)
                block_orders = torch.where(mask.unsqueeze(1), hard_rows, ema_rows)
                policy_name = "OnlineSpectralOrderDistributionHybridDirectEMAMix"
            else:
                if np.random.random() < float(direct_prob):
                    block_orders = hard_order.unsqueeze(0).expand(batch_size_local, -1)
                    policy_name = "OnlineSpectralOrderDistributionHybridHardOrder"
                else:
                    block_orders = ema_order.unsqueeze(0).expand(batch_size_local, -1)
                    policy_name = "OnlineSpectralOrderDistributionHybridEMAOrder"
            block_orders, policy_name = _online_spectral_apply_late_l2r_anneal(block_orders, policy_name)
            ordered_units = _singleton_units_for_block_orders(block_orders) if return_units else None
            return block_orders, ordered_units, policy_name
        distribution_sample_mode = str(online_spectral_policy_distribution_sample_mode).lower()
        if distribution_sample_mode in {"map", "ema_map", "argmax"}:
            if online_spectral_policy_map_order is not None:
                map_order = online_spectral_policy_map_order.to(device=idx.device, dtype=torch.long)
            else:
                map_order = torch.argsort(priority, descending=True).to(dtype=torch.long)
            block_orders = map_order.unsqueeze(0).expand(batch_size_local, -1)
            block_orders, policy_name = _online_spectral_apply_late_l2r_anneal(
                block_orders,
                "OnlineSpectralOrderDistributionMapOrder",
            )
            ordered_units = _singleton_units_for_block_orders(block_orders) if return_units else None
            return block_orders, ordered_units, policy_name
        if distribution_sample_mode not in {"gumbel", "sample", "sampled"}:
            raise ValueError(
                "Unsupported online_spectral_policy_distribution_sample_mode="
                f"{online_spectral_policy_distribution_sample_mode!r}; expected 'gumbel' or 'map'."
            )
        if order_usage_mode == "temperature_sampling":
            sample_temperature = max(float(_online_spectral_policy_temperature_for_iter(iter_num)), 1e-6)
            random_mix_prob = 0.0
            policy_base_name = "OnlineSpectralOrderDistributionTemperatureSample"
        else:
            sample_temperature = max(float(online_spectral_policy_sample_temperature), 1e-6)
            random_mix_prob = max(0.0, min(1.0, float(online_spectral_policy_random_mix_prob)))
            policy_base_name = "OnlineSpectralOrderDistribution"
        if bool(online_spectral_policy_distribution_per_sample):
            rows = []
            random_mix_count = 0
            for _ in range(batch_size_local):
                if np.random.random() < random_mix_prob:
                    rows.append(torch.randperm(num_blocks, device=idx.device))
                    random_mix_count += 1
                else:
                    rows.append(sample_order_from_priority(priority, temperature=sample_temperature).to(dtype=torch.long))
            block_orders = torch.stack(rows, dim=0)
            if torch.is_grad_enabled():
                online_spectral_policy_random_count += int(random_mix_count)
        else:
            if np.random.random() < random_mix_prob:
                order = torch.randperm(num_blocks, device=idx.device)
                if torch.is_grad_enabled():
                    online_spectral_policy_random_count += 1
            else:
                order = sample_order_from_priority(priority, temperature=sample_temperature).to(dtype=torch.long)
            block_orders = order.unsqueeze(0).expand(batch_size_local, -1)
        block_orders, policy_name = _online_spectral_apply_late_l2r_anneal(
            block_orders,
            policy_base_name,
        )
        ordered_units = _singleton_units_for_block_orders(block_orders) if return_units else None
        return block_orders, ordered_units, policy_name

    if torch.is_grad_enabled():
        if (
            online_spectral_policy_priority_ema is None
            and (
                (order_usage_mode == "temperature_sampling" and temperature_stage in {"sample", "map"})
                or (order_usage_mode == "probability_anneal" and prob > 0.0)
            )
        ):
            online_spectral_policy_fallback_count += 1
        else:
            online_spectral_policy_random_count += 1
    fallback = str(online_spectral_policy_fallback)
    if fallback == "random":
        block_orders = _online_spectral_random_block_orders(batch_size_local, idx.device)
        ordered_units = _singleton_units_for_block_orders(block_orders) if return_units else None
        return block_orders, ordered_units, "OnlineSpectralDistributionAnnealRandom"
    if fallback == "ar":
        block_orders = _online_spectral_ar_block_orders(batch_size_local, idx.device)
        ordered_units = _singleton_units_for_block_orders(block_orders) if return_units else None
        return block_orders, ordered_units, "OnlineSpectralDistributionAnnealAR"
    raise ValueError(f"Unsupported online_spectral_policy_fallback={fallback!r}")


def _online_spectral_policy_attention_matrix_from_outputs(outputs, block_orders):
    attentions = _attn_mlp_extract_attentions(outputs)
    if not attentions:
        return None
    layer_idx = int(online_spectral_policy_layer)
    if layer_idx < 0:
        layer_idx = len(attentions) + layer_idx
    if layer_idx < 0 or layer_idx >= len(attentions):
        raise ValueError(
            f"online_spectral_policy_layer={online_spectral_policy_layer} is outside available layers 0..{len(attentions)-1}."
        )
    layer_heads = _aggregate_layerhead_attention_to_current_blocks(
        attentions[layer_idx].detach(),
        block_orders,
        str(online_spectral_policy_export_type),
    )
    head_idx = int(online_spectral_policy_head)
    if head_idx < 0:
        matrix = layer_heads.mean(dim=0)
    else:
        if head_idx >= int(layer_heads.size(0)):
            raise ValueError(
                f"online_spectral_policy_head={online_spectral_policy_head} is outside available heads 0..{int(layer_heads.size(0))-1}."
            )
        matrix = layer_heads[head_idx]
    matrix = matrix.detach().float()
    matrix.fill_diagonal_(0.0)
    return matrix


def _online_spectral_weighted_mean_matrices(weighted_matrices):
    if not weighted_matrices:
        return None, 0
    total = sum(max(1, int(count)) for _, count in weighted_matrices)
    accum = None
    for matrix, count in weighted_matrices:
        weight = float(max(1, int(count)))
        value = matrix.detach().float() * weight
        accum = value if accum is None else accum + value
    return accum / float(max(1, total)), int(total)


def _online_spectral_policy_zscore(values):
    values = values.float()
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return torch.zeros_like(values)
    mean = finite.mean()
    std = finite.std(unbiased=False)
    safe = torch.where(torch.isfinite(values), values, mean)
    if float(std.item()) < 1e-8:
        return torch.zeros_like(safe)
    return (safe - mean) / std


@torch.no_grad()
def _online_spectral_policy_loss_rerank_candidates(candidates):
    """Rerank current-frame candidates with current-model train/probe losses."""
    if not bool(online_spectral_policy_loss_rerank_enabled):
        return candidates
    if not candidates:
        return candidates
    rerank_batches = int(online_spectral_policy_loss_rerank_batches)
    if rerank_batches <= 0:
        return candidates
    rerank_split = str(online_spectral_policy_loss_rerank_split)
    if rerank_split not in {"train", "val"}:
        raise ValueError(f"Unsupported online_spectral_policy_loss_rerank_split={rerank_split!r}")

    candidate_orders = torch.tensor(
        [[int(value) for value in candidate["order"]] for candidate in candidates],
        dtype=torch.long,
        device=device,
    )
    num_candidates = int(candidate_orders.size(0))
    prefix_k = max(1, min(int(online_spectral_policy_loss_rerank_prefix_k), int(num_blocks)))
    candidate_batch_size = max(1, int(online_spectral_policy_loss_rerank_candidate_batch_size))
    rerank_batch_size = max(1, int(online_spectral_policy_loss_rerank_batch_size))
    prefix_loss_sum = torch.zeros(num_candidates, dtype=torch.float64, device=device)
    full_loss_sum = torch.zeros(num_candidates, dtype=torch.float64, device=device)
    count_sum = torch.zeros(num_candidates, dtype=torch.float64, device=device)

    was_training = model.training
    model.eval()
    try:
        for _ in range(rerank_batches):
            X_rerank, _ = get_batch(rerank_split, batch_size_override=rerank_batch_size)
            batch_size_local = int(X_rerank.size(0))
            for start in range(0, num_candidates, candidate_batch_size):
                end = min(num_candidates, start + candidate_batch_size)
                chunk = candidate_orders[start:end]
                chunk_size = int(chunk.size(0))
                block_orders = (
                    chunk[:, None, :]
                    .expand(chunk_size, batch_size_local, int(num_blocks))
                    .reshape(chunk_size * batch_size_local, int(num_blocks))
                )
                X_expanded = (
                    X_rerank[None, :, :]
                    .expand(chunk_size, batch_size_local, int(X_rerank.size(1)))
                    .reshape(chunk_size * batch_size_local, int(X_rerank.size(1)))
                )
                token_orders = _expand_training_block_orders(block_orders)
                with ctx:
                    outputs = model(
                        X_expanded,
                        mode=None,
                        orders=token_orders,
                        return_token_loss=True,
                        return_logits=False,
                    )
                token_losses = outputs[2]
                block_losses = token_losses_to_block_losses(
                    token_losses.detach(),
                    block_len=effective_order_block_len,
                ).float()
                block_losses = block_losses.view(chunk_size, batch_size_local, int(num_blocks))
                prefix_loss_sum[start:end] += block_losses[:, :, :prefix_k].double().mean(dim=2).sum(dim=1)
                full_loss_sum[start:end] += block_losses.double().mean(dim=2).sum(dim=1)
                count_sum[start:end] += float(batch_size_local)
    finally:
        if was_training:
            model.train()

    if ddp:
        dist.all_reduce(prefix_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(full_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_sum, op=dist.ReduceOp.SUM)

    count_sum = count_sum.clamp_min(1.0)
    prefix_loss = (prefix_loss_sum / count_sum).float()
    full_loss = (full_loss_sum / count_sum).float()
    attention_scores = torch.tensor(
        [float(candidate.get("score", 0.0)) for candidate in candidates],
        dtype=torch.float32,
        device=device,
    )
    final_scores = (
        float(online_spectral_policy_loss_rerank_attention_weight)
        * _online_spectral_policy_zscore(attention_scores)
        - float(online_spectral_policy_loss_rerank_prefix_weight)
        * _online_spectral_policy_zscore(prefix_loss)
        - float(online_spectral_policy_loss_rerank_full_weight)
        * _online_spectral_policy_zscore(full_loss)
    )

    reranked = []
    for idx, candidate in enumerate(candidates):
        updated = dict(candidate)
        meta = dict(updated.get("meta", {}))
        meta.update(
            {
                "attention_spectral_score": float(attention_scores[idx].detach().cpu().item()),
                "loss_rerank_prefix_loss": float(prefix_loss[idx].detach().cpu().item()),
                "loss_rerank_full_loss": float(full_loss[idx].detach().cpu().item()),
                "loss_rerank_score": float(final_scores[idx].detach().cpu().item()),
                "loss_rerank_prefix_k": int(prefix_k),
                "loss_rerank_batches": int(rerank_batches),
                "loss_rerank_batch_size": int(rerank_batch_size),
                "loss_rerank_split": str(rerank_split),
                "loss_rerank_attention_weight": float(online_spectral_policy_loss_rerank_attention_weight),
                "loss_rerank_prefix_weight": float(online_spectral_policy_loss_rerank_prefix_weight),
                "loss_rerank_full_weight": float(online_spectral_policy_loss_rerank_full_weight),
                "loss_rerank_rule": (
                    "current-frame train/probe teacher-forced loss only; no original-frame or tau signal is used"
                ),
            }
        )
        updated["meta"] = meta
        updated["score"] = float(final_scores[idx].detach().cpu().item())
        reranked.append(updated)
    reranked.sort(key=lambda item: (-float(item["score"]), str(item["name"])))
    return reranked


def _online_spectral_policy_recover_candidates(recovery_matrix, top_m):
    recovery_top_m = int(top_m)
    if bool(online_spectral_policy_loss_rerank_enabled):
        recovery_top_m = max(recovery_top_m, int(online_spectral_policy_loss_rerank_top_k))
    candidates = recover_fixed_head_spectral_candidates(
        recovery_matrix,
        online_spectral_policy_config,
        top_m=recovery_top_m,
    )
    candidates = _online_spectral_policy_loss_rerank_candidates(candidates)
    return candidates[: int(top_m)]


def _online_spectral_policy_order_history_file():
    if str(online_spectral_policy_order_history_path).strip():
        return str(online_spectral_policy_order_history_path)
    return os.path.join(out_dir, "online_spectral_policy_order_history.jsonl")


def _online_spectral_order_to_list(order):
    if order is None:
        return None
    if torch.is_tensor(order):
        return [int(value) for value in order.detach().cpu().view(-1).tolist()]
    return [int(value) for value in order]


def _online_spectral_order_to_original(order):
    order_current = _online_spectral_order_to_list(order)
    if order_current is None or not permute_data or fixed_block_perm is None:
        return None
    mapper = fixed_block_perm.to(dtype=torch.long, device="cpu")
    current = torch.tensor(order_current, dtype=torch.long, device="cpu")
    return [int(value) for value in mapper[current].tolist()]


def _online_spectral_tensor_to_float_list(values):
    if values is None:
        return None
    tensor = torch.as_tensor(values).detach().float().cpu().view(-1)
    out = []
    for value in tensor.tolist():
        value = float(value)
        out.append(value if math.isfinite(value) else None)
    return out


def _online_spectral_candidate_history_row(candidate, include_order=True):
    meta = candidate.get("meta", {}) if isinstance(candidate, dict) else {}
    if not isinstance(meta, dict):
        meta = {}
    row = {
        "name": str(candidate.get("name", "")),
        "score": float(candidate.get("score", float("nan"))),
        "attention_spectral_score": float(meta.get("attention_spectral_score", candidate.get("score", float("nan")))),
        "adjacency_path_score": float(meta.get("adjacency_path_score", float("nan"))),
        "best_directed_path_score": float(meta.get("best_directed_path_score", float("nan"))),
        "best_directed_name": str(meta.get("best_directed_name", "")),
        "band_quality": float(meta.get("band_quality", float("nan"))),
        "loss_rerank_prefix_loss": float(meta.get("loss_rerank_prefix_loss", float("nan"))),
        "loss_rerank_full_loss": float(meta.get("loss_rerank_full_loss", float("nan"))),
        "loss_rerank_score": float(meta.get("loss_rerank_score", float("nan"))),
    }
    if include_order:
        order_current = _online_spectral_order_to_list(candidate.get("order"))
        row["order_current"] = order_current
        order_original = _online_spectral_order_to_original(order_current)
        if order_original is not None:
            row["order_original"] = order_original
    return row


def _online_spectral_json_safe(value):
    if isinstance(value, dict):
        return {str(k): _online_spectral_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_online_spectral_json_safe(v) for v in value]
    if torch.is_tensor(value):
        return _online_spectral_json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return _online_spectral_json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _online_spectral_policy_write_order_history(current_iter, candidates):
    if not bool(online_spectral_policy_order_history_enabled) or not master_process:
        return
    candidates = list(candidates or [])
    top_n = max(0, int(online_spectral_policy_order_history_top_candidates))
    include_priority = bool(online_spectral_policy_order_history_include_priority)
    top_candidate = candidates[0] if candidates else online_spectral_policy_last_meta
    row = {
        "version": 1,
        "iter": int(current_iter),
        "wall_time": float(time.time()),
        "aogpt_train_mode": str(aogpt_train_mode),
        "policy_updates": int(online_spectral_policy_updates),
        "distribution_updates": int(online_spectral_policy_distribution_updates),
        "attention_updates": int(online_spectral_policy_attention_updates),
        "last_attention_samples": int(online_spectral_policy_last_attention_samples),
        "num_blocks": int(num_blocks),
        "block_order_block_len": int(effective_order_block_len),
        "permute_data": bool(permute_data),
        "loss_rerank_enabled": bool(online_spectral_policy_loss_rerank_enabled),
        "late_l2r_anneal_enabled": bool(online_spectral_policy_late_l2r_anneal_enabled),
        "late_l2r_prob": float(_online_spectral_policy_late_l2r_prob_for_iter(current_iter)),
        "late_l2r_count": int(online_spectral_policy_late_l2r_count),
        "hybrid_direct_prob": float(_online_spectral_policy_hybrid_direct_prob_for_iter(current_iter)),
        "hybrid_direct_prob_schedule": str(online_spectral_policy_hybrid_direct_prob_schedule),
        "hybrid_direct_prob_points": str(online_spectral_policy_hybrid_direct_prob_points),
        "hybrid_per_sample": bool(online_spectral_policy_hybrid_per_sample),
        "hybrid_freeze_order": str(online_spectral_policy_hybrid_freeze_order),
        "hybrid_mix_hard_after_ema_stop": bool(online_spectral_policy_hybrid_mix_hard_after_ema_stop),
        "hybrid_update_hard_after_ema_stop": bool(online_spectral_policy_hybrid_update_hard_after_ema_stop),
        "note": (
            "All *_current orders are current-frame block ids. *_original fields are diagnostics only "
            "and are not used by the policy."
        ),
    }
    if online_spectral_policy_cached_order is not None:
        cached_current = _online_spectral_order_to_list(online_spectral_policy_cached_order)
        row["cached_order_current"] = cached_current
        cached_original = _online_spectral_order_to_original(cached_current)
        if cached_original is not None:
            row["cached_order_original"] = cached_original
    if online_spectral_policy_map_order is not None:
        map_current = _online_spectral_order_to_list(online_spectral_policy_map_order)
        row["map_order_current"] = map_current
        map_original = _online_spectral_order_to_original(map_current)
        if map_original is not None:
            row["map_order_original"] = map_original
    if online_spectral_policy_hybrid_order is not None:
        hybrid_current = _online_spectral_order_to_list(online_spectral_policy_hybrid_order)
        row["hybrid_order_current"] = hybrid_current
        hybrid_original = _online_spectral_order_to_original(hybrid_current)
        if hybrid_original is not None:
            row["hybrid_order_original"] = hybrid_original
    if isinstance(top_candidate, dict):
        row["top_candidate"] = _online_spectral_candidate_history_row(top_candidate, include_order=True)
    if top_n > 0 and candidates:
        row["top_candidates"] = [
            _online_spectral_candidate_history_row(candidate, include_order=True)
            for candidate in candidates[: min(top_n, len(candidates))]
        ]
    if include_priority:
        if online_spectral_policy_priority_ema is not None:
            row["priority_ema"] = _online_spectral_tensor_to_float_list(online_spectral_policy_priority_ema)
        if online_spectral_policy_last_priority_update is not None:
            row["priority_update"] = _online_spectral_tensor_to_float_list(online_spectral_policy_last_priority_update)
        if online_spectral_policy_last_candidate_weights is not None:
            row["last_candidate_weights"] = _online_spectral_tensor_to_float_list(
                online_spectral_policy_last_candidate_weights
            )
        if isinstance(online_spectral_policy_last_distribution_meta, dict):
            row["last_distribution_meta"] = _online_spectral_json_safe(online_spectral_policy_last_distribution_meta)
    if isinstance(online_spectral_policy_try19_bridge_last_meta, dict):
        row["try19_bridge"] = _online_spectral_json_safe(online_spectral_policy_try19_bridge_last_meta)
    path = _online_spectral_policy_order_history_file()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(_online_spectral_json_safe(row), ensure_ascii=True, sort_keys=True) + "\n")


@torch.no_grad()
def _online_spectral_policy_probe_attention_matrices_if_due(current_iter):
    if not _online_spectral_policy_should_collect_attention(current_iter):
        return []
    probe_batches = int(online_spectral_policy_probe_batches)
    if probe_batches <= 0:
        return []
    probe_batch_size = max(1, int(online_spectral_policy_probe_batch_size))
    probe_split = str(online_spectral_policy_probe_split)
    if probe_split not in {"train", "val"}:
        raise ValueError(f"Unsupported online_spectral_policy_probe_split={probe_split!r}")
    was_training = model.training
    model.eval()
    matrices = []
    try:
        for _ in range(probe_batches):
            X_probe, _ = get_batch(probe_split, batch_size_override=probe_batch_size)
            with ctx:
                forward_outputs, _, order_info = _forward_with_active_training_policy(
                    X_probe,
                    return_token_loss=False,
                    return_order_info=False,
                    return_attentions=True,
                    return_logits=False,
                )
            if order_info is None:
                continue
            matrix = _online_spectral_policy_attention_matrix_from_outputs(
                forward_outputs,
                order_info.get("block_orders"),
            )
            if matrix is not None:
                matrices.append((matrix, int(X_probe.size(0))))
    finally:
        if was_training:
            model.train()
    return matrices


def _online_spectral_sync_attention_matrix(matrix):
    if ddp:
        dist.all_reduce(matrix, op=dist.ReduceOp.SUM)
        matrix = matrix / float(ddp_world_size)
    return matrix


@torch.no_grad()
def _online_spectral_policy_update_from_matrix(matrix, current_iter, num_attention_samples=0):
    global online_spectral_policy_A_ema, online_spectral_policy_cached_order, online_spectral_policy_last_meta
    global online_spectral_policy_priority_ema, online_spectral_policy_last_priority_update
    global online_spectral_policy_map_order, online_spectral_policy_distribution_updates
    global online_spectral_policy_last_candidate_weights, online_spectral_policy_last_distribution_meta
    global online_spectral_policy_map_order_changed_kendall
    global online_spectral_policy_updates, online_spectral_policy_attention_updates
    global online_spectral_policy_failed_updates
    global online_spectral_policy_last_update_iter, online_spectral_policy_last_attention_iter
    global online_spectral_policy_last_attention_samples
    if not _online_spectral_policy_mode_enabled() or matrix is None:
        return
    matrix = _online_spectral_sync_attention_matrix(matrix.detach().float())
    matrix_cpu = matrix.detach().float().cpu()
    if online_spectral_policy_A_ema is None:
        online_spectral_policy_A_ema = matrix_cpu
    else:
        decay = min(0.9999, max(0.0, float(online_spectral_policy_ema_decay)))
        online_spectral_policy_A_ema = online_spectral_policy_A_ema * decay + matrix_cpu * (1.0 - decay)
    online_spectral_policy_attention_updates += 1
    online_spectral_policy_last_attention_iter = int(current_iter)
    online_spectral_policy_last_attention_samples = int(num_attention_samples) * int(ddp_world_size)

    recovery_matrix = online_spectral_policy_A_ema if bool(online_spectral_policy_use_ema) else matrix_cpu
    if _online_spectral_distribution_mode_enabled():
        try:
            candidates = _online_spectral_policy_recover_candidates(
                recovery_matrix,
                top_m=int(online_spectral_policy_top_m),
            )
            r_update, weights, dist_meta = candidates_to_priority_vector(
                candidates,
                num_blocks=num_blocks,
                teacher_temperature=float(online_spectral_policy_teacher_temperature),
                score_normalization=str(online_spectral_policy_score_normalization),
            )
        except Exception as exc:
            online_spectral_policy_failed_updates += 1
            if master_process:
                print(f"online spectral order-distribution recovery failed at iter {int(current_iter)}: {exc}")
            _online_spectral_policy_refresh_latest_stats(policy_refreshed=False)
            return
        if r_update.numel() != num_blocks or not torch.isfinite(r_update).all():
            raise ValueError("Online spectral order distribution produced an invalid priority vector.")
        previous_map_order = online_spectral_policy_map_order
        decay = min(0.9999, max(0.0, float(online_spectral_policy_priority_ema_decay)))
        if online_spectral_policy_priority_ema is None:
            online_spectral_policy_priority_ema = r_update.detach().cpu()
        else:
            online_spectral_policy_priority_ema = (
                online_spectral_policy_priority_ema.float() * decay
                + r_update.detach().cpu().float() * (1.0 - decay)
            )
        online_spectral_policy_last_priority_update = r_update.detach().cpu()
        online_spectral_policy_map_order = torch.argsort(
            online_spectral_policy_priority_ema.float(),
            descending=True,
        ).to(dtype=torch.long, device="cpu")
        if previous_map_order is not None:
            online_spectral_policy_map_order_changed_kendall = _kendall_tau_between_orders(
                previous_map_order,
                online_spectral_policy_map_order,
            )
        else:
            online_spectral_policy_map_order_changed_kendall = None
        online_spectral_policy_last_candidate_weights = weights.detach().cpu()
        online_spectral_policy_last_distribution_meta = dict(dist_meta)
        online_spectral_policy_last_distribution_meta.update(
            {
                "top_candidate_name": str(candidates[0].get("name", "")),
                "top_candidate_score": float(candidates[0].get("score", float("nan"))),
                "priority_ema_decay": float(online_spectral_policy_priority_ema_decay),
            }
        )
        online_spectral_policy_last_meta = candidates[0]
        online_spectral_policy_distribution_updates += 1
    else:
        try:
            fixed_history_top_m = 1
            if bool(online_spectral_policy_order_history_enabled):
                fixed_history_top_m = max(1, int(online_spectral_policy_order_history_top_candidates))
            candidates = _online_spectral_policy_recover_candidates(
                recovery_matrix,
                top_m=fixed_history_top_m,
            )
        except Exception as exc:
            online_spectral_policy_failed_updates += 1
            if master_process:
                print(f"online spectral fixed-head recovery failed at iter {int(current_iter)}: {exc}")
            _online_spectral_policy_refresh_latest_stats(policy_refreshed=False)
            return
        meta = candidates[0]
        order_values = meta["order"]
        if len(order_values) != num_blocks or sorted(order_values) != list(range(num_blocks)):
            raise ValueError("Online spectral fixed-head policy produced an invalid block order.")
        online_spectral_policy_cached_order = torch.tensor(order_values, dtype=torch.long, device="cpu")
        online_spectral_policy_last_meta = meta
    online_spectral_policy_updates += 1
    online_spectral_policy_last_update_iter = int(current_iter)
    _online_spectral_policy_refresh_latest_stats(policy_refreshed=True)
    _online_spectral_policy_write_order_history(current_iter, candidates)
    log_every = int(online_spectral_policy_log_interval)
    if master_process and log_every > 0 and int(online_spectral_policy_updates) % log_every == 0:
        if _online_spectral_distribution_mode_enabled():
            first16 = [int(v) for v in online_spectral_policy_map_order[:16].tolist()]
            print(
                "online spectral order-distribution update "
                f"{int(online_spectral_policy_distribution_updates)} at iter {int(current_iter)}: "
                f"top_score={float(online_spectral_policy_last_meta.get('score', float('nan'))):.4f}, "
                f"weight_entropy={float(online_spectral_policy_last_distribution_meta.get('weight_entropy', float('nan'))):.4f}, "
                f"map_order_current_first16={first16}"
            )
        else:
            first16 = [int(v) for v in online_spectral_policy_cached_order[:16].tolist()]
            print(
                "online spectral fixed-head policy update "
                f"{int(online_spectral_policy_updates)} at iter {int(current_iter)}: "
                f"score={float(online_spectral_policy_last_meta.get('score', float('nan'))):.4f}, "
                f"cached_order_current_first16={first16}"
            )


def _online_spectral_try19_bridge_mode():
    return str(online_spectral_policy_try19_bridge_mode).strip().lower()


def _online_spectral_try19_bridge_should_update(current_iter):
    if not bool(online_spectral_policy_try19_bridge_enabled):
        return False
    if not _online_spectral_policy_mode_enabled():
        return False
    mode_name = _online_spectral_try19_bridge_mode()
    if mode_name in {"", "none", "off", "false"}:
        return False
    start_iter = int(online_spectral_policy_try19_bridge_start_iter)
    if start_iter < 0:
        start_iter = int(online_spectral_policy_update_start_iter)
    stop_iter = int(online_spectral_policy_try19_bridge_stop_iter)
    if stop_iter < 0:
        stop_iter = int(online_spectral_policy_update_stop_iter)
    if int(current_iter) < int(start_iter):
        return False
    if stop_iter >= 0 and int(current_iter) >= int(stop_iter):
        return False
    update_every = max(1, int(online_spectral_policy_update_every))
    if int(current_iter) % int(update_every) != 0:
        return False
    return True


def _online_spectral_try19_bridge_candidates_from_rows(rows):
    order_field = str(online_spectral_policy_try19_bridge_order_field).strip()
    score_field = str(online_spectral_policy_try19_bridge_score_field).strip()
    min_score = max(float(online_spectral_policy_try19_bridge_min_candidate_score), 1e-12)
    max_candidates = int(online_spectral_policy_try19_bridge_max_candidates)
    candidates = []
    expected = list(range(int(num_blocks)))
    for row in rows or []:
        order = row.get(order_field)
        if order is None and order_field != "consensus_order_current":
            order = row.get("consensus_order_current")
        if order is None:
            continue
        order = [int(value) for value in order]
        if len(order) != int(num_blocks) or sorted(order) != expected:
            continue
        try:
            raw_score = float(row.get(score_field, float("nan")))
        except (TypeError, ValueError):
            raw_score = float("nan")
        if not math.isfinite(raw_score):
            try:
                raw_score = abs(float(row.get("loss_profile_consensus_alignment", float("nan"))))
            except (TypeError, ValueError):
                raw_score = float("nan")
        score = float(raw_score) if math.isfinite(raw_score) and raw_score > 0.0 else float(min_score)
        layer_idx = int(row.get("layer", -1))
        head_idx = int(row.get("head", -1))
        head_label = str(row.get("head_label", _head_signal_head_label(head_idx)))
        candidate = {
            "name": f"try19_bridge_L{layer_idx}H{head_label}",
            "order": order,
            "score": float(score),
            "meta": {
                "try19_bridge": True,
                "try19_bridge_source": "head_signal_probe",
                "try19_bridge_order_field": str(order_field),
                "try19_bridge_score_field": str(score_field),
                "layer": int(layer_idx),
                "head": int(head_idx),
                "head_label": str(head_label),
                "raw_score": float(row.get("raw_score", float("nan"))),
                "loss_profile_score_gap": float(row.get("loss_profile_score_gap", float("nan"))),
                "loss_profile_consensus_alignment": float(
                    row.get("loss_profile_consensus_alignment", float("nan"))
                ),
                "loss_profile_used_fallback": bool(row.get("loss_profile_used_fallback", False)),
                "loss_profile_low_confidence": bool(row.get("loss_profile_low_confidence", False)),
                "loss_profile_low_confidence_reasons": str(
                    row.get("loss_profile_low_confidence_reasons", "")
                ),
                "attention_spectral_score": float(row.get("raw_score", float("nan"))),
                "band_quality": float(row.get("raw_band_quality", float("nan"))),
                "adjacency_path_score": float(row.get("raw_adjacency_path_score", float("nan"))),
                "best_directed_path_score": float(row.get("raw_directed_path_score", float("nan"))),
                "no_prior_rule": (
                    "current attention + current reveal/profile loss + same-probe cross-head consensus; "
                    "no original L2R/R2L, tau/PPL oracle, target-position anchor, or historical sign"
                ),
            },
        }
        candidates.append(candidate)
    candidates.sort(key=lambda item: (-float(item["score"]), str(item["name"])))
    if max_candidates > 0:
        candidates = candidates[:max_candidates]
    return candidates


def _online_spectral_try19_bridge_summary(candidates, selected_order, current_iter, total_samples):
    scores = [float(candidate.get("score", 0.0)) for candidate in candidates]
    gaps = [
        float(candidate.get("meta", {}).get("loss_profile_score_gap", float("nan")))
        for candidate in candidates
    ]
    finite_gaps = [value for value in gaps if math.isfinite(value)]
    fallback_count = sum(
        1
        for candidate in candidates
        if bool(candidate.get("meta", {}).get("loss_profile_used_fallback", False))
    )
    selected_original = _online_spectral_order_to_original(selected_order)
    selected_tau = None
    if selected_original is not None:
        selected_tau = float(_head_signal_tau_to_l2r(selected_original))
    return {
        "enabled": True,
        "mode": str(_online_spectral_try19_bridge_mode()),
        "iter": int(current_iter),
        "updates": int(online_spectral_policy_try19_bridge_updates),
        "candidate_count": int(len(candidates)),
        "probe_samples": int(total_samples) * int(ddp_world_size),
        "score_mean": float(np.mean(scores)) if scores else float("nan"),
        "score_max": float(np.max(scores)) if scores else float("nan"),
        "score_gap_mean": float(np.mean(finite_gaps)) if finite_gaps else float("nan"),
        "score_gap_min": float(np.min(finite_gaps)) if finite_gaps else float("nan"),
        "fallback_frac": float(fallback_count) / float(max(1, len(candidates))),
        "selected_order_current": [int(value) for value in selected_order],
        "selected_order_original_diagnostic": selected_original,
        "selected_tau_diagnostic": selected_tau,
        "selected_abs_tau_diagnostic": abs(float(selected_tau)) if selected_tau is not None else None,
        "order_field": str(online_spectral_policy_try19_bridge_order_field),
        "score_field": str(online_spectral_policy_try19_bridge_score_field),
        "no_prior_rule": (
            "Bridge selection uses only current attention, current model loss/profile scores, "
            "and same-probe consensus. Original-frame tau/L2R/PPL/history are diagnostic-only."
        ),
    }


def _online_spectral_bridge_score_weights(scores):
    scores_tensor = torch.tensor([float(value) for value in scores], dtype=torch.float32, device="cpu")
    if scores_tensor.numel() == 0:
        return scores_tensor, {"score_mean": float("nan"), "score_std": float("nan"), "uniform_weights": True}
    finite_scores = scores_tensor[torch.isfinite(scores_tensor)]
    if finite_scores.numel() == 0:
        weights = torch.full_like(scores_tensor, 1.0 / float(max(1, scores_tensor.numel())))
        return weights, {"score_mean": float("nan"), "score_std": float("nan"), "uniform_weights": True}
    score_mean = float(finite_scores.mean().item())
    score_std_value = float(finite_scores.std(unbiased=False).item())
    safe_scores = torch.where(torch.isfinite(scores_tensor), scores_tensor, torch.tensor(score_mean))
    normalization = str(online_spectral_policy_score_normalization).strip().lower()
    if normalization == "zscore":
        use_uniform = bool(score_std_value < 1e-8)
        scores_norm = torch.zeros_like(safe_scores) if use_uniform else (safe_scores - score_mean) / score_std_value
    elif normalization in {"none", "raw"}:
        use_uniform = False
        scores_norm = safe_scores
    else:
        raise ValueError(f"unknown score_normalization={online_spectral_policy_score_normalization!r}")
    if use_uniform:
        weights = torch.full_like(scores_tensor, 1.0 / float(max(1, scores_tensor.numel())))
    else:
        temperature = max(float(online_spectral_policy_teacher_temperature), 1e-6)
        weights = torch.softmax(scores_norm / temperature, dim=0).to(dtype=torch.float32)
    return weights, {
        "score_mean": float(score_mean),
        "score_std": float(score_std_value),
        "uniform_weights": bool(use_uniform),
    }


def _online_spectral_try19_bridge_continuous_priority_from_rows(rows):
    order_field = str(online_spectral_policy_try19_bridge_order_field).strip()
    score_field = str(online_spectral_policy_try19_bridge_score_field).strip()
    normalization = str(online_spectral_policy_continuous_priority_normalization).strip().lower()
    if normalization not in {"minmax", "range", "zero_one", "0_1"}:
        raise ValueError(
            "continuous Fiedler priority bridge only supports minmax normalization, "
            f"got {online_spectral_policy_continuous_priority_normalization!r}"
        )
    eps = max(float(online_spectral_policy_continuous_priority_eps), 1e-12)
    expected = list(range(int(num_blocks)))
    priorities = []
    scores = []
    row_summaries = []
    for row in rows or []:
        vector = row.get("raw_fiedler_vector")
        raw_order = row.get("raw_order_current")
        selected_order = row.get(order_field)
        if selected_order is None and order_field != "consensus_order_current":
            selected_order = row.get("consensus_order_current")
        if vector is None or raw_order is None or selected_order is None:
            continue
        vector = [float(value) for value in vector]
        raw_order = [int(value) for value in raw_order]
        selected_order = [int(value) for value in selected_order]
        if len(vector) != int(num_blocks):
            continue
        if len(raw_order) != int(num_blocks) or sorted(raw_order) != expected:
            continue
        if len(selected_order) != int(num_blocks) or sorted(selected_order) != expected:
            continue
        reverse_raw = list(reversed(raw_order))
        if selected_order == raw_order:
            selected_reverse = False
        elif selected_order == reverse_raw:
            selected_reverse = True
        else:
            continue
        coord = torch.tensor(vector, dtype=torch.float32, device="cpu")
        if selected_reverse:
            coord = -coord
        finite = coord[torch.isfinite(coord)]
        if finite.numel() != int(num_blocks):
            continue
        coord_min = float(coord.min().item())
        coord_max = float(coord.max().item())
        coord_range = float(coord_max - coord_min)
        if coord_range <= eps:
            continue
        priority = 1.0 - (coord - coord_min) / float(coord_range + eps)
        priority = priority.clamp(0.0, 1.0).to(dtype=torch.float32)
        try:
            raw_score = float(row.get(score_field, float("nan")))
        except (TypeError, ValueError):
            raw_score = float("nan")
        if not math.isfinite(raw_score) or raw_score <= 0.0:
            raw_score = float(online_spectral_policy_try19_bridge_min_candidate_score)
        priorities.append(priority)
        scores.append(float(raw_score))
        row_summaries.append(
            {
                "layer": int(row.get("layer", -1)),
                "head": int(row.get("head", -1)),
                "head_label": str(row.get("head_label", "")),
                "score": float(raw_score),
                "selected_reverse": bool(selected_reverse),
                "coord_min": float(coord_min),
                "coord_max": float(coord_max),
                "coord_range": float(coord_range),
                "priority_min": float(priority.min().item()),
                "priority_max": float(priority.max().item()),
                "priority_std": float(priority.std(unbiased=False).item()),
            }
        )
    if not priorities:
        raise ValueError("continuous Fiedler priority bridge found no valid oriented Fiedler vectors")
    weights, score_meta = _online_spectral_bridge_score_weights(scores)
    stacked = torch.stack(priorities, dim=0)
    r_update = torch.sum(weights.view(-1, 1) * stacked, dim=0).to(dtype=torch.float32)
    map_update_order = torch.argsort(r_update, descending=True).to(dtype=torch.long, device="cpu")
    entropy = float((-(weights * weights.clamp_min(1e-12).log()).sum()).item())
    dist_meta = {
        "top_m_used": int(len(priorities)),
        "score_mean": float(score_meta["score_mean"]),
        "score_std": float(score_meta["score_std"]),
        "score_normalization": str(online_spectral_policy_score_normalization),
        "teacher_temperature": float(online_spectral_policy_teacher_temperature),
        "weight_entropy": entropy,
        "top_weight": float(weights.max().item()) if weights.numel() else float("nan"),
        "effective_num_candidates": float(math.exp(entropy)) if math.isfinite(entropy) else float("nan"),
        "uniform_weights": bool(score_meta["uniform_weights"]),
        "priority_std": float(r_update.std(unbiased=False).item()),
        "priority_min": float(r_update.min().item()),
        "priority_max": float(r_update.max().item()),
        "priority_range": float((r_update.max() - r_update.min()).item()),
        "priority_effectively_constant": bool(float((r_update.max() - r_update.min()).item()) <= eps),
        "reverse_pair_weight_mass": float("nan"),
        "continuous_priority_bridge": True,
        "continuous_priority_normalization": str(normalization),
        "continuous_priority_eps": float(eps),
        "continuous_priority_rows": row_summaries,
    }
    return r_update, weights, dist_meta, [int(value) for value in map_update_order.tolist()]


@torch.no_grad()
def _online_spectral_apply_try19_bridge(rows, current_iter, total_samples):
    global online_spectral_policy_cached_order, online_spectral_policy_last_meta
    global online_spectral_policy_priority_ema, online_spectral_policy_last_priority_update
    global online_spectral_policy_map_order, online_spectral_policy_hybrid_order
    global online_spectral_policy_distribution_updates
    global online_spectral_policy_last_candidate_weights, online_spectral_policy_last_distribution_meta
    global online_spectral_policy_map_order_changed_kendall
    global online_spectral_policy_updates, online_spectral_policy_attention_updates
    global online_spectral_policy_failed_updates
    global online_spectral_policy_last_update_iter, online_spectral_policy_last_attention_iter
    global online_spectral_policy_last_attention_samples
    global online_spectral_policy_try19_bridge_updates, online_spectral_policy_try19_bridge_last_meta

    if not _online_spectral_try19_bridge_should_update(current_iter):
        return False
    mode_name = _online_spectral_try19_bridge_mode()
    candidates = _online_spectral_try19_bridge_candidates_from_rows(rows)
    if not candidates:
        online_spectral_policy_failed_updates += 1
        if master_process:
            print(f"try19 bridge policy update failed at iter {int(current_iter)}: no valid oriented candidates")
        _online_spectral_policy_refresh_latest_stats(policy_refreshed=False)
        return False

    online_spectral_policy_attention_updates += 1
    online_spectral_policy_last_attention_iter = int(current_iter)
    online_spectral_policy_last_attention_samples = int(total_samples) * int(ddp_world_size)

    if mode_name in {"fixed", "fixed_top1", "top1", "direct"}:
        if _online_spectral_distribution_mode_enabled():
            raise ValueError(
                "online_spectral_policy_try19_bridge_mode='fixed_top1' expects "
                "aogpt_train_mode='OnlineSpectralFixedHeadOrder'."
            )
        weights = [
            max(float(candidate.get("score", 0.0)), float(online_spectral_policy_try19_bridge_min_candidate_score))
            for candidate in candidates
        ]
        denom = max(float(sum(weights)), 1e-8)
        q_matrix = sum(
            float(weight) * _head_signal_precedence_matrix(candidate["order"])
            for weight, candidate in zip(weights, candidates)
        ) / float(denom)
        consensus_order = _head_signal_order_from_precedence_matrix(q_matrix)
        bridge_candidate = {
            "name": "try19_bridge_fixed_top1_consensus",
            "order": [int(value) for value in consensus_order],
            "score": float(np.mean(weights)) if weights else 0.0,
            "meta": {
                "try19_bridge": True,
                "try19_bridge_mode": str(mode_name),
                "try19_bridge_candidate_count": int(len(candidates)),
                "try19_bridge_weight_sum": float(denom),
                "try19_bridge_weight_max": float(max(weights)) if weights else float("nan"),
                "try19_bridge_weight_entropy": float(
                    -sum(
                        (float(weight) / denom) * math.log(max(float(weight) / denom, 1e-12))
                        for weight in weights
                    )
                ),
                "no_prior_rule": (
                    "complete order from same-probe oriented head consensus; no original tau/PPL/history"
                ),
            },
        }
        online_spectral_policy_cached_order = torch.tensor(
            consensus_order,
            dtype=torch.long,
            device="cpu",
        )
        online_spectral_policy_last_meta = bridge_candidate
        history_candidates = [bridge_candidate] + candidates
        selected_order = consensus_order
    elif mode_name in {"distribution", "oriented_distribution", "candidate_distribution", "dist"}:
        if not _online_spectral_distribution_mode_enabled():
            raise ValueError(
                "online_spectral_policy_try19_bridge_mode='oriented_distribution' expects "
                "aogpt_train_mode='OnlineSpectralOrderDistribution'."
            )
        try:
            r_update, weights, dist_meta = candidates_to_priority_vector(
                candidates,
                num_blocks=num_blocks,
                teacher_temperature=float(online_spectral_policy_teacher_temperature),
                score_normalization=str(online_spectral_policy_score_normalization),
            )
        except Exception as exc:
            online_spectral_policy_failed_updates += 1
            if master_process:
                print(f"try19 bridge distribution update failed at iter {int(current_iter)}: {exc}")
            _online_spectral_policy_refresh_latest_stats(policy_refreshed=False)
            return False
        if r_update.numel() != int(num_blocks) or not torch.isfinite(r_update).all():
            raise ValueError("Try19 bridge distribution produced an invalid priority vector.")
        previous_map_order = online_spectral_policy_map_order
        decay = min(0.9999, max(0.0, float(online_spectral_policy_priority_ema_decay)))
        if online_spectral_policy_priority_ema is None:
            online_spectral_policy_priority_ema = r_update.detach().cpu()
        else:
            online_spectral_policy_priority_ema = (
                online_spectral_policy_priority_ema.float() * decay
                + r_update.detach().cpu().float() * (1.0 - decay)
            )
        online_spectral_policy_last_priority_update = r_update.detach().cpu()
        online_spectral_policy_map_order = torch.argsort(
            online_spectral_policy_priority_ema.float(),
            descending=True,
        ).to(dtype=torch.long, device="cpu")
        if previous_map_order is not None:
            online_spectral_policy_map_order_changed_kendall = _kendall_tau_between_orders(
                previous_map_order,
                online_spectral_policy_map_order,
            )
        else:
            online_spectral_policy_map_order_changed_kendall = None
        online_spectral_policy_last_candidate_weights = weights.detach().cpu()
        online_spectral_policy_last_distribution_meta = dict(dist_meta)
        online_spectral_policy_last_distribution_meta.update(
            {
                "top_candidate_name": str(candidates[0].get("name", "")),
                "top_candidate_score": float(candidates[0].get("score", float("nan"))),
                "priority_ema_decay": float(online_spectral_policy_priority_ema_decay),
                "try19_bridge": True,
                "try19_bridge_mode": str(mode_name),
                "try19_bridge_candidate_count": int(len(candidates)),
            }
        )
        online_spectral_policy_last_meta = candidates[0]
        online_spectral_policy_distribution_updates += 1
        history_candidates = candidates
        selected_order = [int(value) for value in online_spectral_policy_map_order.tolist()]
    elif mode_name in {
        "continuous_fiedler_minmax",
        "continuous_fiedler_priority",
        "continuous_priority_ema",
        "fiedler_priority_ema",
        "continuous_minmax",
    }:
        if not _online_spectral_distribution_mode_enabled():
            raise ValueError(
                "continuous Fiedler priority bridge expects "
                "aogpt_train_mode='OnlineSpectralOrderDistribution'."
            )
        try:
            r_update, weights, dist_meta, update_order = (
                _online_spectral_try19_bridge_continuous_priority_from_rows(rows)
            )
        except Exception as exc:
            online_spectral_policy_failed_updates += 1
            if master_process:
                print(f"continuous Fiedler priority bridge update failed at iter {int(current_iter)}: {exc}")
            _online_spectral_policy_refresh_latest_stats(policy_refreshed=False)
            return False
        if r_update.numel() != int(num_blocks) or not torch.isfinite(r_update).all():
            raise ValueError("Continuous Fiedler priority bridge produced an invalid priority vector.")
        previous_map_order = online_spectral_policy_map_order
        decay = min(0.9999, max(0.0, float(online_spectral_policy_priority_ema_decay)))
        if online_spectral_policy_priority_ema is None:
            online_spectral_policy_priority_ema = r_update.detach().cpu()
        else:
            online_spectral_policy_priority_ema = (
                online_spectral_policy_priority_ema.float() * decay
                + r_update.detach().cpu().float() * (1.0 - decay)
            )
        online_spectral_policy_last_priority_update = r_update.detach().cpu()
        online_spectral_policy_map_order = torch.argsort(
            online_spectral_policy_priority_ema.float(),
            descending=True,
        ).to(dtype=torch.long, device="cpu")
        if previous_map_order is not None:
            online_spectral_policy_map_order_changed_kendall = _kendall_tau_between_orders(
                previous_map_order,
                online_spectral_policy_map_order,
            )
        else:
            online_spectral_policy_map_order_changed_kendall = None
        bridge_candidate = {
            "name": "try19_bridge_continuous_fiedler_minmax",
            "order": [int(value) for value in update_order],
            "score": float(dist_meta.get("priority_range", 0.0)),
            "meta": {
                "try19_bridge": True,
                "try19_bridge_mode": str(mode_name),
                "try19_bridge_candidate_count": int(len(candidates)),
                "continuous_priority_bridge": True,
                "continuous_priority_normalization": str(
                    online_spectral_policy_continuous_priority_normalization
                ),
                "no_prior_rule": (
                    "oriented Fiedler coordinates are min-max normalized into current-frame "
                    "continuous priority updates; original tau/PPL/history are diagnostics only"
                ),
            },
        }
        online_spectral_policy_last_candidate_weights = weights.detach().cpu()
        online_spectral_policy_last_distribution_meta = dict(dist_meta)
        online_spectral_policy_last_distribution_meta.update(
            {
                "top_candidate_name": str(bridge_candidate.get("name", "")),
                "top_candidate_score": float(bridge_candidate.get("score", float("nan"))),
                "priority_ema_decay": float(online_spectral_policy_priority_ema_decay),
                "try19_bridge": True,
                "try19_bridge_mode": str(mode_name),
                "try19_bridge_candidate_count": int(len(candidates)),
            }
        )
        online_spectral_policy_last_meta = bridge_candidate
        online_spectral_policy_distribution_updates += 1
        history_candidates = [bridge_candidate] + candidates
        selected_order = [int(value) for value in online_spectral_policy_map_order.tolist()]
    elif mode_name in {"hybrid", "hybrid_direct_ema", "direct_ema", "fixed_distribution_mix", "hard_ema"}:
        if not _online_spectral_distribution_mode_enabled():
            raise ValueError(
                "online_spectral_policy_try19_bridge_mode='hybrid_direct_ema' expects "
                "aogpt_train_mode='OnlineSpectralOrderDistribution'."
            )
        weights_list = [
            max(float(candidate.get("score", 0.0)), float(online_spectral_policy_try19_bridge_min_candidate_score))
            for candidate in candidates
        ]
        denom = max(float(sum(weights_list)), 1e-8)
        q_matrix = sum(
            float(weight) * _head_signal_precedence_matrix(candidate["order"])
            for weight, candidate in zip(weights_list, candidates)
        ) / float(denom)
        consensus_order = _head_signal_order_from_precedence_matrix(q_matrix)
        skip_ema_update = (
            bool(online_spectral_policy_hybrid_update_hard_after_ema_stop)
            and _online_spectral_policy_updates_frozen(current_iter)
            and online_spectral_policy_priority_ema is not None
            and online_spectral_policy_map_order is not None
        )
        if skip_ema_update:
            weights = torch.as_tensor(weights_list, dtype=torch.float32, device="cpu")
            weights = weights / weights.sum().clamp_min(1e-8)
            dist_meta = {
                "ema_update_skipped_after_stop": True,
                "ema_update_stop_iter": int(online_spectral_policy_update_stop_iter),
                "priority_ema_frozen": True,
            }
        else:
            try:
                r_update, weights, dist_meta = candidates_to_priority_vector(
                    candidates,
                    num_blocks=num_blocks,
                    teacher_temperature=float(online_spectral_policy_teacher_temperature),
                    score_normalization=str(online_spectral_policy_score_normalization),
                )
            except Exception as exc:
                online_spectral_policy_failed_updates += 1
                if master_process:
                    print(f"try19 bridge hybrid update failed at iter {int(current_iter)}: {exc}")
                _online_spectral_policy_refresh_latest_stats(policy_refreshed=False)
                return False
            if r_update.numel() != int(num_blocks) or not torch.isfinite(r_update).all():
                raise ValueError("Try19 bridge hybrid produced an invalid priority vector.")
        previous_map_order = online_spectral_policy_map_order
        online_spectral_policy_cached_order = torch.tensor(
            consensus_order,
            dtype=torch.long,
            device="cpu",
        )
        if not skip_ema_update:
            decay = min(0.9999, max(0.0, float(online_spectral_policy_priority_ema_decay)))
            if online_spectral_policy_priority_ema is None:
                online_spectral_policy_priority_ema = r_update.detach().cpu()
            else:
                online_spectral_policy_priority_ema = (
                    online_spectral_policy_priority_ema.float() * decay
                    + r_update.detach().cpu().float() * (1.0 - decay)
                )
            online_spectral_policy_last_priority_update = r_update.detach().cpu()
            online_spectral_policy_map_order = torch.argsort(
                online_spectral_policy_priority_ema.float(),
                descending=True,
            ).to(dtype=torch.long, device="cpu")
        direct_prob = _online_spectral_policy_hybrid_direct_prob_for_iter(current_iter)
        if online_spectral_policy_map_order is not None:
            online_spectral_policy_hybrid_order = _online_spectral_blend_hard_ema_orders(
                online_spectral_policy_cached_order,
                online_spectral_policy_map_order,
                direct_prob,
            )
        else:
            online_spectral_policy_hybrid_order = online_spectral_policy_cached_order.detach().cpu()
        if skip_ema_update and online_spectral_policy_map_order is not None:
            online_spectral_policy_map_order_changed_kendall = 1.0
        elif previous_map_order is not None:
            online_spectral_policy_map_order_changed_kendall = _kendall_tau_between_orders(
                previous_map_order,
                online_spectral_policy_map_order,
            )
        else:
            online_spectral_policy_map_order_changed_kendall = None
        online_spectral_policy_last_candidate_weights = weights.detach().cpu()
        online_spectral_policy_last_distribution_meta = dict(dist_meta)
        online_spectral_policy_last_distribution_meta.update(
            {
                "top_candidate_name": str(candidates[0].get("name", "")),
                "top_candidate_score": float(candidates[0].get("score", float("nan"))),
                "priority_ema_decay": float(online_spectral_policy_priority_ema_decay),
                "priority_ema_update_skipped": bool(skip_ema_update),
                "try19_bridge": True,
                "try19_bridge_mode": str(mode_name),
                "try19_bridge_candidate_count": int(len(candidates)),
                "hybrid_direct_prob": float(direct_prob),
                "hybrid_freeze_order": str(online_spectral_policy_hybrid_freeze_order),
                "hybrid_hard_weight_sum": float(denom),
                "hybrid_hard_weight_max": float(max(weights_list)) if weights_list else float("nan"),
            }
        )
        bridge_candidate = {
            "name": "try19_bridge_hybrid_direct_ema",
            "order": [int(value) for value in online_spectral_policy_hybrid_order.tolist()],
            "score": float(np.mean(weights_list)) if weights_list else 0.0,
            "meta": {
                "try19_bridge": True,
                "try19_bridge_mode": str(mode_name),
                "try19_bridge_candidate_count": int(len(candidates)),
                "try19_bridge_weight_sum": float(denom),
                "try19_bridge_weight_max": float(max(weights_list)) if weights_list else float("nan"),
                "hybrid_direct_prob": float(direct_prob),
                "hybrid_freeze_order": str(online_spectral_policy_hybrid_freeze_order),
                "no_prior_rule": (
                    "sample-level hard/EMA hybrid from same-probe oriented head consensus and "
                    "current priority_ema; no original tau/PPL/history"
                ),
            },
        }
        online_spectral_policy_last_meta = bridge_candidate
        if not skip_ema_update:
            online_spectral_policy_distribution_updates += 1
        history_candidates = [bridge_candidate] + candidates
        selected_order = [int(value) for value in online_spectral_policy_hybrid_order.tolist()]
    else:
        raise ValueError(
            "Unsupported online_spectral_policy_try19_bridge_mode="
            f"{online_spectral_policy_try19_bridge_mode!r}; expected fixed_top1, oriented_distribution, "
            "continuous_fiedler_minmax, or hybrid_direct_ema."
        )

    online_spectral_policy_try19_bridge_updates += 1
    bridge_summary = _online_spectral_try19_bridge_summary(
        candidates,
        selected_order,
        current_iter,
        total_samples,
    )
    bridge_summary["updates"] = int(online_spectral_policy_try19_bridge_updates)
    online_spectral_policy_try19_bridge_last_meta = bridge_summary
    if isinstance(online_spectral_policy_last_meta, dict):
        meta = dict(online_spectral_policy_last_meta.get("meta", {}))
        meta["try19_bridge_summary"] = dict(bridge_summary)
        online_spectral_policy_last_meta["meta"] = meta
    online_spectral_policy_updates += 1
    online_spectral_policy_last_update_iter = int(current_iter)
    _online_spectral_policy_refresh_latest_stats(policy_refreshed=True)
    _online_spectral_policy_write_order_history(current_iter, history_candidates)
    if master_process:
        first16 = [int(value) for value in selected_order[:16]]
        print(
            f"try19 bridge {mode_name} policy update "
            f"{int(online_spectral_policy_try19_bridge_updates)} at iter {int(current_iter)}: "
            f"candidates={int(len(candidates))}, "
            f"gap_mean={float(bridge_summary.get('score_gap_mean', float('nan'))):.6f}, "
            f"fallback_frac={float(bridge_summary.get('fallback_frac', float('nan'))):.3f}, "
            f"selected_order_current_first16={first16}"
        )
    return True


def _online_spectral_policy_refresh_latest_stats(policy_refreshed=False):
    global online_spectral_policy_latest_stats
    if not _online_spectral_policy_mode_enabled():
        online_spectral_policy_latest_stats = None
        return
    stats = {
        "online_spectral_policy_prob": float(_online_spectral_policy_prob_for_iter(iter_num)),
        "online_spectral_policy_updates": float(online_spectral_policy_updates),
        "online_spectral_distribution_updates": float(online_spectral_policy_distribution_updates),
        "online_spectral_policy_attention_updates": float(online_spectral_policy_attention_updates),
        "online_spectral_policy_random_count": float(online_spectral_policy_random_count),
        "online_spectral_policy_fallback_count": float(online_spectral_policy_fallback_count),
        "online_spectral_late_l2r_prob": float(_online_spectral_policy_late_l2r_prob_for_iter(iter_num)),
        "online_spectral_late_l2r_count": float(online_spectral_policy_late_l2r_count),
        "online_spectral_policy_failed_updates": float(online_spectral_policy_failed_updates),
        "online_spectral_policy_last_attention_samples": float(online_spectral_policy_last_attention_samples),
        "online_spectral_policy_cached": float(
            online_spectral_policy_cached_order is not None or online_spectral_policy_priority_ema is not None
        ),
        "online_spectral_policy_refreshed": float(bool(policy_refreshed)),
        "online_spectral_policy_last_update_iter": float(online_spectral_policy_last_update_iter),
        "online_spectral_policy_last_attention_iter": float(online_spectral_policy_last_attention_iter),
        "online_spectral_policy_update_start_iter": float(online_spectral_policy_update_start_iter),
        "online_spectral_policy_update_stop_iter": float(online_spectral_policy_update_stop_iter),
        "online_spectral_policy_updates_frozen": float(_online_spectral_policy_updates_frozen(iter_num)),
        "online_spectral_freeze_to_map_order_after_stop": float(
            bool(online_spectral_policy_freeze_to_map_order_after_stop)
        ),
        "online_spectral_random_mix_prob": float(online_spectral_policy_random_mix_prob),
        "online_spectral_sample_temperature": float(online_spectral_policy_sample_temperature),
        "online_spectral_distribution_sample_mode": str(online_spectral_policy_distribution_sample_mode),
        "online_spectral_order_usage_mode": str(_online_spectral_policy_order_usage_mode()),
        "online_spectral_temperature_sampling_stage": str(
            _online_spectral_policy_temperature_sampling_stage(iter_num)
        ),
        "online_spectral_temperature_sampling_stage_code": float(
            _online_spectral_policy_temperature_sampling_stage_code(iter_num)
        ),
        "online_spectral_temperature_sampling_current_temperature": float(
            _online_spectral_policy_temperature_for_iter(iter_num)
        ),
        "online_spectral_temperature_sampling_start_temperature": float(
            online_spectral_policy_temperature_sampling_start_temperature
        ),
        "online_spectral_temperature_sampling_end_temperature": float(
            online_spectral_policy_temperature_sampling_end_temperature
        ),
        "online_spectral_temperature_sampling_schedule": str(online_spectral_policy_temperature_sampling_schedule),
        "online_spectral_try19_bridge_enabled": float(bool(online_spectral_policy_try19_bridge_enabled)),
        "online_spectral_try19_bridge_updates": float(online_spectral_policy_try19_bridge_updates),
        "online_spectral_try19_bridge_mode": str(online_spectral_policy_try19_bridge_mode),
        "online_spectral_continuous_priority_normalization": str(
            online_spectral_policy_continuous_priority_normalization
        ),
        "online_spectral_hybrid_direct_prob": float(
            _online_spectral_policy_hybrid_direct_prob_for_iter(iter_num)
        ),
        "online_spectral_hybrid_per_sample": float(bool(online_spectral_policy_hybrid_per_sample)),
        "online_spectral_hybrid_freeze_order": str(online_spectral_policy_hybrid_freeze_order),
        "online_spectral_hybrid_order_cached": float(online_spectral_policy_hybrid_order is not None),
    }
    if online_spectral_policy_A_ema is not None:
        finite = online_spectral_policy_A_ema[torch.isfinite(online_spectral_policy_A_ema)]
        if finite.numel() > 0:
            stats.update(
                {
                    "online_spectral_A_ema_mean": float(finite.mean().item()),
                    "online_spectral_A_ema_std": float(finite.std(unbiased=False).item()),
                }
            )
    if isinstance(online_spectral_policy_last_meta, dict):
        stats["online_spectral_score"] = float(online_spectral_policy_last_meta.get("score", float("nan")))
        meta = online_spectral_policy_last_meta.get("meta", {})
        if isinstance(meta, dict):
            stats["online_spectral_band_quality"] = float(meta.get("band_quality", float("nan")))
            stats["online_spectral_adjacency_path_score"] = float(meta.get("adjacency_path_score", float("nan")))
            stats["online_spectral_directed_path_score"] = float(meta.get("best_directed_path_score", float("nan")))
            stats["online_spectral_num_unique_candidates"] = float(meta.get("num_unique_candidates", float("nan")))
            stats["online_spectral_attention_spectral_score"] = float(
                meta.get("attention_spectral_score", online_spectral_policy_last_meta.get("score", float("nan")))
            )
            stats["online_spectral_loss_rerank_prefix_loss"] = float(
                meta.get("loss_rerank_prefix_loss", float("nan"))
            )
            stats["online_spectral_loss_rerank_full_loss"] = float(
                meta.get("loss_rerank_full_loss", float("nan"))
            )
            stats["online_spectral_loss_rerank_score"] = float(meta.get("loss_rerank_score", float("nan")))
    if online_spectral_policy_priority_ema is not None:
        priority = online_spectral_policy_priority_ema.float()
        finite = priority[torch.isfinite(priority)]
        if finite.numel() > 0:
            weights = torch.softmax(priority, dim=0)
            priority_entropy = float((-(weights * weights.clamp_min(1e-12).log()).sum()).item())
            stats.update(
                {
                    "online_spectral_priority_mean": float(finite.mean().item()),
                    "online_spectral_priority_std": float(finite.std(unbiased=False).item()),
                    "online_spectral_priority_entropy": float(priority_entropy),
                }
            )
    if isinstance(online_spectral_policy_last_distribution_meta, dict):
        stats["online_spectral_top_m_used"] = float(
            online_spectral_policy_last_distribution_meta.get("top_m_used", float("nan"))
        )
        stats["online_spectral_candidate_weight_entropy"] = float(
            online_spectral_policy_last_distribution_meta.get("weight_entropy", float("nan"))
        )
        stats["online_spectral_candidate_top_weight"] = float(
            online_spectral_policy_last_distribution_meta.get("top_weight", float("nan"))
        )
        stats["online_spectral_priority_update_std"] = float(
            online_spectral_policy_last_distribution_meta.get("priority_std", float("nan"))
        )
        stats["online_spectral_priority_update_range"] = float(
            online_spectral_policy_last_distribution_meta.get("priority_range", float("nan"))
        )
        stats["online_spectral_priority_update_effectively_constant"] = float(
            bool(online_spectral_policy_last_distribution_meta.get("priority_effectively_constant", False))
        )
        stats["online_spectral_reverse_pair_weight_mass"] = float(
            online_spectral_policy_last_distribution_meta.get("reverse_pair_weight_mass", float("nan"))
        )
    if online_spectral_policy_map_order_changed_kendall is not None:
        stats["online_spectral_map_order_changed_kendall"] = float(online_spectral_policy_map_order_changed_kendall)
    if isinstance(online_spectral_policy_try19_bridge_last_meta, dict):
        stats["online_spectral_try19_bridge_candidate_count"] = float(
            online_spectral_policy_try19_bridge_last_meta.get("candidate_count", float("nan"))
        )
        stats["online_spectral_try19_bridge_score_gap_mean"] = float(
            online_spectral_policy_try19_bridge_last_meta.get("score_gap_mean", float("nan"))
        )
        stats["online_spectral_try19_bridge_score_gap_min"] = float(
            online_spectral_policy_try19_bridge_last_meta.get("score_gap_min", float("nan"))
        )
        stats["online_spectral_try19_bridge_fallback_frac"] = float(
            online_spectral_policy_try19_bridge_last_meta.get("fallback_frac", float("nan"))
        )
        selected_tau = online_spectral_policy_try19_bridge_last_meta.get("selected_tau_diagnostic")
        selected_abs_tau = online_spectral_policy_try19_bridge_last_meta.get("selected_abs_tau_diagnostic")
        if selected_tau is not None:
            stats["online_spectral_try19_bridge_selected_tau_diagnostic"] = float(selected_tau)
        if selected_abs_tau is not None:
            stats["online_spectral_try19_bridge_selected_abs_tau_diagnostic"] = float(selected_abs_tau)
    online_spectral_policy_latest_stats = stats


def _online_spectral_policy_checkpoint_state():
    if not _online_spectral_policy_mode_enabled():
        return None
    state = {
        "enabled": bool(_online_spectral_policy_mode_enabled()),
        "layer": int(online_spectral_policy_layer),
        "head": int(online_spectral_policy_head),
        "export_type": str(online_spectral_policy_export_type),
        "update_every": int(online_spectral_policy_update_every),
        "update_start_iter": int(online_spectral_policy_update_start_iter),
        "update_stop_iter": int(online_spectral_policy_update_stop_iter),
        "freeze_to_map_order_after_stop": bool(online_spectral_policy_freeze_to_map_order_after_stop),
        "probe_batches": int(online_spectral_policy_probe_batches),
        "probe_batch_size": int(online_spectral_policy_probe_batch_size),
        "probe_split": str(online_spectral_policy_probe_split),
        "probe_include_train_step_attention": bool(online_spectral_policy_probe_include_train_step_attention),
        "ema_decay": float(online_spectral_policy_ema_decay),
        "use_ema": bool(online_spectral_policy_use_ema),
        "anneal_start_iter": int(online_spectral_policy_anneal_start_iter),
        "anneal_end_iter": int(online_spectral_policy_anneal_end_iter),
        "start_prob": float(online_spectral_policy_start_prob),
        "end_prob": float(online_spectral_policy_end_prob),
        "prob_schedule": str(online_spectral_policy_prob_schedule),
        "prob_points": str(online_spectral_policy_prob_points),
        "policy_updates": int(online_spectral_policy_updates),
        "distribution_updates": int(online_spectral_policy_distribution_updates),
        "attention_updates": int(online_spectral_policy_attention_updates),
        "random_count": int(online_spectral_policy_random_count),
        "fallback_count": int(online_spectral_policy_fallback_count),
        "late_l2r_count": int(online_spectral_policy_late_l2r_count),
        "failed_updates": int(online_spectral_policy_failed_updates),
        "last_update_iter": int(online_spectral_policy_last_update_iter),
        "last_attention_iter": int(online_spectral_policy_last_attention_iter),
        "last_attention_samples": int(online_spectral_policy_last_attention_samples),
        "config": vars(online_spectral_policy_config) if online_spectral_policy_config is not None else {},
        "last_meta": online_spectral_policy_last_meta,
        "top_m": int(online_spectral_policy_top_m),
        "teacher_temperature": float(online_spectral_policy_teacher_temperature),
        "score_normalization": str(online_spectral_policy_score_normalization),
        "priority_ema_decay": float(online_spectral_policy_priority_ema_decay),
        "continuous_priority_normalization": str(online_spectral_policy_continuous_priority_normalization),
        "continuous_priority_eps": float(online_spectral_policy_continuous_priority_eps),
        "distribution_sample_mode": str(online_spectral_policy_distribution_sample_mode),
        "sample_temperature": float(online_spectral_policy_sample_temperature),
        "random_mix_prob": float(online_spectral_policy_random_mix_prob),
        "order_usage_mode": str(_online_spectral_policy_order_usage_mode()),
        "temperature_sampling_stage": str(_online_spectral_policy_temperature_sampling_stage(iter_num)),
        "temperature_sampling_current_temperature": float(_online_spectral_policy_temperature_for_iter(iter_num)),
        "temperature_sampling_start_temperature": float(
            online_spectral_policy_temperature_sampling_start_temperature
        ),
        "temperature_sampling_end_temperature": float(online_spectral_policy_temperature_sampling_end_temperature),
        "temperature_sampling_schedule": str(online_spectral_policy_temperature_sampling_schedule),
        "distribution_per_sample": bool(online_spectral_policy_distribution_per_sample),
        "loss_rerank_enabled": bool(online_spectral_policy_loss_rerank_enabled),
        "loss_rerank_top_k": int(online_spectral_policy_loss_rerank_top_k),
        "loss_rerank_batches": int(online_spectral_policy_loss_rerank_batches),
        "loss_rerank_batch_size": int(online_spectral_policy_loss_rerank_batch_size),
        "loss_rerank_split": str(online_spectral_policy_loss_rerank_split),
        "loss_rerank_candidate_batch_size": int(online_spectral_policy_loss_rerank_candidate_batch_size),
        "loss_rerank_prefix_k": int(online_spectral_policy_loss_rerank_prefix_k),
        "loss_rerank_attention_weight": float(online_spectral_policy_loss_rerank_attention_weight),
        "loss_rerank_prefix_weight": float(online_spectral_policy_loss_rerank_prefix_weight),
        "loss_rerank_full_weight": float(online_spectral_policy_loss_rerank_full_weight),
        "late_l2r_anneal_enabled": bool(online_spectral_policy_late_l2r_anneal_enabled),
        "late_l2r_anneal_start_iter": int(online_spectral_policy_late_l2r_anneal_start_iter),
        "late_l2r_anneal_end_iter": int(online_spectral_policy_late_l2r_anneal_end_iter),
        "late_l2r_start_prob": float(online_spectral_policy_late_l2r_start_prob),
        "late_l2r_end_prob": float(online_spectral_policy_late_l2r_end_prob),
        "late_l2r_per_sample": bool(online_spectral_policy_late_l2r_per_sample),
        "late_l2r_current_prob": float(_online_spectral_policy_late_l2r_prob_for_iter(iter_num)),
        "last_distribution_meta": online_spectral_policy_last_distribution_meta,
        "map_order_changed_kendall": online_spectral_policy_map_order_changed_kendall,
        "try19_bridge_enabled": bool(online_spectral_policy_try19_bridge_enabled),
        "try19_bridge_mode": str(online_spectral_policy_try19_bridge_mode),
        "try19_bridge_updates": int(online_spectral_policy_try19_bridge_updates),
        "try19_bridge_last_meta": online_spectral_policy_try19_bridge_last_meta,
        "hybrid_direct_prob_start": float(online_spectral_policy_hybrid_direct_prob_start),
        "hybrid_direct_prob_end": float(online_spectral_policy_hybrid_direct_prob_end),
        "hybrid_direct_prob_anneal_start_iter": int(
            online_spectral_policy_hybrid_direct_prob_anneal_start_iter
        ),
        "hybrid_direct_prob_anneal_end_iter": int(
            online_spectral_policy_hybrid_direct_prob_anneal_end_iter
        ),
        "hybrid_direct_prob_current": float(_online_spectral_policy_hybrid_direct_prob_for_iter(iter_num)),
        "hybrid_direct_prob_schedule": str(online_spectral_policy_hybrid_direct_prob_schedule),
        "hybrid_direct_prob_points": str(online_spectral_policy_hybrid_direct_prob_points),
        "hybrid_per_sample": bool(online_spectral_policy_hybrid_per_sample),
        "hybrid_freeze_order": str(online_spectral_policy_hybrid_freeze_order),
        "hybrid_mix_hard_after_ema_stop": bool(online_spectral_policy_hybrid_mix_hard_after_ema_stop),
        "hybrid_update_hard_after_ema_stop": bool(online_spectral_policy_hybrid_update_hard_after_ema_stop),
        "note": (
            "cached_order, A_ema, priority_ema, priority_update, map_order, hybrid_order, "
            "and sampled orders are in current-frame block ids/coordinates. Original-frame fields "
            "are diagnostics only and are not used by this policy."
        ),
    }
    if online_spectral_policy_A_ema is not None:
        state["A_ema"] = online_spectral_policy_A_ema.detach().cpu()
    if online_spectral_policy_cached_order is not None:
        state["cached_order"] = online_spectral_policy_cached_order.detach().cpu()
        if bool(online_spectral_policy_save_orders):
            order_current = [int(v) for v in online_spectral_policy_cached_order.detach().cpu().tolist()]
            state["cached_order_current"] = order_current
            if permute_data and fixed_block_perm is not None:
                mapper = fixed_block_perm.to(dtype=torch.long, device=online_spectral_policy_cached_order.device)
                state["cached_order_original"] = [
                    int(v) for v in mapper[online_spectral_policy_cached_order.long()].detach().cpu().tolist()
                ]
    if online_spectral_policy_priority_ema is not None:
        state["priority_ema"] = online_spectral_policy_priority_ema.detach().cpu()
    if online_spectral_policy_last_priority_update is not None:
        state["priority_update"] = online_spectral_policy_last_priority_update.detach().cpu()
    if online_spectral_policy_map_order is not None:
        state["map_order"] = online_spectral_policy_map_order.detach().cpu()
        if bool(online_spectral_policy_save_orders):
            map_order_current = [int(v) for v in online_spectral_policy_map_order.detach().cpu().tolist()]
            state["map_order_current"] = map_order_current
            if permute_data and fixed_block_perm is not None:
                mapper = fixed_block_perm.to(dtype=torch.long, device=online_spectral_policy_map_order.device)
                state["map_order_original"] = [
                    int(v) for v in mapper[online_spectral_policy_map_order.long()].detach().cpu().tolist()
                ]
                state["map_order_original_note"] = (
                    "Original-frame order is saved only for diagnostics and is not used by policy."
                )
    if online_spectral_policy_hybrid_order is not None:
        state["hybrid_order"] = online_spectral_policy_hybrid_order.detach().cpu()
        if bool(online_spectral_policy_save_orders):
            hybrid_order_current = [
                int(v) for v in online_spectral_policy_hybrid_order.detach().cpu().tolist()
            ]
            state["hybrid_order_current"] = hybrid_order_current
            if permute_data and fixed_block_perm is not None:
                mapper = fixed_block_perm.to(dtype=torch.long, device=online_spectral_policy_hybrid_order.device)
                state["hybrid_order_original"] = [
                    int(v) for v in mapper[online_spectral_policy_hybrid_order.long()].detach().cpu().tolist()
                ]
                state["hybrid_order_original_note"] = (
                    "Original-frame order is saved only for diagnostics and is not used by policy."
                )
    if online_spectral_policy_last_candidate_weights is not None:
        state["last_candidate_weights"] = online_spectral_policy_last_candidate_weights.detach().cpu()
    return state


_init_attn_mlp_policy()
_init_online_spectral_policy()


def _save_iteration_checkpoint(checkpoint_payload, iter_value):
    if not bool(save_iter_checkpoints):
        return None
    if str(save_iter_checkpoint_steps).strip():
        allowed_steps = {
            int(item.strip())
            for item in str(save_iter_checkpoint_steps).split(',')
            if item.strip()
        }
        if int(iter_value) not in allowed_steps:
            return None
    checkpoint_dir = (
        save_iter_checkpoint_dir
        if str(save_iter_checkpoint_dir).strip()
        else os.path.join(out_dir, 'checkpoints')
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, f'ckpt_iter{int(iter_value):07d}.pt')
    torch.save(checkpoint_payload, save_path)

    keep_n = int(save_iter_checkpoint_keep)
    if keep_n > 0:
        checkpoint_files = sorted(
            [
                filename
                for filename in os.listdir(checkpoint_dir)
                if filename.startswith('ckpt_iter') and filename.endswith('.pt')
            ]
        )
        stale_files = checkpoint_files[:-keep_n]
        for filename in stale_files:
            stale_path = os.path.join(checkpoint_dir, filename)
            if stale_path != save_path and os.path.exists(stale_path):
                os.remove(stale_path)
    return save_path


def _aggregate_top_pairs_to_segments(top_pairs, num_blocks_local, top_k, max_segment_len):
    top_k = max(1, min(int(top_k), len(top_pairs)))
    max_segment_len = max(2, int(max_segment_len))
    next_map = {}
    prev_map = {}

    def trace_start(node):
        while node in prev_map:
            node = prev_map[node]
        return node

    def build_segment(start):
        values = [start]
        seen = {start}
        node = start
        while node in next_map:
            node = next_map[node]
            if node in seen:
                break
            values.append(node)
            seen.add(node)
        return values

    for pair in top_pairs[:top_k]:
        first = int(pair["first"])
        second = int(pair["second"])
        if first == second:
            continue
        if first in next_map or second in prev_map:
            continue
        start_first = trace_start(first)
        start_second = trace_start(second)
        if start_first == start_second:
            continue
        seg_first = build_segment(start_first)
        seg_second = build_segment(start_second)
        if len(seg_first) + len(seg_second) > max_segment_len:
            continue
        next_map[first] = second
        prev_map[second] = first

    starts = [node for node in range(num_blocks_local) if node not in prev_map and node in next_map]
    segments = []
    for start in starts:
        values = build_segment(start)
        if len(values) >= 2:
            segments.append(values)
    segments.sort(key=lambda values: (-len(values), values[0], values[-1]))
    return segments


def _load_segment_library(segment_json_path):
    if not segment_json_path:
        return []
    if not os.path.exists(segment_json_path):
        raise FileNotFoundError(f"segment_source_json not found: {segment_json_path}")
    with open(segment_json_path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    raw_segments = None
    if payload.get('final_units'):
        raw_segments = payload.get('final_units', [])
    elif payload.get('aggregated_segments'):
        raw_segments = payload.get('aggregated_segments')
    elif payload.get('levels'):
        levels = payload.get('levels', [])
        if levels:
            last_level = levels[-1]
            raw_segments = last_level.get('next_units') or last_level.get('aggregated_segments', [])
    if raw_segments:
        segments = []
        for row in raw_segments:
            if "segment" in row:
                segment = row.get("segment", [])
            elif "blocks" in row:
                segment = row.get("blocks", [])
            else:
                segment = []
            if len(segment) >= 2:
                segments.append([int(v) for v in segment])
    else:
        top_pairs = payload.get('top_pairs', [])
        segments = _aggregate_top_pairs_to_segments(
            top_pairs,
            num_blocks_local=num_blocks,
            top_k=segment_top_k_pairs,
            max_segment_len=segment_max_len,
        )
    cleaned = []
    for segment in segments:
        if any(v < 0 or v >= num_blocks for v in segment):
            continue
        if len(segment) >= 2:
            cleaned.append(segment)
    return cleaned


def _segment_source_uses_locked_units(segment_json_path):
    if not segment_json_path or not os.path.exists(segment_json_path):
        return False
    with open(segment_json_path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    return bool(payload.get('final_units')) or bool(payload.get('levels'))


def _parse_segment_schedule_csv(raw_value, cast_fn, field_name):
    if raw_value is None or not str(raw_value).strip():
        return []
    values = []
    for item in str(raw_value).split(','):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(cast_fn(item))
        except Exception as exc:
            raise ValueError(f"Could not parse {field_name} entry {item!r}") from exc
    return values


def _parse_segment_schedule_sources(raw_value):
    if raw_value is None or not str(raw_value).strip():
        return []
    return [item.strip() for item in str(raw_value).split(',') if item.strip()]


def _build_segment_schedule_entries():
    boundaries = _parse_segment_schedule_csv(segment_schedule_boundaries, int, "segment_schedule_boundaries")
    ratios = _parse_segment_schedule_csv(segment_schedule_ratios, float, "segment_schedule_ratios")
    sources = _parse_segment_schedule_sources(segment_schedule_source_jsons)
    if not boundaries and not ratios and not sources:
        return []
    if not (boundaries and ratios and sources):
        raise ValueError(
            "segment_schedule_boundaries, segment_schedule_ratios, and "
            "segment_schedule_source_jsons must be provided together."
        )
    if len(boundaries) != len(ratios) or len(boundaries) != len(sources):
        raise ValueError(
            "segment schedule lengths must match: "
            f"boundaries={len(boundaries)}, ratios={len(ratios)}, sources={len(sources)}"
        )
    previous_boundary = -1
    entries = []
    missing_sources = []
    for idx, (boundary, ratio, source) in enumerate(zip(boundaries, ratios, sources)):
        boundary = int(boundary)
        if boundary <= previous_boundary:
            raise ValueError("segment_schedule_boundaries must be strictly increasing.")
        previous_boundary = boundary
        ratio = float(ratio)
        if ratio > 0.0 and not os.path.exists(source):
            missing_sources.append(source)
        entries.append(
            {
                "index": int(idx),
                "boundary": boundary,
                "ratio": ratio,
                "source": source,
            }
        )
    if missing_sources:
        missing_text = "\n".join(f"  - {path}" for path in missing_sources)
        raise FileNotFoundError(
            "segment schedule source JSON files are missing:\n"
            f"{missing_text}\n"
            "A single-process A2/A2a run reuses precomputed policies, so all "
            "scheduled sources must exist before training starts."
        )
    return entries


segment_library = []
segment_source_locked_units = False
effective_segment_use_all_units = bool(segment_use_all_units)
segment_schedule_entries = _build_segment_schedule_entries()
_active_segment_schedule_index = None


def _set_active_segment_policy(segment_json_path, ratio, schedule_index=None, reason="initial"):
    global segment_guided_ratio
    global segment_source_json
    global segment_library
    global segment_source_locked_units
    global effective_segment_use_all_units

    segment_source_json = str(segment_json_path or '')
    segment_guided_ratio = float(ratio)
    segment_library = _load_segment_library(segment_source_json)
    segment_source_locked_units = _segment_source_uses_locked_units(segment_source_json)
    effective_segment_use_all_units = bool(segment_use_all_units) or (
        bool(segment_lock_final_units) and bool(segment_source_locked_units)
    )
    if master_process and segment_guided_ratio > 0.0:
        prefix = "segment schedule" if schedule_index is not None else "segment-guided mode"
        stage_text = f" stage_index={int(schedule_index)}" if schedule_index is not None else ""
        if len(segment_library) == 0:
            print(
                f"warning: {prefix}{stage_text} ratio={segment_guided_ratio} but no usable segments "
                f"were loaded from {segment_source_json or '[none]'}; falling back to pure random orders."
            )
        else:
            print(
                f"{prefix}{stage_text}: loaded {len(segment_library)} aggregated segments "
                f"from {segment_source_json or '[none]'} ({reason})"
            )
            if bool(effective_segment_use_all_units):
                print("segment-guided mode: using all non-overlapping aggregated segments in each guided sample")
            else:
                print(
                    "segment-guided mode: sampling a subset of aggregated segments "
                    f"(max_units_per_order={int(segment_max_units_per_order)}) in each guided sample"
                )


def _segment_schedule_index_for_iter(current_iter):
    active_index = None
    for entry in segment_schedule_entries:
        if int(current_iter) >= int(entry["boundary"]):
            active_index = int(entry["index"])
        else:
            break
    return active_index


def _maybe_update_segment_schedule(current_iter):
    global _active_segment_schedule_index
    if not segment_schedule_entries:
        return
    active_index = _segment_schedule_index_for_iter(current_iter)
    if active_index is None or active_index == _active_segment_schedule_index:
        return
    entry = segment_schedule_entries[active_index]
    _set_active_segment_policy(
        entry["source"],
        entry["ratio"],
        schedule_index=active_index,
        reason=f"iter {int(current_iter)} reached boundary {int(entry['boundary'])}",
    )
    _active_segment_schedule_index = active_index


_set_active_segment_policy(segment_source_json, segment_guided_ratio)
if master_process and segment_schedule_entries:
    print("segment schedule enabled:")
    for entry in segment_schedule_entries:
        print(
            "  "
            f"iter >= {int(entry['boundary'])}: ratio={float(entry['ratio'])}, "
            f"source={entry['source']}"
        )


def _sample_mixed_segment_guided_block_orders(batch_size_local, device_local, return_units=False):
    orders = []
    ordered_units_per_sample = []
    for _ in range(batch_size_local):
        if len(segment_library) == 0 or np.random.random() >= float(segment_guided_ratio):
            random_order = torch.randperm(num_blocks, device=device_local)
            orders.append(random_order)
            if return_units:
                ordered_units_per_sample.append([[int(v)] for v in random_order.detach().cpu().tolist()])
            continue

        shuffled_segment_indices = np.random.permutation(len(segment_library)).tolist()
        chosen_segments = []
        used_blocks = set()
        for seg_idx in shuffled_segment_indices:
            segment = segment_library[seg_idx]
            if (not bool(effective_segment_use_all_units)) and len(chosen_segments) >= int(segment_max_units_per_order):
                break
            if any(value in used_blocks for value in segment):
                continue
            chosen_segments.append(segment)
            used_blocks.update(segment)

        units = []
        for segment in chosen_segments:
            units.append(list(segment))
        for block_idx in range(num_blocks):
            if block_idx not in used_blocks:
                units.append([block_idx])

        unit_perm = np.random.permutation(len(units)).tolist()
        order = []
        ordered_units = []
        for unit_idx in unit_perm:
            unit = [int(value) for value in units[unit_idx]]
            ordered_units.append(unit)
            order.extend(unit)
        orders.append(torch.tensor(order, device=device_local, dtype=torch.long))
        if return_units:
            ordered_units_per_sample.append(ordered_units)
    stacked = torch.stack(orders, dim=0)
    if return_units:
        return stacked, ordered_units_per_sample
    return stacked


def _build_segment_guided_units():
    shuffled_segment_indices = np.random.permutation(len(segment_library)).tolist()
    chosen_segments = []
    used_blocks = set()
    for seg_idx in shuffled_segment_indices:
        segment = segment_library[seg_idx]
        if (not bool(effective_segment_use_all_units)) and len(chosen_segments) >= int(segment_max_units_per_order):
            break
        if any(value in used_blocks for value in segment):
            continue
        chosen_segments.append(segment)
        used_blocks.update(segment)

    units = []
    for segment in chosen_segments:
        units.append(list(segment))
    for block_idx in range(num_blocks):
        if block_idx not in used_blocks:
            units.append([block_idx])
    return units


def _segment_guided_policy_active():
    return (
        train_stage == 'standard'
        and aogpt_train_mode == 'Random'
        and float(segment_guided_ratio) > 0.0
        and len(segment_library) > 0
    )


def _singleton_units_for_block_orders(block_orders):
    return [
        [[int(value)] for value in row]
        for row in block_orders.detach().cpu().tolist()
    ]


def _expand_training_block_orders(block_orders):
    return expand_block_orders_to_token_orders(
        block_orders,
        block_len=effective_order_block_len,
        block_order_layout=block_order_layout,
        image_size=image_size,
        image_block_size=image_block_size,
        image_block_height=image_block_height,
        image_block_width=image_block_width,
    )


fixed_block_order_cached = None


def _fixed_block_order_tensor(device_local):
    global fixed_block_order_cached
    if fixed_block_order_cached is None:
        raw = str(fixed_block_order).strip()
        if not raw:
            raise ValueError(
                "aogpt_train_mode='FixedBlockOrder' requires fixed_block_order "
                "as a comma/space separated permutation of block ids."
            )
        tokens = raw.replace(",", " ").split()
        values = [int(token) for token in tokens]
        if len(values) != int(num_blocks):
            raise ValueError(
                f"fixed_block_order has {len(values)} entries; expected num_blocks={num_blocks}."
            )
        expected = list(range(int(num_blocks)))
        if sorted(values) != expected:
            raise ValueError(
                "fixed_block_order must be a permutation of block ids "
                f"0..{int(num_blocks) - 1}; got sorted={sorted(values)}."
            )
        fixed_block_order_cached = torch.tensor(values, dtype=torch.long)
        print(f"FixedBlockOrder loaded with {len(values)} blocks: {values}")
    return fixed_block_order_cached.to(device=device_local, dtype=torch.long)


def _sample_explicit_training_block_orders(idx, return_units=False):
    global attn_mlp_policy_fallback_count
    if _segment_guided_policy_active():
        sampled = _sample_mixed_segment_guided_block_orders(
            idx.size(0),
            idx.device,
            return_units=return_units,
        )
        if return_units:
            block_orders, ordered_units = sampled
        else:
            block_orders = sampled
            ordered_units = None
        return block_orders, ordered_units, "segment_guided_random"

    if aogpt_train_mode == 'Random':
        block_orders = sample_random_block_orders(
            batch_size=idx.size(0),
            num_blocks=num_blocks,
            device=idx.device,
        )
        ordered_units = _singleton_units_for_block_orders(block_orders) if return_units else None
        return block_orders, ordered_units, "Random"

    if aogpt_train_mode == 'AR':
        block_orders = torch.arange(num_blocks, device=idx.device).unsqueeze(0).expand(idx.size(0), -1)
        ordered_units = _singleton_units_for_block_orders(block_orders) if return_units else None
        return block_orders, ordered_units, "AR"

    if aogpt_train_mode == 'FixedBlockOrder':
        fixed_order = _fixed_block_order_tensor(idx.device)
        block_orders = fixed_order.unsqueeze(0).expand(idx.size(0), -1)
        ordered_units = _singleton_units_for_block_orders(block_orders) if return_units else None
        return block_orders, ordered_units, "FixedBlockOrder"

    if aogpt_train_mode == 'AttnMLPFrozenOrder':
        if _attn_mlp_policy_active_for_iter(iter_num):
            block_orders = _attn_mlp_cached_block_order(idx.size(0), idx.device)
            prob = _attn_mlp_policy_prob_for_iter(iter_num)
            if block_orders is not None and np.random.random() < float(prob):
                ordered_units = _singleton_units_for_block_orders(block_orders) if return_units else None
                return block_orders, ordered_units, "AttnMLPFrozenOrder"
            fallback = str(attn_mlp_policy_fallback)
            if fallback == "random":
                if torch.is_grad_enabled():
                    attn_mlp_policy_fallback_count += 1
                block_orders = _attn_mlp_random_fallback_block_orders(idx.size(0), idx.device)
                ordered_units = _singleton_units_for_block_orders(block_orders) if return_units else None
                return block_orders, ordered_units, "AttnMLPFrozenOrderAnnealRandom"
            if fallback == "ar":
                if torch.is_grad_enabled():
                    attn_mlp_policy_fallback_count += 1
                block_orders = torch.arange(num_blocks, device=idx.device).unsqueeze(0).expand(idx.size(0), -1)
                ordered_units = _singleton_units_for_block_orders(block_orders) if return_units else None
                return block_orders, ordered_units, "AttnMLPFrozenOrderFallbackAR"
            raise ValueError(f"Unsupported attn_mlp_policy_fallback={fallback!r}")
        block_orders = _attn_mlp_random_fallback_block_orders(idx.size(0), idx.device)
        ordered_units = _singleton_units_for_block_orders(block_orders) if return_units else None
        return block_orders, ordered_units, "AttnMLPWarmupRandom"

    if aogpt_train_mode == 'OnlineSpectralFixedHeadOrder':
        return _online_spectral_policy_sample_block_orders(idx, return_units=return_units)

    if aogpt_train_mode == 'OnlineSpectralOrderDistribution':
        return _online_spectral_distribution_sample_block_orders(idx, return_units=return_units)

    if aogpt_train_mode == 'GBetaFrozenOrder':
        # Frozen gβ: batch-mean strict65 B -> argsort -> ONE model-frame order,
        # broadcast to all rows. is_eval detects the outer no_grad (eval) context.
        block_orders = _gbeta_provider.block_orders(
            raw_model, idx, iter_num, is_eval=not torch.is_grad_enabled())
        ordered_units = _singleton_units_for_block_orders(block_orders) if return_units else None
        return block_orders, ordered_units, "GBetaFrozenOrder"

    return None, None, str(aogpt_train_mode)


def _forward_with_explicit_block_orders(
    idx,
    block_orders,
    return_token_loss=False,
    return_attentions=False,
    return_logits=True,
):
    token_orders = _expand_training_block_orders(block_orders)
    return model(
        idx,
        mode=None,
        orders=token_orders,
        return_token_loss=return_token_loss,
        return_attentions=return_attentions,
        return_logits=return_logits,
    )


def _forward_with_segment_guided_policy(idx, return_token_loss=False):
    mixed_block_orders = _sample_mixed_segment_guided_block_orders(
        idx.size(0),
        idx.device,
    )
    return _forward_with_explicit_block_orders(
        idx,
        mixed_block_orders,
        return_token_loss=return_token_loss,
    )


def _forward_with_active_training_policy(
    idx,
    return_token_loss=False,
    return_order_info=False,
    return_attentions=False,
    return_logits=True,
):
    needs_explicit_policy = (
        return_order_info
        or return_attentions
        or str(aogpt_train_mode) == "FixedBlockOrder"
        or str(aogpt_train_mode) == "AttnMLPFrozenOrder"
        or str(aogpt_train_mode) == "OnlineSpectralFixedHeadOrder"
        or str(aogpt_train_mode) == "OnlineSpectralOrderDistribution"
        or _segment_guided_policy_active()
    )
    if needs_explicit_policy:
        block_orders, ordered_units, policy_name = _sample_explicit_training_block_orders(
            idx,
            return_units=return_order_info,
        )
        if block_orders is not None:
            outputs = _forward_with_explicit_block_orders(
                idx,
                block_orders,
                return_token_loss=return_token_loss,
                return_attentions=return_attentions,
                return_logits=return_logits,
            )
            info = {"block_orders": block_orders}
            if return_order_info:
                info["ordered_units"] = ordered_units
            return outputs, policy_name, info

    if _segment_guided_policy_active():
        return _forward_with_segment_guided_policy(idx, return_token_loss=return_token_loss), "segment_guided_random", None
    return model(idx, mode=aogpt_train_mode), str(aogpt_train_mode), None


def _forward_with_main_eval_policy(idx, return_token_loss=False, return_logits=True):
    return model(
        idx,
        mode=main_eval_mode,
        return_token_loss=return_token_loss,
        return_logits=return_logits,
    ), str(main_eval_mode)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(
    weight_decay,
    learning_rate,
    (beta1, beta2),
    device_type,
)
if init_from in ('resume', 'ckpt') and resume_optimizer_state \
        and isinstance(checkpoint.get('optimizer'), dict) and checkpoint['optimizer']:
    optimizer.load_state_dict(checkpoint['optimizer'])

# ── V3 frozen-gβ order provider (constructed once; used by the dispatch branch) ──
_gbeta_provider = None
if aogpt_train_mode == 'GBetaFrozenOrder':
    from orderhead_v3.gbeta_provider import GBetaFrozenProvider
    _gbeta_provider = GBetaFrozenProvider(
        gbeta_ckpt, init_from_ckpt,
        batch_mean_probes=gbeta_batch_mean_probes,
        refresh_every=gbeta_refresh_every,
        seed=seed, device=device, probe_mode=gbeta_probe_mode)
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    eval_batch_size_override = int(eval_batch_size) if int(eval_batch_size) > 0 else None
    original_token_orders = None
    aligned_policy_name = None
    main_eval_policy_name = None
    log_main_eval_reference = _segment_guided_policy_active()
    if permute_data and inverse_block_perm is not None:
        original_block_orders = inverse_block_perm.unsqueeze(0)
        original_token_orders = expand_block_orders_to_token_orders(
            original_block_orders,
            block_len=effective_order_block_len,
            block_order_layout=block_order_layout,
            image_size=image_size,
            image_block_size=image_block_size,
            image_block_height=image_block_height,
            image_block_width=image_block_width,
        )
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        main_eval_losses = torch.zeros(eval_iters) if log_main_eval_reference else None
        for k in range(eval_iters):
            X, Y = get_batch(split, batch_size_override=eval_batch_size_override)
            with ctx:
                forward_outputs, aligned_policy_name, _ = _forward_with_active_training_policy(
                    X,
                    return_token_loss=True,
                    return_logits=False,
                )
                loss = forward_outputs[1]
                if log_main_eval_reference:
                    main_eval_outputs, main_eval_policy_name = _forward_with_main_eval_policy(
                        X,
                        return_token_loss=True,
                        return_logits=False,
                    )
                    main_eval_loss = main_eval_outputs[1]
            losses[k] = loss.item()
            if log_main_eval_reference:
                main_eval_losses[k] = main_eval_loss.item()
        out[split] = losses.mean().item()
        if log_main_eval_reference:
            out[f"{split}_main_eval_loss"] = main_eval_losses.mean().item()
        if split == 'val':
            l2r_losses = torch.zeros(eval_iters)
            origin_l2r_losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split, batch_size_override=eval_batch_size_override)
                with ctx:
                    l2r_outputs = model(
                        X,
                        mode='AR',
                        return_token_loss=True,
                        return_logits=False,
                    )
                    l2r_loss = l2r_outputs[1]
                    if original_token_orders is not None:
                        batch_original_token_orders = original_token_orders.to(device=X.device).expand(X.size(0), -1)
                        origin_l2r_outputs = model(
                            X,
                            mode=None,
                            orders=batch_original_token_orders,
                            return_token_loss=True,
                            return_logits=False,
                        )
                        origin_l2r_loss = origin_l2r_outputs[1]
                    else:
                        origin_l2r_loss = l2r_loss
                l2r_losses[k] = l2r_loss.item()
                origin_l2r_losses[k] = origin_l2r_loss.item()
            out["val_l2r_loss"] = l2r_losses.mean().item()
            out["val_origin_l2r_loss"] = origin_l2r_losses.mean().item()
    if aligned_policy_name is not None:
        out["eval_policy_name"] = str(aligned_policy_name)
    if log_main_eval_reference and main_eval_policy_name is not None:
        out["main_eval_policy_name"] = str(main_eval_policy_name)
    model.train()
    return out

@torch.no_grad()
def estimate_eval_generate_step_block_loss_curves(num_batches_override=None):
    model.eval()
    local_num_batches = int(eval_generate_step_batches if num_batches_override is None else num_batches_override)
    ar_block_curves = []
    random_block_curves = []
    original_l2r_block_curves = [] if permute_data and inverse_block_perm is not None else None

    for _ in range(local_num_batches):
        X, _ = get_batch('val')
        with ctx:
            _, _, ar_token_losses = model(
                X,
                mode='AR',
                return_token_loss=True,
            )
            _, _, random_token_losses = model(
                X,
                mode='Random',
                return_token_loss=True,
            )
            if original_l2r_block_curves is not None:
                original_block_orders = inverse_block_perm.to(device=X.device).unsqueeze(0).expand(X.size(0), -1)
                original_token_orders = expand_block_orders_to_token_orders(
                    original_block_orders,
                    block_len=effective_order_block_len,
                    block_order_layout=block_order_layout,
                    image_size=image_size,
                    image_block_size=image_block_size,
                    image_block_height=image_block_height,
                    image_block_width=image_block_width,
                )
                _, _, original_l2r_token_losses = model(
                    X,
                    mode=None,
                    orders=original_token_orders,
                    return_token_loss=True,
                )
        ar_block_losses = token_losses_to_block_losses(
            ar_token_losses,
            block_len=effective_order_block_len,
        )
        random_block_losses = token_losses_to_block_losses(
            random_token_losses,
            block_len=effective_order_block_len,
        )
        ar_block_curves.append(ar_block_losses.mean(dim=0).float().cpu())
        random_block_curves.append(random_block_losses.mean(dim=0).float().cpu())
        if original_l2r_block_curves is not None:
            original_l2r_block_losses = token_losses_to_block_losses(
                original_l2r_token_losses,
                block_len=effective_order_block_len,
            )
            original_l2r_block_curves.append(original_l2r_block_losses.mean(dim=0).float().cpu())

    model.train()
    ar_curve = torch.stack(ar_block_curves, dim=0).mean(dim=0).numpy()
    random_curve = torch.stack(random_block_curves, dim=0).mean(dim=0).numpy()
    original_l2r_curve = None
    if original_l2r_block_curves is not None and len(original_l2r_block_curves) > 0:
        original_l2r_curve = torch.stack(original_l2r_block_curves, dim=0).mean(dim=0).numpy()
    return ar_curve, random_curve, original_l2r_curve

def build_generate_step_block_loss_figure(ar_curve, random_curve, original_l2r_curve=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    steps = np.arange(len(ar_curve))
    ax.plot(steps, ar_curve, label='AR mode', linewidth=2.0)
    ax.plot(steps, random_curve, label='Random mode', linewidth=2.0)
    if original_l2r_curve is not None:
        ax.plot(steps, original_l2r_curve, label='Original-frame L2R', linewidth=2.0)
    ax.set_title(f'Val Mean Per-Step Block Loss ({eval_generate_step_batches} batches)')
    ax.set_xlabel('Generate / Reveal Block Step')
    ax.set_ylabel('Mean Block Loss')
    ax.legend()
    ax.grid(alpha=0.25)
    return fig


@torch.no_grad()
def estimate_sampled_order_kendall_distance_stats(num_orders_override=None):
    local_num_orders = int(
        eval_kendall_num_orders if num_orders_override is None else num_orders_override
    )
    if local_num_orders <= 0:
        raise ValueError("eval_kendall_num_orders must be positive.")
    fixed_policy_order = None
    sample_mode = None
    map_order = None
    if (
        str(aogpt_train_mode) == "OnlineSpectralOrderDistribution"
        and _online_spectral_hybrid_mode_enabled()
        and online_spectral_policy_priority_ema is not None
        and online_spectral_policy_cached_order is not None
        and online_spectral_policy_map_order is not None
    ):
        direct_prob = _online_spectral_policy_hybrid_direct_prob_for_iter(iter_num)
        if (
            _online_spectral_policy_updates_frozen(iter_num)
            and bool(online_spectral_policy_freeze_to_map_order_after_stop)
        ):
            if bool(online_spectral_policy_hybrid_mix_hard_after_ema_stop):
                hard_order = online_spectral_policy_cached_order.to(device=device, dtype=torch.long)
                ema_order = online_spectral_policy_map_order.to(device=device, dtype=torch.long)
                rows = []
                for _ in range(local_num_orders):
                    if np.random.random() < float(direct_prob):
                        rows.append(hard_order)
                    else:
                        rows.append(ema_order)
                sampled_orders = torch.stack(rows, dim=0)
                if online_spectral_policy_hybrid_order is not None:
                    map_order = online_spectral_policy_hybrid_order.to(device=device, dtype=torch.long).unsqueeze(0)
                else:
                    map_order = _online_spectral_blend_hard_ema_orders(
                        online_spectral_policy_cached_order,
                        online_spectral_policy_map_order,
                        direct_prob,
                    ).to(device=device, dtype=torch.long).unsqueeze(0)
                max_unique_orders = 2
                sample_mode = "online_spectral_order_distribution_frozen_ema_hard_mix"
            else:
                frozen_order, frozen_source = _online_spectral_hybrid_frozen_order()
                if frozen_order is None:
                    frozen_order = online_spectral_policy_map_order.detach().cpu().to(dtype=torch.long)
                    frozen_source = "EMAFallback"
                sampled_orders = frozen_order.to(device=device, dtype=torch.long).unsqueeze(0)
                fixed_policy_order = sampled_orders
                map_order = sampled_orders
                max_unique_orders = 1
                sample_mode = f"online_spectral_order_distribution_frozen_{str(frozen_source).lower()}_order"
        else:
            hard_order = online_spectral_policy_cached_order.to(device=device, dtype=torch.long)
            ema_order = online_spectral_policy_map_order.to(device=device, dtype=torch.long)
            rows = []
            for _ in range(local_num_orders):
                if np.random.random() < float(direct_prob):
                    rows.append(hard_order)
                else:
                    rows.append(ema_order)
            sampled_orders = torch.stack(rows, dim=0)
            if online_spectral_policy_hybrid_order is not None:
                map_order = online_spectral_policy_hybrid_order.to(device=device, dtype=torch.long).unsqueeze(0)
            else:
                map_order = _online_spectral_blend_hard_ema_orders(
                    online_spectral_policy_cached_order,
                    online_spectral_policy_map_order,
                    direct_prob,
                ).to(device=device, dtype=torch.long).unsqueeze(0)
            max_unique_orders = 2
            sample_mode = "online_spectral_order_distribution_hybrid_direct_ema"
    elif (
        str(aogpt_train_mode) == "OnlineSpectralOrderDistribution"
        and online_spectral_policy_priority_ema is not None
    ):
        priority = online_spectral_policy_priority_ema.to(device=device, dtype=torch.float32)
        sampled_orders = torch.stack(
            [
                sample_order_from_priority(
                    priority,
                    temperature=float(online_spectral_policy_sample_temperature),
                ).to(device=device, dtype=torch.long)
                for _ in range(local_num_orders)
            ],
            dim=0,
        )
        map_order = torch.argsort(priority, descending=True).to(device=device, dtype=torch.long).unsqueeze(0)
        max_unique_orders = None
        sample_mode = "online_spectral_order_distribution_sampled"
    elif str(aogpt_train_mode) == "OnlineSpectralFixedHeadOrder" and online_spectral_policy_cached_order is not None:
        fixed_policy_order = online_spectral_policy_cached_order.to(device=device, dtype=torch.long).unsqueeze(0)
        sample_mode = "online_spectral_cached_order"
    elif str(aogpt_train_mode) == "AttnMLPFrozenOrder" and attn_mlp_policy_cached_order is not None:
        fixed_policy_order = attn_mlp_policy_cached_order.to(device=device, dtype=torch.long).unsqueeze(0)
        sample_mode = "attn_mlp_cached_order"
    elif str(aogpt_train_mode) == "FixedBlockOrder":
        fixed_policy_order = _fixed_block_order_tensor(device).unsqueeze(0)
        sample_mode = "fixed_block_order"
    using_segment_guided = (
        train_stage == 'standard'
        and aogpt_train_mode == 'Random'
        and float(segment_guided_ratio) > 0.0
        and len(segment_library) > 0
    )
    if sample_mode == "online_spectral_order_distribution_sampled":
        pass
    elif fixed_policy_order is not None:
        max_unique_orders = 1
        sampled_orders = fixed_policy_order
    elif using_segment_guided:
        units = _build_segment_guided_units()
        num_units = len(units)
        max_unique_orders = math.factorial(num_units) if num_units <= 8 else None
        if max_unique_orders is not None and max_unique_orders <= local_num_orders:
            orders = []
            for perm in itertools.permutations(range(num_units)):
                order = []
                for unit_idx in perm:
                    order.extend(units[unit_idx])
                orders.append(torch.tensor(order, device=device, dtype=torch.long))
            sampled_orders = torch.stack(orders, dim=0)
        else:
            sampled_orders = _sample_mixed_segment_guided_block_orders(
                local_num_orders,
                device,
            )
        sample_mode = "segment_guided"
    else:
        max_unique_orders = None
        sampled_orders = sample_random_block_orders(
            batch_size=local_num_orders,
            num_blocks=num_blocks,
            device=device,
        )
        sample_mode = "pure_random"
    if permute_data and fixed_block_perm is not None:
        order_mapper = fixed_block_perm.to(device=sampled_orders.device, dtype=torch.long)
        sampled_orders_origin = order_mapper[sampled_orders.long()]
        map_order_origin = None if map_order is None else order_mapper[map_order.long()]
    else:
        sampled_orders_origin = sampled_orders
        map_order_origin = map_order
    kendall_tau = kendall_tau_to_l2r_per_sample(sampled_orders_origin).float().cpu()
    map_kendall_tau = None
    if map_order_origin is not None:
        map_kendall_tau = float(kendall_tau_to_l2r_per_sample(map_order_origin).float().cpu()[0].item())
    total_pairs = num_blocks * (num_blocks - 1) / 2.0
    kendall_distance = (1.0 - kendall_tau) * (total_pairs / 2.0)
    normalized_distance = kendall_distance / total_pairs if total_pairs > 0 else kendall_distance
    payload = {
        "origin_kendall_tau": kendall_tau.numpy(),
        "origin_kendall_distance": kendall_distance.numpy(),
        "origin_normalized_distance": normalized_distance.numpy(),
        "total_pairs": float(total_pairs),
        "sample_mode": sample_mode,
        "num_orders_used": int(sampled_orders.size(0)),
        "is_fixed_policy_order": bool(fixed_policy_order is not None),
    }
    if map_kendall_tau is not None:
        payload["origin_kendall_tau_map_order"] = float(map_kendall_tau)
    return payload


def save_figure_to_out_dir(fig, filename, dpi=200):
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, filename)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return save_path


online_pair_stats = {}
online_pair_stats_samples = 0
online_pair_stats_updates = 0
online_spectral_matrices_sum = None
online_spectral_matrices_ema = None
online_spectral_samples = 0
online_spectral_updates = 0
online_spectral_subspace_q = None
online_spectral_global_subspace_q = None
online_spectral_last_write_update = -1


def _online_stats_enabled():
    return bool(online_pair_stats_enabled)


def _online_stats_dir():
    if str(online_pair_stats_out_dir).strip():
        return str(online_pair_stats_out_dir)
    return os.path.join(out_dir, "online_pair_stats")


def _online_spectral_enabled():
    return bool(online_spectral_enabled)


def _online_spectral_dir():
    if str(online_spectral_out_dir).strip():
        return str(online_spectral_out_dir)
    return os.path.join(out_dir, "online_spectral")


def _online_pair_key(first_unit, second_unit):
    first = tuple(int(value) for value in first_unit)
    second = tuple(int(value) for value in second_unit)
    return first, second


def _online_pair_key_text(first_unit, second_unit):
    first, second = _online_pair_key(first_unit, second_unit)
    return json.dumps([list(first), list(second)], separators=(",", ":"))


def _online_pair_row(first_unit, second_unit):
    return {
        "first": [int(value) for value in first_unit],
        "second": [int(value) for value in second_unit],
        "count": 0,
        "first_loss_sum": 0.0,
        "second_loss_sum": 0.0,
        "signed_drop_sum": 0.0,
        "signed_drop_sq_sum": 0.0,
        "positive_count": 0,
    }


def _update_online_pair_stats(order_info, token_losses):
    global online_pair_stats_samples, online_pair_stats_updates
    if not _online_stats_enabled() or order_info is None or token_losses is None:
        return
    ordered_units = order_info.get("ordered_units")
    if not ordered_units:
        return
    block_losses = token_losses_to_block_losses(
        token_losses.detach(),
        block_len=effective_order_block_len,
    ).float().detach().cpu()
    for sample_idx, units in enumerate(ordered_units):
        if len(units) < 2:
            continue
        first_unit = [int(value) for value in units[0]]
        second_unit = [int(value) for value in units[1]]
        first_len = len(first_unit)
        second_len = len(second_unit)
        if first_len <= 0 or second_len <= 0 or first_len + second_len > block_losses.size(1):
            continue
        first_loss = float(block_losses[sample_idx, :first_len].mean().item())
        second_loss = float(block_losses[sample_idx, first_len:first_len + second_len].mean().item())
        signed_drop = first_loss - second_loss
        key = _online_pair_key_text(first_unit, second_unit)
        row = online_pair_stats.get(key)
        if row is None:
            row = _online_pair_row(first_unit, second_unit)
            online_pair_stats[key] = row
        row["count"] += 1
        row["first_loss_sum"] += first_loss
        row["second_loss_sum"] += second_loss
        row["signed_drop_sum"] += signed_drop
        row["signed_drop_sq_sum"] += signed_drop * signed_drop
        if signed_drop > 0.0:
            row["positive_count"] += 1
        online_pair_stats_samples += 1
    online_pair_stats_updates += 1


def _finalize_online_pair_rows():
    rows = []
    keyed_means = {}
    for row in online_pair_stats.values():
        count = int(row.get("count", 0))
        if count <= 0:
            continue
        signed_drop_mean = float(row["signed_drop_sum"] / count)
        signed_drop_sq_mean = float(row["signed_drop_sq_sum"] / count)
        signed_drop_var = max(0.0, signed_drop_sq_mean - signed_drop_mean * signed_drop_mean)
        out_row = {
            "first": [int(value) for value in row["first"]],
            "second": [int(value) for value in row["second"]],
            "count": count,
            "first_loss_mean": float(row["first_loss_sum"] / count),
            "second_loss_mean": float(row["second_loss_sum"] / count),
            "signed_drop_mean": signed_drop_mean,
            "signed_drop_std": float(math.sqrt(signed_drop_var)),
            "signed_drop_positive_rate": float(row["positive_count"] / count),
        }
        rows.append(out_row)
        keyed_means[(_online_pair_key(out_row["first"], out_row["second"]))] = signed_drop_mean
    for row in rows:
        reverse_key = _online_pair_key(row["second"], row["first"])
        reverse_mean = keyed_means.get(reverse_key)
        if reverse_mean is None:
            row["reverse_signed_drop_mean"] = None
            row["reverse_margin"] = None
        else:
            row["reverse_signed_drop_mean"] = float(reverse_mean)
            row["reverse_margin"] = float(row["signed_drop_mean"] - reverse_mean)
    rows.sort(
        key=lambda item: (
            -float(item["reverse_margin"] if item["reverse_margin"] is not None else item["signed_drop_mean"]),
            -float(item["signed_drop_mean"]),
            -int(item["count"]),
            item["first"],
            item["second"],
        )
    )
    return rows


def _write_online_stats(force=False):
    if not _online_stats_enabled():
        return
    if (not force) and int(online_pair_stats_write_every) > 0:
        if int(online_pair_stats_updates) % int(online_pair_stats_write_every) != 0:
            return
    stats_dir = _online_stats_dir()
    os.makedirs(stats_dir, exist_ok=True)
    pair_payload = {
        "meta": {
            "enabled": bool(online_pair_stats_enabled),
            "rank": int(ddp_rank if ddp else 0),
            "world_size": int(ddp_world_size),
            "iter_num": int(iter_num),
            "num_blocks": int(num_blocks),
            "block_order_block_len": int(effective_order_block_len),
            "block_order_layout": str(block_order_layout),
            "image_size": int(image_size),
            "image_block_size": int(image_block_size),
            "image_block_height": int(image_block_height),
            "image_block_width": int(image_block_width),
            "train_policy": str(aogpt_train_mode),
            "segment_guided_ratio": float(segment_guided_ratio),
            "segment_library_size": int(len(segment_library)),
            "samples": int(online_pair_stats_samples),
            "updates": int(online_pair_stats_updates),
            "signal": (
                "Observed first-two current-frame reveal units and their teacher-forced "
                "training losses under the active training policy."
            ),
            "forbidden_priors_used": False,
        },
        "pair_stats": _finalize_online_pair_rows(),
    }
    rank_suffix = f"_rank{int(ddp_rank)}" if ddp else ""
    rank_path = os.path.join(stats_dir, f"online_pair_stats{rank_suffix}.json")
    with open(rank_path, "w", encoding="utf-8") as handle:
        json.dump(pair_payload, handle, ensure_ascii=False, indent=2)
    if master_process:
        with open(os.path.join(stats_dir, "online_pair_stats.json"), "w", encoding="utf-8") as handle:
            json.dump(pair_payload, handle, ensure_ascii=False, indent=2)


def _aggregate_layerhead_attention_to_current_blocks(layer_attn, block_orders, export_type):
    if str(export_type) == "with_none":
        shifted = layer_attn[:, :, :-1, :-1]
        exposure_correct = False
    elif str(export_type) == "without_none":
        shifted = layer_attn[:, :, 1:, 1:]
        exposure_correct = False
    elif str(export_type) == "target_to_observed":
        shifted = layer_attn[:, :, :-1, 1:]
        exposure_correct = True
    else:
        raise ValueError(f"Unsupported online_spectral_export_type={export_type!r}")
    batch, heads, seq_a, seq_b = shifted.shape
    if seq_a != seq_b or seq_a % int(effective_order_block_len) != 0:
        raise ValueError(f"attention shape {tuple(shifted.shape)} is incompatible with block_len={effective_order_block_len}")
    local_num_blocks = seq_a // int(effective_order_block_len)
    block_batch = shifted.float().view(
        batch,
        heads,
        local_num_blocks,
        int(effective_order_block_len),
        local_num_blocks,
        int(effective_order_block_len),
    ).mean(dim=(3, 5))
    total = torch.zeros(
        (heads, local_num_blocks, local_num_blocks),
        dtype=torch.float64,
        device=block_batch.device,
    )
    counts = torch.zeros(
        (local_num_blocks, local_num_blocks),
        dtype=torch.float64,
        device=block_batch.device,
    )
    exposure_mask_reveal = torch.tril(
        torch.ones(
            (local_num_blocks, local_num_blocks),
            dtype=torch.float64,
            device=block_batch.device,
        ),
        diagonal=-1,
    )
    for sample_idx in range(batch):
        inverse = invert_permutation(block_orders[sample_idx].detach()).to(device=block_batch.device)
        current = block_batch[sample_idx][:, inverse, :][:, :, inverse].double()
        if exposure_correct:
            mask = exposure_mask_reveal[inverse, :][:, inverse]
            total += current * mask.unsqueeze(0)
            counts += mask
        else:
            total += current
    if not exposure_correct:
        counts += float(max(1, batch))
    return total / counts.clamp_min(1.0).unsqueeze(0)


def _aggregate_layerhead_attention_to_current_blocks_per_sample(layer_attn, block_orders, export_type):
    if str(export_type) == "with_none":
        shifted = layer_attn[:, :, :-1, :-1]
        exposure_correct = False
    elif str(export_type) == "without_none":
        shifted = layer_attn[:, :, 1:, 1:]
        exposure_correct = False
    elif str(export_type) == "target_to_observed":
        shifted = layer_attn[:, :, :-1, 1:]
        exposure_correct = True
    else:
        raise ValueError(f"Unsupported online_spectral_export_type={export_type!r}")
    batch, heads, seq_a, seq_b = shifted.shape
    if seq_a != seq_b or seq_a % int(effective_order_block_len) != 0:
        raise ValueError(f"attention shape {tuple(shifted.shape)} is incompatible with block_len={effective_order_block_len}")
    local_num_blocks = seq_a // int(effective_order_block_len)
    block_batch = shifted.float().view(
        batch,
        heads,
        local_num_blocks,
        int(effective_order_block_len),
        local_num_blocks,
        int(effective_order_block_len),
    ).mean(dim=(3, 5))
    aligned_samples = []
    exposure_mask_reveal = torch.tril(
        torch.ones(
            (local_num_blocks, local_num_blocks),
            dtype=block_batch.dtype,
            device=block_batch.device,
        ),
        diagonal=-1,
    )
    for sample_idx in range(batch):
        inverse = invert_permutation(block_orders[sample_idx].detach()).to(device=block_batch.device)
        current = block_batch[sample_idx][:, inverse, :][:, :, inverse]
        if exposure_correct:
            mask = exposure_mask_reveal[inverse, :][:, inverse]
            current = current * mask.unsqueeze(0)
        aligned_samples.append(current)
    return torch.stack(aligned_samples, dim=0).float()


_all_head_attn_dataset_pending = []
_all_head_attn_dataset_shard_idx = 0
_all_head_attn_dataset_records = 0
_all_head_attn_dataset_probe_samples = 0
_all_head_attn_dataset_last_iter = None


def _all_head_attn_dataset_dir():
    if str(all_head_attn_dataset_out_dir).strip():
        return str(all_head_attn_dataset_out_dir)
    return os.path.join(out_dir, "all_head_attn_dataset")


def _all_head_attn_dataset_torch_dtype():
    name = str(all_head_attn_dataset_dtype).strip().lower()
    if name in {"float16", "fp16", "half"}:
        return torch.float16
    if name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if name in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(
        f"Unsupported all_head_attn_dataset_dtype={all_head_attn_dataset_dtype!r}; "
        "expected float16, bfloat16, or float32."
    )


def _all_head_attn_dataset_should_collect(current_iter):
    if not bool(all_head_attn_dataset_enabled) or not master_process:
        return False
    current_iter = int(current_iter)
    if current_iter < int(all_head_attn_dataset_start_iter):
        return False
    stop_iter = int(all_head_attn_dataset_stop_iter)
    if stop_iter >= 0 and current_iter > stop_iter:
        return False
    interval = max(1, int(all_head_attn_dataset_interval))
    if current_iter % interval != 0:
        return False
    max_records = int(all_head_attn_dataset_max_records)
    if max_records > 0 and int(_all_head_attn_dataset_records) >= max_records:
        return False
    return True


def _all_head_attn_dataset_block_orders(batch_size_local, device_local, seed):
    mode = str(all_head_attn_dataset_order_mode).strip().lower()
    if mode == "random":
        generator_device = "cuda" if str(device_local).startswith("cuda") else "cpu"
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(int(seed))
        return sample_random_block_orders(
            batch_size=int(batch_size_local),
            num_blocks=num_blocks,
            device=device_local,
            generator=generator,
        )
    if mode in {"ar", "current_ar", "l2r_current"}:
        return torch.arange(num_blocks, device=device_local).unsqueeze(0).expand(int(batch_size_local), -1).clone()
    if mode in {"original_l2r", "original"}:
        if bool(permute_data) and inverse_block_perm is not None:
            order = inverse_block_perm.to(device=device_local, dtype=torch.long)
        else:
            order = torch.arange(num_blocks, device=device_local)
        return order.unsqueeze(0).expand(int(batch_size_local), -1).clone()
    raise ValueError(
        f"Unsupported all_head_attn_dataset_order_mode={all_head_attn_dataset_order_mode!r}; "
        "expected random, ar, or original_l2r."
    )


def _all_head_attn_dataset_metadata():
    return {
        "version": 1,
        "format": "all_head_block_attention_dataset",
        "description": (
            "No-grad probe attention matrices collected during Random training. "
            "Matrices are aligned to the current block frame; no teacher order or MLP target is stored."
        ),
        "out_dir": str(out_dir),
        "dataset": str(dataset),
        "split": str(all_head_attn_dataset_split),
        "iter_start": int(all_head_attn_dataset_start_iter),
        "iter_stop": int(all_head_attn_dataset_stop_iter),
        "interval": int(all_head_attn_dataset_interval),
        "probe_batch_size": int(all_head_attn_dataset_batch_size),
        "micro_batches_per_record": int(all_head_attn_dataset_micro_batches_per_record),
        "probe_samples_per_record": int(all_head_attn_dataset_batch_size) * int(all_head_attn_dataset_micro_batches_per_record),
        "batches_per_step": int(all_head_attn_dataset_batches_per_step),
        "export_type": str(all_head_attn_dataset_export_type),
        "order_mode": str(all_head_attn_dataset_order_mode),
        "dtype": str(all_head_attn_dataset_dtype),
        "seed": int(all_head_attn_dataset_seed),
        "num_layers": int(model.config.n_layer),
        "num_heads": int(model.config.n_head),
        "num_blocks": int(num_blocks),
        "block_len": int(effective_order_block_len),
        "block_size": int(block_size),
        "permute_data": bool(permute_data),
        "permute_seed": int(permute_seed),
        "permute_mode": str(permute_mode),
        "model_type": str(model_type),
        "train_mode": str(aogpt_train_mode),
        "main_eval_mode": str(main_eval_mode),
        "run_name": str(wandb_run_name),
    }


def _all_head_attn_dataset_write_manifest():
    dataset_dir = _all_head_attn_dataset_dir()
    os.makedirs(dataset_dir, exist_ok=True)
    manifest_path = os.path.join(dataset_dir, "manifest.json")
    if os.path.exists(manifest_path):
        return
    metadata = _all_head_attn_dataset_metadata()
    metadata["model_args"] = {
        key: (int(value) if isinstance(value, (np.integer,)) else value)
        for key, value in model_args.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }
    with open(manifest_path, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def _all_head_attn_dataset_write_summary():
    if not bool(all_head_attn_dataset_enabled) or not master_process:
        return
    dataset_dir = _all_head_attn_dataset_dir()
    os.makedirs(dataset_dir, exist_ok=True)
    summary = _all_head_attn_dataset_metadata()
    summary.update({
        "records": int(_all_head_attn_dataset_records),
        "probe_samples": int(_all_head_attn_dataset_probe_samples),
        "shards": int(_all_head_attn_dataset_shard_idx),
        "pending_records": int(len(_all_head_attn_dataset_pending)),
        "last_iter": None if _all_head_attn_dataset_last_iter is None else int(_all_head_attn_dataset_last_iter),
        "wall_time": float(time.time()),
    })
    with open(os.path.join(dataset_dir, "collection_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def _all_head_attn_dataset_flush(force=False):
    global _all_head_attn_dataset_pending
    global _all_head_attn_dataset_shard_idx
    if not bool(all_head_attn_dataset_enabled) or not master_process:
        return
    if not _all_head_attn_dataset_pending:
        _all_head_attn_dataset_write_summary()
        return
    shard_size = max(1, int(all_head_attn_dataset_shard_size))
    if not bool(force) and len(_all_head_attn_dataset_pending) < shard_size:
        return
    dataset_dir = _all_head_attn_dataset_dir()
    os.makedirs(dataset_dir, exist_ok=True)
    _all_head_attn_dataset_write_manifest()
    records_to_write = _all_head_attn_dataset_pending[:shard_size]
    _all_head_attn_dataset_pending = _all_head_attn_dataset_pending[shard_size:]
    payload = {
        "attention": torch.stack([item["attention"] for item in records_to_write], dim=0),
        "iter": torch.tensor([item["iter"] for item in records_to_write], dtype=torch.long),
        "probe_seed": torch.tensor([item["probe_seed"] for item in records_to_write], dtype=torch.long),
        "sample_count": torch.tensor([item["sample_count"] for item in records_to_write], dtype=torch.long),
        "block_orders": torch.stack([item["block_orders"] for item in records_to_write], dim=0),
        "metadata": _all_head_attn_dataset_metadata(),
    }
    shard_path = os.path.join(
        dataset_dir,
        f"all_head_attn_shard_{int(_all_head_attn_dataset_shard_idx):05d}.pt",
    )
    torch.save(payload, shard_path)
    _all_head_attn_dataset_shard_idx += 1
    print(
        f"all_head_attn_dataset wrote {shard_path} "
        f"records={len(records_to_write)} total_records={int(_all_head_attn_dataset_records)}"
    )
    if _all_head_attn_dataset_pending and bool(force):
        _all_head_attn_dataset_flush(force=True)
    else:
        _all_head_attn_dataset_write_summary()


def _all_head_attn_dataset_collect_if_due(current_iter):
    global _all_head_attn_dataset_records
    global _all_head_attn_dataset_probe_samples
    global _all_head_attn_dataset_last_iter
    if not _all_head_attn_dataset_should_collect(current_iter):
        return

    cpu_rng_state = torch.random.get_rng_state()
    np_rng_state = np.random.get_state()
    cuda_rng_state = None
    if device_type == 'cuda' and torch.cuda.is_available():
        cuda_rng_state = torch.cuda.get_rng_state_all()

    was_training = model.training
    model.eval()
    try:
        _all_head_attn_dataset_write_manifest()
        probe_batch_size = max(1, int(all_head_attn_dataset_batch_size))
        split = str(all_head_attn_dataset_split)
        output_dtype = _all_head_attn_dataset_torch_dtype()
        max_records = int(all_head_attn_dataset_max_records)
        batches = max(1, int(all_head_attn_dataset_batches_per_step))
        micro_batches = max(1, int(all_head_attn_dataset_micro_batches_per_record))
        for batch_idx in range(batches):
            if max_records > 0 and int(_all_head_attn_dataset_records) >= max_records:
                break
            record_seed = (
                int(all_head_attn_dataset_seed)
                + int(current_iter) * 1000003
                + int(batch_idx) * 9176
            )
            matrix_sums = None
            total_samples = 0
            block_order_chunks = []
            order_dtype = torch.uint8 if int(num_blocks) <= 255 else torch.int16
            for micro_idx in range(micro_batches):
                probe_seed = record_seed + int(micro_idx) * 1299827
                X_probe, _ = get_batch(split, batch_size_override=probe_batch_size)
                block_orders = _all_head_attn_dataset_block_orders(
                    int(X_probe.size(0)),
                    X_probe.device,
                    probe_seed,
                )
                with torch.no_grad():
                    with ctx:
                        outputs = _forward_with_explicit_block_orders(
                            X_probe,
                            block_orders,
                            return_attentions=True,
                            return_logits=False,
                        )
                attentions = _attn_mlp_extract_attentions(outputs)
                if not attentions:
                    raise RuntimeError("all_head_attn_dataset expected attention tensors, but model returned none.")
                layer_matrices = []
                for layer_attn in attentions:
                    layer_heads = _aggregate_layerhead_attention_to_current_blocks(
                        layer_attn.detach(),
                        block_orders,
                        str(all_head_attn_dataset_export_type),
                    ).detach().cpu()
                    layer_matrices.append(layer_heads)
                micro_matrices = torch.stack(layer_matrices, dim=0).to(dtype=torch.float32, device="cpu")
                if matrix_sums is None:
                    matrix_sums = micro_matrices * float(X_probe.size(0))
                else:
                    matrix_sums += micro_matrices * float(X_probe.size(0))
                total_samples += int(X_probe.size(0))
                block_order_chunks.append(block_orders.detach().cpu().to(dtype=order_dtype))
            if matrix_sums is None or total_samples <= 0:
                continue
            matrices = matrix_sums / float(total_samples)
            for layer_idx in range(matrices.size(0)):
                for head_idx in range(matrices.size(1)):
                    matrices[layer_idx, head_idx].fill_diagonal_(0.0)
            _all_head_attn_dataset_pending.append({
                "attention": matrices.to(dtype=output_dtype),
                "iter": int(current_iter),
                "probe_seed": int(record_seed),
                "sample_count": int(total_samples),
                "block_orders": torch.cat(block_order_chunks, dim=0),
            })
            _all_head_attn_dataset_records += 1
            _all_head_attn_dataset_probe_samples += int(total_samples)
            _all_head_attn_dataset_last_iter = int(current_iter)
        _all_head_attn_dataset_flush(force=False)
    finally:
        if was_training:
            model.train()
        torch.random.set_rng_state(cpu_rng_state)
        np.random.set_state(np_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)


_all_head_pairwise_dataset_pending = {"train": [], "val": []}
_all_head_pairwise_dataset_shard_idx = {"train": 0, "val": 0}
_all_head_pairwise_dataset_records = 0
_all_head_pairwise_dataset_samples = {"train": 0, "val": 0}
_all_head_pairwise_dataset_record_counts = {"train": 0, "val": 0}
_all_head_pairwise_dataset_errors = []
_all_head_pairwise_dataset_last_iter = None
_all_head_pairwise_dataset_resume_checked = False


def _all_head_pairwise_dataset_dir():
    if str(all_head_pairwise_dataset_out_dir).strip():
        return str(all_head_pairwise_dataset_out_dir)
    return os.path.join(out_dir, "all_head_pairwise_dataset")


def _all_head_pairwise_dataset_shard_paths(split):
    split_dir = os.path.join(_all_head_pairwise_dataset_dir(), str(split))
    if not os.path.isdir(split_dir):
        return []
    paths = []
    for name in os.listdir(split_dir):
        if name.startswith("shard_") and name.endswith(".pt"):
            paths.append(os.path.join(split_dir, name))
    return sorted(paths)


def _all_head_pairwise_dataset_resume_existing_if_needed():
    global _all_head_pairwise_dataset_resume_checked
    global _all_head_pairwise_dataset_records
    global _all_head_pairwise_dataset_last_iter
    if _all_head_pairwise_dataset_resume_checked:
        return
    _all_head_pairwise_dataset_resume_checked = True
    if not bool(all_head_pairwise_dataset_resume_existing) or not master_process:
        return

    total_samples = 0
    max_record_index = -1
    max_iter = None
    for split in ("train", "val"):
        paths = _all_head_pairwise_dataset_shard_paths(split)
        _all_head_pairwise_dataset_shard_idx[split] = len(paths)
        split_samples = 0
        split_records = set()
        for path in paths:
            payload = torch.load(path, map_location="cpu")
            if "record_index" not in payload:
                continue
            record_index = payload["record_index"].long().view(-1)
            n = int(record_index.numel())
            split_samples += n
            total_samples += n
            if n > 0:
                max_record_index = max(max_record_index, int(record_index.max().item()))
                split_records.update(int(v) for v in record_index.tolist())
            if "iter" in payload:
                iter_values = payload["iter"].long().view(-1)
                if int(iter_values.numel()) > 0:
                    iter_max = int(iter_values.max().item())
                    max_iter = iter_max if max_iter is None else max(max_iter, iter_max)
        _all_head_pairwise_dataset_samples[split] = int(split_samples)
        _all_head_pairwise_dataset_record_counts[split] = int(len(split_records))
    _all_head_pairwise_dataset_records = max(int(total_samples), int(max_record_index) + 1)
    _all_head_pairwise_dataset_last_iter = max_iter
    if total_samples > 0:
        print(
            "all_head_pairwise_dataset resumed existing shards: "
            f"records={int(_all_head_pairwise_dataset_records)} "
            f"train_samples={int(_all_head_pairwise_dataset_samples['train'])} "
            f"val_samples={int(_all_head_pairwise_dataset_samples['val'])} "
            f"next_train_shard={int(_all_head_pairwise_dataset_shard_idx['train'])} "
            f"next_val_shard={int(_all_head_pairwise_dataset_shard_idx['val'])}",
            flush=True,
        )
        _all_head_pairwise_dataset_write_summary()


def _all_head_pairwise_dataset_torch_dtype():
    name = str(all_head_pairwise_dataset_dtype).strip().lower()
    if name in {"float16", "fp16", "half"}:
        return torch.float16
    if name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if name in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(
        f"Unsupported all_head_pairwise_dataset_dtype={all_head_pairwise_dataset_dtype!r}; "
        "expected float16, bfloat16, or float32."
    )


def _all_head_pairwise_dataset_should_collect(current_iter):
    if not bool(all_head_pairwise_dataset_enabled) or not master_process:
        return False
    _all_head_pairwise_dataset_resume_existing_if_needed()
    current_iter = int(current_iter)
    if current_iter < int(all_head_pairwise_dataset_start_iter):
        return False
    stop_iter = int(all_head_pairwise_dataset_stop_iter)
    if stop_iter >= 0 and current_iter > stop_iter:
        return False
    interval = max(1, int(all_head_pairwise_dataset_interval))
    if current_iter % interval != 0:
        return False
    max_records = int(all_head_pairwise_dataset_max_records)
    if max_records > 0 and int(_all_head_pairwise_dataset_records) >= max_records:
        return False
    return True


def _all_head_pairwise_dataset_block_orders(batch_size_local, device_local, seed):
    mode = str(all_head_pairwise_dataset_attention_order_mode).strip().lower()
    if mode == "random":
        generator_device = "cuda" if str(device_local).startswith("cuda") else "cpu"
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(int(seed))
        return sample_random_block_orders(
            batch_size=int(batch_size_local),
            num_blocks=num_blocks,
            device=device_local,
            generator=generator,
        )
    if mode in {"ar", "current_ar", "l2r_current"}:
        return torch.arange(num_blocks, device=device_local).unsqueeze(0).expand(int(batch_size_local), -1).clone()
    if mode in {"original_l2r", "original"}:
        if bool(permute_data) and inverse_block_perm is not None:
            order = inverse_block_perm.to(device=device_local, dtype=torch.long)
        else:
            order = torch.arange(num_blocks, device=device_local)
        return order.unsqueeze(0).expand(int(batch_size_local), -1).clone()
    raise ValueError(
        f"Unsupported all_head_pairwise_dataset_attention_order_mode={all_head_pairwise_dataset_attention_order_mode!r}; "
        "expected random, ar, or original_l2r."
    )


def _all_head_pairwise_rank_from_order(order):
    rank = torch.empty((int(num_blocks),), dtype=torch.uint8)
    for idx, value in enumerate([int(v) for v in order]):
        rank[int(value)] = int(idx)
    return rank


def _all_head_pairwise_pair_indices():
    return torch.triu_indices(int(num_blocks), int(num_blocks), offset=1)


def _all_head_pairwise_q_from_rank(rank):
    rank_long = rank.long()
    q = torch.where(
        rank_long[:, None] < rank_long[None, :],
        torch.ones((int(num_blocks), int(num_blocks)), dtype=torch.int8),
        -torch.ones((int(num_blocks), int(num_blocks)), dtype=torch.int8),
    )
    q.fill_diagonal_(0)
    return q


def _all_head_pairwise_loss_profile_score(loss_item, score_name):
    score_name = str(score_name).strip().lower()
    if score_name in {"prefix", "prefix_loss"}:
        return float(loss_item["prefix_loss"])
    if score_name in {"full", "full_loss"}:
        return float(loss_item["full_loss"])
    if score_name in {"linear", "linear_profile", "linear_profile_loss"}:
        return float(loss_item["linear_profile_loss"])
    if score_name in {"exp", "exp_profile", "exp_profile_loss"}:
        return float(loss_item["exp_profile_loss"])
    raise ValueError(f"Unsupported all_head_pairwise_dataset_loss_score={score_name!r}")


@torch.no_grad()
def _all_head_pairwise_dataset_evaluate_order_losses(order_list):
    unique_orders = []
    seen = set()
    for order in order_list:
        key = tuple(int(v) for v in order)
        if key in seen:
            continue
        if len(key) != int(num_blocks) or sorted(key) != list(range(int(num_blocks))):
            raise ValueError(f"Invalid pairwise dataset order={key}")
        seen.add(key)
        unique_orders.append(list(key))
    if not unique_orders:
        return {}
    order_tensor = torch.tensor(unique_orders, dtype=torch.long, device=device)
    candidate_batch_size = max(1, int(all_head_pairwise_dataset_loss_candidate_batch_size))
    probe_batch_size = max(1, int(all_head_pairwise_dataset_loss_batch_size))
    prefix_k = max(1, min(int(all_head_pairwise_dataset_loss_prefix_k), int(num_blocks)))
    stats = {
        tuple(order): {
            "full_sum": 0.0,
            "prefix_sum": 0.0,
            "profile_sum": np.zeros(int(num_blocks), dtype=np.float64),
            "count": 0,
        }
        for order in unique_orders
    }
    for _ in range(max(1, int(all_head_pairwise_dataset_loss_batches))):
        X_probe, _ = get_batch('train', batch_size_override=probe_batch_size)
        batch_size_local = int(X_probe.size(0))
        for start in range(0, len(unique_orders), candidate_batch_size):
            end = min(len(unique_orders), start + candidate_batch_size)
            chunk = order_tensor[start:end]
            chunk_size = int(chunk.size(0))
            block_orders = (
                chunk[:, None, :]
                .expand(chunk_size, batch_size_local, int(num_blocks))
                .reshape(chunk_size * batch_size_local, int(num_blocks))
            )
            X_expanded = (
                X_probe[None, :, :]
                .expand(chunk_size, batch_size_local, int(X_probe.size(1)))
                .reshape(chunk_size * batch_size_local, int(X_probe.size(1)))
            )
            with ctx:
                outputs = _forward_with_explicit_block_orders(
                    X_expanded,
                    block_orders,
                    return_token_loss=True,
                    return_logits=False,
                )
            token_losses = None
            for value in outputs[2:]:
                if torch.is_tensor(value) and value.ndim == 2:
                    token_losses = value
                    break
            if token_losses is None:
                raise RuntimeError("Could not find token losses for pairwise dataset orientation.")
            block_losses = token_losses_to_block_losses(
                token_losses.detach(),
                block_len=effective_order_block_len,
            ).float().view(chunk_size, batch_size_local, int(num_blocks))
            prefix_loss = block_losses[:, :, :prefix_k].mean(dim=2)
            full_loss = block_losses.mean(dim=2)
            for local_idx, order_idx in enumerate(range(start, end)):
                key = tuple(unique_orders[order_idx])
                stats[key]["prefix_sum"] += float(prefix_loss[local_idx].double().sum().item())
                stats[key]["full_sum"] += float(full_loss[local_idx].double().sum().item())
                stats[key]["profile_sum"] += (
                    block_losses[local_idx].double().sum(dim=0).detach().cpu().numpy()
                )
                stats[key]["count"] += int(batch_size_local)
    linear_weights = np.linspace(1.0, 0.1, int(num_blocks), dtype=np.float64)
    linear_weights /= float(max(linear_weights.sum(), 1e-12))
    exp_weights = np.exp(
        -np.arange(int(num_blocks), dtype=np.float64)
        / max(float(all_head_pairwise_dataset_loss_exp_tau), 1e-6)
    )
    exp_weights /= float(max(exp_weights.sum(), 1e-12))
    out = {}
    for key, item in stats.items():
        count = max(1, int(item["count"]))
        profile = np.asarray(item["profile_sum"], dtype=np.float64) / float(count)
        out[key] = {
            "prefix_loss": float(item["prefix_sum"] / count),
            "full_loss": float(item["full_sum"] / count),
            "linear_profile_loss": float((linear_weights * profile).sum()),
            "exp_profile_loss": float((exp_weights * profile).sum()),
            "loss_profile": [float(value) for value in profile.tolist()],
            "count": int(item["count"]),
        }
    return out


def _all_head_pairwise_dataset_metadata():
    candidate_source = str(all_head_pairwise_dataset_candidate_source).strip().lower()
    return {
        "version": 1,
        "format": f"online_oriented_all_head_{candidate_source}_pairwise_dataset",
        "description": (
            "Collected during Random training. Each sample is one selected layer/head or layer-mean "
            "attention matrix; the configured candidate source proposes raw/reverse orders, then "
            "current-model train loss selects orientation."
        ),
        "out_dir": str(out_dir),
        "dataset": str(dataset),
        "iter_start": int(all_head_pairwise_dataset_start_iter),
        "iter_stop": int(all_head_pairwise_dataset_stop_iter),
        "interval": int(all_head_pairwise_dataset_interval),
        "attention_batch_size": int(all_head_pairwise_dataset_attention_batch_size),
        "attention_micro_batches_per_record": int(all_head_pairwise_dataset_attention_micro_batches_per_record),
        "attention_samples_per_record": int(all_head_pairwise_dataset_attention_batch_size)
        * int(all_head_pairwise_dataset_attention_micro_batches_per_record),
        "export_type": str(all_head_pairwise_dataset_export_type),
        "attention_order_mode": str(all_head_pairwise_dataset_attention_order_mode),
        "dtype": str(all_head_pairwise_dataset_dtype),
        "seed": int(all_head_pairwise_dataset_seed),
        "train_records": int(all_head_pairwise_dataset_train_records),
        "resume_existing": bool(all_head_pairwise_dataset_resume_existing),
        "heads": str(all_head_pairwise_dataset_heads),
        "candidate_source": str(all_head_pairwise_dataset_candidate_source),
        "direct_asym_eig_mode": str(all_head_pairwise_dataset_direct_asym_eig_mode),
        "orientation": "train_loss_raw_reverse",
        "loss_batches": int(all_head_pairwise_dataset_loss_batches),
        "loss_batch_size": int(all_head_pairwise_dataset_loss_batch_size),
        "loss_candidate_batch_size": int(all_head_pairwise_dataset_loss_candidate_batch_size),
        "loss_prefix_k": int(all_head_pairwise_dataset_loss_prefix_k),
        "loss_score": str(all_head_pairwise_dataset_loss_score),
        "loss_exp_tau": float(all_head_pairwise_dataset_loss_exp_tau),
        "num_layers": int(model.config.n_layer),
        "num_heads": int(model.config.n_head),
        "num_blocks": int(num_blocks),
        "pairs_per_sample": int(int(num_blocks) * (int(num_blocks) - 1) // 2),
        "block_len": int(effective_order_block_len),
        "block_size": int(block_size),
        "permute_data": bool(permute_data),
        "permute_seed": int(permute_seed),
        "permute_mode": str(permute_mode),
        "train_mode": str(aogpt_train_mode),
        "no_prior_note": (
            "Orientation uses current-model train loss only. Original-frame order, tau, and validation PPL "
            "are diagnostics only and are not used for target construction."
        ),
    }


def _all_head_pairwise_dataset_write_manifest():
    dataset_dir = _all_head_pairwise_dataset_dir()
    os.makedirs(dataset_dir, exist_ok=True)
    manifest_path = os.path.join(dataset_dir, "manifest.json")
    if os.path.exists(manifest_path):
        return
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(_all_head_pairwise_dataset_metadata(), handle, indent=2, sort_keys=True)


def _all_head_pairwise_dataset_write_summary():
    if not bool(all_head_pairwise_dataset_enabled) or not master_process:
        return
    dataset_dir = _all_head_pairwise_dataset_dir()
    os.makedirs(dataset_dir, exist_ok=True)
    summary = _all_head_pairwise_dataset_metadata()
    summary.update({
        "records": int(_all_head_pairwise_dataset_records),
        "samples": {key: int(value) for key, value in _all_head_pairwise_dataset_samples.items()},
        "record_counts": {key: int(value) for key, value in _all_head_pairwise_dataset_record_counts.items()},
        "shards": {key: int(value) for key, value in _all_head_pairwise_dataset_shard_idx.items()},
        "pending": {key: int(len(value)) for key, value in _all_head_pairwise_dataset_pending.items()},
        "errors": list(_all_head_pairwise_dataset_errors),
        "last_iter": None if _all_head_pairwise_dataset_last_iter is None else int(_all_head_pairwise_dataset_last_iter),
        "wall_time": float(time.time()),
    })
    with open(os.path.join(dataset_dir, "collection_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def _all_head_pairwise_dataset_heads():
    raw_items = [item.strip() for item in str(all_head_pairwise_dataset_heads).split(',') if item.strip()]
    if not raw_items or any(item.lower() in {"all", "*"} for item in raw_items):
        return [
            (layer, head)
            for layer in range(int(model.config.n_layer))
            for head in range(int(model.config.n_head))
        ]
    pairs = []
    seen = set()
    for item in raw_items:
        if ':' not in item:
            raise ValueError(
                f"all_head_pairwise_dataset_heads item {item!r} must use 'layer:head' format."
            )
        layer_text, head_text = item.split(':', 1)
        layer = int(layer_text)
        if layer < 0 or layer >= int(model.config.n_layer):
            raise ValueError(f"all_head_pairwise_dataset layer={layer} outside 0..{int(model.config.n_layer)-1}")
        head_label = head_text.strip().lower()
        if head_label in {"mean", "avg", "average", "all", "*"}:
            head = -1
        else:
            head = int(head_text)
            if head < 0 or head >= int(model.config.n_head):
                raise ValueError(f"all_head_pairwise_dataset head={head} outside 0..{int(model.config.n_head)-1}")
        key = (layer, head)
        if key not in seen:
            seen.add(key)
            pairs.append(key)
    return pairs


def _all_head_pairwise_dataset_matrix_for_head(matrices, layer_idx, head_idx):
    layer_idx = int(layer_idx)
    head_idx = int(head_idx)
    if head_idx == -1:
        return matrices[layer_idx].mean(dim=0)
    return matrices[layer_idx, head_idx]


def _all_head_pairwise_dataset_recover_candidates(matrix):
    source = str(all_head_pairwise_dataset_candidate_source).strip().lower()
    if source in {"direct_asym_eig", "asym_eig", "direct_eig"}:
        return _head_signal_recover_direct_asym_eig_candidates(
            matrix,
            str(all_head_pairwise_dataset_direct_asym_eig_mode),
            top_m=1,
        )
    if source in {
        "pairwise_max_fiedler",
        "layer_pairwise_max_fiedler",
        "max_fiedler",
        "fiedler",
        "laplacian_fiedler",
    }:
        return _head_signal_recover_pairwise_max_fiedler_candidates(matrix, top_m=1)
    raise ValueError(
        "Unsupported all_head_pairwise_dataset_candidate_source="
        f"{all_head_pairwise_dataset_candidate_source!r}; expected direct_asym_eig or pairwise_max_fiedler."
    )


def _all_head_pairwise_dataset_save_shard(split, samples):
    dataset_dir = _all_head_pairwise_dataset_dir()
    os.makedirs(os.path.join(dataset_dir, split), exist_ok=True)
    pair_i, pair_j = _all_head_pairwise_pair_indices()
    metadata = _all_head_pairwise_dataset_metadata()
    payload = {
        "attention": torch.stack([sample["attention"] for sample in samples], dim=0),
        "teacher_order": torch.stack([sample["teacher_order"] for sample in samples], dim=0).to(torch.uint8),
        "teacher_rank": torch.stack([sample["teacher_rank"] for sample in samples], dim=0).to(torch.uint8),
        "raw_order": torch.stack([sample["raw_order"] for sample in samples], dim=0).to(torch.uint8),
        "raw_rank": torch.stack([sample["raw_rank"] for sample in samples], dim=0).to(torch.uint8),
        "reverse_order": torch.stack([sample["reverse_order"] for sample in samples], dim=0).to(torch.uint8),
        "reverse_rank": torch.stack([sample["reverse_rank"] for sample in samples], dim=0).to(torch.uint8),
        "pair_i": pair_i.to(torch.uint8),
        "pair_j": pair_j.to(torch.uint8),
        "pair_target": torch.stack([sample["pair_target"] for sample in samples], dim=0).to(torch.bool),
        "pair_sign": torch.stack([sample["pair_sign"] for sample in samples], dim=0).to(torch.int8),
        "record_index": torch.tensor([sample["record_index"] for sample in samples], dtype=torch.int64),
        "iter": torch.tensor([sample["iter"] for sample in samples], dtype=torch.int64),
        "layer": torch.tensor([sample["layer"] for sample in samples], dtype=torch.uint8),
        "head": torch.tensor([sample["head"] for sample in samples], dtype=torch.int16),
        "source_sample_count": torch.tensor([sample["source_sample_count"] for sample in samples], dtype=torch.int64),
        "source_probe_seed": torch.tensor([sample["source_probe_seed"] for sample in samples], dtype=torch.int64),
        "loss_count": torch.tensor([sample["loss_count"] for sample in samples], dtype=torch.int64),
        "raw_loss_score": torch.tensor([sample["raw_loss_score"] for sample in samples], dtype=torch.float32),
        "reverse_loss_score": torch.tensor([sample["reverse_loss_score"] for sample in samples], dtype=torch.float32),
        "selected_loss_score": torch.tensor([sample["selected_loss_score"] for sample in samples], dtype=torch.float32),
        "loss_score_gap": torch.tensor([sample["loss_score_gap"] for sample in samples], dtype=torch.float32),
        "selected_reverse": torch.tensor([sample["selected_reverse"] for sample in samples], dtype=torch.bool),
        "eigval_real": torch.tensor([sample["eigval_real"] for sample in samples], dtype=torch.float32),
        "eigval_imag": torch.tensor([sample["eigval_imag"] for sample in samples], dtype=torch.float32),
        "eigval_abs": torch.tensor([sample["eigval_abs"] for sample in samples], dtype=torch.float32),
        "vector_std": torch.tensor([sample["vector_std"] for sample in samples], dtype=torch.float32),
        "metadata": metadata,
    }
    if bool(all_head_pairwise_dataset_store_pairwise_q):
        payload["pairwise_Q"] = torch.stack([sample["pairwise_Q"] for sample in samples], dim=0).to(torch.int8)
    shard_path = os.path.join(
        dataset_dir,
        split,
        f"shard_{int(_all_head_pairwise_dataset_shard_idx[split]):03d}.pt",
    )
    torch.save(payload, shard_path)
    _all_head_pairwise_dataset_shard_idx[split] += 1
    print(
        f"all_head_pairwise_dataset wrote {shard_path} "
        f"samples={len(samples)} total_{split}={int(_all_head_pairwise_dataset_samples[split])}",
        flush=True,
    )


def _all_head_pairwise_dataset_flush(force=False):
    if not bool(all_head_pairwise_dataset_enabled) or not master_process:
        return
    shard_size = max(1, int(all_head_pairwise_dataset_shard_size))
    for split in ("train", "val"):
        while _all_head_pairwise_dataset_pending[split] and (
            bool(force) or len(_all_head_pairwise_dataset_pending[split]) >= shard_size
        ):
            if bool(force):
                count = len(_all_head_pairwise_dataset_pending[split])
            else:
                count = shard_size
            samples = _all_head_pairwise_dataset_pending[split][:count]
            _all_head_pairwise_dataset_pending[split] = _all_head_pairwise_dataset_pending[split][count:]
            _all_head_pairwise_dataset_save_shard(split, samples)
    _all_head_pairwise_dataset_write_summary()


@torch.no_grad()
def _all_head_pairwise_dataset_collect_if_due(current_iter):
    global _all_head_pairwise_dataset_records
    global _all_head_pairwise_dataset_last_iter
    if not _all_head_pairwise_dataset_should_collect(current_iter):
        return

    cpu_rng_state = torch.random.get_rng_state()
    np_rng_state = np.random.get_state()
    cuda_rng_state = None
    if device_type == 'cuda' and torch.cuda.is_available():
        cuda_rng_state = torch.cuda.get_rng_state_all()

    was_training = model.training
    model.eval()
    try:
        _all_head_pairwise_dataset_write_manifest()
        output_dtype = _all_head_pairwise_dataset_torch_dtype()
        attention_batch_size = max(1, int(all_head_pairwise_dataset_attention_batch_size))
        micro_batches = max(1, int(all_head_pairwise_dataset_attention_micro_batches_per_record))
        record_seed = int(all_head_pairwise_dataset_seed) + int(current_iter) * 1000003
        matrix_sums = None
        total_samples = 0
        for micro_idx in range(micro_batches):
            probe_seed = record_seed + int(micro_idx) * 1299827
            X_probe, _ = get_batch('train', batch_size_override=attention_batch_size)
            block_orders = _all_head_pairwise_dataset_block_orders(
                int(X_probe.size(0)),
                X_probe.device,
                probe_seed,
            )
            with ctx:
                outputs = _forward_with_explicit_block_orders(
                    X_probe,
                    block_orders,
                    return_attentions=True,
                    return_logits=False,
                )
            attentions = _attn_mlp_extract_attentions(outputs)
            if not attentions:
                raise RuntimeError("all_head_pairwise_dataset expected attention tensors, but model returned none.")
            layer_matrices = []
            for layer_attn in attentions:
                layer_matrices.append(
                    _aggregate_layerhead_attention_to_current_blocks(
                        layer_attn.detach(),
                        block_orders,
                        str(all_head_pairwise_dataset_export_type),
                    ).detach().cpu()
                )
            micro_matrices = torch.stack(layer_matrices, dim=0).to(dtype=torch.float32, device="cpu")
            if matrix_sums is None:
                matrix_sums = micro_matrices * float(X_probe.size(0))
            else:
                matrix_sums += micro_matrices * float(X_probe.size(0))
            total_samples += int(X_probe.size(0))
        if matrix_sums is None or total_samples <= 0:
            return
        matrices = matrix_sums / float(total_samples)
        for layer_idx in range(matrices.size(0)):
            for head_idx in range(matrices.size(1)):
                matrices[layer_idx, head_idx].fill_diagonal_(0.0)

        record_index = int(_all_head_pairwise_dataset_records)
        split = "train" if record_index < int(all_head_pairwise_dataset_train_records) else "val"
        order_requests = []
        rows = []
        for layer_idx, head_idx in _all_head_pairwise_dataset_heads():
            row = {
                "record_index": int(record_index),
                "iter": int(current_iter),
                "layer": int(layer_idx),
                "head": int(head_idx),
            }
            try:
                matrix = _all_head_pairwise_dataset_matrix_for_head(matrices, layer_idx, head_idx)
                candidate = _all_head_pairwise_dataset_recover_candidates(matrix)[0]
                raw_order = [int(v) for v in candidate["order"]]
                reverse_order = list(reversed(raw_order))
                row["raw_order"] = raw_order
                row["reverse_order"] = reverse_order
                row["candidate_meta"] = candidate.get("meta", {})
                order_requests.extend([raw_order, reverse_order])
            except Exception as exc:
                row["error"] = str(exc)
                _all_head_pairwise_dataset_errors.append(dict(row))
            rows.append(row)

        loss_by_order = _all_head_pairwise_dataset_evaluate_order_losses(order_requests)
        score_name = str(all_head_pairwise_dataset_loss_score).strip().lower()
        pair_i, pair_j = _all_head_pairwise_pair_indices()
        for row in rows:
            raw_order = row.get("raw_order")
            reverse_order = row.get("reverse_order")
            if not raw_order or not reverse_order:
                continue
            raw_key = tuple(int(v) for v in raw_order)
            reverse_key = tuple(int(v) for v in reverse_order)
            raw_loss = loss_by_order.get(raw_key)
            reverse_loss = loss_by_order.get(reverse_key)
            if raw_loss is None or reverse_loss is None:
                error = dict(row)
                error["error"] = "missing_loss"
                _all_head_pairwise_dataset_errors.append(error)
                continue
            raw_score = _all_head_pairwise_loss_profile_score(raw_loss, score_name)
            reverse_score = _all_head_pairwise_loss_profile_score(reverse_loss, score_name)
            if reverse_score < raw_score:
                teacher_order = reverse_order
                selected_loss = reverse_loss
                selected_score = reverse_score
                other_score = raw_score
                selected_reverse = True
            else:
                teacher_order = raw_order
                selected_loss = raw_loss
                selected_score = raw_score
                other_score = reverse_score
                selected_reverse = False
            teacher_rank = _all_head_pairwise_rank_from_order(teacher_order)
            raw_rank = _all_head_pairwise_rank_from_order(raw_order)
            reverse_rank = _all_head_pairwise_rank_from_order(reverse_order)
            pair_target = (teacher_rank.long()[pair_i.long()] < teacher_rank.long()[pair_j.long()])
            pair_sign = torch.where(
                pair_target,
                torch.ones_like(pair_target, dtype=torch.int8),
                -torch.ones_like(pair_target, dtype=torch.int8),
            )
            sample = {
                "attention": _all_head_pairwise_dataset_matrix_for_head(
                    matrices,
                    int(row["layer"]),
                    int(row["head"]),
                ).to(dtype=output_dtype),
                "teacher_order": torch.tensor(teacher_order, dtype=torch.uint8),
                "teacher_rank": teacher_rank,
                "raw_order": torch.tensor(raw_order, dtype=torch.uint8),
                "raw_rank": raw_rank,
                "reverse_order": torch.tensor(reverse_order, dtype=torch.uint8),
                "reverse_rank": reverse_rank,
                "pair_target": pair_target,
                "pair_sign": pair_sign,
                "pairwise_Q": _all_head_pairwise_q_from_rank(teacher_rank),
                "record_index": int(record_index),
                "iter": int(current_iter),
                "layer": int(row["layer"]),
                "head": int(row["head"]),
                "source_sample_count": int(total_samples),
                "source_probe_seed": int(record_seed),
                "loss_count": int(selected_loss.get("count", 0)),
                "raw_loss_score": float(raw_score),
                "reverse_loss_score": float(reverse_score),
                "selected_loss_score": float(selected_score),
                "loss_score_gap": float(max(0.0, float(other_score) - float(selected_score))),
                "selected_reverse": bool(selected_reverse),
                "eigval_real": float(
                    row.get("candidate_meta", {}).get(
                        "eigval_real",
                        row.get("candidate_meta", {}).get("laplacian_fiedler_eigval", float("nan")),
                    )
                ),
                "eigval_imag": float(row.get("candidate_meta", {}).get("eigval_imag", 0.0)),
                "eigval_abs": float(
                    row.get("candidate_meta", {}).get(
                        "eigval_abs",
                        row.get("candidate_meta", {}).get("spectral_gap", float("nan")),
                    )
                ),
                "vector_std": float(row.get("candidate_meta", {}).get("vector_std", float("nan"))),
            }
            _all_head_pairwise_dataset_pending[split].append(sample)
            _all_head_pairwise_dataset_samples[split] += 1
        _all_head_pairwise_dataset_record_counts[split] += 1
        _all_head_pairwise_dataset_records += 1
        _all_head_pairwise_dataset_last_iter = int(current_iter)
        reached_split_boundary = (
            split == "train"
            and int(_all_head_pairwise_dataset_records) == int(all_head_pairwise_dataset_train_records)
        )
        _all_head_pairwise_dataset_flush(force=bool(reached_split_boundary))
    finally:
        if was_training:
            model.train()
        torch.random.set_rng_state(cpu_rng_state)
        np.random.set_state(np_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)


def _head_direction_logger_should_update(current_iter):
    if not bool(head_direction_logger_enabled):
        return False
    interval = int(head_direction_logger_interval)
    if interval <= 0:
        return False
    if int(current_iter) < int(head_direction_logger_start_iter):
        return False
    stop_iter = int(head_direction_logger_stop_iter)
    if stop_iter >= 0 and int(current_iter) >= stop_iter:
        return False
    return int(current_iter) % interval == 0


def _head_direction_logger_dir():
    if str(head_direction_logger_out_dir).strip():
        return str(head_direction_logger_out_dir)
    return os.path.join(out_dir, "head_direction_logger")


def _head_direction_logger_to_frame(matrices):
    frame = str(head_direction_logger_frame).strip().lower()
    if frame in {"current", "model", "permuted", "current_frame"}:
        return matrices
    if frame in {"true_original", "original", "original_l2r", "physical", "physical_l2r"}:
        if permute_data and fixed_block_perm is not None:
            mapper = fixed_block_perm.to(dtype=torch.long, device=matrices.device)
            original_to_current = invert_permutation(mapper)
            return matrices.index_select(-2, original_to_current).index_select(-1, original_to_current)
        return matrices
    raise ValueError(
        f"Unsupported head_direction_logger_frame={head_direction_logger_frame!r}. "
        "Expected current or true_original."
    )


def _head_direction_matrix_metrics(matrix):
    values = matrix.detach().to(dtype=torch.float64, device="cpu")
    n = int(values.size(0))
    upper_mask = torch.triu(torch.ones((n, n), dtype=torch.bool), diagonal=1)
    lower_mask = torch.tril(torch.ones((n, n), dtype=torch.bool), diagonal=-1)
    upper_sum = float(values[upper_mask].sum().item())
    lower_sum = float(values[lower_mask].sum().item())
    denom = max(abs(upper_sum) + abs(lower_sum), 1e-12)
    upper_fraction = float(upper_sum / max(upper_sum + lower_sum, 1e-12))
    direction_score = float((upper_sum - lower_sum) / denom)
    best_offset = 0
    best_sum = -float("inf")
    best_mean = 0.0
    for offset in range(-(n - 1), n):
        if offset == 0:
            continue
        band = torch.diagonal(values, offset=offset)
        if band.numel() == 0:
            continue
        band_sum = float(band.sum().item())
        if band_sum > best_sum:
            best_offset = int(offset)
            best_sum = band_sum
            best_mean = float(band.mean().item())
    if not math.isfinite(best_sum):
        best_sum = 0.0
    return {
        "upper_sum": float(upper_sum),
        "lower_sum": float(lower_sum),
        "upper_fraction": float(upper_fraction),
        "lower_fraction": float(1.0 - upper_fraction),
        "direction_score": float(direction_score),
        "dominant_fraction": float(max(upper_fraction, 1.0 - upper_fraction)),
        "max": float(values.max().item()),
        "sum": float(values.sum().item()),
        "top_band_offset": int(best_offset),
        "top_band_sum": float(best_sum),
        "top_band_mean": float(best_mean),
    }


def _head_direction_logger_vmax(matrices_np):
    values = np.asarray(matrices_np, dtype=np.float64)
    finite = values[np.isfinite(values)]
    finite = finite[finite > 0.0]
    if finite.size == 0:
        return 1.0
    percentile = float(head_direction_logger_vmax_percentile)
    if percentile > 0.0 and percentile < 100.0:
        vmax = float(np.percentile(finite, percentile))
    else:
        vmax = float(np.max(finite))
    if not math.isfinite(vmax) or vmax <= 0.0:
        vmax = float(np.max(finite))
    return max(vmax, 1e-12)


def _head_direction_logger_rgb(matrix_np, vmax):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import cm

    values = np.asarray(matrix_np, dtype=np.float64)
    values = np.where(np.isfinite(values), values, 0.0)
    normed = np.clip(values / float(max(vmax, 1e-12)), 0.0, 1.0)
    cmap = cm.get_cmap(str(head_direction_logger_cmap))
    rgb = (cmap(normed)[..., :3] * 255.0).astype(np.uint8)
    return rgb


def _head_direction_logger_grid_figure(matrices_np, metrics, vmax):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_layers_local, num_heads_local = matrices_np.shape[:2]
    fig, axes = plt.subplots(
        num_layers_local,
        num_heads_local,
        figsize=(2.05 * num_heads_local, 1.95 * num_layers_local),
        constrained_layout=True,
    )
    if num_layers_local == 1:
        axes = np.expand_dims(axes, axis=0)
    if num_heads_local == 1:
        axes = np.expand_dims(axes, axis=1)
    last_im = None
    for layer_idx in range(num_layers_local):
        for head_idx in range(num_heads_local):
            ax = axes[layer_idx, head_idx]
            last_im = ax.imshow(
                matrices_np[layer_idx, head_idx],
                cmap=str(head_direction_logger_cmap),
                vmin=0.0,
                vmax=float(vmax),
                interpolation="nearest",
            )
            item = metrics[(int(layer_idx), int(head_idx))]
            ax.set_title(
                f"L{layer_idx}H{head_idx} u={item['upper_fraction']:.2f} "
                f"d={item['direction_score']:+.2f} b={int(item['top_band_offset']):+d}",
                fontsize=8,
            )
            ax.set_xticks([])
            ax.set_yticks([])
    if last_im is not None:
        fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.72, pad=0.01)
    fig.suptitle(
        "Head direction logger | "
        f"iter={int(iter_num)} | {str(head_direction_logger_export_type)} / "
        f"{str(head_direction_logger_frame)}",
        fontsize=14,
        fontweight="bold",
    )
    return fig


def _head_direction_logger_write_latest(matrices_np, metrics, total_samples, vmax, grid_figure=None):
    if not bool(head_direction_logger_save_latest) or not master_process:
        return
    stats_dir = _head_direction_logger_dir()
    os.makedirs(stats_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(stats_dir, "latest_head_direction_matrices.npz"),
        matrices=np.asarray(matrices_np, dtype=np.float32),
    )
    rows = []
    for (layer_idx, head_idx), item in sorted(metrics.items()):
        rows.append(
            {
                "iter": int(iter_num),
                "layer": int(layer_idx),
                "head": int(head_idx),
                **{
                    key: (int(value) if key == "top_band_offset" else float(value))
                    for key, value in item.items()
                },
            }
        )
    payload = {
        "iter": int(iter_num),
        "wall_time": float(time.time()),
        "num_samples": int(total_samples),
        "num_layers": int(matrices_np.shape[0]),
        "num_heads": int(matrices_np.shape[1]),
        "num_blocks": int(matrices_np.shape[2]),
        "export_type": str(head_direction_logger_export_type),
        "frame": str(head_direction_logger_frame),
        "vmax": float(vmax),
        "rows": rows,
    }
    with open(os.path.join(stats_dir, "latest_head_direction_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    if grid_figure is not None:
        grid_figure.savefig(
            os.path.join(stats_dir, "latest_head_direction_grid.png"),
            dpi=180,
            bbox_inches="tight",
        )


@torch.no_grad()
def _head_direction_logger_update_if_due():
    if not _head_direction_logger_should_update(iter_num):
        return
    if not bool(wandb_log):
        if master_process:
            print("head_direction_logger enabled but wandb_log=False; skipping W&B-only logger")
        return

    rng_state_cpu = None
    rng_state_cuda = None
    rng_state_np = None
    if bool(head_direction_logger_restore_rng):
        rng_state_cpu = torch.random.get_rng_state()
        rng_state_np = np.random.get_state()
        if device_type == 'cuda' and torch.cuda.is_available():
            rng_state_cuda = torch.cuda.get_rng_state_all()
    if bool(head_direction_logger_deterministic):
        rank_offset = int(ddp_rank if ddp else 0)
        seed = int(head_direction_logger_seed) + int(iter_num) * 1009 + rank_offset
        torch.manual_seed(seed)
        if device_type == 'cuda' and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed % (2**32 - 1))

    was_training = model.training
    model.eval()
    grid_figure = None
    try:
        matrix_sums = None
        total_samples = 0
        probe_batch_size = max(1, int(head_direction_logger_batch_size))
        probe_batches = max(1, int(head_direction_logger_batches))
        for _ in range(probe_batches):
            X_probe, _ = get_batch('train', batch_size_override=probe_batch_size)
            block_orders = sample_random_block_orders(
                batch_size=int(X_probe.size(0)),
                num_blocks=num_blocks,
                device=X_probe.device,
            )
            with ctx:
                outputs = _forward_with_explicit_block_orders(
                    X_probe,
                    block_orders,
                    return_attentions=True,
                    return_logits=False,
                )
            attentions = _attn_mlp_extract_attentions(outputs)
            if not attentions:
                continue
            layer_matrices = []
            for layer_attn in attentions:
                layer_matrices.append(
                    _aggregate_layerhead_attention_to_current_blocks(
                        layer_attn.detach(),
                        block_orders,
                        str(head_direction_logger_export_type),
                    ).detach().cpu()
                )
            matrices = torch.stack(layer_matrices, dim=0).to(dtype=torch.float64, device="cpu")
            matrices = _head_direction_logger_to_frame(matrices)
            samples = int(X_probe.size(0))
            matrix_sums = matrices * float(samples) if matrix_sums is None else matrix_sums + matrices * float(samples)
            total_samples += int(samples)
        if matrix_sums is None or total_samples <= 0:
            return

        if ddp:
            reduce_device = torch.device(device) if device_type == 'cuda' else torch.device('cpu')
            matrix_reduce = matrix_sums.to(device=reduce_device)
            count_reduce = torch.tensor([float(total_samples)], dtype=torch.float64, device=reduce_device)
            dist.all_reduce(matrix_reduce, op=dist.ReduceOp.SUM)
            dist.all_reduce(count_reduce, op=dist.ReduceOp.SUM)
            matrix_sums = matrix_reduce.detach().cpu()
            total_samples = int(round(float(count_reduce.detach().cpu().item())))

        matrices = matrix_sums / float(max(1, total_samples))
        metrics = {}
        for layer_idx in range(int(matrices.size(0))):
            for head_idx in range(int(matrices.size(1))):
                metrics[(int(layer_idx), int(head_idx))] = _head_direction_matrix_metrics(
                    matrices[layer_idx, head_idx]
                )
        if not master_process:
            return

        wandb_module = globals().get("wandb", None)
        if wandb_module is None:
            return

        matrices_np = matrices.detach().cpu().numpy().astype(np.float32, copy=False)
        vmax = _head_direction_logger_vmax(matrices_np)
        prefix = str(head_direction_logger_wandb_prefix).strip() or "head_direction"
        payload = {
            "iter": int(iter_num),
            f"{prefix}/num_samples": int(total_samples),
            f"{prefix}/num_layers": int(matrices_np.shape[0]),
            f"{prefix}/num_heads": int(matrices_np.shape[1]),
            f"{prefix}/num_blocks": int(matrices_np.shape[2]),
            f"{prefix}/vmax": float(vmax),
        }
        for (layer_idx, head_idx), item in sorted(metrics.items()):
            head_prefix = f"{prefix}/L{int(layer_idx)}H{int(head_idx)}"
            payload[f"{head_prefix}/upper_fraction"] = float(item["upper_fraction"])
            payload[f"{head_prefix}/direction_score"] = float(item["direction_score"])
            payload[f"{head_prefix}/dominant_fraction"] = float(item["dominant_fraction"])
            payload[f"{head_prefix}/top_band_offset"] = int(item["top_band_offset"])
            payload[f"{head_prefix}/top_band_sum"] = float(item["top_band_sum"])
            payload[f"{head_prefix}/max"] = float(item["max"])
        if bool(head_direction_logger_log_grid):
            grid_figure = _head_direction_logger_grid_figure(matrices_np, metrics, vmax)
            payload[f"{prefix}/attnmap_grid"] = wandb_module.Image(
                grid_figure,
                caption=(
                    f"iter={int(iter_num)} {str(head_direction_logger_export_type)} "
                    f"{str(head_direction_logger_frame)}"
                ),
            )
        individual_interval = int(head_direction_logger_individual_interval)
        if individual_interval <= 0:
            individual_interval = max(1, int(head_direction_logger_interval))
        log_individual = bool(head_direction_logger_log_individual_maps) and int(iter_num) % individual_interval == 0
        if log_individual:
            for layer_idx in range(int(matrices_np.shape[0])):
                for head_idx in range(int(matrices_np.shape[1])):
                    item = metrics[(int(layer_idx), int(head_idx))]
                    rgb = _head_direction_logger_rgb(matrices_np[layer_idx, head_idx], vmax)
                    payload[f"{prefix}/attnmap_L{layer_idx}H{head_idx}"] = wandb_module.Image(
                        rgb,
                        caption=(
                            f"iter={int(iter_num)} L{layer_idx}H{head_idx} "
                            f"upper={item['upper_fraction']:.3f} "
                            f"dir={item['direction_score']:+.3f} "
                            f"band={int(item['top_band_offset']):+d}"
                        ),
                    )
        _head_direction_logger_write_latest(
            matrices_np,
            metrics,
            total_samples,
            vmax,
            grid_figure=grid_figure,
        )
        wandb_module.log(payload)
        print(
            "head_direction_logger iter "
            f"{int(iter_num)} samples={int(total_samples)} "
            f"frame={str(head_direction_logger_frame)} export={str(head_direction_logger_export_type)}"
        )
    finally:
        if grid_figure is not None:
            import matplotlib.pyplot as plt
            plt.close(grid_figure)
        if rng_state_cpu is not None:
            torch.random.set_rng_state(rng_state_cpu)
        if rng_state_cuda is not None:
            torch.cuda.set_rng_state_all(rng_state_cuda)
        if rng_state_np is not None:
            np.random.set_state(rng_state_np)
        if was_training:
            model.train()


def _head_asym_selector_should_update(current_iter):
    if not bool(head_asym_selector_enabled):
        return False
    return int(current_iter) == int(head_asym_selector_target_iter)


def _head_asym_selector_dir():
    if str(head_asym_selector_out_dir).strip():
        return str(head_asym_selector_out_dir)
    return os.path.join(out_dir, "head_asym_selector")


def _head_asym_selector_matrix_metrics(matrix):
    values = matrix.detach().to(dtype=torch.float64, device="cpu")
    n = int(values.size(0))
    eye = torch.eye(n, dtype=torch.bool)
    offdiag = ~eye
    values = values.clone()
    values[eye] = 0.0
    antisym = values - values.t()
    asym_sum = float(antisym[offdiag].abs().sum().item())
    attn_sum = float(values[offdiag].abs().sum().item())
    pair_count = int(offdiag.sum().item())
    return {
        "asym_score": float(asym_sum / max(attn_sum, 1e-12)),
        "mean_abs_asym": float(asym_sum / float(max(1, pair_count))),
        "sum_abs_asym": float(asym_sum),
        "sum_abs_attn": float(attn_sum),
        "max_attn": float(values.max().item()),
    }


def _head_asym_selector_frame_matrices(matrices, frame_name):
    frame = str(frame_name).strip().lower()
    if frame in {"current", "current_l2r", "current_frame", "permuted"}:
        return matrices
    if frame in {"original", "original_l2r", "true_original"}:
        if permute_data and fixed_block_perm is not None:
            mapper = fixed_block_perm.to(dtype=torch.long, device=matrices.device)
            original_to_current = invert_permutation(mapper)
            return matrices.index_select(-2, original_to_current).index_select(-1, original_to_current)
        return matrices
    raise ValueError(f"Unsupported head_asym_selector frame={frame_name!r}.")


def _head_asym_selector_grid_figure(matrices, selected, frame_name, vmax):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mats_np = matrices.detach().cpu().numpy().astype(np.float32, copy=False)
    num_layers, num_heads = int(mats_np.shape[0]), int(mats_np.shape[1])
    fig, axes = plt.subplots(
        num_layers,
        num_heads,
        figsize=(2.0 * num_heads, 2.05 * num_layers),
        constrained_layout=True,
    )
    cmap = str(head_asym_selector_all_head_maps_cmap)
    selected_layer = int(selected.get("layer", -1))
    selected_head = int(selected.get("head", -1))
    image = None
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            ax = axes[layer_idx, head_idx] if num_layers > 1 else axes[head_idx]
            matrix = mats_np[layer_idx, head_idx].copy()
            if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
                np.fill_diagonal(matrix, 0.0)
            image = ax.imshow(
                matrix,
                cmap=cmap,
                vmin=0.0,
                vmax=float(vmax),
                interpolation="nearest",
            )
            title = f"L{layer_idx}H{head_idx}"
            if layer_idx == selected_layer and head_idx == selected_head:
                title += " selected"
                for spine in ax.spines.values():
                    spine.set_edgecolor("red")
                    spine.set_linewidth(2.0)
            ax.set_title(title, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(
        f"iter {int(iter_num)} all heads {frame_name} | {str(head_asym_selector_export_type)}",
        fontsize=12,
    )
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.018, pad=0.01)
    return fig


def _head_asym_selector_write_all_head_maps(matrices, rows, selected):
    if not bool(head_asym_selector_save_all_head_maps) or not master_process:
        return
    if str(head_asym_selector_all_head_maps_dir).strip():
        maps_dir = str(head_asym_selector_all_head_maps_dir)
    else:
        maps_dir = os.path.join(_head_asym_selector_dir(), f"all_head_attn_iter{int(iter_num)}")
    os.makedirs(maps_dir, exist_ok=True)

    frames = [
        ("current_l2r", _head_asym_selector_frame_matrices(matrices, "current_l2r")),
        ("original_l2r", _head_asym_selector_frame_matrices(matrices, "original_l2r")),
    ]
    percentile = float(head_asym_selector_all_head_maps_vmax_percentile)
    saved_figures = {}
    for frame_name, frame_matrices in frames:
        values = frame_matrices.detach().float().cpu().numpy()
        if values.size == 0:
            vmax = 1e-8
        else:
            vmax = float(np.percentile(values, max(0.0, min(100.0, percentile))))
            vmax = max(vmax, 1e-8)
        figure = _head_asym_selector_grid_figure(frame_matrices, selected, frame_name, vmax)
        path = os.path.join(
            maps_dir,
            f"head_asym_all_heads_{frame_name}_{str(head_asym_selector_export_type)}_iter{int(iter_num)}.png",
        )
        try:
            figure.savefig(path, dpi=180)
            saved_figures[frame_name] = path
        finally:
            import matplotlib.pyplot as plt
            plt.close(figure)

    metrics_path = os.path.join(maps_dir, "head_asym_all_heads_current_original_metrics.csv")
    fieldnames = [
        "frame",
        "iter",
        "layer",
        "head",
        "selected",
        "asym_score",
        "mean_abs_asym",
        "sum_abs_asym",
        "sum_abs_attn",
        "max_attn",
    ]
    with open(metrics_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for frame_name, frame_matrices in frames:
            for layer_idx in range(int(frame_matrices.size(0))):
                for head_idx in range(int(frame_matrices.size(1))):
                    item = _head_asym_selector_matrix_metrics(frame_matrices[layer_idx, head_idx])
                    writer.writerow(
                        {
                            "frame": frame_name,
                            "iter": int(iter_num),
                            "layer": int(layer_idx),
                            "head": int(head_idx),
                            "selected": int(
                                int(layer_idx) == int(selected.get("layer", -1))
                                and int(head_idx) == int(selected.get("head", -1))
                            ),
                            **{key: float(value) for key, value in item.items()},
                        }
                    )
    summary_path = os.path.join(maps_dir, "head_asym_all_heads_current_original_summary.json")
    payload = {
        "iter": int(iter_num),
        "export_type": str(head_asym_selector_export_type),
        "selected": selected,
        "figures": saved_figures,
        "metrics_csv": metrics_path,
        "permute_seed": int(permute_seed),
        "note": (
            "current_l2r is the permuted/current block coordinate used by the model. "
            "original_l2r is a post-hoc diagnostic remapping with the fixed data permutation; "
            "it is not used for head selection or training."
        ),
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _head_asym_selector_write(rows, selected, matrices, total_samples):
    if not master_process:
        return
    stats_dir = _head_asym_selector_dir()
    os.makedirs(stats_dir, exist_ok=True)
    csv_path = os.path.join(stats_dir, "head_asym_scores.csv")
    fieldnames = [
        "iter",
        "layer",
        "head",
        "asym_score",
        "mean_abs_asym",
        "sum_abs_asym",
        "sum_abs_attn",
        "max_attn",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})
    if bool(head_asym_selector_save_matrices):
        np.savez_compressed(
            os.path.join(stats_dir, "head_asym_matrices.npz"),
            matrices=matrices.detach().cpu().numpy().astype(np.float32, copy=False),
        )
    _head_asym_selector_write_all_head_maps(matrices, rows, selected)
    payload = {
        "iter": int(iter_num),
        "wall_time": float(time.time()),
        "method": str(selected.get("selection_rule", head_asym_selector_selection_rule)),
        "formula": "sum_{i!=j} |A_ij - A_ji| / sum_{i!=j} |A_ij|",
        "selection_rule": str(selected.get("selection_rule", head_asym_selector_selection_rule)),
        "selection_detail": str(
            selected.get(
                "selection_detail",
                (
                    "argmax(asym_score), tie-broken by lower layer then lower head"
                    if str(selected.get("selection_rule", head_asym_selector_selection_rule)) == "asym_argmax"
                    else (
                        "rank heads by current-frame asym_score, keep asym_top_k, recover one "
                        "direct_asym_eig order per head, orient raw/reverse by current-model "
                        "loss profile, then choose the lowest-loss candidate"
                    )
                ),
            )
        ),
        "no_prior_note": str(
            selected.get(
                "selection_prior_note",
                (
                    "Uses current-frame attention only. Original-frame order, upper/lower "
                    "triangle labels, tau, validation loss, and oracle orders are not used."
                ),
            )
        ),
        "split": str(head_asym_selector_split),
        "num_samples": int(total_samples),
        "num_layers": int(matrices.size(0)),
        "num_heads": int(matrices.size(1)),
        "num_blocks": int(matrices.size(2)),
        "export_type": str(head_asym_selector_export_type),
        "selected": selected,
        "top_heads": rows[: min(10, len(rows))],
        "rows": rows,
    }
    with open(os.path.join(stats_dir, "head_asym_selection.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(
        "head_asym_selector iter "
        f"{int(iter_num)} selected=L{int(selected['layer'])}H{int(selected['head'])} "
        f"asym_score={float(selected['asym_score']):.6f} "
        f"rule={str(selected.get('selection_rule', head_asym_selector_selection_rule))} "
        f"export={str(head_asym_selector_export_type)} samples={int(total_samples)}"
    )


def _head_asym_selector_loss_profile_score(loss_item, score_name):
    score_name = str(score_name).strip().lower()
    if score_name in {"prefix", "prefix_loss"}:
        return float(loss_item["prefix_loss"])
    if score_name in {"full", "full_loss"}:
        return float(loss_item["full_loss"])
    if score_name in {"linear", "linear_profile", "linear_profile_loss"}:
        return float(loss_item["linear_profile_loss"])
    if score_name in {"exp", "exp_profile", "exp_profile_loss"}:
        return float(loss_item["exp_profile_loss"])
    raise ValueError(f"Unsupported head_asym_selector_loss_score={score_name!r}")


@torch.no_grad()
def _head_asym_selector_evaluate_order_losses(order_list):
    unique_orders = []
    seen = set()
    for order in order_list:
        key = tuple(int(v) for v in order)
        if key not in seen:
            seen.add(key)
            unique_orders.append(list(key))
    if not unique_orders:
        return {}
    order_tensor = torch.tensor(unique_orders, dtype=torch.long, device=device)
    candidate_batch_size = max(1, int(head_asym_selector_loss_candidate_batch_size))
    probe_batch_size = max(1, int(head_asym_selector_loss_batch_size))
    prefix_k = max(1, min(int(head_asym_selector_loss_prefix_k), int(num_blocks)))
    stats = {
        tuple(order): {
            "full_sum": 0.0,
            "prefix_sum": 0.0,
            "profile_sum": np.zeros(int(num_blocks), dtype=np.float64),
            "count": 0,
        }
        for order in unique_orders
    }
    for _ in range(max(1, int(head_asym_selector_loss_batches))):
        X_probe, _ = get_batch('train', batch_size_override=probe_batch_size)
        batch_size_local = int(X_probe.size(0))
        for start in range(0, len(unique_orders), candidate_batch_size):
            end = min(len(unique_orders), start + candidate_batch_size)
            chunk = order_tensor[start:end]
            chunk_size = int(chunk.size(0))
            block_orders = (
                chunk[:, None, :]
                .expand(chunk_size, batch_size_local, int(num_blocks))
                .reshape(chunk_size * batch_size_local, int(num_blocks))
            )
            X_expanded = (
                X_probe[None, :, :]
                .expand(chunk_size, batch_size_local, int(X_probe.size(1)))
                .reshape(chunk_size * batch_size_local, int(X_probe.size(1)))
            )
            with ctx:
                outputs = _forward_with_explicit_block_orders(
                    X_expanded,
                    block_orders,
                    return_token_loss=True,
                    return_logits=False,
                )
            token_losses = outputs[2]
            block_losses = token_losses_to_block_losses(
                token_losses.detach(),
                block_len=effective_order_block_len,
            ).float().view(chunk_size, batch_size_local, int(num_blocks))
            prefix_loss = block_losses[:, :, :prefix_k].mean(dim=2)
            full_loss = block_losses.mean(dim=2)
            for local_idx, order_idx in enumerate(range(start, end)):
                key = tuple(unique_orders[order_idx])
                stats[key]["prefix_sum"] += float(prefix_loss[local_idx].double().sum().item())
                stats[key]["full_sum"] += float(full_loss[local_idx].double().sum().item())
                stats[key]["profile_sum"] += (
                    block_losses[local_idx].double().sum(dim=0).detach().cpu().numpy()
                )
                stats[key]["count"] += int(batch_size_local)
    out = {}
    linear_weights = np.linspace(1.0, 0.1, int(num_blocks), dtype=np.float64)
    linear_weights /= float(max(linear_weights.sum(), 1e-12))
    exp_weights = np.exp(
        -np.arange(int(num_blocks), dtype=np.float64)
        / max(float(head_asym_selector_loss_exp_tau), 1e-6)
    )
    exp_weights /= float(max(exp_weights.sum(), 1e-12))
    for key, item in stats.items():
        count = max(1, int(item["count"]))
        profile = np.asarray(item["profile_sum"], dtype=np.float64) / float(count)
        out[key] = {
            "prefix_loss": float(item["prefix_sum"] / count),
            "full_loss": float(item["full_sum"] / count),
            "linear_profile_loss": float((linear_weights * profile).sum()),
            "exp_profile_loss": float((exp_weights * profile).sum()),
            "loss_profile": [float(value) for value in profile.tolist()],
            "count": int(item["count"]),
        }
    return out


def _head_asym_selector_select(rows, matrices):
    rule = str(head_asym_selector_selection_rule).strip().lower()
    if rule in {"", "asym", "asym_argmax", "argmax", "current_attention_asymmetry"}:
        for rank, row in enumerate(rows, start=1):
            row["asym_rank"] = int(rank)
            row["selection_rule"] = "asym_argmax"
        selected = dict(rows[0]) if rows else None
        return rows, selected
    if rule in {
        "oracle_direct_asym_eig_original_tau",
        "direct_asym_eig_original_tau",
        "direct_asym_eig_tau_oracle",
        "oracle_tau_direct_asym_eig",
    }:
        valid = []
        invalid = []
        for rank, row in enumerate(rows, start=1):
            row["asym_rank"] = int(rank)
            row["selection_rule"] = "oracle_direct_asym_eig_original_tau"
            layer_idx = int(row["layer"])
            head_idx = int(row["head"])
            try:
                candidates = _head_signal_recover_direct_asym_eig_candidates(
                    matrices[layer_idx, head_idx],
                    str(head_asym_selector_direct_asym_eig_mode),
                    top_m=1,
                )
                top = candidates[0]
                raw_order = [int(value) for value in top["order"]]
                reverse_order = list(reversed(raw_order))
                raw_original = _head_signal_order_to_original(raw_order)
                reverse_original = _head_signal_order_to_original(reverse_order)
                raw_tau = _head_signal_tau_to_l2r(raw_original)
                reverse_tau = _head_signal_tau_to_l2r(reverse_original)
                if float(reverse_tau) > float(raw_tau):
                    selected_order = reverse_order
                    selected_original = reverse_original
                    selected_tau = reverse_tau
                    selected_reverse = True
                    other_tau = raw_tau
                else:
                    selected_order = raw_order
                    selected_original = raw_original
                    selected_tau = raw_tau
                    selected_reverse = False
                    other_tau = reverse_tau
                row.update(
                    {
                        "selector_candidate_source": "direct_asym_eig",
                        "selector_candidate_name": str(top.get("name", "")),
                        "selector_raw_score": float(top.get("score", float("nan"))),
                        "selector_raw_order_current": raw_order,
                        "selector_raw_order_original_diagnostic": raw_original,
                        "selector_raw_tau_diagnostic": float(raw_tau),
                        "selector_raw_abs_tau_diagnostic": abs(float(raw_tau)),
                        "selector_reverse_order_current": reverse_order,
                        "selector_reverse_order_original_diagnostic": reverse_original,
                        "selector_reverse_tau_diagnostic": float(reverse_tau),
                        "selector_reverse_abs_tau_diagnostic": abs(float(reverse_tau)),
                        "selector_selected_reverse": bool(selected_reverse),
                        "selector_selected_order_current": selected_order,
                        "selector_selected_order_original_diagnostic": selected_original,
                        "selector_selected_tau_diagnostic": float(selected_tau),
                        "selector_selected_abs_tau_diagnostic": abs(float(selected_tau)),
                        "selector_selected_sign_diagnostic": int(_head_signal_sign(selected_tau)),
                        "selector_tau_gap": float(float(selected_tau) - float(other_tau)),
                        "selector_uses_original_tau_oracle": True,
                    }
                )
                meta = top.get("meta", {})
                if isinstance(meta, dict):
                    for key in (
                        "mode",
                        "input",
                        "vector_side",
                        "eigen_selector",
                        "vector_part",
                    ):
                        if key in meta:
                            row[f"selector_eig_{key}"] = str(meta[key])
                    for key in (
                        "eigval_real",
                        "eigval_imag",
                        "eigval_abs",
                        "vector_min",
                        "vector_max",
                        "vector_std",
                    ):
                        if key in meta:
                            row[f"selector_eig_{key}"] = float(meta.get(key, float("nan")))
                valid.append(row)
            except Exception as exc:
                row["selector_error"] = str(exc)
                invalid.append(row)
        if not valid:
            raise RuntimeError("head_asym_selector oracle direct_asym_eig original-tau selection produced no valid candidates.")
        valid.sort(
            key=lambda row: (
                -float(row["selector_selected_tau_diagnostic"]),
                -float(row["asym_score"]),
                int(row["layer"]),
                int(row["head"]),
            )
        )
        for rank, row in enumerate(valid, start=1):
            row["selector_oracle_tau_rank"] = int(rank)
        selected = dict(valid[0])
        selected["selection_rule"] = "oracle_direct_asym_eig_original_tau"
        selected["selection_top_k"] = int(len(valid))
        selected["selection_detail"] = (
            "ORACLE ABLATION: for every head, recover a direct_asym_eig order, "
            "compare raw/reverse by original-frame Kendall tau, and choose the "
            "head whose best orientation has the highest original tau."
        )
        selected["selection_prior_note"] = (
            "ORACLE/PRIOR HEAD-SELECTION ONLY: this selection rule uses original-frame "
            "Kendall tau to choose the head. After the head is assigned, Attn-MLP "
            "training still consumes current-frame attention with the configured "
            "directed-ribbon loss; original tau/order is not fed into later MLP updates."
        )
        return valid + invalid, selected
    if rule not in {
        "asym_topk_direct_asym_eig_loss",
        "direct_asym_eig_loss",
        "asym_topk_eig_loss",
    }:
        raise ValueError(f"Unsupported head_asym_selector_selection_rule={head_asym_selector_selection_rule!r}")

    for rank, row in enumerate(rows, start=1):
        row["asym_rank"] = int(rank)
        row["selection_rule"] = "asym_topk_direct_asym_eig_loss"
    top_k = max(1, min(int(head_asym_selector_asym_top_k), len(rows)))
    candidate_rows = rows[:top_k]
    order_loss_requests = []
    for row in candidate_rows:
        layer_idx = int(row["layer"])
        head_idx = int(row["head"])
        try:
            candidates = _head_signal_recover_direct_asym_eig_candidates(
                matrices[layer_idx, head_idx],
                str(head_asym_selector_direct_asym_eig_mode),
                top_m=1,
            )
            top = candidates[0]
            raw_order = [int(value) for value in top["order"]]
            reverse_order = list(reversed(raw_order))
            raw_original = _head_signal_order_to_original(raw_order)
            raw_tau = _head_signal_tau_to_l2r(raw_original)
            row.update(
                {
                    "selector_candidate_source": "direct_asym_eig",
                    "selector_candidate_name": str(top.get("name", "")),
                    "selector_raw_score": float(top.get("score", float("nan"))),
                    "selector_raw_order_current": raw_order,
                    "selector_raw_order_original_diagnostic": raw_original,
                    "selector_raw_tau_diagnostic": float(raw_tau),
                    "selector_raw_abs_tau_diagnostic": abs(float(raw_tau)),
                }
            )
            meta = top.get("meta", {})
            if isinstance(meta, dict):
                for key in (
                    "mode",
                    "input",
                    "vector_side",
                    "eigen_selector",
                    "vector_part",
                ):
                    if key in meta:
                        row[f"selector_eig_{key}"] = str(meta[key])
                for key in (
                    "eigval_real",
                    "eigval_imag",
                    "eigval_abs",
                    "vector_min",
                    "vector_max",
                    "vector_std",
                ):
                    if key in meta:
                        row[f"selector_eig_{key}"] = float(meta.get(key, float("nan")))
            order_loss_requests.extend([raw_order, reverse_order])
        except Exception as exc:
            row["selector_error"] = str(exc)

    loss_by_order = _head_asym_selector_evaluate_order_losses(order_loss_requests)
    score_name = str(head_asym_selector_loss_score).strip().lower()
    valid = []
    for row in candidate_rows:
        raw_order = row.get("selector_raw_order_current")
        if not raw_order:
            continue
        raw_key = tuple(int(value) for value in raw_order)
        reverse_order = list(reversed(raw_order))
        reverse_key = tuple(int(value) for value in reverse_order)
        raw_loss = loss_by_order.get(raw_key)
        reverse_loss = loss_by_order.get(reverse_key)
        if raw_loss is None or reverse_loss is None:
            row["selector_error"] = "missing_loss"
            continue
        raw_score = _head_asym_selector_loss_profile_score(raw_loss, score_name)
        reverse_score = _head_asym_selector_loss_profile_score(reverse_loss, score_name)
        if reverse_score < raw_score:
            selected_order = reverse_order
            selected_loss = reverse_loss
            selected_score = reverse_score
            other_score = raw_score
            selected_reverse = True
        else:
            selected_order = [int(value) for value in raw_order]
            selected_loss = raw_loss
            selected_score = raw_score
            other_score = reverse_score
            selected_reverse = False
        selected_original = _head_signal_order_to_original(selected_order)
        selected_tau = _head_signal_tau_to_l2r(selected_original)
        row.update(
            {
                "selector_loss_score_name": str(score_name),
                "selector_raw_loss_score": float(raw_score),
                "selector_reverse_loss_score": float(reverse_score),
                "selector_selected_score": float(selected_score),
                "selector_score_gap": float(max(0.0, float(other_score) - float(selected_score))),
                "selector_selected_reverse": bool(selected_reverse),
                "selector_selected_prefix_loss": float(selected_loss.get("prefix_loss", float("nan"))),
                "selector_selected_full_loss": float(selected_loss.get("full_loss", float("nan"))),
                "selector_selected_linear_profile_loss": float(selected_loss.get("linear_profile_loss", float("nan"))),
                "selector_selected_exp_profile_loss": float(selected_loss.get("exp_profile_loss", float("nan"))),
                "selector_selected_order_current": selected_order,
                "selector_selected_order_original_diagnostic": selected_original,
                "selector_selected_tau_diagnostic": float(selected_tau),
                "selector_selected_abs_tau_diagnostic": abs(float(selected_tau)),
            }
        )
        valid.append(row)

    if not valid:
        raise RuntimeError("head_asym_selector direct_asym_eig loss selection produced no valid candidates.")
    valid.sort(
        key=lambda row: (
            float(row["selector_selected_score"]),
            -float(row["asym_score"]),
            int(row["layer"]),
            int(row["head"]),
        )
    )
    for rank, row in enumerate(valid, start=1):
        row["selector_loss_rank_within_asym_top_k"] = int(rank)
    selected = dict(valid[0])
    selected["selection_rule"] = "asym_topk_direct_asym_eig_loss"
    selected["selection_top_k"] = int(top_k)
    selected["selection_loss_score_name"] = str(score_name)
    selected["selection_no_prior_note"] = (
        "Head selection uses current-frame attention asymmetry plus current-model "
        "direct_asym_eig raw/reverse loss only. Original-frame tau/order fields are diagnostic-only."
    )
    return rows, selected


@torch.no_grad()
def _head_asym_selector_update_if_due():
    global attn_mlp_policy_layer, attn_mlp_policy_head, attn_mlp_policy_export_type
    if not _head_asym_selector_should_update(iter_num):
        return

    rng_state_cpu = None
    rng_state_cuda = None
    rng_state_np = None
    if bool(head_asym_selector_restore_rng):
        rng_state_cpu = torch.random.get_rng_state()
        rng_state_np = np.random.get_state()
        if device_type == 'cuda' and torch.cuda.is_available():
            rng_state_cuda = torch.cuda.get_rng_state_all()
    if bool(head_asym_selector_deterministic):
        rank_offset = int(ddp_rank if ddp else 0)
        seed = int(head_asym_selector_seed) + int(iter_num) * 1009 + rank_offset
        torch.manual_seed(seed)
        if device_type == 'cuda' and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed % (2**32 - 1))

    was_training = model.training
    model.eval()
    try:
        matrix_sums = None
        total_samples = 0
        probe_batch_size = max(1, int(head_asym_selector_batch_size))
        probe_batches = max(1, int(head_asym_selector_batches))
        probe_split = str(head_asym_selector_split)
        if probe_split not in {"train", "val"}:
            raise ValueError(f"Unsupported head_asym_selector_split={head_asym_selector_split!r}.")
        for _ in range(probe_batches):
            X_probe, _ = get_batch(probe_split, batch_size_override=probe_batch_size)
            block_orders = sample_random_block_orders(
                batch_size=int(X_probe.size(0)),
                num_blocks=num_blocks,
                device=X_probe.device,
            )
            with ctx:
                outputs = _forward_with_explicit_block_orders(
                    X_probe,
                    block_orders,
                    return_attentions=True,
                    return_logits=False,
                )
            attentions = _attn_mlp_extract_attentions(outputs)
            if not attentions:
                continue
            layer_matrices = []
            for layer_attn in attentions:
                layer_matrices.append(
                    _aggregate_layerhead_attention_to_current_blocks(
                        layer_attn.detach(),
                        block_orders,
                        str(head_asym_selector_export_type),
                    ).detach().cpu()
                )
            matrices = torch.stack(layer_matrices, dim=0).to(dtype=torch.float64, device="cpu")
            samples = int(X_probe.size(0))
            matrix_sums = matrices * float(samples) if matrix_sums is None else matrix_sums + matrices * float(samples)
            total_samples += int(samples)
        if matrix_sums is None or total_samples <= 0:
            return

        if ddp:
            reduce_device = torch.device(device) if device_type == 'cuda' else torch.device('cpu')
            matrix_reduce = matrix_sums.to(device=reduce_device)
            count_reduce = torch.tensor([float(total_samples)], dtype=torch.float64, device=reduce_device)
            dist.all_reduce(matrix_reduce, op=dist.ReduceOp.SUM)
            dist.all_reduce(count_reduce, op=dist.ReduceOp.SUM)
            matrix_sums = matrix_reduce.detach().cpu()
            total_samples = int(round(float(count_reduce.detach().cpu().item())))

        matrices = matrix_sums / float(max(1, total_samples))
        rows = []
        for layer_idx in range(int(matrices.size(0))):
            for head_idx in range(int(matrices.size(1))):
                item = _head_asym_selector_matrix_metrics(matrices[layer_idx, head_idx])
                rows.append(
                    {
                        "iter": int(iter_num),
                        "layer": int(layer_idx),
                        "head": int(head_idx),
                        **{key: float(value) for key, value in item.items()},
                    }
                )
        rows.sort(key=lambda row: (-float(row["asym_score"]), int(row["layer"]), int(row["head"])))
        rows, selected = _head_asym_selector_select(rows, matrices)
        if selected is None:
            return
        if bool(head_asym_selector_assign_to_attn_mlp_policy):
            if attn_mlp_policy is not None and (
                int(attn_mlp_policy_updates) > 0 or int(attn_mlp_policy_attention_updates) > 0
            ):
                raise RuntimeError(
                    "Cannot assign head_asym_selector result to an AttnMLP policy "
                    "after policy updates have already started."
                )
            attn_mlp_policy_layer = int(selected["layer"])
            attn_mlp_policy_head = int(selected["head"])
            attn_mlp_policy_export_type = str(head_asym_selector_export_type)
            selected["assigned_to_attn_mlp_policy"] = True
            selected["attn_mlp_policy_layer"] = int(attn_mlp_policy_layer)
            selected["attn_mlp_policy_head"] = int(attn_mlp_policy_head)
            selected["attn_mlp_policy_export_type"] = str(attn_mlp_policy_export_type)
            if bool(head_asym_selector_init_attn_mlp_policy):
                _init_attn_mlp_policy()
                selected["attn_mlp_policy_initialized"] = bool(attn_mlp_policy is not None)
        else:
            selected["assigned_to_attn_mlp_policy"] = False

        _head_asym_selector_write(rows, selected, matrices, total_samples)
        if master_process and bool(wandb_log) and bool(head_asym_selector_log_wandb):
            wandb_module = globals().get("wandb", None)
            if wandb_module is not None:
                prefix = str(head_asym_selector_wandb_prefix).strip() or "head_asym_selector"
                payload = {
                    "iter": int(iter_num),
                    f"{prefix}/selected_layer": int(selected["layer"]),
                    f"{prefix}/selected_head": int(selected["head"]),
                    f"{prefix}/selected_asym_score": float(selected["asym_score"]),
                    f"{prefix}/num_samples": int(total_samples),
                    f"{prefix}/export_type_id": 0.0 if str(head_asym_selector_export_type) == "with_none" else 1.0,
                }
                if "selector_selected_score" in selected:
                    payload[f"{prefix}/selected_loss_score"] = float(selected["selector_selected_score"])
                    payload[f"{prefix}/selected_reverse"] = float(bool(selected.get("selector_selected_reverse", False)))
                    payload[f"{prefix}/selected_tau_diagnostic"] = float(
                        selected.get("selector_selected_tau_diagnostic", float("nan"))
                    )
                    payload[f"{prefix}/selection_rule_id"] = 1.0
                elif "selector_selected_tau_diagnostic" in selected:
                    payload[f"{prefix}/selected_tau_diagnostic"] = float(
                        selected.get("selector_selected_tau_diagnostic", float("nan"))
                    )
                    payload[f"{prefix}/selected_abs_tau_diagnostic"] = float(
                        selected.get("selector_selected_abs_tau_diagnostic", float("nan"))
                    )
                    payload[f"{prefix}/selected_reverse"] = float(bool(selected.get("selector_selected_reverse", False)))
                    payload[f"{prefix}/selection_rule_id"] = 2.0
                else:
                    payload[f"{prefix}/selection_rule_id"] = 0.0
                for row in rows:
                    head_prefix = f"{prefix}/L{int(row['layer'])}H{int(row['head'])}"
                    payload[f"{head_prefix}/asym_score"] = float(row["asym_score"])
                    if "selector_selected_score" in row:
                        payload[f"{head_prefix}/selector_selected_score"] = float(row["selector_selected_score"])
                    if "selector_selected_tau_diagnostic" in row:
                        payload[f"{head_prefix}/selector_selected_tau_diagnostic"] = float(
                            row["selector_selected_tau_diagnostic"]
                        )
                        payload[f"{head_prefix}/selector_selected_abs_tau_diagnostic"] = float(
                            row.get("selector_selected_abs_tau_diagnostic", float("nan"))
                        )
                wandb_module.log(payload)
    finally:
        if rng_state_cpu is not None:
            torch.random.set_rng_state(rng_state_cpu)
        if rng_state_cuda is not None:
            torch.cuda.set_rng_state_all(rng_state_cuda)
        if rng_state_np is not None:
            np.random.set_state(rng_state_np)
        if was_training:
            model.train()


def _head_signal_position_anchor_scores(source_override=None, force=False):
    if not bool(force) and not bool(head_signal_probe_position_anchor_enabled):
        return None
    source = str(source_override if source_override is not None else head_signal_probe_position_anchor_source).strip().lower()
    model_ref = raw_model if "raw_model" in globals() else model
    if not hasattr(model_ref, "transformer"):
        return None
    transformer = model_ref.transformer
    block_len = int(effective_order_block_len)
    if int(block_size) != int(num_blocks) * block_len:
        return None

    if source in {"target_index", "causal_index", "causal_context", "target_precedence", "causal_precedence"}:
        scores = torch.arange(int(num_blocks), dtype=torch.float32)
        scores = scores - scores.mean()
        scale = float(scores.norm().item())
        if not math.isfinite(scale) or scale <= 1e-12:
            return None
        return (scores.to(dtype=torch.float64) / scale).numpy()

    causal_pc1 = source in {"wtpe_causal_pc1", "wpe_causal_pc1"}
    feature_source = source.replace("_causal_pc1", "_pc1") if causal_pc1 else source
    if feature_source.startswith("wtpe"):
        if "wtpe" not in transformer:
            return None
        token_features = transformer.wtpe.weight[: int(block_size)].detach().float()
    elif feature_source.startswith("wpe"):
        if "wpe" not in transformer:
            return None
        token_features = transformer.wpe.weight[1 : int(block_size) + 1].detach().float()
    else:
        raise ValueError(
            "Unsupported head_signal_probe_position_anchor_source="
            f"{head_signal_probe_position_anchor_source!r}. Expected wtpe_pc1, wtpe_causal_pc1, "
            "wtpe_norm, wpe_pc1, wpe_causal_pc1, wpe_norm, target_index, or target_precedence."
        )

    block_features = token_features.view(int(num_blocks), block_len, -1).mean(dim=1)
    if feature_source.endswith("_norm"):
        scores = block_features.norm(dim=-1)
    elif feature_source.endswith("_pc1"):
        centered = block_features - block_features.mean(dim=0, keepdim=True)
        try:
            _, _, vh = torch.linalg.svd(centered.float(), full_matrices=False)
            axis = vh[0]
            scores = centered @ axis
        except Exception:
            return None
        if causal_pc1:
            causal_scores = torch.arange(
                int(num_blocks),
                dtype=scores.dtype,
                device=scores.device,
            )
            causal_scores = causal_scores - causal_scores.mean()
            if float((scores * causal_scores).sum().item()) < 0.0:
                scores = -scores
        elif scores.numel() > 0:
            max_idx = int(torch.argmax(scores.abs()).item())
            if float(scores[max_idx].item()) < 0.0:
                scores = -scores
    else:
        raise ValueError(
            "Unsupported head_signal_probe_position_anchor_source="
            f"{head_signal_probe_position_anchor_source!r}. Expected wtpe_pc1, wtpe_causal_pc1, "
            "wtpe_norm, wpe_pc1, wpe_causal_pc1, wpe_norm, target_index, or target_precedence."
        )

    scores = scores.detach().to(dtype=torch.float64, device="cpu")
    scores = scores - scores.mean()
    scale = float(scores.norm().item())
    if not math.isfinite(scale) or scale <= 1e-12:
        return None
    return (scores / scale).numpy()


def _head_signal_orient_order_with_position_anchor(raw_order, anchor_scores):
    if anchor_scores is None:
        return None
    raw_order = [int(value) for value in raw_order]
    if len(raw_order) != len(anchor_scores):
        return None
    if str(head_signal_probe_position_anchor_source).strip().lower() in {
        "target_precedence",
        "causal_precedence",
    }:
        target_order = list(range(len(raw_order)))
        margin = float(
            (_head_signal_precedence_matrix(raw_order) * _head_signal_precedence_matrix(target_order)).sum()
        )
    else:
        ranks = np.zeros(len(raw_order), dtype=np.float64)
        for rank, block_idx in enumerate(raw_order):
            ranks[int(block_idx)] = float(rank)
        ranks = ranks - float(ranks.mean())
        anchor = np.asarray(anchor_scores, dtype=np.float64)
        anchor = anchor - float(anchor.mean())
        margin = float((ranks * anchor).sum())
    selected_reverse = bool(margin < 0.0)
    selected_order = list(reversed(raw_order)) if selected_reverse else raw_order
    selected_original = _head_signal_order_to_original(selected_order)
    selected_tau = _head_signal_tau_to_l2r(selected_original)
    return {
        "position_anchor_source": str(head_signal_probe_position_anchor_source),
        "position_anchor_alignment_margin": float(margin),
        "position_anchor_selected_reverse": bool(selected_reverse),
        "position_anchor_tau_diagnostic": float(selected_tau),
        "position_anchor_abs_tau_diagnostic": abs(float(selected_tau)),
        "position_anchor_sign_diagnostic": int(_head_signal_sign(selected_tau)),
        "position_anchor_order_current": selected_order,
        "position_anchor_order_original_diagnostic": selected_original,
    }


def _head_signal_apply_position_consensus(rows):
    if not bool(head_signal_probe_position_consensus_enabled):
        return
    signed_precedence = []
    weights = []
    indexed_rows = []
    for row in rows:
        if "raw_order_current" not in row or "position_anchor_selected_reverse" not in row:
            continue
        raw_order = [int(value) for value in row["raw_order_current"]]
        sign = -1.0 if bool(row.get("position_anchor_selected_reverse", False)) else 1.0
        margin = abs(float(row.get("position_anchor_alignment_margin", 0.0))) + 1e-4
        indexed_rows.append(row)
        signed_precedence.append(_head_signal_precedence_matrix(raw_order) * sign)
        weights.append(float(margin))
    if not signed_precedence:
        return
    for idx, row in enumerate(indexed_rows):
        if bool(head_signal_probe_position_consensus_leave_one_out) and len(signed_precedence) > 1:
            denom = sum(float(weight) for j, weight in enumerate(weights) if j != idx)
            if denom <= 0.0:
                denom = sum(float(weight) for weight in weights)
            q_matrix = sum(
                float(weights[j]) * signed_precedence[j]
                for j in range(len(signed_precedence))
                if j != idx
            ) / float(max(denom, 1e-8))
        else:
            denom = sum(float(weight) for weight in weights)
            q_matrix = sum(
                float(weight) * matrix
                for weight, matrix in zip(weights, signed_precedence)
            ) / float(max(denom, 1e-8))
        raw_order = [int(value) for value in row["raw_order_current"]]
        raw_matrix = _head_signal_precedence_matrix(raw_order)
        align = float((raw_matrix * q_matrix).sum())
        selected_reverse = bool(align < 0.0)
        selected_order = list(reversed(raw_order)) if selected_reverse else raw_order
        selected_original = _head_signal_order_to_original(selected_order)
        selected_tau = _head_signal_tau_to_l2r(selected_original)
        row.update(
            {
                "position_consensus_alignment_margin": float(align),
                "position_consensus_selected_reverse": bool(selected_reverse),
                "position_consensus_tau_diagnostic": float(selected_tau),
                "position_consensus_abs_tau_diagnostic": abs(float(selected_tau)),
                "position_consensus_sign_diagnostic": int(_head_signal_sign(selected_tau)),
                "position_consensus_order_current": selected_order,
                "position_consensus_order_original_diagnostic": selected_original,
            }
        )


def _online_spectral_affinity_from_adjacency(matrix):
    values = matrix.detach().to(dtype=torch.float64, device="cpu").clone()
    values.fill_diagonal_(float("nan"))
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        raise ValueError("online spectral attention adjacency has no finite entries")
    median = torch.median(finite)
    q25 = torch.quantile(finite, 0.25)
    q75 = torch.quantile(finite, 0.75)
    scale = (q75 - q25) / 1.349
    if (not torch.isfinite(scale)) or float(scale) < 1e-8:
        scale = finite.std()
    if (not torch.isfinite(scale)) or float(scale) < 1e-8:
        scale = torch.tensor(1.0, dtype=torch.float64)
    z = (values - median) / scale
    finite_z = z[torch.isfinite(z)]
    threshold = torch.quantile(finite_z, float(online_spectral_threshold_percentile) / 100.0)
    shifted = z - threshold
    transform = str(online_spectral_transform)
    temperature = max(float(online_spectral_temperature), 1e-6)
    if transform == "relu":
        affinity = torch.clamp(shifted, min=0.0)
    elif transform == "exp":
        affinity = torch.exp(torch.clamp(shifted / temperature, min=-20.0, max=20.0))
        affinity = torch.where(shifted < 0.0, torch.zeros_like(affinity), affinity)
    elif transform == "softplus":
        affinity = torch.log1p(torch.exp(torch.clamp(shifted / temperature, min=-20.0, max=20.0)))
        affinity = torch.where(shifted < 0.0, affinity * 0.1, affinity)
    else:
        raise ValueError(f"Unsupported online_spectral_transform={transform!r}")
    affinity = torch.where(torch.isfinite(affinity), affinity, torch.zeros_like(affinity))
    affinity = 0.5 * (affinity + affinity.t())
    affinity.fill_diagonal_(0.0)
    return affinity


def _online_spectral_normalized_adjacency(matrix):
    affinity = _online_spectral_affinity_from_adjacency(matrix)
    degree = affinity.sum(dim=1)
    if float(degree.max()) <= 0.0:
        raise ValueError("online spectral affinity graph is empty")
    degree = torch.clamp(degree, min=1e-8)
    inv_sqrt = 1.0 / torch.sqrt(degree)
    return affinity * inv_sqrt[:, None] * inv_sqrt[None, :]


def _online_spectral_init_subspace(num_matrices, local_num_blocks, rank):
    rank = max(1, min(int(rank), int(local_num_blocks)))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(online_spectral_subspace_seed))
    q = torch.randn(
        int(num_matrices),
        int(local_num_blocks),
        int(rank),
        generator=generator,
        dtype=torch.float64,
    )
    chunks = []
    for idx in range(q.size(0)):
        chunks.append(torch.linalg.qr(q[idx], mode="reduced").Q)
    return torch.stack(chunks, dim=0)


def _online_spectral_update_subspaces(matrices):
    global online_spectral_subspace_q, online_spectral_global_subspace_q
    num_layers, num_heads, local_num_blocks, _ = matrices.shape
    flat = matrices.reshape(num_layers * num_heads, local_num_blocks, local_num_blocks)
    rank = max(1, min(int(online_spectral_subspace_rank), int(local_num_blocks)))
    if online_spectral_subspace_q is None or tuple(online_spectral_subspace_q.shape) != (
        int(num_layers),
        int(num_heads),
        int(local_num_blocks),
        int(rank),
    ):
        online_spectral_subspace_q = _online_spectral_init_subspace(
            int(num_layers) * int(num_heads),
            int(local_num_blocks),
            int(rank),
        ).reshape(num_layers, num_heads, local_num_blocks, rank)
    q_flat = online_spectral_subspace_q.reshape(num_layers * num_heads, local_num_blocks, rank)
    steps = max(1, int(online_spectral_subspace_steps_per_update))
    updated = []
    for matrix_idx in range(flat.size(0)):
        q = q_flat[matrix_idx]
        try:
            normalized = _online_spectral_normalized_adjacency(flat[matrix_idx])
            for _ in range(steps):
                q = torch.linalg.qr(normalized @ q, mode="reduced").Q
        except ValueError:
            pass
        updated.append(q)
    online_spectral_subspace_q = torch.stack(updated, dim=0).reshape(num_layers, num_heads, local_num_blocks, rank)

    global_matrix = matrices.mean(dim=(0, 1))
    if online_spectral_global_subspace_q is None or tuple(online_spectral_global_subspace_q.shape) != (
        int(local_num_blocks),
        int(rank),
    ):
        online_spectral_global_subspace_q = _online_spectral_init_subspace(1, int(local_num_blocks), int(rank))[0]
    q = online_spectral_global_subspace_q
    try:
        normalized = _online_spectral_normalized_adjacency(global_matrix)
        for _ in range(steps):
            q = torch.linalg.qr(normalized @ q, mode="reduced").Q
    except ValueError:
        pass
    online_spectral_global_subspace_q = q


def _spectral_matrix_with_nan_diagonal(matrix):
    out = matrix.detach().to(dtype=torch.float64, device="cpu").clone().numpy()
    if out.ndim == 4:
        for layer_idx in range(out.shape[0]):
            for head_idx in range(out.shape[1]):
                np.fill_diagonal(out[layer_idx, head_idx], np.nan)
    elif out.ndim == 2:
        np.fill_diagonal(out, np.nan)
    return out.astype(np.float32, copy=False)


def _write_online_spectral_state(force=False):
    global online_spectral_last_write_update
    if not _online_spectral_enabled() or online_spectral_matrices_sum is None or int(online_spectral_samples) <= 0:
        return
    write_every = int(online_spectral_write_every)
    if (not force) and write_every > 0 and int(online_spectral_updates) % write_every != 0:
        return
    if (not force) and int(online_spectral_last_write_update) == int(online_spectral_updates):
        return
    stats_dir = _online_spectral_dir()
    os.makedirs(stats_dir, exist_ok=True)
    rank_suffix = f"_rank{int(ddp_rank)}" if ddp else ""
    matrices = online_spectral_matrices_sum / float(max(1, int(online_spectral_samples)))
    payload = {
        "matrices": _spectral_matrix_with_nan_diagonal(matrices),
    }
    if online_spectral_matrices_ema is not None:
        payload["matrices_ema"] = _spectral_matrix_with_nan_diagonal(online_spectral_matrices_ema)
    if online_spectral_subspace_q is not None:
        payload["subspace_q"] = online_spectral_subspace_q.detach().cpu().numpy().astype(np.float32, copy=False)
    if online_spectral_global_subspace_q is not None:
        payload["global_subspace_q"] = (
            online_spectral_global_subspace_q.detach().cpu().numpy().astype(np.float32, copy=False)
        )
    rank_path = os.path.join(stats_dir, f"online_spectral_state{rank_suffix}.npz")
    np.savez_compressed(rank_path, **payload)
    meta = {
        "enabled": bool(online_spectral_enabled),
        "mode": str(online_spectral_mode),
        "rank": int(ddp_rank if ddp else 0),
        "world_size": int(ddp_world_size),
        "iter_num": int(iter_num),
        "num_blocks": int(num_blocks),
        "num_layers": int(matrices.shape[0]),
        "num_heads": int(matrices.shape[1]),
        "samples": int(online_spectral_samples),
        "updates": int(online_spectral_updates),
        "interval": int(online_spectral_interval),
        "batch_size": int(online_spectral_batch_size),
        "max_updates": int(online_spectral_max_updates),
        "export_type": str(online_spectral_export_type),
        "ema_decay": float(online_spectral_ema_decay),
        "subspace_rank": int(online_spectral_subspace_rank),
        "subspace_steps_per_update": int(online_spectral_subspace_steps_per_update),
        "threshold_percentile": float(online_spectral_threshold_percentile),
        "transform": str(online_spectral_transform),
        "temperature": float(online_spectral_temperature),
        "state_npz": rank_path,
        "note": (
            "Current-frame layer/head block attention state collected during training "
            "for attention-spectral recovery. No original-grid, hand scan, FID, or "
            "decoded metric is used."
        ),
    }
    meta_path = os.path.join(stats_dir, f"online_spectral_meta{rank_suffix}.json")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    if master_process:
        generic_npz = os.path.join(stats_dir, "online_spectral_state.npz")
        generic_meta = os.path.join(stats_dir, "online_spectral_meta.json")
        if rank_path != generic_npz:
            np.savez_compressed(generic_npz, **payload)
        with open(generic_meta, "w", encoding="utf-8") as handle:
            json.dump({**meta, "state_npz": generic_npz}, handle, ensure_ascii=False, indent=2)
    online_spectral_last_write_update = int(online_spectral_updates)


@torch.no_grad()
def _online_spectral_probe_update_if_due():
    global online_spectral_matrices_sum, online_spectral_matrices_ema
    global online_spectral_samples, online_spectral_updates
    if not _online_spectral_enabled():
        return
    interval = int(online_spectral_interval)
    if interval <= 0 or int(iter_num) % interval != 0:
        return
    if int(online_spectral_max_updates) > 0 and int(online_spectral_updates) >= int(online_spectral_max_updates):
        return
    mode = str(online_spectral_mode)
    if mode not in {"cache_exact", "subspace"}:
        raise ValueError(f"Unsupported online_spectral_mode={mode!r}")
    was_training = model.training
    model.eval()
    try:
        probe_batch_size = max(1, int(online_spectral_batch_size))
        X_probe, _ = get_batch('train', batch_size_override=probe_batch_size)
        block_orders, _, _ = _sample_explicit_training_block_orders(X_probe, return_units=False)
        if block_orders is None:
            return
        with ctx:
            outputs = _forward_with_explicit_block_orders(
                X_probe,
                block_orders,
                return_attentions=True,
                return_logits=False,
            )
        attentions = None
        for value in outputs[2:]:
            if isinstance(value, list) and value and torch.is_tensor(value[0]):
                attentions = value
                break
        if not attentions:
            return
        layer_matrices = []
        for layer_attn in attentions:
            layer_matrices.append(
                _aggregate_layerhead_attention_to_current_blocks(
                    layer_attn.detach(),
                    block_orders,
                    str(online_spectral_export_type),
                ).detach().cpu()
            )
        matrices = torch.stack(layer_matrices, dim=0).to(dtype=torch.float64, device="cpu")
        for layer_idx in range(matrices.size(0)):
            for head_idx in range(matrices.size(1)):
                matrices[layer_idx, head_idx].fill_diagonal_(0.0)
        samples = int(X_probe.size(0))
        if online_spectral_matrices_sum is None:
            online_spectral_matrices_sum = matrices * float(samples)
        else:
            online_spectral_matrices_sum += matrices * float(samples)
        decay = min(0.999, max(0.0, float(online_spectral_ema_decay)))
        if online_spectral_matrices_ema is None:
            online_spectral_matrices_ema = matrices
        else:
            online_spectral_matrices_ema = online_spectral_matrices_ema * decay + matrices * (1.0 - decay)
        online_spectral_samples += int(samples)
        online_spectral_updates += 1
        if mode == "subspace":
            _online_spectral_update_subspaces(online_spectral_matrices_ema)
        _write_online_spectral_state(force=False)
    finally:
        if was_training:
            model.train()


def _head_signal_probe_heads():
    raw_items = [item.strip() for item in str(head_signal_probe_heads).split(',') if item.strip()]
    if any(item.lower() in {"all", "*"} for item in raw_items):
        return [
            (layer, head)
            for layer in range(int(model.config.n_layer))
            for head in range(int(model.config.n_head))
        ]
    pairs = []
    seen = set()
    for item in raw_items:
        if ':' not in item:
            raise ValueError(
                f"head_signal_probe_heads item {item!r} must use 'layer:head' format."
            )
        layer_text, head_text = item.split(':', 1)
        layer_label = layer_text.strip().lower()
        head_label = head_text.strip().lower()
        if layer_label in {"all", "*", "global", "all_layers", "alllayers"}:
            if head_label not in {"mean", "avg", "average", "all", "*"}:
                raise ValueError(
                    "global head_signal_probe aggregation requires "
                    f"'all:mean' style syntax; got {item!r}."
                )
            key = (-1, -1)
            if key not in seen:
                seen.add(key)
                pairs.append(key)
            continue
        layer = int(layer_text)
        if layer < 0 or layer >= int(model.config.n_layer):
            raise ValueError(f"head_signal_probe layer={layer} outside 0..{int(model.config.n_layer)-1}")
        if head_label in {"mean", "avg", "average", "all", "*"}:
            head = -1
        else:
            head = int(head_text)
            if head < 0 or head >= int(model.config.n_head):
                raise ValueError(f"head_signal_probe head={head} outside 0..{int(model.config.n_head)-1}")
        key = (layer, head)
        if key not in seen:
            seen.add(key)
            pairs.append(key)
    return pairs


def _head_signal_head_label(head_idx):
    head_idx = int(head_idx)
    if head_idx == -1:
        return "mean"
    return str(head_idx)


def _head_signal_probe_matrix_for_head(matrices, layer_idx, head_idx):
    layer_idx = int(layer_idx)
    head_idx = int(head_idx)
    if layer_idx == -1:
        if head_idx != -1:
            raise ValueError("global head_signal_probe matrix only supports head_idx=-1.")
        return matrices.mean(dim=(0, 1))
    if head_idx == -1:
        return matrices[layer_idx].mean(dim=0)
    return matrices[layer_idx, head_idx]


def _head_signal_probe_dir():
    if str(head_signal_probe_out_dir).strip():
        return str(head_signal_probe_out_dir)
    return os.path.join(out_dir, "head_signal_probe")


def _head_signal_tau_to_l2r(order):
    values = [int(v) for v in order]
    n = len(values)
    total = n * (n - 1) / 2.0
    if total <= 0:
        return 1.0
    inversions = 0
    for i in range(n):
        left = values[i]
        for j in range(i + 1, n):
            if left > values[j]:
                inversions += 1
    return float(1.0 - 2.0 * float(inversions) / float(total))


def _head_signal_order_to_original(order_current):
    order_current = [int(v) for v in order_current]
    if not permute_data or fixed_block_perm is None:
        return list(order_current)
    mapper = fixed_block_perm.to(dtype=torch.long, device="cpu")
    current = torch.tensor(order_current, dtype=torch.long, device="cpu")
    return [int(value) for value in mapper[current].tolist()]


def _head_signal_sign(value, eps=1e-9):
    value = float(value)
    if value > float(eps):
        return 1
    if value < -float(eps):
        return -1
    return 0


def _head_signal_precedence_matrix(order):
    order = [int(v) for v in order]
    rank = {value: idx for idx, value in enumerate(order)}
    matrix = np.zeros((len(order), len(order)), dtype=np.float64)
    for i in range(len(order)):
        for j in range(len(order)):
            if i == j:
                continue
            matrix[i, j] = 1.0 if rank[i] < rank[j] else -1.0
    return matrix


def _head_signal_order_from_precedence_matrix(matrix):
    values = np.asarray(matrix, dtype=np.float64)
    scores = values.sum(axis=1)
    return [
        int(idx)
        for idx in sorted(
            range(int(values.shape[0])),
            key=lambda item: (-float(scores[item]), int(item)),
        )
    ]


def _head_signal_candidate_axes_for_fixed_angle(coords, component_pairs, num_angles, angle_idx):
    for a, b in component_pairs:
        if a < 0 or b < 0 or a >= coords.shape[1] or b >= coords.shape[1] or a == b:
            continue
        plane = coords[:, [a, b]]
        theta = float(angle_idx) * math.pi / float(num_angles)
        direction = np.asarray([math.cos(theta), math.sin(theta)], dtype=np.float64)
        axis = plane @ direction
        meta = {
            "component_pair": [int(a + 1), int(b + 1)],
            "angle_index": int(angle_idx),
            "angle_radians": float(theta),
        }
        yield f"c{a + 1}{b + 1}_ang{int(angle_idx):03d}", axis, meta


def _head_signal_recover_fixed_angle_candidates(matrix, config, angle_idx, top_m):
    values = matrix.detach().float().cpu().numpy() if torch.is_tensor(matrix) else np.asarray(matrix, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).copy()
    num_local_blocks = int(values.shape[0])
    np.fill_diagonal(values, np.nan)
    z = robust_z(values)
    score_adjacency = robust_z(sym(z, config.score_adjacency_sym))
    directed_variants = {
        "attn_query_key": anti(z),
        "attn_key_query": anti(z.T),
    }
    affinity = affinity_from_adjacency(
        values,
        threshold_percentile=float(config.threshold_percentile),
        transform=str(config.transform),
        temperature=float(config.temperature),
    )
    coords, eigvals = spectral_coordinates(affinity, num_components=int(config.num_components))
    axes = list(
        _head_signal_candidate_axes_for_fixed_angle(
            coords,
            parse_pairs(config.component_pairs),
            int(config.num_angles),
            int(angle_idx),
        )
    )
    candidates = []
    seen = set()
    k_values = parse_ints(config.k_values)
    group_methods = [item.strip() for item in str(config.group_methods).split(',') if item.strip()]
    direction_lambdas = parse_floats(config.direction_lambdas)
    for primary_name, primary_axis, primary_meta in axes:
        for secondary_name, secondary_axis, secondary_meta in axes:
            for k in k_values:
                for group_method in group_methods:
                    for primary_reverse in (False, True):
                        for secondary_reverse in (False, True):
                            for group_order_reverse in (False, True):
                                order, bands, band_score = grouped_order(
                                    primary_axis,
                                    secondary_axis,
                                    k=int(k),
                                    group_method=str(group_method),
                                    primary_reverse=bool(primary_reverse),
                                    secondary_reverse=bool(secondary_reverse),
                                    group_order_reverse=bool(group_order_reverse),
                                )
                                key = tuple(int(value) for value in order)
                                if key in seen:
                                    continue
                                seen.add(key)
                                score, score_meta = score_order(
                                    order,
                                    score_adjacency,
                                    directed_variants,
                                    direction_lambdas,
                                    directed_weight=float(config.directed_score_weight),
                                    band_score=float(band_score),
                                    band_weight=float(config.band_quality_weight),
                                )
                                candidates.append(
                                    {
                                        "name": (
                                            f"{primary_name}_x_{secondary_name}_k{int(k):02d}_{group_method}"
                                            f"_p{'d' if primary_reverse else 'a'}"
                                            f"_s{'d' if secondary_reverse else 'a'}"
                                            f"_g{'d' if group_order_reverse else 'a'}"
                                        ),
                                        "order": [int(value) for value in order],
                                        "score": float(score),
                                        "meta": {
                                            "primary_axis": primary_meta,
                                            "secondary_axis": secondary_meta,
                                            "k": int(k),
                                            "group_method": str(group_method),
                                            "primary_reverse": bool(primary_reverse),
                                            "secondary_reverse": bool(secondary_reverse),
                                            "group_order_reverse": bool(group_order_reverse),
                                            "band_quality": float(band_score),
                                            "group_sizes": [int(len(group)) for group in bands],
                                            "eigvals": [float(value) for value in eigvals.tolist()],
                                            "num_unique_candidates": int(len(seen)),
                                            "num_blocks": int(num_local_blocks),
                                            **score_meta,
                                        },
                                    }
                                )
    if not candidates:
        raise ValueError("head-signal fixed-angle recovery produced no candidates")
    candidates.sort(key=lambda item: (-float(item["score"]), str(item["name"])))
    for candidate in candidates:
        candidate["meta"]["num_unique_candidates"] = int(len(seen))
    return candidates[: int(top_m)]


def _head_signal_recover_direct_asym_eig_candidates(matrix, mode, top_m):
    values = matrix.detach().float().cpu().numpy() if torch.is_tensor(matrix) else np.asarray(matrix, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).copy()
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"direct_asym_eig expects a square matrix, got shape={tuple(values.shape)}")
    num_local_blocks = int(values.shape[0])
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(values, 0.0)

    mode_name = str(mode).strip().lower()
    if mode_name in {"", "direct_asym_eig"}:
        mode_name = "raw_right_largest_real_real"
    parts = set(item for item in mode_name.split("_") if item)
    eig_input = values
    input_name = "raw"
    if "center" in parts or "centered" in parts:
        eig_input = values - values.mean(axis=1, keepdims=True)
        input_name = "row_centered"
    if "left" in parts:
        eig_input = eig_input.T
        vector_side = "left"
    else:
        vector_side = "right"

    try:
        eigvals, eigvecs = np.linalg.eig(eig_input)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"direct_asym_eig failed: {exc}") from exc
    if eigvals.size <= 0 or eigvecs.size <= 0:
        raise ValueError("direct_asym_eig produced no eigenvectors")

    if "smallest" in parts and "real" in parts:
        eig_idx = int(np.argmin(np.real(eigvals)))
        eigen_selector = "smallest_real"
    elif "abs" in parts or "magnitude" in parts:
        eig_idx = int(np.argmax(np.abs(eigvals)))
        eigen_selector = "largest_abs"
    else:
        eig_idx = int(np.argmax(np.real(eigvals)))
        eigen_selector = "largest_real"

    vector_complex = eigvecs[:, eig_idx]
    if "imag" in parts or "imaginary" in parts:
        vector = np.imag(vector_complex)
        vector_part = "imag"
    elif "absvec" in parts or "vectorabs" in parts:
        vector = np.abs(vector_complex)
        vector_part = "abs"
    else:
        vector = np.real(vector_complex)
        vector_part = "real"
    vector = np.asarray(vector, dtype=np.float64)
    if not np.isfinite(vector).all() or float(np.nanstd(vector)) <= 1e-12:
        fallback = values.sum(axis=1) - values.sum(axis=0)
        if np.isfinite(fallback).all() and float(np.nanstd(fallback)) > 1e-12:
            vector = np.asarray(fallback, dtype=np.float64)
            vector_part = f"{vector_part}_fallback_row_minus_col_sum"
    if not np.isfinite(vector).all():
        raise ValueError("direct_asym_eig produced a non-finite ordering vector")
    if float(np.std(vector)) <= 1e-12:
        raise ValueError("direct_asym_eig produced a degenerate ordering vector")

    order = [
        int(idx)
        for idx in sorted(
            range(num_local_blocks),
            key=lambda item: (float(vector[item]), int(item)),
        )
    ]
    candidate = {
        "name": f"direct_asym_eig_{mode_name}",
        "order": order,
        "score": float(abs(eigvals[eig_idx])),
        "meta": {
            "candidate_source": "direct_asym_eig",
            "mode": str(mode_name),
            "input": str(input_name),
            "vector_side": str(vector_side),
            "eigen_selector": str(eigen_selector),
            "vector_part": str(vector_part),
            "eigval_real": float(np.real(eigvals[eig_idx])),
            "eigval_imag": float(np.imag(eigvals[eig_idx])),
            "eigval_abs": float(abs(eigvals[eig_idx])),
            "vector_min": float(np.min(vector)),
            "vector_max": float(np.max(vector)),
            "vector_std": float(np.std(vector)),
            "num_unique_candidates": 1,
            "num_blocks": int(num_local_blocks),
        },
    }
    return [candidate][: max(1, int(top_m))]


def _head_signal_recover_pairwise_max_fiedler_candidates(matrix, top_m):
    values = matrix.detach().float().cpu().numpy() if torch.is_tensor(matrix) else np.asarray(matrix, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).copy()
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"pairwise_max_fiedler expects a square matrix, got shape={tuple(values.shape)}")
    num_local_blocks = int(values.shape[0])
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(values, 0.0)
    affinity = np.maximum(values, values.T)
    affinity = np.nan_to_num(affinity, nan=0.0, posinf=0.0, neginf=0.0)
    affinity = np.maximum(affinity, 0.0)
    np.fill_diagonal(affinity, 0.0)
    degree = affinity.sum(axis=1)
    laplacian = np.diag(degree) - affinity
    try:
        eigvals, eigvecs = np.linalg.eigh(laplacian)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"pairwise_max_fiedler failed: {exc}") from exc
    if eigvals.size <= 1 or eigvecs.size <= 0:
        raise ValueError("pairwise_max_fiedler produced no nontrivial eigenvectors")
    eig_idx = 1
    vector = np.asarray(eigvecs[:, eig_idx], dtype=np.float64)
    if not np.isfinite(vector).all():
        raise ValueError("pairwise_max_fiedler produced a non-finite Fiedler vector")
    if float(np.std(vector)) <= 1e-12:
        raise ValueError("pairwise_max_fiedler produced a degenerate Fiedler vector")
    order = [
        int(idx)
        for idx in sorted(
            range(num_local_blocks),
            key=lambda item: (float(vector[item]), int(item)),
        )
    ]
    spectral_gap = (
        float(eigvals[2] - eigvals[1])
        if eigvals.size > 2
        else float(abs(eigvals[1]))
    )
    candidate = {
        "name": "pairwise_max_fiedler",
        "order": order,
        "score": float(max(spectral_gap, 1e-12)),
        "meta": {
            "candidate_source": "pairwise_max_fiedler",
            "input": "pairwise_max_affinity",
            "affinity": "max(A,A.T)",
            "laplacian": "D-W",
            "eigen_selector": "second_smallest_laplacian",
            "vector_part": "fiedler_real",
            "laplacian_eigval0": float(eigvals[0]),
            "laplacian_fiedler_eigval": float(eigvals[1]),
            "laplacian_next_eigval": float(eigvals[2]) if eigvals.size > 2 else float("nan"),
            "spectral_gap": float(spectral_gap),
            "affinity_sum": float(affinity.sum()),
            "affinity_density": float(np.mean(affinity > 0.0)),
            "vector_min": float(np.min(vector)),
            "vector_max": float(np.max(vector)),
            "vector_std": float(np.std(vector)),
            "fiedler_vector": [float(value) for value in vector.tolist()],
            "num_unique_candidates": 1,
            "num_blocks": int(num_local_blocks),
            "no_prior_rule": (
                "Layer/head attention defines an undirected local-order axis via max(A,A.T) "
                "and Laplacian Fiedler vector; current loss/profile orients axis vs reverse."
            ),
        },
    }
    return [candidate][: max(1, int(top_m))]


def _head_signal_recover_candidates(matrix, recovery_config):
    source = str(head_signal_probe_candidate_source).strip().lower()
    if source in {"", "spectral", "attention_spectral", "fixed_angle", "affinity"}:
        return _head_signal_recover_fixed_angle_candidates(
            matrix,
            recovery_config,
            int(head_signal_probe_fixed_angle_idx),
            int(head_signal_probe_top_m),
        )
    if source in {"direct_asym_eig", "asym_eig", "direct_eig"}:
        return _head_signal_recover_direct_asym_eig_candidates(
            matrix,
            str(head_signal_probe_direct_asym_eig_mode),
            int(head_signal_probe_top_m),
        )
    if source in {
        "pairwise_max_fiedler",
        "layer_pairwise_max_fiedler",
        "max_fiedler",
        "fiedler",
        "laplacian_fiedler",
    }:
        return _head_signal_recover_pairwise_max_fiedler_candidates(
            matrix,
            int(head_signal_probe_top_m),
        )
    raise ValueError(
        "Unsupported head_signal_probe_candidate_source="
        f"{head_signal_probe_candidate_source!r}; expected spectral, direct_asym_eig, or pairwise_max_fiedler."
    )


@torch.no_grad()
def _head_signal_evaluate_order_losses(order_list):
    unique_orders = []
    seen = set()
    for order in order_list:
        key = tuple(int(v) for v in order)
        if key not in seen:
            seen.add(key)
            unique_orders.append(list(key))
    if not unique_orders:
        return {}
    order_tensor = torch.tensor(unique_orders, dtype=torch.long, device=device)
    candidate_batch_size = max(1, int(head_signal_probe_candidate_batch_size))
    probe_batch_size = max(1, int(head_signal_probe_loss_batch_size))
    prefix_k = max(1, min(int(head_signal_probe_prefix_k), int(num_blocks)))
    stats = {
        tuple(order): {
            "full_sum": 0.0,
            "prefix_sum": 0.0,
            "profile_sum": np.zeros(int(num_blocks), dtype=np.float64),
            "count": 0,
        }
        for order in unique_orders
    }
    for _ in range(max(1, int(head_signal_probe_loss_batches))):
        X_probe, _ = get_batch('train', batch_size_override=probe_batch_size)
        batch_size_local = int(X_probe.size(0))
        for start in range(0, len(unique_orders), candidate_batch_size):
            end = min(len(unique_orders), start + candidate_batch_size)
            chunk = order_tensor[start:end]
            chunk_size = int(chunk.size(0))
            block_orders = (
                chunk[:, None, :]
                .expand(chunk_size, batch_size_local, int(num_blocks))
                .reshape(chunk_size * batch_size_local, int(num_blocks))
            )
            X_expanded = (
                X_probe[None, :, :]
                .expand(chunk_size, batch_size_local, int(X_probe.size(1)))
                .reshape(chunk_size * batch_size_local, int(X_probe.size(1)))
            )
            with ctx:
                outputs = _forward_with_explicit_block_orders(
                    X_expanded,
                    block_orders,
                    return_token_loss=True,
                    return_logits=False,
                )
            token_losses = outputs[2]
            block_losses = token_losses_to_block_losses(
                token_losses.detach(),
                block_len=effective_order_block_len,
            ).float().view(chunk_size, batch_size_local, int(num_blocks))
            prefix_loss = block_losses[:, :, :prefix_k].mean(dim=2)
            full_loss = block_losses.mean(dim=2)
            for local_idx, order_idx in enumerate(range(start, end)):
                key = tuple(unique_orders[order_idx])
                stats[key]["prefix_sum"] += float(prefix_loss[local_idx].double().sum().item())
                stats[key]["full_sum"] += float(full_loss[local_idx].double().sum().item())
                stats[key]["profile_sum"] += (
                    block_losses[local_idx].double().sum(dim=0).detach().cpu().numpy()
                )
                stats[key]["count"] += int(batch_size_local)
    out = {}
    linear_weights = np.linspace(1.0, 0.1, int(num_blocks), dtype=np.float64)
    linear_weights /= float(max(linear_weights.sum(), 1e-12))
    exp_weights = np.exp(
        -np.arange(int(num_blocks), dtype=np.float64)
        / max(float(head_signal_probe_candidate_loss_profile_exp_tau), 1e-6)
    )
    exp_weights /= float(max(exp_weights.sum(), 1e-12))
    for key, item in stats.items():
        count = max(1, int(item["count"]))
        profile = np.asarray(item["profile_sum"], dtype=np.float64) / float(count)
        out[key] = {
            "prefix_loss": float(item["prefix_sum"] / count),
            "full_loss": float(item["full_sum"] / count),
            "linear_profile_loss": float((linear_weights * profile).sum()),
            "exp_profile_loss": float((exp_weights * profile).sum()),
            "loss_profile": [float(value) for value in profile.tolist()],
            "count": int(item["count"]),
        }
    return out


def _head_signal_order_loss_profile_score(loss_item, score_name):
    score_name = str(score_name).strip().lower()
    if score_name in {"prefix", "prefix_loss"}:
        return float(loss_item["prefix_loss"])
    if score_name in {"full", "full_loss"}:
        return float(loss_item["full_loss"])
    if score_name in {"prefix_full", "mixed", "prefix_full_loss"}:
        return (
            float(head_signal_probe_prefix_weight) * float(loss_item["prefix_loss"])
            + float(head_signal_probe_full_weight) * float(loss_item["full_loss"])
        )
    if score_name in {"linear", "linear_profile", "linear_profile_loss"}:
        return float(loss_item["linear_profile_loss"])
    if score_name in {"exp", "exp_profile", "exp_profile_loss"}:
        return float(loss_item["exp_profile_loss"])
    raise ValueError(f"Unsupported head_signal_probe_candidate_loss_profile_score={score_name!r}")


def _head_signal_apply_candidate_loss_profile_consensus(rows, loss_by_order):
    score_name = str(head_signal_probe_candidate_loss_profile_score).strip().lower()
    indexed = []
    winner_matrices = []
    winner_weights = []
    for row in rows:
        candidates = row.get("_head_signal_candidate_orders_current")
        if not candidates:
            continue
        scored = []
        for order in candidates:
            key = tuple(int(value) for value in order)
            loss_item = loss_by_order.get(key)
            if loss_item is None:
                continue
            score = _head_signal_order_loss_profile_score(loss_item, score_name)
            scored.append((float(score), [int(value) for value in order]))
        if not scored:
            continue
        scored.sort(key=lambda item: (float(item[0]), json.dumps(item[1], separators=(",", ":"))))
        best_score, best_order = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else best_score
        gap = max(0.0, float(second_score) - float(best_score))
        row["_head_signal_candidate_loss_profile_index"] = int(len(indexed))
        row["_head_signal_candidate_loss_profile_scored"] = scored
        row["_head_signal_candidate_loss_profile_best_order"] = best_order
        row["_head_signal_candidate_loss_profile_best_score"] = float(best_score)
        row["_head_signal_candidate_loss_profile_score_gap"] = float(gap)
        indexed.append(row)
        winner_matrices.append(_head_signal_precedence_matrix(best_order))
        winner_weights.append(float(gap) + 1e-4)

    if not indexed:
        return

    for row in indexed:
        idx = int(row["_head_signal_candidate_loss_profile_index"])
        if bool(head_signal_probe_consensus_leave_one_out) and len(winner_matrices) > 1:
            denom = sum(
                float(weight)
                for j, weight in enumerate(winner_weights)
                if j != idx
            )
            if denom <= 0.0:
                denom = sum(float(weight) for weight in winner_weights)
            q_matrix = sum(
                float(winner_weights[j]) * winner_matrices[j]
                for j in range(len(winner_matrices))
                if j != idx
            ) / float(max(denom, 1e-8))
        else:
            denom = sum(float(weight) for weight in winner_weights)
            q_matrix = sum(
                float(weight) * matrix
                for weight, matrix in zip(winner_weights, winner_matrices)
            ) / float(max(denom, 1e-8))

        aligned = []
        for score, order in row.get("_head_signal_candidate_loss_profile_scored", []):
            matrix = _head_signal_precedence_matrix(order)
            align = float((matrix * q_matrix).sum())
            aligned.append((align, -float(score), order, float(score)))
        if not aligned:
            continue
        aligned.sort(key=lambda item: (-float(item[0]), -float(item[1]), json.dumps(item[2], separators=(",", ":"))))
        align, _neg_score, selected_order, selected_score = aligned[0]
        low_confidence_reasons = []
        score_gap = float(row["_head_signal_candidate_loss_profile_score_gap"])
        if score_gap < float(head_signal_probe_candidate_loss_profile_min_gap):
            low_confidence_reasons.append("score_gap")
        if abs(float(align)) < float(head_signal_probe_candidate_loss_profile_min_alignment):
            low_confidence_reasons.append("alignment")
        fallback_mode = str(head_signal_probe_candidate_loss_profile_low_confidence_fallback).strip().lower()
        used_fallback = False
        if low_confidence_reasons and fallback_mode in {"global_q", "global", "q"}:
            selected_order = _head_signal_order_from_precedence_matrix(q_matrix)
            selected_score = float("nan")
            align = float((_head_signal_precedence_matrix(selected_order) * q_matrix).sum())
            used_fallback = True
        selected_original = _head_signal_order_to_original(selected_order)
        selected_tau = _head_signal_tau_to_l2r(selected_original)
        raw_order = [int(value) for value in row.get("raw_order_current", [])]
        reverse_raw = list(reversed(raw_order)) if raw_order else []
        row.update(
            {
                "loss_profile_score_name": str(score_name),
                "loss_profile_best_score": float(row["_head_signal_candidate_loss_profile_best_score"]),
                "loss_profile_score_gap": float(score_gap),
                "loss_profile_consensus_alignment": float(align),
                "loss_profile_consensus_selected_score": float(selected_score),
                "loss_profile_low_confidence": bool(bool(low_confidence_reasons)),
                "loss_profile_low_confidence_reasons": ",".join(low_confidence_reasons),
                "loss_profile_low_confidence_fallback": str(fallback_mode),
                "loss_profile_used_fallback": bool(used_fallback),
                "loss_profile_consensus_tau_diagnostic": float(selected_tau),
                "loss_profile_consensus_abs_tau_diagnostic": abs(float(selected_tau)),
                "loss_profile_consensus_sign_diagnostic": int(_head_signal_sign(selected_tau)),
                "loss_profile_consensus_order_current": selected_order,
                "loss_profile_consensus_order_original_diagnostic": selected_original,
                "consensus_rule": "linear_profile_consensus_candidate",
                "consensus_alignment_margin": float(align),
                "consensus_selected_reverse": bool(reverse_raw and selected_order == reverse_raw),
                "consensus_tau_diagnostic": float(selected_tau),
                "consensus_abs_tau_diagnostic": abs(float(selected_tau)),
                "consensus_sign_diagnostic": int(_head_signal_sign(selected_tau)),
                "consensus_order_current": selected_order,
                "consensus_order_original_diagnostic": selected_original,
            }
        )


def _head_signal_apply_candidate_loss_profile_selection(rows, loss_by_order):
    score_name = str(head_signal_probe_candidate_loss_profile_score).strip().lower()
    for row in rows:
        candidates = row.get("_head_signal_candidate_orders_current")
        if not candidates:
            continue
        scored = []
        for order in candidates:
            key = tuple(int(value) for value in order)
            loss_item = loss_by_order.get(key)
            if loss_item is None:
                continue
            score = _head_signal_order_loss_profile_score(loss_item, score_name)
            scored.append((float(score), [int(value) for value in order], loss_item))
        if not scored:
            continue
        scored.sort(key=lambda item: (float(item[0]), json.dumps(item[1], separators=(",", ":"))))
        best_score, selected_order, selected_loss = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else best_score
        score_gap = max(0.0, float(second_score) - float(best_score))
        selected_original = _head_signal_order_to_original(selected_order)
        selected_tau = _head_signal_tau_to_l2r(selected_original)
        raw_order = [int(value) for value in row.get("raw_order_current", [])]
        reverse_raw = list(reversed(raw_order)) if raw_order else []
        row.update(
            {
                "loss_profile_score_name": str(score_name),
                "loss_profile_best_score": float(best_score),
                "loss_profile_score_gap": float(score_gap),
                "loss_profile_selected_score": float(best_score),
                "loss_profile_selected_prefix_loss": float(selected_loss.get("prefix_loss", float("nan"))),
                "loss_profile_selected_full_loss": float(selected_loss.get("full_loss", float("nan"))),
                "loss_profile_selected_linear_profile_loss": float(
                    selected_loss.get("linear_profile_loss", float("nan"))
                ),
                "loss_profile_selected_exp_profile_loss": float(
                    selected_loss.get("exp_profile_loss", float("nan"))
                ),
                "loss_profile_selected_reverse": bool(reverse_raw and selected_order == reverse_raw),
                "loss_profile_selected_tau_diagnostic": float(selected_tau),
                "loss_profile_selected_abs_tau_diagnostic": abs(float(selected_tau)),
                "loss_profile_selected_sign_diagnostic": int(_head_signal_sign(selected_tau)),
                "loss_profile_selected_order_current": selected_order,
                "loss_profile_selected_order_original_diagnostic": selected_original,
                "consensus_rule": "none_loss_profile_direct",
                "consensus_alignment_margin": float("nan"),
                "consensus_selected_reverse": bool(reverse_raw and selected_order == reverse_raw),
                "consensus_tau_diagnostic": float(selected_tau),
                "consensus_abs_tau_diagnostic": abs(float(selected_tau)),
                "consensus_sign_diagnostic": int(_head_signal_sign(selected_tau)),
                "consensus_order_current": selected_order,
                "consensus_order_original_diagnostic": selected_original,
            }
        )


@torch.no_grad()
def _head_signal_probe_update_if_due():
    if not bool(head_signal_probe_enabled):
        return
    if int(iter_num) < int(head_signal_probe_start_iter):
        return
    stop_iter = int(head_signal_probe_stop_iter)
    if stop_iter >= 0 and int(iter_num) >= int(stop_iter):
        return
    interval = int(head_signal_probe_interval)
    if interval <= 0 or int(iter_num) % interval != 0:
        return
    heads = _head_signal_probe_heads()
    if not heads:
        return
    deterministic_probe = bool(head_signal_probe_deterministic)
    rng_state_cpu = None
    rng_state_cuda = None
    rng_state_np = None
    if deterministic_probe:
        rng_state_cpu = torch.random.get_rng_state()
        rng_state_np = np.random.get_state()
        if device_type == 'cuda' and torch.cuda.is_available():
            rng_state_cuda = torch.cuda.get_rng_state_all()
        seed = int(head_signal_probe_seed)
        torch.manual_seed(seed)
        if device_type == 'cuda' and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed % (2**32 - 1))
    was_training = model.training
    model.eval()
    try:
        probe_batch_size = max(1, int(head_signal_probe_batch_size))
        matrix_sums = None
        total_samples = 0
        for _ in range(max(1, int(head_signal_probe_batches))):
            X_probe, _ = get_batch('train', batch_size_override=probe_batch_size)
            block_orders = sample_random_block_orders(
                batch_size=int(X_probe.size(0)),
                num_blocks=num_blocks,
                device=X_probe.device,
            )
            with ctx:
                outputs = _forward_with_explicit_block_orders(
                    X_probe,
                    block_orders,
                    return_attentions=True,
                    return_logits=False,
                )
            attentions = _attn_mlp_extract_attentions(outputs)
            if not attentions:
                continue
            layer_matrices = []
            for layer_attn in attentions:
                layer_matrices.append(
                    _aggregate_layerhead_attention_to_current_blocks(
                        layer_attn.detach(),
                        block_orders,
                        str(head_signal_probe_export_type),
                    ).detach().cpu()
                )
            matrices = torch.stack(layer_matrices, dim=0).to(dtype=torch.float64, device="cpu")
            if matrix_sums is None:
                matrix_sums = matrices * float(X_probe.size(0))
            else:
                matrix_sums += matrices * float(X_probe.size(0))
            total_samples += int(X_probe.size(0))
        if matrix_sums is None or total_samples <= 0:
            return
        matrices = matrix_sums / float(total_samples)
        for layer_idx in range(matrices.size(0)):
            for head_idx in range(matrices.size(1)):
                matrices[layer_idx, head_idx].fill_diagonal_(0.0)

        recovery_config = FixedHeadSpectralPolicyConfig(
            num_components=int(head_signal_probe_num_components),
            component_pairs=str(head_signal_probe_component_pairs),
            num_angles=int(head_signal_probe_num_angles),
            k_values=str(head_signal_probe_k_values),
            group_methods=str(head_signal_probe_group_methods),
            threshold_percentile=float(head_signal_probe_threshold_percentile),
            transform=str(head_signal_probe_transform),
            temperature=float(head_signal_probe_temperature),
            direction_lambdas=str(head_signal_probe_direction_lambdas),
            directed_score_weight=float(head_signal_probe_directed_score_weight),
            band_quality_weight=float(head_signal_probe_band_quality_weight),
            score_adjacency_sym=str(head_signal_probe_score_adjacency_sym),
        )
        position_anchor_scores = _head_signal_position_anchor_scores()
        orientation_rule = str(head_signal_probe_orientation_rule).strip().lower()
        direct_loss_profile_rules = {
            "linear_profile_candidate",
            "linear_profile_direct_candidate",
            "loss_profile_candidate",
            "candidate_loss_profile",
            "candidate_loss_profile_direct",
        }
        candidate_loss_profile_enabled = bool(head_signal_probe_candidate_loss_profile_enabled) or orientation_rule in {
            "linear_profile_consensus_candidate",
            "loss_profile_consensus_candidate",
            "candidate_loss_profile_consensus",
            *direct_loss_profile_rules,
        }

        rows = []
        order_loss_requests = []
        for layer_idx, head_idx in heads:
            head_label = _head_signal_head_label(head_idx)
            row = {
                "version": 1,
                "iter": int(iter_num),
                "wall_time": float(time.time()),
                "layer": int(layer_idx),
                "head": int(head_idx),
                "head_label": str(head_label),
                "head_aggregation": "mean" if int(head_idx) == -1 else "single",
                "samples": int(total_samples),
                "probe_batches": int(head_signal_probe_batches),
                "probe_batch_size": int(probe_batch_size),
                "loss_batches": int(head_signal_probe_loss_batches),
                "loss_batch_size": int(head_signal_probe_loss_batch_size),
                "fixed_angle_idx": int(head_signal_probe_fixed_angle_idx),
                "candidate_source": str(head_signal_probe_candidate_source),
                "direct_asym_eig_mode": str(head_signal_probe_direct_asym_eig_mode),
                "deterministic_probe": bool(deterministic_probe),
                "deterministic_seed": int(head_signal_probe_seed) if deterministic_probe else None,
                "orientation_rule": str(head_signal_probe_orientation_rule),
                "note": (
                    "Online training probe. Tau/original order fields are diagnostic-only; "
                    + (
                        "orientation uses current-model loss only; no cross-head consensus."
                        if orientation_rule in direct_loss_profile_rules
                        else "orientation uses current-model loss and same-iter cross-head consensus."
                    )
                ),
            }
            try:
                head_matrix = _head_signal_probe_matrix_for_head(matrices, layer_idx, head_idx)
                _online_spectral_log_input_attention_if_due(
                    head_matrix,
                    iter_num,
                    layer_idx,
                    head_idx,
                    head_label,
                    str(head_signal_probe_export_type),
                )
                candidates = _head_signal_recover_candidates(
                    head_matrix,
                    recovery_config,
                )
                top = candidates[0]
                raw_order = [int(value) for value in top["order"]]
                raw_original = _head_signal_order_to_original(raw_order)
                raw_tau = _head_signal_tau_to_l2r(raw_original)
                reverse_order = list(reversed(raw_order))
                candidate_orders = []
                candidate_seen = set()
                max_rank = max(1, int(head_signal_probe_candidate_loss_profile_max_rank))
                for candidate in candidates[:max_rank]:
                    candidate_order = [int(value) for value in candidate["order"]]
                    for order_variant in (
                        [candidate_order, list(reversed(candidate_order))]
                        if bool(head_signal_probe_candidate_loss_profile_include_reverse)
                        else [candidate_order]
                    ):
                        key = tuple(int(value) for value in order_variant)
                        if key in candidate_seen:
                            continue
                        candidate_seen.add(key)
                        candidate_orders.append(list(key))
                row.update(
                    {
                        "raw_score": float(top.get("score", float("nan"))),
                        "raw_tau_diagnostic": float(raw_tau),
                        "raw_abs_tau_diagnostic": abs(float(raw_tau)),
                        "raw_sign_diagnostic": int(_head_signal_sign(raw_tau)),
                        "raw_order_current": raw_order,
                        "raw_order_original_diagnostic": raw_original,
                        "raw_candidate_name": str(top.get("name", "")),
                    }
                )
                position_anchor_payload = _head_signal_orient_order_with_position_anchor(
                    raw_order,
                    position_anchor_scores,
                )
                if position_anchor_payload is not None:
                    row.update(position_anchor_payload)
                meta = top.get("meta", {})
                if isinstance(meta, dict):
                    row["raw_candidate_source"] = str(
                        meta.get("candidate_source", str(head_signal_probe_candidate_source))
                    )
                    row["raw_directed_path_score"] = float(meta.get("best_directed_path_score", float("nan")))
                    row["raw_adjacency_path_score"] = float(meta.get("adjacency_path_score", float("nan")))
                    row["raw_band_quality"] = float(meta.get("band_quality", float("nan")))
                    for meta_key in (
                        "mode",
                        "input",
                        "vector_side",
                        "eigen_selector",
                        "vector_part",
                    ):
                        if meta_key in meta:
                            row[f"raw_eig_{meta_key}"] = str(meta.get(meta_key))
                    for meta_key in (
                        "eigval_real",
                        "eigval_imag",
                        "eigval_abs",
                        "laplacian_eigval0",
                        "laplacian_fiedler_eigval",
                        "laplacian_next_eigval",
                        "spectral_gap",
                        "affinity_sum",
                        "affinity_density",
                        "vector_min",
                        "vector_max",
                        "vector_std",
                    ):
                        if meta_key in meta:
                            row[f"raw_eig_{meta_key}"] = float(meta.get(meta_key, float("nan")))
                    fiedler_vector = meta.get("fiedler_vector", None)
                    if isinstance(fiedler_vector, (list, tuple)) and len(fiedler_vector) == int(num_blocks):
                        row["raw_fiedler_vector"] = [float(value) for value in fiedler_vector]
                if candidate_loss_profile_enabled:
                    row["_head_signal_candidate_orders_current"] = candidate_orders
                    order_loss_requests.extend(candidate_orders)
                else:
                    order_loss_requests.extend([raw_order, reverse_order])
            except Exception as exc:
                row["error"] = str(exc)
            rows.append(row)

        _head_signal_apply_position_consensus(rows)

        sync_global_state = None
        if orientation_rule in {"sync_global_loss", "global_loss_sync", "sync_loss"}:
            sync_rows = []
            sync_matrices = []
            for row in rows:
                if "raw_order_current" not in row:
                    continue
                raw_order = [int(value) for value in row["raw_order_current"]]
                row["_head_signal_sync_index"] = int(len(sync_rows))
                sync_rows.append(row)
                sync_matrices.append(_head_signal_precedence_matrix(raw_order))
            if sync_matrices:
                num_sync = int(len(sync_matrices))
                agreement = np.zeros((num_sync, num_sync), dtype=np.float64)
                for i in range(num_sync):
                    for j in range(i, num_sync):
                        value = float((sync_matrices[i] * sync_matrices[j]).mean())
                        agreement[i, j] = value
                        agreement[j, i] = value
                try:
                    eigvals, eigvecs = np.linalg.eigh(agreement)
                    top = eigvecs[:, int(np.argmax(eigvals))]
                    sync_signs = np.where(top >= 0.0, 1.0, -1.0).astype(np.float64)
                    sync_eigval = float(np.max(eigvals))
                except Exception:
                    sync_signs = np.ones(num_sync, dtype=np.float64)
                    sync_eigval = float("nan")
                q_matrix = sum(
                    float(sync_signs[idx]) * sync_matrices[idx]
                    for idx in range(num_sync)
                ) / float(max(num_sync, 1))
                global_order = _head_signal_order_from_precedence_matrix(q_matrix)
                global_reverse = list(reversed(global_order))
                order_loss_requests.extend([global_order, global_reverse])
                sync_global_state = {
                    "rows": sync_rows,
                    "matrices": sync_matrices,
                    "signs": sync_signs,
                    "q_matrix": q_matrix,
                    "global_order": global_order,
                    "global_reverse": global_reverse,
                    "sync_eigval": sync_eigval,
                    "mean_abs_agreement": float(np.mean(np.abs(agreement))),
                }

        loss_by_order = _head_signal_evaluate_order_losses(order_loss_requests)
        prefix_weight = float(head_signal_probe_prefix_weight)
        full_weight = float(head_signal_probe_full_weight)
        signed_precedence = []
        signed_weights = []
        for row in rows:
            if "raw_order_current" not in row:
                continue
            raw_order = [int(value) for value in row["raw_order_current"]]
            reverse_order = list(reversed(raw_order))
            raw_loss = loss_by_order.get(tuple(raw_order))
            reverse_loss = loss_by_order.get(tuple(reverse_order))
            if raw_loss is None or reverse_loss is None:
                continue
            raw_selection_loss = (
                prefix_weight * float(raw_loss["prefix_loss"])
                + full_weight * float(raw_loss["full_loss"])
            )
            reverse_selection_loss = (
                prefix_weight * float(reverse_loss["prefix_loss"])
                + full_weight * float(reverse_loss["full_loss"])
            )
            margin = float(reverse_selection_loss - raw_selection_loss)
            loss_order = reverse_order if margin < 0.0 else raw_order
            loss_original = _head_signal_order_to_original(loss_order)
            loss_tau = _head_signal_tau_to_l2r(loss_original)
            row.update(
                {
                    "raw_prefix_loss": float(raw_loss["prefix_loss"]),
                    "raw_full_loss": float(raw_loss["full_loss"]),
                    "reverse_prefix_loss": float(reverse_loss["prefix_loss"]),
                    "reverse_full_loss": float(reverse_loss["full_loss"]),
                    "loss_selection_margin_reverse_minus_forward": float(margin),
                    "loss_oriented_tau_diagnostic": float(loss_tau),
                    "loss_oriented_abs_tau_diagnostic": abs(float(loss_tau)),
                    "loss_oriented_sign_diagnostic": int(_head_signal_sign(loss_tau)),
                    "loss_oriented_selected_reverse": bool(margin < 0.0),
                    "loss_oriented_order_current": loss_order,
                    "loss_oriented_order_original_diagnostic": loss_original,
                    "loss_eval_count": int(raw_loss.get("count", 0)),
                }
            )
            row["_head_signal_consensus_index"] = int(len(signed_precedence))
            signed_precedence.append(
                _head_signal_precedence_matrix(raw_order) * (1.0 if margin >= 0.0 else -1.0)
            )
            signed_weights.append(abs(float(margin)) + 1e-4)

        if (
            bool(head_signal_probe_consensus_enabled)
            and sync_global_state is not None
            and orientation_rule in {"sync_global_loss", "global_loss_sync", "sync_loss"}
        ):
            global_order = sync_global_state["global_order"]
            global_reverse = sync_global_state["global_reverse"]
            global_loss = loss_by_order.get(tuple(global_order))
            global_reverse_loss = loss_by_order.get(tuple(global_reverse))
            if global_loss is not None and global_reverse_loss is not None:
                prefix_weight = float(head_signal_probe_prefix_weight)
                full_weight = float(head_signal_probe_full_weight)
                global_selection_loss = (
                    prefix_weight * float(global_loss["prefix_loss"])
                    + full_weight * float(global_loss["full_loss"])
                )
                global_reverse_selection_loss = (
                    prefix_weight * float(global_reverse_loss["prefix_loss"])
                    + full_weight * float(global_reverse_loss["full_loss"])
                )
                global_margin = float(global_reverse_selection_loss - global_selection_loss)
                oriented_q = np.asarray(sync_global_state["q_matrix"], dtype=np.float64)
                if global_margin < 0.0:
                    oriented_q = -oriented_q
                global_selected_order = global_reverse if global_margin < 0.0 else global_order
                global_selected_original = _head_signal_order_to_original(global_selected_order)
                global_selected_tau = _head_signal_tau_to_l2r(global_selected_original)
                for row in sync_global_state["rows"]:
                    raw_order = [int(value) for value in row["raw_order_current"]]
                    raw_matrix = _head_signal_precedence_matrix(raw_order)
                    align = float((raw_matrix * oriented_q).sum())
                    consensus_order = list(reversed(raw_order)) if align < 0.0 else raw_order
                    consensus_original = _head_signal_order_to_original(consensus_order)
                    consensus_tau = _head_signal_tau_to_l2r(consensus_original)
                    row.update(
                        {
                            "consensus_rule": "sync_global_loss",
                            "consensus_alignment_margin": float(align),
                            "consensus_selected_reverse": bool(align < 0.0),
                            "consensus_tau_diagnostic": float(consensus_tau),
                            "consensus_abs_tau_diagnostic": abs(float(consensus_tau)),
                            "consensus_sign_diagnostic": int(_head_signal_sign(consensus_tau)),
                            "consensus_order_current": consensus_order,
                            "consensus_order_original_diagnostic": consensus_original,
                            "sync_global_order_current": global_order,
                            "sync_global_selected_order_current": global_selected_order,
                            "sync_global_selected_order_original_diagnostic": global_selected_original,
                            "sync_global_selected_tau_diagnostic": float(global_selected_tau),
                            "sync_global_loss": float(global_selection_loss),
                            "sync_global_reverse_loss": float(global_reverse_selection_loss),
                            "sync_global_margin_reverse_minus_forward": float(global_margin),
                            "sync_global_selected_reverse": bool(global_margin < 0.0),
                            "sync_global_eigval": float(sync_global_state["sync_eigval"]),
                            "sync_global_mean_abs_head_agreement": float(
                                sync_global_state["mean_abs_agreement"]
                            ),
                        }
                    )
        elif bool(head_signal_probe_consensus_enabled) and signed_precedence:
            for row in rows:
                consensus_idx = row.get("_head_signal_consensus_index", None)
                if "raw_order_current" not in row or consensus_idx is None:
                    continue
                consensus_idx = int(consensus_idx)
                if consensus_idx < 0 or consensus_idx >= len(signed_precedence):
                    continue
                if bool(head_signal_probe_consensus_leave_one_out) and len(signed_precedence) > 1:
                    denom = sum(
                        float(weight)
                        for j, weight in enumerate(signed_weights)
                        if j != consensus_idx
                    )
                    if denom <= 0.0:
                        denom = sum(float(weight) for weight in signed_weights)
                    q_matrix = sum(
                        float(signed_weights[j]) * signed_precedence[j]
                        for j in range(len(signed_precedence))
                        if j != consensus_idx
                    ) / float(denom)
                else:
                    denom = sum(float(weight) for weight in signed_weights)
                    q_matrix = sum(
                        float(weight) * matrix
                        for weight, matrix in zip(signed_weights, signed_precedence)
                    ) / float(max(denom, 1e-8))
                raw_order = [int(value) for value in row["raw_order_current"]]
                raw_matrix = _head_signal_precedence_matrix(raw_order)
                align = float((raw_matrix * q_matrix).sum())
                consensus_order = list(reversed(raw_order)) if align < 0.0 else raw_order
                consensus_original = _head_signal_order_to_original(consensus_order)
                consensus_tau = _head_signal_tau_to_l2r(consensus_original)
                row.update(
                    {
                        "consensus_alignment_margin": float(align),
                        "consensus_selected_reverse": bool(align < 0.0),
                        "consensus_tau_diagnostic": float(consensus_tau),
                        "consensus_abs_tau_diagnostic": abs(float(consensus_tau)),
                        "consensus_sign_diagnostic": int(_head_signal_sign(consensus_tau)),
                        "consensus_order_current": consensus_order,
                        "consensus_order_original_diagnostic": consensus_original,
                    }
                )

        if candidate_loss_profile_enabled:
            if orientation_rule in direct_loss_profile_rules:
                _head_signal_apply_candidate_loss_profile_selection(rows, loss_by_order)
            else:
                _head_signal_apply_candidate_loss_profile_consensus(rows, loss_by_order)

        _online_spectral_apply_try19_bridge(rows, iter_num, total_samples)

        if master_process:
            probe_dir = _head_signal_probe_dir()
            os.makedirs(probe_dir, exist_ok=True)
            history_path = os.path.join(probe_dir, "head_signal_probe_history.jsonl")
            with open(history_path, "a", encoding="utf-8") as handle:
                for row in rows:
                    for key in list(row.keys()):
                        if str(key).startswith("_head_signal_candidate_loss_profile") or key == "_head_signal_candidate_orders_current":
                            row.pop(key, None)
                    handle.write(json.dumps(_online_spectral_json_safe(row), ensure_ascii=False) + "\n")
            compact = []
            for row in rows:
                head_label = str(row.get("head_label", _head_signal_head_label(row.get("head", -1))))
                if "consensus_tau_diagnostic" in row:
                    compact.append(
                        f"L{row['layer']}H{head_label}:{row['consensus_tau_diagnostic']:+.3f}"
                    )
                elif "loss_oriented_tau_diagnostic" in row:
                    compact.append(
                        f"L{row['layer']}H{head_label}:{row['loss_oriented_tau_diagnostic']:+.3f}"
                    )
            print(
                f"head_signal_probe iter {int(iter_num)} "
                f"samples={int(total_samples)} " + " ".join(compact)
            )
    finally:
        if rng_state_cpu is not None:
            torch.random.set_rng_state(rng_state_cpu)
        if rng_state_cuda is not None:
            torch.cuda.set_rng_state_all(rng_state_cuda)
        if rng_state_np is not None:
            np.random.set_state(rng_state_np)
        if was_training:
            model.train()


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb_init_kwargs = {
        "project": wandb_project,
        "name": wandb_run_name,
        "config": config,
    }
    if str(wandb_run_id).strip():
        wandb_init_kwargs["id"] = str(wandb_run_id).strip()
        wandb_init_kwargs["resume"] = "allow"
    wandb.init(**wandb_init_kwargs)
    if hasattr(wandb, "define_metric"):
        wandb.define_metric("iter")
        wandb.define_metric("*", step_metric="iter")

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
latest_policy_train_stats = None

while True:
    _maybe_update_segment_schedule(iter_num)

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * float(param_group.get('lr_scale', 1.0))

    _head_asym_selector_update_if_due()

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            log_payload = {
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100,
            }
            if "eval_policy_name" in losses:
                log_payload["policy/eval_policy_name"] = losses["eval_policy_name"]
            if "main_eval_policy_name" in losses:
                log_payload["policy/main_eval_policy_name"] = losses["main_eval_policy_name"]
            if "train_main_eval_loss" in losses:
                log_payload["train/main_eval_loss"] = losses["train_main_eval_loss"]
            if "val_main_eval_loss" in losses:
                log_payload["val/main_eval_loss"] = losses["val_main_eval_loss"]
            if "val_l2r_loss" in losses:
                log_payload["val/l2r_loss"] = losses["val_l2r_loss"]
            if "val_origin_l2r_loss" in losses:
                log_payload["val/origin_l2r_loss"] = losses["val_origin_l2r_loss"]
            if latest_policy_train_stats is not None:
                optional_policy_fields = {
                    "policy/segment_guided_ratio": "segment_guided_ratio",
                    "policy/segment_library_size": "segment_library_size",
                    "policy/attn_mlp_active": "attn_mlp_policy_active",
                    "policy/attn_mlp_prob": "attn_mlp_policy_prob",
                    "policy/attn_mlp_updates": "attn_mlp_policy_updates",
                    "policy/attn_mlp_attention_updates": "attn_mlp_policy_attention_updates",
                    "policy/attn_mlp_attention_batches_per_update": "attn_mlp_policy_attention_batches_per_update",
                    "policy/attn_mlp_attention_matrix_buffer": "attn_mlp_policy_attention_matrix_buffer",
                    "policy/attn_mlp_fallback_count": "attn_mlp_policy_fallback_count",
                    "policy/attn_mlp_last_update_iter": "attn_mlp_policy_last_update_iter",
                    "policy/attn_mlp_last_attention_iter": "attn_mlp_policy_last_attention_iter",
                    "policy/attn_mlp_cached": "attn_mlp_policy_cached",
                    "policy/attn_mlp_refreshed": "attn_mlp_policy_refreshed",
                    "policy/attn_mlp_updates_frozen": "attn_mlp_policy_updates_frozen",
                    "policy/attn_mlp_current_lr": "attn_mlp_policy_current_lr",
                    "policy/attn_mlp_loss_stop_enabled": "attn_mlp_policy_loss_stop_enabled",
                    "policy/attn_mlp_loss_stop_triggered": "attn_mlp_policy_loss_stop_triggered",
                    "policy/attn_mlp_loss_stop_iter": "attn_mlp_policy_loss_stop_iter",
                    "policy/attn_mlp_loss_stop_bad_steps": "attn_mlp_policy_loss_stop_bad_steps",
                    "policy/attn_mlp_loss_stop_checks": "attn_mlp_policy_loss_stop_checks",
                    "policy/attn_mlp_loss_stop_best": "attn_mlp_policy_loss_stop_best",
                    "policy/attn_mlp_loss_stop_best_iter": "attn_mlp_policy_loss_stop_best_iter",
                    "policy/attn_mlp_loss_stop_last_metric": "attn_mlp_policy_loss_stop_last_metric",
                    "policy/attn_mlp_A_ema_mean": "attn_mlp_A_ema_mean",
                    "policy/attn_mlp_A_ema_std": "attn_mlp_A_ema_std",
                    "policy/attn_mlp_logits_mean": "attn_mlp_logits_mean",
                    "policy/attn_mlp_logits_std": "attn_mlp_logits_std",
                    "policy/attn_mlp_logits_ema_enabled": "attn_mlp_policy_logits_ema_enabled",
                    "policy/attn_mlp_logits_ema_available": "attn_mlp_policy_logits_ema_available",
                    "policy/attn_mlp_logits_ema_decay": "attn_mlp_policy_logits_ema_decay",
                    "policy/attn_mlp_attention_ema_enabled": "attn_mlp_attention_ema_enabled",
                    "policy/attn_mlp_logits_ema_mean": "attn_mlp_logits_ema_mean",
                    "policy/attn_mlp_logits_ema_std": "attn_mlp_logits_ema_std",
                    "policy/attn_mlp_logits_ema_min": "attn_mlp_logits_ema_min",
                    "policy/attn_mlp_logits_ema_max": "attn_mlp_logits_ema_max",
                    "policy/attn_mlp_teacher_diag_enabled": "attn_mlp_policy_teacher_diag_enabled",
                    "policy/attn_mlp_teacher_diag_interval": "attn_mlp_policy_teacher_diag_interval",
                    "policy/attn_mlp_teacher_diag_error": "attn_mlp_teacher_diag_error",
                    "policy/attn_mlp_teacher_diag_selected_reverse": "attn_mlp_teacher_diag_selected_reverse",
                    "policy/attn_mlp_teacher_diag_mse_raw": "attn_mlp_teacher_diag_mse_raw",
                    "policy/attn_mlp_teacher_diag_mse_reverse": "attn_mlp_teacher_diag_mse_reverse",
                    "policy/attn_mlp_teacher_diag_mse_match": "attn_mlp_teacher_diag_mse_match",
                    "policy/attn_mlp_teacher_diag_mae_match": "attn_mlp_teacher_diag_mae_match",
                    "policy/attn_mlp_teacher_diag_pearson_match": "attn_mlp_teacher_diag_pearson_match",
                    "policy/attn_mlp_teacher_diag_score_pair_tau_match": "attn_mlp_teacher_diag_score_pair_tau_match",
                    "policy/attn_mlp_teacher_diag_cached_order_tau_match": "attn_mlp_teacher_diag_cached_order_tau_match",
                    "policy/attn_mlp_teacher_diag_raw_order_tau_match": "attn_mlp_teacher_diag_raw_order_tau_match",
                    "policy/attn_mlp_teacher_diag_order_logits_order_tau_match": "attn_mlp_teacher_diag_order_logits_order_tau_match",
                    "policy/attn_mlp_teacher_diag_mlp_cached_tau_original": "attn_mlp_teacher_diag_mlp_cached_tau_original_diagnostic",
                    "policy/attn_mlp_teacher_diag_mlp_raw_tau_original": "attn_mlp_teacher_diag_mlp_raw_tau_original_diagnostic",
                    "policy/attn_mlp_teacher_diag_teacher_raw_tau_original": "attn_mlp_teacher_diag_teacher_raw_tau_original_diagnostic",
                    "policy/attn_mlp_teacher_diag_teacher_reverse_tau_original": "attn_mlp_teacher_diag_teacher_reverse_tau_original_diagnostic",
                    "policy/attn_mlp_teacher_diag_teacher_match_tau_original": "attn_mlp_teacher_diag_teacher_match_tau_original_diagnostic",
                    "policy/attn_mlp_teacher_diag_spectral_gap": "attn_mlp_teacher_diag_spectral_gap",
                    "policy/attn_mlp_teacher_diag_affinity_sum": "attn_mlp_teacher_diag_affinity_sum",
                    "policy/attn_mlp_teacher_diag_affinity_density": "attn_mlp_teacher_diag_affinity_density",
                    "policy/attn_mlp_train_enabled": "attn_mlp_train_enabled",
                    "policy/attn_mlp_train_loss": "attn_mlp_train_loss",
                    "policy/attn_mlp_train_lr": "attn_mlp_train_lr",
                    "policy/attn_mlp_train_skipped_loss_stop": "attn_mlp_train_skipped_loss_stop",
                    "policy/attn_mlp_train_shadow_mse_active": "attn_mlp_train_shadow_mse_active",
                    "policy/attn_mlp_train_shadow_mse_eval_after_stop": "attn_mlp_train_shadow_mse_eval_after_stop",
                    "policy/attn_mlp_train_shadow_mse_frozen": "attn_mlp_train_shadow_mse_frozen",
                    "policy/attn_mlp_train_shadow_mse_error": "attn_mlp_train_shadow_mse_error",
                    "policy/attn_mlp_train_shadow_mse_waiting_for_items": "attn_mlp_train_shadow_mse_waiting_for_items",
                    "policy/attn_mlp_train_shadow_mse_items": "attn_mlp_train_shadow_mse_items",
                    "policy/attn_mlp_train_shadow_mse_train_items": "attn_mlp_train_shadow_mse_train_items",
                    "policy/attn_mlp_train_shadow_mse_val_items": "attn_mlp_train_shadow_mse_val_items",
                    "policy/attn_mlp_train_shadow_mse_train_count": "attn_mlp_train_shadow_mse_train_count",
                    "policy/attn_mlp_train_shadow_mse_val_count": "attn_mlp_train_shadow_mse_val_count",
                    "policy/attn_mlp_train_shadow_mse_batches_per_item": "attn_mlp_train_shadow_mse_batches_per_item",
                    "policy/attn_mlp_train_shadow_mse_train_items_per_update": "attn_mlp_train_shadow_mse_train_items_per_update",
                    "policy/attn_mlp_train_shadow_mse_val_items_per_update": "attn_mlp_train_shadow_mse_val_items_per_update",
                    "policy/attn_mlp_train_shadow_mse_train_samples_per_step": "attn_mlp_train_shadow_mse_train_samples_per_step",
                    "policy/attn_mlp_train_shadow_mse_val_samples_per_step": "attn_mlp_train_shadow_mse_val_samples_per_step",
                    "policy/attn_mlp_train_shadow_mse_orientation_x_mode_last_step": "attn_mlp_train_shadow_mse_orientation_x_mode_last_step",
                    "policy/attn_mlp_train_shadow_mse_batch_buffer": "attn_mlp_train_shadow_mse_batch_buffer",
                    "policy/attn_mlp_train_shadow_mse_item_buffer": "attn_mlp_train_shadow_mse_item_buffer",
                    "policy/attn_mlp_train_shadow_mse_train_loss": "attn_mlp_train_shadow_mse_train_loss",
                    "policy/attn_mlp_train_shadow_mse_train_mae": "attn_mlp_train_shadow_mse_train_mae",
                    "policy/attn_mlp_train_shadow_mse_train_pearson": "attn_mlp_train_shadow_mse_train_pearson",
                    "policy/attn_mlp_train_shadow_mse_train_score_pair_tau": "attn_mlp_train_shadow_mse_train_score_pair_tau",
                    "policy/attn_mlp_train_shadow_mse_train_order_pair_tau": "attn_mlp_train_shadow_mse_train_order_pair_tau",
                    "policy/attn_mlp_train_shadow_mse_val_loss": "attn_mlp_train_shadow_mse_val_loss",
                    "policy/attn_mlp_train_shadow_mse_val_mae": "attn_mlp_train_shadow_mse_val_mae",
                    "policy/attn_mlp_train_shadow_mse_val_pearson": "attn_mlp_train_shadow_mse_val_pearson",
                    "policy/attn_mlp_train_shadow_mse_val_score_pair_tau": "attn_mlp_train_shadow_mse_val_score_pair_tau",
                    "policy/attn_mlp_train_shadow_mse_val_order_pair_tau": "attn_mlp_train_shadow_mse_val_order_pair_tau",
                    "policy/attn_mlp_train_shadow_mse_selected_reverse_frac": "attn_mlp_train_shadow_mse_selected_reverse_frac",
                    "policy/attn_mlp_train_shadow_mse_raw_weighted_loss": "attn_mlp_train_shadow_mse_raw_weighted_loss",
                    "policy/attn_mlp_train_shadow_mse_reverse_weighted_loss": "attn_mlp_train_shadow_mse_reverse_weighted_loss",
                    "policy/attn_mlp_train_shadow_mse_loss_gap_raw_minus_reverse": "attn_mlp_train_shadow_mse_loss_gap_raw_minus_reverse",
                    "policy/attn_mlp_train_pair_loss": "attn_mlp_train_pair_loss",
                    "policy/attn_mlp_train_axis_loss": "attn_mlp_train_axis_loss",
                    "policy/attn_mlp_train_pair_acc": "attn_mlp_train_pair_acc",
                    "policy/attn_mlp_train_pair_weight_mean": "attn_mlp_train_pair_weight_mean",
                    "policy/attn_mlp_train_pair_weight": "attn_mlp_train_pair_weight",
                    "policy/attn_mlp_train_axis_weight": "attn_mlp_train_axis_weight",
                    "policy/attn_mlp_train_target_axis_std": "attn_mlp_train_target_axis_std",
                    "policy/attn_mlp_train_logit_l2": "attn_mlp_train_logit_l2",
                    "policy/attn_mlp_train_logit_std": "attn_mlp_train_logit_std",
                    "policy/attn_mlp_train_logit_entropy": "attn_mlp_train_logit_entropy",
                    "policy/attn_mlp_train_directed_ribbon_loss": "attn_mlp_train_directed_ribbon_loss",
                    "policy/attn_mlp_train_directed_ribbon_side_loss": "attn_mlp_train_directed_ribbon_side_loss",
                    "policy/attn_mlp_train_directed_ribbon_band_loss": "attn_mlp_train_directed_ribbon_band_loss",
                    "policy/attn_mlp_train_directed_ribbon_flow_sign": "attn_mlp_train_directed_ribbon_flow_sign",
                    "policy/attn_mlp_train_directed_ribbon_active_edges": "attn_mlp_train_directed_ribbon_active_edges",
                    "policy/attn_mlp_train_directed_ribbon_weight_mean": "attn_mlp_train_directed_ribbon_weight_mean",
                    "policy/attn_mlp_train_axis_profile_axis_loss": "attn_mlp_train_axis_profile_axis_loss",
                    "policy/attn_mlp_train_axis_profile_dir_loss": "attn_mlp_train_axis_profile_dir_loss",
                    "policy/attn_mlp_train_axis_profile_axis_weight": "attn_mlp_train_axis_profile_axis_weight",
                    "policy/attn_mlp_train_axis_profile_dir_weight": "attn_mlp_train_axis_profile_dir_weight",
                    "policy/attn_mlp_train_axis_profile_corr": "attn_mlp_train_axis_profile_corr",
                    "policy/attn_mlp_train_axis_profile_abs_corr": "attn_mlp_train_axis_profile_abs_corr",
                    "policy/attn_mlp_train_axis_profile_pair_acc": "attn_mlp_train_axis_profile_pair_acc",
                    "policy/attn_mlp_train_axis_profile_pair_count": "attn_mlp_train_axis_profile_pair_count",
                    "policy/attn_mlp_train_axis_profile_q_abs_mean": "attn_mlp_train_axis_profile_q_abs_mean",
                    "policy/attn_mlp_train_axis_profile_q_abs_max": "attn_mlp_train_axis_profile_q_abs_max",
                    "policy/attn_mlp_train_axis_profile_dir_margin_mean": "attn_mlp_train_axis_profile_dir_margin_mean",
                    "policy/attn_mlp_train_axis_profile_dir_margin_abs_mean": "attn_mlp_train_axis_profile_dir_margin_abs_mean",
                    "policy/attn_mlp_train_axis_profile_dir_accept_rate": "attn_mlp_train_axis_profile_dir_accept_rate",
                    "policy/attn_mlp_train_axis_profile_dir_forward_better_frac": "attn_mlp_train_axis_profile_dir_forward_better_frac",
                    "policy/attn_mlp_train_axis_profile_dir_map_score": "attn_mlp_train_axis_profile_dir_map_score",
                    "policy/attn_mlp_train_axis_profile_dir_reverse_score": "attn_mlp_train_axis_profile_dir_reverse_score",
                    "policy/attn_mlp_train_axis_profile_dir_pair_acc": "attn_mlp_train_axis_profile_dir_pair_acc",
                    "policy/attn_mlp_train_pg_loss": "attn_mlp_train_pg_loss",
                    "policy/attn_mlp_train_move_loss": "attn_mlp_train_move_loss",
                    "policy/attn_mlp_train_move_acc": "attn_mlp_train_move_acc",
                    "policy/attn_mlp_train_move_pairs": "attn_mlp_train_move_pairs",
                    "policy/attn_mlp_train_move_delta_mean": "attn_mlp_train_move_delta_mean",
                    "policy/attn_mlp_train_move_delta_abs_mean": "attn_mlp_train_move_delta_abs_mean",
                    "policy/attn_mlp_train_move_weight_mean": "attn_mlp_train_move_weight_mean",
                    "policy/attn_mlp_train_move_weight": "attn_mlp_train_move_weight",
                    "policy/attn_mlp_train_move_base_full_loss": "attn_mlp_train_move_base_full_loss",
                    "policy/attn_mlp_train_move_swap_full_loss": "attn_mlp_train_move_swap_full_loss",
                    "policy/attn_mlp_train_move_base_score": "attn_mlp_train_move_base_score",
                    "policy/attn_mlp_train_move_best_score": "attn_mlp_train_move_best_score",
                    "policy/attn_mlp_train_attn_pair_weight": "attn_mlp_train_attn_pair_weight",
                    "policy/attn_mlp_train_attn_pair_loss": "attn_mlp_train_attn_pair_loss",
                    "policy/attn_mlp_train_attn_pair_acc": "attn_mlp_train_attn_pair_acc",
                    "policy/attn_mlp_train_attn_pair_count": "attn_mlp_train_attn_pair_count",
                    "policy/attn_mlp_train_attn_pair_weight_mean": "attn_mlp_train_attn_pair_weight_mean",
                    "policy/attn_mlp_train_attn_close_weight": "attn_mlp_train_attn_close_weight",
                    "policy/attn_mlp_train_attn_close_loss": "attn_mlp_train_attn_close_loss",
                    "policy/attn_mlp_train_attn_close_dist_mean": "attn_mlp_train_attn_close_dist_mean",
                    "policy/attn_mlp_train_head_profile_weight": "attn_mlp_train_head_profile_weight",
                    "policy/attn_mlp_train_head_profile_loss": "attn_mlp_train_head_profile_loss",
                    "policy/attn_mlp_train_head_profile_acc": "attn_mlp_train_head_profile_acc",
                    "policy/attn_mlp_train_head_profile_pairs": "attn_mlp_train_head_profile_pairs",
                    "policy/attn_mlp_train_head_profile_pair_weight_mean": "attn_mlp_train_head_profile_pair_weight_mean",
                    "policy/attn_mlp_train_head_profile_refreshed": "attn_mlp_train_head_profile_refreshed",
                    "policy/attn_mlp_train_head_profile_last_iter": "attn_mlp_train_head_profile_last_iter",
                    "policy/attn_mlp_train_head_profile_heads": "attn_mlp_train_head_profile_heads",
                    "policy/attn_mlp_train_head_profile_samples": "attn_mlp_train_head_profile_samples",
                    "policy/attn_mlp_train_head_profile_score_gap_mean": "attn_mlp_train_head_profile_score_gap_mean",
                    "policy/attn_mlp_train_head_profile_alignment_mean": "attn_mlp_train_head_profile_alignment_mean",
                    "policy/attn_mlp_train_head_profile_fallback_frac": "attn_mlp_train_head_profile_fallback_frac",
                    "policy/attn_mlp_train_head_profile_q_abs_mean": "attn_mlp_train_head_profile_q_abs_mean",
                    "policy/attn_mlp_train_head_profile_q_abs_max": "attn_mlp_train_head_profile_q_abs_max",
                    "policy/attn_mlp_train_head_profile_tau_diagnostic": "attn_mlp_train_head_profile_tau_diagnostic",
                    "policy/attn_mlp_train_head_profile_abs_tau_diagnostic": "attn_mlp_train_head_profile_abs_tau_diagnostic",
                    "policy/attn_mlp_train_head_profile_head_tau_mean_diagnostic": "attn_mlp_train_head_profile_head_tau_mean_diagnostic",
                    "policy/attn_mlp_train_head_profile_gate_enabled": "attn_mlp_train_head_profile_gate_enabled",
                    "policy/attn_mlp_train_head_profile_gate_checked": "attn_mlp_train_head_profile_gate_checked",
                    "policy/attn_mlp_train_head_profile_gate_accepted": "attn_mlp_train_head_profile_gate_accepted",
                    "policy/attn_mlp_train_head_profile_gate_alignment": "attn_mlp_train_head_profile_gate_alignment",
                    "policy/attn_mlp_train_head_profile_gate_new_score": "attn_mlp_train_head_profile_gate_new_score",
                    "policy/attn_mlp_train_head_profile_gate_cached_score": "attn_mlp_train_head_profile_gate_cached_score",
                    "policy/attn_mlp_train_head_profile_gate_best_score": "attn_mlp_train_head_profile_gate_best_score",
                    "policy/attn_mlp_train_head_profile_gate_selected_score": "attn_mlp_train_head_profile_gate_selected_score",
                    "policy/attn_mlp_train_head_profile_gate_score_delta": "attn_mlp_train_head_profile_gate_score_delta",
                    "policy/attn_mlp_train_head_profile_gate_margin": "attn_mlp_train_head_profile_gate_margin",
                    "policy/attn_mlp_train_head_profile_gate_selected_source_id": "attn_mlp_train_head_profile_gate_selected_source_id",
                    "policy/attn_mlp_train_head_profile_gate_selected_is_new": "attn_mlp_train_head_profile_gate_selected_is_new",
                    "policy/attn_mlp_train_head_profile_gate_selected_is_cached": "attn_mlp_train_head_profile_gate_selected_is_cached",
                    "policy/attn_mlp_train_head_profile_gate_selected_is_best": "attn_mlp_train_head_profile_gate_selected_is_best",
                    "policy/attn_mlp_train_head_profile_best_memory_enabled": "attn_mlp_train_head_profile_best_memory_enabled",
                    "policy/attn_mlp_train_head_profile_best_memory_available": "attn_mlp_train_head_profile_best_memory_available",
                    "policy/attn_mlp_train_head_profile_best_memory_updated": "attn_mlp_train_head_profile_best_memory_updated",
                    "policy/attn_mlp_train_head_profile_best_memory_iter": "attn_mlp_train_head_profile_best_memory_iter",
                    "policy/attn_mlp_train_std_floor": "attn_mlp_train_std_floor",
                    "policy/attn_mlp_train_entropy_floor": "attn_mlp_train_entropy_floor",
                    "policy/attn_mlp_train_entropy_ceiling": "attn_mlp_train_entropy_ceiling",
                    "policy/attn_mlp_train_grad_norm": "attn_mlp_train_grad_norm",
                    "policy/attn_mlp_train_sample_minus_random_mean": "attn_mlp_train_sample_minus_random_mean",
                    "policy/attn_mlp_train_prefix_minus_random_mean": "attn_mlp_train_prefix_minus_random_mean",
                    "policy/attn_mlp_train_full_advantage_mean": "attn_mlp_train_full_advantage_mean",
                    "policy/attn_mlp_train_prefix_advantage_mean": "attn_mlp_train_prefix_advantage_mean",
                    "policy/attn_mlp_train_sample_full_loss": "attn_mlp_train_sample_full_loss",
                    "policy/attn_mlp_train_random_full_loss": "attn_mlp_train_random_full_loss",
                    "policy/attn_mlp_train_sample_prefix_loss": "attn_mlp_train_sample_prefix_loss",
                    "policy/attn_mlp_train_random_prefix_loss": "attn_mlp_train_random_prefix_loss",
                    "policy/attn_mlp_train_reward_scale": "attn_mlp_train_reward_scale",
                    "policy/attn_mlp_train_pl_logp": "attn_mlp_train_pl_logp",
                    "policy/attn_mlp_train_nll_states": "attn_mlp_train_nll_states",
                    "policy/attn_mlp_train_sampled_orders_per_state": "attn_mlp_train_sampled_orders_per_state",
                    "policy/attn_mlp_train_random_baseline_orders": "attn_mlp_train_random_baseline_orders",
                    "policy/online_spectral_prob": "online_spectral_policy_prob",
                    "policy/online_spectral_updates": "online_spectral_policy_updates",
                    "policy/online_spectral_distribution_updates": "online_spectral_distribution_updates",
                    "policy/online_spectral_attention_updates": "online_spectral_policy_attention_updates",
                    "policy/online_spectral_attention_samples": "online_spectral_policy_last_attention_samples",
                    "policy/online_spectral_random_count": "online_spectral_policy_random_count",
                    "policy/online_spectral_fallback_count": "online_spectral_policy_fallback_count",
                    "policy/online_spectral_late_l2r_prob": "online_spectral_late_l2r_prob",
                    "policy/online_spectral_late_l2r_count": "online_spectral_late_l2r_count",
                    "policy/online_spectral_failed_updates": "online_spectral_policy_failed_updates",
                    "policy/online_spectral_cached": "online_spectral_policy_cached",
                    "policy/online_spectral_score": "online_spectral_score",
                    "policy/online_spectral_attention_spectral_score": "online_spectral_attention_spectral_score",
                    "policy/online_spectral_band_quality": "online_spectral_band_quality",
                    "policy/online_spectral_adjacency_path_score": "online_spectral_adjacency_path_score",
                    "policy/online_spectral_directed_path_score": "online_spectral_directed_path_score",
                    "policy/online_spectral_loss_rerank_prefix_loss": "online_spectral_loss_rerank_prefix_loss",
                    "policy/online_spectral_loss_rerank_full_loss": "online_spectral_loss_rerank_full_loss",
                    "policy/online_spectral_loss_rerank_score": "online_spectral_loss_rerank_score",
                    "policy/online_spectral_priority_mean": "online_spectral_priority_mean",
                    "policy/online_spectral_priority_std": "online_spectral_priority_std",
                    "policy/online_spectral_priority_entropy": "online_spectral_priority_entropy",
                    "policy/online_spectral_top_m_used": "online_spectral_top_m_used",
                    "policy/online_spectral_candidate_weight_entropy": "online_spectral_candidate_weight_entropy",
                    "policy/online_spectral_candidate_top_weight": "online_spectral_candidate_top_weight",
                    "policy/online_spectral_priority_update_std": "online_spectral_priority_update_std",
                    "policy/online_spectral_priority_update_range": "online_spectral_priority_update_range",
                    "policy/online_spectral_priority_update_effectively_constant": "online_spectral_priority_update_effectively_constant",
                    "policy/online_spectral_reverse_pair_weight_mass": "online_spectral_reverse_pair_weight_mass",
                    "policy/online_spectral_random_mix_prob": "online_spectral_random_mix_prob",
                    "policy/online_spectral_sample_temperature": "online_spectral_sample_temperature",
                    "policy/online_spectral_temperature_sampling_stage_code": (
                        "online_spectral_temperature_sampling_stage_code"
                    ),
                    "policy/online_spectral_temperature_sampling_current_temperature": (
                        "online_spectral_temperature_sampling_current_temperature"
                    ),
                    "policy/online_spectral_temperature_sampling_start_temperature": (
                        "online_spectral_temperature_sampling_start_temperature"
                    ),
                    "policy/online_spectral_temperature_sampling_end_temperature": (
                        "online_spectral_temperature_sampling_end_temperature"
                    ),
                    "policy/online_spectral_update_start_iter": "online_spectral_policy_update_start_iter",
                    "policy/online_spectral_update_stop_iter": "online_spectral_policy_update_stop_iter",
                    "policy/online_spectral_updates_frozen": "online_spectral_policy_updates_frozen",
                    "policy/online_spectral_freeze_to_map_order_after_stop": "online_spectral_freeze_to_map_order_after_stop",
                    "policy/online_spectral_map_order_changed_kendall": "online_spectral_map_order_changed_kendall",
                    "policy/online_spectral_try19_bridge_enabled": "online_spectral_try19_bridge_enabled",
                    "policy/online_spectral_try19_bridge_updates": "online_spectral_try19_bridge_updates",
                    "policy/online_spectral_try19_bridge_candidate_count": "online_spectral_try19_bridge_candidate_count",
                    "policy/online_spectral_try19_bridge_score_gap_mean": "online_spectral_try19_bridge_score_gap_mean",
                    "policy/online_spectral_try19_bridge_score_gap_min": "online_spectral_try19_bridge_score_gap_min",
                    "policy/online_spectral_try19_bridge_fallback_frac": "online_spectral_try19_bridge_fallback_frac",
                    "policy/online_spectral_try19_bridge_selected_tau_diagnostic": "online_spectral_try19_bridge_selected_tau_diagnostic",
                    "policy/online_spectral_try19_bridge_selected_abs_tau_diagnostic": "online_spectral_try19_bridge_selected_abs_tau_diagnostic",
                }
                for wandb_key, stats_key in optional_policy_fields.items():
                    if stats_key in latest_policy_train_stats:
                        log_payload[wandb_key] = latest_policy_train_stats[stats_key]
                if "attn_mlp_policy_name" in latest_policy_train_stats:
                    log_payload["policy/attn_mlp_policy_name"] = latest_policy_train_stats["attn_mlp_policy_name"]
                if "online_spectral_policy_name" in latest_policy_train_stats:
                    log_payload["policy/online_spectral_policy_name"] = latest_policy_train_stats["online_spectral_policy_name"]
            if eval_generate_step_loss_log and train_stage == 'standard' and aogpt_train_mode == 'Random':
                ar_curve, random_curve, original_l2r_curve = estimate_eval_generate_step_block_loss_curves()
                figure = build_generate_step_block_loss_figure(
                    ar_curve,
                    random_curve,
                    original_l2r_curve=original_l2r_curve,
                )
                latest_plot_path = save_figure_to_out_dir(
                    figure,
                    eval_generate_step_loss_filename,
                )
                log_payload["val/generate_step_block_loss_mean_ar"] = float(np.mean(ar_curve))
                log_payload["val/generate_step_block_loss_mean_random"] = float(np.mean(random_curve))
                if original_l2r_curve is not None:
                    log_payload["val/generate_step_block_loss_mean_original_l2r"] = float(np.mean(original_l2r_curve))
                log_payload["val/generate_step_block_loss_plot"] = wandb.Image(figure)
                log_payload["val/generate_step_block_loss_plot_path"] = latest_plot_path
                print(f"saved generate_step_block_loss_plot to {latest_plot_path}")
                import matplotlib.pyplot as plt
                plt.close(figure)
            if eval_kendall_distance_log:
                kendall_payload = estimate_sampled_order_kendall_distance_stats()
                kendall_distance = np.asarray(kendall_payload["origin_kendall_distance"], dtype=np.float64)
                kendall_distance_normalized = np.asarray(
                    kendall_payload["origin_normalized_distance"],
                    dtype=np.float64,
                )
                sample_mode = str(kendall_payload.get("sample_mode", "sampled_orders"))
                log_payload["val/origin_kendall_distance_sample_mode"] = sample_mode
                log_payload["val/origin_kendall_distance_sampled_mean"] = float(
                    kendall_distance_normalized.mean()
                )
                log_payload["val/origin_kendall_distance_sampled_raw_mean"] = float(
                    kendall_distance.mean()
                )
                log_payload["val/origin_kendall_distance_sampled_num_orders"] = int(
                    kendall_payload.get("num_orders_used", 0)
                )
                kendall_tau_values = np.asarray(kendall_payload["origin_kendall_tau"], dtype=np.float64)
                log_payload["val/origin_kendall_tau_mean"] = float(kendall_tau_values.mean())
                if "origin_kendall_tau_map_order" in kendall_payload:
                    log_payload["val/origin_kendall_tau_map_order"] = float(
                        kendall_payload["origin_kendall_tau_map_order"]
                    )
                if bool(kendall_payload.get("is_fixed_policy_order", False)):
                    log_payload["val/origin_kendall_tau_cached_order"] = float(kendall_tau_values[0])
                    log_payload["val/origin_kendall_distance_cached_order"] = float(kendall_distance[0])
                    log_payload["val/origin_kendall_distance_normalized_cached_order"] = float(
                        kendall_distance_normalized[0]
                    )
                print(
                    "computed origin_kendall_distance_sampled_mean="
                    f"{log_payload['val/origin_kendall_distance_sampled_mean']:.4f} "
                    f"using {log_payload['val/origin_kendall_distance_sampled_num_orders']} "
                    f"orders ({sample_mode})"
                )
            wandb.log(log_payload)
        val_loss_for_ckpt = losses['val']
        if val_loss_for_ckpt < best_val_loss or always_save_checkpoint:
            best_val_loss = val_loss_for_ckpt
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                if permute_data and fixed_block_perm is not None:
                    checkpoint['data_permutation'] = {
                        'permute_mode': permute_mode,
                        'permute_seed': int(permute_seed),
                        'block_perm': [int(v) for v in fixed_block_perm.tolist()],
                        'inverse_block_perm': [int(v) for v in inverse_block_perm.tolist()],
                        'block_order_layout': str(block_order_layout),
                        'image_size': int(image_size),
                        'image_block_size': int(image_block_size),
                        'image_block_height': int(image_block_height),
                        'image_block_width': int(image_block_width),
                    }
                attn_mlp_state = _attn_mlp_checkpoint_state()
                if attn_mlp_state is not None:
                    checkpoint['attn_mlp_policy_state'] = attn_mlp_state
                online_spectral_policy_state = _online_spectral_policy_checkpoint_state()
                if online_spectral_policy_state is not None:
                    checkpoint['online_spectral_policy_state'] = online_spectral_policy_state
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
                iter_ckpt_path = _save_iteration_checkpoint(checkpoint, iter_num)
                if iter_ckpt_path is not None:
                    print(f"saving iteration checkpoint to {iter_ckpt_path}")
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    attn_mlp_policy_matrices = []
    attn_mlp_policy_probe_batches = []
    attn_mlp_policy_shadow_mse_records = []
    online_spectral_policy_matrices = []
    collect_attn_mlp_attention = _attn_mlp_policy_should_collect_attention(iter_num)
    attn_mlp_policy_needs_token_loss = collect_attn_mlp_attention and str(attn_mlp_policy_feature_mode or "attention").lower() not in {
        "attention",
        "matrix",
        "single",
        "attention_direct",
        "direct_attention",
        "attention_only_direct",
    }
    collect_online_spectral_policy_attention = _online_spectral_policy_should_collect_train_step_attention(iter_num)
    collect_policy_attention = collect_attn_mlp_attention or collect_online_spectral_policy_attention
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            forward_outputs, active_policy_name, order_info = _forward_with_active_training_policy(
                X,
                return_token_loss=(_online_stats_enabled() or bool(attn_mlp_policy_needs_token_loss)),
                return_order_info=(_online_stats_enabled() or bool(collect_attn_mlp_attention)),
                return_attentions=collect_policy_attention,
            )
            logits, loss = forward_outputs[:2]
            token_losses_for_stats = forward_outputs[2] if _online_stats_enabled() and len(forward_outputs) > 2 else None
            _update_online_pair_stats(order_info, token_losses_for_stats)
            if collect_attn_mlp_attention and order_info is not None:
                policy_input = _attn_mlp_policy_input_from_outputs(
                    forward_outputs,
                    order_info.get("block_orders"),
                )
                if policy_input is not None:
                    attn_mlp_policy_matrices.append(policy_input)
                    if (
                        _attn_mlp_policy_shadow_mse_train_active(iter_num)
                        or _attn_mlp_policy_shadow_mse_eval_after_stop_active(iter_num)
                    ):
                        shadow_record = _attn_mlp_shadow_mse_record_from_outputs(
                            forward_outputs,
                            order_info.get("block_orders"),
                            X,
                        )
                        if shadow_record is not None:
                            attn_mlp_policy_shadow_mse_records.append(shadow_record)
                    if str(attn_mlp_policy_train_loss or "none").lower() in {
                        "sampled_nll_pg",
                        "full_nll_pg",
                        "nll_pg",
                        "axis_profile_simple",
                        "attention_axis_profile",
                        "direct_axis_profile",
                        "sampled_nll_pg_move_pref",
                        "nll_pg_move_pref",
                        "move_pref_pg",
                    }:
                        max_probe_batches = max(1, int(attn_mlp_policy_nll_states_per_update))
                        if len(attn_mlp_policy_probe_batches) < max_probe_batches:
                            attn_mlp_policy_probe_batches.append(X.detach().clone())
            if collect_online_spectral_policy_attention and order_info is not None:
                spectral_matrix = _online_spectral_policy_attention_matrix_from_outputs(
                    forward_outputs,
                    order_info.get("block_orders"),
                )
                if spectral_matrix is not None:
                    online_spectral_policy_matrices.append((spectral_matrix, int(X.size(0))))
            if active_policy_name == "segment_guided_random":
                latest_policy_train_stats = {
                    "segment_guided_ratio": float(segment_guided_ratio),
                    "segment_library_size": float(len(segment_library)),
                }
            elif str(active_policy_name).startswith("AttnMLP"):
                _attn_mlp_refresh_latest_stats(policy_refreshed=False)
                latest_policy_train_stats = dict(attn_mlp_policy_latest_stats or {})
                latest_policy_train_stats["attn_mlp_policy_name"] = str(active_policy_name)
            elif str(active_policy_name).startswith("OnlineSpectral"):
                _online_spectral_policy_refresh_latest_stats(policy_refreshed=False)
                latest_policy_train_stats = dict(online_spectral_policy_latest_stats or {})
                latest_policy_train_stats["online_spectral_policy_name"] = str(active_policy_name)
            else:
                latest_policy_train_stats = None
            loss = loss / gradient_accumulation_steps
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)
    if attn_mlp_policy_matrices:
        policy_matrix = _attn_mlp_policy_matrix_for_update(attn_mlp_policy_matrices)
        shadow_mse_items = _attn_mlp_shadow_mse_items_for_update(
            attn_mlp_policy_shadow_mse_records,
            emit=policy_matrix is not None,
        )
        if policy_matrix is not None:
            attn_mlp_policy_shadow_mse_items_current = shadow_mse_items if shadow_mse_items else None
            try:
                _attn_mlp_update_policy_from_matrix(
                    policy_matrix,
                    iter_num,
                    probe_batches=attn_mlp_policy_probe_batches,
                )
            finally:
                attn_mlp_policy_shadow_mse_items_current = None
            if latest_policy_train_stats is not None and str(latest_policy_train_stats.get("attn_mlp_policy_name", "")).startswith("AttnMLP"):
                policy_name_for_stats = latest_policy_train_stats["attn_mlp_policy_name"]
                latest_policy_train_stats = dict(attn_mlp_policy_latest_stats or {})
                latest_policy_train_stats["attn_mlp_policy_name"] = policy_name_for_stats
            elif attn_mlp_policy_latest_stats is not None:
                latest_policy_train_stats = dict(attn_mlp_policy_latest_stats or {})
                latest_policy_train_stats["attn_mlp_policy_name"] = "AttnMLPShadow"
        else:
            _attn_mlp_refresh_latest_stats(policy_refreshed=False)
            latest_policy_train_stats = dict(attn_mlp_policy_latest_stats or {})
            latest_policy_train_stats["attn_mlp_policy_name"] = "AttnMLPWaitingForAttentionBatch"
    if _online_spectral_policy_should_collect_attention(iter_num):
        online_spectral_policy_matrices.extend(_online_spectral_policy_probe_attention_matrices_if_due(iter_num))
    if online_spectral_policy_matrices:
        spectral_policy_matrix, spectral_policy_samples = _online_spectral_weighted_mean_matrices(
            online_spectral_policy_matrices
        )
        _online_spectral_policy_update_from_matrix(
            spectral_policy_matrix,
            iter_num,
            num_attention_samples=spectral_policy_samples,
        )
        if latest_policy_train_stats is not None and str(latest_policy_train_stats.get("online_spectral_policy_name", "")).startswith("OnlineSpectral"):
            policy_name_for_stats = latest_policy_train_stats["online_spectral_policy_name"]
            latest_policy_train_stats = dict(online_spectral_policy_latest_stats or {})
            latest_policy_train_stats["online_spectral_policy_name"] = policy_name_for_stats
    _online_spectral_probe_update_if_due()
    _head_direction_logger_update_if_due()
    _head_signal_probe_update_if_due()
    _all_head_attn_dataset_collect_if_due(iter_num)
    _all_head_pairwise_dataset_collect_if_due(iter_num)
    _write_online_stats(force=False)
    _write_online_spectral_state(force=False)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(
            f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, "
            f"mfu {running_mfu*100:.2f}%"
        )
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

_write_online_stats(force=True)
_write_online_spectral_state(force=True)
_all_head_attn_dataset_flush(force=True)
_all_head_pairwise_dataset_flush(force=True)

if ddp:
    destroy_process_group()
