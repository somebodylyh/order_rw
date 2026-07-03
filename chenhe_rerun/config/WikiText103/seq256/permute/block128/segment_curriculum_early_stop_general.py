# Preset config for scripts/runner/hierarchical_segment_curriculum_runner.py
# Usage:
#   python scripts/runner/hierarchical_segment_curriculum_runner.py config/WikiText103/seq256/permute/block128/segment_curriculum_early_stop_general.py
#
# Motivation:
#   Simplified high-sample block128 ablation for gain-filtered multi-level
#   aggregation. This variant keeps only the basic aggregation sanity checks
#   and the target-specific gain gate, then accepts the gain-passing subset.
#   Policy and boundary gates are disabled. No original frame or L2R diagnostics
#   are used for decisions.

config = 'config/WikiText103/seq256/permute/block128/random.py'

train_out_dir = 'out/curriculum/permute/seq256/block128/out-wikitext103-seq256-random-b128-permute-hcurr-early-stop-gain-only-6-stage'
benchmark_root = 'Report/curriculum/permute/seq256/block128/hierarchical_block128_early_stop_gain_only-6-stage'

wandb_project = 'AOGPT-order-block-128-final'
wandb_run_name = 'seq256-random-b128-permute-hcurr-early-stop-gain-only-6-stage'

warmup_iters = 8000
stage_iters = 7000
num_curriculum_stages = 6
segment_guided_ratios = '1,1,1,1,1,1'
segment_max_lens = '2,2,2,2,2,2'
segment_max_units_per_order = 999999
segment_use_all_units = True
segment_top_k_pairs = 96

benchmark_split = 'train'
benchmark_batch_size = 2
pair_mining_batches = 96
pair_eval_batch_size = 16
pair_chunk_size = pair_eval_batch_size
forward_eval_batch_size = benchmark_batch_size * pair_chunk_size

aggregation_margin_threshold = 0.06
aggregation_margin_thresholds = '0.06,0.055,0.05,0.045,0.045,0.045'
aggregation_rank_key = 'margin_then_score'
pair_aggregation_mode = 'disjoint_pairs'
benchmark_num_levels = 128
aggregate_top_k_pairs = 128
aggregate_top_k_pairs_per_stage = '128,96,48,16,6,2'
pair_score_k = 2
pair_score_mode = 'signed_drop'
tv_weight = 0.3
drop_weight = 0.15
drop_weights = '0.20,0.18,0.15,0.10,0.05,0.00'

early_stop_enabled = True
early_stop_eval_split = 'val'
early_stop_min_pairs_after_margin = 1

# Target-specific gain gate:
# gain(i -> j) = loss(j | length-matched random context) - loss(j | i).
early_stop_gain_eval_batches = 48
early_stop_gain_eval_batch_size = 32
early_stop_gain_random_contexts = 8
early_stop_gain_threshold = 0.01
# Values <= -1e20 disable the p10 gate in hierarchical_structured_benchmark.py.
early_stop_gain_p10_threshold = -1e30
early_stop_gain_positive_rate = 0.70
early_stop_min_gain_pass_rate = 0.0
early_stop_filter_segments_by_gain = True

# Candidate policy gate disabled for this simplified run.
early_stop_policy_eval_batches = 0
early_stop_policy_eval_batch_size = 32
early_stop_policy_epsilon = 0.005

# Boundary gate disabled for this simplified run.
early_stop_boundary_eval_batches = 0
early_stop_boundary_eval_batch_size = 32
early_stop_boundary_random_contexts = 4
early_stop_boundary_gain_threshold = 0.0
early_stop_boundary_gain_p10_threshold = -1e30
early_stop_boundary_positive_rate = 0.60
early_stop_max_bad_boundary_ratio = 0.30

pair_mining_mode = 'attention_pruned'
attn_top_k = 8
attn_top_ks = '8,8,8,8,6,6'
attn_num_batches = 64
attn_batch_size = 64
attn_mode = 'Random'
attn_symmetrize = 'mean'
attn_export_type = 'with_none'
