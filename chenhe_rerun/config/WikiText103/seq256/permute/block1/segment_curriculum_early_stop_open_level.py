# Preset config for scripts/runner/hierarchical_segment_curriculum_runner.py
# Usage:
#   python scripts/runner/hierarchical_segment_curriculum_runner.py config/WikiText103/seq256/permute/block1/segment_curriculum_early_stop_open_level.py
#
# Motivation:
#   Token/block1 permuted segment-curriculum ablation with open-level
#   aggregation. This keeps the existing block1 segment setup mostly intact,
#   but uses gain-only early stop and benchmark_num_levels=256 so each
#   benchmark stage can keep aggregating while the candidate merges remain
#   valid. For block1 there are at most 256 singleton units initially, so 256 is
#   effectively "unlimited" under the early-stop no-progress/no-pair checks.
#   Policy and boundary gates are disabled. No original frame or L2R diagnostics
#   are used for decisions.

config = 'config/WikiText103/seq256/permute/block1/random.py'

train_out_dir = 'out/curriculum/permute/seq256/block1/out-wikitext103-seq256-random-b1-permute-hcurr-early-stop-open-level-gain-only'
benchmark_root = 'Report/curriculum/permute/seq256/block1/hierarchical_block1_early_stop_open_level_gain_only'

wandb_project = 'AOGPT-order-block-1-final'
wandb_run_name = 'seq256-random-b1-permute-hcurr-early-stop-open-level-gain-only'

warmup_iters = 8000
stage_iters = 8000
num_curriculum_stages = 4
segment_guided_ratios = '1,1,1,1'
segment_max_lens = '2,2,2,2'
segment_max_units_per_order = 999999
segment_use_all_units = True
segment_top_k_pairs = 64

benchmark_split = 'train'
benchmark_batch_size = 1
pair_mining_batches = 24
pair_eval_batch_size = 8
pair_chunk_size = pair_eval_batch_size
forward_eval_batch_size = benchmark_batch_size * pair_chunk_size

aggregation_margin_threshold = 0.02
aggregation_rank_key = 'margin_then_score'
pair_aggregation_mode = 'disjoint_pairs'
benchmark_num_levels = 256
aggregate_top_k_pairs = 128
pair_score_k = 2
pair_score_mode = 'signed_drop'
tv_weight = 0.3
drop_weight = 0.2

early_stop_enabled = True
early_stop_eval_split = 'val'
early_stop_min_pairs_after_margin = 1

# Target-specific gain gate:
# gain(i -> j) = loss(j | length-matched random context) - loss(j | i).
early_stop_gain_eval_batches = 24
early_stop_gain_eval_batch_size = 16
early_stop_gain_random_contexts = 4
early_stop_gain_threshold = 0.01
# Values <= -1e20 disable the p10 gate in hierarchical_structured_benchmark.py.
early_stop_gain_p10_threshold = -1e30
early_stop_gain_positive_rate = 0.70
early_stop_min_gain_pass_rate = 0.0
early_stop_filter_segments_by_gain = True

# Candidate policy gate disabled for this simplified run.
early_stop_policy_eval_batches = 0
early_stop_policy_eval_batch_size = 16
early_stop_policy_epsilon = 0.005

# Boundary gate disabled for this simplified run.
early_stop_boundary_eval_batches = 0
early_stop_boundary_eval_batch_size = 16
early_stop_boundary_random_contexts = 4
early_stop_boundary_gain_threshold = 0.0
early_stop_boundary_gain_p10_threshold = -1e30
early_stop_boundary_positive_rate = 0.60
early_stop_max_bad_boundary_ratio = 0.30

pair_mining_mode = 'attention_pruned'
attn_top_k = 4
attn_num_batches = 24
attn_batch_size = 32
attn_mode = 'Random'
attn_symmetrize = 'mean'
attn_export_type = 'with_none'
