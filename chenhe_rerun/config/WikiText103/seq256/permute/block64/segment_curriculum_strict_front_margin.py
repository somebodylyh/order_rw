# Preset config for scripts/runner/hierarchical_segment_curriculum_runner.py
# Usage:
#   python scripts/runner/hierarchical_segment_curriculum_runner.py config/WikiText103/seq256/permute/block64/segment_curriculum_strict_front_margin.py
#
# Motivation:
#   Compared with segment_curriculum.py, this variant makes early pair
#   acceptance stricter and reduces the number of aggregation candidates
#   stage by stage. The goal is to reduce noisy pair seeds before they are
#   amplified into long units, without using any original/L2R supervision.

config = 'config/WikiText103/seq256/permute/block64/random.py'

train_out_dir = 'out/curriculum/permute/seq256/block64/out-wikitext103-seq256-random-b64-permute-hcurr-strict-front-margin-decay-6-stage'
benchmark_root = 'Report/curriculum/permute/seq256/block64/hierarchical_block64_strict_front_margin_decay-6-stage'

wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-random-b64-permute-hcurr-strict-front-margin-decay-6-stage'

warmup_iters = 8000
stage_iters = 7000
num_curriculum_stages = 6
segment_guided_ratios = '1,1,1,1,1,1'
segment_max_lens = '2,2,2,2,2,2'
segment_max_units_per_order = 999999
segment_use_all_units = True
segment_top_k_pairs = 96

benchmark_batch_size = 1
pair_mining_batches = 48
pair_eval_batch_size = 16
pair_chunk_size = pair_eval_batch_size
forward_eval_batch_size = benchmark_batch_size * pair_chunk_size

aggregation_margin_threshold = 0.08
aggregation_margin_thresholds = '0.08,0.07,0.06,0.055,0.055,0.055'
aggregation_rank_key = 'margin_then_score'
pair_aggregation_mode = 'disjoint_pairs'
benchmark_num_levels = 1
aggregate_top_k_pairs = 96
aggregate_top_k_pairs_per_stage = '128,96,48,16,6,2'
pair_score_k = 2
pair_score_mode = 'signed_drop'
tv_weight = 0.3
drop_weight = 0.15
drop_weights = '0.20,0.18,0.15,0.10,0.05,0.00'

pair_mining_mode = 'attention_pruned'
attn_top_k = 6
attn_top_ks = '6,6,6,5,4,4'
attn_num_batches = 32
attn_batch_size = 32
attn_mode = 'Random'
attn_symmetrize = 'mean'
attn_export_type = 'with_none'
