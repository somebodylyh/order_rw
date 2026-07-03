# Preset config for scripts/runner/hierarchical_segment_curriculum_runner.py
# Usage:
#   python scripts/runner/hierarchical_segment_curriculum_runner.py config/WikiText103/seq256/permute/block128/segment_curriculum.py

config = 'config/WikiText103/seq256/permute/block128/random.py'

train_out_dir = 'out/curriculum/permute/seq256/block128/out-wikitext103-seq256-random-b128-permute-hcurr-margin-aligned-eval-6-stage'
benchmark_root = 'Report/curriculum/permute/seq256/block128/hierarchical_block128_margin_aligned_eval-6-stage'

wandb_project = 'AOGPT-order-block-128-final'
wandb_run_name = 'seq256-random-b128-permute-hcurr-margin-6-stage'

warmup_iters = 8000
stage_iters = 7000
num_curriculum_stages = 6
segment_guided_ratios = '1,1,1,1,1,1'
segment_max_lens = '2,2,2,2,2,2'
segment_max_units_per_order = 999999
segment_use_all_units = True
segment_top_k_pairs = 96

benchmark_batch_size = 1
pair_mining_batches = 24
pair_eval_batch_size = 16
pair_chunk_size = pair_eval_batch_size
forward_eval_batch_size = benchmark_batch_size * pair_chunk_size

aggregation_margin_threshold = 0.03
aggregation_margin_thresholds = '0.03,0.03,0.025,0.025,0.02,0.02'
aggregation_rank_key = 'margin_then_score'
pair_aggregation_mode = 'disjoint_pairs'
benchmark_num_levels = 1
aggregate_top_k_pairs = 128
pair_score_k = 2
pair_score_mode = 'signed_drop'
tv_weight = 0.3
drop_weight = 0.3
drop_weights = '0.2,0.2,0.25,0.25,0.3,0.3'

pair_mining_mode = 'attention_pruned'
attn_top_k = 8
attn_top_ks = '8,8,8,8,8,8'
attn_num_batches = 24
attn_batch_size = 32
attn_mode = 'Random'
attn_symmetrize = 'mean'
attn_export_type = 'with_none'
