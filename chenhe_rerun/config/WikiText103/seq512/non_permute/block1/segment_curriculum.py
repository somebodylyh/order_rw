# Preset config for scripts/runner/hierarchical_segment_curriculum_runner.py
# Usage:
#   python scripts/runner/hierarchical_segment_curriculum_runner.py config/WikiText103/seq512/non_permute/block1/segment_curriculum.py

config = 'config/WikiText103/seq512/non_permute/block1/random.py'

train_out_dir = 'out/curriculum/nonpermute/seq512/block1/out-wikitext103-seq512-random-b1-curriculum'
benchmark_root = 'Report/curriculum/nonpermute/seq512/block1/block1'

wandb_project = 'AOGPT-order-block'
wandb_run_name = 'seq512-random-b1-curriculum'

warmup_iters = 8000
stage_iters = 8000
num_curriculum_stages = 4
segment_guided_ratios = '1,1,1,1'
segment_max_lens = '2,2,2,2'
segment_max_units_per_order = 999999
segment_use_all_units = True
segment_top_k_pairs = 24

benchmark_batch_size = 16
pair_mining_batches = 8
pair_eval_batch_size = 2
pair_chunk_size = pair_eval_batch_size
forward_eval_batch_size = benchmark_batch_size * pair_chunk_size

aggregation_margin_threshold = 0.02
pair_aggregation_mode = 'disjoint_pairs'
benchmark_num_levels = 1
aggregate_top_k_pairs = 24
pair_score_k = 2
pair_score_mode = 'signed_drop'
tv_weight = 0.3
drop_weight = 0.2

pair_mining_mode = 'attention_pruned'
attn_top_k = 8
attn_num_batches = 8
attn_batch_size = 8
attn_mode = 'Random'
attn_symmetrize = 'mean'
attn_export_type = 'with_none'
