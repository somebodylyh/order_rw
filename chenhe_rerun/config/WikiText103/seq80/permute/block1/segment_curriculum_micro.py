# Conservative token-level micro-segment curriculum config.
# Usage:
#   python scripts/runner/hierarchical_segment_curriculum_runner.py config/WikiText103/seq80/permute/block1/segment_curriculum_micro.py

config = 'config/WikiText103/seq80/permute/block1/random.py'

train_out_dir = 'out/curriculum/permute/seq80/block1/out-wikitext103-seq80-random-b1-hcurr-micro-score-big'
benchmark_root = 'Report/curriculum/permute/seq80/block1/hierarchical_block1_micro_score-big'

wandb_project = 'AOGPT-order-block-seq80'
wandb_run_name = 'seq80-random-b1-permute-hcurr-token-micro-score-big'

warmup_iters = 8000
stage_iters = 8000
num_curriculum_stages = 4
segment_guided_ratios = '1,1,1,1'
segment_max_lens = '999999,999999,999999,999999'
segment_max_units_per_order = 16
segment_use_all_units = False
segment_top_k_pairs = 96

benchmark_batch_size = 2
pair_mining_batches = 24
pair_eval_batch_size = 8
pair_chunk_size = pair_eval_batch_size
forward_eval_batch_size = benchmark_batch_size * pair_chunk_size

# token_micro_segments intentionally skips directed margin filtering and uses
# score-based undirected affinity for neighborhood proposal.
aggregation_margin_threshold = -1e30
pair_aggregation_mode = 'token_micro_segments'
benchmark_num_levels = 2
aggregate_top_k_pairs = 160
pair_score_k = 2
pair_score_mode = 'signed_drop'
tv_weight = 0.3
drop_weight = 0.2

pair_mining_mode = 'attention_pruned'
attn_top_k = 8
attn_num_batches = 24
attn_batch_size = 32
attn_mode = 'Random'
attn_symmetrize = 'mean'
attn_export_type = 'with_none'

token_micro_topk = 4
token_micro_min_affinity = -1e30
token_micro_max_degree = 5
token_micro_max_segment_size = 4
token_micro_min_segment_size = 2
token_micro_min_density = 0.5
# 0 means enumerate all internal permutations. With max size 4 this is at most
# 24 candidate orders per component.
token_micro_num_order_candidates = 0
token_micro_eval_batches = 32
token_micro_eval_batch_size = 32
token_micro_random_suffixes = 4
token_micro_random_orders = 8
# These two are kept for CLI compatibility/diagnostics only. The simplified
# token path accepts by best-vs-random margin, not pair consistency or reverse.
token_micro_min_pair_consistency = -1e30
token_micro_reverse_margin_threshold = -1e30
token_micro_random_margin_threshold = 0.01
token_micro_keep_singletons = False
token_micro_max_accepted_segments = 0
