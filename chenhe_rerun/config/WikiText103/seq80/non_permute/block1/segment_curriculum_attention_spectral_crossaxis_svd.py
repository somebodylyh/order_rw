# Offline attention-spectral/SVD-style curriculum with no-prior model-signal
# rerank for WikiText103 seq80 non-permuted token-level training.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 /data/users/chenhe/conda_envs/X1/bin/python \
#     scripts/runner/hierarchical_segment_curriculum_runner.py \
#     config/WikiText103/seq80/non_permute/block1/segment_curriculum_attention_spectral_crossaxis_svd.py
#
# This is the seq80 counterpart of the seq256/block64 model-signal no-prior
# attention-spectral curriculum. Each stage mines checkpoint attention, builds
# a normalized attention graph, runs the exact offline spectral decomposition
# path, generates order candidates, reranks the top candidates with
# checkpoint-side model signals, and feeds the best learned token order back
# through train.py's segment-guided Random carrier.
#
# The candidate generator remains the old exact SVD/spectral family. The final
# selection is now scored by no-prior model signals: token loss, prefix loss,
# early reveal signed drop, reverse-margin, positive rate, and stability.

config = 'config/WikiText103/seq80/non_permute/block1/random.py'

train_out_dir = 'out/curriculum/nonpermute/seq80/block1/out-wikitext103-seq80-random-b1-attn-spectral-crossaxis-svd-modelsignal-noprior-4-stage-50000-iters'
benchmark_root = 'Report/curriculum/nonpermute/seq80/block1/attention_spectral_crossaxis_svd_modelsignal_noprior-4-stage-50000-iters'

wandb_project = 'AOGPT-order-block-seq80-final'
wandb_run_name = 'seq80-random-b1-nonpermute-attn-spectral-crossaxis-svd-modelsignal-noprior-4-stage-50000-iters'

# Match the 50k random baseline: 8000 + 4 * 10500 = 50000.
warmup_iters = 8000
stage_iters = 10500
num_curriculum_stages = 4
resume_existing = True
run_to_end_after_single_unit = False

segment_guided_ratios = '0.5,0.75,0.9,1'
segment_max_lens = '80,80,80,80'
segment_max_units_per_order = 999999
segment_top_k_pairs = 80
segment_use_all_units = True

# Once the carrier reaches full guidance, keep the previous recovered order
# instead of searching again under a deterministic learned order.
freeze_policy_at_full_guidance = True
freeze_policy_ratio_threshold = 1.0

benchmark_split = 'train'
benchmark_batch_size = 1

curriculum_recovery_mode = 'attention_spectral_cross_axis'
attention_spectral_backend = 'offline'

# Offline attention mining. Batch size follows the seq256/block64 no-prior
# config to keep recovery memory conservative while reranking is enabled.
attn_num_batches = 24
attn_batch_size = 64
attn_mode = 'Random'
attn_export_type = 'with_none'

# Exact benchmark-time spectral recovery. The implementation uses EVD on the
# normalized symmetric affinity matrix; this is the same old SVD/spectral-axis
# family of methods, with no online subspace approximation.
spectral_max_primary_matrices = 16
spectral_max_primary_matrices_per_stage = '16,16,16,16'
spectral_num_components = 4
spectral_component_pairs = '1-2'
spectral_num_angles = 16
spectral_num_angles_per_stage = '16,16,16,16'
spectral_k_values = '4,6,7,8,9,10,12'
spectral_k_values_per_stage = '4,6,7,8,9,10,12;4,6,7,8,9,10,12;4,6,7,8,9,10,12;4,6,7,8,9,10,12'
spectral_group_methods = 'gap'
spectral_threshold_percentile = 60.0
spectral_transform = 'relu'
spectral_temperature = 1.0
spectral_direction_lambdas = '0,0.1,0.25'
spectral_directed_score_weight = 0.25
spectral_band_quality_weight = 0.05
spectral_score_adjacency_sym = 'max'
spectral_max_candidates = 8192
spectral_export_top_candidates = 512
spectral_output_policy = 'single_order'

# Score the top attention/SVD candidates and select the best no-prior model
# signal order, matching the seq256/block64 model-signal rerank rule.
spectral_rerank_num_candidates = 96
spectral_rerank_num_batches = 4
spectral_rerank_batch_size = 8
spectral_rerank_forward_batch_size = 16
spectral_rerank_prefix_k = 16
spectral_rerank_include_reverse = 1
spectral_rerank_attention_weight = 1.0
spectral_rerank_full_loss_weight = 0.10
spectral_rerank_prefix_loss_weight = 0.20
spectral_rerank_signed_drop_weight = 0.15
spectral_rerank_reverse_margin_weight = 0.35
spectral_rerank_positive_rate_weight = 0.10
spectral_rerank_stability_weight = 0.10

spectral_seed = 12345
spectral_device = 'cuda'
spectral_dtype = 'bfloat16'
