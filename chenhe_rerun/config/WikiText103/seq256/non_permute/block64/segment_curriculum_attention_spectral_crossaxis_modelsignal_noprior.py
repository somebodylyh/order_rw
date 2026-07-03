# No-prior model-signal attention-spectral curriculum for WikiText103 block64.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 /data/users/chenhe/conda_envs/X1/bin/python \
#     scripts/runner/hierarchical_segment_curriculum_runner.py \
#     config/WikiText103/seq256/non_permute/block64/segment_curriculum_attention_spectral_crossaxis_modelsignal_noprior.py
#
# This is the language counterpart of the ImageNet64 model-signal no-prior
# curriculum. Candidate orders come from checkpoint attention spectral axes,
# then the top attention candidates are reranked only with checkpoint-side
# token loss / early reveal / reverse-margin / stability signals.
#
# No L2R order, hand scan, text-position heuristic, 2D coordinate prior, FID,
# or image metric is used for candidate generation, ranking, or stopping.
# Because this is non_permute text, absolute-position and natural sequential
# priors should still be treated as diagnostic caveats.

config = 'config/WikiText103/seq256/non_permute/block64/random.py'

train_out_dir = 'out/curriculum/nonpermute/seq256/block64/out-wikitext103-seq256-random-b64-attn-spectral-crossaxis-modelsignal-noprior-4-stage'
benchmark_root = 'Report/curriculum/nonpermute/seq256/block64/attention_spectral_crossaxis_modelsignal_noprior-4-stage'

wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-random-b64-nonpermute-attn-spectral-crossaxis-modelsignal-noprior-4-stage'

# Keep final_max_iters aligned to the 50k random baseline:
# 8000 + 4 * 10500 = 50000.
warmup_iters = 8000
stage_iters = 10500
num_curriculum_stages = 4
resume_existing = True
run_to_end_after_single_unit = False

# The benchmark writes one learned 64-block chain. Ratios below 1.0 mix that
# chain with ordinary random reveal orders; the final stage trains exactly the
# recovered learned chain.
segment_guided_ratios = '0.5,0.75,0.9,1'
segment_max_lens = '64,64,64,64'
segment_max_units_per_order = 999999
segment_top_k_pairs = 64

# Once training becomes fully guided, keep the last pre-full-guidance policy
# fixed. With these ratios, stage 4 reuses the stage 3 policy instead of
# running another order recovery pass at ratio 1.
freeze_policy_at_full_guidance = True
freeze_policy_ratio_threshold = 1.0

benchmark_split = 'train'
benchmark_batch_size = 1

curriculum_recovery_mode = 'attention_spectral_cross_axis'

# Attention mining. Training batch size is still inherited from random.py.
# 64 is intentionally below the old text config's 128, which could OOM during
# benchmark recovery on a busy 24GB card.
attn_num_batches = 24
attn_batch_size = 64
attn_mode = 'Random'
attn_export_type = 'with_none'

# Wider no-prior candidate generation. group_methods='gap' means band
# boundaries are chosen from discontinuities in learned spectral coordinates,
# not from equal-size chunks or hand-designed positions.
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

# Pure model-signal rerank over the top attention candidates. This uses only
# teacher-forced token loss, prefix loss, early reveal signed drop, reverse
# prefix margin, positive rate, and batch stability from the current checkpoint.
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
