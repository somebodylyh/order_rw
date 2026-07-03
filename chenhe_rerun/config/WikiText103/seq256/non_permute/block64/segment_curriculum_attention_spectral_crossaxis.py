# Attention-spectral cross-axis curriculum for WikiText103 seq256/block64.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 /data/users/chenhe/conda_envs/X1/bin/python \
#     scripts/runner/hierarchical_segment_curriculum_runner.py \
#     config/WikiText103/seq256/non_permute/block64/segment_curriculum_attention_spectral_crossaxis.py
#
# This is the text counterpart of the ImageNet64 attention-spectral curriculum:
# every stage recovers one 64-block learned order from checkpoint attention and
# feeds it back through train.py's segment-guided Random carrier.

config = 'config/WikiText103/seq256/non_permute/block64/random.py'

train_out_dir = 'out/curriculum/nonpermute/seq256/block64/out-wikitext103-seq256-random-b64-attn-spectral-crossaxis-4-stage'
benchmark_root = 'Report/curriculum/nonpermute/seq256/block64/attention_spectral_crossaxis-4-stage'

wandb_project = 'AOGPT-order-block-64'
wandb_run_name = 'seq256-random-b64-nonpermute-attn-spectral-crossaxis-4-stage'

# Keep the curriculum directly comparable to the 50k random baseline:
# final_max_iters = warmup_iters + stage_iters * num_curriculum_stages = 50000.
warmup_iters = 6000
stage_iters = 11000
num_curriculum_stages = 4
resume_existing = True
run_to_end_after_single_unit = False

segment_guided_ratios = '0.5,0.75,0.9,1'
segment_max_lens = '64,64,64,64'
segment_max_units_per_order = 999999
segment_top_k_pairs = 64

benchmark_split = 'train'
benchmark_batch_size = 1

curriculum_recovery_mode = 'attention_spectral_cross_axis'
# Text recovery is lighter than ImageNet here, so use a larger forward-only
# attention mining batch while leaving the actual training batch inherited from
# random.py unchanged.
attn_num_batches = 16
attn_batch_size = 128
attn_mode = 'Random'
attn_export_type = 'with_none'

spectral_max_primary_matrices = 16
spectral_max_primary_matrices_per_stage = '8,12,16,16'
spectral_num_components = 4
spectral_component_pairs = '1-2'
spectral_num_angles = 16
spectral_num_angles_per_stage = '8,12,16,16'
spectral_k_values = '8,10'
spectral_k_values_per_stage = '4,6;6,8;8,10;10,12'
spectral_group_methods = 'gap'
spectral_threshold_percentile = 60.0
spectral_transform = 'relu'
spectral_temperature = 1.0
spectral_direction_lambdas = '0,0.1,0.25'
spectral_directed_score_weight = 0.25
spectral_band_quality_weight = 0.05
spectral_score_adjacency_sym = 'max'
spectral_max_candidates = 4096
spectral_export_top_candidates = 128
spectral_output_policy = 'single_order'
spectral_seed = 12345
spectral_device = 'cuda'
spectral_dtype = 'bfloat16'
