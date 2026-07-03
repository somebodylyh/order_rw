# Continue the seq256/block64 non-permuted Random checkpoint while probing all
# 4x8 heads inside the training loop. This matches the fixed-checkpoint
# consensus setting that stabilized most heads.

out_dir = 'out/base/nonpermute/seq256/block64/out-wikitext103-seq256-random-b64-head-signal-stable-try4-all-heads-consensus-5k-to5600'
eval_interval = 200
eval_iters = 100
log_interval = 20

wandb_log = False
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-random-b64-head-signal-stable-try4-all-heads-consensus-5k-to5600'

dataset = 'wikitext103'
batch_size = 64
block_size = 256
gradient_accumulation_steps = 2
permute_data = False
permute_seed = 42

model_type = 'aogpt'
train_stage = 'standard'
aogpt_train_mode = 'Random'
main_eval_mode = 'Random'
generalization_eval_mode = ''
n_layer = 4
n_head = 8
n_embd = 384
dropout = 0

block_order_block_len = 4

init_from = 'resume'
resume_optimizer_state = True
learning_rate = 1e-3
max_iters = 5600
lr_decay_iters = 50000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 0

always_save_checkpoint = True
save_iter_checkpoints = False
save_iter_checkpoint_steps = ''
save_iter_checkpoint_keep = 0

compile = False
eval_generate_step_loss_log = False
eval_kendall_distance_log = False

head_signal_probe_enabled = True
head_signal_probe_interval = 20
head_signal_probe_out_dir = 'Report/head_singal_Stable/try_4/online_training_probe'
head_signal_probe_heads = 'all'
head_signal_probe_batches = 8
head_signal_probe_batch_size = 16
head_signal_probe_loss_batches = 4
head_signal_probe_loss_batch_size = 16
head_signal_probe_candidate_batch_size = 8
head_signal_probe_prefix_k = 16
head_signal_probe_prefix_weight = 0.7
head_signal_probe_full_weight = 0.3
head_signal_probe_export_type = 'with_none'
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
head_signal_probe_deterministic = True
head_signal_probe_seed = 24681357
