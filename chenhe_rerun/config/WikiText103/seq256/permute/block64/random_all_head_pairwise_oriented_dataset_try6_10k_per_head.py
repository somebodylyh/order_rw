# Long online oriented all-head pairwise teacher dataset for Attn-MLP distillation.
#
# Target scale: 10k records. Since each record contains all 4x8 heads, this is
# 10k matrices per head, 320k head-matrix samples total.

_base_config = 'config/WikiText103/seq256/permute/block64/random.py'
exec(open(_base_config).read())

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-random-b64-permute-block-allhead-pairwise-oriented-try6-10k-per-head'
init_from = 'resume'
resume_optimizer_state = True

max_iters = 170000
lr_decay_iters = 170000
eval_interval = 5000
eval_iters = 100
log_interval = 10
always_save_checkpoint = True

wandb_log = False
wandb_run_name = 'seq256-random-b64-permute-allhead-pairwise-oriented-try6-10k-per-head'

aogpt_train_mode = 'Random'
main_eval_mode = 'Random'
attn_mlp_policy_enabled = False
online_spectral_policy_enabled = False
online_spectral_policy_probe_enabled = False
online_attention_probe_enabled = False
head_direction_logger_enabled = False
head_signal_probe_enabled = False

all_head_attn_dataset_enabled = False

all_head_pairwise_dataset_enabled = True
all_head_pairwise_dataset_start_iter = 10000
all_head_pairwise_dataset_stop_iter = 169984
all_head_pairwise_dataset_interval = 16
all_head_pairwise_dataset_attention_batch_size = 16
all_head_pairwise_dataset_attention_micro_batches_per_record = 16
all_head_pairwise_dataset_export_type = 'without_none'
all_head_pairwise_dataset_attention_order_mode = 'random'
all_head_pairwise_dataset_out_dir = 'Report/MLP_distillation/try_6/oriented_pairwise_dataset_10k_per_head'
all_head_pairwise_dataset_shard_size = 512
all_head_pairwise_dataset_dtype = 'float16'
all_head_pairwise_dataset_seed = 161803
all_head_pairwise_dataset_max_records = 10000
all_head_pairwise_dataset_train_records = 8000
all_head_pairwise_dataset_direct_asym_eig_mode = 'raw_right_largest_real_real'
all_head_pairwise_dataset_loss_batches = 4
all_head_pairwise_dataset_loss_batch_size = 16
all_head_pairwise_dataset_loss_candidate_batch_size = 4
all_head_pairwise_dataset_loss_prefix_k = 16
all_head_pairwise_dataset_loss_score = 'linear_profile'
all_head_pairwise_dataset_loss_exp_tau = 16.0
all_head_pairwise_dataset_store_pairwise_q = True
