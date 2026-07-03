# Online oriented all-head pairwise teacher dataset for Attn-MLP distillation.
#
# This try collects every layer/head attention matrix during Random training,
# runs direct-asym-eig per head, evaluates raw/reverse by current-model train
# loss, and stores the lower-loss orientation as pairwise supervision.

_base_config = 'config/WikiText103/seq256/permute/block64/random.py'
exec(open(_base_config).read())

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-random-b64-permute-block-allhead-pairwise-oriented-try5-10k15k'
init_from = 'resume'
resume_optimizer_state = True

max_iters = 15000
lr_decay_iters = 50000
eval_interval = 1000
eval_iters = 100
log_interval = 10
always_save_checkpoint = True

wandb_log = False
wandb_run_name = 'seq256-random-b64-permute-allhead-pairwise-oriented-try5-10k15k'

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
all_head_pairwise_dataset_stop_iter = 14992
all_head_pairwise_dataset_interval = 16
all_head_pairwise_dataset_attention_batch_size = 16
all_head_pairwise_dataset_attention_micro_batches_per_record = 16
all_head_pairwise_dataset_export_type = 'without_none'
all_head_pairwise_dataset_attention_order_mode = 'random'
all_head_pairwise_dataset_out_dir = 'Report/MLP_distillation/try_5/oriented_pairwise_dataset'
all_head_pairwise_dataset_shard_size = 512
all_head_pairwise_dataset_dtype = 'float16'
all_head_pairwise_dataset_seed = 161803
all_head_pairwise_dataset_max_records = 0
all_head_pairwise_dataset_train_records = 250
all_head_pairwise_dataset_direct_asym_eig_mode = 'raw_right_largest_real_real'
all_head_pairwise_dataset_loss_batches = 4
all_head_pairwise_dataset_loss_batch_size = 16
all_head_pairwise_dataset_loss_candidate_batch_size = 4
all_head_pairwise_dataset_loss_prefix_k = 16
all_head_pairwise_dataset_loss_score = 'linear_profile'
all_head_pairwise_dataset_loss_exp_tau = 16.0
all_head_pairwise_dataset_store_pairwise_q = True
