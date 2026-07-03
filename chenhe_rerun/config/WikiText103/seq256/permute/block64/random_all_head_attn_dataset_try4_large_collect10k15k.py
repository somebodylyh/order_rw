# Larger all-head attention dataset for Attn-MLP distillation.
#
# One record contains every layer/head block attention matrix at one training
# iteration. Each record is averaged over 256 probe samples via 16 micro-batches
# of size 16 to keep the probe memory footprint close to try3.

_base_config = 'config/WikiText103/seq256/permute/block64/random.py'
exec(open(_base_config).read())

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-random-b64-permute-block-allhead-attn-dataset-try4-large-10k15k'
init_from = 'resume'
resume_optimizer_state = True

max_iters = 15000
lr_decay_iters = 50000
eval_interval = 1000
eval_iters = 100
log_interval = 10
always_save_checkpoint = True

wandb_log = False
wandb_run_name = 'seq256-random-b64-permute-allhead-attn-dataset-try4-large-10k15k'

aogpt_train_mode = 'Random'
main_eval_mode = 'Random'
attn_mlp_policy_enabled = False
online_spectral_policy_enabled = False
online_spectral_policy_probe_enabled = False
online_attention_probe_enabled = False
head_direction_logger_enabled = False
head_signal_probe_enabled = False

all_head_attn_dataset_enabled = True
all_head_attn_dataset_start_iter = 10000
all_head_attn_dataset_stop_iter = 14992
all_head_attn_dataset_interval = 16
all_head_attn_dataset_batch_size = 16
all_head_attn_dataset_micro_batches_per_record = 16
all_head_attn_dataset_batches_per_step = 1
all_head_attn_dataset_export_type = 'without_none'
all_head_attn_dataset_order_mode = 'random'
all_head_attn_dataset_out_dir = 'Report/MLP_distillation/try_4/all_head_attn_dataset'
all_head_attn_dataset_shard_size = 64
all_head_attn_dataset_dtype = 'float16'
all_head_attn_dataset_seed = 271828
all_head_attn_dataset_split = 'train'
all_head_attn_dataset_max_records = 0
