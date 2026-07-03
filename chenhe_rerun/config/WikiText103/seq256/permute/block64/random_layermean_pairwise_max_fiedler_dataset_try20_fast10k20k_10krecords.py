# Online oriented L0 layer-mean pairwise-max Fiedler teacher dataset for
# Attn-MLP distillation.
#
# This collects the latest try23 teacher target:
# - average all layer-0 heads into one 64x64 attention matrix A;
# - build W=max(A,A.T), diag(W)=0;
# - recover the graph Laplacian Fiedler axis;
# - compare raw axis vs reverse with current-model train linear_profile_loss
#   on a heavy full-probe batch bank;
# - store the lower-loss orientation as pairwise supervision.
#
# Compared with try8, each record contributes one layer-level sample instead of
# 32 single-head samples.

_base_config = 'config/WikiText103/seq256/permute/block64/random.py'
exec(open(_base_config).read())

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-random-b64-permute-block-layermean-fiedler-pairwise-try20-fast10k20k-10krecords'
init_from = 'resume'
resume_optimizer_state = True

max_iters = 20000
lr_decay_iters = 50000
eval_interval = 5000
eval_iters = 100
log_interval = 10
always_save_checkpoint = True

wandb_log = False
wandb_run_name = 'seq256-random-b64-permute-layermean-fiedler-pairwise-try20-fast10k20k-10krecords'

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
all_head_pairwise_dataset_stop_iter = 19999
all_head_pairwise_dataset_interval = 1
all_head_pairwise_dataset_attention_batch_size = 16
all_head_pairwise_dataset_attention_micro_batches_per_record = 16
all_head_pairwise_dataset_export_type = 'without_none'
all_head_pairwise_dataset_attention_order_mode = 'random'
all_head_pairwise_dataset_out_dir = 'Report/language/wikitext103/mlp/distillation/try_20/l0_layermean_fiedler_oriented_dataset_fast10k20k'
all_head_pairwise_dataset_shard_size = 64
all_head_pairwise_dataset_dtype = 'float16'
all_head_pairwise_dataset_seed = 161803
all_head_pairwise_dataset_max_records = 10000
all_head_pairwise_dataset_train_records = 8000
all_head_pairwise_dataset_heads = '0:mean'
all_head_pairwise_dataset_candidate_source = 'pairwise_max_fiedler'
all_head_pairwise_dataset_loss_batches = 256
all_head_pairwise_dataset_loss_batch_size = 16
all_head_pairwise_dataset_loss_candidate_batch_size = 2
all_head_pairwise_dataset_loss_prefix_k = 16
all_head_pairwise_dataset_loss_score = 'linear_profile'
all_head_pairwise_dataset_loss_exp_tau = 16.0
all_head_pairwise_dataset_store_pairwise_q = True
all_head_pairwise_dataset_resume_existing = True
