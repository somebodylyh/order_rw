# Fixed-random50k L0 layer-mean pairwise-max Fiedler teacher dataset.
#
# This is the same teacher construction as the latest graph-Laplacian
# distribution method:
# - average all layer-0 heads into one 64x64 attention matrix A;
# - build the pairwise-max/Fiedler teacher candidate;
# - compare raw vs reverse by current-model train linear_profile_loss;
# - store the selected orientation as pairwise supervision.
#
# The run script copies the 50k random checkpoint into this out_dir before
# launching. We resume from that copy with lr=0 and no optimizer state, so the
# model acts as a fixed random50k base while the training loop is used only as a
# data-extraction driver.

_base_config = 'config/WikiText103/seq256/permute/block64/random.py'
exec(open(_base_config).read())

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-random50k-l0-layermean-fiedler-pairwise-try24-attn512-loss1024-6krecords'
init_from = 'resume'
resume_optimizer_state = False

# The source checkpoint is iter=50000. Collect exactly 6000 records:
# records 0..4999 -> train, records 5000..5999 -> val.
max_iters = 55999
lr_decay_iters = 55999
eval_interval = 100000
eval_iters = 20
log_interval = 10
always_save_checkpoint = False

# Keep the resumed 50k model fixed during extraction.
learning_rate = 0.0
min_lr = 0.0
warmup_iters = 0

wandb_log = False
wandb_run_name = 'seq256-random50k-l0-layermean-fiedler-pairwise-try24-attn512-loss1024-6krecords'

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
all_head_pairwise_dataset_start_iter = 50000
all_head_pairwise_dataset_stop_iter = 55999
all_head_pairwise_dataset_interval = 1

# Attention teacher input: 16 * 32 = 512 train samples per record.
all_head_pairwise_dataset_attention_batch_size = 16
all_head_pairwise_dataset_attention_micro_batches_per_record = 32
all_head_pairwise_dataset_export_type = 'without_none'
all_head_pairwise_dataset_attention_order_mode = 'random'
all_head_pairwise_dataset_out_dir = 'Report/language/wikitext103/mlp/distillation/try_24/l0_layermean_fiedler_random50k_attn512_loss1024_5k1k'
all_head_pairwise_dataset_shard_size = 64
all_head_pairwise_dataset_dtype = 'float16'
all_head_pairwise_dataset_seed = 271828
all_head_pairwise_dataset_max_records = 6000
all_head_pairwise_dataset_train_records = 5000

# L0 layer-mean sample only.
all_head_pairwise_dataset_heads = '0:mean'
all_head_pairwise_dataset_candidate_source = 'pairwise_max_fiedler'

# Direction/orientation probe: 64 * 16 = 1024 train samples per record.
all_head_pairwise_dataset_loss_batches = 64
all_head_pairwise_dataset_loss_batch_size = 16
all_head_pairwise_dataset_loss_candidate_batch_size = 2
all_head_pairwise_dataset_loss_prefix_k = 16
all_head_pairwise_dataset_loss_score = 'linear_profile'
all_head_pairwise_dataset_loss_exp_tau = 16.0
all_head_pairwise_dataset_store_pairwise_q = True
all_head_pairwise_dataset_resume_existing = True
