# WikiText103 seq80 non-permuted token-level online spectral distribution.
# Schedule:
# - 0..20000: pure random warmup, no online probe/rerank updates
# - 20000..40000: update every 10 steps and train from the learned distribution
# - 40000..50000: stop updates and freeze to the current MAP order
#
# This is the top-1 variant: loss rerank still scores a broad candidate set, but
# only the best reranked order is converted into the priority EMA.

_base_config = 'config/WikiText103/seq80/non_permute/block1/online_spectral_order_distribution.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

out_dir = 'out/base/nonpermute/seq80/block1/out-wikitext103-seq80-online-spectral-order-distribution-loss-rerank-top1-warmup20k-dist20k40k-freeze10k-update10-top96-prefix16-kall-lowtemp-no-random-mix-b1-nonpermute-50000-iters'
wandb_run_name = 'seq80-online-spectral-order-distribution-loss-rerank-top1-warmup20k-dist20k40k-freeze10k-update10-top96-prefix16-kall-lowtemp-no-random-mix-b1-nonpermute-50000-iters'
compile = False
eval_batch_size = 64

# First recovery happens at iter 20000. The training batch at iter 20000 still
# uses random because the distribution is created after the probe/update.
online_spectral_policy_update_start_iter = 20000
online_spectral_policy_anneal_start_iter = 20000
online_spectral_policy_anneal_end_iter = 20001
online_spectral_policy_start_prob = 0.0
online_spectral_policy_end_prob = 1.0
online_spectral_policy_fallback = 'random'

online_spectral_policy_update_every = 10
online_spectral_policy_update_stop_iter = 40000
online_spectral_policy_freeze_to_map_order_after_stop = True

# Keep the attention probe budget comparable to the seq80 curriculum-side run.
online_spectral_policy_probe_batches = 16
online_spectral_policy_probe_batch_size = 256

# Generate the same broader spectral family used by the offline curriculum.
online_spectral_policy_k_values = '4,6,7,8,9,10,12'

# Top-1 distribution: use only the best reranked candidate for each EMA update.
online_spectral_policy_top_m = 1
online_spectral_policy_teacher_temperature = 1.0
online_spectral_policy_score_normalization = 'zscore'
online_spectral_policy_priority_ema_decay = 0.95
online_spectral_policy_sample_temperature = 0.1
online_spectral_policy_random_mix_prob = 0.0
online_spectral_policy_distribution_per_sample = False

online_spectral_policy_loss_rerank_enabled = True
online_spectral_policy_loss_rerank_top_k = 96
online_spectral_policy_loss_rerank_batches = 4
online_spectral_policy_loss_rerank_batch_size = 128
online_spectral_policy_loss_rerank_candidate_batch_size = 1
online_spectral_policy_loss_rerank_prefix_k = 16
online_spectral_policy_loss_rerank_attention_weight = 1.0
online_spectral_policy_loss_rerank_prefix_weight = 0.4
online_spectral_policy_loss_rerank_full_weight = 0.1

online_spectral_policy_order_history_top_candidates = 16
