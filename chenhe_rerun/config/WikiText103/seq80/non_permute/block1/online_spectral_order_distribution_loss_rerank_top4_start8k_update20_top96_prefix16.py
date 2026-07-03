# WikiText103 seq80 non-permuted token-level online spectral distribution.
# This keeps a distribution policy, but makes it behave closer to the
# successful offline single-order curriculum:
# - wait until 8k random warmup before probing/updating
# - rerank a wider candidate set
# - convert the best four reranked candidates into the priority distribution
# - train from a low-temperature shared sampled order, then freeze MAP at 40k

_base_config = 'config/WikiText103/seq80/non_permute/block1/online_spectral_order_distribution.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

out_dir = 'out/base/nonpermute/seq80/block1/out-wikitext103-seq80-online-spectral-order-distribution-loss-rerank-top4-start8k-update20-top96-prefix16-kall-lowtemp-no-random-mix-freeze40k-map-b1-nonpermute-50000-iters'
wandb_run_name = 'seq80-online-spectral-order-distribution-loss-rerank-top4-start8k-update20-top96-prefix16-kall-lowtemp-no-random-mix-freeze40k-map-b1-nonpermute-50000-iters'
compile = False
eval_batch_size = 64

# Curriculum-like schedule: pure random until the model has a useful attention
# signal, then slowly hand over to the learned distribution.
online_spectral_policy_update_start_iter = 8000
online_spectral_policy_anneal_start_iter = 8000
online_spectral_policy_anneal_end_iter = 30000
online_spectral_policy_start_prob = 0.0
online_spectral_policy_end_prob = 1.0
online_spectral_policy_fallback = 'random'

online_spectral_policy_update_every = 20
online_spectral_policy_update_stop_iter = 40000
online_spectral_policy_freeze_to_map_order_after_stop = True

# Keep the attention probe budget comparable to the previous seq80 run.
online_spectral_policy_probe_batches = 16
online_spectral_policy_probe_batch_size = 256

# Generate the same broader spectral family used by the offline curriculum.
online_spectral_policy_k_values = '4,6,7,8,9,10,12'

# Distribution mode remains enabled, but top_m=4 keeps only a small local
# ensemble of the best reranked directions.
online_spectral_policy_top_m = 4
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
