# WikiText103 seq80 non-permuted token-level online spectral fixed-head config.
#
# Schedule:
# - 0..5000: pure random training orders, no online probe/rerank updates
# - 5000..15000: linearly anneal fixed-head policy usage from 0 to 1
# - 15000..50000: train with the latest fixed-head order, still updating online
# - no final stop/freeze stage
#
# Rerank budget:
# - attention probe: 16 x 256 = 4096 samples
# - loss rerank:     96 candidates x (4 x 128 = 512 samples)

_base_config = 'config/WikiText103/seq80/non_permute/block1/online_spectral_cached_order.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

out_dir = 'out/base/nonpermute/seq80/block1/out-wikitext103-seq80-online-spectral-fixed-head-loss-rerank-warmup5k-anneal15k-update20-top96-prefix16-kall-no-freeze-b1-nonpermute-50000-iters'
wandb_run_name = 'seq80-online-spectral-fixed-head-loss-rerank-warmup5k-anneal15k-update20-top96-prefix16-kall-no-freeze-b1-nonpermute-50000-iters'
compile = False
eval_batch_size = 64

online_spectral_policy_update_start_iter = 5000
online_spectral_policy_anneal_start_iter = 5000
online_spectral_policy_anneal_end_iter = 15000
online_spectral_policy_start_prob = 0.0
online_spectral_policy_end_prob = 1.0
online_spectral_policy_fallback = 'random'

online_spectral_policy_update_every = 20
online_spectral_policy_update_stop_iter = -1
online_spectral_policy_freeze_to_map_order_after_stop = False

# Keep the probe budget aligned with the seq80 distribution runs.
online_spectral_policy_probe_batches = 16
online_spectral_policy_probe_batch_size = 256

# Generate the same broader spectral family used by the seq80 distribution runs.
online_spectral_policy_k_values = '4,6,7,8,9,10,12'

online_spectral_policy_loss_rerank_enabled = True
online_spectral_policy_loss_rerank_top_k = 96
online_spectral_policy_loss_rerank_batches = 4
online_spectral_policy_loss_rerank_batch_size = 128
online_spectral_policy_loss_rerank_candidate_batch_size = 1
online_spectral_policy_loss_rerank_prefix_k = 16
online_spectral_policy_loss_rerank_attention_weight = 1.0
online_spectral_policy_loss_rerank_prefix_weight = 0.4
online_spectral_policy_loss_rerank_full_weight = 0.1

# Fixed-head caches the single best reranked order, but keeps top candidates in
# the jsonl history so we can inspect stability and near-ties later.
online_spectral_policy_top_m = 1
online_spectral_policy_order_history_top_candidates = 16
