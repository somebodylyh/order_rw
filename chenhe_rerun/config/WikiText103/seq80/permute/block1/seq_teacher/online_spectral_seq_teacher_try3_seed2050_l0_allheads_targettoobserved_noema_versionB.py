# WikiText103 seq80 permuted block1 seq-teacher try_3.
#
# Matched seed follow-up to try_2. All method/schedule/sample/readout settings
# are inherited from try_2; only the global training seed and run identity paths
# are changed.

_parent_config = (
    'config/WikiText103/seq80/permute/block1/seq_teacher/'
    'online_spectral_seq_teacher_try2_l0_allheads_targettoobserved_noema_versionB.py'
)
with open(_parent_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

seed = 2050

out_dir = (
    'out/seq/permute/seq80/block1/'
    'out-wikitext103-seq80-seqteacher-try3-seed2050-online-spectral-'
    'l0-allheads-targettoobserved-noema-versionB-update20-warmup10k-'
    'stop32k-prob08-freeze35k-b1-permute'
)

wandb_run_name = (
    'seq80-seqteacher-try3-seed2050-online-spectral-l0-allheads-'
    'targettoobserved-noema-versionB-update20-warmup10k-stop32k-'
    'prob08-freeze35k'
)

_report_root = (
    'Report/language/wikitext103/order_teacher_distribution/token/seq80/try_3'
)
head_signal_probe_out_dir = _report_root + '/online_training_probe'

online_spectral_policy_log_input_attn_prefix = (
    'seq80_seqteacher_try3_seed2050_l0_allheads_targettoobserved_'
    'noema_versionB_input_attn'
)
online_spectral_policy_log_input_attn_out_dir = _report_root + '/input_attn'

assert int(seed) == 2050
assert int(permute_seed) == 42
assert int(head_signal_probe_seed) == 24681357
assert str(head_signal_probe_heads) == '0:mean'
assert str(head_signal_probe_export_type) == 'target_to_observed'
assert int(head_signal_probe_batches) * int(head_signal_probe_batch_size) == 1024
assert int(head_signal_probe_loss_batches) * int(head_signal_probe_loss_batch_size) == 512
assert not bool(online_spectral_policy_use_ema)
assert not bool(attn_mlp_policy_enabled)
