# WikiText103 seq256 permuted block64 try_65.
#
# Example config for applying the latest seq readout to the block64 MLP line.
# It intentionally inherits try60 and changes only the attention readout family:
# L0 all-head mean target_to_observed attention instead of without_none.
#
# Stage schedule, seed, sample accounting, MLP objective, and no-EMA settings are
# kept matched to try60.

_base_config = (
    'config/WikiText103/seq256/permute/block64/'
    'attn_mlp_try60_seed2053_shadow_teacher_score_mse_train10k18k_'
    'shadowtrain512_alltrain_laststepdir_policyattn1024_fixed18k_anneal18k35k.py'
)
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

out_dir = (
    'out/base/permute/seq256/block64/'
    'out-wikitext103-seq256-try65-seed2053-targettoobserved-readout-'
    'shadow-teacher-score-mse-train10k18k-shadowtrain512-alltrain-'
    'laststepdir-policyattn1024-linearlabel-stop32k-prob08-freeze35k-'
    'b64-permute-block'
)

wandb_run_name = (
    'seq256-try65-seed2053-targettoobserved-readout-shadow-teacher-score-mse-'
    'train10k18k-shadowtrain512-alltrain-laststepdir-policyattn1024-'
    'linearlabel-stop32k-prob08-freeze35k-b64-permute'
)

# Latest readout method: predictor/target-aligned target_to_observed attention,
# aggregated exactly like try60 otherwise: layer 0, all heads, one matrix.
attn_mlp_policy_export_type = 'target_to_observed'

_report_root = (
    'Report/language/wikitext103/mlp/block/seq256/block64/distillation/'
    'tries_by_stage/07_target_to_observed_readout_examples/try_65'
)
attn_mlp_policy_shadow_mse_log_path = _report_root + '/input_attn/attn_mlp_shadow_mse_history.jsonl'
attn_mlp_policy_log_input_attn_prefix = (
    'try65_seed2053_targettoobserved_readout_shadow_teacher_score_mse_'
    'train10k18k_shadowtrain512_alltrain_laststepdir_policyattn1024_'
    'linearlabel_stop32k_prob08_freeze35k_input_attn'
)
attn_mlp_policy_log_input_attn_out_dir = _report_root + '/input_attn'

assert int(seed) == 2053
assert int(permute_seed) == 42
assert str(aogpt_train_mode) == 'AttnMLPFrozenOrder'
assert bool(attn_mlp_policy_enabled)
assert bool(attn_mlp_policy_random_init)
assert not bool(attn_mlp_policy_freeze)
assert str(attn_mlp_policy_train_loss) == 'teacher_score_mse'
assert not bool(attn_mlp_policy_use_global)
assert int(attn_mlp_policy_layer) == 0
assert int(attn_mlp_policy_head) == -1
assert str(attn_mlp_policy_export_type) == 'target_to_observed'
assert str(attn_mlp_policy_shadow_mse_label_orientation) == 'linear_profile_loss'
assert int(attn_mlp_policy_attention_batches_per_update) * int(batch_size) * int(gradient_accumulation_steps) == 1024
assert int(attn_mlp_policy_shadow_mse_batches_per_item) * int(batch_size) * int(gradient_accumulation_steps) == 512
assert int(attn_mlp_policy_shadow_mse_start_iter) == 10000
assert int(attn_mlp_policy_shadow_mse_stop_iter) == 18000
assert int(attn_mlp_policy_update_stop_iter) == 32000
assert int(attn_mlp_policy_anneal_end_iter) == 35000
assert not bool(attn_mlp_policy_attention_ema_enabled)
assert not bool(attn_mlp_policy_logits_ema_enabled)
assert not bool(online_spectral_policy_enabled)
assert not bool(head_signal_probe_enabled)
