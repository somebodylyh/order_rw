# WikiText103 seq80 permuted block1 Attn-MLP try_3.
#
# Seed-only counterpart to try_2. Everything else is intentionally inherited:
# L0 all-head target_to_observed readout, teacher_score_mse, no EMA, and the
# block64 try60-style 10k-18k / 18k-32k / 32k-35k / 35k-50k schedule.

_base_config = (
    'config/WikiText103/seq80/permute/block1/seq_mlp/'
    'attn_mlp_seq80_try2_l0_allheads_targettoobserved_shadow_teacher_score_mse_'
    'train10k18k_shadow512_policy1024_stop32k_prob08_fixed35k.py'
)
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

seed = 2054
permute_seed = 42

out_dir = (
    'out/seq/permute/seq80/block1/'
    'out-wikitext103-seq80-attnmlp-try3-seed2054-l0-allheads-targettoobserved-'
    'shadow-teacher-score-mse-train10k18k-shadow512-policy1024-'
    'stop32k-prob08-fixed35k-b1-permute'
)

wandb_run_name = (
    'seq80-attnmlp-try3-seed2054-l0-allheads-targettoobserved-'
    'shadow-teacher-score-mse-train10k18k-shadow512-policy1024-'
    'stop32k-prob08-fixed35k'
)

_report_root = 'Report/language/wikitext103/mlp/token/seq80/try_3'
attn_mlp_policy_shadow_mse_log_path = _report_root + '/input_attn/attn_mlp_shadow_mse_history.jsonl'
attn_mlp_policy_log_input_attn_prefix = (
    'seq80_attnmlp_try3_seed2054_l0_allheads_targettoobserved_'
    'shadow_teacher_score_mse_train10k18k_shadow512_policy1024_'
    'stop32k_prob08_fixed35k_input_attn'
)
attn_mlp_policy_log_input_attn_out_dir = _report_root + '/input_attn'

assert int(seed) == 2054
assert int(permute_seed) == 42
assert str(attn_mlp_policy_export_type) == 'target_to_observed'
assert not bool(attn_mlp_policy_use_global)
assert int(attn_mlp_policy_layer) == 0
assert int(attn_mlp_policy_head) == -1
assert int(attn_mlp_policy_shadow_mse_start_iter) == 10000
assert int(attn_mlp_policy_shadow_mse_stop_iter) == 18000
assert int(attn_mlp_policy_update_stop_iter) == 32000
assert int(attn_mlp_policy_anneal_end_iter) == 35000
