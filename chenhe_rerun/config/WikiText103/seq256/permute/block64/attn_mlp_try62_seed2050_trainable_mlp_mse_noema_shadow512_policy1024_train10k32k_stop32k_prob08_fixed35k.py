# WikiText103 seq256 permuted block64 try_62:
# Counterpart to try57. Keep seed and all baseline parameters fixed, but keep
# teacher-score MSE updates open through the 18k-32k policy-refresh phase.

_baseline_config = 'config/WikiText103/seq256/permute/block64/attn_mlp_try57_seed2050_shadow_teacher_score_mse_train10k20k_step10val2_fixed20k_anneal20k35k.py'
with open(_baseline_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-try62-seed2050-trainable-mlp-mse-online-noema-shadow512-policy1024-train10k32k-stop32k-prob08-fixed35k-b64-permute-block'
wandb_run_name = 'seq256-try62-seed2050-trainable-mlp-mse-online-noema-shadow512-policy1024-train10k32k-stop32k-prob08-fixed35k'

mse_train_stop = 32000
attn_mlp_policy_shadow_mse_stop_iter = mse_train_stop
attn_mlp_policy_shadow_mse_log_path = 'Report/language/wikitext103/mlp/distillation/try_62/input_attn/attn_mlp_shadow_mse_history.jsonl'
attn_mlp_policy_log_input_attn_prefix = 'try62_seed2050_trainable_mlp_mse_online_noema_shadow512_policy1024_train10k32k_stop32k_prob08_fixed35k_input_attn'
attn_mlp_policy_log_input_attn_out_dir = 'Report/language/wikitext103/mlp/distillation/try_62/input_attn'

assert seed == 2050
assert int(attn_mlp_policy_attention_batches_per_update) == 8
assert int(attn_mlp_policy_shadow_mse_batches_per_item) == 4
assert int(attn_mlp_policy_shadow_mse_train_items_per_update) == 2
assert int(attn_mlp_policy_update_stop_iter) == 32000
assert not bool(attn_mlp_policy_attention_ema_enabled)
assert not bool(attn_mlp_policy_logits_ema_enabled)
