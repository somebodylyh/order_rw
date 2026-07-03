# Clean WikiText103 seq256 permuted block64 Random base for Attn-MLP resume.
#
# Purpose:
# - Train only the backbone with pure Random block order for the first 10k iters.
# - Do not attach or shadow-train any MLP/order policy.
# - Keep the original 50k LR schedule (`lr_decay_iters=50000`) so a later
#   10k->50k resume uses the same backbone LR phase as the standard baseline.

_base_config = 'config/WikiText103/seq256/permute/block64/random.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-random-b64-permute-block-clean-10k-for-attn-mlp-resume'

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-random-b64-permute-clean-10k-for-attn-mlp-resume'

max_iters = 10000
lr_decay_iters = 50000
always_save_checkpoint = True

aogpt_train_mode = 'Random'
main_eval_mode = 'Random'
generalization_eval_mode = ''
attn_mlp_policy_enabled = False
online_spectral_policy_enabled = False
head_signal_probe_enabled = False
head_direction_logger_enabled = False
