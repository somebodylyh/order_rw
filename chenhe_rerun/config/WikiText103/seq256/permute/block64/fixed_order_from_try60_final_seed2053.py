# WikiText103 seq256 permuted block64 fixed-order run.
#
# This uses the final current-frame cached order from try60 as a deterministic
# backbone order for the whole run. It does not enable MLP/teacher updates,
# EMA, attention collection, or a new try-numbered report path.

_base_config = 'config/WikiText103/seq256/permute/block64/random.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

seed = 2053
permute_seed = 42

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-fixed-order-from-try60-final-seed2053-b64-permute-block-50000-iters'

init_from = 'scratch'
resume_optimizer_state = False
max_iters = 50000
lr_decay_iters = 50000
always_save_checkpoint = True
compile = False

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-fixed-order-from-try60-final-seed2053-b64-permute-50000-iters'

aogpt_train_mode = 'FixedBlockOrder'
main_eval_mode = 'Random'
generalization_eval_mode = ''

# try60 final cached_order_current at iter=31999.
# In permuted data this current-frame order maps to original-frame tau ~= 0.995.
fixed_block_order = (
    '11,32,38,42,19,22,55,16,9,61,4,29,45,57,15,62,'
    '37,28,43,30,53,60,8,34,1,35,63,47,36,5,12,50,'
    '21,27,41,18,44,6,0,56,51,20,40,46,14,49,24,17,'
    '2,39,10,7,13,59,58,26,25,23,31,33,54,48,52,3'
)

attn_mlp_policy_enabled = False
online_spectral_policy_enabled = False
online_spectral_enabled = False
head_signal_probe_enabled = False
head_direction_logger_enabled = False
head_asym_selector_enabled = False
