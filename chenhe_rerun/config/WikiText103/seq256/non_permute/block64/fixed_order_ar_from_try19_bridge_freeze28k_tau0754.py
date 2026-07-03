# WikiText103 seq256/block64 fixed-order AR run using the try19-bridge
# freeze28k MAP order from the non_permute distribution experiment.
#
# Fixed reveal order:
#
#   [17, 15, 18, 11, 14, 16, 9, 13,
#    25, 1, 6, 4, 19, 8, 10, 12,
#    20, 0, 29, 21, 5, 7, 2, 3,
#    22, 23, 26, 24, 32, 27, 28, 30,
#    31, 37, 36, 33, 39, 34, 35, 40,
#    38, 41, 42, 43, 44, 45, 48, 47,
#    50, 46, 49, 51, 62, 61, 63, 60,
#    52, 59, 58, 53, 54, 57, 55, 56]
#
# Implementation note: train.py's segment-guided Random carrier is used as a
# fixed-order carrier. The paired json has one locked unit covering all 64
# blocks and segment_guided_ratio=1.0, so every training sample uses exactly
# this order. permute_data=False means block ids are in the original text frame.

out_dir = 'out/base/nonpermute/seq256/block64/out-wikitext103-seq256-block64-fixed-try19-bridge-freeze28k-tau0754-ar-b64-nonpermute-50000-iters'
eval_interval = 250
eval_iters = 200
log_interval = 10

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-block64-fixed-try19-bridge-freeze28k-tau0754-ar-b64-nonpermute-50000-iters'

dataset = 'wikitext103'
data_record_mode = 'stream'
batch_size = 64
block_size = 256
gradient_accumulation_steps = 2
permute_data = False
permute_seed = 42

model_type = 'aogpt'
train_stage = 'standard'
aogpt_train_mode = 'Random'
main_eval_mode = 'AR'
generalization_eval_mode = ''
n_layer = 4
n_head = 8
n_embd = 384
dropout = 0

block_order_block_len = 4
block_order_layout = 'contiguous'

segment_guided_ratio = 1.0
segment_source_json = 'config/WikiText103/seq256/non_permute/block64/fixed_order_nonpermute_try19_bridge_freeze28k_tau0754.json'
segment_max_units_per_order = 999999
segment_use_all_units = True

learning_rate = 1e-3
max_iters = 50000
lr_decay_iters = 50000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 0

# The active training/eval policy is already the fixed try19 order. Disable the
# extra generate-step diagnostic plot to keep eval checkpoints quick.
eval_generate_step_loss_log = False
