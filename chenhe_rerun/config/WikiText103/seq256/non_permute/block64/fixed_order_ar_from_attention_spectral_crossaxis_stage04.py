# WikiText103 seq256/block64 fixed learned-order AR run using the stage_04
# attention-spectral cross-axis order learned by the non_permute curriculum run.
#
# This mirrors the ImageNet64 fixed-order config:
# config/Imagenet64VQ_f4/seq256/non_permute/block64/
# fixed_order_ar_from_attention_spectral_crossaxis_stage06_l8h8e512.py
#
# Fixed reveal order:
#
#   [1, 2, 3, 0, 4, 5, 6, 7,
#    8, 9, 10, 11, 12, 13, 14, 15,
#    16, 17, 18, 19, 20, 21, 22, 23,
#    24, 25, 26, 27, 28, 29, 30, 31,
#    32, 33, 59, 60, 61, 62, 63, 36,
#    35, 34, 58, 57, 56, 55, 54, 53,
#    52, 51, 50, 49, 48, 47, 46, 45,
#    44, 43, 42, 41, 40, 39, 38, 37]
#
# Implementation note: train.py's segment-guided Random carrier is used as a
# fixed-order carrier. The paired json has one locked unit covering all 64
# blocks and segment_guided_ratio=1.0, so every training sample uses exactly
# this order. permute_data=False means block ids are in the original text frame.

out_dir = 'out/base/nonpermute/seq256/block64/out-wikitext103-seq256-block64-fixed-attn-spectral-crossaxis-stage04-ar-b64-nonpermute-50000-iters'
eval_interval = 250
eval_iters = 200
log_interval = 10

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-block64-fixed-attn-spectral-crossaxis-stage04-ar-b64-nonpermute-50000-iters'

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
segment_source_json = 'config/WikiText103/seq256/non_permute/block64/fixed_order_nonpermute_attention_spectral_crossaxis_stage04.json'
segment_max_units_per_order = 999999
segment_use_all_units = True

learning_rate = 1e-3
max_iters = 50000
lr_decay_iters = 50000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 0

# The active training policy is already the fixed learned order. Disable the
# extra generate-step diagnostic plot to keep eval checkpoints quick.
eval_generate_step_loss_log = False
