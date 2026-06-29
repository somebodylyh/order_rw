# collab_extract_on_our_ckpt_L0H0

This directory stores block-attention analysis for the checkpoint
`/home/admin/lyuyuhuan/order_lyu/block_lo_arm_order_network/probe_results/clean_base_random_perm/ckpt_step10000.pt` evaluated with `mode=Random`.

Top-level groups:

- `with_none/`: predictor-aligned block attention including `[None]`
- `without_none/`: real-token-only block attention excluding `[None]`
- `diff/`: `with_none - without_none`

Inside each top-level group:

- `reveal/`: reveal-order block coordinates in the current permuted input frame
- `current_original/`: reordered back to the original block order of the current permuted input frame
- `true_original/`: reordered further back to the true unpermuted data block order
