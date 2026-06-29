# 65-node None-separated head/method search

Protocol: node0=None/BOS, node 1+i = physical content block i = x_{4i}..x_{4i+3}.
No training was run. Attention is token-level observation aggregated into a block-level graph.

Config: `{"M": 8, "batch_size": 8, "best_val_loss": null, "ckpt": "block_lo_arm_order_network/probe_results/clean_base_random_perm/ckpt_step0.pt", "fwd_batch": 8, "iter_num": 0, "perm_orientation": "phys_to_model", "seed": 0, "shape": [4, 8, 64, 65]}`

Rows: 96
Strong pass: 0
Weak pass: 0
