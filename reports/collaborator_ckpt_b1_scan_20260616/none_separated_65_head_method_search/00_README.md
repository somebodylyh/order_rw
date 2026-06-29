# 65-node None-separated head/method search

Protocol: node0=None/BOS, node 1+i = physical content block i = x_{4i}..x_{4i+3}.
No training was run. Attention is token-level observation aggregated into a block-level graph.

Config: `{"M": 20, "batch_size": 8, "best_val_loss": 3.640097141265869, "ckpt": "/home/admin/ych/nanogpt-learned-order/ckpt/ckpt.pt", "fwd_batch": 8, "iter_num": 50000, "perm_orientation": "model_to_phys", "seed": 0, "shape": [4, 8, 64, 65]}`

Rows: 256
Strong pass: 20
Weak pass: 11
