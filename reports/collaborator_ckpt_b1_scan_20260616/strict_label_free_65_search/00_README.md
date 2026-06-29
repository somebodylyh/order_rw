# Strict Label-Free 65-node None-separated head/method search

Protocol: node0=None/BOS, node 1+i = MODEL block i (NOT physical).
inv_perm is NEVER used during extraction or CDL rollout.
It is applied ONLY posthoc to translate sigma_model → sigma_phys for scoring.

Config: `{"M": 20, "batch_size": 8, "best_val_loss": 3.640097141265869, "ckpt": "/home/admin/ych/nanogpt-learned-order/ckpt/ckpt.pt", "extraction": "model-frame loss-aligned AR, [None] separate, NO inv_perm during construction", "fwd_batch": 8, "iter_num": 50000, "perm_orientation": "model_to_phys", "posthoc": "inv_perm applied only to translate sigma_model \u2192 sigma_phys for scoring", "protocol": "strict_label_free_65", "seed": 0, "shape": [4, 8, 64, 65]}`

Rows: 256
Strong pass: 20
Weak pass: 11

Compare with oracle-remapped version in `none_separated_65_head_method_search/`.
