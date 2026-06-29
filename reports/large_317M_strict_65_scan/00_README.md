# Strict Label-Free 65-node None-separated head/method search

Protocol: node0=None/BOS, node 1+i = MODEL block i (NOT physical).
inv_perm is NEVER used during extraction or CDL rollout.
It is applied ONLY posthoc to translate sigma_model → sigma_phys for scoring.

Config: `{"M": 40, "batch_size": 4, "best_val_loss": null, "ckpt": "block_lo_arm_order_network/probe_results/large_random_baseline_16l16h1024d/ckpt_step5000.pt", "extraction": "model-frame loss-aligned AR, [None] separate, NO inv_perm during construction", "fwd_batch": 4, "iter_num": 5000, "perm_orientation": "model_to_phys", "posthoc": "inv_perm applied only to translate sigma_model \u2192 sigma_phys for scoring", "protocol": "strict_label_free_65", "seed": 42, "shape": [16, 16, 64, 65]}`

Rows: 512
Strong pass: 0
Weak pass: 0

Compare with oracle-remapped version in `none_separated_65_head_method_search/`.
