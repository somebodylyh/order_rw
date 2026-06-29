# g_beta Loss Comparison

| Run | Loss | Tau | Pairwise | Prefix@8 | Prefix@16 | Train sec | Peak MB |
|---|---|---:|---:|---:|---:|---:|---:|
| listmle | listmle | 0.9840 | 0.9920 | 0.9556 | 0.9937 | 73.9 | 0.0 |
| pairwise_bce | pairwise_bce | 0.9835 | 0.9918 | 0.9731 | 0.9934 | 80.9 | 0.0 |
| rank_kl_tau16 | rank_kl | 0.9796 | 0.9898 | 0.9725 | 0.9937 | 72.2 | 0.0 |
| rank_kl_tau2 | rank_kl | 0.5071 | 0.7536 | 0.9775 | 0.8313 | 67.9 | 0.0 |
| rank_kl_tau4 | rank_kl | 0.7057 | 0.8528 | 0.9769 | 0.9931 | 69.4 | 0.0 |
| rank_kl_tau8 | rank_kl | 0.9362 | 0.9681 | 0.9737 | 0.9937 | 69.2 | 0.0 |

Reference Pairwise BCE: tau 0.984, pairwise accuracy 0.997, Prefix@8 0.973.

Do not compare primary loss values across different objective types.
