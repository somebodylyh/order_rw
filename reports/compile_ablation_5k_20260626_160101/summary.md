# Compile / No-Compile 5k Ablation

Run date: 2026-06-26

## Setup

- Root: `block_lo_arm_order_network/probe_results/compile_ablation_5k_20260626_160101`
- Runs:
  - `compile_a`, `compile_b`: default `torch.compile(mode="reduce-overhead")`
  - `nocompile_a`, `nocompile_b`: `--no-compile-model`
- Shared config:
  - `run_kind=baseline`
  - `data_source=continuous`
  - `seed=123`, `permute_seed=123`
  - `max_steps=5000`
  - `lr_decay_steps=50000`
  - `save_steps=0,5000`
  - no W&B, no attention trajectory

## Step 5k Eval

| run | train_loss | val_train_objective | val_ori_l2r | val_unstructured | lr |
|---|---:|---:|---:|---:|---:|
| compile_a | 4.0691 | 4.1260 | 3.9533 | 4.1260 | 9.7798e-04 |
| compile_b | 4.0648 | 4.1257 | 3.9666 | 4.1257 | 9.7798e-04 |
| nocompile_a | 4.0665 | 4.1289 | 3.9789 | 4.1289 | 9.7798e-04 |
| nocompile_b | 4.0778 | 4.1311 | 3.9844 | 4.1311 | 9.7798e-04 |

Losses are effectively matched across all four runs at step 5k.

## Weight Comparison at Step 5k

| pair | total cosine | mean abs diff | c_attn cos | c_proj cos | mlp cos | wte cos | wpe cos | q/k norm cos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compile_a vs compile_b | 0.7950 | 0.02791 | 0.3428 | 0.3481 | 0.4808 | 0.7481 | 0.4862 | 0.9871 |
| nocompile_a vs nocompile_b | 0.8020 | 0.02746 | 0.3501 | 0.3796 | 0.5099 | 0.7545 | 0.4570 | 0.9866 |
| compile_a vs nocompile_a | 0.8325 | 0.02528 | 0.4559 | 0.4793 | 0.5862 | 0.7844 | 0.4500 | 0.9913 |
| compile_b vs nocompile_b | 0.8262 | 0.02582 | 0.4319 | 0.4551 | 0.5732 | 0.7759 | 0.4781 | 0.9899 |

Disabling `torch.compile` does not make repeated same-config runs bitwise or trajectory deterministic. The no-compile repeat pair is about as divergent as the compile repeat pair.

## Strict65 CDL Audit at Step 5k

Protocol: L0 all heads, strict65, physical-frame B, `M=500`, `bs_mean=4`, fixed probe seed 0, fast quality with `n_destroy_replicas=0`.

| run | consensus_tau_l2r | consensus_tau_rev_l2r | top_head | top_head_tau_l2r | top_head_abs_tau_gt_0.5 | near_diag_mass |
|---|---:|---:|---|---:|---:|---:|
| compile_a | +0.0373 | -0.0373 | H3 | +0.1730 | 0.016 | 0.3885 |
| compile_b | +0.0730 | -0.0730 | H6 | +0.2288 | 0.080 | 0.3769 |
| nocompile_a | +0.0215 | -0.0215 | H2 | -0.2941 | 0.128 | 0.3900 |
| nocompile_b | +0.1193 | -0.1193 | H2 | +0.3055 | 0.212 | 0.3914 |

All four 5k static CDL consensus signals are weak. Per-head top signals vary by run, including a sign flip for H2 between `nocompile_a` and `nocompile_b`.

## Interpretation

This ablation does not support "`torch.compile` alone caused the random-baseline divergence." Repeated same-config runs diverge substantially even with `--no-compile-model`; total weight cosine is about 0.80 in both compile and no-compile repeat pairs, while losses remain nearly identical.

The likely interpretation is broader CUDA/training nondeterminism plus a highly degenerate random-order objective. `torch.compile` may still be one nondeterministic source, but disabling it is not sufficient. Remaining suspects include fused AdamW, CUDA reduction/kernel ordering, TF32/matmul behavior, and general chaotic amplification of tiny numerical differences.

For experiment design, the conclusion remains: static random-baseline CDL tau is not a stable property of the training setup. The reliable test is same-checkpoint paired intervention: random continuation vs CDL/g_beta continuation from exactly the same checkpoint.
