# Spec — Image MLP-refresh Alternating (single-arm self-bootstrapping feasibility)

Date: 2026-05-24
Branch base: `attn-order-alternating`
Status: design approved by user; pending spec review → writing-plans

## 1. 定位 / Goal (读这一节定边界)

这一步**不是**严格跨-arm task baseline comparison,而是 **image MLP-refresh 自举闭环可行性验证**。

单臂实验回答的问题:
> 在 image 上,attention-derived MLP order 能不能随训练 refresh 起来,形成 **"order → attention → better order"** 的正反馈?

它**不**回答:它是否严格 beat random/raster training(那需要以后补 arm)。

**结论边界(写报告时严格遵守)**:
- 可以说:image MLP-refresh alternating **能/不能**自举出稳定 order,并在**同一模型内**形成 matched-order task gain。
- 不可以说:它已优于 random/raster training。
- 已有 `cont_random` (val_random≈7.69) / `cont_raster` (val_raster≈7.52) 仅作 **historical reference**,因为它们是 e2-continuation、与本 from-0 实验**不同 init,不能数字对位**。

背景与前置发现见 memory `image_substrate_bottleneck`:MLP readout 不是瓶颈;raster-curriculum 已证 order 对 task 有用且与 locality 无关;random-model attention 平、one-shot attention-derived order 还没赢——本实验测"refresh 迭代能否把它滚起来"。

## 2. 单臂定义 — `image_mlp_refresh_from0`

```
模型: random init image AOGPT
  arch = e2 同构: block_size=64, vocab_size=8192, n_layer=4, n_head=8,
                  n_embd=256, block_order_block_len=1, order_impl='block'
数据: Imagenet32VQ_f4_800k_seq64 (8×8=64 single-token VQ latent, GRID=8)
坐标系: from-0 un-permuted (physical raster = model coords;
        fixed_token_perm=None, inv_block_perm=None, 无 remap)

0–3k:   random-order warmup (alpha=0, 不提 B / 不蒸馏 / 不需 β)
3k 起每 3k refresh 一次 (3k,6k,...,30k):
  1. 从当前 θ inline 提取 A_global (64×64) → B = Aᵀ (build_directed_graph, 对角清零)
  2. 用 C-D+L teacher 训练/finetune 12-d MLP β (CPU, warm-start 上次 β)
  3. 用 GPU-batched MLP sampler 并行采样 block order
  4. 继续训练 θ (alpha-mixed: β-order vs random-order)
total: 30k

alpha schedule: 0 until 3k → 3k–13k 线性 ramp → 0.9 → 之后恒 0.9
sampler: image plain orientation (无 source_start, 无 reversed); tau_beta=0.5, top_k=4
teacher: 纯 C-D+L on B (attention-only); 不用 Manhattan distance
distill: finetune (β warm-start), CPU
输出: probe_results_image/vq64_alt_from0_mlp/
```

### C-D+L teacher 复用
`q = C - D + L`,C=mean B[S,v]、D=mean B[U\{v},v]、L=B[last,v](`attn_order_teacher`)。距离/几何**一律不用**。

## 3. 代码结构

- **新建 `scripts/train_vq64_alternating.py`**,**不动 `train_vq64_round2.py`**。
- 复用 `train_vq64_round2`:`AOGPT` 构造、`_forward_with_block_orders`、`evaluate_*`、`get_lr`、`get_alpha`(直接 import;from-0 不调 `load_baseline_model`,改为随机初始化同构模型)。
- 复用 attn_order 模块:`attn_order_distill.distill_order_mlp`(finetune/scratch + refresh_diagnostics)、`attn_order_teacher`、`attn_order_features`、`attn_order_mlp_policy`(batched sampler)。
- **新增**:
  - inline 图像 attention 提取(搬 `extract_image_attention_e2.extract_a_global` 逻辑:forward_fn random orders、全层全头平均、model-pos→physical remap、对角清零);
  - refresh + 蒸馏块(每 3k);
  - `val_mlp_order` eval 列;
  - `--alpha-warmup-start`(延迟 ramp 到 warmup 之后)。

## 4. Eval

每 `eval-interval` 保留现有 8 列,**新增 1 列**:

| 列 | 含义 |
|---|---|
| **val_mlp_order** (新增) | 同一模型在 **π_β order** 下的 NLL,用来判断模型是否逐渐适配自己学出的 order。order 由**最近一次 refresh 的 β + B** 在 eval 时现采(**eval 不重新提取 attention**,只复用上次 refresh 的 β/B;B/β 每 3k 才更新)。首次 refresh 前(<3k,β=None)该列置空/回退随机(守卫)。 |
| val_random / val_raster / val_hilbert / val_Bcov_balanced / val_distance_only_coverage / val_rw_top4_eps0 / val_rw_eps015 / val_rw_topk8 | 沿用 `train_vq64_round2.evaluate_7orders` 现成 |

## 5. 主判据(报告主看)

**(1) self-bootstrapping 是否成立** — 看每次 refresh(`refresh_diagnostics.jsonl`)是否一起变好:
- teacher/student KL 下降;top1 / top4 提高;B_edge_ratio 增强;attention peak/mean(max/mean)提高;`val_mlp_order` 随 refresh 下降。
- 一起变好 ⇒ loop 滚起来了。

**(2) MLP order 是否成为模型偏好的有效 order** — 同一模型内:
- `val_mlp_order < val_random`;或至少 `val_mlp_order` 持续下降并逐渐接近/超过其他 structured eval order。

**(3) 是否过度 specialization**:
- 若 `val_mlp_order` 下降 **但** `val_random / val_hilbert / val_Bcov` 大幅恶化 ⇒ 有效但 matched-order specialized(类似 raster)。
- 若 `val_mlp_order` 下降 **且** cross-order 没明显崩 ⇒ 更好。

## 6. 安全不变量(沿用 text alternating)

- B **只**由当前 θ inline 提取;**禁 `--a-block-path` 外部 A**(本脚本不接受外部 A)。
- teacher = 纯 C-D+L on B(attention-only),**不碰 image distance/manh**(不走 Bcov/distance 采样器路径)。
- 不用 readiness / source_start / reversed。
- warmup(alpha=0)阶段不采 β、不蒸馏(behavior-preserving)。
- 新代码隔离在 `train_vq64_alternating.py`;`train_vq64_round2.py` 与既有 run 不受影响。

## 7. 测试(TDD;先过再发车)

**单测**:
1. inline attention 提取与 `extract_image_attention_e2.extract_a_global` 在同一 ckpt + 同 seed 下数值一致。
2. GPU-batched MLP sampler 的打分/采样 == `attn_order_mlp_policy` 的 canonical numpy build_features+MLP 参考(沿用已有测式)。
3. refresh 后 B 确实改变(非 frozen):连续两次 refresh 的 B 在 θ 更新后不 bit-identical。
4. alpha schedule:`--alpha-warmup-start` 下 step<3k → alpha=0,3k–13k ramp,≥13k → 0.9。
5. 坐标系:un-permuted forward(`fixed_token_perm=None`)与 block-order→token-order 展开正确。

**GPU smoke**:~400 步 from-0(缩小 warmup/refresh,如 warmup100/refresh100)端到端跑通;β 在首 refresh 诞生;first eval finite(无 NaN);`val_mlp_order` 列正常写出。

**发车**:spec + 单测 + 400-step smoke 全过后,再启动单臂 30k(~1hr,单卡;自等空卡,chenhe 占 GPU1)。

## 8. 输出物

- `probe_results_image/vq64_alt_from0_mlp/`:`config.json`、`eval_curve.tsv`(含 `val_mlp_order`)、`train_log.txt`、`refresh_diagnostics.jsonl`、`A_global_step{N}.npy`、`beta_step{N}.pt`、`ckpt_step{N}.pt`(save-steps 同 text:如 3k/15k/30k)。
- 报告读法严格按 §5 主判据 + §1 结论边界。
