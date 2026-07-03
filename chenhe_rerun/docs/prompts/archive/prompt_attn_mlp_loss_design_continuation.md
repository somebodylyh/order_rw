> Archive note: this is a historical Attn-MLP loss-design continuation prompt.
> It remains useful for failure analysis, but current work should start from
> the Fiedler-teacher distillation/insertion path under
> `Report/language/wikitext103/mlp/distillation/`.
> Translate old `Report/attn_mlp_*` paths before reuse.

# WikiText103 Attn-MLP Loss Design Continuation Prompt

Use this prompt for a fresh agent continuing the WikiText103 / AO-GPT /
Attn-MLP loss-design exploration after the completed full-scale
`try1_energy_listwise_pl` run.

## Prior Conversation Recovery

The previous Codex conversation for this continuation can be resumed or
inspected with:

```bash
codex resume 019eb0b1-4a49-7981-b968-c210c2ce7686
```

If the VSCode Codex panel cannot show that conversation, the local session
record is still available at:

```text
/home/chenhe/.codex/sessions/2026/06/10/rollout-2026-06-10T16-41-09-019eb0b1-4a49-7981-b968-c210c2ce7686.jsonl
```

Use that transcript as context for what was attempted and why the run was
paused. Treat the repo files and `Report/attn_mlp_loss_design_exploration/`
artifacts as the ground-truth experiment record before launching any new run.

## Prompt

你现在进入 Goal + Plan 模式，同时开始一个持久目标和一个可执行计划。

目标：
在 `/home/chenhe/nanogpt-learned-order` 中，继续 WikiText103 / AO-GPT /
Attn-MLP 的 loss 设计探索。目标不是做 smoke test，而是系统探索一种
loss，使一个额外的 MLP 在 frozen AO-GPT 下能够学到有效的顺序分布。后续
计划是把这个 MLP 和主网络一起训练并共同收敛，得到一条好的 order；但当前
阶段必须先用 frozen 主网络验证 MLP 的 loss 是否真的能学到分布。

根基约束保持不变：

1. 不能使用任何 original-order 先验训练 MLP。
2. 不能拟合、蒸馏、回归由 attention 直接生成的一条 order。
3. 不能用 original L2R、original tau、OriginalL2R PPL、human scan、
   validation PPL shortcut 做训练、方向选择、模型选择、early stop 或
   loss 设计选择。
4. `original_l2r_tau` 只能作为实验完全结束后的 diagnostic / acceptance
   gate。
5. 不能 smoke test 通过就退出。smoke 只能检查代码能跑，不能作为结论。
6. 每个正式 try 必须记录完整实验结果和失败原因。
7. 只有同时满足以下硬性条件才允许退出为成功：
   - learned MLP order / distribution 在 frozen AO-GPT 下的 loss 明显优于
     random mode loss；
   - learned order 与 original L2R 的 post-hoc diagnostic Kendall tau
     `> 0.4`。
8. 如果没有达到这两个条件，不能宣称成功；必须继续提出下一种 loss 设计，
   或者给出明确、证据充分的 blocker。

当前研究问题：
之前实验说明 attention 等 current-frame signals 中确实可能包含有用信息，
但已有 loss 设计容易学到“对 frozen NLL 有用但不恢复 original L2R tau 的
非原始顺序”。现在重点不是证明 attention 有信息，而是探索如何设定 MLP 的
loss，让 MLP 学到一个好的顺序分布。

## 最新已完成实验

已经完成一个正式 full-scale try，不是 smoke test：

```text
Report/attn_mlp_loss_design_exploration/try1_energy_listwise_pl/
```

这个 try 使用：

- loss family: random-candidate frozen-NLL Plackett-Luce distribution
- frozen checkpoint:
  `out/base/permute/seq256/block64/out-wikitext103-seq256-random-b64-permute-block-50000-iters/ckpt.pt`
- train states: `2000`
- val states: `512`
- train order probes: `10000`
- epochs: `50`
- eval states: `512`
- random eval orders: `16`
- policy sample eval orders: `8`
- GPU: `CUDA_VISIBLE_DEVICES=1`

try1 的 loss 形式：
对每个 current-frame state，随机采样候选 order，评估每条 candidate order
的 frozen AO-GPT NLL：

```text
q_k = softmax(-(L_k - mean(L)) / target_loss_temp)
p_k = softmax(log PL_s(order_k) / pl_temp)
L = CE(q, p)
    + pairwise candidate contrast
    + expected-loss surrogate
    + entropy/std/diversity/index-bias regularizers
```

其中 attention 只是 MLP 输入，不被转换成 teacher order。没有使用 original
L2R / tau / OriginalL2R PPL 做训练、方向选择、模型选择或 early stop。

try1 的关键结果：
try1 没有成功。

```text
mlp_map_full_loss = 3.630112885031849
policy_sample_mean_full_loss = 3.631475218164269
random_mean_full_loss = 3.6321273927460425
random_best_full_loss = 3.621053149923682
mlp_minus_random_mean = -0.0020145077141933143
policy_sample_mean_minus_random_mean = -0.0006521745817735791
mlp_minus_random_best = 0.009059735108166933
forward_minus_reverse_loss = -0.00452768336981535
diagnostic/original_l2r_tau = 0.049938383556547616
```

接受条件检查：

```text
MAP loss clearly better than random mean: False
sampled distribution loss clearly better than random mean: False
forward better than reverse: True
diagnostic original_l2r_tau > 0.4: False
success: False
```

try1 的失败模式：
这个 loss 能学训练集里的静态 random candidate energy，但没有泛化成对
unseen current-frame states 有用的 order distribution。

证据：

```text
train candidate_top1_match: 0.208984375 -> 0.89208984375
val candidate_top1_match:   0.21875 -> 0.2109375
train loss_total:           2.398501396179199 -> 1.1627564653754234
val loss_total:             2.4221200942993164 -> 2.4645802080631256
val candidate_expected_minus_mean_loss: -0.00017892103642225266
```

解释：
静态 cached random-candidate energy target 可能让 MLP 记住了训练 candidate
set 的 idiosyncrasy。它能把 train candidates 排好，但没有学到稳健的顺序
分布。不要把这个结果解释为 “Attn-MLP 不可行”；它只说明 try1 这种静态
candidate listwise energy 不够。

## 需要先读的文件

1. `Report/attn_mlp_loss_design_exploration/SUMMARY.md`
2. `Report/attn_mlp_loss_design_exploration/HANDOFF_NEXT.md`
3. `Report/attn_mlp_loss_design_exploration/DESIGN.md`
4. `Report/attn_mlp_loss_design_exploration/try_summary.csv`
5. `Report/attn_mlp_loss_design_exploration/try1_energy_listwise_pl/failure_analysis.md`
6. `Report/attn_mlp_loss_design_exploration/try1_energy_listwise_pl/method.md`
7. `Report/attn_mlp_loss_design_exploration/try1_energy_listwise_pl/eval_summary.json`
8. `scripts/train/train_attn_mlp_order_distribution_energy.py`
9. 旧语义 loss 相关文件：
   - `Report/attn_mlp_loss_semantics/SUMMARY.md`
   - `Report/attn_mlp_loss_semantics/try_summary.csv`
   - `scripts/train/train_attn_mlp_loss_semantics.py`

## Signal Boundaries

Allowed current-frame model-side signals:

- current-frame attention as MLP input
- current-frame random-reveal block losses as MLP input
- random current-frame candidate orders
- frozen AO-GPT NLL of random candidate orders
- entropy / diversity / stability / transitivity regularizers that do not use
  original order
- reverse-order contrast as a model-side comparison

Forbidden signals:

- original L2R
- original tau
- OriginalL2R PPL
- human scan
- validation PPL shortcut
- fitting or distilling any attention-derived hard/soft teacher order

注意：
`diagnostic/original_l2r_tau` 只能在正式 try 完全结束后报告，不能用于训练、
方向选择、超参选择、模型选择、early stop 或 loss 设计选择。

## Next Try

下一步不要重复：
不要原样重复 try1 的 static cached random-candidate listwise energy。它已经
full-scale 失败，主要 blocker 是静态 candidate target 泛化差。

推荐下一次正式 try：
优先实现 “on-policy / resampled candidate energy”。

核心思想：
不要把 candidate orders 固定缓存成一个静态 dataset。每个 epoch 或 minibatch
重新采样 candidate orders，重新评估 frozen AO-GPT NLL，让 MLP 不能记住固定
候选集合。

建议 try2 方向：

```text
Try2: resampled / on-policy candidate energy

For each current-frame state:
1. MLP(A_current) -> logits s in R^64
2. sample or generate candidate orders from a mixture:
   - random orders
   - MLP MAP order
   - MLP Gumbel / Plackett-Luce sampled orders
   - local perturbations of MLP order
   - reverse MLP order as contrast
3. evaluate frozen AO-GPT NLL for candidates
4. define candidate energy target using only current-frame frozen NLL
5. train PL distribution / pairwise candidate preference / expected-NLL surrogate
6. refresh candidates every epoch or minibatch
```

候选 loss 可以从下面选择，但不能停在理论，必须实现并跑正式 try。

1. Resampled listwise energy:

```text
q_k = softmax(-(L_k - mean(L)) / tau_loss)
p_k = softmax(log PL_s(order_k) / tau_pl)
L = CE(q, p) + entropy/diversity regularization
```

2. Pairwise candidate-energy contrast:

```text
target(a better than b) = sigmoid((L_b - L_a) / tau_loss)
pred(a better than b) = sigmoid((log PL_s(a) - log PL_s(b)) / tau_pl)
L_pair = BCE(pred, target)
```

3. REINFORCE-style frozen-NLL reward:

```text
order_sample ~ PL_s
reward = -(NLL(order_sample) - random_baseline_NLL)
L_pg = - stopgrad(reward) * log P_s(order_sample)
```

4. Stability-augmented energy:
对同一个 data window 的多个 random reveals，要求 MLP distribution 兼容，但不能
用 attention-derived order 当 teacher。

执行策略：

- 先基于 try1 失败原因提出 2-4 个候选 loss。
- 默认优先做 resampled/on-policy candidate energy，因为它最直接测试 try1 的
  blocker。
- 可以先做 tiny smoke，只检查代码不崩；smoke 不能写成研究结论。
- smoke 通过后必须跑正式规模。
- 如果 try2 失败，必须写清失败模式，并提出 try3 的具体 loss 修改。

正式 try 规模不能小于：

- train samples >= 2000
- val samples >= 512
- pair/order probes >= 10000
- epochs >= 50
- eval samples >= 512
- random eval orders >= 16

每轮都必须比较：

- MLP MAP loss vs random mean loss
- MLP sampled-distribution loss vs random mean loss
- MLP loss vs random best loss
- forward order vs reverse order loss
- distribution entropy / collapse
- pairwise/listwise consistency
- train-vs-val candidate ranking generalization
- post-hoc `original_l2r_tau`

## Experiment Outputs

继续记录到：

```text
Report/attn_mlp_loss_design_exploration/
```

下一次正式 try 建议命名：

```text
Report/attn_mlp_loss_design_exploration/try2_resampled_candidate_energy/
```

每个正式 try 至少包含：

- `method.md`
- `failure_analysis.md`
- `eval_summary.json`
- `metrics.csv`
- `eval_rows.csv`
- `commands.sh`
- `policy.pt`
- 如有 replay/candidate cache，也保存路径
- 更新 `try_summary.csv`
- 更新 `SUMMARY.md`
- 更新 `HANDOFF_NEXT.md`

每个 try 必须说明：

- loss 公式
- allowed signals
- forbidden signals 如何隔离
- 样本数
- frozen checkpoint
- random baseline 定义
- selection criterion
- post-hoc diagnostic
- 为什么成功或失败
- 失败后下一步改什么

成功判断：
只有当以下两个条件同时满足，才能停止并报告成功：

```text
1. learned MLP order / distribution 在 frozen AO-GPT 下的 loss 明显优于 random mode loss
2. post-hoc diagnostic original_l2r_tau > 0.4
```

如果只满足 frozen NLL 优于 random，但 tau 仍低于 0.4，不能宣称成功；只能说学到
了 current-frame utility order，还没有恢复 original L2R 结构。

如果 tau 高但 frozen NLL 不优于 random，也不能宣称成功。

最终回复用户时必须用中文报告：

- 读了哪些文件
- 新实现或修改了哪些脚本
- 本次 try 的 loss 公式
- 样本数和正式规模是否达标
- frozen checkpoint 路径
- policy 路径
- report 路径
- MAP/random/reverse/sampled-distribution 指标
- train-vs-val 泛化情况
- post-hoc tau diagnostic，并明确它没有用于训练或选择
- 是否达到成功门槛
- 如果失败，下一步具体怎么改

非常重要：
不要把“attention 能恢复一条不错的信息”直接变成 teacher order supervision。这里要
探索的是：在不给 original-order 先验、不拟合 Attn order 的条件下，什么 loss
能让 MLP 自己学到好的 order distribution。
