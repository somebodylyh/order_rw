> Archive note: this is an older teacher-free / utility-oriented Attn-MLP
> prompt. For current work, first read
> `docs/prompts/prompt_language_block_current_task.md` and the Fiedler-teacher
> distillation reports. Translate old `Report/attn_mlp_*` paths before reuse.

你是一位严谨的 ML research engineer / research scientist。请基于我的 GitHub 项目
`Yangxiaohehehe/nanogpt-learned-order` 推进一个新的 WikiText103 / AO-GPT order-learning 实验。

请用中文和我沟通；代码、文件名、命令、日志字段保持英文。不要泛泛讲 learned order。
不要把还没跑的实验说成已经成功。凡是未完成、失败、不确定的部分必须明确说明。

============================================================
0. 当前我要做什么
============================================================

我现在不是单纯复现 LO-ARM 或 REOrder。我想在它们的基础上，开发一个更适合我
AO-GPT / nanogpt-learned-order 项目的 order-learning 方法。

核心思想：

    backbone AO-GPT 可以先固定不变；
    从当前模型 attention matrix 接一个 MLP；
    MLP 输出每个 current-frame block 的 priority logits；
    logits 诱导候选 reveal orders；
    用 frozen/current AO-GPT 的 current-frame NLL / token loss 判断候选 order 好不好；
    用这些候选的 utility 更新 MLP，使 MLP 学到能降低 PPL 的顺序。

这个实验的关键词是：

    Attention-conditioned, PPL-oriented order policy learning for AO-GPT

它不是：

    - 学 original L2R
    - 学 human scan
    - 蒸馏 original-frame Kendall tau
    - 直接让梯度穿过 argsort / hard permutation
    - 只复现 REINFORCE paper

它是：

    A_current -> MLP logits -> sampled candidate orders
    candidate orders -> AO-GPT current-frame NLL / prefix loss / attention score
    utilities -> update MLP order policy
    goal: low NLL / low PPL order

============================================================
1. 必读文件
============================================================

先读项目背景：

    README.md
    base.md
    base_cn.md
    docs/prompts/prompt_online_language_current.md
    docs/findings/findings_language.md
    docs/structure/PROJECT_STRUCTURE_language.md

再读核心代码：

    train.py
    AOGPT.py
    AOGPT_block.py
    AOGPT_token.py
    order_utils.py
    online_spectral_order_policy.py
    attn_mlp_order_policy.py

重点读 eval / analysis：

    scripts/eval/eval_lm_original_order_ppl.py
    scripts/eval/eval_lm_ppl_sweep.py
    scripts/eval/eval_lm_order_pool_ppl.py
    scripts/analysis/head_angle_tau_diagnostic.py
    scripts/analysis/seq80_random_ckpt_online_order_attn_batch.py
    scripts/analysis/seq80_loss_rerank_existing_attention_candidates.py

重点读 configs：

    config/WikiText103/seq256/permute/block64/attn_mlp_frozen_order.py
    config/WikiText103/seq256/non_permute/block64/online_spectral_fixed_head_order_loss_rerank.py
    config/WikiText103/seq256/non_permute/block64/online_spectral_order_distribution_loss_rerank.py
    config/WikiText103/seq256/permute/block64/online_spectral_fixed_head_order.py
    config/WikiText103/seq256/permute/block64/online_spectral_order_distribution.py

如果本轮上下文里有 LO-ARM 和 REOrder 两篇论文，也请阅读。它们的启发是：

    - order 是离散 policy / latent permutation；
    - 不能简单地让 gradient 直接穿过 hard order / argsort；
    - 可以用 Plackett-Luce / Gumbel-top-k 采样 permutation；
    - 用 task loss / reward / ELBO / ranking objective 更新 order policy。

但我的方法要适配 AO-GPT 和 WikiText103 PPL，不要只是照搬论文。

============================================================
2. 必须遵守的 frame 约束
============================================================

所有训练、candidate selection、utility、MLP input、MLP target、policy update 都只能使用
current-frame model-side signals。

允许使用：

    current-frame attention matrix
    current-frame attention path score
    current-frame directed attention score
    current-frame band quality
    current-frame train/probe teacher-forced token loss
    current-frame prefix loss
    current-frame full loss / NLL

禁止用于训练、target、candidate selection、policy update、early stop、model selection：

    original L2R order
    original-frame Kendall tau
    OriginalL2R PPL
    *_original diagnostics
    human scan / raster / Morton / Hilbert / column-major
    validation PPL for selecting order or MLP checkpoint

如果 permute_data=True：

    MLP logits / sampled orders / cached_order / candidate orders 都必须是 current-frame block ids。
    block_perm 和 *_original 只能用于 diagnostics，不得作为训练信号。
    OriginalL2R 是 oracle upper bound，不是 no-prior learned-order 结果。

如果 permute_data=False：

    current frame 等于自然文本 frame；
    但仍要描述为 current-model attention/loss signal，不是用 tau 或 oracle 选择顺序。

============================================================
3. 第一版主实验 setting
============================================================

第一版先做语言侧最稳的 block-level setting，不要先从 seq80/block1 开始。

主实验：

    dataset = WikiText103
    block_size = 256
    block_order_block_len = 4
    num_blocks = 64
    frame = permute / block64
    layer = 0
    head = 7
    export_type = with_none

建议 checkpoint：

    seq256 / permute / block64 random checkpoint
    或 seq256 / permute / block64 online spectral checkpoint

目标是先在 frozen backbone 上证明：

    MLP(A_current) 生成的 order 能比 random order 得到更低 current-frame NLL / lower PPL proxy。

============================================================
4. 模型设计
============================================================

使用或扩展：

    attn_mlp_order_policy.py::FlatAttentionOrderMLP

输入：

    A_current ∈ R^{N x N}

输出：

    s = MLP(A_current) ∈ R^N

语义：

    s[i] 越大，current-frame block i 越应该更早 reveal。

hard MAP order：

    order_map = argsort(s, descending=True)

因此 config 必须使用：

    attn_mlp_policy_order_mode = "argsort_desc"

不要使用默认 ascending argsort，除非你明确把 logit 方向反过来了。第一版统一：

    high logit = early reveal

============================================================
5. 为什么不能直接对 hard order 反传
============================================================

AO-GPT forward 需要显式 hard order：

    model(x, mode=None, orders=token_orders)

模型会按 orders shuffle token，再用 causal decoder 做 teacher-forced prediction。
所以：

    order = argsort(MLP(A))

这个 hard permutation / argsort 不能直接普通反传。

正确做法：

    用 MLP logits 定义一个 stochastic order policy；
    从 policy 采样或构造多条 candidate orders；
    用 frozen/current AO-GPT 给候选 order 打分；
    再用 order log-prob 或 per-block priority regression 更新 MLP。

============================================================
6. Candidate orders 如何生成
============================================================

不要枚举 64!。每次只构造一个小候选集 C。

第一版 candidate set：

    K ≈ 16

组成：

    1  MAP order:
        argsort(s, descending=True)

    8  Gumbel-top-k sampled orders:
        argsort(s / T + GumbelNoise, descending=True)

    4  Random orders:
        torch.randperm(num_blocks)

    2  Local mutations:
        对 MAP order 或 previous order 做 insert / swap / segment reverse

    1  Previous cached order:
        上一次 policy order，作为 stability anchor

可选：

    1-4 spectral candidates:
        从 online_spectral_order_policy.py recover_fixed_head_spectral_candidates(A)
        得到 top candidates，只作为 proposal，不作为 label。

Gumbel 的作用：

    给 logits 加随机噪声，从 Plackett-Luce / permutation policy 中采样完整 order。
    高 logit block 更容易靠前，但仍然保持探索。

Gumbel sampling 伪代码：

    def sample_gumbel_order(logits, temperature=0.7):
        eps = 1e-6
        u = torch.rand_like(logits).clamp(eps, 1 - eps)
        g = -torch.log(-torch.log(u))
        return torch.argsort(logits / temperature + g, descending=True)

推荐：

    gumbel_temperature = 0.7
    early training random candidates 比例高一点
    later training 可以减少 random，增加 MLP/Gumbel/local candidates

============================================================
7. Candidate utility 如何定义
============================================================

我的目标是低 PPL，所以 utility 必须以 AO-GPT current-frame NLL / full loss 为主。
attention score 只是结构 prior / proposal signal，不能替代 NLL。

对每条 candidate order o_m，计算：

    full_loss_m
    prefix_loss_m
    attention_score_m

其中：

    full_loss_m:
        candidate order 下 frozen/current AO-GPT 的 teacher-forced average token NLL。
        这是低 PPL 的主目标。

    prefix_loss_m:
        前 prefix_k 个 reveal blocks 的 average loss。
        保留 early reveal / curriculum 信号。

    attention_score_m:
        该 order 是否沿着 current attention graph 的强边走。
        可以复用 online_spectral_order_policy.py 里的 path score / directed score 逻辑。

推荐第一版 utility：

    U_m =
      -1.00 * z(full_loss_m)
      -0.25 * z(prefix_loss_m)
      +0.25 * z(attention_score_m)

其中 z 是在同一 candidate set 内做 zscore。

最简 smoke 版本可以先用：

    U_m = -z(full_loss_m)

如果这个最简版本完全没有 signal，不要急着加复杂项，先检查 candidate set 是否有好 order。

============================================================
8. MLP policy loss 设计
============================================================

MLP logits s 定义 Plackett-Luce permutation policy：

    P_phi(o | A)
      = product_t exp(s[o_t]) / sum_{j not yet selected} exp(s[j])

给定 candidate orders o_1...o_K 和 utilities U_1...U_K：

    q_m = softmax(U_m / tau_u)

其中：

    tau_u = 1.0 initially
    如果 utility noisy，tau_u 可增大到 1.5
    如果 utility 差异清楚，tau_u 可降低到 0.5

主 loss：

    L_PL = - sum_m stopgrad(q_m) * log P_phi(o_m | A)

含义：

    high utility order 应该有更高 policy probability。

辅助 per-block priority regression：

对候选 order o_m：

    pos_m(i) = block i 在 o_m 中的位置
    v_m(i) = 1 - pos_m(i) / (N - 1)

用 candidate weight 加权：

    r_i = sum_m q_m * v_m(i)

让 MLP logits 拟合这个 soft priority：

    L_prio = MSE(zscore(s), zscore(r))

最终第一版 loss：

    L = L_PL + 0.5 * L_prio

可选 regularization：

    L = L_PL
      + 0.5 * L_prio
      + 0.01 * KL(policy || old_policy)
      - 0.001 * entropy(policy)

但第一版请先实现：

    L = L_PL + 0.5 * L_prio

不要一开始做 PPO、straight-through argsort、soft sort、joint backbone+MLP backprop。

============================================================
9. Frozen-backbone first proof
============================================================

第一阶段请固定 backbone，只训练 MLP。

目的：

    在固定 AO-GPT checkpoint 下，证明 MLP policy 能从 attention 中学到低 NLL order。

步骤：

    1. load AO-GPT checkpoint
    2. model.eval()
    3. freeze all backbone params
    4. collect A_current from train/probe batches
    5. MLP(A_current) -> logits
    6. build candidate orders
    7. no_grad evaluate candidate full_loss / prefix_loss / attention_score
    8. compute utilities U
    9. update MLP with L_PL + 0.5 L_prio
    10. save replay buffer and metrics

不要一开始就同时更新 backbone 和 MLP。

============================================================
10. Replay buffer / cost control
============================================================

训练成本要控制。不要每次都在线暴力评估后丢掉结果。

建议使用 replay buffer：

    A_current
    candidate_orders
    candidate_utilities
    target_priority
    metadata

因为 backbone frozen，candidate utility 对同一 checkpoint 是相对稳定的，可以缓存。

推荐流程：

    Round 0:
        random + spectral + local proposals
        evaluate utilities
        train MLP on replay

    Round 1:
        用训练后的 MLP 生成 MAP/Gumbel proposals
        加 random/spectral/local proposals
        evaluate new utilities
        merge into replay
        train MLP

    Round 2:
        repeat once

一般 2-3 rounds 足够。

成本控制：

    smoke:
        num_A = 200
        K = 8
        probe_samples_per_candidate = 16

    main:
        num_A = 2,000
        K = 12 or 16
        probe_samples_per_candidate = 32

如果成本太高，使用 two-stage candidate filtering：

    generate 64 cheap proposals
    use attention_score to keep top 12
    add 4 random
    only evaluate token loss for 16 candidates

============================================================
11. Gate checks：确保真的能学到好顺序
============================================================

必须先做这些 sanity checks，不要直接跑大实验。

Gate 0: Candidate oracle sanity

    对每个 A，有 K 条 candidate。
    计算：

        candidate_oracle_loss = min_m full_loss(o_m)
        random_mean_loss = mean full_loss(random candidates)
        map_loss = full_loss(MAP order)
        spectral_loss = full_loss(spectral order), if available

    如果 candidate_oracle_loss 没有明显低于 random_mean_loss，
    说明候选池没有好 order，MLP 没东西可学。

Gate 1: Offline MLP improvement

    heldout A 上比较：

        MLP MAP full_loss
        MLP Gumbel best full_loss
        Random mean full_loss
        Candidate oracle full_loss
        Spectral proposal full_loss, if available

    成功标准：

        MLP MAP 或 MLP Gumbel best 比 random mean 更低；
        MLP utility percentile 上升；
        best candidate source 从 random/spectral 逐渐转向 MAP/Gumbel。

Gate 2: Stability

    记录：

        policy entropy
        logit std
        order_change_kendall_current
        random_best_rate
        map_best_rate
        gumbel_best_rate
        spectral_best_rate
        utility_std

    如果 entropy collapse 或 order 每轮乱跳，需要降低 lr、增大 random mix、加 KL。

Gate 3: Final PPL-oriented eval

    最终用 eval_lm_ppl_sweep.py 统一评估。
    不要跨 run 直接比较 train.py 默认 val/loss。

============================================================
12. 需要实现的脚本
============================================================

请优先实现或修改下面这些脚本。

A. Offline utility policy trainer:

    scripts/train/train_attn_mlp_utility_policy.py

功能：

    - load frozen AO-GPT checkpoint
    - collect A_current
    - build candidate orders
    - evaluate candidate utilities
    - compute L_PL + L_prio
    - train MLP
    - save replay buffer
    - save metrics
    - save policy.pt

B. Candidate oracle diagnostic:

    scripts/analysis/attn_mlp_candidate_oracle_diagnostic.py

功能：

    - 对每个 A 生成 candidates
    - evaluate full_loss / prefix_loss / attention_score
    - report oracle gap vs random
    - report best_source distribution

C. Eval trained MLP policy on frozen checkpoint:

    scripts/eval/eval_attn_mlp_policy_order_nll.py

功能：

    - load frozen backbone
    - load trained MLP
    - collect heldout A
    - generate MAP/Gumbel orders
    - compare NLL vs Random / Spectral / candidate oracle
    - save summary.json / csv / README

D. Optional config for later online/frozen train.py integration:

    config/WikiText103/seq256/permute/block64/attn_mlp_utility_policy_frozen_backbone.py
    config/WikiText103/seq256/permute/block64/attn_mlp_utility_policy_online.py

第一阶段先 offline，不要急着接 train.py full training。

============================================================
13. 推荐命令模板
============================================================

Candidate oracle smoke:

    python scripts/analysis/attn_mlp_candidate_oracle_diagnostic.py \
      --ckpt_path out/base/permute/seq256/block64/<RUN>/ckpt.pt \
      --out_dir Report/attn_mlp_utility/seq256_perm_b64/oracle_smoke \
      --dataset wikitext103 \
      --split train \
      --num_attention_samples 200 \
      --candidate_count 8 \
      --probe_batch_size 16 \
      --layer 0 \
      --head 7 \
      --export_type with_none \
      --utility full_loss \
      --device cuda \
      --dtype bfloat16

Offline utility MLP smoke:

    python scripts/train/train_attn_mlp_utility_policy.py \
      --ckpt_path out/base/permute/seq256/block64/<RUN>/ckpt.pt \
      --out_dir checkpoints/attn_mlp_utility/seq256_perm_b64_l0h7_withnone_smoke \
      --dataset wikitext103 \
      --split train \
      --num_attention_samples 200 \
      --candidate_count 8 \
      --probe_batch_size 16 \
      --hidden_dims 1024,1024 \
      --input_normalization robust_zscore \
      --lr 1e-4 \
      --epochs 20 \
      --loss_pl_weight 1.0 \
      --loss_prio_weight 0.5 \
      --gumbel_temperature 0.7 \
      --utility_temperature 1.0 \
      --device cuda \
      --dtype bfloat16

Main utility MLP:

    python scripts/train/train_attn_mlp_utility_policy.py \
      --ckpt_path out/base/permute/seq256/block64/<RUN>/ckpt.pt \
      --out_dir checkpoints/attn_mlp_utility/seq256_perm_b64_l0h7_withnone_main \
      --dataset wikitext103 \
      --split train \
      --num_attention_samples 2000 \
      --candidate_count 16 \
      --probe_batch_size 32 \
      --hidden_dims 1024,1024 \
      --input_normalization robust_zscore \
      --lr 1e-4 \
      --epochs 50 \
      --refresh_rounds 2 \
      --loss_pl_weight 1.0 \
      --loss_prio_weight 0.5 \
      --gumbel_temperature 0.7 \
      --utility_temperature 1.0 \
      --device cuda \
      --dtype bfloat16

============================================================
14. Checkpoint 格式
============================================================

保存的 MLP policy 必须兼容或容易改造成兼容 attn_mlp_order_policy.load_frozen_attn_mlp_policy：

    {
      "model_state_dict": model.state_dict(),
      "config": {
        "num_blocks": 64,
        "hidden_dims": [1024, 1024],
        "dropout": 0.0,
        "activation": "gelu",
        "input_normalization": "robust_zscore"
      },
      "policy_type": "attention_conditioned_utility_ranked_pl_policy",
      "target_direction": "larger_logit_reveals_earlier",
      "training_meta": {
        "ckpt_path": "...",
        "frame": "current",
        "layer": 0,
        "head": 7,
        "export_type": "with_none",
        "utility": "-full_loss - prefix_loss + attention_score",
        "candidate_sources": ["map", "gumbel", "random", "mutation", "previous", "spectral_optional"],
        "forbidden_signals": ["original_l2r", "original_tau", "OriginalL2R_ppl"]
      },
      "metrics": {...}
    }

============================================================
15. 最终报告
============================================================

请写报告：

    Report/attn_mlp_utility/seq256_perm_b64_l0h7_withnone/README.md

报告必须包含：

    1. 实验目标
    2. 和 LO-ARM / REOrder 的关系：借鉴 order-policy / PL / Gumbel / non-direct-gradient；但方法适配 AO-GPT PPL
    3. frame 约束
    4. candidate generation 设计
    5. utility 公式
    6. MLP loss 公式
    7. 成本估算
    8. candidate oracle sanity 结果
    9. offline MLP improvement 结果
    10. replay / refresh rounds
    11. failure cases
    12. 下一步是否值得接回 train.py

必须说明：

    - 这不是 original L2R supervision
    - 这不是 direct gradient through argsort
    - 这是 current-frame, PPL-oriented order policy learning
    - OriginalL2R 只作为 oracle diagnostic

============================================================
16. 完成任务后给我的回复格式
============================================================

请用中文总结：

    - 新增/修改了哪些文件
    - 读了哪些关键 repo 文件
    - 使用了哪个 checkpoint
    - 是否完成 candidate oracle diagnostic
    - oracle gap 是否存在
    - 训练了多少 attention samples / candidates
    - MLP loss 是否下降
    - heldout 上 MLP MAP/Gumbel 是否优于 random mean
    - policy.pt 保存路径
    - replay buffer 保存路径
    - 报告路径
    - 未完成或失败的部分
    - 下一步建议

不要只说“完成了”。必须给路径、命令、metrics 和 caveats。
