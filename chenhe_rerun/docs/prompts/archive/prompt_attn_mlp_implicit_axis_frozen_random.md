> Archive note: this is an older implicit-axis frozen-random Attn-MLP prompt.
> It is provenance, not the current entry point. Current work should start from
> the L0 layer-mean pairwise-max Fiedler distillation/insertion reports.

你是一位严谨的 ML research engineer / research scientist。请基于我的 GitHub 项目
`Yangxiaohehehe/nanogpt-learned-order` 推进一个新的 WikiText103 / AO-GPT order-learning 实验。

请用中文和我沟通；代码、文件名、命令、日志字段保持英文。不要泛泛讲 learned order。
不要把未实际运行的实验说成已经成功。任何失败、未完成、不确定的地方必须明确说明。

============================================================
0. 当前实验一句话
============================================================

我要做的不是复现 LO-ARM 或 REOrder，也不是继续做 top-k attention-spectral teacher 蒸馏。

我要测试一个更直接的问题：

    在一个已经训练好的 random-order AO-GPT checkpoint 上，
    冻结 backbone，
    只用 current-frame attention matrix A_current 构造一个无外部顺序先验的 loss，
    训练一个 Attn-MLP 输出每个 current-frame block 的 priority logits，
    看它能不能学出非平凡、有结构、可能低 NLL / 高 tau diagnostic 的 order。

核心实验名：

    Frozen Random Checkpoint Implicit Attention-Axis Attn-MLP

核心思想：

    A_current -> MLP -> logits s ∈ R^N
    s_i 越大，current-frame block i 越早 reveal

    不先算 a_signed target。
    不做 top-k candidate order selection。
    不用 original L2R / original tau / human scan。
    直接把 attention graph smoothness + directed attention preference 写成 loss。

最终 hard order：

    order = argsort(s, descending=True)

============================================================
1. 必须遵守的约束
============================================================

训练和选择只能使用 current-frame model-side signals。

允许：

    current-frame attention matrix A_current
    robust-normalized attention
    symmetric attention affinity graph
    directed attention asymmetry
    frozen AO-GPT current-frame teacher-forced full_loss / prefix_loss for diagnostics only
    current-frame random-order baselines

禁止用于训练 loss / model selection / direction selection：

    original L2R order
    original-frame Kendall tau
    OriginalL2R PPL
    *_original diagnostics
    human scan / raster / Morton / Hilbert / column-major
    validation PPL for selecting order or checkpoint

如果 permute_data=True：

    MLP logits / learned order / cached order 必须是 current-frame block ids。
    block_perm 和 *_original 只能用于 logging / diagnostics。
    OriginalL2R 是 oracle upper bound，不是 no-prior result。

重要：

    本实验中的 tau 只能作为 diagnostic。
    不能用 tau 选模型、选方向、选超参、early stop。
    主选择指标应该是 current-frame objective / heldout objective / frozen checkpoint NLL vs Random。
    使用GPU1进行实验

============================================================
2. 必读文件
============================================================

先读项目背景：

    README.md
    base.md
    base_cn.md
    docs/prompts/prompt_online_language_current.md
    docs/findings/findings_language.md
    docs/structure/PROJECT_STRUCTURE_language.md

核心代码：

    train.py
    AOGPT.py
    AOGPT_block.py
    AOGPT_token.py
    order_utils.py
    online_spectral_order_policy.py
    attn_mlp_order_policy.py

重点函数 / 模块：

    train.py:
        _aggregate_layerhead_attention_to_current_blocks
        _attn_mlp_attention_matrix_from_outputs
        _forward_with_explicit_block_orders
        _forward_with_active_training_policy

    AOGPT_block.py:
        forward(...)
        forward_fn(...)
        sample_random_orders(...)
        set_ascending_orders(...)

    order_utils.py:
        expand_block_orders_to_token_orders
        token_losses_to_block_losses
        build_fixed_block_permutation
        invert_permutation

    online_spectral_order_policy.py:
        robust_z
        affinity_from_adjacency
        sym
        anti
        spectral_coordinates
        score_order

    attn_mlp_order_policy.py:
        FlatAttentionOrderMLP
        logits_to_order

重点 config：

    config/WikiText103/seq256/permute/block64/random.py
    config/WikiText103/seq256/permute/block64/attn_mlp_frozen_order.py
    config/WikiText103/seq256/non_permute/block64/online_spectral_fixed_head_order_loss_rerank.py
    config/WikiText103/seq80/non_permute/block1/random_save_attn_ckpts_1k2k5k8k10k15k20k25k50k.py

============================================================
3. 第一版实验 setting
============================================================

优先使用：

    WikiText103 / seq256 / permute / block64
    block_size = 256
    block_order_block_len = 4
    num_blocks = 64
    layer = 0
    head = 7
    export_type = "with_none"

优先 checkpoint：

    一个已经训练好的 random-order block64 checkpoint。

请先在 repo 中查找类似路径：

    out/base/permute/seq256/block64/*random*b64*permute*/ckpt.pt
    out/base/permute/seq256/block64/out-wikitext103-seq256-random-b64-permute-block-50000-iters/ckpt.pt

如果不存在，搜索 README / Report / config 中记录的 random block64 checkpoint 路径。
如果仍找不到，请明确告诉我缺少 checkpoint，不要编造路径。

本阶段：

    backbone frozen
    MLP trainable
    不接回 train.py full training loop
    不更新 AO-GPT backbone
    只做 offline frozen-checkpoint diagnostic / MLP training

============================================================
4. 实验目标
============================================================

第一阶段要回答：

    给定 frozen random AO-GPT checkpoint 的 current-frame attention matrix A_current，
    一个 MLP 能否仅凭 implicit attention graph loss 学到非平凡 order？

具体看：

    1. MLP objective 是否下降；
    2. MLP logits 是否非塌缩；
    3. MLP order 是否比 random order 有更低 frozen-checkpoint NLL；
    4. MLP order 的 tau diagnostic 是否出现结构；
    5. direct free-vector optimization 是否能在同一 loss 下找到结构；
    6. 如果 free-vector 可以但 MLP 不行，说明 MLP/数据/优化问题；
       如果 free-vector 也不行，说明 loss 公式本身没有编码有效 order signal。

============================================================
5. 不要做什么
============================================================

不要：

    - 不要先运行 attention-spectral top-k candidate search 生成 teacher order；
    - 不要先算 fixed 45° axis a_signed 再让 MLP 拟合；
    - 不要用 stopgrad(zscore(a_signed)) 当 target；
    - 不要使用 original L2R / original tau / OriginalL2R PPL；
    - 不要用 human scan / raster / Hilbert / Morton；
    - 不要让 gradient 直接穿过 hard argsort；
    - 不要一开始做 REINFORCE / PPO / Plackett-Luce RL；
    - 不要同时训练 backbone。

本实验要做的是：

    A_current -> build graph loss coefficients S,D
    s = MLP(A_current)
    loss(s; A_current) = graph smoothness + directed attention pair preference

============================================================
6. Attention graph loss 设计
============================================================

输入：

    A = A_current ∈ R^{N x N}

来自 frozen AO-GPT 的 attention matrix，current-frame block coordinates。

先构造 robust-normalized matrix：

    Z = robust_z(A)
    Z_ii = 0

注意 robust_z 必须只基于当前 matrix 的 finite off-diagonal entries。
可以复用 online_spectral_order_policy.robust_z，或实现 torch 版本。

------------------------------------------------------------
6.1 Symmetric affinity W / normalized graph S
------------------------------------------------------------

构造无向 affinity：

    Sym_ij = max(Z_ij, Z_ji)

或者做 ablation：

    Sym_ij = 0.5 * (Z_ij + Z_ji)

第一版主用：

    sym_mode = "max"

threshold：

    threshold = percentile(Sym off-diagonal, 60)

affinity：

    W_ij = relu(Sym_ij - threshold)
    W_ii = 0

degree：

    d_i = sum_j W_ij + eps

normalized adjacency：

    S_ij = W_ij / sqrt(d_i * d_j)

或者 normalized Laplacian：

    L_norm = I - S

------------------------------------------------------------
6.2 Directed matrix D
------------------------------------------------------------

构造 directed asymmetry：

    D_ij = 0.5 * (Z_ij - Z_ji)
    D_ii = 0

方向解释：

    D_ij > 0 表示 current attention asymmetry 倾向 i before j。
    这个方向可能有歧义，所以必须同时跑两个 ablation：

        dir_mode = "query_key":
            D = 0.5 * (Z - Z.T)

        dir_mode = "key_query":
            D = -0.5 * (Z - Z.T)

不要用 tau 选择 dir_mode。
两个 mode 都跑，比较 current-frame objective / frozen NLL / heldout loss。
tau 只做 diagnostic。

============================================================
7. MLP 输出和 loss
============================================================

MLP：

    s = FlatAttentionOrderMLP(A) ∈ R^N

语义：

    s_i 越大，block i 越早 reveal。

标准化：

    x = zscore(s)

必须防止 collapse：

    如果 std(s) < 1e-4，加 variance penalty 或跳过该 sample。
    可加入：
        L_var = relu(min_std - std(s))^2
    推荐 min_std = 0.1

------------------------------------------------------------
7.1 Graph smoothness / spectral-axis loss
------------------------------------------------------------

目标：

    让 x 成为 attention graph 上的非平凡 1D axis，
    强 affinity 的 blocks 在 axis 上接近，
    但 x 不能塌成常数。

推荐 loss：

    L_smooth =
        sum_{i,j} W_ij * (x_i - x_j)^2
        / sum_{i,j} W_ij

也可以等价使用 normalized adjacency Rayleigh objective：

    L_spec =
        - x^T S x / (x^T x + eps)

第一版主用：

    L_smooth

因为它更直观、更稳定。

注意：

    只用 L_smooth 不决定方向；
    x 和 -x 的 L_smooth 一样；
    所以必须加入 directed term。

------------------------------------------------------------
7.2 Directed pair loss
------------------------------------------------------------

teacher soft pair target：

    Y_ij = sigmoid(D_ij / tau_d)

confidence weight：

    W_dir_ij = clamp(abs(D_ij) / margin_d, 0, 1)

student pair logit：

    P_logit_ij = (x_i - x_j) / tau_s

loss：

    L_dir =
      weighted BCEWithLogits(P_logit_ij, Y_ij)
      over i < j where W_dir_ij > min_conf

推荐：

    tau_d = 0.5
    tau_s = 1.0
    margin_d = 0.5
    min_conf = 0.2

如果 direction 太 noisy，增加 min_conf 到 0.3 或降低 lambda_dir。

------------------------------------------------------------
7.3 Final loss
------------------------------------------------------------

第一版：

    L = L_smooth + lambda_dir * L_dir + lambda_var * L_var

推荐：

    lambda_dir = 0.3
    lambda_var = 0.01

ablation：

    lambda_dir in {0.1, 0.3, 1.0}
    sym_mode in {"max", "mean"}
    dir_mode in {"query_key", "key_query"}

不要加入 L_PL。
不要加入 top-k candidate loss。
不要加入 teacher-axis regression。

============================================================
8. 必须先做 direct free-vector sanity check
============================================================

在训练 MLP 前，必须先做一个最小 sanity check：

    对单个 A_current，不用 MLP。
    直接令 x ∈ R^N 是可学习参数。
    用同样的 L_smooth + lambda_dir * L_dir 优化 x。
    看优化出来的 order = argsort(x, descending=True) 是否非平凡。

目的：

    检查这个 implicit loss 本身是否能从 attention matrix 中产生结构。
    如果 free-vector 都学不到结构，MLP 更不可能学到。

实现：

    scripts/analysis/attn_axis_free_vector_sanity.py

输入：

    ckpt_path
    num_attention_samples
    layer/head/export_type
    dir_mode
    lambda_dir

输出：

    Report/attn_mlp_implicit_axis/free_vector_sanity/...

记录：

    loss_smooth
    loss_dir
    logit_std
    order_current_first16
    frozen full_loss of order
    random_mean_full_loss
    tau_to_current_l2r diagnostic
    tau_to_original_l2r diagnostic if permute_data=True, diagnostics only
    attention graph stats
    D confidence stats

成功标准不是 tau。
成功标准优先是：

    x 不塌缩；
    objective 明显下降；
    frozen full_loss 不差于 random mean；
    diagnostic tau / adjacency 有结构可报告但不用于选择。

============================================================
9. MLP offline training
============================================================

如果 free-vector sanity 有信号，再训练 MLP。

新增脚本：

    scripts/train/train_attn_mlp_implicit_axis.py

功能：

    - load frozen random AO-GPT checkpoint
    - freeze backbone
    - collect current-frame attention matrices A_current
    - build W/S/D for each A
    - train FlatAttentionOrderMLP with implicit loss
    - save policy.pt
    - save metrics and attention graph diagnostics
    - run heldout evaluation

推荐模型：

    FlatAttentionOrderMLP(
        num_blocks=64,
        hidden_dims=[1024, 1024],
        dropout=0.0,
        activation="gelu",
        input_normalization="robust_zscore"
    )

训练数据：

    train attention samples: 2k
    val attention samples: 512
    smoke: train 200, val 64

attention collection：

    reveal_mode = Random
    layer = 0
    head = 7
    export_type = with_none
    probe_batch_size = 64
    attention samples can be averaged over 1 probe batch each

重要：

    A_current 通过 Random reveal orders 收集。
    不使用 AR probe 作为第一版主数据。
    可以后续 ablate AR probe，但第一版主张必须是 no explicit AR prior。

训练 loss：

    L = L_smooth + lambda_dir * L_dir + lambda_var * L_var

optimizer：

    AdamW
    lr = 1e-4
    weight_decay = 0.01
    grad_clip = 1.0
    epochs = 20 smoke / 50 main
    batch_size = 32 or 64 attention matrices

============================================================
10. Frozen NLL evaluation
============================================================

训练后，在 heldout A 上：

    s = MLP(A)
    order = argsort(s, descending=True)

用 frozen AO-GPT 评估 current-frame teacher-forced loss：

    MLP_MAP_full_loss
    random_mean_full_loss
    random_best_full_loss
    reverse_MLP_order_full_loss
    optional attention_spectral_order_full_loss for diagnostic only if available

不要用 original L2R 做选择。

需要输出：

    MLP_MAP_full_loss - random_mean_full_loss
    MLP_MAP_full_loss - random_best_full_loss
    reverse_order_loss - forward_order_loss
    order stability across A samples
    MLP logit entropy/std
    pairwise direction agreement with D
    smoothness objective on heldout
    tau diagnostics only

脚本：

    scripts/eval/eval_attn_mlp_implicit_axis_order_nll.py

============================================================
11. 推荐命令
============================================================

Free-vector smoke：

    python scripts/analysis/attn_axis_free_vector_sanity.py \
      --ckpt_path out/base/permute/seq256/block64/<RANDOM_RUN>/ckpt.pt \
      --out_dir Report/attn_mlp_implicit_axis/seq256_perm_b64/free_vector_smoke \
      --dataset wikitext103 \
      --split train \
      --num_attention_samples 32 \
      --probe_batch_size 64 \
      --layer 0 \
      --head 7 \
      --export_type with_none \
      --sym_mode max \
      --dir_mode query_key \
      --threshold_percentile 60 \
      --lambda_dir 0.3 \
      --device cuda \
      --dtype bfloat16

MLP smoke：

    python scripts/train/train_attn_mlp_implicit_axis.py \
      --ckpt_path out/base/permute/seq256/block64/<RANDOM_RUN>/ckpt.pt \
      --out_dir checkpoints/attn_mlp_implicit_axis/seq256_perm_b64_l0h7_withnone_smoke \
      --dataset wikitext103 \
      --split train \
      --num_train_attention_samples 200 \
      --num_val_attention_samples 64 \
      --probe_batch_size 64 \
      --layer 0 \
      --head 7 \
      --export_type with_none \
      --sym_mode max \
      --dir_mode query_key \
      --threshold_percentile 60 \
      --lambda_dir 0.3 \
      --lambda_var 0.01 \
      --hidden_dims 1024,1024 \
      --input_normalization robust_zscore \
      --lr 1e-4 \
      --epochs 20 \
      --batch_size 32 \
      --device cuda \
      --dtype bfloat16

MLP main：

    python scripts/train/train_attn_mlp_implicit_axis.py \
      --ckpt_path out/base/permute/seq256/block64/<RANDOM_RUN>/ckpt.pt \
      --out_dir checkpoints/attn_mlp_implicit_axis/seq256_perm_b64_l0h7_withnone_main \
      --dataset wikitext103 \
      --split train \
      --num_train_attention_samples 2000 \
      --num_val_attention_samples 512 \
      --probe_batch_size 64 \
      --layer 0 \
      --head 7 \
      --export_type with_none \
      --sym_mode max \
      --dir_mode query_key \
      --threshold_percentile 60 \
      --lambda_dir 0.3 \
      --lambda_var 0.01 \
      --hidden_dims 1024,1024 \
      --input_normalization robust_zscore \
      --lr 1e-4 \
      --epochs 50 \
      --batch_size 64 \
      --device cuda \
      --dtype bfloat16

Run ablations:

    dir_mode = query_key
    dir_mode = key_query

    lambda_dir = 0.1
    lambda_dir = 0.3
    lambda_dir = 1.0

Do not choose by tau.
Choose/report by heldout implicit objective and frozen NLL vs random.

============================================================
12. 保存格式
============================================================

MLP checkpoint：

    checkpoints/attn_mlp_implicit_axis/.../policy.pt

格式：

    {
      "model_state_dict": model.state_dict(),
      "config": {
        "num_blocks": 64,
        "hidden_dims": [1024, 1024],
        "dropout": 0.0,
        "activation": "gelu",
        "input_normalization": "robust_zscore"
      },
      "policy_type": "implicit_attention_axis_mlp",
      "target_direction": "larger_logit_reveals_earlier",
      "training_meta": {
        "ckpt_path": "...",
        "frame": "current",
        "layer": 0,
        "head": 7,
        "export_type": "with_none",
        "loss": "L_smooth + lambda_dir * L_dir + lambda_var * L_var",
        "sym_mode": "max",
        "dir_mode": "query_key",
        "threshold_percentile": 60,
        "forbidden_signals": [
          "original_l2r",
          "original_tau",
          "OriginalL2R_ppl",
          "human_scan"
        ]
      },
      "metrics": {...}
    }

Reports：

    Report/attn_mlp_implicit_axis/seq256_perm_b64_l0h7_withnone/README.md
    Report/attn_mlp_implicit_axis/seq256_perm_b64_l0h7_withnone/metrics.csv
    Report/attn_mlp_implicit_axis/seq256_perm_b64_l0h7_withnone/train_log.jsonl
    Report/attn_mlp_implicit_axis/seq256_perm_b64_l0h7_withnone/eval_summary.json

============================================================
13. 必须报告的指标
============================================================

训练指标：

    train/loss_total
    train/loss_smooth
    train/loss_dir
    train/loss_var
    train/logit_std
    train/logit_entropy
    train/dir_pair_conf_mean
    train/graph_degree_mean
    train/graph_edge_density

验证指标：

    val/loss_total
    val/loss_smooth
    val/loss_dir
    val/logit_std
    val/order_change_kendall_current
    val/pairwise_direction_agreement
    val/smoothness_score

Frozen NLL eval：

    eval/mlp_map_full_loss
    eval/random_mean_full_loss
    eval/random_best_full_loss
    eval/mlp_minus_random_mean
    eval/reverse_mlp_full_loss
    eval/forward_minus_reverse_loss

Diagnostics only：

    diagnostic/current_l2r_tau
    diagnostic/original_l2r_tau if permute_data=True
    diagnostic/within_unit_adjacent_rate
    diagnostic/order_first16_current
    diagnostic/order_first16_original if available

报告里必须写明：

    diagnostic tau 未用于训练、选择、early stop。
    OriginalL2R 未用于任何训练或选择。

============================================================
14. 判断标准
============================================================

优先判断：

    1. free-vector sanity 是否能优化出非塌缩 axis；
    2. MLP 是否能在 heldout A 上降低 implicit objective；
    3. MLP MAP order frozen full_loss 是否优于 random_mean；
    4. forward order 是否优于 reverse order；
    5. logits 是否稳定，非全平；
    6. tau diagnostic 是否出现结构，但不作为主标准。

可能结果解释：

    A. objective 降，NLL 优于 random，tau 仍约 0：
        说明学到的是 current-attention utility order，不是 L2R。可以继续做 PPL-oriented path。

    B. objective 降，但 NLL 不优于 random：
        implicit attention-axis loss 学到了 graph structure，但不一定是低 PPL order。
        下一步加入 current-frame full_loss direction / rerank。

    C. objective 不降，logits collapse：
        loss 公式或 D/W 构造无效。先看 free-vector sanity。

    D. free-vector 有结构，MLP 没结构：
        MLP 输入归一化、容量、学习率、数据量或 batching 有问题。

    E. query_key 与 key_query 方向相反：
        不用 tau 选方向；比较 frozen NLL / current objective，并报告两个结果。

============================================================
15. 最终回复给用户的格式
============================================================

完成后请用中文总结：

    - 读了哪些 repo 文件；
    - 找到了哪个 frozen random checkpoint；
    - 新增/修改了哪些脚本；
    - free-vector sanity 是否完成；
    - MLP smoke/main 是否完成；
    - 使用的 loss 公式和超参；
    - 训练/验证 attention samples 数量；
    - policy.pt 保存路径；
    - report 保存路径；
    - objective 是否下降；
    - frozen NLL 是否优于 random mean；
    - tau diagnostic 结果，但明确只是 diagnostic；
    - 未完成或失败的部分；
    - 下一步建议。

不要只说“完成了”。必须给路径、命令、样本数、metrics 和 caveats。
