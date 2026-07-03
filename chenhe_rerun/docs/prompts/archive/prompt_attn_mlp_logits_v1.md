> Archive note: this is an older first-version Attn-MLP logits prompt. Current
> work should use the Fiedler-teacher distillation line and frozen try28 MLP
> insertion tests unless the user explicitly asks to revisit this design.

你是一位严谨的 ML research engineer。请基于我的 GitHub 项目
`Yangxiaohehehe/nanogpt-learned-order` 推进当前 WikiText103 language 方向的第一版
Attn-MLP distillation 实验。

请用中文和我沟通；但代码、文件名、命令、日志字段保持英文。不要泛泛讨论理论，
要直接读 repo、实现脚本、生成数据、训练 MLP、保存 artifacts、给出可复现实验记录。
不要编造已完成的结果；凡是没实际运行成功的部分必须明确说明。

============================================================
0. 当前任务一句话
============================================================

当前我要做的是：

    第一版 Frozen Attention-Amortized Order Policy / Attn-MLP distillation。

目标不是让 MLP 学 original L2R，也不是让 MLP 学人类 scan order。
目标是让 MLP 学会近似我已有的 current-frame online spectral/loss-rerank teacher operator。

更具体地说：

    input:
        current-frame block attention matrix A_current ∈ R^{N x N}

    teacher:
        online_spectral_order_policy.py 中的 attention-spectral candidate recovery
        + optional current-frame train/probe loss rerank

    student:
        attn_mlp_order_policy.py 中的 FlatAttentionOrderMLP

    student output:
        logits ∈ R^N

这些 logits 是“每个 current-frame block position 的 reveal priority logit”。
logit 越大，表示这个 block 越应该更早 reveal。

第一版学习目标不是 hard order，而是每个位置的 teacher logits：

    target_logits[i] = teacher 给 current-frame block i 的 priority logit

训练后 hard order 由：

    order = argsort(student_logits, descending=True)

得到。

必须统一这个方向：高 logit = 早 reveal。
因此 frozen MLP config 里应该使用：

    attn_mlp_policy_order_mode = "argsort_desc"

如果现有 config 还是 "argsort"，请新建或修改一个实验 config，避免方向反了。

============================================================
1. 必读文件
============================================================

先读这些项目背景和语言方向文件：

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

重点读 eval / analysis 代码：

    scripts/eval/eval_lm_original_order_ppl.py
    scripts/eval/eval_lm_ppl_sweep.py
    scripts/eval/eval_lm_order_pool_ppl.py
    scripts/analysis/seq80_random_ckpt_online_order_attn_batch.py
    scripts/analysis/seq80_loss_rerank_existing_attention_candidates.py
    scripts/analysis/head_angle_tau_diagnostic.py

重点读 configs：

    config/WikiText103/seq256/non_permute/block64/online_spectral_fixed_head_order.py
    config/WikiText103/seq256/non_permute/block64/online_spectral_fixed_head_order_loss_rerank.py
    config/WikiText103/seq256/non_permute/block64/online_spectral_order_distribution.py
    config/WikiText103/seq256/non_permute/block64/online_spectral_order_distribution_loss_rerank.py
    config/WikiText103/seq256/permute/block64/online_spectral_fixed_head_order.py
    config/WikiText103/seq256/permute/block64/online_spectral_order_distribution.py
    config/WikiText103/seq256/permute/block64/attn_mlp_frozen_order.py

第一版主实验优先使用：

    WikiText103 / seq256 / permute / block64
    block_size = 256
    block_order_block_len = 4
    num_blocks = 64
    layer = 0
    head = 7
    export_type = "with_none"

seq80/block1 暂时作为后续 stress test，不作为第一版唯一入口。

============================================================
2. 必须遵守的 frame / oracle 约束
============================================================

所有训练、teacher target、candidate selection、rerank、MLP 输入、MLP label 都只能使用
current-frame model-side signals。

允许使用：

    current-frame attention matrix
    current-frame spectral graph recovery
    current-frame candidate attention path score
    current-frame directed attention score
    current-frame band quality
    current-frame train/probe teacher-forced prefix loss
    current-frame train/probe teacher-forced full loss

禁止用于训练、target、selection、rerank、early stop、MLP checkpoint selection：

    original L2R order
    original-frame Kendall tau
    original-frame oracle diagnostics
    OriginalL2R PPL
    human scan / raster / Morton / Hilbert / column-major
    validation PPL for choosing MLP or order

如果 permute_data=True：

    cached_order / teacher order / target_logits / student logits 都必须是 current-frame block ids
    block_perm 和 *_original 只能用于 logging / diagnostics
    OriginalL2R 是 oracle upper bound，不是 no-prior learned-order result

如果 permute_data=False：

    current frame 等于自然文本 frame
    但仍然要把方法解释为 current-model attention/loss signal，不是用 tau 或 oracle 选择 order

============================================================
3. 第一版实验目标
============================================================

请实现并运行第一版实验：

    A. 导出 Attn-MLP teacher dataset
    B. 保存训练集 / 验证集 / 测试集 shards
    C. 训练一个输出 per-block logits 的 MLP，并利用wandb记录
    D. 保存 best policy.pt
    E. 做 offline imitation eval
    F. 准备 frozen MLP online run config

第一版不要直接做 joint policy learning，不要 unfreeze MLP 接进 AO-GPT optimizer。
第一版只做 supervised distillation + frozen MLP policy。

============================================================
4. Teacher dataset 定义
============================================================

每条样本是一张 current-frame block attention matrix 及其 teacher logits：

    A_current:      float16 or float32 tensor [N, N]
    target_logits:  float32 tensor [N]
    target_priority: optional float32 tensor [N]
    teacher_top_order: int64 tensor [N]
    teacher_candidate_orders: optional int64 tensor [M, N]
    teacher_candidate_scores: optional float32 tensor [M]
    metadata: source_ckpt, iter, split, probe_mode, layer, head, export_type, etc.

其中：

    N = num_blocks
    block64 时 N = 64

核心语义：

    target_logits[i] 越大，current-frame block i 越应该更早 reveal。

不要保存成“block id 的 label”。要保存每个 current-frame 位置/块的 scalar logit target。

============================================================
5. target_logits 如何从 teacher candidates 构造
============================================================

给定 attention matrix A_current，先调用 teacher candidate recovery：

    recover_fixed_head_spectral_candidates(
        A_current,
        config=FixedHeadSpectralPolicyConfig(...),
        top_m=top_m_recovery
    )

建议第一版参数：

    top_m_recovery = 64
    top_m_target = 32
    teacher_temperature = 1.0
    score_normalization = "zscore"

对每个 candidate order m：

    order_m = [block_id_0, block_id_1, ..., block_id_{N-1}]
    pos_m(i) = block i 在 order_m 里的 reveal 位置
    v_m(i) = 1 - pos_m(i) / (N - 1)

对 candidate scores：

    scores = candidate final scores
    scores_z = zscore(scores)
    w_m = softmax(scores_z / teacher_temperature)

soft priority:

    target_priority(i) = sum_m w_m * v_m(i)

最终用于训练的 teacher logits：

    target_logits = zscore(target_priority)

也可以保存：

    target_priority
    target_logits
    teacher_top_order = candidates[0]["order"]

但训练主 target 用 target_logits。

重要：target_logits 不是 calibrated probability；它是每个 block 的相对 reveal-priority logit。
训练和 eval 都按“数值越大越早 reveal”解释。

============================================================
6. attention-only 数据和 loss-rerank 数据
============================================================

为了让数据集尽量大，请做两层数据：

    1. large attention-only dataset
    2. smaller loss-rerank dataset

attention-only target：

    A_current
    -> recover top-M candidates
    -> 用 attention-side candidate score 构造 target_logits

loss-rerank target：

    A_current
    -> recover top-M candidates
    -> 对 candidates 用 current-frame train/probe teacher-forced loss rerank
    -> 用 reranked final_score 构造 target_logits

loss-rerank final score 先沿用当前项目设置：

    final_score =
        1.0  * z(attention_spectral_score)
      - 0.25 * z(prefix_loss)
      - 0.15 * z(full_loss)

block64 默认：

    prefix_k = 8
    loss_rerank_top_k = 64
    loss_rerank_batches = 16
    loss_rerank_batch_size = 64

如果算力不够，先做：

    loss_rerank_batches = 4
    loss_rerank_batch_size = 64

但 metadata 里必须写清楚。

============================================================
7. 训练集 / 验证集 / 测试集设计
============================================================

不要只做 random row split，因为同一个 checkpoint / probe seed 的样本高度相关。
请做 group split。

每个样本 group key 至少包括：

    source_run
    source_ckpt_path
    source_iter
    probe_mode
    probe_seed_bucket
    ema_decay

建议 split：

    train:
        70% groups

    val_id:
        10% groups
        与 train 同 run / 同 checkpoint 分布，但 probe seed 或 sample range 不同

    val_time:
        10% groups
        held-out checkpoint iterations，用来检查 temporal generalization

    test_run:
        10% groups
        held-out run 或 held-out checkpoint source，用来检查跨 run generalization

如果第一版资源有限，至少要有：

    train
    val_id
    val_time

不要用 WikiText103 val PPL 选择 MLP checkpoint。MLP 选择只能用 offline imitation metrics。

============================================================
8. 数据规模
============================================================

先做 smoke set，确保 shape、方向、保存、训练都没 bug：

    attention-only smoke:
        1k samples

    loss-rerank smoke:
        100 samples

通过后做 main set：

    attention-only main:
        50k - 200k samples

    loss-rerank main:
        10k - 30k samples

如果算力和存储允许，再做 large set：

    attention-only:
        500k - 1M samples

    loss-rerank:
        50k - 100k samples

A_current 建议保存 float16，训练时转 float32。
block64 下 [64,64] fp16 约 8 KB，所以 100k samples 约 0.8 GB，1M samples 约 8 GB。

============================================================
9. Probe mode 设计
============================================================

采集 A_current 时必须给模型一个 hard reveal order。第一版不要只用单一 probe mode。

建议混合：

    Random probe:              70%
    TeacherTopOrder probe:     20%
    DistributionSampled probe: 10%

如果实现复杂，smoke set 可以先只用 Random probe。
main set 应该尽量加入 TeacherTopOrder / DistributionSampled，以模拟 MLP 接回训练后的 attention distribution。

不要把 AR probe 放进第一版主训练集。AR probe 可以作为 diagnostic，但不要作为 teacher dataset 的主要来源。

============================================================
10. EMA input
============================================================

train.py 的 frozen MLP path 使用 attention EMA：

    A_ema = decay * A_ema_prev + (1 - decay) * A_raw

因此 dataset 中建议保存多种输入：

    A_raw, ema_decay = 0.0
    A_ema, ema_decay = 0.90
    A_ema, ema_decay = 0.95

注意：每个 A_input 的 teacher target 应该由同一个 A_input 重新计算。
不要把 A_raw 的 target 直接贴给 A_ema。

============================================================
11. 需要实现的脚本 1：导出 teacher dataset
============================================================

新增脚本：

    scripts/analysis/export_attn_mlp_teacher_logits_dataset.py

脚本功能：

    - 读取一个或多个 checkpoints
    - 采样 train split probe batches
    - 生成 explicit block_orders
    - forward AO-GPT with return_attentions=True
    - 聚合到 current-frame block attention matrix A_current
    - 维护 A_raw / A_ema
    - 调用 online_spectral_order_policy.py 的 candidate recovery
    - 可选 current-frame loss rerank
    - 构造 target_priority 和 target_logits
    - 按 shard 保存 train / val_id / val_time / test_run
    - 写 manifest.json、metadata.jsonl、summary.csv、README.md

命令示例：

    python scripts/analysis/export_attn_mlp_teacher_logits_dataset.py \
      --ckpt_glob 'out/base/permute/seq256/block64/*/ckpt.pt' \
      --out_dir data/attn_mlp_teacher/wikitext103_seq256_perm_b64_l0h7_withnone_logits_v1_smoke \
      --dataset wikitext103 \
      --split train \
      --layer 0 \
      --head 7 \
      --export_type with_none \
      --num_examples_per_ckpt 500 \
      --probe_batch_size 64 \
      --probe_batches_per_example 1 \
      --ema_decays 0.0,0.95 \
      --probe_modes Random \
      --top_m_recovery 64 \
      --top_m_target 32 \
      --teacher_temperature 1.0 \
      --loss_rerank_fraction 0.1 \
      --loss_rerank_top_k 64 \
      --loss_rerank_batches 4 \
      --loss_rerank_batch_size 64 \
      --loss_rerank_prefix_k 8 \
      --dtype float16 \
      --seed 12345

main set 示例：

    python scripts/analysis/export_attn_mlp_teacher_logits_dataset.py \
      --ckpt_glob 'out/base/permute/seq256/block64/*/ckpt.pt' \
      --out_dir data/attn_mlp_teacher/wikitext103_seq256_perm_b64_l0h7_withnone_logits_v1_main \
      --dataset wikitext103 \
      --split train \
      --layer 0 \
      --head 7 \
      --export_type with_none \
      --num_examples_per_ckpt 5000 \
      --probe_batch_size 64 \
      --probe_batches_per_example 1 \
      --ema_decays 0.0,0.90,0.95 \
      --probe_modes Random,TeacherTopOrder,DistributionSampled \
      --probe_mode_weights 0.7,0.2,0.1 \
      --top_m_recovery 64 \
      --top_m_target 32 \
      --teacher_temperature 1.0 \
      --loss_rerank_fraction 0.1 \
      --loss_rerank_top_k 64 \
      --loss_rerank_batches 8 \
      --loss_rerank_batch_size 64 \
      --loss_rerank_prefix_k 8 \
      --dtype float16 \
      --seed 12345

如果没有足够 checkpoints，不要失败；请允许显式传入 --ckpt_paths 或 --ckpt_file。
如果没有 online spectral checkpoint，也可以先只用 random checkpoints。

============================================================
12. Dataset shard 格式
============================================================

目录结构：

    data/attn_mlp_teacher/wikitext103_seq256_perm_b64_l0h7_withnone_logits_v1_main/
      manifest.json
      README.md
      summary.csv
      train/
        shard_00000.pt
        shard_00001.pt
      val_id/
        shard_00000.pt
      val_time/
        shard_00000.pt
      test_run/
        shard_00000.pt

每个 shard 保存：

    {
      "A": FloatTensor[num_samples, N, N],
      "target_logits": FloatTensor[num_samples, N],
      "target_priority": FloatTensor[num_samples, N],
      "teacher_top_order": LongTensor[num_samples, N],
      "teacher_top_score": FloatTensor[num_samples],
      "teacher_entropy": FloatTensor[num_samples],
      "teacher_score_gap": FloatTensor[num_samples],
      "sample_weight": FloatTensor[num_samples],
      "metadata": list[dict]
    }

metadata 里可以包含 block_perm，但明确标注：

    "block_perm is diagnostics only; not used as input or target."

manifest.json 必须记录：

    num_blocks
    block_order_block_len
    layer
    head
    export_type
    target_type = "per_block_teacher_logits"
    target_direction = "larger_logit_reveals_earlier"
    frame = "current"
    top_m_recovery
    top_m_target
    teacher_temperature
    loss_rerank settings
    split policy
    source checkpoints
    command used

============================================================
13. 需要实现的脚本 2：训练 MLP logits policy
============================================================

新增脚本：

    scripts/train/train_attn_mlp_order_logits.py

使用模型：

    attn_mlp_order_policy.FlatAttentionOrderMLP

第一版模型：

    num_blocks = 64
    hidden_dims = [1024, 1024]
    activation = "gelu"
    dropout = 0.0
    input_normalization = "robust_zscore"

输入：

    A_current [B, N, N]

输出：

    student_logits [B, N]

target：

    target_logits [B, N]

训练 loss：

    student_z = zscore(student_logits, dim=-1)
    target_z = zscore(target_logits, dim=-1)

    loss_mse = MSE(student_z, target_z)

pairwise rank loss：

    对所有 i,j:
        如果 target_logits[i] > target_logits[j] + target_margin
        则希望 student_logits[i] > student_logits[j]

    loss_rank =
        mean softplus(-(student_logits[i] - student_logits[j]) / rank_temperature)

optional KL：

    p_teacher = softmax(target_logits / teacher_kl_temperature)
    p_student = log_softmax(student_logits / student_kl_temperature)
    loss_kl = KL(p_teacher || p_student)

总 loss：

    loss =
        1.0 * loss_mse
      + lambda_rank * loss_rank
      + lambda_kl * loss_kl

默认：

    lambda_rank = 0.2
    lambda_kl = 0.0
    rank_temperature = 0.25
    target_margin = 0.02

sample_weight：

    如果 shard 里有 sample_weight，用它加权 loss。
    sample_weight 只能来自 teacher entropy / score_gap 等 current-frame teacher confidence，
    不能来自 original tau。

训练命令 smoke：

    python scripts/train/train_attn_mlp_order_logits.py \
      --dataset_dir data/attn_mlp_teacher/wikitext103_seq256_perm_b64_l0h7_withnone_logits_v1_smoke \
      --out_dir checkpoints/attn_mlp/wikitext103_seq256_perm_b64_l0h7_withnone_logits_v1_smoke \
      --num_blocks 64 \
      --hidden_dims 1024,1024 \
      --input_normalization robust_zscore \
      --activation gelu \
      --dropout 0.0 \
      --batch_size 512 \
      --epochs 20 \
      --lr 1e-3 \
      --weight_decay 0.01 \
      --warmup_steps 200 \
      --lambda_rank 0.2 \
      --lambda_kl 0.0 \
      --rank_temperature 0.25 \
      --target_margin 0.02 \
      --eval_every 200 \
      --save_best_by val_pairwise_acc

训练命令 main：

    python scripts/train/train_attn_mlp_order_logits.py \
      --dataset_dir data/attn_mlp_teacher/wikitext103_seq256_perm_b64_l0h7_withnone_logits_v1_main \
      --out_dir checkpoints/attn_mlp/wikitext103_seq256_perm_b64_l0h7_withnone_logits_v1_main \
      --num_blocks 64 \
      --hidden_dims 1024,1024 \
      --input_normalization robust_zscore \
      --activation gelu \
      --dropout 0.0 \
      --batch_size 1024 \
      --epochs 50 \
      --lr 1e-3 \
      --weight_decay 0.01 \
      --warmup_steps 1000 \
      --lambda_rank 0.2 \
      --lambda_kl 0.0 \
      --rank_temperature 0.25 \
      --target_margin 0.02 \
      --eval_every 1000 \
      --save_best_by val_pairwise_acc

checkpoint 必须兼容 train.py 的 load_frozen_attn_mlp_policy：

    {
      "model_state_dict": model.state_dict(),
      "config": {
        "num_blocks": 64,
        "hidden_dims": [1024, 1024],
        "dropout": 0.0,
        "activation": "gelu",
        "input_normalization": "robust_zscore"
      },
      "teacher_meta": {...},
      "metrics": {...}
    }

保存路径：

    checkpoints/attn_mlp/wikitext103_seq256_perm_b64_l0h7_withnone_logits_v1_main/policy.pt
    checkpoints/attn_mlp/wikitext103_seq256_perm_b64_l0h7_withnone_logits_v1_main/best_policy.pt
    checkpoints/attn_mlp/wikitext103_seq256_perm_b64_l0h7_withnone_logits_v1_main/train_log.jsonl
    checkpoints/attn_mlp/wikitext103_seq256_perm_b64_l0h7_withnone_logits_v1_main/metrics.csv
    checkpoints/attn_mlp/wikitext103_seq256_perm_b64_l0h7_withnone_logits_v1_main/README.md

============================================================
14. Offline imitation eval
============================================================

训练脚本或单独 eval 脚本必须输出这些 metrics：

    target_logits_mse
    target_logits_z_mse
    pearson_corr
    spearman_corr
    pairwise_preference_accuracy
    pairwise_preference_accuracy_confident_pairs

order metrics:

    mlp_order = argsort(student_logits, descending=True)
    teacher_order = teacher_top_order

    kendall_tau_mlp_vs_teacher_current
    top4_prefix_overlap
    top8_prefix_overlap
    top16_prefix_overlap
    adjacent agreement diagnostics if useful

logit diagnostics:

    student_logit_mean
    student_logit_std
    student_logit_entropy = entropy(softmax(student_logits))
    student_top1_gap
    target_logit_entropy
    teacher_entropy

breakdown by:

    split
    source_checkpoint
    source_iter
    probe_mode
    ema_decay
    loss_rerank_used

不要用 original-frame Kendall tau 选择 best checkpoint。
可以在 README 里报告 original diagnostics，但必须标注 diagnostics-only。

============================================================
15. Frozen MLP online run config
============================================================

训练好 MLP 后，准备 frozen run config。可以复制：

    config/WikiText103/seq256/permute/block64/attn_mlp_frozen_order.py

新建：

    config/WikiText103/seq256/permute/block64/attn_mlp_frozen_order_logits_v1.py

必须设置：

    aogpt_train_mode = "AttnMLPFrozenOrder"
    attn_mlp_policy_enabled = True
    attn_mlp_policy_path = "checkpoints/attn_mlp/wikitext103_seq256_perm_b64_l0h7_withnone_logits_v1_main/best_policy.pt"
    attn_mlp_policy_freeze = True
    attn_mlp_policy_start_iter = 8000
    attn_mlp_policy_layer = 0
    attn_mlp_policy_head = 7
    attn_mlp_policy_export_type = "with_none"
    attn_mlp_policy_input_normalization = "robust_zscore"
    attn_mlp_policy_ema_decay = 0.95
    attn_mlp_policy_update_every = 1
    attn_mlp_policy_order_mode = "argsort_desc"
    attn_mlp_policy_fallback = "random"
    attn_mlp_policy_collect_warmup_attention = True

注意：

    train.py 的 AttnMLPFrozenOrder 是 one-pass lag：
        order_t -> loss_t + attention_t -> MLP(attention_t) -> order_{t+1}

不要描述成 same-step attention 选择 same-step order。

如果 train.py 当前没有 "argsort_desc" config 流程可用，检查 attn_mlp_order_policy.logits_to_order。
它已经应该支持 argsort_desc / argsort_descending；如果没有，最小改动添加。

============================================================
16. 最终 PPL eval
============================================================

不要直接跨 run 比默认 val/loss。使用 eval_lm_ppl_sweep.py。

对 frozen MLP checkpoint 至少评估：

    Random
    AR
    CheckpointAttnMLPOrder
    OriginalL2R

如果有 online spectral 对照 checkpoint，也评估：

    CheckpointOnlineSpectralOrder
    CheckpointOnlineSpectralDistributionMAP
    CheckpointOnlineSpectralDistributionSampled

命令模板：

    python scripts/eval/eval_lm_ppl_sweep.py \
      --ckpt_paths out/base/permute/seq256/block64/out-wikitext103-seq256-attn-mlp-frozen-order-logits-v1-b64-permute-block-50000-iters/ckpt.pt \
      --eval_modes Random,AR,CheckpointAttnMLPOrder,OriginalL2R \
      --out_dir Report/eval/wikitext103_seq256_perm_block64_attn_mlp_logits_v1_ppl \
      --split val \
      --batch_size 64 \
      --num_batches 0 \
      --sample_mode sequential \
      --sequential_cover_tail \
      --device cuda \
      --dtype bfloat16

OriginalL2R 只能作为 oracle diagnostic。不要把它写成 no-prior learned-order 结果。

============================================================
17. 实验报告
============================================================

请为本实验写报告：

    Report/attn_mlp_distill/wikitext103_seq256_perm_b64_l0h7_withnone_logits_v1/README.md

报告必须包含：

    1. 目标：学习 per-current-frame-block priority logits
    2. teacher 定义
    3. 数据来源
    4. split 方法
    5. target_logits 构造公式
    6. 模型结构
    7. 训练命令
    8. offline imitation metrics
    9. frozen run config path
    10. PPL eval 命令和结果，如果已跑
    11. caveats

caveats 必须写：

    - MLP 蒸馏的是 current-frame teacher operator，不是 original L2R
    - target_logits 高表示更早 reveal
    - OriginalL2R 只作为 oracle diagnostic
    - offline imitation 不等于 PPL 提升
    - frozen MLP 是 one-step lagged policy

============================================================
18. 完成任务时给我的最终回复
============================================================

完成后请用中文总结：

    - 修改/新增了哪些文件
    - 数据保存在哪里
    - 训练集/验证集/测试集各多少样本
    - teacher 是否启用了 loss rerank
    - MLP checkpoint 保存在哪里
    - offline imitation metrics
    - 是否创建了 frozen run config
    - 是否启动/完成 frozen AO-GPT run
    - 是否完成 PPL sweep
    - 失败或未完成的部分，以及原因
    - 下一步建议

不要只说“完成了”。必须给出路径、命令、样本数、metrics 和 caveats。
