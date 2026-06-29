# L0H5 Order 信号随训练进程的演化 — 设计

**日期**: 2026-05-29
**线**: BR-1 派生诊断(承接多 seed 单 head 扫描的 L0H5 winner 发现)
**相关记忆**: [[br1_batch_readout_status]]

## 背景与问题

之前在 `alt_from0_random/ckpt_step5000.pt`(text, 4 层 ×8 head, n_embd=384, block_size=256, block_len=4, N=64 blocks)上发现:order 信号高度 head-specific,集中在 **Layer 0 Head 5**,跨 3 个**抽样 seed**(0/1/2)稳定(τ_vs_l2r = +0.562/+0.583/+0.638)。注意那里的"seed"控制的是**数据抽样 + randperm reveal 顺序**,不是训练过程。

本设计回答一个**不同的轴**:在与 L0H5 发现完全相同的 regime(纯随机序 baseline)下,**order 信号随训练步如何演化**?具体两个问题:

- **Q1(强度演化)**: L0H5 的 τ_vs_l2r 随训练步怎么变?是从随机初始化就存在,还是训练中长出来(涌现)?后期是否衰减/饱和?
- **Q2(定位稳定性)**: order 信号的 head 定位是否训练不变?winner head 始终是 L0H5,还是会漂移到别的 (layer, head)?

## 关键约束 / 已知事实

- 目前 `probe_results/attention_order_mlp/` 下**所有 checkpoint 都是同一架构**(4L8H384),L0H5 这个 head index 是相对固定 (L=4, H=8) 定义的。
- L0H5 发现所在的 `alt_from0_random` 是 `run_kind=baseline`、`mlp_alternating=False` 的**纯随机序训练**(order 全程随机,无 alpha 课程、无 MLP 诱导),是最干净的 regime。但它**只存了 step5000 一个 ckpt**。
- 要拿到训练轨迹只能重训(几小时 GPU),这是唯一不可省的真实成本。
- **扫描成本与 head 数量无关**:开销全在 forward pass,提 1 个 head 与 32 个 head 成本相同。现有诊断脚本 `diag_br1_head_layer_scan.py` 用 batch=1 串行跑 3200 次 forward 才慢;批量化后可降到分钟级。

## 组件

### ① 密集存点重训 random baseline

复刻 `alt_from0_random` 的训练配置,**仅改 save_steps**:

- `run_kind=baseline`、纯随机序、`mlp_alternating=False`
- `seed=42`、`permute_seed=42`、`lr=1e-3`、`min_lr=1e-4`、`batch_size=64`、`grad_accum=2`
- `n_layer=4`、`n_head=8`、`n_embd=384`、`block_size=256`、`block_order_block_len=4`、`vocab_size=50304`
- `max_steps=30000`(复刻原 baseline,保证 step5000 复现门成立)
- `save_steps = {init, 1000, 2000, 5000, 10000, 15000, 30000}`
  - "init" = 训练前的近初始化 checkpoint(step0 若训练器支持保存,否则最早可保存步,如 step10)
- 输出目录:`probe_results/attention_order_mlp/random_baseline_dense/`

**复现门(hard gate)**: 新 run 的 step5000 在 seed 0/1/2 上的 L0H5 τ_vs_l2r 必须落在已知值范围 **+0.56 ~ +0.64 的 ±0.05** 内。若不复现,轨迹不可信,先排查训练配置差异,不进入聚合判读。

### ② 批量化 head-layer 扫描内核

新建 `scripts/diag_l0h5_evolution_scan.py`(或将 `diag_br1_head_layer_scan.py` 重构出可复用的 `scan()`),**核心改造 = 批量化 forward**:

- 把 `idx_chunks[i:i+1]`(batch=1)改为按 `batch_size`(32 或 64)堆叠:一次 forward 吃 `(B, t)` 的 tokens + `(B, t)` 的**每样本独立 random reveal orders**(`forward_fn(idx, orders, return_attentions=True)` 已支持任意 batch,`attn_outputs` 每层 `(B, H, t+1, t+1)`)。
- 仍保留**全 32 head 扫描**(为检测 Q2 的 winner 漂移;额外 head 是 forward 后的 numpy 切片,零额外 forward 成本)。
- 保留与原扫描一致的 pipeline:single-head attention → reveal→physical remap → 256→64 block-agg → `B = A^T` 去对角 → batch-mean over B → teacher CDL(`generate_teacher_label`, `alpha_dep=0.5`)→ σ。
- 保留 heavy baseline 行(top-4 var head + all-layer mean + 0.1·[None])作为对照锚点。

参数:`M=100`、`batch_size=32`(=3200 graphs/scan,与原跨-seed 数值可直接对比),对每个 (step, seed∈{0,1,2}) 跑一遍,dump 一个 JSON 到 `block_lo_arm_order_network/batch_readout/logs/l0h5_evo/scan_step{S}_seed{D}.json`。

**预算**: 批量化后单 scan ~10–30s × (7 step × 3 seed = 21) ≈ **~10 分钟以内**。

### ③ 聚合 + 出图

新建 `scripts/aggregate_l0h5_evolution.py`,吃所有 scan JSON,产出 `report.json` + matplotlib 图:

- **L0H5 强度曲线**: τ_vs_l2r(L0,H5) 的 3-seed mean±std vs step(回答 Q1)
- **winner-head-per-step 表**: 每步 |τ_vs_l2r| 最大的 (layer,head) 是谁(回答 Q2 核心)
- **L0H5 rank-per-step**: L0H5 在 |τ_vs_l2r| 排序里的名次 vs step
- **layer-0 集中度 vs step**: max|τ| over heads, 按 layer 分,看 order 信号是否始终在 Layer 0
- **热图**: step × (L,H) 的 τ_vs_l2r
- **符号检查**: L0H5 应为 +(L2R-leaning)、L0H1 应为 −(anti-L2R),逐步标注
- **多样性健康度 vs step**: first_step_entropy 和 mean_pairwise_τ(确认非塌缩)

### ④ 判读标准(verdict)

- **稳定**: 在 ≥某训练步后,L0H5 始终是 top-1/top-2 |τ_vs_l2r| head、符号恒 +、|τ| 不塌 → 结论"L0H5 是该 regime 下训练不变的早层 order head"。
- **涌现**: 近 init 时 |τ| 低/噪声,随训练单调升高 → 报涌现曲线 + 起飞步。
- **漂移**: winner head index 跨步变化 → L0H5 **不是**训练不变量,报它漂到哪个 (layer,head),以及 L0H5 自身曲线如何。

任一结论都是可写进 paper 的 mechanistic 观察(order 信息在 head 上的非均匀分布如何随训练形成/迁移)。

## 不做(YAGNI)

- **不改模型规模/架构/训练 seed**(那是另外的轴,本设计明确选"训练进程")。
- **不引入 NLL 到任何 selection / 判读**(沿用 BR-1/NR-1 红线)。
- **不重定义 §5.1 阈值**;本诊断的核心 metric 是 τ_vs_l2r 与 winner-head 定位,不是 §5.1 gate。
- 不复用 full `extract_A_matrices` 的 top-4 var head 选择作为主信号(只作 heavy 对照行)。

## 复用 vs 新代码

- **复用**: 扫描 pipeline(remap / block-agg / B=A^T)、`generate_teacher_label`(CDL)、`teacher_diversity_stats`、`kendall_tau_batch`、`_load_model_and_chunks`、训练脚本 `train_clean_aogpt` / `train_vq64_alternating`(取 baseline 路径)。
- **新代码**: (a) 密集存点的训练启动配置/脚本;(b) 批量化扫描 driver;(c) 聚合出图脚本。

## 风险

- **复现门不过**: 重训轨迹若 step5000 对不上已知 L0H5,需先排查(随机性来源、数据路径、extract randperm 未播种问题——见 [[cdl_evolution_clean_base_20260527]],`extract_A_matrices` 跨 run randperm 方差)。本设计在扫描内 reveal-order 用 `seed+i` 显式播种,规避该坑。
- **GPU 占用**: 重训需一张空卡;若卡忙,训练阶段排队等待(扫描+聚合阶段成本可忽略)。
