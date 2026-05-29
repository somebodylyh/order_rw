# L0H5 Order 信号跨 checkpoint / 跨采样-seed 稳定性验证

日期: 2026-05-29
作者: lyuyuhuan + Claude
关联记忆: `br1_batch_readout_status.md`, `cdl_evolution_clean_base_20260527.md`, `text_training_config.md`

## 1. 背景与动机

BR-1 轻量 hook 设计阶段,在一个 **text alternating-from-0** run(`alt_from0_random/ckpt_step5000.pt`,wikitext seq256/block64,L=4 H=8,47M)上做过多采样-seed 的单 head 扫描,结论是 order 信号高度 head-specific:`L0H5` 的 `τ_vs_l2r ≈ 0.56–0.64`(接近 heavy top-4 的 0.67),在 3 个采样 seed 上稳定(0.562/0.583/0.577);`L0H1` 是 anti-L2R(τ≈−0.38),head-mean 把两者抵消。

该结论目前只建立在:
- **单个 checkpoint**(step5000),且该 ckpt **已从仓库丢失**,扫描脚本也未提交(无 git 记录)。
- **采样-seed 维度**(同一 ckpt 的随机 order 采样种子),**不是训练-seed**。

本验证回答:**order 信号的 head-specific 结构在训练过程中是否稳定出现、何时出现、单个主序 head 的 τ_vs_l2r 是否对采样噪声稳健。**

## 2. 范围与判据

### 2.1 模型选择(已定)
用 `probe_results/clean_base_random_perm/` 的完整 ladder:`ckpt_step{0,1000,5000,10000,20000,30000,40000,50000,60000}.pt`(9 个)。

这是一个**独立的** text random-perm baseline run,**不是**原始 alt_from0 run。⚠️ 因此:
- 它的 L0 主序 head **不保证是字面的 head 5**。不同 run 间 head index 因 attention head permutation 对称性**不可直接比较**。
- 本验证**数据驱动地**识别"L0 内 τ_vs_l2r 最强的正向 head"和"最强的负向(anti-L2R)head",而**不预设 H5/H1**。
- "是否恰好落在 H5/H1" 作为**观察结果报告**,不作为成败判据。

### 2.2 采样-seed(已定)
每个 checkpoint 用 **5 个采样 seed**(0–4),每个 seed `M=100` 个 batch、`batch_size=32`,与原扫描口径一致。

### 2.3 成功判据(verdict gates)
本轮是诊断,不设 hard pass/fail 训练 gate。报告以下结论:
- **G1 head-concentration**:在中后期 ckpt(≥20k),L0 存在 `|τ_vs_l2r|` 显著高于 head-mean 的单 head(主序 head),且其 `τ_vs_l2r` 跨 5 seed 的 std 小(目标 std < 0.05,沿用原 0.562/0.583/0.577 的离散度量级)。
- **G2 head-index 跨 step 稳定**:同一 run 内(权重连续演化),主序 head 的 **index 在相邻 step 间稳定**(允许早期 <5k 抖动)。
- **G3 演化合理性**:step0(随机初始化)的主序 head τ_vs_l2r 应接近 0 / 噪声水平(null 对照);随训练上升。
- **G4 抵消结构**:检验是否复现"主序 head 正 τ + 另一 head anti-L2R、head-mean 抵消"的现象(报告 head-mean τ_vs_l2r vs 单 head)。

### 2.4 红线(继承 BR-1 / NR-1 spec)
- 只用 attention-derived 信息;**严禁 NLL / L2R-raster oracle 进入 readout 或 selection**。
- readout 固定 CDL-source-start(`alpha_dep=0.5`),与既有 teacher 口径一致。
- `extract_A_matrices` 的 `torch.randperm` **必须播种**(per-seed `torch.Generator`),否则跨 run B 方差污染(见 `cdl_evolution_clean_base_20260527.md`)。

## 3. 复用的现成积木

| 用途 | 文件 | 函数 |
|---|---|---|
| 加载 text AOGPT ckpt | `train_clean_aogpt.py` / `layer_head_locality_scan.load_model` | — |
| 前向 + per-layer-head attention(physical order remap) | `layer_head_locality_scan.py` | `extract_per_layer_head` |
| token→block 聚合(BLOCK_LEN=4, NUM_BLOCKS=64) | `layer_head_locality_scan.py` | `aggregate_to_blocks` |
| 单 head B→order readout (CDL-source-start) | `neural_readout/teacher_labels.py` | `generate_teacher_label(B, alpha_dep=0.5)` |
| diversity 统计 | `batch_readout/diversity_batch.py` | `teacher_diversity_stats(sigma)` |
| Kendall tau | `block_lo_arm_order_network/utils.py` / scipy | `kendall_tau` / `kendalltau` |

数据:wikitext-103 chunked(`train_clean_aogpt.py` 的加载路径,见 `probe_data_loading.md`)。

## 4. 新增组件

新增一个独立脚本 `block_lo_arm_order_network/per_head_order_scan.py`(**重写丢失的脚本**),职责单一:对一个 ckpt 跑 per-head order 扫描,输出与原 JSON schema 对齐的结果。

### 4.1 核心函数(可单测)
```
per_head_order_scan(model, data_chunks, M, batch_size, seed, alpha_dep=0.5)
    -> {
        "config": {M, batch_size, seed, L, H, alpha_dep},
        "heavy_baseline": {tau_vs_l2r, diversity{...}},
        "per_head_layer_sorted_by_abs_tau_vs_l2r": [
            {layer, head, tau_vs_l2r, mean_pairwise_tau, first_step_entropy, tau_vs_heavy}, ...
        ],
    }
```
内部流程(单 seed):
1. 播种 `torch.Generator(seed)`。
2. 循环 M 个 batch,每 batch 取 batch_size 个序列,各自 `randperm`(seeded)→ `model.forward_fn(..., return_attentions=True)` → per-(layer,head) token attention,remap 回 physical order,聚合到 64×64,batch-mean。
3. **heavy**:按 off-diagonal variance 选 top-4 head,平均 → A_heavy → B=A^T(diag 置 0)→ `generate_teacher_label` → `sigma_heavy[m]`。
4. **per head (l,h)**:B=A_lh^T(diag 置 0)→ `generate_teacher_label` → `sigma_lh[m]`。
5. M 个 batch 后:对 heavy 与每个 head,
   - `tau_vs_l2r` = mean_m kendalltau(sigma[m], arange(64))
   - `diversity` = `teacher_diversity_stats(sigma_MxN)`
   - `tau_vs_heavy`(仅 per-head)= mean_m kendalltau(sigma_lh[m], sigma_heavy[m])
6. 按 `|tau_vs_l2r|` 降序排列 per-head。

### 4.2 驱动 / 聚合层
`scripts/run_per_head_order_scan_ladder.sh`(或 driver py):
- 对 9 个 ckpt × 5 seed 调用核心函数,落盘 `batch_readout/logs/per_head_scan/ckpt{STEP}_seed{S}.json`。
- 聚合脚本读所有 JSON → 每 (step, layer, head) 的 `tau_vs_l2r` mean±std(over 5 seed)→ 汇总 `per_head_scan_summary.json` + TSV。

### 4.3 产出图
`τ_vs_l2r vs training_step` 折线图,3 条线 + error bar(seed std):
- 主序 head(每 step 数据驱动选 |τ| 最大正向 head;若 index 跨 step 稳定则锁定单一 index 画一条线,否则标注变化);
- anti head(最强负向);
- heavy top-4 baseline。
另存 head×step 的 `τ_vs_l2r` 热力图(L×H 每 step 一张或拼图)。

## 5. 验证策略

1. **复刻 sanity**:先只跑 `clean_base step5000, seed0`,人工核对输出 JSON 字段齐全、`first_step_entropy_max==log(64)`、heavy τ_vs_l2r 在合理范围(注意:clean_base ≠ alt_from0,数字不必等于原 0.67,但结构应类似)。
2. **核心函数单测**(`tests/`):
   - 退化图(全 0 off-diag)→ readout 不崩、tau 良定义;
   - 已知 B(构造一个 out-degree 单调递减的图)→ sigma == 预期顺序、tau_vs_l2r 符号正确;
   - seed 可复现:同 seed 两次跑 bit-identical;不同 seed 结果不同。
3. **跑全 ladder**:9×5=45 次扫描,纯前向(无训练),后台跑;按 G1–G4 出结论。

## 6. 非目标(YAGNI)

- 不训练新模型,不做训练-seed 维度(本轮明确只跨 ckpt + 采样 seed)。
- 不复活原 alt_from0 ckpt。
- 不接 hook、不动 Phase-1/Phase-3 训练。
- 不引入新 readout 变体或 NLL。
