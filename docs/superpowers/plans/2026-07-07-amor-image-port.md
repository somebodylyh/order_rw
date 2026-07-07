# AMOR → VQ-Image 移植 Implementation Plan（极简 · text 管线复用版）

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans。Steps 用 checkbox (`- [ ]`)。

**Goal:** 把已锁死的 text AMOR 管线（head选择 → CDL(attention) → gβ 预训练 → anchor+σ_cache 部署 → periodic PG）跑在 **VQ 图像 token** 上（Imagenet64 VQ, patch2x2），验证零先验注意力涌现序在图像模态是否有 headroom。

**关键发现（2026-07-07，已 probe 证实）:** "image 底座" `probe_results_image/vq64_fixed_random_l8h8e512/ckpt_step30000.pt` **就是 text AOGPT 架构**（`vocab_size=8192` VQ codes, `n_layer=8, n_head=8, n_embd=512, block_size=256, block_order_block_len=4`）。text loader `load_chenhe_backbone` 加载它 `state_dict key 完全匹配`；`single_head_B_from_forward` 在 VQ .bin 上抽出有限 (B,64,64) B。layout 与 text AMOR 逐位相同（N=64/BLOCK_LEN=4/HEADS=8/strict257）→ **text 管线原样复用,不写新模块**。原"连续 patch"移植方案（image_order/amor/*）已废弃。

**Architecture:** 复用 `chenhe_rerun/` 全部 orderhead_v3 + gβ + provider + train，只改数据指向（VQ .bin）+ deploy config 的 model dims（8/8/512, vocab 8192）+ ckpt 路径。VQ code 是 uint16 token 流,`_sample_data_windows` 直接读。loss = CE over 8192 codes（同 text CE，PG advantage 通用）。

## Global Constraints

- **零先验不变式**：gβ 只读模型自己的 patch attention；teacher 恒 CDL(C−D+L) on attention；VQ 的 2D patch 布局 / raster 序**绝不进** gβ/teacher/selection，只作 post-hoc oracle（FID / τ_vs_raster）。
- **不碰 text 参照**：`chenhe_rerun/orderhead_v3/*`、`gbeta_cdl_pretrain.py`、`train.py`、`gbeta_provider.py` **只读不改**。VQ 全走参数化入口（`pretrain_gbeta_cdl(data_bin=...)`）+ 新 config 文件。若发现某处硬编码 text 需改动 → STOP 报告,不擅改锁死文件。
- **layout 固定**：N=64、HEADS=8、BLOCK_LEN=4、strict257、None node0。
- **headline metric**：own-order val CE（VQ code NLL）。head selection 只用无监督结构分（不看 val loss）。
- **底座 / 数据**：backbone = `probe_results_image/vq64_fixed_random_l8h8e512/ckpt_step30000.pt`；data = `nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch2x2/{train,val}.bin`。
- **产出目录**：`out/rerun_vq/`（gβ + deploy 输出）；VQ deploy config 放 `chenhe_rerun/config/imagenet64vq/`。

---

## Task 1: gβ 预训练 smoke（证 `pretrain_gbeta_cdl` 在 VQ 上端到端跑通）

**Files:**
- Test: `chenhe_rerun/tests/test_gbeta_vq_pretrain_smoke.py`

**Interfaces:**
- Consumes: `gbeta_cdl_pretrain.pretrain_gbeta_cdl(parent_ckpt, out_dir, *, n_groups, batch_mean_size, epochs, data_bin, device, seed)`
- Produces: 证实小规模调用产出 `g_beta_best.pt` + `gbeta_provenance.json`（含 sel_layer/sel_head、best_val_acc）

- [ ] **Step 1: 写 smoke 测试**

```python
import os, tempfile, torch, pytest

CKPT = "probe_results_image/vq64_fixed_random_l8h8e512/ckpt_step30000.pt"
VQ_BIN = "nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch2x2/train.bin"

@pytest.mark.skipif(not (os.path.exists(CKPT) and os.path.exists(VQ_BIN)),
                    reason="VQ ckpt/data missing")
def test_pretrain_gbeta_cdl_on_vq_smoke():
    import sys; sys.path.insert(0, "chenhe_rerun")
    from gbeta_cdl_pretrain import pretrain_gbeta_cdl
    with tempfile.TemporaryDirectory() as d:
        path = pretrain_gbeta_cdl(
            parent_ckpt=CKPT, out_dir=d,
            n_select=64, n_groups=16, batch_mean_size=16, n_reveal=2,
            epochs=2, data_bin=VQ_BIN, device="cpu", seed=0)
        assert os.path.exists(path)
        assert os.path.exists(os.path.join(d, "gbeta_provenance.json"))
        st = torch.load(path, map_location="cpu", weights_only=False)
        assert st["config"]["N"] == 64
        assert 0 <= st["config"]["sel_layer"] < 8
        assert 0 <= st["config"]["sel_head"] < 8
```

- [ ] **Step 2: 跑,确认 PASS**（run from repo root）

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest chenhe_rerun/tests/test_gbeta_vq_pretrain_smoke.py -v -s`
Expected: PASS。若 `select_best_head` / `_sample_data_windows` 对 VQ 报错 → STOP,记录 traceback,判断是否 text 硬编码。

- [ ] **Step 3: Commit**

```bash
git add chenhe_rerun/tests/test_gbeta_vq_pretrain_smoke.py
git commit -m "test(amor-vq): gβ CDL pretrain smoke on VQ image tokens"
```

---

## Task 2: gβ 全量预训练（VQ backbone + VQ 数据）— GPU 运行

**需用户确认 GPU 后启动。**

**Files:**
- Create: `chenhe_rerun/run_gbeta_vq_pretrain.py`（薄 wrapper 调 `pretrain_gbeta_cdl`,固化 VQ 参数）
- Output: `out/rerun_vq/gbeta_vq_bm16/{g_beta_best.pt, gbeta_provenance.json}`

**Interfaces:**
- Consumes: Task 1 验证过的 `pretrain_gbeta_cdl`
- Produces: VQ gβ ckpt（sel_layer/sel_head = VQ 底座上重选的 canonical head）

- [ ] **Step 1: 写 wrapper**

```python
"""Run gβ CDL pretrain on the VQ image backbone (Imagenet64 VQ patch2x2)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from gbeta_cdl_pretrain import pretrain_gbeta_cdl

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(REPO, "probe_results_image/vq64_fixed_random_l8h8e512/ckpt_step30000.pt")
VQ_BIN = os.path.join(REPO, "nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch2x2/train.bin")
OUT = os.path.join(REPO, "out/rerun_vq/gbeta_vq_bm16")

if __name__ == "__main__":
    pretrain_gbeta_cdl(
        parent_ckpt=CKPT, out_dir=OUT,
        n_select=800, n_groups=1500, batch_mean_size=16, n_reveal=8,
        epochs=40, lr=3e-4, data_bin=VQ_BIN, device="cuda", seed=0)
```

- [ ] **Step 2: 跑全量**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -u chenhe_rerun/run_gbeta_vq_pretrain.py`
Expected: 打印 `Stage A: selected head = LxHy`、逐 epoch `val_pairwise_acc`、末行 `GBETA: .../g_beta_best.pt (LxHy, val_acc=...)`。记录 head 与 val_acc（text 侧 98.44% 仅参考,VQ 不设同等门槛）。

- [ ] **Step 3: Commit**

```bash
git add chenhe_rerun/run_gbeta_vq_pretrain.py
git commit -m "feat(amor-vq): VQ gβ pretrain runner + artifact"
```

---

## Task 3: Headroom gate（CDL-attention σ vs random，own-order CE Δ）

**Files:**
- Create: `chenhe_rerun/eval_cdl_headroom_vq.py`

**Interfaces:**
- Consumes: `load_chenhe_backbone`；`single_head_B_from_forward`；`build_none_separated_B`+`rollout_from_none`（CDL）；VQ val.bin；Task 2 的 sel_layer/sel_head
- Produces: `{ce_cdl, ce_random, delta}`（own-order VQ-code CE，delta<0 表示 CDL 序更好）

- [ ] **Step 1: 实现**（对 val 窗口：random-reveal 提单头 B → CDL rollout 得 σ_cdl → 用 σ_cdl 做 own-order forward 取 CE；对照 random σ 的 CE。全 batch 平均。forward 用 `model.forward_fn(idx, token_orders, return_logits=True)` 取 CE loss。）

- [ ] **Step 2: 跑于 VQ val（≥1000 窗口）**

Run: `python -u chenhe_rerun/eval_cdl_headroom_vq.py --n 1024 --device cuda`
Expected: 打印 `ce_cdl, ce_random, delta`。

- [ ] **Gate 判据（人工）**：
  - `delta` 显著 < 0 → **PASS**,进 Task 4 部署。
  - `delta` ≈ 0 或 >0 → 记 **VQ-image clean negative**（"零先验 CDL(attention) 在 VQ 图像上无 headroom"）。**不灌 raster/locality 先验救**。STOP 报告用户。

- [ ] **Step 3: Commit**

```bash
git add chenhe_rerun/eval_cdl_headroom_vq.py
git commit -m "feat(amor-vq): CDL-attention headroom gate on VQ image"
```

---

## Task 4: VQ deploy config + 三臂对照训练 — GPU 运行

**仅当 Task 3 PASS。需用户确认 GPU。**

**Files:**
- Create: `chenhe_rerun/config/imagenet64vq/gbeta_anchor_pgonly_vq.py`（PG-only 生产臂）
- Create: `chenhe_rerun/config/imagenet64vq/gbeta_anchor_frozen_vq.py`（frozen 对照）
- Create: `chenhe_rerun/config/imagenet64vq/random_baseline_vq.py`（random 序对照）
- 前置确认：train.py 如何把 `dataset` 解析到 VQ .bin（读 `chenhe_rerun/train.py` data 加载段；若只认注册的 dataset 名,加一个指向 VQ 目录的 dataset 分支或用显式 bin 路径参数）。

**Interfaces:**
- Consumes: VQ 底座（init_from_ckpt / gbeta_parent_ckpt）；Task 2 VQ gβ（gbeta_ckpt）
- Produces: 三个 out_dir 下的 own-order val CE 曲线

**Config 关键差异（相对 text `gbeta_anchor_joint_pgonly_from20k.py`）:**
```python
# —— 必改：model dims 匹配 VQ 底座 ——
n_layer = 8; n_head = 8; n_embd = 512        # text 是 4/8/384
vocab_size = 8192                             # VQ codes（若 config 支持;否则从 ckpt model_args 继承）
block_size = 256; block_order_block_len = 4   # 不变
# —— 数据指向 VQ ——
dataset = 'imagenet64vq_patch2x2'             # 需 train.py 能解析到 nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch2x2
permute_seed = 42; permute_mode = 'block'
# —— ckpt 路径 ——
init_from = 'ckpt'
init_from_ckpt = 'probe_results_image/vq64_fixed_random_l8h8e512/ckpt_step30000.pt'
gbeta_ckpt = 'out/rerun_vq/gbeta_vq_bm16/g_beta_best.pt'
gbeta_parent_ckpt = 'probe_results_image/vq64_fixed_random_l8h8e512/ckpt_step30000.pt'
gbeta_anchor_size = 16; gbeta_probe_order_mode = 'deployment'
aogpt_train_mode = 'GBetaFrozenOrder'
# PG-only 臂沿用 text 生产值：tau=0.05, k=4, update_every=100, adv_clip=0.1, pg_lr=3e-5
```

- [ ] **Step 1:** 读 `chenhe_rerun/train.py` 的 dataset→bin 解析段,确认 VQ 接入方式（注册 or 显式路径）。若需改 train.py 的 dataset 注册表 → 只加分支不改现有,否则 STOP。
- [ ] **Step 2:** 写三个 config。
- [ ] **Step 3:** 各跑（示例 PG-only）：
```bash
python -u chenhe_rerun/train.py chenhe_rerun/config/imagenet64vq/gbeta_anchor_pgonly_vq.py
```
- [ ] **Step 4:** 对比三臂 own-order val CE @ 终点。判据：PG-only ≤ frozen ≤ random。
- [ ] **Step 5: post-hoc oracle（不进训练）**：VQ decode 涌现 σ → 像素 → FID(相对)；σ vs raster 的 Kendall τ。仅报告对齐度。
- [ ] **Step 6: Commit** configs + 结果摘要。

---

## Self-Review

- **Spec 覆盖**：head选择+CDL+gβ(Task2) → headroom gate(Task3) → 部署+PG(Task4)，AMOR 全阶段。零先验落在 Global Constraints + Task3 gate + Task4 post-hoc oracle。
- **相对原 plan 的收敛**：因底座=text 架构,原 6 阶段新模块（image_A_align/strict65/head_selection/provider/train）全部消除,塌缩为「参数化调用 + config」。
- **执行期需现场核对（已标注,非 placeholder）**：Task1 证 `select_best_head`/`_sample_data_windows` 不含 text 硬编码；Task4 Step1 证 train.py dataset 解析可接 VQ。任一处需改锁死文件 → STOP。
- **Type consistency**：VQ token uint16→int64；单头 B (·,64,64)；gβ config N=64；provenance permute_seed=42/none_mode=model/strict65/single_head 与 provider 断言一致。
