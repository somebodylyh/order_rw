# 迁移到新服务器 — 部署指南

> 本仓库 (`order_rw`) 是 attn-order / AO-GPT 实验主线。代码分散在 **3 个 git 仓库**，
> 大文件（数据 / checkpoint / 结果）**不在 git 里**，需在新服务器单独获取。
> 环境基线：conda env `X1`，Python 3.8.5，CUDA 12.1，2× 24GB GPU。

---

## 1. 代码：三个仓库的布局

| 子目录 | 归属仓库 | 内容 |
|---|---|---|
| 外层（`block_lo_arm_order_network/`, `scripts/`, `image_order/`, `analyses/`, `docs/`） | `github.com/somebodylyh/order_rw` | attn-order 主线代码、训练/FID/诊断脚本 |
| `nanogpt-learned-order/` | `github.com/Yangxiaohehehe/nanogpt-learned-order` (分支 `main`) | **核心模型** `AOGPT.py`、`order_utils.py`、nanogpt baseline 训练器、configs |
| `AO-GPT-MDM/` | 独立 repo（外层已 gitignore） | 旧 MDM 线；**本主线不需要**——所需的 `model_AOGPT_AdaLN6_*` 在 `block_lo_arm_order_network/` 已有副本 |

脚本靠 `sys.path.insert(_REPO/"nanogpt-learned-order")` 找核心模型，所以 **`nanogpt-learned-order` 必须 clone 成外层仓库的子目录**：

```bash
git clone https://github.com/somebodylyh/order_rw.git
cd order_rw
git checkout attn-order-alternating
git clone https://github.com/Yangxiaohehehe/nanogpt-learned-order.git   # 落到 ./nanogpt-learned-order
# AO-GPT-MDM 本主线不需要，可跳过
```

---

## 2. 环境

```bash
conda create -n X1 python=3.8.5 -y && conda activate X1
# 先装匹配目标机 CUDA 的 torch（基线是 cu121）：
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

离线环境用 `hf-mirror.com`：`export HF_ENDPOINT=https://hf-mirror.com`。

---

## 3. 不在 git 里、需在新服务器获取的资产

| 资产 | 大小 | 路径 | 怎么拿 |
|---|---|---|---|
| **VQ-f4 VAE** | 212M | `~/.cache/huggingface/hub/models--xvjiarui--ldm-vq-f4` | `huggingface_hub` 下载 `xvjiarui/ldm-vq-f4`，或直接 rsync 旧机缓存 |
| **patch2x2 数据** | ~430M/份 | `nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch2x2/{train,val}.bin + *_patch_order_indices.npy + meta.pkl` | rsync 旧机最快；或用 `block_lo_arm_order_network/data/Imagenet64VQ_f4_800k/{download_*,rearrange_to_patches.py}` 重新生成 |
| **checkpoints** | 33MB/个 | `probe_results_image/.../ckpt_step*.pt`, `beta_step*.pt`, `A_global_step*.npy` | 不在 git；要么 rsync 旧机，要么按下面重训 |

> 最省事：把旧机的 `~/.cache/huggingface` + 需要的 `data/` + 关键 `ckpt` 直接 `rsync` 过去，避免重新下载/生成 ImageNet。

---

## 4. 跑通验证（迁移后冒烟）

```bash
# A) FID / teacher-forced 重建（需要一个已有 ckpt + 数据 + VAE）
python scripts/fid_alt_vq64.py --mode teacher_forced --tf-pred argmax \
  --orders raster --num-samples 64 --fid-real-samples 64 --device cuda:0 \
  --out-dir /tmp/fid_smoke
# 看 /tmp/fid_smoke/real_val_recon_grid.png 是否为连贯 ImageNet → 数据/VAE/frame 正确

# B) alternating 训练冒烟（从 0，小步数）
python scripts/train_vq64_alternating.py \
  --data-train nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch2x2/train.bin \
  --data-val   nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch2x2/val.bin \
  --meta       nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch2x2/meta.pkl \
  --output-dir /tmp/alt_smoke --max-steps 200 --warmup-start 100 \
  --n-layer 8 --n-head 8 --n-embd 512 --device cuda:0
```

---

## 5. 对齐合作者 config（8L/512）后重训

合作者基线：`n_layer=8, n_head=8, n_embd=512, dropout=0`（我们之前是 4L/256，约 8.3M 参数，
是 FID 到不了 ~40 的主因）。`train_vq64_alternating.py` 现已支持 `--n-layer/--n-head/--n-embd`
（默认即 8/8/512）。现成的 nanogpt-baseline 8L/512 config 在
`block_lo_arm_order_network/configs/image_large/imagenet64_l8h8e512_*.py`。

完整重训（30k，patch2x2，8L/512）：

```bash
python scripts/train_vq64_alternating.py \
  --data-train nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch2x2/train.bin \
  --data-val   nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch2x2/val.bin \
  --meta       nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch2x2/meta.pkl \
  --output-dir probe_results_image/vq64_alt_from0_mlp_patch2x2_l8e512 \
  --max-steps 30000 --warmup-start 3000 --refresh-interval 3000 \
  --alpha-max 0.9 --alpha-ramp 10000 \
  --n-layer 8 --n-head 8 --n-embd 512 --device cuda:0
```
