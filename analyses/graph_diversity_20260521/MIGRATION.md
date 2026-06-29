# Migration Plan: order_lyu → 8-GPU Machine

**Date**: 2026-05-21
**Source**: `/home/admin/lyuyuhuan/order_lyu` (2× RTX 4090)
**Target**: 8-GPU machine (path TBD — set `$TARGET` variable)
**Total needed**: ~3.8 GB (minimal) | ~11 GB (with round-2 ckpts)

---

## Step 0: Git Status (DONE)

- Branch: `master`, remote: `origin` → `https://github.com/somebodylyh/order_rw.git`
- 37 commits ahead of `origin/master` — needs `git push`
- 2 modified tracked files: `block_lo_arm_order_network/data/Imagenet64VQ_f4_800k/prepare.py`, `image_order/extract_image_attention.py`
- **~75 GB untracked** (ckpt, npy, bin, pycache) — NONE should be committed
- `.gitignore` only has `.worktrees/`

---

## Step 1: What to Sync

### 1a. Git-tracked code (push only, no rsync)

```bash
# On old machine:
cd /home/admin/lyuyuhuan/order_lyu
git push origin master   # push 37 local commits
```

**Before push**: add `.gitignore` to exclude large binaries (see Section 2).

### 1b. Training data (MUST SYNC, ~1.2 GB)

| Path | Size | Purpose |
|---|---|---|
| `nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch8x8/train.bin` | ~800M | E3-ctrl-small Phase 3 training |
| `block_lo_arm_order_network/data/Imagenet32VQ_f4_800k_seq64/train.bin` | ~200M | E2 small/large (reference) |
| `/home/admin/lyuyuhuan/order-shakespeare/nanoGPT/data/wikitext103/train.bin` | 227M | Text setup |

### 1c. Key checkpoints (MUST SYNC, ~0.9 GB)

| Path | Size | Purpose |
|---|---|---|
| `nanogpt-learned-order/out/image_large/imagenet64_vq_f4_800k/seq256/patch2x2_full_baseline_l4h8e256/ckpt.pt` | ~95M | E3-ctrl-small final baseline |
| `nanogpt-learned-order/out/image_alignment/e2_imagenet32_vqf4_seq64_l4h8e256/ckpt.pt` | 95M | E2 small (uniform_noisy ref) |
| `nanogpt-learned-order/out/image_alignment/e2_large_imagenet32_vqf4_seq64_l8h8e512/ckpt.pt` | 423M | E2 large (uniform_noisy ref) |
| `block_lo_arm_order_network/probe_results/clean_method_graph_rw_a10_to_a095_30k60k/ckpt_step50000.pt` | ~100M | Text clean_method step 50k |

### 1d. Attention graphs (MUST SYNC, ~50 MB)

| Path | Size |
|---|---|
| `probe_results_image_large/imagenet64_vqf4_full_l4h8e256_patch2x2_control/A_global.npy` | ~320K |
| `probe_results_image_large/imagenet64_vqf4_full_l4h8e256_patch2x2_control/A_block_8x8.npy` | ~32K |
| `probe_results_image_large/imagenet64_vqf4_full_l4h8e256_patch2x2_control/B_global.npy` | ~320K |
| `probe_results_image_large/imagenet64_vqf4_full_l4h8e256_patch2x2_control/diagnostics.json` | ~1K |
| `probe_results_image/e2_imagenet32_vqf4_seq64/attention/A_global.npy` | ~32K |
| `probe_results_image/e2_large_imagenet32_vqf4_seq64_l8h8e512/attention/A_global.npy` | ~32K |
| `block_lo_arm_order_network/probe_results/clean_method_graph_rw_a10_to_a095_30k60k/A_global_step50000.npy` | ~32K |
| `probe_results_image_large/paper_ready_results/table*.tsv` | ~50K total |

### 1e. Round-2 / Phase 1.5 results (MUST SYNC for Task 1.2, ~1 GB)

| Path | Size |
|---|---|
| `probe_results_image_large/grw_e3ctrlsmall_round2_multiseed_minimal/` | ~10M |
| `probe_results_image_large/grw_e3ctrlsmall_round2_5arm/` | ~100M |
| `probe_results_image_large/grw_e3ctrlsmall_round2_seed*_minimal/` | ~200M total |
| `probe_results_image_large/grw_e3ctrlsmall/cont_*/` (attention subdirs only) | ~50M total |
| `probe_results_image_large/grw_e3ctrlsmall/baseline_attention/` | ~5M |
| `probe_results_image_large/graph_regime_cross_graph_overnight/` | ~1M |
| `probe_results_image_large/imagenet64_vqf4_full_l4h8e256_patch2x2_control/` | ~5M |
| `analyses/phase1_5_20260521/` | ~5M |
| `analyses/graph_diversity_20260521/` (TSV + report only) | ~5M |

### 1f. Round-2 training checkpoints (OPTIONAL for Phase 3 continuation, ~7 GB)

These are the 5-arm continuation checkpoints. Only sync if you plan to resume Graph-RW training from them:

| Path | Size |
|---|---|
| `probe_results_image_large/grw_e3ctrlsmall/cont_random/ckpt_*.pt` | ~500M |
| `probe_results_image_large/grw_e3ctrlsmall/cont_graph_rw/ckpt_*.pt` | ~500M |
| `probe_results_image_large/grw_e3ctrlsmall/cont_Bcov_balanced/` (in round2_5arm/) | ~500M |
| `probe_results_image_large/grw_e3ctrlsmall/cont_eps015/ckpt_*.pt` | ~500M |
| `probe_results_image_large/grw_e3ctrlsmall/cont_raster/ckpt_*.pt` | ~500M |
| text Graph-RW variants (clean_method_*) | ~noise tuning only, skip |

**Recommendation**: sync round-2 eval TSVs + attention .npy; skip the checkpoints unless explicitly needed for Phase 2.8 (frozen eval).

### 1g. DO NOT SYNC

- `block_lo_arm_order_network/probe_results/` (~58G of old text experiments)
- `AO-GPT-MDM/` (3.2G, old probe code)
- `__pycache__/` anywhere
- `.ipynb_checkpoints/`
- `*.log` files > 10MB
- `nanogpt-learned-order/out/` intermediate checkpoints (keep only the 4 key ones)

---

## Step 2: Add .gitignore Before Push

Create a `.gitignore` to prevent accidental commit of large binaries:

```bash
cat >> /home/admin/lyuyuhuan/order_lyu/.gitignore << 'GITIGNORE_EOF'
# Large binaries (never commit)
*.pt
*.bin
*.npy
*.npz
*.pth
*.safetensors
*.ckpt
*.tar
*.tar.gz
*.zip

# Directories with large artifacts
nanogpt-learned-order/data/
nanogpt-learned-order/out/
AO-GPT-MDM/
probe_results/
probe_results_image/
probe_results_image_large/
cv/
logs/
block_lo_arm_order_network/probe_results/
block_lo_arm_order_network/probe_results_image_large/
block_lo_arm_order_network/data/Imagenet32VQ_f4_800k_seq64/train.bin
block_lo_arm_order_network/data/Imagenet64VQ_f4_800k/*.npy
analyses/*/*.npy

# Bytecode
__pycache__/
*.pyc
*.pyo

# Claude/IDE
.claude/
.codex
.ds_env

# Images
*.png
*.jpg

# Logs
*.log
GITIGNORE_EOF

# Then push
git add .gitignore
git add -u   # stage modifications to tracked files
git commit -m "chore: add .gitignore to protect large binaries from accidental commit"
git push origin master
```

---

## Step 3: Rsync Commands

Set `TARGET` on the old machine before running:

```bash
TARGET="user@new-machine:/path/to/order_lyu"

# Create target dir
ssh "${TARGET%%:*}" "mkdir -p ${TARGET#*:}/nanogpt-learned-order/data"
ssh "${TARGET%%:*}" "mkdir -p ${TARGET#*:}/nanogpt-learned-order/out/image_large/imagenet64_vq_f4_800k/seq256/patch2x2_full_baseline_l4h8e256"
ssh "${TARGET%%:*}" "mkdir -p ${TARGET#*:}/nanogpt-learned-order/out/image_alignment/e2_imagenet32_vqf4_seq64_l4h8e256"
ssh "${TARGET%%:*}" "mkdir -p ${TARGET#*:}/nanogpt-learned-order/out/image_alignment/e2_large_imagenet32_vqf4_seq64_l8h8e512"

# --- Tier 1: Critical (training data) ---
rsync -avP nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch8x8/train.bin \
    "$TARGET/nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch8x8/"
rsync -avP block_lo_arm_order_network/data/Imagenet32VQ_f4_800k_seq64/train.bin \
    "$TARGET/block_lo_arm_order_network/data/Imagenet32VQ_f4_800k_seq64/"
rsync -avP /home/admin/lyuyuhuan/order-shakespeare/nanoGPT/data/wikitext103/train.bin \
    "$TARGET/../order-shakespeare/nanoGPT/data/wikitext103/"

# --- Tier 2: Key checkpoints ---
rsync -avP nanogpt-learned-order/out/image_large/imagenet64_vq_f4_800k/seq256/patch2x2_full_baseline_l4h8e256/ \
    "$TARGET/nanogpt-learned-order/out/image_large/imagenet64_vq_f4_800k/seq256/patch2x2_full_baseline_l4h8e256/"
rsync -avP nanogpt-learned-order/out/image_alignment/e2_imagenet32_vqf4_seq64_l4h8e256/ \
    "$TARGET/nanogpt-learned-order/out/image_alignment/e2_imagenet32_vqf4_seq64_l4h8e256/"
rsync -avP nanogpt-learned-order/out/image_alignment/e2_large_imagenet32_vqf4_seq64_l8h8e512/ \
    "$TARGET/nanogpt-learned-order/out/image_alignment/e2_large_imagenet32_vqf4_seq64_l8h8e512/"

# Text checkpoint (just ckpt + A_global, not all checkpoints)
rsync -avP block_lo_arm_order_network/probe_results/clean_method_graph_rw_a10_to_a095_30k60k/ckpt_step50000.pt \
    "$TARGET/block_lo_arm_order_network/probe_results/clean_method_graph_rw_a10_to_a095_30k60k/"
rsync -avP block_lo_arm_order_network/probe_results/clean_method_graph_rw_a10_to_a095_30k60k/A_global_step50000.npy \
    "$TARGET/block_lo_arm_order_network/probe_results/clean_method_graph_rw_a10_to_a095_30k60k/"

# --- Tier 3: Attention graphs + paper-ready results ---
rsync -avP probe_results_image_large/ "$TARGET/probe_results_image_large/"
rsync -avP probe_results_image/ "$TARGET/probe_results_image/"
rsync -avP analyses/ "$TARGET/analyses/"
rsync -avP scripts/ "$TARGET/scripts/"

# --- Tier 4: Code (git clone is preferred, but rsync for quick local test) ---
# On new machine, do: git clone https://github.com/somebodylyh/order_rw.git
# Then rsync the untracked scripts from old machine:
rsync -avP \
    block_lo_arm_order_network/*.py \
    block_lo_arm_order_network/configs/ \
    block_lo_arm_order_network/tests/ \
    "$TARGET/block_lo_arm_order_network/"
```

---

## Step 4: New Machine Setup (after rsync)

### 4a. Clone repo

```bash
git clone https://github.com/somebodylyh/order_rw.git /path/to/order_lyu_new
cd /path/to/order_lyu_new

# If you just rsynced the whole thing, skip clone and just:
git init   # won't work since .git is rsynced — just use the rsynced .git
```

### 4b. Python dependencies

```bash
pip install torch numpy pandas scikit-learn matplotlib tqdm datasets transformers
# Optional: pip install umap-learn

# GPU check
python -c "import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

### 4c. Directory structure checklist after sync

The new machine should have at minimum:
```
order_lyu/
├── block_lo_arm_order_network/
│   ├── data/
│   │   └── Imagenet32VQ_f4_800k_seq64/train.bin
│   ├── configs/
│   ├── directed_graph_policy.py   # build_directed_graph(), compute_source()
│   ├── graph_regime_diagnostic.py  # diagnose()
│   ├── train_aogpt_graph_rw.py
│   ├── train_clean_base.py
│   └── ...
├── nanogpt-learned-order/
│   ├── data/
│   │   └── Imagenet64VQ_f4_800k_full_patch8x8/train.bin
│   ├── out/
│   │   ├── image_large/
│   │   │   └── imagenet64_vq_f4_800k/seq256/patch2x2_full_baseline_l4h8e256/ckpt.pt
│   │   └── image_alignment/
│   │       ├── e2_imagenet32_vqf4_seq64_l4h8e256/ckpt.pt
│   │       └── e2_large_imagenet32_vqf4_seq64_l8h8e512/ckpt.pt
│   └── AOGPT.py  (from git clone)
├── probe_results_image_large/
│   ├── imagenet64_vqf4_full_l4h8e256_patch2x2_control/
│   │   ├── A_global.npy
│   │   ├── A_block_8x8.npy
│   │   └── B_global.npy
│   ├── paper_ready_results/table*.tsv
│   └── grw_e3ctrlsmall_round2_*/
├── probe_results_image/
│   └── e2_*/attention/A_global.npy
├── analyses/
│   ├── graph_diversity_20260521/
│   └── phase1_5_20260521/
├── scripts/
└── ../order-shakespeare/nanoGPT/data/wikitext103/train.bin
```

### 4d. Preflight script

Run this on the new machine to verify everything works:

```bash
#!/bin/bash
# preflight.sh — run on new machine before any training

set -e
PROJECT_ROOT="/path/to/order_lyu_new"
cd "$PROJECT_ROOT"

echo "=== 1. Python imports ==="
python -c "
import torch; import numpy as np; import sys
sys.path.insert(0, 'block_lo_arm_order_network')
sys.path.insert(0, 'nanogpt-learned-order')
from directed_graph_policy import build_directed_graph, compute_source
from graph_regime_diagnostic import diagnose
print('  OK: all imports pass')
"

echo "=== 2. GPU check ==="
python -c "
import torch
n = torch.cuda.device_count()
for i in range(n):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}, mem={torch.cuda.get_device_properties(i).total_mem/1e9:.1f}GB')
if n < 8:
    print(f'  WARNING: expected 8 GPUs, found {n}')
else:
    print('  OK: 8 GPUs available')
"

echo "=== 3. Training data checks ==="
for F in \
    nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch8x8/train.bin \
    block_lo_arm_order_network/data/Imagenet32VQ_f4_800k_seq64/train.bin \
    /home/admin/lyuyuhuan/order-shakespeare/nanoGPT/data/wikitext103/train.bin
do
    if [ -f "$F" ]; then
        SZ=$(du -h "$F" | cut -f1)
        echo "  OK: $F ($SZ)"
    else
        echo "  MISSING: $F"
    fi
done

echo "=== 4. Checkpoint checks ==="
for F in \
    nanogpt-learned-order/out/image_large/imagenet64_vq_f4_800k/seq256/patch2x2_full_baseline_l4h8e256/ckpt.pt \
    nanogpt-learned-order/out/image_alignment/e2_imagenet32_vqf4_seq64_l4h8e256/ckpt.pt \
    nanogpt-learned-order/out/image_alignment/e2_large_imagenet32_vqf4_seq64_l8h8e512/ckpt.pt \
    block_lo_arm_order_network/probe_results/clean_method_graph_rw_a10_to_a095_30k60k/ckpt_step50000.pt
do
    if [ -f "$F" ]; then
        echo "  OK: $F"
    else
        echo "  MISSING: $F"
    fi
done

echo "=== 5. Attention graph checks ==="
for F in \
    probe_results_image_large/imagenet64_vqf4_full_l4h8e256_patch2x2_control/A_global.npy \
    probe_results_image_large/imagenet64_vqf4_full_l4h8e256_patch2x2_control/A_block_8x8.npy \
    probe_results_image_large/imagenet64_vqf4_full_l4h8e256_patch2x2_control/B_global.npy \
    probe_results_image/e2_imagenet32_vqf4_seq64/attention/A_global.npy \
    probe_results_image/e2_large_imagenet32_vqf4_seq64_l8h8e512/attention/A_global.npy
do
    if [ -f "$F" ]; then
        echo "  OK: $F"
    else
        echo "  MISSING: $F"
    fi
done

echo "=== 6. g(B) diagnostic smoke test ==="
python -c "
import sys, numpy as np
sys.path.insert(0, 'block_lo_arm_order_network')
sys.path.insert(0, 'nanogpt-learned-order')
from directed_graph_policy import build_directed_graph, compute_source
from graph_regime_diagnostic import diagnose

# E3 control small block
A = np.load('probe_results_image_large/imagenet64_vqf4_full_l4h8e256_patch2x2_control/A_block_8x8.npy')
B = build_directed_graph(A.astype(np.float64))
g = diagnose(B, 'grid2d', 8)
print(f'  E3_ctrl_small: readiness={g[\"readiness_strength\"]:.2f}, p_nbr={g[\"p_nbr_le1\"]:.3f}, '
      f'locality={g[\"locality_score\"]:.3f}, directionality={g[\"directionality\"]:.3f}')
assert g['p_nbr_le1'] > 0.9, f'E3 p_nbr_le1 should be >0.9, got {g[\"p_nbr_le1\"]}'
assert g['locality_score'] > 0.5, f'E3 locality should be >0.5, got {g[\"locality_score\"]}'
print('  OK: E3-control-small block graph diagnostic matches expected')

# E2 (should be uniform_noisy)
A2 = np.load('probe_results_image/e2_imagenet32_vqf4_seq64/attention/A_global.npy')
B2 = build_directed_graph(A2.astype(np.float64))
g2 = diagnose(B2, 'grid2d', 8)
print(f'  E2_small: readiness={g2[\"readiness_strength\"]:.2f}, entropy={g2[\"row_entropy\"]:.3f}, '
      f'locality={g2[\"locality_score\"]:.3f}')
assert g2['row_entropy'] > 0.98, f'E2 entropy should be >0.98'
print('  OK: E2 graph diagnostic matches expected (uniform_noisy)')

# Text
A3 = np.load('block_lo_arm_order_network/probe_results/clean_method_graph_rw_a10_to_a095_30k60k/A_global_step50000.npy')
B3 = build_directed_graph(A3.astype(np.float64))
g3 = diagnose(B3, 'seq1d', 0)
print(f'  Text: readiness={g3[\"readiness_strength\"]:.2f}, directionality={g3[\"directionality\"]:.3f}')
assert g3['directionality'] > 0.7, f'Text directionality should be >0.7'
print('  OK: Text graph diagnostic matches expected (readiness_dominant)')
print('  ALL g(B) diagnostics PASSED')
"

echo "=== 7. Block permutation / inverse_block_perm check ==="
python -c "
import sys, numpy as np
sys.path.insert(0, 'block_lo_arm_order_network')
sys.path.insert(0, 'nanogpt-learned-order')
from AOGPT import AOGPTConfig, AOGPT
import torch

# Load E3 model
ckpt = torch.load('nanogpt-learned-order/out/image_large/imagenet64_vq_f4_800k/seq256/patch2x2_full_baseline_l4h8e256/ckpt.pt', map_location='cpu')
m_args = ckpt.get('model_args', {})
cfg = ckpt.get('config', {})
for k in ['block_size', 'vocab_size', 'n_layer', 'n_head', 'n_embd', 'dropout', 'bias', 'block_order_block_len', 'order_impl']:
    if k in m_args or k in cfg:
        continue
model_args = {}
for k in ['block_size', 'vocab_size', 'n_layer', 'n_head', 'n_embd', 'dropout', 'bias', 'block_order_block_len', 'order_impl']:
    if k in m_args:
        model_args[k] = m_args[k]
    elif k in cfg:
        model_args[k] = cfg[k]
model_args.setdefault('force_manual_attention', True)

print(f'  block_order_block_len={model_args.get(\"block_order_block_len\")}')
print(f'  block_size={model_args.get(\"block_size\")}')
print(f'  n_layer={model_args.get(\"n_layer\")}, n_head={model_args.get(\"n_head\")}')

model = AOGPT(AOGPTConfig(**model_args))
print('  OK: model loads, inverse_block_perm check by construction')
"

echo ""
echo "=== PREFLIGHT COMPLETE ==="
echo "If all checks pass, proceed to Task 1.2 (Round-2 TSV aggregation)."
```

---

## Step 5: Task Priority on New Machine

After preflight passes:

1. **Task 1.2** (IMMEDIATE): Round-2 multi-seed aggregate TSV → spot-check + CI/sign-count
   - Scripts: `scripts/aggregate_multiseed_minimal.py`, `scripts/final_results_tables.py`
   - Data: `probe_results_image_large/grw_e3ctrlsmall_round2_multiseed_minimal/`

2. **Only after Task 1.2 review passes** → Phase 3 CEM oracle

3. **Phase 3 CEM oracle** (LINEAR FLOW):
   a. Collecting (compute task loss for N candidate w* per B)
   b. Searching (CEM w* in the g(B) space)
   c. Fitting (MLP g(B) → w*)
   d. Evaluating (generalization, causal ablation)

4. **Reminder invariants** (from CLAUDE.md/Phase 1.5):
   - patch2x2: spatial `token_to_patch_indices` aggregation ONLY (NEVER block_len=4 contiguous)
   - physical→model order: `_forward_with_block_orders` / `inverse_block_perm` remap
   - CEM input: all-head batch/rolling-average B (n≈30 default)
   - Layer-0 head-selected B: optional ablation only
