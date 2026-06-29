# A-conditioned Graph Order Network Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a permutation-equivariant GNN Order Network that generalizes to held-out A matrices (target >= 75% val next-step acc), with a DeepSets ablation baseline.

**Architecture:** Two models share pure-topological node features (7 dim: is_visited, is_last_node, 4 degree features, reveal_ratio). GNN adds 3-layer edge-conditioned message-passing (~60K params). DeepSets scores nodes independently (~15K params). Both trained on 6000 examples (400 seqs), validated on 1500 examples (100 seqs).

**Tech Stack:** PyTorch, numpy, existing DP solver (dp_solver.py), existing AOGPT model for P0 generation.

---

### Task 1: Generate 500 A matrices via P0 N16 direct NLL scoring

**Files:**
- No code changes needed — use existing `p0_n16_direct.py`

`p0_n16_direct.py` already supports `--num_seqs` and `--output` CLI args.

- [ ] **Step 1: Run P0 generation on GPU**

```bash
cd /home/admin/lyuyuhuan/order_lyu/block_lo_arm_order_network
CUDA_VISIBLE_DEVICES=1 python p0_n16_direct.py \
  --num_seqs 500 \
  --num_repeats 5 \
  --output probe_results/A_n16_direct_500x5.npy
```

Expected: ~50 min on 4090. Output: `probe_results/A_n16_direct_500x5.npy` shape (500, 16, 16) float32.
Signal check at end: Adjacent vs Far diff close to 0.02, confirming signal quality.

### Task 2: DP solve all 500 A matrices

**Files:**
- No code changes needed — use existing `p1_validate_dp.py`

- [ ] **Step 1: Run DP batch solve**

```bash
cd /home/admin/lyuyuhuan/order_lyu/block_lo_arm_order_network
python p1_validate_dp.py \
  --input probe_results/A_n16_direct_500x5.npy \
  --output probe_results/dp_n16_direct_500x5.npz \
  --num-blocks 16 \
  --workers 4
```

Expected: ~2-3 min. Output: `probe_results/dp_n16_direct_500x5.npz` with paths, weights, summary stats.
Verify: L2R=0/500, R2L=0/500, avg_adj < 4/15.

### Task 3: Build training data from 500 DP paths

**Files:**
- No code changes needed — use existing `p2_build_training_data.py`

- [ ] **Step 1: Build step-level training examples**

```bash
cd /home/admin/lyuyuhuan/order_lyu/block_lo_arm_order_network
python p2_build_training_data.py \
  --input probe_results/dp_n16_direct_500x5.npz \
  --output probe_results/on_training_data_500.npz \
  --num-blocks 16
```

Expected: <5s. Output: `probe_results/on_training_data_500.npz` with 7500 examples (500 × 15 steps).
Keys: visited_masks, last_nodes, next_nodes, seq_indices, step_indices.

- [ ] **Step 2: Verify data integrity**

```bash
python3 -c "
import numpy as np
d = np.load('probe_results/on_training_data_500.npz')
print(f'Examples: {len(d[\"next_nodes\"])}')  # 7500
print(f'Unique seqs: {len(set(d[\"seq_indices\"]))}')  # 500
# Each seq has exactly 15 steps
for seq_idx in range(5):
    count = (d['seq_indices'] == seq_idx).sum()
    print(f'  Seq {seq_idx}: {count} steps')
"
```

Expected: 7500 examples, 500 unique seqs, 15 steps per seq.

### Task 4: Implement GNN and DeepSets architectures

**Files:**
- Modify: `order_network.py` — append `GraphOrderNetwork` and `DeepSetOrderNetwork` classes

- [ ] **Step 1: Add pure-topological node feature extractor**

Append to `order_network.py`:

```python
# ── Topological Node Features (no position encoding) ────────────────────────

def extract_topological_node_features(
    A: torch.Tensor,
    visited_masks: torch.Tensor,
    last_nodes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pure topological/state features per node. No absolute position.

    Returns:
        features: (B, N, 7) — [is_visited, is_last, in_deg_w, out_deg_w,
                               in_deg_uw, out_deg_uw, reveal_ratio]
        revealed_mask: (B, N) bool
    """
    B, N, _ = A.shape
    device = A.device

    A_safe = torch.where(torch.isfinite(A), A, torch.zeros((), device=device, dtype=A.dtype))
    revealed_mask = masks_to_revealed_bool(visited_masks.to(device), N)

    is_visited = revealed_mask.float().unsqueeze(-1)  # (B, N, 1)
    is_last = torch.zeros(B, N, 1, device=device)
    last_idx = last_nodes.to(device).view(B, 1)
    is_last.scatter_(1, last_idx.unsqueeze(-1), 1.0)

    A_sp = F.softplus(A_safe)  # smooth positive edge weights
    in_deg_w = A_sp.sum(dim=1, keepdim=True)   # (B, N, 1)
    out_deg_w = A_sp.sum(dim=2, keepdim=True)  # (B, N, 1)

    A_mean = A_safe.mean(dim=(1, 2), keepdim=True)
    in_deg_uw = (A_safe > A_mean).float().sum(dim=1, keepdim=True)
    out_deg_uw = (A_safe > A_mean).float().sum(dim=2, keepdim=True)

    reveal_ratio = revealed_mask.float().sum(dim=-1, keepdim=True).unsqueeze(-1) / N

    features = torch.cat([
        is_visited,
        is_last,
        in_deg_w,
        out_deg_w,
        in_deg_uw,
        out_deg_uw,
        reveal_ratio.expand(-1, N, -1),
    ], dim=-1)  # (B, N, 7)

    return features, revealed_mask
```

- [ ] **Step 2: Verify node features compile**

```bash
cd /home/admin/lyuyuhuan/order_lyu/block_lo_arm_order_network
python3 -c "
import torch
from order_network import extract_topological_node_features
A = torch.randn(4, 16, 16)
vm = torch.randint(0, 65536, (4,))
ln = torch.randint(0, 16, (4,))
f, m = extract_topological_node_features(A, vm, ln)
print(f'Features: {f.shape}')  # (4, 16, 7)
print(f'Mask: {m.shape}')       # (4, 16)
assert f.shape == (4, 16, 7)
assert not torch.isnan(f).any()
print('PASS')
"
```

Expected: `Features: torch.Size([4, 16, 7])`, PASS.

- [ ] **Step 3: Add GNN layer**

Append to `order_network.py`:

```python
# ── Edge-Conditioned GNN Layer ──────────────────────────────────────────────

class EdgeGNNLayer(nn.Module):
    """One message-passing layer with learnable edge temperature."""

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_e = nn.Linear(1, d_model, bias=False)
        self.W_update = nn.Linear(2 * d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.log_tau = nn.Parameter(torch.zeros(1))  # temperature for tanh

    def forward(self, h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, N, d_model) node states
            A: (B, N, N) raw edge weights, diagonal -inf

        Returns:
            h_new: (B, N, d_model)
        """
        B, N, _ = A.shape
        A_safe = torch.where(torch.isfinite(A), A, torch.zeros((), device=A.device, dtype=A.dtype))
        tau = F.softplus(self.log_tau) + 1e-3
        w = torch.tanh(A_safe / tau)  # (B, N, N), in [-1, 1]

        h_proj = self.W_v(h)  # (B, N, d)
        messages = torch.einsum('bij,bjd->bid', w, h_proj)  # (B, N, d)

        w_scalar = w.unsqueeze(-1)  # (B, N, N, 1)
        edge_feat = self.W_e(w_scalar).sum(dim=2)  # (B, N, d)

        agg = messages + edge_feat
        combined = torch.cat([h, agg], dim=-1)  # (B, N, 2d)
        h_new = self.norm(h + F.gelu(self.W_update(combined)))
        return h_new
```

- [ ] **Step 4: Verify GNN layer shapes**

```bash
cd /home/admin/lyuyuhuan/order_lyu/block_lo_arm_order_network
python3 -c "
import torch
from order_network import EdgeGNNLayer
layer = EdgeGNNLayer(d_model=64)
h = torch.randn(4, 16, 64)
A = torch.randn(4, 16, 16)
A[:, range(16), range(16)] = float('-inf')
h2 = layer(h, A)
assert h2.shape == (4, 16, 64)
assert not torch.isnan(h2).any()
print(f'Params: {sum(p.numel() for p in layer.parameters()):,}')
print('PASS')
"
```

Expected: ~16K params per layer, PASS.

- [ ] **Step 5: Add GraphOrderNetwork**

Append to `order_network.py`:

```python
# ── Graph Order Network ─────────────────────────────────────────────────────

class GraphOrderNetwork(nn.Module):
    """
    Permutation-equivariant GNN for block ordering.

    Pure topological node features → 3-layer edge-conditioned message-passing
    → readout MLP → per-node logits.
    """

    def __init__(self, num_blocks: int = 16, d_model: int = 64, num_layers: int = 3):
        super().__init__()
        self.num_blocks = num_blocks
        self.d_model = d_model

        self.feature_proj = nn.Linear(7, d_model, bias=False)
        self.gnn_layers = nn.ModuleList([EdgeGNNLayer(d_model) for _ in range(num_layers)])
        self.readout = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self, A: torch.Tensor, visited_masks: torch.Tensor, last_nodes: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            A: (B, N, N) NLL pair score matrices
            visited_masks: (B,) int bitmask
            last_nodes: (B,) long last-revealed block index

        Returns:
            logits: (B, N), visited positions set to -inf
        """
        features, revealed_mask = extract_topological_node_features(
            A, visited_masks, last_nodes
        )  # (B, N, 7)

        h = self.feature_proj(features)  # (B, N, d)
        for layer in self.gnn_layers:
            h = layer(h, A)

        scores = self.readout(h).squeeze(-1)  # (B, N)
        return scores.masked_fill(revealed_mask, float('-inf'))
```

- [ ] **Step 6: Add DeepSetOrderNetwork**

Append to `order_network.py`:

```python
# ── DeepSet Order Network (message-passing ablation) ─────────────────────────

class DeepSetOrderNetwork(nn.Module):
    """
    No message-passing baseline. Same features, same readout MLP.
    Any accuracy delta vs GraphOrderNetwork is purely from graph convolution.
    """

    def __init__(self, num_blocks: int = 16, d_model: int = 64):
        super().__init__()
        self.readout = nn.Sequential(
            nn.Linear(7, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self, A: torch.Tensor, visited_masks: torch.Tensor, last_nodes: torch.Tensor
    ) -> torch.Tensor:
        features, revealed_mask = extract_topological_node_features(
            A, visited_masks, last_nodes
        )  # (B, N, 7)
        scores = self.readout(features).squeeze(-1)  # (B, N)
        return scores.masked_fill(revealed_mask, float('-inf'))
```

- [ ] **Step 7: Verify both models forward pass**

```bash
cd /home/admin/lyuyuhuan/order_lyu/block_lo_arm_order_network
python3 -c "
import torch
from order_network import GraphOrderNetwork, DeepSetOrderNetwork

B = 4
A = torch.randn(B, 16, 16)
A[:, range(16), range(16)] = float('-inf')
vm = torch.randint(0, 65536, (B,))
ln = torch.randint(0, 16, (B,))

gnn = GraphOrderNetwork()
ds = DeepSetOrderNetwork()

gnn_out = gnn(A, vm, ln)
ds_out = ds(A, vm, ln)

print(f'GNN params: {sum(p.numel() for p in gnn.parameters()):,}')
print(f'DeepSets params: {sum(p.numel() for p in ds.parameters()):,}')
print(f'GNN output: {gnn_out.shape}')  # (4, 16)
print(f'DeepSets output: {ds_out.shape}')  # (4, 16)

# Visited nodes are masked to -inf
for b in range(B):
    visited = [(vm[b] >> i) & 1 for i in range(16)]
    for i, v in enumerate(visited):
        if v:
            assert gnn_out[b, i] == float('-inf'), f'GNN node {i} not masked'
            assert ds_out[b, i] == float('-inf'), f'DeepSets node {i} not masked'

print('PASS: all visited nodes correctly masked to -inf')
"
```

Expected: GNN ~60K params, DeepSets ~15K params, PASS.

### Task 5: Write training script

**Files:**
- Create: `train_gnn_on.py`

- [ ] **Step 1: Write train_gnn_on.py**

```python
"""Train GraphOrderNetwork and DeepSetOrderNetwork on DP next-step labels.

Train/val split by A matrix (sequence), not by example.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config import Config
from order_network import GraphOrderNetwork, DeepSetOrderNetwork


DEFAULT_DATA = "probe_results/on_training_data_500.npz"
DEFAULT_A = "probe_results/A_n16_direct_500x5.npy"
DEFAULT_DP = "probe_results/dp_n16_direct_500x5.npz"
DEFAULT_OUTPUT_DIR = "probe_results"


class SequenceSplitDataset(Dataset):
    """Dataset with train/val split by sequence index, not by example."""

    def __init__(self, examples, A_matrices, seq_indices_subset):
        self.A_matrices = torch.as_tensor(A_matrices, dtype=torch.float32)
        mask = np.isin(examples["seq_indices"], seq_indices_subset)
        self.visited_masks = torch.as_tensor(
            examples["visited_masks"][mask].astype(np.int64), dtype=torch.long
        )
        self.last_nodes = torch.as_tensor(
            examples["last_nodes"][mask].astype(np.int64), dtype=torch.long
        )
        self.next_nodes = torch.as_tensor(
            examples["next_nodes"][mask].astype(np.int64), dtype=torch.long
        )
        self.seq_indices = torch.as_tensor(
            examples["seq_indices"][mask].astype(np.int64), dtype=torch.long
        )

    def __len__(self):
        return len(self.next_nodes)

    def __getitem__(self, index):
        seq_idx = self.seq_indices[index]
        return (
            self.A_matrices[seq_idx],
            self.visited_masks[index],
            self.last_nodes[index],
            self.next_nodes[index],
        )


def split_by_sequence(num_sequences, val_fraction, seed):
    """Split sequence indices into train/val sets."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(num_sequences)
    val_size = int(round(num_sequences * val_fraction))
    val_indices = set(indices[:val_size].tolist())
    train_indices = set(indices[val_size:].tolist())
    return train_indices, val_indices


def evaluate(model, dataset, batch_size, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for A, visited_masks, last_nodes, next_nodes in loader:
            A = A.to(device)
            visited_masks = visited_masks.to(device)
            last_nodes = last_nodes.to(device)
            next_nodes = next_nodes.to(device)
            logits = model(A, visited_masks, last_nodes)
            loss = F.cross_entropy(logits, next_nodes, reduction="sum")
            pred = logits.argmax(dim=-1)
            correct += (pred == next_nodes).sum().item()
            total += next_nodes.numel()
            loss_sum += loss.item()
    return {
        "accuracy": correct / max(total, 1),
        "loss": loss_sum / max(total, 1),
        "num_examples": total,
    }


def train(model, train_dataset, val_dataset, args, device):
    loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6
    )

    history = []
    best_val_acc = 0.0
    best_epoch = 0
    best_state = None
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        total = 0
        correct = 0
        loss_sum = 0.0
        for A, visited_masks, last_nodes, next_nodes in loader:
            A = A.to(device)
            visited_masks = visited_masks.to(device)
            last_nodes = last_nodes.to(device)
            next_nodes = next_nodes.to(device)

            logits = model(A, visited_masks, last_nodes)
            loss = F.cross_entropy(logits, next_nodes)

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            loss_sum += loss.item() * next_nodes.numel()
            correct += (logits.argmax(dim=-1) == next_nodes).sum().item()
            total += next_nodes.numel()

        train_acc = correct / max(total, 1)
        train_loss = loss_sum / max(total, 1)
        val_metrics = evaluate(model, val_dataset, args.batch_size, device)
        scheduler.step(val_metrics["accuracy"])

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        })

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.3f}",
            flush=True,
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch + 1} (patience={args.patience})")
            break

    return history, best_state, best_val_acc, best_epoch


def compute_greedy_path_weight(model, A, device):
    """Greedy rollout: argmax next block at each step, sum edge weights."""
    model.eval()
    B, N, _ = A.shape
    A_dev = A.to(device)
    total_weight = torch.zeros(B, device=device)
    paths = []
    visited = torch.zeros(B, dtype=torch.long, device=device)
    last = torch.zeros(B, dtype=torch.long, device=device)  # arbitrary init

    for step in range(N):
        with torch.no_grad():
            logits = model(A_dev, visited, last)
        chosen = logits.argmax(dim=-1)  # (B,)
        paths.append(chosen.cpu().tolist())

        if step > 0:
            prev_last = last
            edge_w = A_dev[torch.arange(B, device=device), chosen, prev_last]
            edge_w = torch.where(
                torch.isfinite(edge_w), edge_w, torch.zeros((), device=device)
            )
            total_weight += edge_w

        visited = visited | (1 << chosen)
        last = chosen

    paths_t = list(zip(*paths))
    return paths_t, total_weight


def evaluate_greedy_ratio(model, A_val, val_seq_indices, device):
    """Greedy/DP weight ratio on held-out A matrices."""
    import numpy as np
    A_tensor = torch.as_tensor(A_val, dtype=torch.float32)
    paths, weights = compute_greedy_path_weight(model, A_tensor, device)
    return {"greedy_paths": paths, "greedy_weights": weights.cpu().numpy()}


def parse_args():
    parser = argparse.ArgumentParser(description="Train Graph/DeepSet Order Network")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--a-matrices", default=DEFAULT_A)
    parser.add_argument("--dp-data", default=DEFAULT_DP)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", choices=["gnn", "deepsets"], default="gnn")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    Config.seed = args.seed
    Config.set_seed()
    device = torch.device(args.device)
    print(f"Device: {device}", flush=True)

    examples = np.load(args.data)
    A_matrices = np.load(args.a_matrices)
    num_sequences = A_matrices.shape[0]
    print(f"Loaded {num_sequences} sequences, {len(examples['next_nodes'])} examples")

    train_seqs, val_seqs = split_by_sequence(
        num_sequences, args.val_fraction, args.seed
    )
    print(f"Train seqs: {len(train_seqs)}, Val seqs: {len(val_seqs)}")

    train_dataset = SequenceSplitDataset(examples, A_matrices, train_seqs)
    val_dataset = SequenceSplitDataset(examples, A_matrices, val_seqs)
    print(f"Train examples: {len(train_dataset)}, Val examples: {len(val_dataset)}")

    if args.model == "gnn":
        model = GraphOrderNetwork(
            num_blocks=Config.num_blocks,
            d_model=args.d_model,
            num_layers=args.num_layers,
        )
        output_name = "gnn_on_best.pt"
    else:
        model = DeepSetOrderNetwork(
            num_blocks=Config.num_blocks, d_model=args.d_model
        )
        output_name = "deepsets_on_best.pt"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}, params: {n_params:,}")

    history, best_state, best_val_acc, best_epoch = train(
        model, train_dataset, val_dataset, args, device
    )

    print(f"\nBest val acc: {best_val_acc:.4f} at epoch {best_epoch}")

    # Greedy/DP ratio on validation A matrices
    val_seq_list = sorted(val_seqs)
    val_A = A_matrices[val_seq_list]
    result = evaluate_greedy_ratio(model, val_A, val_seq_list, device)

    # Load DP weights for comparison
    dp_data = np.load(args.dp_data)
    dp_weights = dp_data["weights"][val_seq_list]
    ratio = result["greedy_weights"] / dp_weights
    print(f"Greedy/DP weight ratio: mean={ratio.mean():.4f}, min={ratio.min():.4f}")

    output = Path(args.output_dir) / output_name
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "args": vars(args),
            "history": history,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "greedy_dp_ratio_mean": float(ratio.mean()),
        },
        output,
    )
    print(f"Saved checkpoint: {output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify training script compiles**

```bash
cd /home/admin/lyuyuhuan/order_lyu/block_lo_arm_order_network
python3 -m py_compile train_gnn_on.py && echo "PASS: compile OK"
```

### Task 6: Train both models and compare

**Files:**
- Run: `train_gnn_on.py` (twice — gnn + deepsets)

- [ ] **Step 1: Train GNN**

```bash
cd /home/admin/lyuyuhuan/order_lyu/block_lo_arm_order_network
CUDA_VISIBLE_DEVICES=1 python3 -u train_gnn_on.py \
  --model gnn \
  --epochs 200 \
  --patience 30 \
  --batch-size 128 \
  --lr 1e-3 \
  --d-model 64 \
  --num-layers 3 \
  --output-dir probe_results
```

Expected: ~30s-2min on GPU. Watch for val_acc climbing past 67.7%.

- [ ] **Step 2: Train DeepSets**

```bash
cd /home/admin/lyuyuhuan/order_lyu/block_lo_arm_order_network
CUDA_VISIBLE_DEVICES=1 python3 -u train_gnn_on.py \
  --model deepsets \
  --epochs 200 \
  --patience 30 \
  --batch-size 128 \
  --lr 1e-3 \
  --d-model 64 \
  --output-dir probe_results
```

Expected: ~15s on GPU. Baseline for message-passing ablation.

- [ ] **Step 3: Compare results**

```bash
python3 -c "
import torch

gnn = torch.load('probe_results/gnn_on_best.pt', map_location='cpu', weights_only=True)
ds = torch.load('probe_results/deepsets_on_best.pt', map_location='cpu', weights_only=True)

print('=== GNN ===')
print(f'Best val acc: {gnn[\"best_val_acc\"]:.4f}')
print(f'Best epoch:  {gnn[\"best_epoch\"]}')
print(f'Greedy/DP ratio: {gnn.get(\"greedy_dp_ratio_mean\", \"N/A\")}')

print()
print('=== DeepSets ===')
print(f'Best val acc: {ds[\"best_val_acc\"]:.4f}')
print(f'Best epoch:  {ds[\"best_epoch\"]}')
print(f'Greedy/DP ratio: {ds.get(\"greedy_dp_ratio_mean\", \"N/A\")}')

print()
delta = gnn['best_val_acc'] - ds['best_val_acc']
print(f'GNN - DeepSets delta: {delta:.4f}')
if delta > 0.05:
    print('Message-passing provides significant benefit over pure features.')
elif delta > 0:
    print('Message-passing provides marginal benefit.')
else:
    print('Message-passing does not help — features alone are sufficient.')
"
```

### Task 7: Run test suite

- [ ] **Step 1: Run all tests**

```bash
cd /home/admin/lyuyuhuan/order_lyu/block_lo_arm_order_network
python3 -m pytest tests/ -v --tb=short
```

Expected: all existing tests pass. New GNN/DeepSets tests should also pass.

### Task 8: Update memory with results

- [ ] **Step 1: Update pipeline_state.md and MEMORY.md**

After training completes, update `pipeline_state.md` description with final val acc numbers and GNN/DeepSets delta. Add new memory index entry if needed for the GNN checkpoint.
