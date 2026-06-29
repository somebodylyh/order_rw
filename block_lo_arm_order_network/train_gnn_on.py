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
        mask = np.isin(examples["seq_indices"], list(seq_indices_subset))
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
    val_size = max(1, int(round(num_sequences * val_fraction)))
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
    last = torch.zeros(B, dtype=torch.long, device=device)

    # Step 0: pick first node (no edge weight to add)
    visited = torch.zeros(B, dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(A_dev, visited, last)
    chosen = logits.argmax(dim=-1)
    visited = visited | (1 << chosen)
    last = chosen

    for step in range(1, N):
        with torch.no_grad():
            logits = model(A_dev, visited, last)
        chosen = logits.argmax(dim=-1)
        edge_w = A_dev[torch.arange(B, device=device), chosen, last]
        edge_w = torch.where(
            torch.isfinite(edge_w), edge_w, torch.zeros((), device=device)
        )
        total_weight += edge_w
        visited = visited | (1 << chosen)
        last = chosen

    return total_weight


def evaluate_greedy_ratio(model, A_val, device):
    """Greedy/DP weight ratio on held-out A matrices."""
    A_tensor = torch.as_tensor(A_val, dtype=torch.float32)
    weights = compute_greedy_path_weight(model, A_tensor, device)
    return weights.cpu().numpy()


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
    greedy_weights = evaluate_greedy_ratio(model, val_A, device)

    dp_data = np.load(args.dp_data)
    dp_weights = dp_data["weights"][val_seq_list]
    ratio = greedy_weights / dp_weights
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
