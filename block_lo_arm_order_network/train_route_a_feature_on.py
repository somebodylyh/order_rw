"""A-conditioned Route A ON baseline using per-candidate DP features."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from config import Config
from order_network import (
    OrderNetwork,
    extract_candidate_features_from_masks,
    extract_rich_candidate_features_from_masks,
)
from train_route_a_on import resolve_device


DEFAULT_DATA = "probe_results/on_training_data.npz"
DEFAULT_A = "probe_results/A_n16_direct_100x5.npy"
DEFAULT_OUTPUT = "probe_results/route_a_feature_on.pt"


def extract_features(A, visited_masks, last_nodes, mode):
    if mode == "compact":
        return extract_candidate_features_from_masks(A, visited_masks, last_nodes)
    if mode == "rich":
        return extract_rich_candidate_features_from_masks(A, visited_masks, last_nodes)
    raise ValueError(f"Unknown feature mode: {mode}")


class RouteAFeatureDataset(Dataset):
    """Dataset that joins next-step examples with their sequence A matrix."""

    def __init__(self, examples, A_matrices):
        self.A_matrices = torch.as_tensor(A_matrices, dtype=torch.float32)
        self.seq_indices = torch.as_tensor(
            examples["seq_indices"].astype(np.int64),
            dtype=torch.long,
        )
        self.visited_masks = torch.as_tensor(
            examples["visited_masks"].astype(np.int64),
            dtype=torch.long,
        )
        self.last_nodes = torch.as_tensor(
            examples["last_nodes"].astype(np.int64),
            dtype=torch.long,
        )
        self.next_nodes = torch.as_tensor(
            examples["next_nodes"].astype(np.int64),
            dtype=torch.long,
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


def load_feature_dataset(data_path, a_path):
    examples = np.load(data_path)
    A_matrices = np.load(a_path)
    return RouteAFeatureDataset(examples, A_matrices)


def split_dataset(dataset, val_fraction, seed):
    val_size = int(round(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


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
            features, revealed = extract_features(
                A,
                visited_masks,
                last_nodes,
                mode=getattr(model, "feature_mode", "compact"),
            )
            logits = model(features, revealed)
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
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    history = []

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
            features, revealed = extract_features(
                A,
                visited_masks,
                last_nodes,
                mode=args.feature_mode,
            )
            logits = model(features, revealed)
            loss = F.cross_entropy(logits, next_nodes)
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            loss_sum += loss.item() * next_nodes.numel()
            correct += (logits.argmax(dim=-1) == next_nodes).sum().item()
            total += next_nodes.numel()

        train_metrics = {
            "loss": loss_sum / max(total, 1),
            "accuracy": correct / max(total, 1),
        }
        val_metrics = evaluate(model, val_dataset, args.batch_size, device)
        row = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(row)
        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"train_loss={row['train_loss']:.4f} "
            f"train_acc={row['train_accuracy']:.3f} | "
            f"val_loss={row['val_loss']:.4f} "
            f"val_acc={row['val_accuracy']:.3f}",
            flush=True,
        )

    return history


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train A-conditioned Route A feature Order Network."
    )
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--a-matrices", default=DEFAULT_A)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--feature-mode", choices=["compact", "rich"], default="rich")
    return parser.parse_args()


def main():
    args = parse_args()
    Config.seed = args.seed
    Config.set_seed()
    device = resolve_device(args.device)
    dataset = load_feature_dataset(args.data, args.a_matrices)
    train_dataset, val_dataset = split_dataset(
        dataset,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    feature_dim = Config.on_feature_dim
    if args.feature_mode == "rich":
        feature_dim = 5 * Config.num_blocks + 2
    model = OrderNetwork(
        feature_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    model.feature_mode = args.feature_mode
    print(
        f"Loaded {len(dataset)} examples "
        f"({len(train_dataset)} train / {len(val_dataset)} val)"
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    history = train(model, train_dataset, val_dataset, args, device)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "history": history,
        },
        output,
    )
    print(f"Saved checkpoint: {output}")


if __name__ == "__main__":
    main()
