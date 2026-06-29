"""Route A supervised Order Network training on DP next-step labels."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from config import Config
from order_network import RouteATransformerOrderNetwork


DEFAULT_DATA = "probe_results/on_training_data.npz"
DEFAULT_OUTPUT = "probe_results/route_a_on_supervised.pt"


class RouteAStateDataset(Dataset):
    """Dataset of (visited_mask, last_node) -> next_node examples."""

    def __init__(self, data):
        self.visited_masks = torch.as_tensor(
            data["visited_masks"].astype(np.int64),
            dtype=torch.long,
        )
        self.last_nodes = torch.as_tensor(
            data["last_nodes"].astype(np.int64),
            dtype=torch.long,
        )
        self.next_nodes = torch.as_tensor(
            data["next_nodes"].astype(np.int64),
            dtype=torch.long,
        )
        if not (
            len(self.visited_masks) == len(self.last_nodes) == len(self.next_nodes)
        ):
            raise ValueError("visited_masks, last_nodes, and next_nodes length mismatch")

    def __len__(self):
        return len(self.next_nodes)

    def __getitem__(self, index):
        return (
            self.visited_masks[index],
            self.last_nodes[index],
            self.next_nodes[index],
        )


def load_state_dataset(path):
    data = np.load(path)
    return RouteAStateDataset(data)


def evaluate_next_step_accuracy(model, dataset, batch_size, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for visited_masks, last_nodes, next_nodes in loader:
            visited_masks = visited_masks.to(device)
            last_nodes = last_nodes.to(device)
            next_nodes = next_nodes.to(device)
            logits = model(visited_masks, last_nodes)
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


def split_dataset(dataset, val_fraction, seed):
    val_size = int(round(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def resolve_device(requested, cuda_available=None):
    if cuda_available is None:
        cuda_available = torch.cuda.is_available()
    if requested.startswith("cuda") and not cuda_available:
        print("CUDA requested but unavailable; falling back to CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(requested)


def train_supervised(model, train_dataset, val_dataset, args, device):
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    model.to(device)

    history = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        for visited_masks, last_nodes, next_nodes in train_loader:
            visited_masks = visited_masks.to(device)
            last_nodes = last_nodes.to(device)
            next_nodes = next_nodes.to(device)

            logits = model(visited_masks, last_nodes)
            loss = F.cross_entropy(logits, next_nodes)
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += loss.item() * next_nodes.numel()
            total_correct += (logits.argmax(dim=-1) == next_nodes).sum().item()
            total += next_nodes.numel()

        train_metrics = {
            "loss": total_loss / max(total, 1),
            "accuracy": total_correct / max(total, 1),
            "num_examples": total,
        }
        val_metrics = evaluate_next_step_accuracy(
            model,
            val_dataset,
            batch_size=args.batch_size,
            device=device,
        )
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
        description="Train Route A Transformer ON on DP next-step labels."
    )
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=4096)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    Config.seed = args.seed
    Config.set_seed()
    device = resolve_device(args.device)

    dataset = load_state_dataset(args.data)
    train_dataset, val_dataset = split_dataset(
        dataset,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    model = RouteATransformerOrderNetwork(
        num_blocks=Config.num_blocks,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )
    print(
        f"Loaded {len(dataset)} examples "
        f"({len(train_dataset)} train / {len(val_dataset)} val)"
    )
    print(f"Model parameters: {model.count_parameters():,}")
    history = train_supervised(model, train_dataset, val_dataset, args, device)

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
