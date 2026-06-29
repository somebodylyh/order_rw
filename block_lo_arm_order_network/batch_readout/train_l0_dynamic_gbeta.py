"""Offline pretraining loop for L0DynamicGBeta.

Label-free: checkpoint selection uses only the selected teacher-imitation
validation loss (ListMLE, pairwise BCE, or Rank-KL). Physical coordinates
(L2R, inv_perm, block_perm) are never read, saved, or used for model
selection.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from batch_readout.l0_dynamic_gbeta import L0DynamicGBeta
from batch_readout.order_distillation_losses import (
    listmle_loss,
    rank_kl_loss,
)
from batch_readout.soft_pairwise import (
    entropy_floor_loss,
    gate_entropy,
    pairwise_accuracy,
    per_head_aux_loss,
    soft_pairwise_bce_loss,
)

LOSS_TYPES = ("listmle", "pairwise_bce", "rank_kl")


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_pretrain_dataset(
    dataset_path: str,
    device: str = "cpu",
    require_consensus_order: bool = False,
    require_pairwise: bool = True,
) -> dict:
    """Load a Task-3 .npz dataset and return tensors + split indices.

    Returns:
        dict with keys: B_raw, teacher_pairwise, teacher_weights (all torch
        tensors on ``device``), plus train_idx, val_idx, test_idx (numpy
        int64 arrays).

    Raises:
        RuntimeError if any forbidden physical field is present.
    """
    forbidden = {"block_perm", "inv_perm", "clean_perm",
                 "physical_order", "l2r_order"}
    with np.load(dataset_path, allow_pickle=True) as z:
        leaked = forbidden & set(z.files)
        if leaked:
            raise RuntimeError(
                f"Dataset {dataset_path} contains forbidden physical fields: "
                f"{sorted(leaked)}.  Refusing to load."
            )

        B_raw = torch.from_numpy(z["B_raw"].copy()).float().to(device)
        if require_pairwise and "teacher_pairwise" not in z.files:
            raise KeyError("teacher_pairwise is required")
        Y_pair = (
            torch.from_numpy(z["teacher_pairwise"].copy()).float().to(device)
            if "teacher_pairwise" in z.files else None
        )
        if require_consensus_order and "teacher_consensus_order" not in z.files:
            raise KeyError("teacher_consensus_order is required")
        consensus_order = (
            torch.from_numpy(z["teacher_consensus_order"].copy()).long().to(device)
            if "teacher_consensus_order" in z.files else None
        )
        wT = torch.from_numpy(z["teacher_weights"].copy()).float().to(device)
        train_idx = z["train_idx"].copy()
        val_idx = z["val_idx"].copy()
        test_idx = z["test_idx"].copy()

    return {
        "B_raw": B_raw,
        "teacher_pairwise": Y_pair,
        "teacher_consensus_order": consensus_order,
        "teacher_weights": wT,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }


# ---------------------------------------------------------------------------
# One-epoch helpers
# ---------------------------------------------------------------------------

def _hard_pairwise_accuracy(
    scores: torch.Tensor,
    teacher_order: torch.Tensor,
) -> float:
    B, N = scores.shape
    ranks = torch.empty_like(teacher_order)
    positions = torch.arange(
        N, device=teacher_order.device, dtype=teacher_order.dtype,
    ).expand_as(teacher_order)
    ranks.scatter_(1, teacher_order, positions)
    target = ranks.unsqueeze(-1) < ranks.unsqueeze(-2)
    predicted = scores.unsqueeze(-1) > scores.unsqueeze(-2)
    mask = torch.triu(
        torch.ones(N, N, dtype=torch.bool, device=scores.device), diagonal=1,
    ).unsqueeze(0).expand(B, -1, -1)
    return float((predicted[mask] == target[mask]).float().mean().item())


def compute_primary_loss(
    scores: torch.Tensor,
    scores_per_head: torch.Tensor,
    teacher_order: torch.Tensor,
    teacher_pairwise: torch.Tensor,
    loss_type: str,
    rank_kl_temperature: float,
) -> tuple:
    """Return primary loss, per-head auxiliary loss, and optional soft acc."""
    if loss_type == "listmle":
        zero = scores.new_zeros(())
        return listmle_loss(scores, teacher_order), zero, None
    if loss_type == "rank_kl":
        zero = scores.new_zeros(())
        return (
            rank_kl_loss(
                scores, teacher_order, temperature=rank_kl_temperature,
            ),
            zero,
            None,
        )
    if loss_type == "pairwise_bce":
        if teacher_pairwise is None or teacher_pairwise.numel() == 0:
            raise ValueError("pairwise_bce requires teacher_pairwise")
        p_mask = torch.triu(
            torch.ones(
                scores.shape[1], scores.shape[1],
                dtype=torch.bool, device=scores.device,
            ),
            diagonal=1,
        )
        final = soft_pairwise_bce_loss(
            scores, teacher_pairwise, pair_mask=p_mask,
        )
        aux = per_head_aux_loss(scores_per_head, teacher_pairwise)
        return final, aux, pairwise_accuracy(scores, teacher_pairwise)
    raise ValueError(
        f"unknown loss_type {loss_type!r}; expected one of {LOSS_TYPES}"
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    lambda_aux: float,
    lambda_ent: float,
    min_gate_entropy: float,
    device: torch.device,
    loss_type: str = "pairwise_bce",
    rank_kl_temperature: float = 4.0,
) -> dict:
    """Run one epoch (train or val).  Returns aggregated metrics."""
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_final = 0.0
    total_aux = 0.0
    total_ent = 0.0
    total_acc = 0.0
    total_soft_acc = 0.0
    soft_acc_batches = 0
    total_entropy = 0.0
    n_batches = 0

    for batch in loader:
        B_raw = batch[0].to(device)
        teacher_order = batch[1].to(device)
        Y_pair = batch[2].to(device)

        if is_train:
            optimizer.zero_grad()

        scores, aux = model(B_raw)
        # (apply_head_dropout defaults to model.training → True in train)

        loss_final, loss_aux, soft_acc = compute_primary_loss(
            scores,
            aux["scores_per_head"],
            teacher_order,
            Y_pair,
            loss_type,
            rank_kl_temperature,
        )
        loss_ent = entropy_floor_loss(aux["alpha"], min_entropy=min_gate_entropy)
        loss = loss_final + lambda_aux * loss_aux + lambda_ent * loss_ent

        if is_train:
            loss.backward()
            optimizer.step()

        acc = _hard_pairwise_accuracy(scores, teacher_order)
        ent = gate_entropy(aux["alpha"]).mean().item()

        total_loss += loss.item()
        total_final += loss_final.item()
        total_aux += loss_aux.item()
        total_ent += loss_ent.item()
        total_acc += acc
        if soft_acc is not None:
            total_soft_acc += soft_acc
            soft_acc_batches += 1
        total_entropy += ent
        n_batches += 1

    metrics = {
        "loss": total_loss / n_batches,
        "loss_final": total_final / n_batches,
        "loss_aux": total_aux / n_batches,
        "loss_ent": total_ent / n_batches,
        "pairwise_acc": total_acc / n_batches,
        "gate_entropy_mean": total_entropy / n_batches,
    }
    if soft_acc_batches:
        metrics["soft_pairwise_acc"] = total_soft_acc / soft_acc_batches
    return metrics


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(
    dataset_path: str,
    out_dir: str,
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    lambda_aux: float = 0.05,
    lambda_ent: float = 0.001,
    min_gate_entropy: float = 1.5,
    seed: int = 0,
    device: str = "cuda:0",
    heads: int = 8,
    scorer_hidden: tuple = (256, 64),
    gate_hidden: int = 32,
    loss_type: str = "pairwise_bce",
    rank_kl_temperature: float = 4.0,
) -> dict:
    """Train L0DynamicGBeta to imitate the dynamic CDL consensus teacher.

    Checkpoint selection: the model with the lowest **validation pairwise
    BCE loss** (``val_loss_final``) is saved as ``g_beta_best.pt``.
    No physical metrics are used.

    Args:
        dataset_path: path to Task-3 .npz dataset.
        out_dir: directory for checkpoints and logs.
        epochs: number of full passes over the training set.
        batch_size: samples per batch.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        lambda_aux: weight on per-head auxiliary loss.
        lambda_ent: weight on entropy floor loss.
        min_gate_entropy: floor for gate entropy (nats).
        seed: random seed for reproducibility.
        device: torch device.
        heads: number of attention heads (default 8).
        scorer_hidden: hidden layer sizes for block scorer.
        gate_hidden: hidden size for gate encoder.

    Returns:
        dict with keys: best_epoch, best_val_loss_final, best_val_acc,
        metrics_history (list of per-epoch dicts), out_dir.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if loss_type not in LOSS_TYPES:
        raise ValueError(
            f"unknown loss_type {loss_type!r}; expected one of {LOSS_TYPES}"
        )
    if rank_kl_temperature <= 0.0:
        raise ValueError("rank_kl_temperature must be > 0")

    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        dev = torch.device("cpu")

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ──
    data = load_pretrain_dataset(
        dataset_path,
        device=str(dev),
        require_consensus_order=True,
        require_pairwise=loss_type == "pairwise_bce",
    )
    B_raw = data["B_raw"]
    Y_pair = data["teacher_pairwise"]
    teacher_order = data["teacher_consensus_order"]
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]

    # ── Build model ──
    model = L0DynamicGBeta(
        heads=heads, nodes=65,
        scorer_hidden=scorer_hidden, gate_hidden=gate_hidden,
    ).to(dev)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    # ── DataLoaders ──
    if Y_pair is None:
        Y_pair = torch.empty(
            (B_raw.shape[0], 0, 0), dtype=B_raw.dtype, device=B_raw.device,
        )
    train_ds = TensorDataset(
        B_raw[train_idx], teacher_order[train_idx], Y_pair[train_idx],
    )
    val_ds = TensorDataset(
        B_raw[val_idx], teacher_order[val_idx], Y_pair[val_idx],
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # ── Save config ──
    config = {
        "model_name": "l0_dynamic_gbeta_v0",
        "heads": heads,
        "nodes": 65,
        "head_identity": False,
        "scorer_hidden": list(scorer_hidden),
        "gate_hidden": gate_hidden,
        "loss_type": loss_type,
        "rank_kl_temperature": rank_kl_temperature,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "lambda_aux": lambda_aux,
        "lambda_ent": lambda_ent,
        "min_gate_entropy": min_gate_entropy,
        "seed": seed,
        "dataset_path": str(dataset_path),
        "dataset_meta": None,
        "selection_metric": "val_loss_final",
        "selection_label_free": True,
    }
    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── Training loop ──
    metrics_history = []
    best_val_loss_final = float("inf")
    best_epoch = -1
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        train_metrics = _run_epoch(
            model, train_loader, optimizer,
            lambda_aux, lambda_ent, min_gate_entropy, dev,
            loss_type, rank_kl_temperature,
        )

        # Val
        model.eval()
        val_metrics = _run_epoch(
            model, val_loader, None,
            lambda_aux, lambda_ent, min_gate_entropy, dev,
            loss_type, rank_kl_temperature,
        )

        # ── Per-epoch alpha diagnostics (val, no grad) ──
        with torch.no_grad():
            _, aux_val = model(B_raw[val_idx].to(dev), apply_head_dropout=False)
            alpha_val = aux_val["alpha"]  # (N_val, H)
            alpha_mean_h = alpha_val.mean(dim=0).cpu().tolist()
            alpha_std_h = alpha_val.std(dim=0).cpu().tolist()
            wT_val = data["teacher_weights"][val_idx].to(dev)
            # Pearson r between mean alpha and mean teacher weight per head
            alpha_bar = alpha_val.mean(dim=0).cpu()
            wT_bar = wT_val.mean(dim=0).cpu()
            v1 = alpha_bar - alpha_bar.mean()
            v2 = wT_bar - wT_bar.mean()
            denom = (v1.norm() * v2.norm()).clamp_min(1e-9)
            r_alpha_wT = float((v1 * v2).sum() / denom)
            alpha_diag = {
                "alpha_mean_per_head": alpha_mean_h,
                "alpha_std_per_head": alpha_std_h,
                "r_alpha_teacher": r_alpha_wT,
            }

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "alpha_diag": alpha_diag,
        }
        metrics_history.append(record)

        # ── Checkpoint selection (label-free) ──
        if val_metrics["loss_final"] < best_val_loss_final:
            best_val_loss_final = val_metrics["loss_final"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                out / "g_beta_best.pt",
            )

        # Log
        elapsed = time.time() - t_start
        print(
            f"[{epoch:3d}/{epochs}] "
            f"train: loss={train_metrics['loss']:.4f} "
            f"acc={train_metrics['pairwise_acc']:.3f} "
            f"H_α={train_metrics['gate_entropy_mean']:.2f} | "
            f"val: loss={val_metrics['loss']:.4f} "
            f"acc={val_metrics['pairwise_acc']:.3f} "
            f"H_α={val_metrics['gate_entropy_mean']:.2f} | "
            f"best_ep={best_epoch}  {elapsed:.0f}s",
            flush=True,
        )

    # ── Save last checkpoint and metrics ──
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": epochs,
            "val_metrics": metrics_history[-1]["val"],
        },
        out / "g_beta_last.pt",
    )

    with open(out / "metrics.jsonl", "w") as f:
        for rec in metrics_history:
            f.write(json.dumps(rec) + "\n")

    train_seconds = time.time() - t_start
    peak_cuda_memory_mb = (
        float(torch.cuda.max_memory_allocated(dev) / (1024 ** 2))
        if dev.type == "cuda" else 0.0
    )
    training_summary = {
        "best_epoch": best_epoch,
        "best_val_loss_final": best_val_loss_final,
        "best_val_acc": metrics_history[best_epoch - 1]["val"]["pairwise_acc"],
        "loss_type": loss_type,
        "rank_kl_temperature": rank_kl_temperature,
        "train_seconds": train_seconds,
        "peak_cuda_memory_mb": peak_cuda_memory_mb,
    }
    with open(out / "training_summary.json", "w") as f:
        json.dump(training_summary, f, indent=2)

    print(f"\nTraining complete.  best epoch={best_epoch}  "
          f"best val_loss_final={best_val_loss_final:.6f}  "
          f"best val_acc={metrics_history[best_epoch-1]['val']['pairwise_acc']:.4f}")

    return {
        "best_epoch": best_epoch,
        "best_val_loss_final": best_val_loss_final,
        "best_val_acc": metrics_history[best_epoch - 1]["val"]["pairwise_acc"],
        "metrics_history": metrics_history,
        "train_seconds": train_seconds,
        "peak_cuda_memory_mb": peak_cuda_memory_mb,
        "out_dir": str(out),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--lambda-aux", type=float, default=0.05)
    parser.add_argument("--lambda-ent", type=float, default=0.001)
    parser.add_argument("--min-gate-entropy", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--loss-type", choices=LOSS_TYPES, default="pairwise_bce")
    parser.add_argument("--rank-kl-temperature", type=float, default=4.0)
    args = parser.parse_args()
    train(
        dataset_path=args.dataset,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_aux=args.lambda_aux,
        lambda_ent=args.lambda_ent,
        min_gate_entropy=args.min_gate_entropy,
        seed=args.seed,
        device=args.device,
        heads=args.heads,
        loss_type=args.loss_type,
        rank_kl_temperature=args.rank_kl_temperature,
    )


if __name__ == "__main__":
    main()
