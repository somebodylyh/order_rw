"""Offline pretraining loop for L0DynamicGBeta (ported).

Label-free: checkpoint selection uses only the validation teacher-imitation loss
(pairwise BCE). Physical coordinates are never read/saved/used for selection.
listmle / rank_kl remain optional (lazy import); the canonical path is pairwise_bce.
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from orderhead_v3.l0_dynamic_gbeta import L0DynamicGBeta
from orderhead_v3.soft_pairwise import (
    entropy_floor_loss, gate_entropy, pairwise_accuracy,
    per_head_aux_loss, soft_pairwise_bce_loss,
)

LOSS_TYPES = ("listmle", "pairwise_bce", "rank_kl")


def load_pretrain_dataset(dataset_path, device="cpu", require_consensus_order=False,
                          require_pairwise=True):
    forbidden = {"block_perm", "inv_perm", "clean_perm", "physical_order", "l2r_order"}
    with np.load(dataset_path, allow_pickle=True) as z:
        leaked = forbidden & set(z.files)
        if leaked:
            raise RuntimeError(f"Dataset {dataset_path} contains forbidden fields: {sorted(leaked)}")
        B_raw = torch.from_numpy(z["B_raw"].copy()).float().to(device)
        if require_pairwise and "teacher_pairwise" not in z.files:
            raise KeyError("teacher_pairwise is required")
        Y_pair = (torch.from_numpy(z["teacher_pairwise"].copy()).float().to(device)
                  if "teacher_pairwise" in z.files else None)
        if require_consensus_order and "teacher_consensus_order" not in z.files:
            raise KeyError("teacher_consensus_order is required")
        consensus_order = (torch.from_numpy(z["teacher_consensus_order"].copy()).long().to(device)
                           if "teacher_consensus_order" in z.files else None)
        wT = torch.from_numpy(z["teacher_weights"].copy()).float().to(device)
        train_idx = z["train_idx"].copy()
        val_idx = z["val_idx"].copy()
        test_idx = z["test_idx"].copy()
    return {"B_raw": B_raw, "teacher_pairwise": Y_pair,
            "teacher_consensus_order": consensus_order, "teacher_weights": wT,
            "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx}


def _hard_pairwise_accuracy(scores, teacher_order):
    B, N = scores.shape
    ranks = torch.empty_like(teacher_order)
    positions = torch.arange(N, device=teacher_order.device, dtype=teacher_order.dtype).expand_as(teacher_order)
    ranks.scatter_(1, teacher_order, positions)
    target = ranks.unsqueeze(-1) < ranks.unsqueeze(-2)
    predicted = scores.unsqueeze(-1) > scores.unsqueeze(-2)
    mask = torch.triu(torch.ones(N, N, dtype=torch.bool, device=scores.device), diagonal=1).unsqueeze(0).expand(B, -1, -1)
    return float((predicted[mask] == target[mask]).float().mean().item())


def compute_primary_loss(scores, scores_per_head, teacher_order, teacher_pairwise,
                         loss_type, rank_kl_temperature):
    if loss_type == "listmle":
        from orderhead_v3.order_distillation_losses import listmle_loss  # lazy
        return listmle_loss(scores, teacher_order), scores.new_zeros(()), None
    if loss_type == "rank_kl":
        from orderhead_v3.order_distillation_losses import rank_kl_loss  # lazy
        return rank_kl_loss(scores, teacher_order, temperature=rank_kl_temperature), scores.new_zeros(()), None
    if loss_type == "pairwise_bce":
        if teacher_pairwise is None or teacher_pairwise.numel() == 0:
            raise ValueError("pairwise_bce requires teacher_pairwise")
        p_mask = torch.triu(torch.ones(scores.shape[1], scores.shape[1], dtype=torch.bool, device=scores.device), diagonal=1)
        final = soft_pairwise_bce_loss(scores, teacher_pairwise, pair_mask=p_mask)
        aux = per_head_aux_loss(scores_per_head, teacher_pairwise)
        return final, aux, pairwise_accuracy(scores, teacher_pairwise)
    raise ValueError(f"unknown loss_type {loss_type!r}; expected one of {LOSS_TYPES}")


def _run_epoch(model, loader, optimizer, lambda_aux, lambda_ent, min_gate_entropy,
               device, loss_type="pairwise_bce", rank_kl_temperature=4.0):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    tot = {"loss": 0.0, "final": 0.0, "aux": 0.0, "ent": 0.0, "acc": 0.0,
           "soft": 0.0, "soft_n": 0, "H": 0.0, "n": 0}
    for batch in loader:
        B_raw = batch[0].to(device)
        teacher_order = batch[1].to(device)
        Y_pair = batch[2].to(device)
        if is_train:
            optimizer.zero_grad()
        scores, aux = model(B_raw)
        loss_final, loss_aux, soft_acc = compute_primary_loss(
            scores, aux["scores_per_head"], teacher_order, Y_pair, loss_type, rank_kl_temperature)
        loss_ent = entropy_floor_loss(aux["alpha"], min_entropy=min_gate_entropy)
        loss = loss_final + lambda_aux * loss_aux + lambda_ent * loss_ent
        if is_train:
            loss.backward()
            optimizer.step()
        tot["loss"] += loss.item(); tot["final"] += loss_final.item()
        tot["aux"] += loss_aux.item(); tot["ent"] += loss_ent.item()
        tot["acc"] += _hard_pairwise_accuracy(scores, teacher_order)
        if soft_acc is not None:
            tot["soft"] += soft_acc; tot["soft_n"] += 1
        tot["H"] += gate_entropy(aux["alpha"]).mean().item(); tot["n"] += 1
    n = max(1, tot["n"])
    metrics = {"loss": tot["loss"] / n, "loss_final": tot["final"] / n,
               "loss_aux": tot["aux"] / n, "loss_ent": tot["ent"] / n,
               "pairwise_acc": tot["acc"] / n, "gate_entropy_mean": tot["H"] / n}
    if tot["soft_n"]:
        metrics["soft_pairwise_acc"] = tot["soft"] / tot["soft_n"]
    return metrics


def train(dataset_path, out_dir, epochs=40, batch_size=32, lr=3e-4, weight_decay=1e-2,
          lambda_aux=0.05, lambda_ent=0.001, min_gate_entropy=1.5, seed=0,
          device="cuda:0", heads=8, scorer_hidden=(256, 64), gate_hidden=32,
          loss_type="pairwise_bce", rank_kl_temperature=4.0):
    """Train L0DynamicGBeta to imitate the CDL teacher; select by val loss_final."""
    torch.manual_seed(seed); np.random.seed(seed)
    if loss_type not in LOSS_TYPES:
        raise ValueError(f"unknown loss_type {loss_type!r}")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        dev = torch.device("cpu")
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    data = load_pretrain_dataset(dataset_path, device=str(dev),
                                 require_consensus_order=True,
                                 require_pairwise=loss_type == "pairwise_bce")
    B_raw = data["B_raw"]; Y_pair = data["teacher_pairwise"]
    teacher_order = data["teacher_consensus_order"]
    train_idx, val_idx = data["train_idx"], data["val_idx"]

    model = L0DynamicGBeta(heads=heads, nodes=65, scorer_hidden=scorer_hidden,
                           gate_hidden=gate_hidden).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if Y_pair is None:
        Y_pair = torch.empty((B_raw.shape[0], 0, 0), dtype=B_raw.dtype, device=B_raw.device)
    train_ds = TensorDataset(B_raw[train_idx], teacher_order[train_idx], Y_pair[train_idx])
    val_ds = TensorDataset(B_raw[val_idx], teacher_order[val_idx], Y_pair[val_idx])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    config = {"model_name": "l0_dynamic_gbeta_v0", "heads": heads, "nodes": 65,
              "head_identity": False, "scorer_hidden": list(scorer_hidden),
              "gate_hidden": gate_hidden, "loss_type": loss_type,
              "seed": seed, "dataset_path": str(dataset_path),
              "selection_metric": "val_loss_final", "selection_label_free": True}
    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    best_val = float("inf"); best_epoch = -1; history = []
    for epoch in range(1, epochs + 1):
        tr = _run_epoch(model, train_loader, optimizer, lambda_aux, lambda_ent,
                        min_gate_entropy, dev, loss_type, rank_kl_temperature)
        va = _run_epoch(model, val_loader, None, lambda_aux, lambda_ent,
                        min_gate_entropy, dev, loss_type, rank_kl_temperature)
        history.append({"epoch": epoch, "train": tr, "val": va})
        if va["loss_final"] < best_val:
            best_val = va["loss_final"]; best_epoch = epoch
            torch.save({"model_state_dict": model.state_dict(), "config": config,
                        "epoch": epoch, "val_metrics": va}, out / "g_beta_best.pt")
        print(f"[{epoch:3d}/{epochs}] train loss={tr['loss']:.4f} acc={tr['pairwise_acc']:.3f} "
              f"| val loss={va['loss']:.4f} acc={va['pairwise_acc']:.3f} best_ep={best_epoch}",
              flush=True)

    with open(out / "metrics.jsonl", "w") as f:
        for rec in history:
            f.write(json.dumps(rec) + "\n")
    return {"best_epoch": best_epoch, "best_val_loss_final": best_val,
            "best_val_acc": history[best_epoch - 1]["val"]["pairwise_acc"],
            "out_dir": str(out)}
