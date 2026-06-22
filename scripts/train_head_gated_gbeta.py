#!/usr/bin/env python3
"""Offline distillation training for head-gated g_beta variants.

Loads a multi-head .npz dataset (from build_gbeta_headset_dataset.py),
trains one head-gated g_beta variant against teacher labels, logs
tau / gate-entropy / top-heads metrics, and saves the best checkpoint.

Usage (CLI):
  python3 scripts/train_head_gated_gbeta.py \
    --dataset headset_l0_manual_l0h2_cd_ckpt_step20000.npz \
    --variant head_gated --gate-mode soft_all \
    --epochs 40 --out-dir logs/run_001
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Optional

import numpy as np
import torch
from scipy.stats import kendalltau

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "block_lo_arm_order_network"))

from batch_readout.head_gated_gbeta import build_head_gated_gbeta  # noqa: E402
from batch_readout.pl_sampling import pl_argsort  # noqa: E402


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _tau(a: np.ndarray, b: np.ndarray) -> float:
    val, _ = kendalltau(np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64))
    return float(val) if not np.isnan(val) else 0.0


def _gate_entropy(gate_weights: torch.Tensor) -> float:
    """Mean entropy of gate distribution across batch (nats)."""
    w = gate_weights.detach().cpu()  # (B, H)
    eps = 1e-9
    ent = -(w * torch.log(w + eps)).sum(dim=1).mean()
    return float(ent.item())


def _top_heads(gate_weights: torch.Tensor, k: int = 3) -> list:
    """Sorted list of top-k head indices by mean weight across batch."""
    mean_w = gate_weights.detach().cpu().mean(dim=0)
    order = torch.argsort(mean_w, descending=True)
    return order[:k].tolist()


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def pairwise_logistic_loss(logits: torch.Tensor, rank: torch.Tensor) -> torch.Tensor:
    """Mean over (i,j) of -log sigmoid(z_i - z_j) for rank_i < rank_j."""
    zi = logits.unsqueeze(2)
    zj = logits.unsqueeze(1)
    mask = (rank.unsqueeze(2) < rank.unsqueeze(1)).float()
    diff = zi - zj
    loss_mat = -torch.nn.functional.logsigmoid(diff) * mask
    denom = mask.sum().clamp_min(1.0)
    return loss_mat.sum() / denom


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_head_gated_gbeta(
    dataset_path: str,
    variant: str,
    N: int = 64,
    H: int = 16,
    gate_mode: str = "soft_all",
    topk: int = 2,
    head_idx: int = 2,
    batch_size: int = 64,
    lr: float = 3e-4,
    epochs: int = 40,
    seed: int = 0,
    out_dir: Optional[str] = None,
    device: str = "cuda:0",
    val_frac: float = 0.2,
):
    """Train a head-gated g_beta variant on a multi-head dataset.

    Returns dict with history, best_epoch, best_metrics.
    """
    torch.manual_seed(seed)

    # --- Load dataset ---
    ds = np.load(dataset_path)
    B_heads = ds["B_heads"].astype(np.float32)  # (M, H, N, N)
    teacher = ds["teacher_order_model"].astype(np.int64)  # (M, N)
    block_perm = ds["block_perm_phys_to_model"].astype(np.int64)  # (N,)
    semantic_path = block_perm[np.arange(N, dtype=np.int64)]  # (N,) model-frame L2R order
    identity_order = np.arange(N, dtype=np.int64)
    ds_H = int(ds["num_heads"])

    if H is None:
        H = ds_H
    elif H != ds_H:
        print(f"Warning: specified H={H} but dataset has H={ds_H}; using {ds_H}")
        H = ds_H

    M = len(B_heads)

    # --- Train/val split ---
    rng = np.random.default_rng(seed)
    perm = rng.permutation(M)
    n_val = max(1, int(M * val_frac))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    B_tr = B_heads[train_idx]
    T_tr = teacher[train_idx]
    B_val = B_heads[val_idx]
    T_val = teacher[val_idx]

    # Convert rank = inverse permutation of teacher order
    def _to_rank(sigma):
        r = np.empty((sigma.shape[0], sigma.shape[1]), dtype=np.int64)
        for i in range(sigma.shape[0]):
            r[i, sigma[i]] = np.arange(sigma.shape[1], dtype=np.int64)
        return r

    rank_tr = _to_rank(T_tr)
    rank_val = _to_rank(T_val)

    # --- Device ---
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    # --- Build model ---
    model = build_head_gated_gbeta(
        variant, N=N, H=H,
        head_idx=head_idx,
        gate_mode=gate_mode, topk=topk,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    # --- Output dir ---
    out_dir_p = pathlib.Path(out_dir) if out_dir else None
    if out_dir_p is not None:
        out_dir_p.mkdir(parents=True, exist_ok=True)

    best = {"pairwise_acc": -1.0, "epoch": -1, "metrics": None}
    history = []
    rng_epoch = np.random.default_rng(seed + 1)

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        epoch_perm = rng_epoch.permutation(len(B_tr))
        losses = []
        for start in range(0, len(B_tr), batch_size):
            idx = epoch_perm[start:start + batch_size]
            Bb = torch.from_numpy(B_tr[idx]).float().to(device)
            Rb = torch.from_numpy(rank_tr[idx]).to(device)
            z, aux = model(Bb)
            loss = pairwise_logistic_loss(z, Rb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        train_loss = float(np.mean(losses))

        # --- Eval ---
        model.eval()
        with torch.no_grad():
            Bb_val = torch.from_numpy(B_val).float().to(device)
            z_val, aux_val = model(Bb_val)
            val_loss = float(pairwise_logistic_loss(
                z_val, torch.from_numpy(rank_val).to(device)
            ).item())

            sigma_pred = pl_argsort(z_val.cpu()).numpy()  # (n_val, N)

        # Per-sample tau metrics
        taus_vs_teacher = []
        taus_vs_semantic = []
        taus_vs_identity = []
        for i in range(len(sigma_pred)):
            taus_vs_teacher.append(_tau(sigma_pred[i], T_val[i]))
            taus_vs_semantic.append(_tau(sigma_pred[i], semantic_path))
            taus_vs_identity.append(_tau(sigma_pred[i], identity_order))

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "tau_model_vs_teacher": float(np.mean(taus_vs_teacher)),
            "tau_model_vs_semantic_path": float(np.mean(taus_vs_semantic)),
            "tau_model_vs_identity": float(np.mean(taus_vs_identity)),
            "gate_entropy": _gate_entropy(aux_val["gate_weights"]),
            "top_heads": _top_heads(aux_val["gate_weights"]),
        }
        history.append(metrics)

        if metrics["tau_model_vs_teacher"] > best.get("tau_model_vs_teacher", -1.0):
            best = {
                "pairwise_acc": -1.0,  # placeholder
                "tau_model_vs_teacher": metrics["tau_model_vs_teacher"],
                "epoch": epoch,
                "metrics": metrics,
            }
            if out_dir_p is not None:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": {
                            "variant": variant,
                            "N": N, "H": H,
                            "gate_mode": gate_mode,
                            "topk": topk,
                            "head_idx": head_idx,
                        },
                        "epoch": epoch,
                        "metrics": metrics,
                    },
                    out_dir_p / "g_beta_best.pt",
                )

    # --- Save history ---
    if out_dir_p is not None:
        (out_dir_p / "history.json").write_text(
            json.dumps({"history": history, "best": best}, indent=2, default=str)
        )
        # Also save as TSV for easy analysis
        tsv_path = out_dir_p / "metrics.tsv"
        keys = ["epoch", "train_loss", "val_loss",
                "tau_model_vs_teacher", "tau_model_vs_semantic_path",
                "tau_model_vs_identity", "gate_entropy"]
        with open(tsv_path, "w") as f:
            f.write("\t".join(keys) + "\n")
            for h in history:
                row = [str(h.get(k, "")) for k in keys]
                f.write("\t".join(row) + "\n")

    return {
        "history": history,
        "best_epoch": best["epoch"],
        "best_metrics": best["metrics"],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Train head-gated g_beta variants offline"
    )
    p.add_argument("--dataset", required=True, help="Path to headset .npz file")
    p.add_argument("--variant", required=True,
                   choices=["single_head", "mean_head", "head_gated"],
                   help="Model variant")
    p.add_argument("--gate-mode", default="soft_all", choices=["soft_all", "topk"],
                   help="Gate mode (head_gated only)")
    p.add_argument("--topk", type=int, default=2, help="Top-k for gate (head_gated only)")
    p.add_argument("--head-idx", type=int, default=2,
                   help="Head index for single_head variant")
    p.add_argument("--N", type=int, default=64, help="Number of blocks")
    p.add_argument("--H", type=int, default=None,
                   help="Number of heads (auto-detected from dataset)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True, help="Output directory for logs/model")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    result = train_head_gated_gbeta(
        dataset_path=args.dataset,
        variant=args.variant,
        N=args.N, H=args.H,
        gate_mode=args.gate_mode,
        topk=args.topk,
        head_idx=args.head_idx,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
        out_dir=args.out_dir,
        device=args.device,
    )

    print(f"\nBest epoch: {result['best_epoch']}")
    print(f"Best metrics: {json.dumps(result['best_metrics'], indent=2, default=str)}")


if __name__ == "__main__":
    main()
