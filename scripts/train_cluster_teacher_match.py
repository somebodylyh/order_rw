#!/usr/bin/env python3
"""Train a minimal teacher-match pairwise readout against a clustered soft teacher.

Consumes a teacher .npz (from build_topk_clustered_pairwise_teacher.py),
trains a ClusterTeacherReadout that gates across K heads and scores per-block
pairwise ordering, then reports metrics (val_bce, pairwise acc, gate stats).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ClusterTeacherReadout(nn.Module):
    """Gated multi-head readout:  alpha = softmax(gate(flatten(B_heads)))
    scores = sum_i alpha_i * scorer_i(B_i)."""

    def __init__(self, K: int, N: int = 64, hidden: int = 256):
        super().__init__()
        self.K = K
        self.N = N
        flat_dim = K * 65 * 65
        self.gate = nn.Sequential(
            nn.Linear(flat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, K),
        )
        self.scorer = nn.Sequential(
            nn.Linear(65 * 65, hidden),
            nn.GELU(),
            nn.Linear(hidden, N),
        )

    def forward(self, B_heads: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """B_heads: (batch, K, 65, 65) → scores: (batch, N), meta dict."""
        batch = B_heads.shape[0]
        flat_all = B_heads.reshape(batch, -1)  # (batch, K*65*65)
        gate_logits = self.gate(flat_all)  # (batch, K)
        alpha = F.softmax(gate_logits, dim=-1)  # (batch, K)

        # per-head scores
        B_flat_heads = B_heads.reshape(batch, self.K, 65 * 65)  # (batch, K, 4225)
        per_head = self.scorer(B_flat_heads)  # (batch, K, N)
        scores = (alpha.unsqueeze(-1) * per_head).sum(dim=1)  # (batch, N)

        return scores, {
            "alpha": alpha.detach(),
            "per_head_scores": per_head.detach(),
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_teacher_match(
    B_heads: np.ndarray,
    Y_pair: np.ndarray,
    confidence: np.ndarray,
    *,
    epochs: int = 200,
    hidden: int = 256,
    lr: float = 3e-4,
    seed: int = 0,
    device: str = "cpu",
) -> dict:
    """Train a single-sample teacher-match model and return metrics.

    Args:
        B_heads: (K, 65, 65) float32 — one per candidate head.
        Y_pair: (N, N) float32 — soft pairwise teacher.
        confidence: (N, N) float32 — confidence weights.
        epochs, hidden, lr, seed: training hyper-parameters.
        device: "cpu" or "cuda".

    Returns:
        dict with val_bce, val_pairwise_acc, gate_mean, gate_entropy, ...
    """
    K, D, _ = B_heads.shape
    N_from_B = D - 1  # 65 × 65 B ⇒ N=64 blocks + None node
    N = Y_pair.shape[0]
    if N != N_from_B:
        raise ValueError(f"Y_pair N={N} does not match B_heads N_from_B={N_from_B} (B shape {B_heads.shape})")
    if Y_pair.shape != (N, N):
        raise ValueError(f"Y_pair shape {Y_pair.shape} != ({N}, {N})")
    if confidence.shape != (N, N):
        raise ValueError(f"confidence shape {confidence.shape} != ({N}, {N})")

    torch.manual_seed(seed)
    np.random.seed(seed)

    B_t = torch.from_numpy(np.asarray(B_heads, dtype="float32")).unsqueeze(0).to(device)  # (1, K, 65, 65)
    Y_t = torch.from_numpy(np.asarray(Y_pair, dtype="float32")).to(device)
    C_t = torch.from_numpy(np.asarray(confidence, dtype="float32")).to(device)

    model = ClusterTeacherReadout(K=K, N=N, hidden=hidden).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    # Pre-compute mask (skip diagonal)
    diag_mask = ~torch.eye(N, dtype=torch.bool, device=device)

    best_bce = float("inf")
    best_acc = 0.0
    best_gate_mean = None
    best_gate_entropy = None

    for _epoch in range(epochs):
        model.train()
        optimiser.zero_grad()
        scores, meta = model(B_t)  # (1, N)
        diff = scores[:, :, None] - scores[:, None, :]  # (1, N, N) — logit for a < b
        diff_masked = diff[0][diag_mask]
        y_masked = Y_t[diag_mask]
        c_masked = C_t[diag_mask]

        loss = F.binary_cross_entropy_with_logits(
            diff_masked, y_masked, weight=c_masked
        )
        loss.backward()
        optimiser.step()

        # Track best
        with torch.no_grad():
            model.eval()
            scores_eval, meta_eval = model(B_t)
            diff_eval = scores_eval[:, :, None] - scores_eval[:, None, :]  # (1, N, N)
            pred = (diff_eval[0] > 0).float()
            correct = (pred[diag_mask] == Y_t[diag_mask]).float()
            acc = correct.mean().item()
            bce = loss.item()
            if bce < best_bce:
                best_bce = bce
                best_acc = acc
                best_gate_mean = meta_eval["alpha"][0].detach().cpu().numpy().copy()
                best_gate_entropy = float(
                    -(meta_eval["alpha"][0] * torch.log(meta_eval["alpha"][0] + 1e-8))
                    .sum()
                    .item()
                )

    # Confidence coverage at thresholds
    coverage = {}
    for thresh in [0.05, 0.10, 0.20]:
        cov_mask = C_t[diag_mask] > thresh
        coverage[f"cov_{thresh}"] = float(cov_mask.float().mean().item())

    return {
        "val_bce": float(best_bce),
        "val_pairwise_acc": float(best_acc),
        "gate_mean": best_gate_mean.tolist() if best_gate_mean is not None else None,
        "gate_entropy": best_gate_entropy,
        "confidence_coverage": coverage,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher-npz", required=True, help="Path to teacher_*.npz")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    data = np.load(args.teacher_npz)
    B_heads = data["B_heads"]  # (K, 65, 65)
    Y_pair = data["Y_pair"]
    confidence = data["confidence"]

    result = train_teacher_match(
        B_heads=B_heads,
        Y_pair=Y_pair,
        confidence=confidence,
        epochs=args.epochs,
        hidden=args.hidden,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "match_result.json").write_text(json.dumps(result, indent=2))
    print(f"val_bce={result['val_bce']:.4f}  acc={result['val_pairwise_acc']:.4f}  "
          f"cov_0.1={result['confidence_coverage'].get('cov_0.10', 'N/A')}  "
          f"→ {out}")


if __name__ == "__main__":
    main()
