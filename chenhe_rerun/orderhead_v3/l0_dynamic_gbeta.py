"""Label-free L0 dynamic g_beta: shared block scorer + shared dynamic gate (ported).

  - 4-channel normalisation: raw, prob (row-normalised), logz (whole-map z-score
    of log), rowz (row-wise z-score).
  - Shared block scorer: row+column features from all 4 channels → small MLP →
    per-block logit. Applied independently to each head.
  - Shared dynamic gate: pooled stats over the 3 non-raw channels → per-head α.
  - No head identity embedding → permutation equivariant across heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_strict65(B: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """4-channel normalisation for strict-65 block graphs.

    Channels: 0 raw, 1 prob (row-normalised), 2 logz (whole-map z of log),
    3 rowz (per-row z). Returns (batch, heads, 4, 65, 65).
    """
    raw = B.float()

    row_sum = raw.sum(dim=-1, keepdim=True)
    prob = raw / row_sum.clamp_min(eps)

    log_raw = torch.log(raw.clamp_min(eps))
    log_mean = log_raw.mean(dim=(-2, -1), keepdim=True)
    log_std = log_raw.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    logz = (log_raw - log_mean) / log_std

    row_mean = raw.mean(dim=-1, keepdim=True)
    row_std = raw.std(dim=-1, keepdim=True).clamp_min(1e-6)
    rowz = (raw - row_mean) / row_std

    return torch.stack([raw, prob, logz, rowz], dim=2)


def build_row_col_features(channels: torch.Tensor) -> torch.Tensor:
    """Per-content-block row+column features from all channels → (B, H, 64, 520)."""
    B, H, C, N65, _ = channels.shape
    Nc = N65 - 1

    content_rows = channels[:, :, :, 1:, :]
    content_cols = channels[:, :, :, :, 1:]
    content_cols_t = content_cols.transpose(-1, -2)

    feat = torch.cat([
        content_rows.reshape(B, H, Nc, C * 65),
        content_cols_t.reshape(B, H, Nc, C * 65),
    ], dim=-1)
    return feat


class SharedBlockScorer(nn.Module):
    """Small MLP scoring one content block from its 520 features (shared)."""

    def __init__(self, in_features: int = 520, hidden: tuple = (256, 64)):
        super().__init__()
        dims = [in_features, *hidden, 1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SharedDynamicGate(nn.Module):
    """Compact gate: pooled stats over 3 non-raw channels → per-head logit."""

    def __init__(self, n_channels: int = 3, hidden: int = 32):
        super().__init__()
        in_dim = n_channels * 4
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, gate_channels: torch.Tensor) -> torch.Tensor:
        B, H, C, _, _ = gate_channels.shape
        flat = gate_channels.reshape(B, H, C, -1)
        summary = torch.cat([
            flat.mean(dim=-1),
            flat.std(dim=-1),
            flat.max(dim=-1).values,
            flat.min(dim=-1).values,
        ], dim=-1)
        return self.encoder(summary).squeeze(-1)


class L0DynamicGBeta(nn.Module):
    """Label-free dynamic g_beta: shared scorer + shared gate, no head ID.

    Input:  B_raw  (batch, H=8, 65, 65).
    Output: scores (batch, 64), aux dict (scores_per_head, gate_logits, alpha, dropped_head).
    Head-permutation equivariant.
    """

    def __init__(self, heads: int = 8, nodes: int = 65,
                 scorer_hidden: tuple = (256, 64), gate_hidden: int = 32):
        super().__init__()
        self.H = heads
        self.N65 = nodes
        self.Nc = nodes - 1
        in_features = 4 * nodes * 2  # 520

        self.block_scorer = SharedBlockScorer(in_features, hidden=scorer_hidden)
        self.gate = SharedDynamicGate(n_channels=3, hidden=gate_hidden)

    def forward(self, B_raw: torch.Tensor,
                apply_head_dropout: bool | None = None) -> tuple[torch.Tensor, dict]:
        Bsz = B_raw.shape[0]
        H = B_raw.shape[1]
        if H != self.H:
            raise ValueError(f"expected H={self.H}, got {H}")

        channels = normalize_strict65(B_raw)

        block_features = build_row_col_features(channels)
        flat = block_features.reshape(Bsz * H * self.Nc, -1)
        scores_flat = self.block_scorer(flat)
        scores_per_head = scores_flat.reshape(Bsz, H, self.Nc)

        gate_channels = channels[:, :, 1:]
        gate_logits = self.gate(gate_channels)

        do_dropout = apply_head_dropout
        if do_dropout is None:
            do_dropout = self.training
        dropped_head = None
        if do_dropout:
            dropped_head = torch.randint(0, H, (Bsz,), device=gate_logits.device)
            masked_logits = gate_logits.clone()
            masked_logits[torch.arange(Bsz, device=gate_logits.device), dropped_head] = float("-inf")
            alpha = F.softmax(masked_logits, dim=1)
        else:
            alpha = F.softmax(gate_logits, dim=1)

        scores = (scores_per_head * alpha[:, :, None]).sum(dim=1)

        return scores, {
            "scores_per_head": scores_per_head,
            "gate_logits": gate_logits,
            "alpha": alpha,
            "dropped_head": dropped_head,
        }
