"""Label-free L0 dynamic g_beta: shared block scorer + shared dynamic gate.

Architecture (v0):
  - 4-channel normalisation: raw, prob (row-normalised), logz (whole-map
    z-score of log), rowz (row-wise z-score).
  - Shared block scorer: row+column features from all 4 channels → small MLP
    → per-block logit.  Applied independently to each head.
  - Shared dynamic gate: compact pooling (mean, std, max, min) over the
    3 non-raw channels → MLP → softmax → per-head weight α.
  - No head identity embedding.  Both scorer and gate share parameters
    across heads → permutation equivariant by construction.
  - Training-time head dropout: randomly masks one head per sample before
    softmax.  ``gate_logits`` in aux is always pre-dropout.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize_strict65(
    B: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """4-channel normalisation for strict-65 block graphs.

    Channels (in order):
      0: B_raw   — original values
      1: B_prob  — row-normalised  (row-sum → 1, or 0 for zero rows)
      2: B_logz  — whole-map z-score of log(B + eps), per sample & head
      3: B_rowz  — row-wise z-score, per sample, head, and row

    Args:
        B: (batch, heads, 65, 65) float tensor.
        eps: floor for divisions.

    Returns:
        channels: (batch, heads, 4, 65, 65) float tensor.
    """
    raw = B.float()  # (B, H, 65, 65)

    # ── prob: row-normalised ──
    row_sum = raw.sum(dim=-1, keepdim=True)  # (B, H, 65, 1)
    prob = raw / row_sum.clamp_min(eps)

    # ── logz: whole-map z-score of log ──
    log_raw = torch.log(raw.clamp_min(eps))
    log_mean = log_raw.mean(dim=(-2, -1), keepdim=True)
    log_std = log_raw.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    logz = (log_raw - log_mean) / log_std

    # ── rowz: per-row z-score ──
    row_mean = raw.mean(dim=-1, keepdim=True)
    row_std = raw.std(dim=-1, keepdim=True).clamp_min(1e-6)
    rowz = (raw - row_mean) / row_std

    return torch.stack([raw, prob, logz, rowz], dim=2)  # (B, H, 4, 65, 65)


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------

def build_row_col_features(channels: torch.Tensor) -> torch.Tensor:
    """Extract per-content-block row+column features from all channels.

    For content block i (node i+1 in the 65-node graph):
      - row  = channels[..., i+1, :]   — attention FROM this node
      - col  = channels[..., :, i+1]   — attention TO this node

    Args:
        channels: (B, H, C, 65, 65).  C = 4 channels.

    Returns:
        features: (B, H, 64, C * 65 * 2) = (B, H, 64, 520) for C=4.
    """
    B, H, C, N65, _ = channels.shape
    Nc = N65 - 1  # 64 content blocks

    # Rows for content blocks 1..64: (B, H, C, 64, 65)
    content_rows = channels[:, :, :, 1:, :]
    # Columns for content blocks 1..64: (B, H, C, 65, 64)
    content_cols = channels[:, :, :, :, 1:]
    # Transpose to (B, H, C, 64, 65) so row/col align on the last dim
    content_cols_t = content_cols.transpose(-1, -2)

    # Concatenate row and column features
    feat = torch.cat([
        content_rows.reshape(B, H, Nc, C * 65),
        content_cols_t.reshape(B, H, Nc, C * 65),
    ], dim=-1)  # (B, H, 64, 2*C*65) = (B, H, 64, 520)
    return feat


# ---------------------------------------------------------------------------
# Shared block scorer
# ---------------------------------------------------------------------------

class SharedBlockScorer(nn.Module):
    """Small MLP that scores a single content block from its 520 features.

    Shared across all heads and all blocks — the same weights process every
    (head, block) tuple independently.
    """

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


# ---------------------------------------------------------------------------
# Shared dynamic gate
# ---------------------------------------------------------------------------

class SharedDynamicGate(nn.Module):
    """Compact gate: pooled statistics over 3 non-raw channels → per-head logit.

    Pooling: mean, std, max, min over the flattened 65×65 entries, per channel.
    Total: 3 channels × 4 stats = 12 features per head → MLP → 1 logit.

    Does NOT receive B_raw — only prob, logz, rowz.
    """

    def __init__(self, n_channels: int = 3, hidden: int = 32):
        super().__init__()
        in_dim = n_channels * 4  # 3 channels × 4 stats = 12
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, gate_channels: torch.Tensor) -> torch.Tensor:
        """gate_channels: (B, H, 3, 65, 65) — prob, logz, rowz."""
        B, H, C, _, _ = gate_channels.shape
        # Flatten spatial dims: (B, H, C, 4225)
        flat = gate_channels.reshape(B, H, C, -1)
        # Pooling stats: (B, H, C) each
        summary = torch.cat([
            flat.mean(dim=-1),           # (B, H, C)
            flat.std(dim=-1),            # (B, H, C)
            flat.max(dim=-1).values,     # (B, H, C)
            flat.min(dim=-1).values,     # (B, H, C)
        ], dim=-1)  # (B, H, 4*C) = (B, H, 12)
        return self.encoder(summary).squeeze(-1)  # (B, H)


# ---------------------------------------------------------------------------
# L0DynamicGBeta
# ---------------------------------------------------------------------------

class L0DynamicGBeta(nn.Module):
    """Label-free dynamic g_beta: shared scorer + shared gate, no head ID.

    Input:  B_raw  (batch, H=8, 65, 65) — Task 1 / Task 3 output.
    Output: scores (batch, 64) — per-block reveal-priority logits.
            aux dict with per-head scores, alpha, gate logits, dropped head.

    Head-permutation equivariant: if the head dimension of B_raw is permuted,
    ``scores`` is unchanged and ``alpha`` is identically permuted.
    """

    def __init__(
        self,
        heads: int = 8,
        nodes: int = 65,
        scorer_hidden: tuple = (256, 64),
        gate_hidden: int = 32,
    ):
        super().__init__()
        self.H = heads
        self.N65 = nodes
        self.Nc = nodes - 1  # 64 content blocks
        in_features = 4 * nodes * 2  # 4 channels × (row+col) = 520

        self.block_scorer = SharedBlockScorer(in_features, hidden=scorer_hidden)
        self.gate = SharedDynamicGate(n_channels=3, hidden=gate_hidden)

    def forward(
        self,
        B_raw: torch.Tensor,
        apply_head_dropout: bool | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Forward pass.

        Args:
            B_raw: (B, H, 65, 65) strict-65 block graphs.
            apply_head_dropout: if None, dropout is active iff model is in
                training mode.  Set explicitly for eval-time testing.

        Returns:
            scores: (B, 64) per-block reveal-priority logits.
            aux: dict with:
                scores_per_head  (B, H, 64)
                gate_logits      (B, H) — pre-dropout
                alpha            (B, H) — post-dropout, sum to 1
                dropped_head     (B,) or None
        """
        Bsz = B_raw.shape[0]
        H = B_raw.shape[1]
        if H != self.H:
            raise ValueError(f"expected H={self.H}, got {H}")

        # ── Normalise ──
        channels = normalize_strict65(B_raw)  # (B, H, 4, 65, 65)

        # ── Block scorer ──
        block_features = build_row_col_features(channels)  # (B, H, 64, 520)
        flat = block_features.reshape(Bsz * H * self.Nc, -1)
        scores_flat = self.block_scorer(flat)  # (B*H*64, 1)
        scores_per_head = scores_flat.reshape(Bsz, H, self.Nc)  # (B, H, 64)

        # ── Gate (no raw channel) ──
        gate_channels = channels[:, :, 1:]  # (B, H, 3, 65, 65)
        gate_logits = self.gate(gate_channels)  # (B, H)

        # ── Head dropout ──
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

        # ── Weighted sum ──
        scores = (scores_per_head * alpha[:, :, None]).sum(dim=1)  # (B, 64)

        return scores, {
            "scores_per_head": scores_per_head,
            "gate_logits": gate_logits,    # pre-dropout
            "alpha": alpha,
            "dropped_head": dropped_head,
        }
