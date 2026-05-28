"""BR-1 Task 5: g_beta readout models.

FlattenReadout (v1):
    vec(B) -> MLP -> z. Permutation-sensitive — encodes slot identity
    directly through the MLP weights. Quick to train, useful as baseline.

NodewiseReadout (v2):
    per-node features x_v = [B[v, :], B[:, v]] -> shared MLP -> 2L
    TransformerEncoder mixer -> per-node scalar head. No LEARNED positional
    embedding is added (the slot identity already lives in the row/column
    ordering of the per-node feature; adding a PE on top would be
    redundant). The mixer itself is permutation-equivariant over its N
    token positions, so all positional information flows through the
    per-node feature.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FlattenReadout(nn.Module):
    def __init__(self, N: int = 64, hidden=(1024, 256)):
        super().__init__()
        self.N = N
        dims = [N * N, *hidden, N]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, B: torch.Tensor) -> torch.Tensor:
        # B: (batch, N, N) -> (batch, N)
        if B.shape[-1] != self.N or B.shape[-2] != self.N:
            raise ValueError(f"expected B with last two dims ({self.N}, {self.N}); got {tuple(B.shape)}")
        return self.net(B.reshape(B.shape[0], -1))


class NodewiseReadout(nn.Module):
    def __init__(self, N: int = 64, d_model: int = 64, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.N = N
        self.d_model = d_model
        self.node_in = nn.Linear(2 * N, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.mixer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, B: torch.Tensor) -> torch.Tensor:
        # B: (batch, N, N). per-node feature = [B[v, :], B[:, v]] in R^{2N}.
        if B.shape[-1] != self.N or B.shape[-2] != self.N:
            raise ValueError(f"expected B with last two dims ({self.N}, {self.N}); got {tuple(B.shape)}")
        x = torch.cat([B, B.transpose(-1, -2)], dim=-1)  # (batch, N, 2N)
        h = self.node_in(x)                               # (batch, N, d_model)
        h = self.mixer(h)                                 # (batch, N, d_model)
        return self.head(h).squeeze(-1)                   # (batch, N)
