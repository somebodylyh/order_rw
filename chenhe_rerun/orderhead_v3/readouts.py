"""Single-head gβ readouts (ported from batch_readout/model.py).

Input:  B  (batch, N, N)  — single-head content block graph (None stripped, N=64).
Output: scores (batch, N)  — per-block reveal-priority logits.

NodewiseReadout is the canonical deployed gβ (nodewise_K1000.pt:
model_name='nodewise', N=64, d_model=64, n_layers=2, n_heads=4). FlattenReadout
kept for parity with uniform_label_free_v1.py Stage B.
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
        if B.shape[-1] != self.N or B.shape[-2] != self.N:
            raise ValueError(f"expected B (..., {self.N}, {self.N}); got {tuple(B.shape)}")
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
        if B.shape[-1] != self.N or B.shape[-2] != self.N:
            raise ValueError(f"expected B (..., {self.N}, {self.N}); got {tuple(B.shape)}")
        x = torch.cat([B, B.transpose(-1, -2)], dim=-1)   # (batch, N, 2N)
        h = self.node_in(x)                                # (batch, N, d_model)
        h = self.mixer(h)                                  # (batch, N, d_model)
        return self.head(h).squeeze(-1)                    # (batch, N)


def build_readout(config: dict):
    """Build a readout from a saved gβ config dict."""
    name = config.get("model_name", "nodewise")
    N = int(config.get("N", 64))
    if name == "flatten":
        return FlattenReadout(N=N, hidden=tuple(config.get("hidden", (1024, 256))))
    if name == "nodewise":
        return NodewiseReadout(N=N, d_model=int(config.get("d_model", 64)),
                               n_layers=int(config.get("n_layers", 2)),
                               n_heads=int(config.get("n_heads", 4)))
    raise ValueError(f"unknown readout model_name={name!r}")
