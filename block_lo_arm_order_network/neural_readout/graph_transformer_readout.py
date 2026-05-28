"""NR-1 Task 6: g_beta -- Graph Transformer over per-node attention features.

Per-node token x_v = concat(B[v, :], B[:, v]) in R^{2N}, projected to d_model,
processed by a standard TransformerEncoder with NO positional embedding,
then read off to a scalar score s_v per node.

Honest framing of equivariance (NR-1 spec §3):
  Even without a node positional embedding, the per-node feature
  x_v = [B[v, :], B[:, v]] carries the row / column coordinates of B,
  and those coordinates encode node identity through the fixed graph
  coordinate system. This is a low-extra-position-bias readout, NOT a
  permutation-invariant architecture. Making it strictly equivariant
  would require edge-list / message-passing / edge-bias-attention, which
  NR-1 deliberately avoids.

NR-1 supervision boundary: this module never sees NLL or any reward signal.
"""
import torch
import torch.nn as nn


class GraphTransformerReadout(nn.Module):
    """Input  B: (batch, N, N) attention graph.
    Output s: (batch, N) per-node score, score convention earliest = highest.
    """

    def __init__(
        self,
        N=64,
        d_model=64,
        n_heads=4,
        n_layers=2,
        ffn_mult=4,
        dropout=0.0,
    ):
        super().__init__()
        self.N = N
        self.in_proj = nn.Linear(2 * N, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_mult * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, B):
        """B: (batch, N, N) -> s: (batch, N)."""
        if B.dim() != 3 or B.shape[-1] != self.N or B.shape[-2] != self.N:
            raise ValueError(
                f"expected (batch, {self.N}, {self.N}), got {tuple(B.shape)}"
            )
        rows = B                       # (b, N, N)
        cols = B.transpose(-1, -2)     # (b, N, N)
        x = torch.cat([rows, cols], dim=-1)  # (b, N, 2N)
        h = self.in_proj(x)                  # (b, N, d_model)
        h = self.encoder(h)                  # (b, N, d_model)
        s = self.head(h).squeeze(-1)         # (b, N)
        return s
