"""Head-gated g_beta readout models.

Three variants sharing one interface::

    scores, aux = model(B_heads)

where:
  - B_heads:  (batch, H, N, N)  — one block-graph per head, model frame
  - scores:   (batch, N)         — per-block reveal-priority logits
  - aux:      dict with 'gate_weights': (batch, H)

Variants:
  - SingleHeadGBeta:   select one fixed head
  - MeanHeadGBeta:     average all heads (expected-weak baseline)
  - HeadGatedGBeta:    learned gate over heads (soft_all or topk mode)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Backbone (same as existing FlattenReadout)
# ---------------------------------------------------------------------------

def _make_readout(N: int, hidden: tuple = (1024, 256)) -> nn.Sequential:
    dims = [N * N, *hidden, N]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Single-Head
# ---------------------------------------------------------------------------

class SingleHeadGBeta(nn.Module):
    """Select one fixed head, apply FlattenReadout."""

    def __init__(self, N: int = 64, H: int = 16, head_idx: int = 0,
                 hidden: tuple = (1024, 256)):
        super().__init__()
        self.N = N
        self.H = H
        self.head_idx = head_idx
        self.readout = _make_readout(N, hidden)

    def forward(self, B_heads: torch.Tensor) -> tuple[torch.Tensor, dict]:
        # B_heads: (B, H, N, N)
        B = B_heads[:, self.head_idx, :, :]  # (B, N, N)
        scores = self.readout(B.reshape(B.shape[0], -1))
        gate = torch.zeros(B_heads.shape[0], self.H, device=B_heads.device)
        gate[:, self.head_idx] = 1.0
        return scores, {"gate_weights": gate}


# ---------------------------------------------------------------------------
# Mean-Head (expected-weak baseline)
# ---------------------------------------------------------------------------

class MeanHeadGBeta(nn.Module):
    """Average all heads, apply FlattenReadout."""

    def __init__(self, N: int = 64, H: int = 16, hidden: tuple = (1024, 256)):
        super().__init__()
        self.N = N
        self.H = H
        self.readout = _make_readout(N, hidden)

    def forward(self, B_heads: torch.Tensor) -> tuple[torch.Tensor, dict]:
        # B_heads: (B, H, N, N)
        B_mix = B_heads.mean(dim=1)  # (B, N, N)
        scores = self.readout(B_mix.reshape(B_mix.shape[0], -1))
        gate = torch.full((B_heads.shape[0], self.H), 1.0 / self.H,
                          device=B_heads.device)
        return scores, {"gate_weights": gate}


# ---------------------------------------------------------------------------
# Head-Gated (sparse / soft attention)
# ---------------------------------------------------------------------------

class HeadGatedGBeta(nn.Module):
    """Learned gate over heads — soft_all or topk mode.

    Gate:  B_h (N,N) -> flatten -> small MLP -> scalar logit per head.
    Mix:   B_mix = sum_h gate_weights_h * B_h.
    Read:  FlattenReadout on B_mix.
    """

    def __init__(self, N: int = 64, H: int = 16,
                 readout_hidden: tuple = (1024, 256),
                 gate_hidden: tuple = (256,),
                 gate_mode: str = "soft_all",
                 topk: int = 2,
                 temperature: float = 1.0):
        super().__init__()
        if gate_mode not in ("soft_all", "topk"):
            raise ValueError(f"gate_mode must be 'soft_all' or 'topk', got {gate_mode!r}")

        self.N = N
        self.H = H
        self.gate_mode = gate_mode
        self.topk = topk
        self.temperature = temperature

        # Per-head feature extractor: flatten B_h -> MLP -> scalar
        in_dim = N * N
        gate_layers = []
        for hdim in gate_hidden:
            gate_layers.append(nn.Linear(in_dim, hdim))
            gate_layers.append(nn.GELU())
            in_dim = hdim
        gate_layers.append(nn.Linear(in_dim, 1))
        self.gate_net = nn.Sequential(*gate_layers)

        # Readout backbone
        self.readout = _make_readout(N, readout_hidden)

    def forward(self, B_heads: torch.Tensor) -> tuple[torch.Tensor, dict]:
        # B_heads: (B, H, N, N)
        B_sz, H, N, _ = B_heads.shape
        if H != self.H:
            raise ValueError(f"expected H={self.H}, got B_heads with H={H}")
        if N != self.N:
            raise ValueError(f"expected N={self.N}, got B_heads with N={N}")

        # Gate logits from per-head flattened B
        B_flat = B_heads.reshape(B_sz * H, N * N)  # (B*H, N*N)
        gate_logits = self.gate_net(B_flat).reshape(B_sz, H)  # (B, H)

        if self.gate_mode == "soft_all":
            gate_weights = F.softmax(gate_logits / self.temperature, dim=1)
        else:  # topk
            k = min(self.topk, H)
            _, top_idx = gate_logits.topk(k, dim=1)
            mask = torch.zeros_like(gate_logits)
            mask.scatter_(1, top_idx, 1.0)
            masked = gate_logits.masked_fill(mask == 0, float("-inf"))
            gate_weights = F.softmax(masked, dim=1)

        # Weighted mix of heads
        B_mix = (B_heads * gate_weights[:, :, None, None]).sum(dim=1)  # (B, N, N)

        # Readout
        scores = self.readout(B_mix.reshape(B_sz, -1))  # (B, N)

        aux = {"gate_weights": gate_weights}
        return scores, aux


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_head_gated_gbeta(variant: str, N: int = 64, H: int = 16,
                           **kwargs) -> nn.Module:
    """Build a head-gated g_beta variant by name.

    Args:
        variant: 'single_head', 'mean_head', or 'head_gated'.
        N: number of blocks (64).
        H: number of heads in the layer.
        **kwargs: passed to the model constructor (head_idx, gate_mode, topk, etc.)

    Returns:
        nn.Module with (B_heads) -> (scores, aux) interface.
    """
    if variant == "single_head":
        return SingleHeadGBeta(N=N, H=H, head_idx=kwargs.get("head_idx", 0),
                               hidden=kwargs.get("readout_hidden", (1024, 256)))
    if variant == "mean_head":
        return MeanHeadGBeta(N=N, H=H,
                             hidden=kwargs.get("readout_hidden", (1024, 256)))
    if variant == "head_gated":
        return HeadGatedGBeta(
            N=N, H=H,
            readout_hidden=kwargs.get("readout_hidden", (1024, 256)),
            gate_hidden=kwargs.get("gate_hidden", (256,)),
            gate_mode=kwargs.get("gate_mode", "soft_all"),
            topk=kwargs.get("topk", 2),
            temperature=kwargs.get("temperature", 1.0),
        )
    raise ValueError(f"unknown variant {variant!r}; choose from "
                     f"{{'single_head', 'mean_head', 'head_gated'}}")
