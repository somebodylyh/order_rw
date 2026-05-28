"""BR-1 Task 7: Plackett-Luce sampling and argsort.

sigma ~ PL(z/tau) without replacement is equivalent to argsort(-(z/tau + Gumbel))
where Gumbel = -log(-log U), U ~ Uniform(0, 1). This is the Gumbel-top-k trick;
see Kool et al. 2019 "Stochastic Beams and Where to Find Them".

The sampler operates on CPU tensors so an explicit torch.Generator can drive
the noise without colliding with the global CUDA stream. Callers should
.cpu() their logits before passing in.
"""
from __future__ import annotations

import torch


def pl_argsort(z: torch.Tensor) -> torch.Tensor:
    """Deterministic order: earliest reveal = argmax z.
    z: (B, N) -> sigma: (B, N) int64."""
    return torch.argsort(-z, dim=-1).to(torch.int64)


def pl_sample(z: torch.Tensor, tau: float = 1.0, generator: "torch.Generator | None" = None) -> torch.Tensor:
    """Sample sigma ~ Plackett-Luce(z/tau) without replacement.
    z: (B, N) on CPU -> sigma: (B, N) int64."""
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    if z.is_cuda:
        raise ValueError(
            "pl_sample expects z on CPU; the Generator path does not cross "
            "device. Call z.cpu() first."
        )
    u = torch.rand(z.shape, generator=generator).clamp_min(1e-20)
    gumbel = -torch.log(-torch.log(u))
    scores = z / tau + gumbel
    return torch.argsort(-scores, dim=-1).to(torch.int64)
