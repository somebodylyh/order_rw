"""Pure group-credit assembly helpers for the V3 OrderHead experiments."""

from numbers import Integral

import numpy as np
import torch


def per_sample_loss(token_losses):
    """Reduce ``(batch, tokens)`` token losses to one mean per sample."""
    if not isinstance(token_losses, torch.Tensor) or token_losses.ndim != 2:
        raise ValueError("token_losses must be a rank-2 torch.Tensor with shape (b, t)")
    return token_losses.mean(dim=1)


def group_ids_for(batch_size, m):
    """Return contiguous NumPy index arrays for groups of exactly ``m`` samples."""
    if not isinstance(batch_size, Integral) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
    if not isinstance(m, Integral) or isinstance(m, bool) or m <= 0:
        raise ValueError(f"m must be a positive integer, got {m!r}")
    if batch_size % m != 0:
        raise ValueError(f"batch_size {batch_size} must be divisible by m {m}")
    return [np.arange(start, start + m) for start in range(0, batch_size, m)]


def group_rewards(per_sample_ell, groups):
    """Reduce ``(batch,)`` sample losses to group means on the input device."""
    if not isinstance(per_sample_ell, torch.Tensor) or per_sample_ell.ndim != 1:
        raise ValueError("per_sample_ell must be a rank-1 torch.Tensor with shape (b,)")
    if not groups:
        raise ValueError("groups must contain at least one non-empty group")

    rewards = []
    for group in groups:
        indices = torch.as_tensor(group, dtype=torch.long, device=per_sample_ell.device)
        if indices.ndim != 1 or indices.numel() == 0:
            raise ValueError("each group must be a non-empty rank-1 index collection")
        if torch.any(indices < 0) or torch.any(indices >= per_sample_ell.numel()):
            raise ValueError("group index is outside per_sample_ell")
        rewards.append(per_sample_ell.index_select(0, indices).mean())
    return torch.stack(rewards)


class GroupEMA:
    """Per-group EMA baseline whose update returns the detached prior state."""

    def __init__(self, G, alpha=0.9):
        if not isinstance(G, Integral) or isinstance(G, bool) or G <= 0:
            raise ValueError(f"G must be a positive integer, got {G!r}")
        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not 0 <= alpha <= 1:
            raise ValueError(f"alpha must be in [0, 1], got {alpha!r}")
        self.G = int(G)
        self.alpha = float(alpha)
        self.b = None

    def update(self, ell_g):
        """Return the detached pre-update baseline, then incorporate ``ell_g``."""
        if not isinstance(ell_g, torch.Tensor) or ell_g.shape != (self.G,):
            shape = getattr(ell_g, "shape", None)
            raise ValueError(f"ell_g must have shape ({self.G},), got {shape}")

        observed = ell_g.detach()
        if self.b is None:
            self.b = observed.clone()
            return self.b.detach().clone()

        self.b = self.b.to(device=observed.device, dtype=observed.dtype)
        baseline = self.b.detach().clone()
        self.b = (self.alpha * self.b + (1.0 - self.alpha) * observed).detach()
        return baseline
