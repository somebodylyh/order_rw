"""Utility functions: Plackett-Luce sampling, Kendall tau, path coherence Q."""

import numpy as np
import torch
from typing import List


def plackett_luce_sample(scores: np.ndarray, mask: np.ndarray,
                         temperature: float = 1.0) -> List[int]:
    """
    Sequential categorical sampling without replacement (Plackett-Luce).

    Args:
        scores: (N,) unnormalized logits for each candidate.
        mask: (N,) bool array, True = already selected (excluded).
        temperature: softmax temperature.

    Returns:
        ordering: list of selected indices in order.
    """
    scores = np.asarray(scores, dtype=np.float64).copy()
    mask = np.asarray(mask, dtype=bool).copy()
    N = len(scores)
    ordering = []
    available = list(np.where(~mask)[0])
    adjusted = scores / max(temperature, 1e-8)
    # Apply -inf mask
    adjusted[mask] = -np.inf
    for _ in range(len(available)):
        probs = _softmax(adjusted)
        # Numerical safety: re-normalize after masking
        probs = np.nan_to_num(probs, nan=0.0)
        if probs.sum() == 0:
            # Fallback: uniform over remaining
            remaining = np.where(~mask)[0]
            chosen = np.random.choice(remaining)
        else:
            probs /= probs.sum()
            chosen = np.random.choice(N, p=probs)
        ordering.append(int(chosen))
        mask[chosen] = True
        adjusted[chosen] = -np.inf
    return ordering


def plackett_luce_sample_batch(scores: torch.Tensor, mask: torch.Tensor,
                                temperature: float = 1.0) -> torch.Tensor:
    """
    Batched Plackett-Luce sampling.

    Args:
        scores: (B, N) logits.
        mask: (B, N) bool, True = already selected.
        temperature: softmax temperature.

    Returns:
        orderings: (B, N) long tensor, each row is the sampled ordering.
    """
    B, N = scores.shape
    scores = scores / max(temperature, 1e-8)
    scores = scores.masked_fill(mask, float('-inf'))
    orderings = torch.zeros(B, N, dtype=torch.long, device=scores.device)
    for step in range(N):
        probs = torch.softmax(scores, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        chosen = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B,)
        orderings[:, step] = chosen
        scores.scatter_(1, chosen.unsqueeze(-1), float('-inf'))
    return orderings


def compute_path_coherence(ordering: List[int], A: np.ndarray) -> float:
    """
    Q(sigma) = sum_{t=1}^{N-1} A[sigma(t+1), sigma(t)]

    A[i, j] = affinity of block i (later, query) attending to block j (earlier, key).
    """
    total = 0.0
    for t in range(len(ordering) - 1):
        later = ordering[t + 1]
        earlier = ordering[t]
        total += A[later, earlier]
    return total


def kendall_tau(ordering1: List[int], ordering2: List[int]) -> float:
    """
    Kendall tau correlation between two orderings.
    Returns value in [-1, 1].
    """
    n = len(ordering1)
    assert n == len(ordering2)
    pos1 = {x: i for i, x in enumerate(ordering1)}
    pos2 = {x: i for i, x in enumerate(ordering2)}
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = ordering1[i]
            b = ordering1[j]
            order1 = pos1[a] < pos1[b]
            order2 = pos2[a] < pos2[b]
            if order1 == order2:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return 0.0
    return (concordant - discordant) / total


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.nanmax(x)
    e = np.exp(x)
    return e / e.sum()
