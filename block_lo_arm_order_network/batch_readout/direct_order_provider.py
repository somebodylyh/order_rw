"""Direct non-learned order scores on model-frame strict65 attention graphs.

All functions use the repository convention ``B[source, target]``. Inputs are
L0 all-head model-frame strict65 graphs, and heads are fused by arithmetic mean.
Larger scores are revealed earlier.
"""

from __future__ import annotations

import torch

from batch_readout.frozen_gbeta_hook import (
    extract_probe_averaged_model_frame_strict65,
)


DIRECT_POLICIES = (
    "initial_cdl_one_shot",
    "source_mass",
    "readiness",
)


def _validate_strict65(B: torch.Tensor) -> torch.Tensor:
    if not isinstance(B, torch.Tensor):
        raise TypeError(f"B must be a torch.Tensor, got {type(B).__name__}")
    if B.ndim != 4 or tuple(B.shape[-2:]) != (65, 65):
        raise ValueError(
            "B must have shape (batch, heads, 65, 65), "
            f"got {tuple(B.shape)}"
        )
    return B


def _content_masses(B: torch.Tensor):
    M = B[:, :, 1:, 1:]
    diag = M.diagonal(dim1=-2, dim2=-1)
    row_sum = M.sum(dim=-1) - diag
    col_sum = M.sum(dim=-2) - diag
    return row_sum, col_sum


def initial_cdl_scores_strict65(B: torch.Tensor):
    """Return the old CDL initial-state score without sequential rollout.

    With None fixed as selected/last and content nodes initially unselected:

        q_0(i) = C_0(i) - D_0(i) + L_0(i)
               = B[0,i] - mean_{u in U excluding i} B[u,i] + B[0,i]
               = 2 B[0,i] - mean_{u in U excluding i} B[u,i]
    """
    B = _validate_strict65(B)
    none_support = B[:, :, 0, 1:]
    _, col_sum = _content_masses(B)
    dependency = col_sum / 63.0
    scores_h = 2.0 * none_support - dependency
    return scores_h, scores_h.mean(dim=1)


def source_mass_scores_strict65(B: torch.Tensor):
    """Score candidate i by row mass: candidate i -> other content targets.

    This is a source heuristic, not the old CDL score.
    """
    B = _validate_strict65(B)
    row_sum, _ = _content_masses(B)
    return row_sum, row_sum.mean(dim=1)


def readiness_scores_strict65(B: torch.Tensor, lambda_dep: float = 1.0):
    """Score source strength minus incoming dependency strength."""
    B = _validate_strict65(B)
    row_sum, col_sum = _content_masses(B)
    scores_h = row_sum - float(lambda_dep) * col_sum
    return scores_h, scores_h.mean(dim=1)


def direct_policy_scores(
    B: torch.Tensor,
    policy: str,
    lambda_dep: float = 1.0,
):
    """Dispatch a direct strict65 policy and return per-head/fused scores."""
    if policy == "initial_cdl_one_shot":
        return initial_cdl_scores_strict65(B)
    if policy == "source_mass":
        return source_mass_scores_strict65(B)
    if policy == "readiness":
        return readiness_scores_strict65(B, lambda_dep=lambda_dep)
    raise ValueError(
        f"unknown direct policy {policy!r}; expected one of {DIRECT_POLICIES}"
    )


class DirectModelFrameOrderProvider:
    """Training-loop provider for direct model-frame strict65 policies."""

    def __init__(
        self,
        policy: str,
        lambda_dep: float = 1.0,
        batch_mean_probes: int = 4,
        refresh_every: int = 10,
        seed: int = 0,
        device: str = "cuda:0",
    ):
        if policy not in DIRECT_POLICIES:
            raise ValueError(
                f"unknown direct policy {policy!r}; expected one of "
                f"{DIRECT_POLICIES}"
            )
        resolved_device = torch.device(device)
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            resolved_device = torch.device("cpu")

        self.policy = policy
        self.lambda_dep = float(lambda_dep)
        self.batch_mean_probes = max(1, int(batch_mean_probes))
        self.refresh_every = max(1, int(refresh_every))
        self.seed = int(seed)
        self.device = resolved_device
        self.none_mode = "model"
        self._sigma = None
        self._last_refresh = None

    @torch.no_grad()
    def physical_order(self, model, idx_batch, global_step):
        if (
            self._sigma is None
            or (int(global_step) - int(self._last_refresh)) >= self.refresh_every
        ):
            B = extract_probe_averaged_model_frame_strict65(
                model,
                idx_batch,
                global_step=int(global_step),
                seed=self.seed,
                batch_mean_probes=self.batch_mean_probes,
                device=self.device,
            )
            _, scores = direct_policy_scores(
                B,
                policy=self.policy,
                lambda_dep=self.lambda_dep,
            )
            self._sigma = scores.argsort(dim=-1, descending=True)[0].cpu()
            self._last_refresh = int(global_step)
        return self._sigma
