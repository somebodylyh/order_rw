"""BR-1 Task 14: in-loop frozen-beta order provider.

Holds a trained g_beta checkpoint and a sampling mode. On `.step(A)` it:
  1. computes B_batch = mean over the batch axis of A^T (zeroed diagonal);
  2. runs the frozen g_beta to get z;
  3. returns either argsort(-z) or PL-sample(z/tau).

The returned sigma is shape (N,) int64 — the controller of the training
loop applies it as the order for the *next* step.
"""
from __future__ import annotations

import torch

from batch_readout.model import FlattenReadout, NodewiseReadout
from batch_readout.pl_sampling import pl_argsort, pl_sample


def _build_from_config(cfg: dict):
    name = cfg["model_name"]
    N = int(cfg.get("N", 64))
    if name == "flatten":
        return FlattenReadout(N=N, hidden=tuple(cfg.get("hidden", (1024, 256))))
    if name == "nodewise":
        return NodewiseReadout(
            N=N,
            d_model=int(cfg.get("d_model", 64)),
            n_layers=int(cfg.get("n_layers", 2)),
            n_heads=int(cfg.get("n_heads", 4)),
        )
    raise ValueError(f"unknown model {name!r} in g_beta config")


class FrozenBetaHook:
    def __init__(
        self,
        g_beta_ckpt: str,
        mode: str = "argsort",
        tau: float = 1.0,
        seed=None,
        device: str = "cuda:0",
    ):
        if mode not in ("argsort", "sample"):
            raise ValueError(f"mode must be 'argsort' or 'sample', got {mode!r}")
        state = torch.load(g_beta_ckpt, map_location="cpu", weights_only=False)
        self.model = _build_from_config(state["config"])
        self.model.load_state_dict(state["model"])
        self.model.eval()
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.model.to(self.device)
        self.mode = mode
        self.tau = float(tau)
        # CPU generator -- the PL sampler runs on CPU tensors.
        self.generator = torch.Generator(device="cpu")
        if seed is not None:
            self.generator.manual_seed(int(seed))

    @torch.no_grad()
    def step(self, attention: torch.Tensor) -> torch.Tensor:
        """attention: (batch, N, N) per-sample A_theta(x). Returns sigma: (N,) int64."""
        if attention.ndim != 3 or attention.shape[1] != attention.shape[2]:
            raise ValueError(f"attention must be (batch, N, N); got {tuple(attention.shape)}")
        A = attention.to(self.device).float()
        # B_batch = mean over batch of A^T, diagonal zeroed.
        B = A.transpose(1, 2).mean(dim=0, keepdim=True)
        N = B.shape[-1]
        diag = torch.arange(N, device=B.device)
        B[:, diag, diag] = 0.0
        z = self.model(B).cpu()
        if self.mode == "argsort":
            return pl_argsort(z)[0]
        return pl_sample(z, tau=self.tau, generator=self.generator)[0]
