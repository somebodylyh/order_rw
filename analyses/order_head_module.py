"""Internal OrderHead: wraps the deployed gβ MLP as a model module.

Batch-mean readout reproduces the external FrozenBetaHook (bit-match target);
per-sample readout is the V3-headline config. MODEL-frame scores throughout —
callers remap MODEL→PHYSICAL before any order_nll (see AOGPTWithOrderHead).
"""
import sys
sys.path.insert(0, "block_lo_arm_order_network")
import numpy as np
import torch
import torch.nn as nn

from batch_readout.integration_hook import FrozenBetaHook
from batch_readout.pl_sampling import pl_argsort
from analyses.p7_gbeta_policy import sample_pl, GBETA_CKPT


def bmatrix_from_A(A, per_sample):
    """A: (Bs,N,N) -> B. per_sample: (Bs,N,N); else batch-mean (1,N,N). Diagonal zeroed.
    Matches FrozenBetaHook.step: B = mean over batch of A^T, diag=0."""
    B = A.transpose(1, 2)
    B = B.clone() if per_sample else B.mean(dim=0, keepdim=True)
    n = B.shape[-1]
    d = torch.arange(n, device=B.device)
    B[:, d, d] = 0.0
    return B


class OrderHeadModule(nn.Module):
    def __init__(self, gbeta_ckpt=GBETA_CKPT, device="cpu"):
        super().__init__()
        hook = FrozenBetaHook(gbeta_ckpt, mode="argsort", device=device)
        self.gbeta = hook.model        # nn.Module, MODEL-frame scores; params ARE gβ
        self.device = device

    def scores(self, A, per_sample):
        B = bmatrix_from_A(A.to(self.device).float(), per_sample)
        return self.gbeta(B)           # (rows, N), grad-enabled through gβ

    def argsort_order(self, A, per_sample=False):
        z = self.scores(A, per_sample)
        return pl_argsort(z.detach().cpu())        # (rows, N) model-frame block orders

    def sample_pl_order(self, A, per_sample, tau):
        z = self.scores(A, per_sample)
        orders, logps, ents = [], [], []
        for r in range(z.shape[0]):
            o, lp, e = sample_pl(z[r], tau)
            orders.append(o); logps.append(lp); ents.append(e)
        return np.stack(orders), torch.stack(logps), torch.stack(ents)
