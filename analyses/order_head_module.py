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
        self._requested_device = device
        self.gbeta.to(device)

    @property
    def device(self):
        """Current gβ device, including moves made through ``module.to(...)``."""
        return next(self.gbeta.parameters()).device

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


# ── Task 2: AOGPTWithOrderHead wrapper ───────────────────────────────────────

from batch_readout.hook_order_provider import (  # noqa: E402
    extract_selected_head_A_for_batch, random_probe_token_orders,  # noqa: F401
)
from analyses.p7_gbeta_policy import HEAD, NONE_MODE  # noqa: E402
from analyses.p5_utility_controller import BLOCK_LEN, N  # noqa: E402, F401
from clean_training_protocol import physical_blocks_to_model_token_order  # noqa: E402


class AOGPTWithOrderHead(nn.Module):
    """Backbone + internal OrderHead. B is detached before the OrderHead so no
    policy gradient reaches the backbone (implicit co-adaptation)."""

    def __init__(self, backbone, order_head, clean_perm, device="cpu"):
        super().__init__()
        self.backbone = backbone
        self.order_head = order_head
        self.clean_perm = clean_perm
        self.inv_perm = clean_perm.inv_perm_model_to_phys.cpu().numpy()  # [model]=phys
        self.device = device

    def extract_B(self, idx_batch, probe):
        A = extract_selected_head_A_for_batch(
            self.backbone, idx_batch.to(self.device), HEAD, self.clean_perm,
            self.device, probe, none_mode=NONE_MODE)
        return A.to(self.device).float()

    def extract_B65(self, idx_batch, probe):
        """Selected-head 65-node strict65 B (None row/col RETAINED) — the CDL
        teacher input. ``extract_B`` strips None to 64x64 for gβ; the greedy
        C-D+L rollout needs the None-support row, so this keeps all 65 nodes."""
        from batch_readout.hook_order_provider import (
            extract_all_layers_all_heads_strict65_A_for_batch,
        )
        allB = extract_all_layers_all_heads_strict65_A_for_batch(
            self.backbone, idx_batch.to(self.device), self.clean_perm,
            self.device, probe)                       # (B, L*H, 65, 65)
        hidx = HEAD[0] * self.backbone.config.n_head + HEAD[1]
        B65 = allB[:, hidx]                            # (B, 65, 65) selected head
        if not torch.is_tensor(B65):
            B65 = torch.as_tensor(B65)
        return B65.to(self.device).float()

    def compute_order_logits(self, idx_batch, probe, per_sample):
        A = self.extract_B(idx_batch, probe).detach()   # DETACH — no grad to backbone
        return self.order_head.scores(A, per_sample)     # grad on order_head only

    def token_orders_from_model_blocks(self, order_model_BN):
        phys = self.inv_perm[np.asarray(order_model_BN, dtype=np.int64)]   # (Bs,N) model->phys
        toks = [physical_blocks_to_model_token_order(
                    torch.from_numpy(phys[r:r + 1]), self.clean_perm, BLOCK_LEN)[0]
                for r in range(phys.shape[0])]
        return torch.stack(toks)                                             # (Bs,T)
