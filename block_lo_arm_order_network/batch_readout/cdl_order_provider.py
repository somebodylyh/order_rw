"""In-loop CDL teacher (C-D+L) order provider — no learned g_β, hand-designed scaffold.

Frozen-baseline for reviewer question: "why not just use CDL directly?"

Mirrors HookOrderProvider's extraction (selected head, random-probe regime, K-step
refresh) but replaces g_β with the CDL teacher. The order is computed by the C-D+L
hand-designed teacher from the extracted attention graph B.
"""
from __future__ import annotations

import numpy as np
import torch

from training_utils import N
from attn_order_teacher import rollout_order  # (64,) int64 block order


class CdlOrderProvider:
    """CDL teacher order provider with K-step refresh.

    `.physical_order(model, idx_batch, global_step)` returns a physical-frame block
    order (N,) int64 computed by CDL teacher from a fresh extraction.
    """

    def __init__(self, head, clean_perm, refresh_every=10, tau_T=1.0,
                 mode="C-D+L", seed=0, device="cuda:0", none_mode="b1",
                 greedy=True, reverse=False):
        self.head = tuple(head)
        self.clean_perm = clean_perm
        self.refresh_every = max(1, int(refresh_every))
        self.tau_T = float(tau_T)
        self.mode = mode
        self.seed = int(seed)
        self.device = torch.device(device)
        self.none_mode = none_mode
        self.greedy = greedy
        self.reverse = bool(reverse)
        self._sigma = None
        self._last_refresh = None

    def physical_order(self, model, idx_batch, global_step):
        if self._sigma is None or (global_step - self._last_refresh) >= self.refresh_every:
            # Delayed imports to avoid circular dependency at module load
            from batch_readout.hook_order_provider import (
                extract_selected_head_A_for_batch, random_probe_token_orders,
            )

            probe = random_probe_token_orders(
                idx_batch.shape[0], self.seed, global_step, self.device
            )
            A_batch = extract_selected_head_A_for_batch(
                model, idx_batch, self.head, self.clean_perm, self.device, probe,
                none_mode=self.none_mode,
            )  # (B, N, N) float32

            # Batch-mean A → B
            A_mean = A_batch.mean(dim=0).cpu().numpy().astype(np.float64)
            B = np.asarray(A_mean.T, dtype=np.float64).copy()
            np.fill_diagonal(B, 0.0)

            # Pure C-D+L greedy rollout (no readiness anchoring)
            order = rollout_order(
                B, tau_T=self.tau_T, seed=self.seed + global_step,
                mode=self.mode, greedy=self.greedy,
            )  # (N,) int64

            sigma = order.astype(np.int64)
            if self.reverse:
                sigma = sigma[::-1].copy()  # flip physical block order
            self._sigma = torch.from_numpy(sigma)
            self._last_refresh = global_step

        return self._sigma
