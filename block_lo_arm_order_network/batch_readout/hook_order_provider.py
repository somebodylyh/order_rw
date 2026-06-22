"""Phase 2 (§5): in-loop frozen-g_β order provider for the training hook.

Wraps `batch_readout.integration_hook.FrozenBetaHook` with the in-loop extraction
of the SELECTED head's physical-frame B1 (65-node) block graph, so the order fed
back into training comes from the SAME extraction path g_β was pretrained on
(per_head_order_scan none_mode='b1' or 'model', one head) — preventing the B_train≠B_hook
mismatch flagged in the spec §3.2.

Cost control (spec §6 "100-step benchmark gate"): the order is recomputed only
every `refresh_every` steps via a no-grad probe forward, and reused in between.
The probe uses RANDOM block orders (matching the offline extraction regime) so
g_β keeps seeing in-distribution graphs even after the hook drives training away
from random order. g_β stays frozen; no CDL / L2R / NLL touches it.
"""
from __future__ import annotations

import numpy as np
import torch

from training_utils import SEQ_LEN, N, BLOCK_LEN
from clean_training_protocol import expand_model_blocks_to_token_order
from per_head_order_scan import (
    _attn_to_A_block_b1_vec,
    _attn_to_A_block_loss_aligned_content_vec,
    _attn_to_A_block_predictor_vec,
    _attn_to_A_block_model_vec,
    _attn_to_A_block_content_vec,
)
from batch_readout.integration_hook import FrozenBetaHook

_AGG_FN = {"b1": _attn_to_A_block_b1_vec,
           "predictor": _attn_to_A_block_predictor_vec,
           "model": _attn_to_A_block_model_vec,
           "content": _attn_to_A_block_content_vec,
           "loss_aligned": _attn_to_A_block_loss_aligned_content_vec}


@torch.no_grad()
def extract_selected_head_A_for_batch(model, idx_batch, head, clean_perm, device, probe_orders,
                                       none_mode="b1"):
    """Forward `idx_batch` under `probe_orders`, extract one head's block graph.

    Returns (batch, N, N) float32 block graphs (diagonal zeroed).
    `probe_orders` are model-coordinate token orders (batch, SEQ_LEN).
    `none_mode` selects the [None]-handling: 'b1' | 'predictor' |
    'model' | 'content' | 'loss_aligned'.
    """
    l, h = head
    device = torch.device(device)
    inv_perm = clean_perm.inv_perm_model_to_phys.cpu().numpy()
    agg = _AGG_FN.get(none_mode)
    if agg is None:
        raise ValueError(f"unknown none_mode={none_mode!r}, expected one of {sorted(_AGG_FN)}")
    model.eval()
    _, _, attn_list = model.forward_fn(idx_batch, probe_orders, return_attentions=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    attn_batch = torch.stack(attn_list).cpu().numpy()  # (L, B, H, T+1, T+1)
    Bsz = int(idx_batch.shape[0])
    probe_np = probe_orders.cpu().numpy()
    A = np.zeros((Bsz, N, N), dtype=np.float32)
    for bi in range(Bsz):
        A[bi] = agg(attn_batch[l, bi, h], probe_np[bi], inv_perm)
    return torch.from_numpy(A).float()


def random_probe_token_orders(batch_size, seed, global_step, device):
    """Random physical-block permutations -> model token orders (batch, SEQ_LEN).

    Deterministic in (seed, global_step); the probe regime is random order to keep
    the extracted graphs in-distribution for g_β.
    """
    rows = []
    for b in range(batch_size):
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed) * 100_000_000 + int(global_step) * 1000 + b)
        blocks = torch.randperm(N, generator=g)
        rows.append(expand_model_blocks_to_token_order(blocks.unsqueeze(0), BLOCK_LEN)[0])
    return torch.stack(rows).to(device)


class HookOrderProvider:
    """Frozen g_β order provider with K-step refresh.

    `.physical_order(model, idx_batch, global_step)` returns a physical-frame block
    order (N,) int64 to use for training; it is recomputed every `refresh_every`
    steps from a fresh random-probe extraction and cached in between.
    """

    def __init__(self, g_beta_ckpt, head, clean_perm, refresh_every=1,
                 mode="argsort", tau=1.0, seed=0, device="cuda:0", none_mode="b1"):
        self.hook = FrozenBetaHook(g_beta_ckpt, mode=mode, tau=tau, seed=seed, device=device)
        self.head = tuple(head)
        self.clean_perm = clean_perm
        self.refresh_every = max(1, int(refresh_every))
        self.seed = int(seed)
        self.device = torch.device(device)
        self.none_mode = none_mode
        self._sigma = None
        self._last_refresh = None

    def physical_order(self, model, idx_batch, global_step):
        if self._sigma is None or (global_step - self._last_refresh) >= self.refresh_every:
            probe = random_probe_token_orders(idx_batch.shape[0], self.seed, global_step, self.device)
            A = extract_selected_head_A_for_batch(
                model, idx_batch, self.head, self.clean_perm, self.device, probe,
                none_mode=self.none_mode,
            )
            self._sigma = self.hook.step(A.to(self.device)).cpu()  # (N,) physical-frame
            self._last_refresh = global_step
        return self._sigma


# ---------------------------------------------------------------------------
# Multi-head extraction (for head-gated g_beta)
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_all_heads_A_for_batch(model, idx_batch, layer, clean_perm, device,
                                  probe_orders, none_mode="model"):
    """Forward `idx_batch` under `probe_orders`, extract ALL heads' block graphs.

    Returns (batch, num_heads, N, N) float32 block graphs (diagonal zeroed)
    in model frame.
    """
    device = torch.device(device) if isinstance(device, str) else device
    from per_head_order_scan import (
        _attn_to_A_block_model_vec,
        _attn_to_A_block_b1_vec,
    )

    agg = _attn_to_A_block_model_vec if none_mode == "model" else _attn_to_A_block_b1_vec
    model.eval()
    _, _, attn_list = model.forward_fn(idx_batch, probe_orders, return_attentions=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    attn_layer = attn_list[layer]  # (B, H, T+1, T+1)
    B_sz, H, Tp1, _ = attn_layer.shape
    attn_np = attn_layer.cpu().numpy()
    probe_np = probe_orders.cpu().numpy()

    inv_perm = clean_perm.inv_perm_model_to_phys.cpu().numpy()
    if none_mode == "model":
        inv_perm = np.arange(N, dtype=np.int64)

    A = np.zeros((B_sz, H, N, N), dtype=np.float32)
    for bi in range(B_sz):
        for h in range(H):
            A_h = agg(attn_np[bi, h, :, :], probe_np[bi], inv_perm)
            A[bi, h] = A_h

    # Zero diagonals
    diag = np.arange(N)
    A[:, :, diag, diag] = 0.0

    return torch.from_numpy(A).float()


class HeadGatedHookOrderProvider:
    """Head-gated g_beta order provider — extracts ALL heads from a layer.

    Replaces the single-head HookOrderProvider with multi-head extraction
    and a learned head-gating MLP (head_gated_gbeta.py).
    """

    def __init__(self, g_beta_ckpt, layer, clean_perm, refresh_every=1,
                 topk=2, mode="argsort", tau=1.0, seed=0, device="cuda:0",
                 none_mode="model"):
        from batch_readout.head_gated_gbeta import HeadGatedGBeta
        import torch

        state = torch.load(g_beta_ckpt, map_location="cpu", weights_only=False)
        cfg = state["config"]
        H = cfg.get("H", 16)
        gate_mode = cfg.get("gate_mode", "topk")
        topk = cfg.get("topk", topk)

        self.model = HeadGatedGBeta(
            N=N, H=H,
            gate_mode=gate_mode,
            topk=topk,
        )
        self.model.load_state_dict(state["model"])
        self.model.eval()

        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        self.model.to(self.device)

        self.layer = int(layer)
        self.clean_perm = clean_perm
        self.refresh_every = max(1, int(refresh_every))
        self.mode = mode
        self.tau = float(tau)
        self.seed = int(seed)
        self.none_mode = none_mode

        self._sigma = None
        self._last_refresh = None

        # CPU generator for PL sampling
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))

    @torch.no_grad()
    def physical_order(self, model, idx_batch, global_step):
        if self._sigma is None or (global_step - self._last_refresh) >= self.refresh_every:
            probe = random_probe_token_orders(
                idx_batch.shape[0], self.seed, global_step, self.device
            )
            # Extract all heads from specified layer
            A_heads = extract_all_heads_A_for_batch(
                model, idx_batch, self.layer, self.clean_perm,
                self.device, probe, none_mode=self.none_mode,
            )  # (B, H, N, N)

            # Compute batch-mean B = A^T (per head)
            B_heads = A_heads.transpose(-1, -2).mean(dim=0, keepdim=True)  # (1, H, N, N)
            # Zero diagonal (already zeroed but redo for safety)
            diag = torch.arange(N, device=B_heads.device)
            B_heads[:, :, diag, diag] = 0.0

            B_heads = B_heads.to(self.device)

            # Run head-gated g_beta
            z, aux = self.model(B_heads)  # z: (1, N)

            # Produce order
            from batch_readout.pl_sampling import pl_argsort, pl_sample
            z_cpu = z.cpu()
            if self.mode == "argsort":
                sigma = pl_argsort(z_cpu)[0]  # (N,)
            else:
                sigma = pl_sample(z_cpu, tau=self.tau, generator=self.generator)[0]

            # sigma is in model frame. For model none_mode, remap to physical.
            if self.none_mode == "model":
                from clean_training_protocol import model_blocks_to_physical_blocks
                sigma = model_blocks_to_physical_blocks(sigma.unsqueeze(0), self.clean_perm)[0]

            self._sigma = sigma
            self._last_refresh = global_step

        return self._sigma
