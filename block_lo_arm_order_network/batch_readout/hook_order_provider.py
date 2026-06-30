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
    _attn_to_A_block_loss_aligned_with_none_model_vec,
)
from batch_readout.integration_hook import FrozenBetaHook


def _attn_to_A_block_strict65_model_vec(attn, reveal_tokens, inv_perm,
                                        seq_len=SEQ_LEN, num_blocks=N, block_len=BLOCK_LEN):
    """Single-head strict65 model-frame block graph, EXACT training construction.

    Reproduces ``uniform_label_free_v1`` g_β training input:
        A65 = loss_aligned_with_none_model(attn, reveal)   # (num_blocks, num_blocks+1)
        B65 = build_none_separated_B(A65)                  # (65, 65), [None] = node 0
        B_in = B65[1:, 1:]                                 # (64, 64), [None] stripped

    The [None]/BOS token is kept as a SEPARATE source node (column 0) so its
    sink mass is isolated, then dropped — NOT folded into block 0 like the
    ``model`` agg. ``inv_perm`` is unused (model frame); the trainer applies the
    posthoc model->physical remap. ``FrozenBetaHook.step`` transposes the
    returned matrix, so we return ``B65[1:,1:].T`` == ``A65[:,1:]`` to land on the
    training-frame B after that transpose.
    """
    del inv_perm  # model frame: no physical remap here
    from none_separated_block_graph import build_none_separated_B
    A65 = _attn_to_A_block_loss_aligned_with_none_model_vec(
        attn, reveal_tokens, seq_len=seq_len, num_blocks=num_blocks, block_len=block_len
    )  # (..., num_blocks, num_blocks+1)
    B65 = build_none_separated_B(A65)        # (65, 65)
    out = B65[1:, 1:].T.astype(np.float32, copy=True)  # step() will transpose back
    di = np.arange(num_blocks)
    out[di, di] = 0.0
    return out


_AGG_FN = {"b1": _attn_to_A_block_b1_vec,
           "predictor": _attn_to_A_block_predictor_vec,
           "model": _attn_to_A_block_model_vec,
           "content": _attn_to_A_block_content_vec,
           "loss_aligned": _attn_to_A_block_loss_aligned_content_vec,
           "strict65_model": _attn_to_A_block_strict65_model_vec}


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
                 mode="argsort", tau=1.0, seed=0, device="cuda:0", none_mode="b1",
                 reverse=False):
        self.hook = FrozenBetaHook(
            g_beta_ckpt, mode=mode, tau=tau, reverse=reverse, seed=seed, device=device
        )
        self.head = tuple(head)
        self.clean_perm = clean_perm
        self.refresh_every = max(1, int(refresh_every))
        self.seed = int(seed)
        self.device = torch.device(device)
        self.none_mode = none_mode
        self.reverse = bool(reverse)
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
    # Free all GPU attention tensors now that we've moved the needed layer to CPU.
    del attn_list, attn_layer
    if device.type == "cuda":
        torch.cuda.empty_cache()

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


# ---------------------------------------------------------------------------
# All-layer all-head extraction (L layers × H heads → L*H heads total)
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_all_layers_all_heads_A_for_batch(model, idx_batch, clean_perm, device,
                                              probe_orders, none_mode="model"):
    """Forward `idx_batch` under `probe_orders`, extract ALL heads from ALL layers.

    Returns (batch, L*H, N, N) float32 block graphs (diagonal zeroed) in the
    frame specified by `none_mode` ("model" or "b1").
    """
    device = torch.device(device) if isinstance(device, str) else device
    agg = _AGG_FN.get(none_mode)
    if agg is None:
        raise ValueError(f"unknown none_mode={none_mode!r}, expected one of {sorted(_AGG_FN)}")

    model.eval()
    _, _, attn_list = model.forward_fn(idx_batch, probe_orders, return_attentions=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    n_layers = len(attn_list)
    # Move each layer to CPU one-at-a-time, stack with numpy (avoids GPU copy
    # from torch.stack which doubles peak memory).
    attn_np_list = []
    for a in attn_list:
        attn_np_list.append(a.cpu().numpy())
    attn_np = np.stack(attn_np_list, axis=0)  # (L, B, H, T+1, T+1)
    L, B_sz, H, Tp1, _ = attn_np.shape
    probe_np = probe_orders.cpu().numpy()
    # Release GPU attention tensors now that we're on CPU.
    del attn_list, attn_np_list
    if device.type == "cuda":
        torch.cuda.empty_cache()

    inv_perm = clean_perm.inv_perm_model_to_phys.cpu().numpy()
    if none_mode == "model":
        inv_perm = np.arange(N, dtype=np.int64)

    total_heads = L * H
    A = np.zeros((B_sz, total_heads, N, N), dtype=np.float32)
    for bi in range(B_sz):
        for li in range(L):
            for hi in range(H):
                A_h = agg(attn_np[li, bi, hi, :, :], probe_np[bi], inv_perm)
                A[bi, li * H + hi] = A_h

    # Zero diagonals
    diag = np.arange(N)
    A[:, :, diag, diag] = 0.0

    return torch.from_numpy(A).float()


@torch.no_grad()
def extract_all_layers_all_heads_strict65_A_for_batch(model, idx_batch, clean_perm, device,
                                                       probe_orders):
    """Like ``extract_all_layers_all_heads_A_for_batch`` but produces strict65
    (65×65 with None node) graphs via ``build_none_separated_B``, suitable for
    ``L0DynamicGBeta`` with heads = L*H.

    Returns (batch, L*H, 65, 65) float32 block graphs in model frame.
    """
    device = torch.device(device) if isinstance(device, str) else device
    from per_head_order_scan import _attn_to_A_block_loss_aligned_with_none_model_vec
    from none_separated_block_graph import build_none_separated_B

    model.eval()
    _, _, attn_list = model.forward_fn(idx_batch, probe_orders, return_attentions=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    n_layers = len(attn_list)
    # Move each layer to CPU one-at-a-time, stack with numpy.
    attn_np_list = []
    for a in attn_list:
        attn_np_list.append(a.cpu().numpy())
    attn_np = np.stack(attn_np_list, axis=0)  # (L, B, H, T+1, T+1)
    L, B_sz, H_np, Tp1, _ = attn_np.shape
    probe_np = probe_orders.cpu().numpy()
    del attn_list, attn_np_list
    if device.type == "cuda":
        torch.cuda.empty_cache()

    total_heads = L * H_np
    B_out = np.zeros((B_sz, total_heads, 65, 65), dtype=np.float32)
    for bi in range(B_sz):
        # _attn_to_A_block_loss_aligned_with_none_model_vec handles arbitrary
        # leading dims, so (L, H, 257, 257) → (L, H, 64, 65).
        A = _attn_to_A_block_loss_aligned_with_none_model_vec(
            attn_np[:, bi, :, :, :],  # (L, H, 257, 257)
            probe_np[bi],             # (256,)
        )  # → (L, H, 64, 65)
        for li in range(L):
            for hi in range(H_np):
                B_out[bi, li * H_np + hi] = build_none_separated_B(A[li, hi])  # (65, 65)

    return torch.from_numpy(B_out).float()


class HeadGatedHookOrderProvider:
    """Head-gated g_beta order provider — extracts ALL heads from a layer.

    Replaces the single-head HookOrderProvider with multi-head extraction
    and a learned head-gating MLP (head_gated_gbeta.py).
    """

    def __init__(self, g_beta_ckpt, layer, clean_perm, refresh_every=1,
                 topk=2, mode="argsort", tau=1.0, seed=0, device="cuda:0",
                 none_mode="model", reverse=False):
        from batch_readout.head_gated_gbeta import build_head_gated_gbeta
        import torch

        state = torch.load(g_beta_ckpt, map_location="cpu", weights_only=False)
        cfg = state["config"]
        variant = cfg.get("variant", "head_gated")
        H = cfg.get("H", 16)
        gate_mode = cfg.get("gate_mode", "topk")
        topk = cfg.get("topk", topk)

        self.model = build_head_gated_gbeta(
            variant,
            N=N,
            H=H,
            head_idx=cfg.get("head_idx", 0),
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
        self.reverse = bool(reverse)

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
            # Extract all heads from specified layer (returns CPU tensor after
            # mem-safety fix — attn_list is freed inside).
            A_heads = extract_all_heads_A_for_batch(
                model, idx_batch, self.layer, self.clean_perm,
                self.device, probe, none_mode=self.none_mode,
            )  # (B, H, N, N) on CPU

            # Compute batch-mean B = A^T (per head) on CPU, then move to GPU.
            B_heads = A_heads.transpose(-1, -2).mean(dim=0, keepdim=True)  # (1, H, N, N)
            diag = torch.arange(N, device=B_heads.device)
            B_heads[:, :, diag, diag] = 0.0
            B_heads = B_heads.to(self.device)
            del A_heads
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            # Run head-gated g_beta
            z, aux = self.model(B_heads)  # z: (1, N)

            # Produce order
            from batch_readout.pl_sampling import pl_argsort, pl_sample
            z_cpu = z.cpu()
            if self.mode == "argsort":
                sigma = pl_argsort(z_cpu)[0]  # (N,)
            else:
                sigma = pl_sample(z_cpu, tau=self.tau, generator=self.generator)[0]
            if self.reverse:
                sigma = torch.flip(sigma, dims=[0])

            # For model/content modes the training loop performs the model->physical
            # remap after calling physical_order(), matching the existing hook path.
            self._sigma = sigma
            self._last_refresh = global_step

        return self._sigma


# ---------------------------------------------------------------------------
# All-layer head-gated order provider (L layers × H heads → L*H total)
# ---------------------------------------------------------------------------

class AllLayerHeadGatedHookOrderProvider:
    """Head-gated g_beta order provider — extracts ALL heads from ALL layers.

    Uses ``extract_all_layers_all_heads_A_for_batch`` to gather B matrices from
    every (layer, head) pair, then gates across L*H heads with
    ``HeadGatedGBeta(H=L*H, gate_mode="topk", topk=K)``.

    The gate automatically learns to focus on the most informative heads
    (identified by the label-free structure score as L1H0/L1H7).
    """

    def __init__(self, g_beta_ckpt, clean_perm, refresh_every=1,
                 topk=4, mode="argsort", tau=1.0, seed=0, device="cuda:0",
                 none_mode="model", reverse=False, batch_mean_probes=1):
        import torch

        state = torch.load(g_beta_ckpt, map_location="cpu", weights_only=False)
        cfg = state["config"]
        model_name = cfg.get("model_name", "?")

        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")

        # Determine total heads
        n_layers = cfg.get("n_layers", 4)
        n_heads_per_layer = cfg.get("H", cfg.get("heads", 8))
        H_all = n_layers * n_heads_per_layer

        if model_name == "l0_dynamic_gbeta_v0":
            # ── L0DynamicGBeta checkpoint (strict65, 65×65) ──
            from batch_readout.l0_dynamic_gbeta import L0DynamicGBeta
            # For l0_dynamic_gbeta_v0, 'heads' in config is the TOTAL number
            # of heads used during training (e.g. 32 = 4 layers × 8 heads).
            # Use it directly — don't multiply by n_layers again.
            H_use = cfg.get("heads", H_all)
            self.model = L0DynamicGBeta(
                heads=H_use, nodes=65,
                scorer_hidden=tuple(cfg.get("scorer_hidden", (256, 64))),
                gate_hidden=cfg.get("gate_hidden", 32),
            )
            self.model.load_state_dict(state["model_state_dict"])
            self.model.eval()
            self.model.to(self.device)
            self._use_strict65 = True
            self._model_type = "L0DynamicGBeta"
            self.H_all = H_use
        else:
            # ── HeadGatedGBeta checkpoint (64×64) ──
            from batch_readout.head_gated_gbeta import build_head_gated_gbeta
            variant = cfg.get("variant", "head_gated")
            gate_mode = cfg.get("gate_mode", "topk")
            readout_hidden = cfg.get("readout_hidden", (1024, 256))
            gate_hidden = cfg.get("gate_hidden", (256,))
            self.model = build_head_gated_gbeta(
                variant, N=N, H=H_all,
                head_idx=cfg.get("head_idx", 0),
                gate_mode=gate_mode, topk=topk,
                readout_hidden=readout_hidden, gate_hidden=gate_hidden,
            )
            missing, unexpected = self.model.load_state_dict(state["model"], strict=False)
            if missing:
                import sys
                print(f"[AllLayerHeadGated] WARNING: missing keys: {missing}", file=sys.stderr)
            self.model.eval()
            self.model.to(self.device)
            self._use_strict65 = False
            self._model_type = "HeadGatedGBeta"
            self.H_all = H_all

        self.clean_perm = clean_perm
        self.refresh_every = max(1, int(refresh_every))
        self.mode = mode
        self.tau = float(tau)
        self.seed = int(seed)
        self.none_mode = none_mode
        self.reverse = bool(reverse)
        self.n_layers = n_layers
        self.batch_mean_probes = max(1, int(batch_mean_probes))

        self._sigma = None
        self._last_refresh = None

        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))

    @torch.no_grad()
    def physical_order(self, model, idx_batch, global_step):
        if self._sigma is None or (global_step - self._last_refresh) >= self.refresh_every:
            n_probes = self.batch_mean_probes
            B = idx_batch.shape[0]

            if self._use_strict65:
                # Strict65 path (L0DynamicGBeta): (B, H, 65, 65) with None node.
                # Accumulate on CPU to avoid holding n_probes copies on GPU.
                B_sum = None  # CPU float32
                for pi in range(n_probes):
                    probe = random_probe_token_orders(B, self.seed,
                                                      global_step * 1000 + pi,
                                                      self.device)
                    B_pi = extract_all_layers_all_heads_strict65_A_for_batch(
                        model, idx_batch, self.clean_perm, self.device, probe,
                    )  # (B, L*H, 65, 65) — already on CPU after mem-safety fix
                    if B_sum is None:
                        B_sum = B_pi.float()
                    else:
                        B_sum += B_pi.float()
                    del B_pi, probe
                B_heads = (B_sum / n_probes).to(self.device)
                del B_sum
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                z, aux = self.model(B_heads, apply_head_dropout=False)  # (B, 64)
            else:
                # HeadGatedGBeta path: (B, L*H, 64, 64).
                # Accumulate on CPU.
                A_sum = None  # CPU float32
                for pi in range(n_probes):
                    probe = random_probe_token_orders(B, self.seed,
                                                      global_step * 1000 + pi,
                                                      self.device)
                    A_pi = extract_all_layers_all_heads_A_for_batch(
                        model, idx_batch, self.clean_perm,
                        self.device, probe, none_mode=self.none_mode,
                    )  # (B, L*H, N, N) — already on CPU after mem-safety fix
                    if A_sum is None:
                        A_sum = A_pi.float()
                    else:
                        A_sum += A_pi.float()
                    del A_pi, probe
                A_heads = A_sum / n_probes
                del A_sum
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                B_heads = A_heads.transpose(-1, -2)  # (B, L*H, N, N)
                diag = torch.arange(N, device=B_heads.device)
                B_heads[:, :, diag, diag] = 0.0
                B_heads = B_heads.to(self.device)
                z, aux = self.model(B_heads)  # z: (B, N)

            # Produce order
            from batch_readout.pl_sampling import pl_argsort, pl_sample
            z_cpu = z.cpu()
            if self.mode == "argsort":
                sigma = pl_argsort(z_cpu)[0]  # (N,)
            else:
                sigma = pl_sample(z_cpu, tau=self.tau, generator=self.generator)[0]
            if self.reverse:
                sigma = torch.flip(sigma, dims=[0])

            self._sigma = sigma
            self._last_refresh = global_step

        return self._sigma
