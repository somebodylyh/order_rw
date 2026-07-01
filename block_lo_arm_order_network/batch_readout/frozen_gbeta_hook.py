"""Model-frame frozen g_beta order provider for AO-GPT training hook.

LABEL-FREE: this module operates entirely in model-frame coordinates.
It does NOT read, use, or save inv_perm, clean_perm, block_perm, or any
physical-coordinate metadata.  Physical remapping is reserved for
post-hoc characterization and oracle baselines only.

Flow:
  1. no-grad probe forward with random model-frame probe orders
  2. extract L0 all-head attention → build_model_frame_strict65 → B [1,8,65,65]
  3. g_beta(B) → scores [1,64]
  4. sigma_model = argsort(-scores)
  5. expand sigma_model → model-frame token order
  6. return token order for AO-GPT training
"""

from __future__ import annotations

import numpy as np
import torch

from training_utils import SEQ_LEN, N, BLOCK_LEN
from clean_training_protocol import expand_model_blocks_to_token_order
from batch_readout.l0_strict65 import build_model_frame_strict65
from batch_readout.l0_dynamic_gbeta import L0DynamicGBeta
from batch_readout.integration_hook import _build_from_config


# ---------------------------------------------------------------------------
# Shared model-frame strict65 extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_probe_averaged_model_frame_strict65(
    aogpt_model: torch.nn.Module,
    idx_batch: torch.Tensor,
    global_step: int,
    seed: int,
    batch_mean_probes: int,
    device: torch.device,
) -> torch.Tensor:
    """Extract probe-averaged L0 all-head model-frame strict65 graphs."""
    device = torch.device(device)
    n_probes = max(1, int(batch_mean_probes))
    batch_size = int(idx_batch.shape[0])
    num_heads = (
        aogpt_model.config.n_head if hasattr(aogpt_model, "config") else 8
    )

    B_samples = np.empty(
        (n_probes, batch_size, num_heads, 65, 65),
        dtype=np.float32,
    )
    for probe_idx in range(n_probes):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            int(seed) * 100_000_000
            + int(global_step) * 1000
            + probe_idx
        )
        random_blocks = torch.randperm(N, generator=generator, device="cpu")
        probe = expand_model_blocks_to_token_order(
            random_blocks.unsqueeze(0),
            BLOCK_LEN,
        ).to(device)

        aogpt_model.eval()
        _, _, attn_list = aogpt_model.forward_fn(
            idx_batch,
            probe,
            return_attentions=True,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        attn_l0 = attn_list[0].cpu().numpy()
        probe_np = probe.cpu().numpy()
        # Free all GPU attention tensors — only L0 was needed.
        del attn_list
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if probe_np.shape[0] == 1 and attn_l0.shape[0] > 1:
            probe_np = np.broadcast_to(
                probe_np,
                (attn_l0.shape[0], probe_np.shape[1]),
            )
        B_samples[probe_idx] = build_model_frame_strict65(attn_l0, probe_np)

    B_mean = B_samples.mean(axis=0)
    return torch.from_numpy(B_mean).float().to(device)


# ---------------------------------------------------------------------------
# Model-frame frozen g_beta order provider
# ---------------------------------------------------------------------------

class FrozenGBetaModelFrameProvider:
    """Label-free frozen g_beta: model-frame order provider.

    Extracts L0 all-head attention under random probe orders, runs the
    pretrained g_beta, and returns a **model-frame** token reveal order
    suitable for ``model.forward_fn(idx, token_order)``.

    Supports both:
    - ``L0DynamicGBeta`` (multi-head, 65-node strict65 input)
    - ``NodewiseReadout`` / ``FlattenReadout`` (single-head, 64-node input;
      heads are averaged before feeding to the model)

    Physical coordinates are NEVER used in this class.
    """

    def __init__(
        self,
        g_beta_ckpt: str,
        device: str = "cuda:0",
        seed: int = 0,
        batch_mean_probes: int = 4,
    ):
        dev = torch.device(device)
        if dev.type == "cuda" and not torch.cuda.is_available():
            dev = torch.device("cpu")
        self.device = dev

        # Load pretrained g_beta — detect checkpoint type
        state = torch.load(g_beta_ckpt, map_location="cpu", weights_only=False)
        cfg = state["config"]
        model_name = cfg.get("model_name", "?")

        if model_name in ("flatten", "nodewise"):
            # Single-head g_beta (64-node).  We extract all L0 heads via
            # strict65, strip the None-token, and mean over heads → (B,64,64).
            self.model = _build_from_config(cfg).to(dev)
            if "model" in state:
                self.model.load_state_dict(state["model"])
            elif "model_state_dict" in state:
                self.model.load_state_dict(state["model_state_dict"])
            self._multi_head_input = False
        else:
            # Multi-head g_beta (L0DynamicGBeta, 65-node).
            H = cfg.get("heads", 8)
            self.model = L0DynamicGBeta(
                heads=H, nodes=65,
                scorer_hidden=tuple(cfg.get("scorer_hidden", (256, 64))),
                gate_hidden=cfg.get("gate_hidden", 32),
            ).to(dev)
            self.model.load_state_dict(state["model_state_dict"])
            self._multi_head_input = True

        self.model.eval()
        self.seed = int(seed)
        self.batch_mean_probes = max(1, int(batch_mean_probes))
        self._cached_order: torch.Tensor | None = None
        self._cached_step: int | None = None

    @torch.no_grad()
    def model_frame_token_order(
        self,
        aogpt_model: torch.nn.Module,
        idx_batch: torch.Tensor,
        global_step: int,
    ) -> torch.Tensor:
        """Return a model-frame token reveal order (1, SEQ_LEN).

        Runs ``batch_mean_probes`` probe forward passes with different
        random probe orders, averages the resulting B matrices over probe
        forwards (matching the deployed g_beta input protocol),
        then runs g_beta to produce the model-block reveal order.

        Args:
            aogpt_model: the AO-GPT model (for attention extraction only).
            idx_batch: (1, SEQ_LEN) model-coordinate token indices.
            global_step: current training step (for probe seeding).

        Returns:
            token_order: (1, SEQ_LEN) long tensor, model-frame token order.
        """
        B_t = extract_probe_averaged_model_frame_strict65(
            aogpt_model,
            idx_batch,
            global_step=global_step,
            seed=self.seed,
            batch_mean_probes=self.batch_mean_probes,
            device=self.device,
        )

        if self._multi_head_input:
            # L0DynamicGBeta: feed strict65 directly (Bsz, H, 65, 65)
            scores, _aux = self.model(B_t, apply_head_dropout=False)  # (Bsz, 64)
        else:
            # NodewiseReadout / FlattenReadout: strip None-token (node 0),
            # mean over L0 heads → (Bsz, 64, 64), then feed.
            B_content = B_t[:, :, 1:, 1:]  # (Bsz, H, 64, 64) — remove None-token
            B_headed = B_content.mean(dim=1)  # (Bsz, 64, 64) — mean over L0 heads
            scores = self.model(B_headed)  # (Bsz, 64)

        # Model-block reveal order (descending scores).
        sigma_model = scores.argsort(dim=1, descending=True)  # (Bsz, 64)

        # Expand to token order: model block i → tokens [4i, 4i+1, 4i+2, 4i+3].
        token_order = expand_model_blocks_to_token_order(
            sigma_model, BLOCK_LEN,
        ).to(self.device)  # (Bsz, SEQ_LEN)

        return token_order


# ---------------------------------------------------------------------------
# Training-loop compatible wrapper (model-frame → physical via training loop)
# ---------------------------------------------------------------------------

class FrozenGBetaModelFrameBlockProvider:
    """Label-free model-frame order provider compatible with training loop.

    Uses ``FrozenGBetaModelFrameProvider`` internally (strict65 extraction +
    batch_mean over probes + multi-head g_beta).  Returns **model-frame block
    order** so that the training loop's existing ``model_blocks_to_physical_blocks``
    path handles the physical remapping — keeping inv_perm / clean_perm out of
    the method path.

    Must be used with ``--frozen-beta-none-mode model`` so the training loop
    knows to remap model→physical.
    """

    def __init__(
        self,
        g_beta_ckpt: str,
        batch_mean_probes: int = 4,
        refresh_every: int = 1,
        seed: int = 0,
        device: str = "cuda:0",
    ):
        self._provider = FrozenGBetaModelFrameProvider(
            g_beta_ckpt=g_beta_ckpt,
            device=device,
            seed=seed,
            batch_mean_probes=batch_mean_probes,
        )
        self.batch_mean_probes = max(1, int(batch_mean_probes))
        self.refresh_every = max(1, int(refresh_every))
        self.seed = int(seed)
        self.device = device
        self.none_mode = "model"  # model-frame provider always returns model-frame order
        self._sigma = None
        self._last_refresh = None

    @property
    def gbeta_module(self):
        """The underlying L0DynamicGBeta (for Stage-3 freeze/unfreeze + optimizer).

        FrozenGBetaModelFrameBlockProvider wraps FrozenGBetaModelFrameProvider as
        ``self._provider``, which builds the readout as ``.model``.
        """
        return self._provider.model

    @torch.no_grad()
    def physical_order(self, model, idx_batch, global_step):
        """Return **model-frame** block order (N,) int64.

        Named ``physical_order`` for compatibility with the training loop.
        The training loop must remap model→physical via
        ``model_blocks_to_physical_blocks(sigma, clean_perm)`` when
        ``--frozen-beta-none-mode model`` is set.
        """
        if (
            self._sigma is None
            or (int(global_step) - int(self._last_refresh)) >= self.refresh_every
        ):
            token_order = self._provider.model_frame_token_order(
                model, idx_batch, global_step,
            )
            self._sigma = (
                token_order[0, ::BLOCK_LEN] // BLOCK_LEN
            ).cpu()
            self._last_refresh = int(global_step)
        return self._sigma


# ---------------------------------------------------------------------------
# Baseline: random model-frame order
# ---------------------------------------------------------------------------

def random_model_frame_token_order(
    batch_size: int,
    seed: int,
    global_step: int,
    device: torch.device,
) -> torch.Tensor:
    """Random model-frame block order → token order.  Label-free baseline."""
    rows = []
    for b in range(batch_size):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed) * 100_000_000 + int(global_step) * 1000 + b)
        blocks = torch.randperm(N, generator=gen, device="cpu")
        rows.append(
            expand_model_blocks_to_token_order(blocks.unsqueeze(0), BLOCK_LEN)[0]
        )
    return torch.stack(rows).to(device)


# ---------------------------------------------------------------------------
# Oracle baselines (use physical coordinates — explicitly labeled)
# ---------------------------------------------------------------------------

def l2r_model_frame_token_order(
    clean_perm,
    device: torch.device,
) -> torch.Tensor:
    """Physical L2R → model-frame token order.  ORACLE BASELINE.

    Uses clean_perm to translate physical L2R into model-frame token order.
    This is a canonical baseline, NOT the label-free method.
    """
    # Physical L2R blocks: [0, 1, ..., 63]
    phys_blocks = torch.arange(N, dtype=torch.long)
    # → model blocks via block_perm
    model_blocks = clean_perm.block_perm_phys_to_model[phys_blocks]
    token_order = expand_model_blocks_to_token_order(
        model_blocks.unsqueeze(0), BLOCK_LEN,
    )
    return token_order.to(device)


def layout_path_model_frame_token_order(
    clean_perm,
    device: torch.device,
) -> torch.Tensor:
    """Layout path (= block_perm order) as token order.  ORACLE BASELINE.

    The layout path is block_perm_phys_to_model itself — the model-block
    order that yields physical L2R under this layout.
    """
    model_blocks = clean_perm.block_perm_phys_to_model.clone()
    token_order = expand_model_blocks_to_token_order(
        model_blocks.unsqueeze(0), BLOCK_LEN,
    )
    return token_order.to(device)
