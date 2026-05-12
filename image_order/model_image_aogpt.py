"""
ImageAOGPT: AO-GPT variant for continuous image patches (MSE loss).

Replaces discrete token embedding / cross-entropy with:
  - patch_proj: Linear(patch_dim, n_embd)   — input projection
  - patch_head: Linear(n_embd, patch_dim)   — output projection
  - MSE loss between predicted and target (shuffled) patches

Everything else — [None] prefix, wpe, wtpe, AdaLN Block, FinalLayer,
and the reveal-order mechanism — is identical to the text model so that
the existing Graph-RW order machinery is plug-compatible.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "block_lo_arm_order_network"))

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from model_AOGPT_AdaLN6_NoRep_cond_128_trunc_qknorm import Block, FinalLayer


@dataclass
class ImageAOGPTConfig:
    n_patches: int = 64
    patch_dim: int = 48
    n_embd: int = 256
    n_layer: int = 4
    n_head: int = 8
    cond_dim: int = 128   # must equal the imported Block's AdaLN cond width
    dropout: float = 0.0
    bias: bool = True


@dataclass
class _BlockCfgShim:
    """Minimal config object the imported Block/FinalLayer constructors read."""
    n_embd: int
    n_head: int
    dropout: float
    bias: bool
    block_size: int   # = n_patches; CausalSelfAttention uses block_size+1 for causal mask


class ImageAOGPT(nn.Module):

    def __init__(self, config: ImageAOGPTConfig):
        super().__init__()
        self.config = config

        shim = _BlockCfgShim(
            n_embd=config.n_embd,
            n_head=config.n_head,
            dropout=config.dropout,
            bias=config.bias,
            block_size=config.n_patches,
        )

        self.patch_proj = nn.Linear(config.patch_dim, config.n_embd, bias=config.bias)
        self.wpe = nn.Embedding(config.n_patches + 1, config.n_embd)
        self.wtpe = nn.Embedding(config.n_patches, config.cond_dim)
        self.wnonee = nn.Embedding(1, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([Block(shim) for _ in range(config.n_layer)])
        self.final_layer = FinalLayer(shim)
        self.patch_head = nn.Linear(config.n_embd, config.patch_dim, bias=config.bias)

        # Init all weights
        self.apply(self._init_weights)
        # Scaled init for residual projections (c_proj), per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                std = 0.02 / math.sqrt(2 * config.n_layer)
                torch.nn.init.trunc_normal_(p, mean=0.0, std=std, a=-3 * std, b=3 * std)

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02,
                                        a=-3 * 0.02, b=3 * 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02,
                                        a=-3 * 0.02, b=3 * 0.02)

    # ------------------------------------------------------------------
    # Order helpers
    # ------------------------------------------------------------------

    def shuffle(self, x: torch.Tensor, orders: torch.LongTensor) -> torch.Tensor:
        """Gather x[b, orders[b, t], :] per batch."""
        batch_size, seq_len = x.shape[:2]
        batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1).expand(-1, seq_len)
        return x[batch_indices, orders]

    def unshuffle(self, shuffled: torch.Tensor, orders: torch.LongTensor) -> torch.Tensor:
        """Inverse scatter: place shuffled[b, t] back at position orders[b, t]."""
        batch_size, seq_len = shuffled.shape[:2]
        batch_indices = torch.arange(batch_size, device=shuffled.device).unsqueeze(1).expand(-1, seq_len)
        out = torch.zeros_like(shuffled)
        out[batch_indices, orders] = shuffled
        return out

    def sample_random_orders(self, x: torch.Tensor) -> torch.LongTensor:
        """Per-sample random permutation of n_patches."""
        B, N = x.shape[0], x.shape[1]
        return torch.stack([torch.randperm(N, device=x.device) for _ in range(B)])

    def set_ascending_orders(self, x: torch.Tensor) -> torch.LongTensor:
        """Per-sample raster-scan (ascending) order."""
        B, N = x.shape[0], x.shape[1]
        return torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def forward_fn(self, patches: torch.Tensor, orders: torch.LongTensor,
                   return_attentions: bool = False):
        """
        patches : (B, N, patch_dim)
        orders  : (B, N) long
        """
        device = patches.device
        B, N, D = patches.shape
        assert N == self.config.n_patches, \
            f"Expected {self.config.n_patches} patches, got {N}"

        # 1. Shuffle patches according to reveal order; shuffled patches are also the targets
        x_shuf = self.shuffle(patches, orders)      # (B, N, D)
        targets = x_shuf                             # same tensor, used as MSE target

        # 2. Project patches to embedding space
        tok_emb = self.patch_proj(x_shuf)           # (B, N, n_embd)

        # 3. Prepend [None] token
        none_emb = self.wnonee(torch.zeros(B, 1, dtype=torch.long, device=device))
        # none_emb: (B, 1, n_embd)
        tok_emb = torch.cat([none_emb, tok_emb], dim=1)   # (B, N+1, n_embd)

        # 4. Position embedding (mirrors text model lines 291-312)
        pos = torch.arange(0, N + 1, dtype=torch.long, device=device)   # (N+1,)
        pos_emb = self.wpe(pos).unsqueeze(0).expand(B, -1, -1)           # (B, N+1, n_embd)
        pos_emb_prefix = pos_emb[:, :1]                                  # (B, 1, n_embd)  [None] pos
        pos_emb_postfix = self.shuffle(pos_emb[:, 1:], orders)           # (B, N, n_embd)  shuffled patch pos

        x = tok_emb + torch.cat([pos_emb_prefix, pos_emb_postfix], dim=1)

        # 5. Target-position conditioning c : (B, N+1, cond_dim)
        target_pos_emb = self.wtpe(pos[:N])                              # (N, cond_dim)
        target_pos_emb = target_pos_emb.unsqueeze(0).expand(B, -1, -1)  # (B, N, cond_dim)
        target_pos_emb_prefix = self.shuffle(target_pos_emb, orders)     # (B, N, cond_dim)  shuffled
        target_pos_emb_postfix = torch.zeros(B, 1, self.config.cond_dim, device=device)  # (B, 1, cond_dim)
        c = torch.cat([target_pos_emb_prefix, target_pos_emb_postfix], dim=1)  # (B, N+1, cond_dim)

        # 6. Transformer blocks
        x = self.drop(x)
        attn_outputs = [] if return_attentions else None
        for blk in self.h:
            if return_attentions:
                x, attn_probs = blk(x, c, return_attn=True)
                attn_outputs.append(attn_probs)
            else:
                x = blk(x, c)

        # 7. Final layer
        x = self.final_layer(x, c)

        # 8. Project to patch space
        pred_patches = self.patch_head(x)            # (B, N+1, patch_dim)

        # 9. Align prediction with shifted target (same shift as text model)
        pred = pred_patches[:, :-1, :]              # (B, N, patch_dim)  — drop last step
        loss = F.mse_loss(pred, targets)

        if return_attentions:
            return pred, loss, attn_outputs
        return pred, loss

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------

    def forward(self, patches: torch.Tensor, mode: str = 'Random',
                orders: torch.LongTensor = None, return_attentions: bool = False):
        """
        patches : (B, N, patch_dim)
        mode    : 'Random' | 'AR' | None
        orders  : required when mode is None, shape (B, N)
        """
        if mode == 'Random':
            orders = self.sample_random_orders(patches)
        elif mode == 'AR':
            orders = self.set_ascending_orders(patches)
        elif mode is None:
            assert orders is not None, "mode=None requires explicit orders"
            assert orders.shape == patches.shape[:2], \
                f"orders shape {orders.shape} must match (B, N)={patches.shape[:2]}"
        else:
            raise ValueError(f"Unknown mode '{mode}'; expected 'Random', 'AR', or None")

        return self.forward_fn(patches, orders, return_attentions=return_attentions)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        num_decay = sum(p.numel() for p in decay_params)
        num_nodecay = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------

if __name__ == '__main__':
    cfg = ImageAOGPTConfig()
    model = ImageAOGPT(cfg)
    param_count = model.get_num_params()

    B, N, D = 4, 64, 48
    patches = torch.randn(B, N, D)

    # Random mode
    pred_r, loss_r = model(patches, mode='Random')
    assert loss_r.ndim == 0 and loss_r.item() == loss_r.item(), "loss_random is not a finite scalar"

    # Fixed order mode
    fixed_orders = torch.stack([torch.randperm(N) for _ in range(B)])
    pred_f, loss_f = model(patches, mode=None, orders=fixed_orders)
    assert loss_f.ndim == 0 and loss_f.item() == loss_f.item(), "loss_fixed is not a finite scalar"

    # Backward pass
    loss_f.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Param {name} has no gradient"

    # Return attentions
    pred_a, loss_a, attn_outputs = model(patches, mode='Random', return_attentions=True)
    assert len(attn_outputs) == cfg.n_layer, \
        f"Expected {cfg.n_layer} attention outputs, got {len(attn_outputs)}"
    expected_attn_shape = (B, cfg.n_head, N + 1, N + 1)
    for i, a in enumerate(attn_outputs):
        assert tuple(a.shape) == expected_attn_shape, \
            f"Layer {i} attn shape {tuple(a.shape)} != {expected_attn_shape}"

    print(
        f"OK model_image_aogpt: params={param_count}, "
        f"loss_random={loss_r.item():.6f}, "
        f"loss_fixed={loss_f.item():.6f}, "
        f"attn_shape={tuple(attn_outputs[0].shape)}"
    )
