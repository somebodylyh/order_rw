"""P5: attention-scaffolded utility controller (Phase 0).

H improves downstream reveal-order utility beyond the attention scaffold; it does
NOT recover physical order. Supervision = downstream teacher-forced AO NLL (never
physical-rank/CDL(B), which structurally ignore H under fixed layout).
See docs/superpowers/specs/2026-06-29-p5-attention-scaffolded-utility-controller-design.md.
"""
import pathlib, sys
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
_BLOCK = ROOT / "block_lo_arm_order_network"
for _p in (str(ROOT), str(_BLOCK)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from neural_readout.extract_b import _load_model_and_chunks  # noqa: E402
from clean_training_protocol import physical_blocks_to_model_token_order  # noqa: E402

SEQ_LEN, N, BLOCK_LEN = 256, 64, 4


def load_p5_ckpt(ckpt_path, M, device="cpu"):
    model, chunks, clean_perm, dev, _ = _load_model_and_chunks(
        ckpt_path, M, seed=0, device=device, split="train")
    return model, chunks, clean_perm, dev


@torch.no_grad()
def order_nll(model, idx_row, sigma_phys, clean_perm, device):
    """Downstream teacher-forced AO NLL of revealing blocks in physical-block order
    sigma_phys (64,). Lower = better utility."""
    sigma = np.asarray(sigma_phys, dtype=np.int64)[None, :]            # (1,64)
    token_order = physical_blocks_to_model_token_order(
        torch.from_numpy(sigma), clean_perm, BLOCK_LEN).to(device)
    _, loss = model.forward_fn(idx_row.to(device), token_order)
    return float(loss)


def utility_pool(model, idx_row, sigmas, clean_perm, device):
    return [order_nll(model, idx_row, s, clean_perm, device) for s in sigmas]
