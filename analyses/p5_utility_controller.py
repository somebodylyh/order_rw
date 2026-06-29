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
_ANALYSES = ROOT / "analyses"
for _p in (str(ROOT), str(_BLOCK), str(_ANALYSES)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from neural_readout.extract_b import _load_model_and_chunks  # noqa: E402
from clean_training_protocol import physical_blocks_to_model_token_order  # noqa: E402
from physical_signal_source import carrier_b65_per_text  # noqa: E402
from none_separated_block_graph import rollout_by_method  # noqa: E402

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


# ── Task 2: Per-sample scaffold ──────────────────────────────────────────────

def block_b_features(B65):
    """Per non-None block features: its row (out-edges) and column (in-edges) of B65."""
    B = np.asarray(B65, dtype=np.float64)
    rows = B[1:, :]                                  # (64, 65) outgoing
    cols = B[:, 1:].T                                # (64, 65) incoming
    return np.concatenate([rows, cols], axis=1).astype(np.float32)   # (64, 130)


def sigma_from_B65(B65):
    """C-D+L rollout order of a B65 -> physical-block order (64,).

    rollout_by_method returns the order ndarray DIRECTLY (P2 wraps it as
    discovery_metrics(rollout_by_method(...))); it is NOT a dict."""
    order = rollout_by_method(np.asarray(B65), "C-D+L")
    return np.asarray(order, dtype=np.int64)


def sample_scaffold(ckpt_path, M, layer=0, head=1, n_reveals=8, fixed_reveal_seed=0,
                    device="cpu"):
    """Per-text B65 (seed123 carrier L0H1), its C-D+L order sigma_B, and the chunks."""
    model, chunks, clean_perm, dev = load_p5_ckpt(ckpt_path, M, device=device)
    B_list, _tau = carrier_b65_per_text(
        ckpt_path, layer, head, M=M, n_reveals=n_reveals,
        fixed_reveal_seed=fixed_reveal_seed, device=device)
    sig = [sigma_from_B65(B) for B in B_list]
    return {"B": B_list, "sigma_B": sig, "chunks": chunks,
            "clean_perm": clean_perm, "model": model, "dev": dev}
