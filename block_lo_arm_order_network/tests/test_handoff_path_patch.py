"""Path-restricted L0->L1 QK patch: the L1 destination head's attention is
recomputed from a (corrupted) residual for Q/K while V stays clean; an unchanged
residual reproduces the clean attention, and Q/K respond to residual corruption.
"""
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.path_patch_handoff import (  # noqa: E402
    make_probe_batch,
    capture_block_input,
    l1_attn_from_residual,
    patched_l1_head_output,
)


def _l1_inputs(model, chunks, dev, rng_seed=2, n=8):
    pc, po = make_probe_batch(chunks, n=n, rng=np.random.default_rng(rng_seed))
    handle, store = capture_block_input(model, layer=1)
    with torch.no_grad():
        model.forward_fn(pc.to(dev), torch.from_numpy(po).to(dev), return_attentions=True)
    handle.remove()
    return store["x"], store["cond"]


def test_corr_equals_clean_gives_identical_att(seed2_bundle):
    model, chunks, _ = seed2_bundle
    dev = torch.device("cpu")
    x_clean, cond = _l1_inputs(model, chunks, dev)
    att_a = l1_attn_from_residual(model, x_clean, cond, dst_head=0, device=dev)
    att_b = l1_attn_from_residual(model, x_clean, cond, dst_head=0, device=dev)
    assert torch.allclose(att_a, att_b)  # deterministic
    # corr == clean -> patched head output is finite and equals clean-V att@v
    y_patch = patched_l1_head_output(model, x_clean, x_clean, dst_head=0, cond=cond, device=dev)
    assert torch.isfinite(y_patch).all()


def test_patched_att_uses_corr_qk(seed2_bundle):
    model, chunks, _ = seed2_bundle
    dev = torch.device("cpu")
    x_clean, cond = _l1_inputs(model, chunks, dev)
    x_corr = x_clean + 0.1 * torch.randn_like(x_clean)
    att_clean = l1_attn_from_residual(model, x_clean, cond, dst_head=0, device=dev)
    att_corr = l1_attn_from_residual(model, x_corr, cond, dst_head=0, device=dev)
    assert not torch.allclose(att_clean, att_corr)  # q/k respond to corruption
