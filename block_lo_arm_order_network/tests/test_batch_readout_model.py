"""BR-1 Task 5: tests for FlattenReadout + NodewiseReadout."""
import sys
import pathlib

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))


def test_flatten_shape_and_finite():
    from batch_readout.model import FlattenReadout
    m = FlattenReadout(N=64, hidden=(256, 64))
    B = torch.randn(3, 64, 64)
    z = m(B)
    assert z.shape == (3, 64)
    assert torch.isfinite(z).all()


def test_flatten_rejects_wrong_N():
    from batch_readout.model import FlattenReadout
    m = FlattenReadout(N=64, hidden=(256, 64))
    with pytest.raises(ValueError, match="expected B with last two dims"):
        m(torch.randn(3, 32, 32))


def test_nodewise_shape_and_finite():
    from batch_readout.model import NodewiseReadout
    m = NodewiseReadout(N=64, d_model=64, n_layers=2, n_heads=4)
    B = torch.randn(3, 64, 64)
    z = m(B)
    assert z.shape == (3, 64)
    assert torch.isfinite(z).all()


def test_nodewise_has_no_learnable_positional_embedding():
    """Check by inspection: no nn.Embedding or registered positional-embedding
    buffer/parameter sits in the model. Slot identity comes only from the
    [B[v,:], B[:,v]] feature ordering, NOT from a learned PE added to inputs."""
    from batch_readout.model import NodewiseReadout
    m = NodewiseReadout(N=64, d_model=64, n_layers=2, n_heads=4)
    for mod in m.modules():
        assert not isinstance(mod, torch.nn.Embedding), (
            f"unexpected nn.Embedding in NodewiseReadout: {mod}"
        )
    pe_like = [n for n, _ in m.named_parameters() if "pos" in n.lower() or "pe" in n.lower()]
    assert pe_like == [], f"unexpected positional-embedding-like params: {pe_like}"
    pe_buf = [n for n, _ in m.named_buffers() if "pos" in n.lower() or "pe" in n.lower()]
    assert pe_buf == [], f"unexpected positional-embedding-like buffers: {pe_buf}"


def test_nodewise_param_count_reasonable():
    """Sanity-check the param count so an accidental over-paramatrization fails loud."""
    from batch_readout.model import NodewiseReadout
    m = NodewiseReadout(N=64, d_model=64, n_layers=2, n_heads=4)
    p = sum(t.numel() for t in m.parameters())
    # 2-layer encoder + node_in + head with d_model=64; expect well under 1M params.
    assert 50_000 < p < 1_000_000, p


def test_flatten_is_position_sensitive():
    """FlattenReadout encodes slot identity directly through vec(B) -> MLP weights.
    Permuting B should NOT yield a permuted output (would mean the model ignored
    its input layout — a bug)."""
    from batch_readout.model import FlattenReadout
    torch.manual_seed(0)
    m = FlattenReadout(N=64, hidden=(64,)).eval()
    B = torch.randn(1, 64, 64)
    perm = torch.randperm(64)
    B_perm = B[:, perm][:, :, perm]
    with torch.no_grad():
        z1 = m(B)[0]
        z2 = m(B_perm)[0]
    assert not torch.allclose(z1[perm], z2, atol=1e-3)
