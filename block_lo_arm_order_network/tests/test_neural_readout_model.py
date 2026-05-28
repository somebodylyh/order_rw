"""NR-1 Task 6: tests for GraphTransformerReadout (g_beta).

Pins:
  - shape: (batch, 64, 64) -> (batch, 64)
  - no node positional embedding registered (matches §3 framing)
  - parameter count in expected order of magnitude (~50-80K)
  - input shape mismatch raises
"""
import sys
import pathlib

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))


def test_model_shape():
    from neural_readout.graph_transformer_readout import GraphTransformerReadout
    g = GraphTransformerReadout(N=64, d_model=64, n_heads=4, n_layers=2)
    B = torch.randn(7, 64, 64)
    s = g(B)
    assert s.shape == (7, 64), s.shape


def test_no_positional_embedding_registered():
    """No parameter or buffer named 'pos*' (case-insensitive)."""
    from neural_readout.graph_transformer_readout import GraphTransformerReadout
    g = GraphTransformerReadout(N=64, d_model=64, n_heads=4, n_layers=2)
    for name, _ in g.named_parameters():
        assert "pos" not in name.lower(), f"unexpected positional param: {name}"
    for name, _ in g.named_buffers():
        assert "pos" not in name.lower(), f"unexpected positional buffer: {name}"


def test_param_count_in_range():
    from neural_readout.graph_transformer_readout import GraphTransformerReadout
    g = GraphTransformerReadout(N=64, d_model=64, n_heads=4, n_layers=2)
    n = sum(p.numel() for p in g.parameters())
    # Order of magnitude check; the spec says ~50-80K, allow some slack for
    # framework-specific bias terms / standard TransformerEncoder defaults.
    assert 30_000 <= n <= 200_000, f"param count out of expected range: {n}"


def test_bad_shape_raises():
    from neural_readout.graph_transformer_readout import GraphTransformerReadout
    import pytest
    g = GraphTransformerReadout(N=64)
    with pytest.raises(ValueError, match=r"expected \(batch, 64, 64\)"):
        g(torch.randn(2, 32, 32))
