import sys, os
import torch

CHENHE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CHENHE)
from orderhead_v3.readouts import NodewiseReadout, FlattenReadout, build_readout
from orderhead_v3.head_selection import select_best_head
from AOGPT_block import AOGPT, AOGPTConfig


def test_readout_shapes():
    B = torch.rand(3, 64, 64)
    for m in [NodewiseReadout(N=64, d_model=64, n_layers=2, n_heads=4),
              FlattenReadout(N=64, hidden=(256, 64))]:
        m.eval()
        with torch.no_grad():
            s = m(B)
        assert s.shape == (3, 64) and torch.isfinite(s).all()
        sigma = s.argsort(dim=1, descending=True)
        assert all(sorted(sigma[i].tolist()) == list(range(64)) for i in range(3))


def test_build_readout_from_config():
    m = build_readout({"model_name": "nodewise", "N": 64, "d_model": 64,
                       "n_layers": 2, "n_heads": 4})
    assert isinstance(m, NodewiseReadout)


def test_head_selection_returns_valid_head():
    cfg = dict(block_size=256, vocab_size=50304, n_layer=2, n_head=8, n_embd=128,
               dropout=0.0, bias=True, block_order_block_len=4,
               block_order_layout="contiguous", position_encoding_mode="absolute")
    model = AOGPT(AOGPTConfig(**cfg)).eval()
    chunks = torch.randint(0, 50304, (16, 256))
    (layer, head), scores = select_best_head(model, chunks, n_reveal=4, seed=0, device="cpu")
    assert 0 <= layer < 2 and 0 <= head < 8            # valid (layer, head)
    assert len(scores) == 2 * 8                          # all layers x heads scored
    assert scores[0]["total"] >= scores[-1]["total"]    # sorted best-first
