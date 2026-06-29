import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aogpt_ych import AOGPT, AOGPTConfig


def test_return_all_attentions_exports_probe_data_for_each_layer():
    model = AOGPT(
        AOGPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=3,
            n_head=2,
            n_embd=8,
            dropout=0.0,
            bias=True,
        )
    )
    inputs = torch.randint(0, 32, (1, 8))
    orders = torch.arange(8).unsqueeze(0)

    logits, loss, probe_data = model(
        inputs,
        mode=None,
        orders=orders,
        return_probe_data=True,
        return_all_attentions=True,
    )

    assert logits.shape == (1, 9, 32)
    assert loss.ndim == 0
    assert len(probe_data["all_attentions"]) == 3
    for layer_attn in probe_data["all_attentions"]:
        assert layer_attn.shape == (1, 2, 9, 9)
        assert torch.isfinite(layer_attn).all()
