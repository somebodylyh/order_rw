"""Tests for Route A Transformer Order Network."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from order_network import RouteATransformerOrderNetwork


def test_route_a_transformer_forward_masks_visited_nodes():
    model = RouteATransformerOrderNetwork(
        num_blocks=16,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.0,
    )
    visited_masks = torch.tensor([0b0000000000000011, 0b0000000000010100])
    last_nodes = torch.tensor([1, 4])

    logits = model(visited_masks, last_nodes)

    assert logits.shape == (2, 16)
    assert torch.isneginf(logits[0, 0])
    assert torch.isneginf(logits[0, 1])
    assert torch.isfinite(logits[0, 2:]).all()
    assert torch.isneginf(logits[1, 2])
    assert torch.isneginf(logits[1, 4])


def test_route_a_transformer_default_is_four_layer_four_head_256d():
    model = RouteATransformerOrderNetwork()

    assert model.num_blocks == 16
    assert model.d_model == 256
    assert len(model.transformer.layers) == 4
    assert model.transformer.layers[0].self_attn.num_heads == 4
    assert model.count_parameters() > 8_000_000
