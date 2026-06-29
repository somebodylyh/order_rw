"""Tests for Route A ON supervised training helpers."""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_route_a_on import (
    RouteAStateDataset,
    evaluate_next_step_accuracy,
    resolve_device,
)


def test_route_a_state_dataset_returns_tensors():
    data = {
        "visited_masks": np.asarray([1, 3], dtype=np.uint32),
        "last_nodes": np.asarray([0, 1], dtype=np.int16),
        "next_nodes": np.asarray([1, 2], dtype=np.int16),
    }
    dataset = RouteAStateDataset(data)

    visited_mask, last_node, next_node = dataset[1]

    assert len(dataset) == 2
    assert visited_mask.dtype == torch.long
    assert last_node.dtype == torch.long
    assert next_node.dtype == torch.long
    assert visited_mask.item() == 3
    assert last_node.item() == 1
    assert next_node.item() == 2


def test_evaluate_next_step_accuracy_uses_argmax_predictions():
    class FixedModel(torch.nn.Module):
        def forward(self, visited_masks, last_nodes):
            logits = torch.zeros(visited_masks.shape[0], 4)
            logits[:, 2] = 3.0
            logits[:, 1] = 1.0
            return logits

    data = {
        "visited_masks": np.asarray([1, 1, 1], dtype=np.uint32),
        "last_nodes": np.asarray([0, 0, 0], dtype=np.int16),
        "next_nodes": np.asarray([2, 1, 2], dtype=np.int16),
    }
    dataset = RouteAStateDataset(data)

    metrics = evaluate_next_step_accuracy(
        FixedModel(),
        dataset,
        batch_size=2,
        device=torch.device("cpu"),
    )

    assert metrics["accuracy"] == 2 / 3
    assert metrics["num_examples"] == 3


def test_resolve_device_falls_back_when_cuda_unavailable():
    assert resolve_device("cuda", cuda_available=False).type == "cpu"
    assert resolve_device("cpu", cuda_available=False).type == "cpu"
