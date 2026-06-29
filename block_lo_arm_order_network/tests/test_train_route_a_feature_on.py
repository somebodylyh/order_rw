"""Tests for A-conditioned Route A feature ON trainer."""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_route_a_feature_on import RouteAFeatureDataset


def test_route_a_feature_dataset_returns_a_matrix_and_state():
    examples = {
        "seq_indices": np.asarray([1], dtype=np.int32),
        "visited_masks": np.asarray([0b101], dtype=np.uint32),
        "last_nodes": np.asarray([2], dtype=np.int16),
        "next_nodes": np.asarray([3], dtype=np.int16),
    }
    A = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    dataset = RouteAFeatureDataset(examples, A)

    A_item, visited_mask, last_node, next_node = dataset[0]

    assert A_item.shape == (4, 4)
    assert A_item.dtype == torch.float32
    assert A_item[0, 0].item() == 16.0
    assert visited_mask.item() == 0b101
    assert last_node.item() == 2
    assert next_node.item() == 3
