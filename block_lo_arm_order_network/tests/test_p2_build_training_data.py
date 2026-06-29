"""Tests for Route A ON training data builder."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p2_build_training_data import build_training_examples


def test_build_training_examples_from_paths():
    paths = np.asarray([
        [2, 0, 3, 1],
        [1, 3, 0, 2],
    ], dtype=np.int16)

    examples = build_training_examples(paths, num_blocks=4)

    assert examples["visited_masks"].tolist() == [
        0b0100,
        0b0101,
        0b1101,
        0b0010,
        0b1010,
        0b1011,
    ]
    assert examples["last_nodes"].tolist() == [2, 0, 3, 1, 3, 0]
    assert examples["next_nodes"].tolist() == [0, 3, 1, 3, 0, 2]
    assert examples["seq_indices"].tolist() == [0, 0, 0, 1, 1, 1]
    assert examples["step_indices"].tolist() == [0, 1, 2, 0, 1, 2]
