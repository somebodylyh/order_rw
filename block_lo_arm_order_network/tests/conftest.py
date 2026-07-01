"""Shared fixtures for Pillar-3 path-patching tests.

Loading the AOGPT model + eval chunks is expensive, so the seed2 step-10000
checkpoint is loaded once per session on CPU and shared across test modules.
"""
import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # import analyses.* as a namespace package

SEED2_CKPT = str(ROOT / "runs/handoff_overnight/seed2/ckpt_step10000.pt")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: opt-in tests needing the 10k ckpt / GPU (run with -m slow)"
    )


@pytest.fixture(scope="session")
def seed2_bundle():
    """(model, chunks, seed) for seed2 step-10000 on CPU."""
    from analyses.path_patch_handoff import load_model_and_chunks_seed

    dev = torch.device("cpu")
    model, chunks, seed = load_model_and_chunks_seed(SEED2_CKPT, total=64, device=dev)
    return model, chunks, seed
