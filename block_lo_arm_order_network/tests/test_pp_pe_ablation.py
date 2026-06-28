"""Reversible wpe/wtpe ablation context manager."""
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.position_prior_decomp import pe_ablation  # noqa: E402
from analyses.path_patch_handoff import (  # noqa: E402
    load_model_and_chunks_seed, make_probe_batch, run_clean)

CKPT0 = str(ROOT / "runs/handoff_overnight/seed2/ckpt_step0.pt")


def test_none_is_bit_identical():
    dev = torch.device("cpu")
    model, chunks, _ = load_model_and_chunks_seed(CKPT0, 64, dev)
    pc, po = make_probe_batch(chunks, 16, np.random.default_rng(0))
    _, tau_plain = run_clean(model, pc, po, dev)
    with pe_ablation(model, "none"):
        _, tau_none = run_clean(model, pc, po, dev)
    assert np.array_equal(tau_plain, tau_none)
    _, tau_after = run_clean(model, pc, po, dev)
    assert np.array_equal(tau_plain, tau_after)  # hooks removed on exit


def test_zero_both_changes_tau():
    dev = torch.device("cpu")
    model, chunks, _ = load_model_and_chunks_seed(CKPT0, 64, dev)
    pc, po = make_probe_batch(chunks, 16, np.random.default_rng(0))
    _, tau_full = run_clean(model, pc, po, dev)
    with pe_ablation(model, "both"):
        _, tau_zero = run_clean(model, pc, po, dev)
    assert not np.array_equal(tau_full, tau_zero)
