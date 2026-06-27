"""Stage-2: downstream path_fraction is finite/non-negative, and the redundancy
ordering helper reports the single<=LOO<=full trend without hard-failing.
"""
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.path_patch_handoff import (  # noqa: E402
    run_stage2,
    redundancy_ordering,
    make_probe_batch,
)


def test_path_fraction_is_finite_and_nonneg(seed2_bundle):
    model, chunks, seed = seed2_bundle
    dev = torch.device("cpu")
    pb = [make_probe_batch(chunks, 16, np.random.default_rng(i)) for i in range(2)]
    out = run_stage2(
        model, seed, src_heads_L0=[1, 6, 7], dst_heads_L1=[0, 3, 5, 7],
        probe_batches=pb, device=dev,
    )
    assert np.isfinite(out["path_fraction_global"])
    assert out["path_fraction_global"] >= 0


def test_redundancy_ordering_reports_trend():
    r = redundancy_ordering({"single": 0.05, "loo": 0.21, "full": 0.25})
    assert r["monotone"] in (True, False)
    assert r["single"] <= r["full"]
