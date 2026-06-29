"""Tests for non-learned model-frame strict65 order policies."""

import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))

from batch_readout.direct_order_provider import (  # noqa: E402
    DirectModelFrameOrderProvider,
    direct_policy_scores,
    initial_cdl_scores_strict65,
    readiness_scores_strict65,
    source_mass_scores_strict65,
)


def _empty_B(batch=1, heads=1):
    return torch.zeros(batch, heads, 65, 65, dtype=torch.float32)


def test_initial_cdl_score_shape_and_mean_head_fusion():
    B = _empty_B(batch=2, heads=2)
    B[:, 0, 0, 1] = 2.0
    B[:, 1, 0, 1] = 4.0

    scores_h, scores = initial_cdl_scores_strict65(B)

    assert scores_h.shape == (2, 2, 64)
    assert scores.shape == (2, 64)
    torch.testing.assert_close(scores[:, 0], torch.full((2,), 6.0))


def test_initial_cdl_uses_column_dependency_and_excludes_diagonal():
    B = _empty_B()
    B[0, 0, 0, 1] = 10.0
    B[0, 0, 2, 1] = 63.0
    B[0, 0, 1, 1] = 999.0

    scores_h, _ = initial_cdl_scores_strict65(B)

    assert scores_h[0, 0, 0].item() == pytest.approx(19.0)


def test_source_mass_uses_rows_and_larger_score_is_earlier():
    B = _empty_B()
    B[0, 0, 1, 3] = 5.0
    B[0, 0, 2, 1] = 2.0

    _, scores = source_mass_scores_strict65(B)
    order = scores.argsort(dim=-1, descending=True)

    assert scores[0, 0].item() == pytest.approx(5.0)
    assert scores[0, 1].item() == pytest.approx(2.0)
    assert order[0, 0].item() == 0


def test_source_mass_excludes_nonzero_diagonal():
    B = _empty_B()
    B[0, 0, 1, 1] = 100.0
    B[0, 0, 1, 2] = 3.0

    scores_h, _ = source_mass_scores_strict65(B)

    assert scores_h[0, 0, 0].item() == pytest.approx(3.0)


def test_readiness_is_row_mass_minus_scaled_column_mass():
    B = _empty_B()
    B[0, 0, 1, 2] = 5.0
    B[0, 0, 3, 1] = 2.0

    scores_h, _ = readiness_scores_strict65(B, lambda_dep=0.5)

    assert scores_h[0, 0, 0].item() == pytest.approx(4.0)
    assert scores_h[0, 0, 1].item() == pytest.approx(-2.5)


@pytest.mark.parametrize(
    "policy",
    ["initial_cdl_one_shot", "source_mass", "readiness"],
)
def test_dispatch_returns_scores_for_all_direct_policies(policy):
    scores_h, scores = direct_policy_scores(
        _empty_B(batch=2, heads=3),
        policy=policy,
        lambda_dep=0.75,
    )
    assert scores_h.shape == (2, 3, 64)
    assert scores.shape == (2, 64)


def test_direct_policy_rejects_non_strict65_shape():
    with pytest.raises(ValueError, match="65, 65"):
        direct_policy_scores(torch.zeros(2, 8, 64, 64), "source_mass")


def test_direct_policy_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown direct policy"):
        direct_policy_scores(_empty_B(), "not_a_policy")


def test_provider_returns_model_frame_order_with_larger_score_first(monkeypatch):
    B = _empty_B(batch=2, heads=1)
    B[:, :, 1, 2] = 7.0

    calls = []

    def fake_extract(*args, **kwargs):
        calls.append(kwargs["global_step"])
        return B

    monkeypatch.setattr(
        "batch_readout.direct_order_provider."
        "extract_probe_averaged_model_frame_strict65",
        fake_extract,
    )
    provider = DirectModelFrameOrderProvider(
        policy="source_mass",
        batch_mean_probes=4,
        refresh_every=10,
        seed=3,
        device="cpu",
    )

    order = provider.physical_order(object(), torch.zeros(2, 256), 20)

    assert order.shape == (64,)
    assert order.dtype == torch.int64
    assert order[0].item() == 0
    assert calls == [20]


def test_provider_reuses_cached_order_until_refresh(monkeypatch):
    calls = []

    def fake_extract(*args, **kwargs):
        calls.append(kwargs["global_step"])
        return _empty_B()

    monkeypatch.setattr(
        "batch_readout.direct_order_provider."
        "extract_probe_averaged_model_frame_strict65",
        fake_extract,
    )
    provider = DirectModelFrameOrderProvider(
        policy="initial_cdl_one_shot",
        refresh_every=5,
        device="cpu",
    )

    first = provider.physical_order(object(), torch.zeros(1, 256), 10)
    cached = provider.physical_order(object(), torch.zeros(1, 256), 14)
    refreshed = provider.physical_order(object(), torch.zeros(1, 256), 15)

    torch.testing.assert_close(first, cached)
    torch.testing.assert_close(first, refreshed)
    assert calls == [10, 15]


def test_provider_constructor_and_source_are_label_free():
    import inspect

    params = set(inspect.signature(DirectModelFrameOrderProvider.__init__).parameters)
    source = inspect.getsource(DirectModelFrameOrderProvider)
    forbidden = {"inv_perm", "clean_perm", "block_perm"}
    assert not params & forbidden
    for term in forbidden:
        assert term not in source
