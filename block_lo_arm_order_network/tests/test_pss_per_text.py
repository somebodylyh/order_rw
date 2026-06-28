import inspect
import pathlib, sys
import numpy as np
import pytest
import torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import analyses.physical_signal_source as pss
from analyses.physical_signal_source import carrier_b65_per_text

CKPT = ROOT / "runs/handoff_overnight/seed2/ckpt_step10000.pt"


def test_per_text_b65_public_signature():
    params = list(inspect.signature(carrier_b65_per_text).parameters.values())
    assert [param.name for param in params] == [
        "ckpt_path",
        "layer",
        "head",
        "M",
        "n_reveals",
        "fixed_reveal_seed",
        "device",
        "return_halves",
    ]
    assert [param.default for param in params[3:]] == [
        24,
        8,
        0,
        "cpu",
        False,
    ]


@pytest.mark.parametrize(
    ("n_reveals", "return_halves", "message"),
    [
        (0, False, "n_reveals must be at least 1"),
        (1, True, "n_reveals must be at least 2 when return_halves=True"),
    ],
)
def test_n_reveals_is_validated_before_checkpoint_load(
        monkeypatch, n_reveals, return_halves, message):
    def fail_if_loaded(*args, **kwargs):
        raise AssertionError("checkpoint loader must not run for invalid n_reveals")

    monkeypatch.setattr(pss, "_load_model_and_chunks", fail_if_loaded)
    with pytest.raises(ValueError, match=message):
        carrier_b65_per_text(
            "missing.pt", layer=0, head=0, M=2, n_reveals=n_reveals,
            return_halves=return_halves)


def test_shared_reveal_orders_are_reused_for_each_text(monkeypatch):
    reveals = np.array([[0, 1, 2], [2, 0, 1], [1, 2, 0]], dtype=np.int64)
    calls = []
    reveal_requests = []

    class RecordingModel:
        def forward_fn(self, chunk, prediction_order, return_attentions):
            calls.append((int(chunk.item()), prediction_order.cpu().numpy()[0].copy()))
            return None, None, [torch.zeros((1, 1, 1, 1))]

    class CleanPerm:
        inv_perm_model_to_phys = torch.arange(3)

    monkeypatch.setattr(
        pss, "_load_model_and_chunks",
        lambda *args, **kwargs: (
            RecordingModel(), torch.arange(2).reshape(2, 1), CleanPerm(),
            torch.device("cpu"), None))

    def fixed_reveals(count, seed):
        reveal_requests.append((count, seed))
        return reveals.copy()

    monkeypatch.setattr(pss, "random_reveal_orders", fixed_reveals)
    monkeypatch.setattr(
        pss, "_attn_to_A_block_loss_aligned_with_none_vec",
        lambda attn, reveal, inv: np.zeros((1, 1, 64, 65)))
    monkeypatch.setattr(pss, "build_none_separated_B", lambda A: np.zeros((65, 65)))
    monkeypatch.setattr(pss, "rollout_by_method", lambda B, method: np.arange(64))
    monkeypatch.setattr(pss, "discovery_metrics", lambda order: {"tau_vs_l2r": 0.0})

    carrier_b65_per_text("unused.pt", layer=0, head=0, M=2, n_reveals=3)

    assert reveal_requests == [(3, 0)]
    for text_index in range(2):
        observed = [reveal for text, reveal in calls if text == text_index]
        np.testing.assert_array_equal(observed, reveals)


@pytest.mark.skipif(not CKPT.exists(), reason="optional real-checkpoint integration test")
def test_per_text_b65_shapes_and_validity():
    B_list, tau_list = carrier_b65_per_text(
        str(CKPT), layer=0, head=2, M=6, n_reveals=8)
    assert len(B_list) == 6 and B_list[0].shape == (65, 65)
    assert len(tau_list) == 6
    # carrier head: most texts read high physical tau
    assert np.mean([abs(t) > 0.9 for t in tau_list]) >= 0.5
