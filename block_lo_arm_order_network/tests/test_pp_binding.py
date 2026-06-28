"""Part-2 binding scores (tau_pos / tau_content) + anchor-validity gate."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.position_prior_decomp import binding_scores, make_layouts  # noqa: E402
from clean_training_protocol import build_clean_block_permutation  # noqa: E402


def test_binding_scores_keys_and_gate():
    train = build_clean_block_permutation(64, seed=7)
    layouts = make_layouts(train, K=4, seed_base=10)
    out = binding_scores(2, 10000, layouts, winning_layer=1, carrier_heads=[3, 5, 7],
                         tier="strong", root=str(ROOT / "runs/handoff_overnight"),
                         n_batches=1)
    for k in ("anchor_tau_pos", "relayout_mean_pos", "relayout_mean_content",
              "anchor_valid", "verdict"):
        assert k in out
    assert out["verdict"] in ("slot-scaffold", "content-bound", "ood-break", "invalid")
