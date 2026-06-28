"""Layout generator: training anchor + fixed-random relayouts."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.position_prior_decomp import make_layouts, save_layouts, load_layouts  # noqa: E402
from clean_training_protocol import build_clean_block_permutation  # noqa: E402


def test_layouts_anchor_and_count(tmp_path):
    train = build_clean_block_permutation(64, seed=7)
    layouts = make_layouts(train, K=8, seed_base=1000)
    assert len(layouts) == 8
    assert layouts[0]["is_training_layout"] is True
    assert all(l["is_training_layout"] is False for l in layouts[1:])
    save_layouts(layouts, str(tmp_path / "layouts.json"))
    back = load_layouts(str(tmp_path / "layouts.json"))
    assert back[3]["perm"] == layouts[3]["perm"]
    assert sorted(layouts[5]["perm"]) == list(range(64))
