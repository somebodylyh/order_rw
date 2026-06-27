"""A4: carrier-B structure across ckpts (pre/post/converged) + Pattern A/B."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.plot_emergence import carrier_b_structure, plot_carrier_b  # noqa: E402


def test_carrier_b_structure_and_render(tmp_path):
    bs = carrier_b_structure(2, winning_layer=1, winner_heads=[3, 5, 7],
                             ckpt_steps=(2000, 10000),
                             root=str(ROOT / "runs/handoff_overnight"))
    assert bs["pattern"] in ("sharpening", "switch")
    assert 10000 in bs["by_step"]
    plot_carrier_b(bs, str(tmp_path))
    assert (tmp_path / "carrier_b_structure.png").exists()
