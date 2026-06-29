import pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.p5_utility_controller import load_p5_ckpt, block_hidden_states

CKPT = str(ROOT / "runs/handoff_overnight/seed123/ckpt_step10000.pt")


def test_hidden_shape_and_determinism():
    model, chunks, clean_perm, dev = load_p5_ckpt(CKPT, M=2, device="cpu")
    sig = np.arange(64)
    H1 = block_hidden_states(model, chunks[0:1], sig, clean_perm, layer=0, device=dev)
    H2 = block_hidden_states(model, chunks[0:1], sig, clean_perm, layer=0, device=dev)
    assert H1.shape[0] == 64 and H1.ndim == 2
    assert np.allclose(H1, H2)                          # deterministic, frozen model
    # different texts give different H
    H3 = block_hidden_states(model, chunks[1:2], sig, clean_perm, layer=0, device=dev)
    assert not np.allclose(H1, H3)
