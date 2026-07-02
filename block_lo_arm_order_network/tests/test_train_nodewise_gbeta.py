import pathlib, sys
import numpy as np, torch, pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "block_lo_arm_order_network", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import train_nodewise_gbeta as prod


def _rand_B65(total, seed=0):
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((total, 65, 65)).astype(np.float64)
    for k in range(total):
        np.fill_diagonal(B[k], 0.0)
    return B


def test_bootstrap_cdl_dataset_shapes_and_antisymmetry():
    B65 = _rand_B65(40, seed=1)
    B_content, Y_pair = prod.bootstrap_cdl_dataset(B65, K=8, m=4, seed=3)
    assert B_content.shape == (8, 64, 64)
    assert Y_pair.shape == (8, 64, 64)
    # soft pairwise: Y[i,j] + Y[j,i] == 1 off-diagonal, diag == 0.5
    Y = Y_pair[0]
    off = ~np.eye(64, dtype=bool)
    assert np.allclose((Y + Y.T)[off], 1.0)
    assert np.allclose(np.diag(Y), 0.5)
    # None node stripped (content is 64×64 with zero diagonal)
    assert np.allclose(np.diagonal(B_content, axis1=1, axis2=2), 0.0)


def test_train_readout_learns_cdl_signal():
    """Dataset Y is derived from B via CDL, so a NodewiseReadout should fit it:
    training loss decreases and val pairwise-acc beats chance."""
    B65 = _rand_B65(60, seed=2)
    B_content, Y_pair = prod.bootstrap_cdl_dataset(B65, K=64, m=6, seed=5)
    model, val_acc, history = prod.train_readout_on_dataset(
        B_content, Y_pair, epochs=15, lr=1e-3, batch_size=16, seed=0, device="cpu")
    assert history[-1]["train_loss"] < history[0]["train_loss"]   # learning
    assert val_acc > 0.55                                          # beats chance


def test_saved_ckpt_loads_via_frozenbetahook(tmp_path):
    from batch_readout.model import NodewiseReadout
    from batch_readout.integration_hook import FrozenBetaHook
    m = NodewiseReadout(N=64, d_model=64, n_layers=2, n_heads=4)
    path = prod.save_gbeta(m, tmp_path / "g_beta_best.pt")
    hook = FrozenBetaHook(path, mode="argsort", device="cpu")
    assert type(hook.model).__name__ == "NodewiseReadout"
    # forward on a 64×64 B produces a (N,) order via the hook
    A = torch.randn(4, 64, 64)
    sigma = hook.step(A)
    assert sigma.shape[0] == 64


@pytest.mark.slow
def test_reproduce_against_nodewise_K1000(tmp_path):
    """Full single-head producer vs the real 10k ckpt: produces a loadable
    NodewiseReadout that learns CDL above chance (small M/K smoke)."""
    if not pathlib.Path(prod.DEFAULT_SOURCE_CKPT).exists():
        pytest.skip(f"10k ckpt absent: {prod.DEFAULT_SOURCE_CKPT}")
    path = prod.train_nodewise_gbeta(
        out_dir=str(tmp_path), head=(1, 7), M=128, n_reveal=1,
        K=64, m=8, epochs=8, seed=123, device="cpu")
    prov = __import__("json").loads((tmp_path / "gbeta_provenance.json").read_text())
    assert prov["head"] == [1, 7]
    assert prov["best_val_pairwise_acc"] > 0.55
    from batch_readout.integration_hook import FrozenBetaHook
    hook = FrozenBetaHook(path, mode="argsort", device="cpu")
    assert type(hook.model).__name__ == "NodewiseReadout"
