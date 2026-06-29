import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.p5_utility_controller import build_dataset, train_b_only, train_residual

CKPT = str(ROOT / "runs/handoff_overnight/seed123/ckpt_step10000.pt")


def test_dataset_and_training_runs():
    samples = build_dataset(CKPT, M=6, n_reveals=4, K_rand=3, K_noisy=3)
    s0 = samples[0]
    assert s0["B_feat"].shape == (64, 130) and s0["P"].shape == (64, 64)
    gB = train_b_only(samples, epochs=20)
    sc = train_residual(samples, gB, h_dim=s0["H"].shape[1], epochs=20, h_mode="real")
    scs = train_residual(samples, gB, h_dim=s0["H"].shape[1], epochs=20, h_mode="shuffle")
    assert sc is not None and scs is not None
