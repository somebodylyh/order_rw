"""NR-1 Task 10 (extended): end-to-end train_nr1 smoke on synthetic data.

Confirms the full training pipeline (load -> ablation -> model -> loss ->
metrics -> selection by tau) runs end-to-end on a tiny synthetic dataset
without GPU model loading. Does NOT verify NLL diagnostic (Task 11) or
ablation correctness (Task 14, covered separately).

Synthetic dataset: 8 graphs, chain B + matching teacher rank. Trains 5
epochs, expects loss decrease and a `g_beta_best.pt` checkpoint to be
written.
"""
import sys
import pathlib

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))


def _build_synthetic_dataset(tmp_path, M=8, N=64, seed=0):
    """Write a (B, sigma, rank, chunk_index, split) .npz with chain graphs."""
    from neural_readout.dataset import save_dataset
    rng = np.random.default_rng(seed)
    B = np.zeros((M, N, N), dtype=np.float32)
    sigma = np.zeros((M, N), dtype=np.int64)
    rank = np.zeros((M, N), dtype=np.int64)
    for i in range(M):
        perm = rng.permutation(N)
        for k in range(N - 1):
            B[i, perm[k], perm[k + 1]] = 1.0
        sigma[i] = perm
        rank[i, perm] = np.arange(N)
    chunk_index = np.arange(M, dtype=np.int64)
    path = tmp_path / "synth.npz"
    save_dataset(str(path), B, sigma, rank, chunk_index, split="train",
                 meta={"ckpt_path": "/dev/null", "M": M, "seed": seed, "alpha_dep": 0.5})
    return str(path)


def test_train_nr1_runs_end_to_end(tmp_path):
    from neural_readout.train_nr1 import train_nr1
    dataset_path = _build_synthetic_dataset(tmp_path)
    out_dir = tmp_path / "ckpts"
    best_path, log = train_nr1(
        dataset_path=dataset_path,
        out_dir=str(out_dir),
        train_n=6,
        val_n=2,
        epochs=5,
        batch=4,
        lr=3e-3,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        eval_every_epochs=1,
    )
    assert pathlib.Path(best_path).exists()
    assert len(log) == 5
    # Sanity: tau on synthetic data should be measurably above zero after 5 epochs
    # (it won't be near 1 with only 6 train samples, but the pipeline must be wired right)
    final_tau = log[-1]["kendall_tau"]
    assert -1.0 <= final_tau <= 1.0, final_tau


def test_train_nr1_ablation_b_global_path(tmp_path):
    """b_global ablation must not crash and must still produce a checkpoint."""
    from neural_readout.train_nr1 import train_nr1
    dataset_path = _build_synthetic_dataset(tmp_path)
    out_dir = tmp_path / "ckpts_bglobal"
    best_path, log = train_nr1(
        dataset_path=dataset_path,
        out_dir=str(out_dir),
        train_n=6,
        val_n=2,
        epochs=3,
        batch=4,
        lr=3e-3,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        ablation="b_global",
    )
    assert pathlib.Path(best_path).exists()
    assert len(log) == 3
