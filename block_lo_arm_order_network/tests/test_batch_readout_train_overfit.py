"""BR-1 Task 9 (overfit): tiny dataset must drive train loss to ~0.

Pure-CPU; no ckpt or wikitext involved. Validates the loop and the loss
plumbing end-to-end on a synthetic problem.
"""
import sys
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))


@pytest.mark.slow
def test_overfit_tiny(tmp_path):
    from batch_readout.train_offline import train

    rng = np.random.default_rng(0)
    N, Mt = 16, 16
    B = rng.standard_normal((Mt, N, N)).astype(np.float32)
    for m in range(Mt):
        np.fill_diagonal(B[m], 0.0)
    sigma = np.stack([rng.permutation(N) for _ in range(Mt)]).astype(np.int64)
    rank = np.argsort(sigma, axis=1).astype(np.int64)
    Y = np.zeros((Mt, N, N), dtype=np.uint8)
    for m in range(Mt):
        Y[m] = (rank[m, :, None] < rank[m, None, :]).astype(np.uint8)
        np.fill_diagonal(Y[m], 0)
    chunks = np.zeros((Mt, 1), dtype=np.int64)

    npz = tmp_path / "tiny.npz"
    np.savez(
        npz,
        train_B_batch=B, train_sigma_T=sigma, train_rank=rank,
        train_pairwise_Y=Y, train_chunks=chunks,
        val_B_batch=B[:2], val_sigma_T=sigma[:2], val_rank=rank[:2],
        val_pairwise_Y=Y[:2], val_chunks=chunks[:2],
        test_B_batch=B[:2], test_sigma_T=sigma[:2], test_rank=rank[:2],
        test_pairwise_Y=Y[:2], test_chunks=chunks[:2],
    )

    out = train(
        dataset_path=str(npz),
        model_name="flatten", loss_name="pairwise",
        N=N, batch_size=4, lr=1e-2, epochs=400, seed=0,
        out_dir=str(tmp_path / "ckpts"), device="cpu",
    )
    # AdamW's wd=1e-2 keeps the asymptote slightly off 0; 0.1 is loose enough
    # to be a "model demonstrably learns" check rather than a wd-sensitive one.
    assert out["final_train_loss"] < 0.1, out["final_train_loss"]
    # best checkpoint exists
    assert (tmp_path / "ckpts" / "g_beta_best.pt").exists()
