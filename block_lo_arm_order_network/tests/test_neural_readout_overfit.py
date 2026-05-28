"""NR-1 Task 9: end-to-end overfit-tiny-batch sanity test.

Builds 4 toy chain graphs (each with a different random permutation of nodes),
trains GraphTransformerReadout via pairwise_logistic_loss with AdamW for a few
hundred steps. With model >> data, the loss must drive to near 0 and the
Kendall tau against the synthetic teacher rank must be ~1.

Verifies the full pipeline (model + loss + rank convention + metrics) is
wired correctly together. Does not touch real wikitext or any ckpt.
"""
import sys
import pathlib

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))


def test_overfit_tiny_batch():
    from neural_readout.graph_transformer_readout import GraphTransformerReadout
    from neural_readout.loss import pairwise_logistic_loss
    from neural_readout.eval_metrics import compute_matching_metrics

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    M, N = 4, 64

    # Synthetic chain B + matching rank: each graph has its own random
    # permutation perm, with edge perm[k] -> perm[k+1]. The teacher rank
    # is rank[perm[k]] = k (so perm[0] is earliest, rank 0).
    B_np = np.zeros((M, N, N), dtype=np.float32)
    rank_np = np.zeros((M, N), dtype=np.int64)
    for i in range(M):
        perm = rng.permutation(N)
        for k in range(N - 1):
            B_np[i, perm[k], perm[k + 1]] = 1.0
        rank_np[i, perm] = np.arange(N)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    g = GraphTransformerReadout().to(device)
    opt = torch.optim.AdamW(g.parameters(), lr=3e-3)

    B = torch.from_numpy(B_np).to(device)
    rank = torch.from_numpy(rank_np).to(device)

    final_loss = None
    for _step in range(300):
        s = g(B)
        loss = pairwise_logistic_loss(s, rank)
        opt.zero_grad()
        loss.backward()
        opt.step()
        final_loss = loss.item()

    assert final_loss is not None and final_loss < 0.05, \
        f"final loss {final_loss:.4f} should overfit to near 0"

    with torch.no_grad():
        scores = g(B).cpu().numpy()
    metrics = compute_matching_metrics(scores=scores, rank=rank.cpu().numpy())
    assert metrics["kendall_tau"] > 0.95, metrics
    assert metrics["pairwise_precedence_acc"] > 0.95, metrics
