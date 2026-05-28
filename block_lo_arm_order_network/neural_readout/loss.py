"""NR-1 Task 7: pairwise logistic loss with rank-0=earliest convention.

For all pairs (i, j) with rank[i] < rank[j] (i revealed earlier than j),
the model is rewarded for predicting s[i] > s[j]:

    L = - (1 / |P|) * sum_{(i,j) in P} log sigma( s[i] - s[j] )
    P = { (i, j) : rank[i] < rank[j] }

For N = 64 all N*(N-1)/2 = 2016 directed pairs are used per graph.

Convention (load-bearing, also enforced by neural_readout.teacher_labels):
    rank[v] = position of v in reveal order
    rank 0 = earliest revealed
    earliest should get the HIGHEST score
"""
import torch
import torch.nn.functional as F


def pairwise_logistic_loss(scores, rank):
    """Compute mean pairwise logistic loss over a batch of graphs.

    Args:
        scores: (batch, N) float tensor.
        rank:   (batch, N) long tensor; rank 0 = earliest revealed.

    Returns:
        Scalar tensor, mean of per-graph losses.
    """
    if scores.dim() != 2 or rank.dim() != 2 or scores.shape != rank.shape:
        raise ValueError(
            f"shape mismatch: scores {tuple(scores.shape)}, rank {tuple(rank.shape)}"
        )
    rank = rank.long()
    # mask[b, i, j] = True iff rank[b, i] < rank[b, j]  -> i should beat j
    mask = rank.unsqueeze(-1) < rank.unsqueeze(-2)          # (b, N, N)
    # logits[b, i, j] = s[b, i] - s[b, j]
    logits = scores.unsqueeze(-1) - scores.unsqueeze(-2)    # (b, N, N)
    # -log sigma(x) = softplus(-x)
    per_pair = F.softplus(-logits)                          # (b, N, N)
    n_pairs = mask.sum(dim=(-1, -2)).clamp_min(1).float()
    loss = (per_pair * mask).sum(dim=(-1, -2)) / n_pairs    # (b,)
    return loss.mean()
