"""NR-1 Task 8: attention-order matching metrics (spec §5.1, §5.2).

Five per-batch metrics:
  - kendall_tau               (§5.1 hard gate)
  - pairwise_precedence_acc   (§5.1 hard gate; numerically ~ (1 + tau) / 2)
  - spearman_rho              (§5.1 hard gate, rank-rank correlation)
  - top1_first_node_match     (§5.2 diagnostic)
  - first3_set_match          (§5.2 diagnostic)

All metrics are derived purely from the predicted order vs the teacher rank;
NO NLL, loss, or external utility signal is read.

Score convention (matches teacher_labels.py + loss.py):
    rank 0 = earliest revealed; earliest should get the HIGHEST score.
"""
import numpy as np
from scipy.stats import kendalltau, spearmanr


def _pred_rank_from_scores(scores):
    """scores: (M, N). Returns (M, N) predicted rank where rank 0 = highest score."""
    order = np.argsort(-scores, axis=-1)  # descending; order[m, t] = node at predicted-rank t
    pred_rank = np.empty_like(order)
    M, N = order.shape
    for m in range(M):
        pred_rank[m, order[m]] = np.arange(N)
    return pred_rank


def compute_matching_metrics(scores, rank):
    """Average §5.1 + §5.2 metrics over a batch.

    Args:
        scores: (M, N) float, model output.
        rank:   (M, N) int, teacher rank (rank 0 = earliest).

    Returns:
        dict with the five metrics above.
    """
    scores = np.asarray(scores)
    rank = np.asarray(rank, dtype=np.int64)
    if scores.shape != rank.shape:
        raise ValueError(f"shape mismatch: scores {scores.shape}, rank {rank.shape}")

    M, N = scores.shape
    pred_rank = _pred_rank_from_scores(scores)

    taus, rhos, accs, top1s, first3s = [], [], [], [], []
    for m in range(M):
        tau, _ = kendalltau(pred_rank[m], rank[m])
        rho, _ = spearmanr(pred_rank[m], rank[m])

        # Pairwise precedence accuracy: over teacher-ordered pairs (i, j) with
        # rank[i] < rank[j], how often does pred_rank[i] < pred_rank[j]?
        true_lt = rank[m][:, None] < rank[m][None, :]
        pred_lt = pred_rank[m][:, None] < pred_rank[m][None, :]
        if true_lt.any():
            acc = float(pred_lt[true_lt].mean())
        else:
            acc = 0.5

        # top-1 first node match: did we predict the same first-revealed node?
        teacher_first = int(np.argmin(rank[m]))     # node with rank 0
        student_first = int(np.argmin(pred_rank[m]))  # equivalently argmax(scores[m])
        top1 = float(teacher_first == student_first)

        # first-3 set match: |intersection| / 3 of the two first-3 sets
        teacher_first3 = set(np.argsort(rank[m])[:3].tolist())
        student_first3 = set(np.argsort(pred_rank[m])[:3].tolist())
        first3 = len(teacher_first3 & student_first3) / 3.0

        taus.append(tau)
        rhos.append(rho)
        accs.append(acc)
        top1s.append(top1)
        first3s.append(first3)

    return {
        "kendall_tau": float(np.nanmean(taus)),
        "pairwise_precedence_acc": float(np.mean(accs)),
        "spearman_rho": float(np.nanmean(rhos)),
        "top1_first_node_match": float(np.mean(top1s)),
        "first3_set_match": float(np.mean(first3s)),
    }
