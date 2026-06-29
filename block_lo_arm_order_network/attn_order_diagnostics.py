r"""Phase 1.5 — order-set diagnostics shared by the rollout diagnostic script.

Pure functions over order sets (K, N) and a graph B. Single source of truth for the
orientation / structure metrics; covered by test_attn_order.py.

Metric formulas mirror the established conventions:
  - locality_stats / mean_manh / P(d<=1|2): image_order/diagnose_image_orders.py:locality_stats
  - top4_follow / B_edge_ratio: structural_probes.py:probe_order_set
  - kendall tau vs raster: scipy.stats.kendalltau (proxy reference, not a target)
"""
import numpy as np

from attn_order_features import build_features
from attn_order_teacher import teacher_scores

GRID = 8


def _atleast2d(orders):
    return np.atleast_2d(np.asarray(orders, dtype=np.int64))


def locality_stats(orders, grid=GRID):
    o = _atleast2d(orders)
    rows, cols = o // grid, o % grid
    d = np.abs(np.diff(rows, axis=1)) + np.abs(np.diff(cols, axis=1))
    flat = d.ravel()
    return dict(mean_manh=float(d.mean()), p_le1=float((flat <= 1).mean()),
                p_le2=float((flat <= 2).mean()))


def top4_follow_and_edge(orders, B):
    o = _atleast2d(orders)
    N = B.shape[0]
    top4 = {i: set(np.argsort(-B[i])[:4]) for i in range(N)}
    follow, edge = [], []
    for k in range(o.shape[0]):
        for t in range(o.shape[1] - 1):
            u, v = int(o[k, t]), int(o[k, t + 1])
            follow.append(1.0 if v in top4[u] else 0.0)
            edge.append(B[u, v])
    base = float(B[~np.eye(N, dtype=bool)].mean())
    return float(np.mean(follow)), float(np.mean(edge)) / (base + 1e-12)


def kendall_tau_vs_raster(orders):
    from scipy.stats import kendalltau
    o = _atleast2d(orders)
    raster = np.arange(o.shape[1])
    return float(np.nanmean([kendalltau(o[k], raster).correlation for k in range(o.shape[0])]))


def forward_reverse_ratio(orders):
    """Fraction of orders with tau>0 (forward) vs tau<0 (reverse) vs raster, and mean |tau|.

    abs_tau near 1 with split fwd/rev = strong chain whose direction is undetermined."""
    from scipy.stats import kendalltau
    o = _atleast2d(orders)
    raster = np.arange(o.shape[1])
    taus = np.array([kendalltau(o[k], raster).correlation for k in range(o.shape[0])])
    fwd = float((taus > 0).mean())
    rev = float((taus < 0).mean())
    return fwd, rev, float(np.nanmean(np.abs(taus)))


def diversity(orders):
    return len({tuple(o.tolist()) for o in _atleast2d(orders)})


def _entropy_from_counts(counts):
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def start_end_distribution(orders, topn=5):
    o = _atleast2d(orders)
    starts, ends = o[:, 0], o[:, -1]
    sv, sc = np.unique(starts, return_counts=True)
    ev, ec = np.unique(ends, return_counts=True)
    s_order = np.argsort(-sc)
    e_order = np.argsort(-ec)
    return dict(
        top_start=[(int(sv[i]), int(sc[i])) for i in s_order[:topn]],
        top_end=[(int(ev[i]), int(ec[i])) for i in e_order[:topn]],
        start_entropy=round(_entropy_from_counts(sc), 4),
        end_entropy=round(_entropy_from_counts(ec), 4),
        n_unique_starts=int(sv.size),
        n_unique_ends=int(ev.size),
    )


def free_running_agreement(B, student_score_fn, N, k=4):
    """Roll out greedily under `student_score_fn`, and at each *student-visited* state compare
    the student's top-1/top-4 to the teacher's top-1/top-4 (NOT teacher-forced).

    student_score_fn(X) -> 1-D scores over the candidate rows of the 12-d feature matrix X.
    High free-running agreement = the student makes the same local decisions as the teacher
    even along its own trajectory (so any global reversal is an orientation/start effect,
    not local drift). Low = genuine policy divergence.
    """
    B = np.asarray(B, dtype=np.float64)
    S, U, last = [], list(range(N)), None
    t1 = t4 = 0.0
    steps = 0
    for t in range(N):
        if len(U) <= 1:
            break
        X, cand, _ = build_features(B, S, U, last, t, N)
        s_sc = np.asarray(student_score_fn(X), dtype=np.float64).ravel()
        q, _ = teacher_scores(B, S, U, last, mode="C-D+L")
        kk = min(k, len(cand))
        s_top = set(np.argsort(-s_sc)[:kk].tolist())
        q_top = set(np.argsort(-q)[:kk].tolist())
        t1 += float(int(np.argmax(s_sc)) == int(np.argmax(q)))
        t4 += len(s_top & q_top) / kk
        steps += 1
        v = int(cand[int(np.argmax(s_sc))])
        S.append(v); U.remove(v); last = v
    return (t1 / steps, t4 / steps) if steps else (float("nan"), float("nan"))
