"""Order readouts: C-D+L on a single graph, graph-level mix, score-level two-branch mix."""
import numpy as np
from attn_order_teacher import rollout_order, teacher_scores
from graph_normalize import shift_nonneg, offdiag_zscore

MODE = "C-D+L"


def cdl_order(B, greedy=True, tau_T=1.0, seed=0, top_k=0):
    """Full reveal order from one graph via C-D+L; readout sees shift-to-nonneg graph."""
    Bn = shift_nonneg(B)
    return rollout_order(Bn, mode=MODE, greedy=greedy, tau_T=tau_T, seed=seed,
                         top_k=top_k, standardize=True).astype(np.int64)


def graph_mix_order(B_A, B_H, lam, greedy=True, tau_T=1.0, seed=0, top_k=0):
    """Order from the graph-level mix lam*B_A_z + (1-lam)*B_H_z."""
    mix = lam * offdiag_zscore(B_A) + (1.0 - lam) * offdiag_zscore(B_H)
    return cdl_order(mix, greedy=greedy, tau_T=tau_T, seed=seed, top_k=top_k)


def _zscore_vec(v):
    v = np.asarray(v, dtype=np.float64)
    sd = v.std()
    return (v - v.mean()) / (sd if sd > 0 else 1.0)


def score_mix_order(B_A, B_H_resid, gamma, greedy=True, tau_T=1.0, seed=0):
    """Sequential C-D+L with per-step score s = z(s_A) + gamma * z(s_H), on shift-nonneg graphs.
    Frozen proxy of the two-branch policy s = f_A(phi_A) + gamma f_H(phi_H_resid)."""
    A = shift_nonneg(B_A)
    H = shift_nonneg(B_H_resid)
    N = A.shape[0]
    rng = np.random.default_rng(seed)
    S, U, last, order = [], list(range(N)), None, []
    for t in range(N):
        if len(U) == 1:
            v = U[0]
        else:
            qa, cand = teacher_scores(A, S, U, last, mode=MODE)
            qh, candh = teacher_scores(H, S, U, last, mode=MODE)
            assert list(cand) == list(candh)
            score = _zscore_vec(qa) + gamma * _zscore_vec(qh)
            if greedy:
                v = int(cand[int(np.argmax(score))])
            else:
                p = np.exp((score - score.max()) / tau_T); p /= p.sum()
                v = int(rng.choice(cand, p=p))
        order.append(v); S.append(v); U.remove(v); last = v
    return np.asarray(order, dtype=np.int64)
