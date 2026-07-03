"""Unified C-D+L attention-only teacher for the order policy (ported verbatim).

For each candidate v in U_t:
    C(v) = mean_{u in S_t}        B[u, v]        # selected-set support
    D(v) = mean_{u in U_t, u!=v}  B[u, v]        # unselected-set dependency
    L(v) = B[last, v]                            # last-step transition edge
    q_t(v) = C(v) - D(v) + L(v)

Attention-only, no readiness / distance / NLL terms. B[u, v] = edge u -> v,
zero diagonal.
"""
import numpy as np

MODES = ("C", "L", "C-D", "C+L", "C-D+L", "-D")
MAIN_MODE = "C-D+L"


def teacher_components(B, S_t, U_t, last):
    """Return (C, D, L, candidates) — the three attention-only components over U_t."""
    B = np.asarray(B, dtype=np.float64)
    S = np.asarray(list(S_t), dtype=np.int64)
    U = np.asarray(list(U_t), dtype=np.int64)
    nU = U.shape[0]

    if S.shape[0] > 0 and nU > 0:
        C = B[np.ix_(S, U)].mean(axis=0)
    else:
        C = np.zeros(nU, dtype=np.float64)

    if nU > 1:
        M = B[np.ix_(U, U)]
        D = M.sum(axis=0) / (nU - 1)
    else:
        D = np.zeros(nU, dtype=np.float64)

    if last is not None and nU > 0:
        L = B[last, U]
    else:
        L = np.zeros(nU, dtype=np.float64)

    return C, D, L, U.copy()


def teacher_scores(B, S_t, U_t, last, mode=MAIN_MODE):
    """Return (q, candidates) for the requested teacher mode."""
    if mode not in MODES:
        raise ValueError(f"unknown teacher mode {mode!r}; choose from {MODES}")
    C, D, L, U = teacher_components(B, S_t, U_t, last)
    if mode == "C":
        q = C
    elif mode == "L":
        q = L
    elif mode == "C-D":
        q = C - D
    elif mode == "C+L":
        q = C + L
    elif mode == "-D":
        q = -D
    else:  # "C-D+L"
        q = C - D + L
    return q, U


def _softmax(q, tau):
    if tau <= 0.0:
        raise ValueError(f"tau_T must be > 0, got {tau}")
    q = np.asarray(q, dtype=np.float64)
    if q.size == 0:
        return q.copy()
    z = (q - q.max()) / tau
    e = np.exp(z)
    return e / e.sum()


def _entropy(p):
    p = np.asarray(p, dtype=np.float64)
    nz = p > 0
    return float(-(p[nz] * np.log(p[nz])).sum())


def teacher_step(B, S_t, U_t, last, tau_T=1.0, mode=MAIN_MODE, top_k=4):
    """Full teacher step: scores, probs, entropy, top-k candidates."""
    q, U = teacher_scores(B, S_t, U_t, last, mode=mode)
    p = _softmax(q, tau_T)
    ent = _entropy(p)
    order = np.argsort(-q)
    k = min(top_k, U.shape[0]) if top_k and top_k > 0 else U.shape[0]
    topk_candidates = U[order[:k]].tolist()
    return {
        "scores": q,
        "probs": p,
        "entropy": ent,
        "topk_candidates": topk_candidates,
        "candidates": U,
        "mode": mode,
        "tau_T": tau_T,
    }


def rollout_order(B, tau_T=1.0, seed=0, mode=MAIN_MODE, standardize=False,
                  top_k=0, greedy=False, start=None, return_entropy=False):
    """Roll out a full reveal order by sequentially sampling from the teacher."""
    B = np.asarray(B, dtype=np.float64)
    N = B.shape[0]
    rng = np.random.default_rng(seed)
    S, U, last = [], list(range(N)), None
    order, entropies = [], []

    for t in range(N):
        if len(U) == 1:
            v = U[0]
        else:
            q, cand = teacher_scores(B, S, U, last, mode=mode)
            if t == 0 and start is not None:
                v = int(start)
            elif greedy:
                v = int(cand[int(np.argmax(q))])
            else:
                qq = q.astype(np.float64).copy()
                if standardize:
                    qq = (qq - qq.mean()) / (qq.std() + 1e-9)
                if top_k and 0 < top_k < qq.size:
                    keep = np.argpartition(-qq, top_k - 1)[:top_k]
                    masked = np.full_like(qq, -np.inf)
                    masked[keep] = qq[keep]
                    qq = masked
                p = _softmax(qq, tau_T)
                v = int(cand[rng.choice(len(cand), p=p)])
                if return_entropy:
                    entropies.append(_entropy(p))
        order.append(v)
        S.append(v)
        U.remove(v)
        last = v

    order = np.asarray(order, dtype=np.int64)
    return (order, entropies) if return_entropy else order
