"""
Directed graph stochastic order policy -- pure NumPy (no PyTorch, no ON, no AOGPT).
"""
import numpy as np
from typing import Dict, Optional, Tuple, Union


def build_directed_graph(A_global: np.ndarray) -> np.ndarray:
    """B = A_global.T with zero diagonal."""
    B = np.asarray(A_global, dtype=np.float64).T.copy()
    np.fill_diagonal(B, 0.0)
    return B


def compute_source(
    B: np.ndarray,
    alpha_dep: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """source[u] = out[u] - alpha_dep * in[u]; returns (source, out_deg, in_deg)."""
    B = np.asarray(B, dtype=np.float64)
    out_deg = B.sum(axis=1)  # (N,)
    in_deg = B.sum(axis=0)   # (N,)
    source = out_deg - alpha_dep * in_deg
    return source, out_deg, in_deg


def _softmax(
    scores: np.ndarray,
    tau: float,
    rng: np.random.Generator,
    top_k: int = 0,
) -> np.ndarray:
    """Numerically stable softmax with temperature."""
    if tau <= 0.0:
        raise ValueError(f"tau must be > 0, got {tau}")
    scores = np.asarray(scores, dtype=np.float64).copy()
    if top_k > 0 and top_k < scores.size:
        keep = np.argpartition(-scores, top_k - 1)[:top_k]
        masked = np.full_like(scores, -np.inf)
        masked[keep] = scores[keep]
        scores = masked
    if np.ptp(scores) < 1e-15:
        scores = scores + 1e-9 * rng.standard_normal(scores.shape)
    s_max = scores.max()
    exp_scores = np.exp((scores - s_max) / tau)
    return exp_scores / exp_scores.sum()


def progressive_rw_step(
    B: np.ndarray,
    S: np.ndarray,
    U: np.ndarray,
    last: int,
    betas: Dict[str, float],
    tau: float,
    source: np.ndarray,
    rng: np.random.Generator,
    top_k: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """One step of directed_progressive_rw; returns (p_t over candidates, raw scores)."""
    B = np.asarray(B, dtype=np.float64)
    S = np.asarray(S, dtype=np.int64)
    U = np.asarray(U, dtype=np.int64)
    source = np.asarray(source, dtype=np.float64)

    # support(v) = revealed nodes' edges into each candidate
    if len(S) > 0:
        support = B[S].sum(axis=0)[U]  # sum over revealed rows, index candidates
    else:
        support = np.zeros(len(U), dtype=np.float64)

    # future(v) = unrevealed prerequisite mass
    if len(U) > 0:
        future = B[U].sum(axis=0)[U]  # sum over unrevealed rows, index candidates
        # B has zero diagonal, so B[v,v]=0 and future already excludes self
    else:
        future = np.zeros(len(U), dtype=np.float64)

    # local = direct edge from last to candidate
    if last >= 0:
        local_score = B[last, U]
    else:
        local_score = np.zeros(len(U), dtype=np.float64)

    # Combine terms
    score = (
        betas.get('sup', 1.0) * support
        - betas.get('fut', 0.5) * future
        + betas.get('src', 0.2) * source[U]
        + betas.get('loc', 0.5) * local_score
    )

    p_t = _softmax(score, tau, rng, top_k=top_k)
    return p_t, score


def self_avoiding_rw_step(
    B: np.ndarray,
    U: np.ndarray,
    last: int,
    tau: float,
    rng: np.random.Generator,
    top_k: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """One step of self_avoiding_rw; returns (p_t, raw scores)."""
    B = np.asarray(B, dtype=np.float64)
    U = np.asarray(U, dtype=np.int64)

    if last < 0:
        score = np.ones(len(U), dtype=np.float64)
    else:
        score = B[last, U].astype(np.float64)

    p_t = _softmax(score, tau, rng, top_k=top_k)
    return p_t, score


def pagerank(
    B: np.ndarray,
    q: np.ndarray,
    alpha_pr: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """Personalized PageRank: r = alpha_pr * P^T @ r + (1-alpha_pr) * q."""
    B = np.asarray(B, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    N = B.shape[0]

    out_deg = B.sum(axis=1)  # (N,)
    out_safe = np.where(out_deg > 0, out_deg, 1.0)
    P = B / out_safe[:, np.newaxis]  # row-normalized, zero rows stay all-zero

    r = q.copy()
    for _ in range(max_iter):
        r_new = alpha_pr * (P.T @ r) + (1.0 - alpha_pr) * q
        delta = float(np.abs(r_new - r).sum())
        r = r_new
        if delta < tol:
            break
    return r


def sample_order(
    B: np.ndarray,
    policy: str,
    params: Dict[str, float],
    seed: int,
    start_source: Optional[Union[np.ndarray, str]] = None,
) -> Tuple[np.ndarray, float]:
    """Sample ONE order; returns (order: (N,) int64, logprob: float)."""
    rng = np.random.default_rng(seed)
    B = np.asarray(B, dtype=np.float64)
    N = B.shape[0]

    tau_start = float(params.get('tau_start', 1.0))
    tau_step = float(params.get('tau_step', 1.0))
    alpha_dep = float(params.get('alpha_dep', 0.5))
    alpha_pr = float(params.get('alpha_pr', 0.85))
    top_k = int(params.get('top_k', 0) or 0)
    epsilon_uniform = float(params.get('epsilon_uniform', 0.0))

    # Compute source
    if start_source is None:
        source, _, _ = compute_source(B, alpha_dep)
    elif isinstance(start_source, str) and start_source == 'uniform':
        source = np.ones(N, dtype=np.float64)
    else:
        source = np.asarray(start_source, dtype=np.float64)

    # --- Progressive RW --------------------------------------------------
    if policy == 'progressive_rw':
        p0 = _softmax(source, tau_start, rng, top_k=top_k)
        if epsilon_uniform > 0.0:
            p0 = (1.0 - epsilon_uniform) * p0 + epsilon_uniform / N
        idx0 = int(rng.choice(N, p=p0))

        order = np.zeros(N, dtype=np.int64)
        order[0] = idx0
        logprob = float(np.log(max(p0[idx0], 1e-300)))

        S = np.array([idx0], dtype=np.int64)
        U = np.setdiff1d(np.arange(N, dtype=np.int64), S)
        last = idx0

        betas = {
            'sup': float(params.get('beta_sup', 1.0)),
            'fut': float(params.get('beta_fut', 0.5)),
            'src': float(params.get('beta_src', 0.2)),
            'loc': float(params.get('beta_loc', 0.5)),
        }

        for t in range(1, N):
            p_t, _scores = progressive_rw_step(
                B, S, U, last, betas, tau_step, source, rng, top_k=top_k
            )
            if epsilon_uniform > 0.0:
                p_t = (1.0 - epsilon_uniform) * p_t + epsilon_uniform / len(U)
            idx_t = int(rng.choice(len(U), p=p_t))
            node_t = int(U[idx_t])

            order[t] = node_t
            logprob += float(np.log(max(p_t[idx_t], 1e-300)))

            S = np.append(S, node_t)
            U = U[U != node_t]
            last = node_t

        return order, logprob

    # --- Self-avoiding RW ------------------------------------------------
    if policy == 'self_avoiding_rw':
        p0 = _softmax(source, tau_start, rng, top_k=top_k)
        if epsilon_uniform > 0.0:
            p0 = (1.0 - epsilon_uniform) * p0 + epsilon_uniform / N
        idx0 = int(rng.choice(N, p=p0))

        order = np.zeros(N, dtype=np.int64)
        order[0] = idx0
        logprob = float(np.log(max(p0[idx0], 1e-300)))

        S = np.array([idx0], dtype=np.int64)
        U = np.setdiff1d(np.arange(N, dtype=np.int64), S)
        last = idx0

        for t in range(1, N):
            p_t, _scores = self_avoiding_rw_step(B, U, last, tau_step, rng, top_k=top_k)
            if epsilon_uniform > 0.0:
                p_t = (1.0 - epsilon_uniform) * p_t + epsilon_uniform / len(U)
            idx_t = int(rng.choice(len(U), p=p_t))
            node_t = int(U[idx_t])

            order[t] = node_t
            logprob += float(np.log(max(p_t[idx_t], 1e-300)))

            S = np.append(S, node_t)
            U = U[U != node_t]
            last = node_t

        return order, logprob

    # --- PageRank deterministic ------------------------------------------
    if policy == 'pagerank_det':
        q = _softmax(source, tau_start, rng)
        r = pagerank(B, q, alpha_pr)
        scores = r + 1e-9 * rng.standard_normal(N)
        order = np.argsort(-scores).astype(np.int64)
        return order, 0.0

    # --- PageRank stochastic ---------------------------------------------
    if policy == 'pagerank_stoch':
        q = _softmax(source, tau_start, rng)
        r = pagerank(B, q, alpha_pr)
        p0 = _softmax(r, tau_start, rng, top_k=top_k)
        if epsilon_uniform > 0.0:
            p0 = (1.0 - epsilon_uniform) * p0 + epsilon_uniform / N
        idx0 = int(rng.choice(N, p=p0))

        order = np.zeros(N, dtype=np.int64)
        order[0] = idx0
        logprob = float(np.log(max(p0[idx0], 1e-300)))

        S = np.array([idx0], dtype=np.int64)
        U = np.setdiff1d(np.arange(N, dtype=np.int64), S)
        last = idx0

        for t in range(1, N):
            p_t, _scores = self_avoiding_rw_step(B, U, last, tau_step, rng, top_k=top_k)
            if epsilon_uniform > 0.0:
                p_t = (1.0 - epsilon_uniform) * p_t + epsilon_uniform / len(U)
            idx_t = int(rng.choice(len(U), p=p_t))
            node_t = int(U[idx_t])

            order[t] = node_t
            logprob += float(np.log(max(p_t[idx_t], 1e-300)))

            S = np.append(S, node_t)
            U = U[U != node_t]
            last = node_t

        return order, logprob

    raise ValueError(f"Unknown policy: {policy}")


def sample_orders(
    B: np.ndarray,
    policy: str,
    params: Dict[str, float],
    K: int,
    seed_base: int = 42,
) -> Dict:
    """Sample K orders and return orders + diagnostic stats."""
    from order_diagnostics import _kendall_tau

    B = np.asarray(B, dtype=np.float64)
    N = B.shape[0]

    orders = np.zeros((K, N), dtype=np.int64)
    logprobs = np.zeros(K, dtype=np.float64)

    for k in range(K):
        seed = seed_base * 10000 + k
        orders[k], logprobs[k] = sample_order(B, policy, params, seed)

    # --- legality ---
    valid = 0
    for k in range(K):
        if sorted(orders[k].tolist()) == list(range(N)):
            valid += 1
    legal_rate = float(valid / K)

    # --- first_node_entropy ---
    first_nodes = orders[:, 0]
    counts = np.bincount(first_nodes, minlength=N).astype(np.float64)
    probs = counts / K
    probs_nz = probs[probs > 0]
    first_node_entropy = float(-np.sum(probs_nz * np.log(probs_nz)))

    # --- pairwise_tau_mean ---
    n_pairs = min(1000, K * (K - 1) // 2)
    # Use deterministic sampling for reproducibility
    pair_rng = np.random.default_rng(seed_base * 10000 + 99999)
    if K >= 2:
        all_i, all_j = np.triu_indices(K, k=1)
        if len(all_i) > n_pairs:
            idx = pair_rng.choice(len(all_i), size=n_pairs, replace=False)
            all_i = all_i[idx]
            all_j = all_j[idx]
        pair_taus = []
        for i, j in zip(all_i, all_j):
            pair_taus.append(_kendall_tau(orders[i], orders[j]))
        pairwise_tau_mean = float(np.mean(pair_taus)) if pair_taus else 0.0
    else:
        pairwise_tau_mean = 0.0

    # --- mean_directed_score ---
    directed_scores = []
    for k in range(K):
        for t in range(N - 1):
            directed_scores.append(B[orders[k, t], orders[k, t + 1]])
    mean_directed_score = float(np.mean(directed_scores)) if directed_scores else 0.0

    # --- mean_progressive_support ---
    supports = []
    for k in range(K):
        S_set = set()
        for t in range(N):
            v = int(orders[k, t])
            if t > 0:
                sup = sum(float(B[u, v]) for u in S_set)
                supports.append(sup / float(t))
            S_set.add(v)
    mean_progressive_support = float(np.mean(supports)) if supports else 0.0

    # --- policy_step_entropy ---
    n_replay = min(100, K)
    all_H = np.zeros((n_replay, N), dtype=np.float64)
    for i in range(n_replay):
        seed = seed_base * 10000 + i
        all_H[i] = _policy_step_entropy_replay(B, policy, params, orders[i], seed)

    mean_per_step = all_H.mean(axis=0)  # (N,)
    third = max(1, N // 3)
    early = float(mean_per_step[:third].mean())
    mid = float(mean_per_step[third:2 * third].mean())
    late = float(mean_per_step[2 * third:].mean())
    overall = float(mean_per_step.mean())

    # --- tau_vs_l2r ---
    L2R = np.arange(N, dtype=np.int64)
    taus_l2r = np.array([_kendall_tau(orders[i], L2R) for i in range(K)], dtype=np.float64)
    tau_vs_l2r_mean = float(taus_l2r.mean())
    tau_vs_l2r_std = float(taus_l2r.std())

    # --- logprob stats ---
    logprob_mean = float(logprobs.mean())
    logprob_std = float(logprobs.std())
    logprob_p10 = float(np.percentile(logprobs, 10))
    logprob_p90 = float(np.percentile(logprobs, 90))

    return {
        'orders': orders,
        'logprobs': logprobs,
        'legal_rate': legal_rate,
        'tau_vs_l2r_mean': tau_vs_l2r_mean,
        'tau_vs_l2r_std': tau_vs_l2r_std,
        'first_node_entropy': first_node_entropy,
        'pairwise_tau_mean': pairwise_tau_mean,
        'mean_directed_score': mean_directed_score,
        'mean_progressive_support': mean_progressive_support,
        'policy_step_entropy': {
            'mean': overall,
            'early': early,
            'mid': mid,
            'late': late,
        },
        'cross_logprob': None,  # computed by Phase 1 script (cross-policy metric)
        'logprob_mean': logprob_mean,
        'logprob_std': logprob_std,
        'logprob_p10': logprob_p10,
        'logprob_p90': logprob_p90,
    }


def _policy_step_entropy_replay(
    B: np.ndarray,
    policy: str,
    params: Dict[str, float],
    order: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Replay one order step-by-step; return per-step entropies H_t (N,) float64."""
    rng = np.random.default_rng(seed)
    B = np.asarray(B, dtype=np.float64)
    order = np.asarray(order, dtype=np.int64)
    N = B.shape[0]

    tau_start = float(params.get('tau_start', 1.0))
    tau_step = float(params.get('tau_step', 1.0))
    alpha_dep = float(params.get('alpha_dep', 0.5))
    alpha_pr = float(params.get('alpha_pr', 0.85))
    top_k = int(params.get('top_k', 0) or 0)

    source, _, _ = compute_source(B, alpha_dep)

    # --- pagerank_det has no per-step distribution ---
    if policy == 'pagerank_det':
        return np.zeros(N, dtype=np.float64)

    betas = {
        'sup': float(params.get('beta_sup', 1.0)),
        'fut': float(params.get('beta_fut', 0.5)),
        'src': float(params.get('beta_src', 0.2)),
        'loc': float(params.get('beta_loc', 0.5)),
    }

    H = np.zeros(N, dtype=np.float64)

    # --- Step 0: p0 ---
    if policy in ('progressive_rw', 'self_avoiding_rw'):
        p0 = _softmax(source, tau_start, rng, top_k=top_k)
    elif policy == 'pagerank_stoch':
        q = _softmax(source, tau_start, rng)
        r = pagerank(B, q, alpha_pr)
        p0 = _softmax(r, tau_start, rng, top_k=top_k)
    else:
        raise ValueError(f"Unknown policy: {policy}")

    eps = 1e-300
    H[0] = float(-np.sum(p0 * np.log(np.maximum(p0, eps))))

    # --- Step t ---
    first = int(order[0])
    S = np.array([first], dtype=np.int64)
    U = np.setdiff1d(np.arange(N, dtype=np.int64), S)
    last = first

    for t in range(1, N):
        if policy == 'progressive_rw':
            p_t, _scores = progressive_rw_step(
                B, S, U, last, betas, tau_step, source, rng, top_k=top_k
            )
        elif policy in ('self_avoiding_rw', 'pagerank_stoch'):
            p_t, _scores = self_avoiding_rw_step(B, U, last, tau_step, rng, top_k=top_k)
        else:
            p_t = np.ones(len(U)) / len(U)  # fallback

        eps_step = 1e-300
        H[t] = float(-np.sum(p_t * np.log(np.maximum(p_t, eps_step))))

        chosen = int(order[t])
        if chosen not in U:
            # Should not happen for valid permutations; break gracefully
            break
        S = np.append(S, chosen)
        U = U[U != chosen]
        last = chosen

    return H


def policy_step_entropy(
    B: np.ndarray,
    S: np.ndarray,
    U: np.ndarray,
    last: int,
    params: Dict[str, float],
) -> float:
    """Per-step entropy H_t for Phase 3 diagnostics (§1.7, §3.1).

    Spec signature: (B, S, U, last, params) → H_t.
    If S is empty (first step), computes H of p₀. Otherwise computes H of p_t
    from progressive_rw_step.
    """
    B = np.asarray(B, dtype=np.float64)
    rng = np.random.default_rng(0)  # rng not used for entropy-only computation

    tau_start = float(params.get('tau_start', 1.0))
    tau_step = float(params.get('tau_step', 1.0))
    alpha_dep = float(params.get('alpha_dep', 0.5))
    top_k = int(params.get('top_k', 0) or 0)

    if len(S) == 0:
        source, _, _ = compute_source(B, alpha_dep)
        p = _softmax(source, tau_start, rng, top_k=top_k)
    else:
        source, _, _ = compute_source(B, alpha_dep)
        betas = {
            'sup': float(params.get('beta_sup', 1.0)),
            'fut': float(params.get('beta_fut', 0.5)),
            'src': float(params.get('beta_src', 0.2)),
            'loc': float(params.get('beta_loc', 0.5)),
        }
        p, _ = progressive_rw_step(B, S, U, last, betas, tau_step, source, rng, top_k=top_k)

    eps = 1e-300
    return float(-np.sum(p * np.log(np.maximum(p, eps))))
