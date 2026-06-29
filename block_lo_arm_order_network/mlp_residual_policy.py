"""
MLP residual policy: augments RW scores with a learned residual.

logit_t(v) = score_RW_t(v) + gamma * MLP(features_t(v))

Features (12-dim, NO raw block index):
  0-3: support, future, source, local (normalized)
  4-5: in_deg, out_deg (normalized)
  6-7: t/N, |S|/N (scalar)
  8-11: rank_support, rank_future, rank_source, rank_local (0-1)
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

# imports from sibling modules (run from block_lo_arm_order_network/)
from directed_graph_policy import (
    compute_source,
    progressive_rw_step,
    _softmax,
)

FEATURE_DIM = 12


class MLPResidualPolicy(nn.Module):
    """Small MLP: 12 -> 32 -> 16 -> 8 -> 1 (ReLU). Last layer zero-init."""

    def __init__(self, gamma: float = 0.0, hidden_dims: Tuple[int, ...] = (32, 16, 8)):
        super().__init__()
        layers = []
        prev = FEATURE_DIM
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        # zero-init last layer
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        self.net = nn.Sequential(*layers)
        self.gamma = nn.Parameter(torch.tensor(float(gamma)), requires_grad=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (*, FEATURE_DIM) -> residuals: (*,)"""
        raw = self.net(features).squeeze(-1)
        return self.gamma * raw


def compute_step_features(
    B: np.ndarray,
    S: np.ndarray,
    U: np.ndarray,
    last: int,
    source: np.ndarray,
    out_deg: np.ndarray,
    in_deg: np.ndarray,
    t: int,
    N: int,
) -> np.ndarray:
    """Compute per-candidate features for MLP input. Returns (len(U), 12) float64."""
    n_candidates = len(U)

    # raw scores
    if len(S) > 0:
        support = B[S].sum(axis=0)[U]
    else:
        support = np.zeros(n_candidates, dtype=np.float64)

    if n_candidates > 0:
        future = B[U].sum(axis=0)[U]
    else:
        future = np.zeros(n_candidates, dtype=np.float64)

    local_score = B[last, U] if last >= 0 else np.zeros(n_candidates, dtype=np.float64)

    src_U = source[U]
    out_U = out_deg[U]
    in_U = in_deg[U]

    # normalization scales (avoid div-by-zero)
    def safe_scale(arr):
        m = float(np.max(np.abs(arr)))
        return m if m > 0.0 else 1.0

    # rank normalizer
    def norm_rank(arr):
        if len(arr) <= 1:
            return np.zeros(len(arr), dtype=np.float64)
        ranks = np.argsort(np.argsort(arr).astype(np.float64))
        return ranks / (len(arr) - 1)

    feats = np.zeros((n_candidates, FEATURE_DIM), dtype=np.float64)
    feats[:, 0] = support / safe_scale(support)
    feats[:, 1] = future / safe_scale(future)
    feats[:, 2] = src_U / safe_scale(src_U)
    feats[:, 3] = local_score / safe_scale(local_score)
    feats[:, 4] = in_U / safe_scale(in_U)
    feats[:, 5] = out_U / safe_scale(out_U)
    feats[:, 6] = t / N
    feats[:, 7] = len(S) / N
    feats[:, 8] = norm_rank(support)
    feats[:, 9] = norm_rank(future)
    feats[:, 10] = norm_rank(src_U)
    feats[:, 11] = norm_rank(local_score)

    return feats


def sample_order_with_mlp(
    B: np.ndarray,
    params: Dict[str, float],
    seed: int,
    mlp: Optional[MLPResidualPolicy] = None,
    mlp_device: str = "cpu",
) -> Tuple[np.ndarray, float, List[np.ndarray], List[np.ndarray]]:
    """Sample ONE order using RW + MLP residual.

    Returns (order (N,), logprob, step_probs, rw_scores).
    If mlp is None or |gamma| < 1e-12: identical to sample_order(progressive_rw).
    """
    rng = np.random.default_rng(seed)
    B = np.asarray(B, dtype=np.float64)
    N = B.shape[0]

    tau_start = float(params.get('tau_start', 1.0))
    tau_step = float(params.get('tau_step', 1.0))
    alpha_dep = float(params.get('alpha_dep', 0.5))
    top_k = int(params.get('top_k', 0) or 0)
    epsilon_uniform = float(params.get('epsilon_uniform', 0.0))

    source, out_deg, in_deg = compute_source(B, alpha_dep)

    betas = {
        'sup': float(params.get('beta_sup', 1.0)),
        'fut': float(params.get('beta_fut', 0.5)),
        'src': float(params.get('beta_src', 0.2)),
        'loc': float(params.get('beta_loc', 0.5)),
    }

    use_mlp = mlp is not None and abs(float(mlp.gamma.item())) > 1e-12

    # --- Step 0: p0 from source ---
    raw_scores_t0 = source.copy()
    if use_mlp:
        features0 = compute_step_features(
            B, np.array([], dtype=np.int64), np.arange(N, dtype=np.int64),
            -1, source, out_deg, in_deg, 0, N,
        )
        with torch.no_grad():
            feats_t = torch.from_numpy(features0.astype(np.float32)).to(mlp_device)
            residuals = mlp(feats_t).cpu().numpy()
        raw_scores_t0 = raw_scores_t0 + residuals.astype(np.float64)

    p0 = _softmax(raw_scores_t0, tau_start, rng, top_k=top_k)
    if epsilon_uniform > 0.0:
        p0 = (1.0 - epsilon_uniform) * p0 + epsilon_uniform / N
    idx0 = int(rng.choice(N, p=p0))

    order = np.zeros(N, dtype=np.int64)
    order[0] = idx0
    logprob = float(np.log(max(p0[idx0], 1e-300)))

    step_probs = [p0]
    rw_scores_list = [source.copy()]

    S = np.array([idx0], dtype=np.int64)
    U = np.setdiff1d(np.arange(N, dtype=np.int64), S)
    last = idx0

    for t in range(1, N):
        _, rw_score = progressive_rw_step(
            B, S, U, last, betas, tau_step, source, rng, top_k=top_k,
        )
        combined_score = rw_score.copy()
        if use_mlp:
            features_t = compute_step_features(
                B, S, U, last, source, out_deg, in_deg, t, N,
            )
            with torch.no_grad():
                feats_t = torch.from_numpy(features_t.astype(np.float32)).to(mlp_device)
                residuals = mlp(feats_t).cpu().numpy()
            combined_score = combined_score + residuals.astype(np.float64)

        p_t = _softmax(combined_score, tau_step, rng, top_k=top_k)
        if epsilon_uniform > 0.0:
            p_t = (1.0 - epsilon_uniform) * p_t + epsilon_uniform / len(U)
        idx_t = int(rng.choice(len(U), p=p_t))
        node_t = int(U[idx_t])

        order[t] = node_t
        logprob += float(np.log(max(p_t[idx_t], 1e-300)))

        step_probs.append(p_t)
        rw_scores_list.append(rw_score)

        S = np.append(S, node_t)
        U = U[U != node_t]
        last = node_t

    return order, logprob, step_probs, rw_scores_list


def compute_logprob_for_order_mlp(
    B: np.ndarray,
    params: Dict[str, float],
    order: np.ndarray,
    mlp: Optional[MLPResidualPolicy] = None,
    mlp_device: str = "cpu",
) -> float:
    """Replay one order step-by-step; return total logprob under MLP-augmented policy.

    Uses the SAME top_k setting as sampling (from params). No epsilon_uniform applied
    during replay (that's a sampling-time noise, not part of the distribution).
    """
    B = np.asarray(B, dtype=np.float64)
    order = np.asarray(order, dtype=np.int64)
    N = B.shape[0]

    tau_start = float(params.get('tau_start', 1.0))
    tau_step = float(params.get('tau_step', 1.0))
    alpha_dep = float(params.get('alpha_dep', 0.5))
    top_k = int(params.get('top_k', 0) or 0)

    source, out_deg, in_deg = compute_source(B, alpha_dep)

    betas = {
        'sup': float(params.get('beta_sup', 1.0)),
        'fut': float(params.get('beta_fut', 0.5)),
        'src': float(params.get('beta_src', 0.2)),
        'loc': float(params.get('beta_loc', 0.5)),
    }

    use_mlp = mlp is not None and abs(float(mlp.gamma.item())) > 1e-12

    # Step 0
    raw_scores_t0 = source.copy()
    if use_mlp:
        features0 = compute_step_features(
            B, np.array([], dtype=np.int64), np.arange(N, dtype=np.int64),
            -1, source, out_deg, in_deg, 0, N,
        )
        with torch.no_grad():
            feats_t = torch.from_numpy(features0.astype(np.float32)).to(mlp_device)
            residuals = mlp(feats_t).cpu().numpy()
        raw_scores_t0 = raw_scores_t0 + residuals.astype(np.float64)

    rng0 = np.random.default_rng(0)
    p0 = _softmax(raw_scores_t0, tau_start, rng0, top_k=top_k)
    idx0 = int(order[0])
    logprob = float(np.log(max(p0[idx0], 1e-300)))

    S = np.array([idx0], dtype=np.int64)
    U = np.setdiff1d(np.arange(N, dtype=np.int64), S)
    last = idx0

    rng_step = np.random.default_rng(1)
    for t in range(1, N):
        chosen = int(order[t])
        if chosen not in U:
            break

        _, rw_score = progressive_rw_step(
            B, S, U, last, betas, tau_step, source, rng_step, top_k=top_k,
        )
        combined_score = rw_score.copy()
        if use_mlp:
            features_t = compute_step_features(
                B, S, U, last, source, out_deg, in_deg, t, N,
            )
            with torch.no_grad():
                feats_t = torch.from_numpy(features_t.astype(np.float32)).to(mlp_device)
                residuals = mlp(feats_t).cpu().numpy()
            combined_score = combined_score + residuals.astype(np.float64)

        p_t = _softmax(combined_score, tau_step, rng_step, top_k=top_k)
        chosen_idx = int(np.where(U == chosen)[0][0])
        logprob += float(np.log(max(p_t[chosen_idx], 1e-300)))

        S = np.append(S, chosen)
        U = U[U != chosen]
        last = chosen

    return logprob


# ── PyTorch-differentiable replay (for Stage C training) ──

def compute_policy_logprob_batch(
    B: np.ndarray,
    params: Dict[str, float],
    orders: np.ndarray,  # (batch_size, K, N) physical orders
    mlp: MLPResidualPolicy,
    mlp_device: str,
) -> torch.Tensor:
    """Compute differentiable logprob of orders under MLP-augmented policy.

    Replays each order step-by-step. At each step:
      score = rw_score (NumPy, constant) + mlp(features) (PyTorch, differentiable)
      p_t = softmax(score / tau) with top-k masking
      logprob += log(p_t[chosen])

    Returns (batch_size, K) float tensor with grad w.r.t. mlp parameters.
    Uses the SAME top_k as params to keep sample/replay consistent.
    """
    B_np = np.asarray(B, dtype=np.float64)
    N = B_np.shape[0]
    batch_size, K = orders.shape[:2]

    tau_start = float(params.get('tau_start', 0.1))
    tau_step = float(params.get('tau_step', 0.1))
    alpha_dep = float(params.get('alpha_dep', 0.5))
    top_k = int(params.get('top_k', 0) or 0)

    source, out_deg, in_deg = compute_source(B_np, alpha_dep)

    betas = {
        'sup': float(params.get('beta_sup', 1.0)),
        'fut': float(params.get('beta_fut', 0.5)),
        'src': float(params.get('beta_src', 0.2)),
        'loc': float(params.get('beta_loc', 0.5)),
    }

    logprobs = torch.zeros(batch_size, K, device=mlp_device)

    for b in range(batch_size):
        for k in range(K):
            order_np = orders[b, k].astype(np.int64)

            # --- Step 0 ---
            features0 = compute_step_features(
                B_np, np.array([], dtype=np.int64), np.arange(N, dtype=np.int64),
                -1, source, out_deg, in_deg, 0, N,
            )
            feats0_t = torch.from_numpy(features0.astype(np.float32)).to(mlp_device)
            residuals0 = mlp(feats0_t)  # (N,)
            rw0 = torch.from_numpy(source.astype(np.float32)).to(mlp_device)
            combined0 = rw0 + residuals0

            p0 = _softmax_torch(combined0, tau_start, top_k=top_k)
            idx0 = int(order_np[0])
            logprobs[b, k] += torch.log(p0[idx0].clamp(min=1e-12))

            # --- Steps 1..N-1 ---
            S = np.array([idx0], dtype=np.int64)
            U = np.setdiff1d(np.arange(N, dtype=np.int64), S)
            last = idx0

            for t in range(1, N):
                chosen = int(order_np[t])
                if chosen not in U:
                    break

                # RW score (NumPy, non-differentiable)
                _, rw_score = progressive_rw_step(
                    B_np, S, U, last, betas, tau_step, source,
                    np.random.default_rng(0), top_k=0,  # no top-k in NumPy step
                )
                rw_score_t = torch.from_numpy(rw_score.astype(np.float32)).to(mlp_device)

                # MLP residual (PyTorch, differentiable)
                features_t = compute_step_features(
                    B_np, S, U, last, source, out_deg, in_deg, t, N,
                )
                feats_t = torch.from_numpy(features_t.astype(np.float32)).to(mlp_device)
                residuals_t = mlp(feats_t)  # (len(U),)

                combined_t = rw_score_t + residuals_t
                p_t = _softmax_torch(combined_t, tau_step, top_k=top_k)

                chosen_idx = int(np.where(U == chosen)[0][0])
                logprobs[b, k] += torch.log(p_t[chosen_idx].clamp(min=1e-12))

                S = np.append(S, chosen)
                U = U[U != chosen]
                last = chosen

    return logprobs


def _softmax_torch(
    scores: torch.Tensor,
    tau: float,
    top_k: int = 0,
) -> torch.Tensor:
    """PyTorch softmax with temperature and optional top-k masking."""
    if top_k > 0 and top_k < scores.numel():
        # mask non-top-k to -inf
        _, top_indices = torch.topk(scores, top_k)
        mask = torch.full_like(scores, float('-inf'))
        mask.scatter_(0, top_indices, scores[top_indices])
        scores = mask
    s_max = scores.max()
    exp_scores = torch.exp((scores - s_max) / tau)
    return exp_scores / exp_scores.sum()
