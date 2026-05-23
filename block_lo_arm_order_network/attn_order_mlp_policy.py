r"""GPU-batched sampler for the distilled Attn-Order MLP policy (Phase 2 continuation).

This wires the Phase-1 distilled per-candidate MLP scorer s_beta(v) = MLP(phi_t(v))
into an on-the-fly order sampler with the SAME interface contract as
`directed_graph_policy.sample_orders_batched_torch` (returns (batch, N) physical
block orders on `device`), so it can be dropped into the clean continuation trainer
in place of `progressive_rw_v3`.

The 12-d features phi_t(v) replicate `attn_order_features.build_features` exactly,
computed in batched torch form (matmuls + masked maxes). A numpy reference
(`score_state_numpy`) reuses the canonical `build_features` + MLP forward and is used
by tests to assert the batched scores match the per-state numpy path.

Three orientations (Phase-2 §1b orientation diagnostic):
  - "original"      : pure MLP scores at every step (incl. t=0).
  - "reversed"      : sample as "original", then reverse each order row.
  - "source_start"  : t=0 starts from softmax(readiness/tau_start) (v3-style source
                      anchor); for t>0, score = z(MLP) + src_rho * z(readiness) over
                      the current candidate set (v3-style readiness direction prior).

Sampling convention matches `train_attn_order_mlp.student_rollout` (standardize=True):
per-step scores are z-scored over the current candidate set U before softmax/tau, so
tau is comparable across modalities and steps. No entropy regularization (entropy is
controlled by tau / top_k at sampling time, per PHASE2_DESIGN_NOTE.md §5).
"""
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from attn_order_features import build_features, NUM_FEATURES
from train_attn_order_mlp import OrderMLP

ORIENTATIONS = ("original", "reversed", "source_start")
_BIG_NEG = -1e30


def load_order_mlp(path, device, hidden: int = 64, layers: int = 2, act: str = "gelu") -> OrderMLP:
    """Load a Phase-1 distilled OrderMLP state_dict (defaults match train_attn_order_mlp)."""
    m = OrderMLP(in_dim=NUM_FEATURES, hidden=hidden, layers=layers, act=act)
    m.load_state_dict(torch.load(str(path), map_location="cpu", weights_only=True))
    m.eval()
    m.to(device)
    return m


def readiness_vector(B_np: np.ndarray, alpha_dep: float = 0.5) -> np.ndarray:
    """r(v) = out(v) - alpha_dep * in(v) (== v3 source/readiness prior)."""
    B = np.asarray(B_np, dtype=np.float64)
    out_deg = B.sum(axis=1)
    in_deg = B.sum(axis=0)
    return out_deg - alpha_dep * in_deg


# ---------------------------------------------------------------------------
# numpy reference (canonical features) — used by tests
# ---------------------------------------------------------------------------

def score_state_numpy(B_np, mlp, S, U, last, t, N) -> np.ndarray:
    """Reference per-candidate MLP scores via the canonical build_features path.

    Returns raw MLP scores aligned to U (np.float64), no standardization/softmax.
    """
    X, cand, _ = build_features(B_np, S, U, last, t, N)
    with torch.no_grad():
        sc = mlp(torch.tensor(X, dtype=torch.float32, device=next(mlp.parameters()).device))
    return sc.detach().cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# batched features (all N candidates at once, garbage for already-selected rows
# is harmless because those columns get masked before sampling)
# ---------------------------------------------------------------------------

def _batched_features(B, BT, selected, last_idx, t, N):
    """Return (Bsz, N, 12) features replicating build_features for every candidate v.

    B, BT: (N, N) float tensors.  selected: (Bsz, N) bool.  last_idx: (Bsz,) long (-1 at t=0).
    All rows share the same t (exactly t nodes selected), so |S|=t, |U\\{v}|=N-t-1.
    """
    device = B.device
    Bsz = selected.size(0)
    feat = torch.zeros(Bsz, N, NUM_FEATURES, device=device, dtype=B.dtype)

    sel_f = selected.to(B.dtype)            # (Bsz, N)
    unsel = (~selected).to(B.dtype)         # (Bsz, N)
    Bexp = B.unsqueeze(0)                   # (1, N, N) over (u, v): B[u,v]
    BTexp = BT.unsqueeze(0)                 # (1, N, N): BT[u,v] = B[v,u]
    sel_u = selected.unsqueeze(-1)          # (Bsz, N, 1) mask over u
    eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)  # u==v

    # A. selected set <-> candidate (features 1-4); zero when |S|=0
    if t > 0:
        feat[..., 0] = (sel_f @ B) / t                       # mean B[S, v]
        feat[..., 2] = (sel_f @ BT) / t                      # mean B[v, S]
        feat[..., 1] = torch.where(sel_u, Bexp, torch.full_like(Bexp, _BIG_NEG)).max(dim=1).values
        feat[..., 3] = torch.where(sel_u, BTexp, torch.full_like(BTexp, _BIG_NEG)).max(dim=1).values

    # B. unselected set \ {v} <-> candidate (features 5-8); zero when |U\{v}|=0
    nUm1 = N - t - 1
    if nUm1 > 0:
        feat[..., 4] = (unsel @ B) / nUm1                    # mean_{u in U\{v}} B[u, v]  (B[v,v]=0)
        feat[..., 6] = (unsel @ BT) / nUm1                   # mean_{u in U\{v}} B[v, u]
        valid_u = (~selected).unsqueeze(-1) & (~eye)         # u in U and u != v
        feat[..., 5] = torch.where(valid_u, Bexp, torch.full_like(Bexp, _BIG_NEG)).max(dim=1).values
        feat[..., 7] = torch.where(valid_u, BTexp, torch.full_like(BTexp, _BIG_NEG)).max(dim=1).values

    # C. last transition (features 9-10); zero where last < 0
    valid_last = last_idx >= 0
    if valid_last.any():
        li = last_idx.clamp(min=0)
        bl = B[li]                                           # (Bsz, N): B[last, v]
        bvl = BT[li]                                         # (Bsz, N): B[v, last]
        vl = valid_last.unsqueeze(-1).to(B.dtype)
        feat[..., 8] = bl * vl
        feat[..., 9] = bvl * vl

    # D. progress (features 11-12)
    feat[..., 10] = float(t) / float(N)
    feat[..., 11] = float(N - t) / float(N)

    # masked maxes can leave _BIG_NEG when a block was skipped — clamp those to 0
    feat = torch.where(feat <= _BIG_NEG / 2.0, torch.zeros_like(feat), feat)
    return feat


def _zscore_over_candidates(scores, selected, t, N):
    """z-score `scores` (Bsz, N) over the unselected candidates per row (mean/std over U)."""
    unsel = (~selected).to(scores.dtype)
    cnt = float(N - t)
    mean = (scores * unsel).sum(dim=1, keepdim=True) / cnt
    var = (((scores - mean) ** 2) * unsel).sum(dim=1, keepdim=True) / cnt
    std = var.clamp_min(0.0).sqrt() + 1e-9
    return (scores - mean) / std


def sample_orders_batched_mlp(
    B_np: np.ndarray,
    batch_size: int,
    mlp: OrderMLP,
    orientation: str,
    base_seed: int,
    device: "torch.device",
    tau: float = 0.5,
    tau_start: float = 0.1,
    top_k: int = 4,
    src_rho: float = 0.0,
    alpha_dep: float = 0.5,
    greedy: bool = False,
    return_entropy: bool = False,
):
    """Sample `batch_size` physical block orders from the distilled MLP policy.

    Returns (batch_size, N) LongTensor on `device`. `orientation` in ORIENTATIONS.
    If `return_entropy`, returns (orders, mean_per_step_entropy: float).
    """
    if orientation not in ORIENTATIONS:
        raise ValueError(f"orientation must be one of {ORIENTATIONS}, got {orientation}")
    N = B_np.shape[0]
    B = torch.from_numpy(np.ascontiguousarray(B_np)).to(device=device, dtype=torch.float32)
    BT = B.t().contiguous()
    readiness = torch.from_numpy(readiness_vector(B_np, alpha_dep)).to(device=device, dtype=torch.float32)

    gen = torch.Generator(device=device)
    gen.manual_seed(int(base_seed))

    orders = torch.zeros(batch_size, N, dtype=torch.long, device=device)
    selected = torch.zeros(batch_size, N, dtype=torch.bool, device=device)
    last_idx = torch.full((batch_size,), -1, dtype=torch.long, device=device)
    src_start = orientation == "source_start"
    ent_sum, ent_steps = 0.0, 0

    for t in range(N):
        if t == 0 and src_start:
            # v3-style source anchor: start from softmax(readiness / tau_start)
            z = _zscore_over_candidates(readiness.unsqueeze(0).expand(batch_size, -1).clone(),
                                        selected, t, N)
            logits = z / max(tau_start, 1e-6)
        else:
            feat = _batched_features(B, BT, selected, last_idx, t, N)
            with torch.no_grad():
                raw = mlp(feat)                                  # (Bsz, N)
            z = _zscore_over_candidates(raw, selected, t, N)
            if src_start and src_rho > 0.0:
                zr = _zscore_over_candidates(
                    readiness.unsqueeze(0).expand(batch_size, -1).clone(), selected, t, N)
                z = z + src_rho * zr
            logits = z / max(tau, 1e-6)

        logits = logits.masked_fill(selected, float("-inf"))

        if top_k and 0 < top_k < N - t:
            kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
            logits = torch.where(logits >= kth, logits, torch.full_like(logits, float("-inf")))

        if greedy:
            choices = torch.argmax(logits, dim=-1)
        else:
            probs = torch.softmax(logits, dim=-1)
            probs = probs.masked_fill(selected, 0.0)
            if return_entropy and t < N - 1:
                ent_sum += float(-(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1).mean())
                ent_steps += 1
            choices = torch.multinomial(probs, num_samples=1, generator=gen).squeeze(-1)

        orders[:, t] = choices
        selected[torch.arange(batch_size, device=device), choices] = True
        last_idx = choices

    if orientation == "reversed":
        orders = torch.flip(orders, dims=[1]).contiguous()
    if return_entropy:
        return orders, (ent_sum / ent_steps if ent_steps else 0.0)
    return orders


def sample_orders_batched_position(
    N: int,
    batch_size: int,
    base_seed: int,
    device: "torch.device",
    pos_tau: float = 0.1,
    top_k: int = 4,
    greedy: bool = False,
    return_entropy: bool = False,
):
    """Position-only order sampler — the entropy-matched attribution control for the
    MLP/source_start arm. Per-step candidate logits depend ONLY on the physical index:
    `logits[v] = -v / pos_tau` (smaller index = more L2R). No graph B, no MLP, no readiness.

    Mirrors `sample_orders_batched_mlp`'s masking / top_k / softmax / multinomial / entropy
    path exactly, so the only difference vs source_start is the score source. Lower `pos_tau`
    => more strictly forward-L2R + lower entropy; calibrate it to match source_start's
    reported avg_step_entropy. Returns (batch_size, N) physical orders on `device`; if
    `return_entropy`, returns (orders, mean_per_step_entropy: float).
    """
    base = -torch.arange(N, device=device, dtype=torch.float32) / max(pos_tau, 1e-6)  # (N,)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(base_seed))

    orders = torch.zeros(batch_size, N, dtype=torch.long, device=device)
    selected = torch.zeros(batch_size, N, dtype=torch.bool, device=device)
    ent_sum, ent_steps = 0.0, 0

    for t in range(N):
        logits = base.unsqueeze(0).expand(batch_size, -1).clone()
        logits = logits.masked_fill(selected, float("-inf"))

        if top_k and 0 < top_k < N - t:
            kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
            logits = torch.where(logits >= kth, logits, torch.full_like(logits, float("-inf")))

        if greedy:
            choices = torch.argmax(logits, dim=-1)
        else:
            probs = torch.softmax(logits, dim=-1)
            probs = probs.masked_fill(selected, 0.0)
            if return_entropy and t < N - 1:
                ent_sum += float(-(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1).mean())
                ent_steps += 1
            choices = torch.multinomial(probs, num_samples=1, generator=gen).squeeze(-1)

        orders[:, t] = choices
        selected[torch.arange(batch_size, device=device), choices] = True

    if return_entropy:
        return orders, (ent_sum / ent_steps if ent_steps else 0.0)
    return orders
