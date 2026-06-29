"""Frozen Old Step-Policy Prior + AO-GPT Local-Utility Reranker.

Core components:
- OldONPrior: wraps frozen CrossAttentionOrderNetwork for feature extraction
- build_reranker_features(): per-candidate 7-dim features
- StepWiseGreedyReranker: no-training linear combination of normalized scores
- StepWiseMLPReranker: trained MLP adapter (7→64→64→1)
- sequential_generate(): autoregressive order generation with any reranker
- compute_candidate_prefix_nll(): per-step AOGPT candidate-local NLL labels
- compute_swap_metrics(): diagnostic order quality metrics
"""

import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

from order_network import (
    CrossAttentionOrderNetwork,
    masks_to_revealed_bool,
)

# ── Model loading ──────────────────────────────────────────────────────────────

def load_old_on(ckpt_path: str, device: str = "cpu") -> CrossAttentionOrderNetwork:
    """Load frozen CrossAttentionOrderNetwork. Infers d_edge, d_model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]
    for k in list(sd.keys()):
        sd[k.replace("_orig_mod.", "")] = sd.pop(k)
    d_edge = sd["edge_mlp.0.weight"].shape[0]
    d_model = sd["score_mlp.0.weight"].shape[0]
    model = CrossAttentionOrderNetwork(num_blocks=16, d_edge=d_edge, d_model=d_model)
    model.load_state_dict(sd)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_aogpt(ckpt_path: str, device: str = "cpu"):
    """Load frozen AO-GPT. Returns (model, block_perm, inv_perm)."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "AO-GPT-MDM"))
    from model_AOGPT_AdaLN6_NoRep_cond_128_trunc_qknorm import AOGPT, AOGPTConfig

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sig = list(AOGPTConfig.__init__.__code__.co_varnames)
    valid = {k: v for k, v in dict(ckpt["model_args"]).items() if k in sig}
    model = AOGPT(AOGPTConfig(**valid))
    sd = ckpt["model"]
    for k in list(sd.keys()):
        sd[k.replace("_orig_mod.", "")] = sd.pop(k)
    model.load_state_dict(sd)
    model.crop_block_size(256)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    bp = torch.tensor(ckpt["data_permutation"]["block_perm"], dtype=torch.long)
    inv_perm = torch.tensor(ckpt["data_permutation"]["inverse_block_perm"], dtype=torch.long)
    return model, bp, inv_perm


# ── Token order utilities ──────────────────────────────────────────────────────

def n16_order_to_token_order(
    n16_order: torch.Tensor,
    block_perm: torch.Tensor,
    sub_blocks: int = 4,
    block_len: int = 4,
) -> torch.Tensor:
    """Convert physical N16 block order → model-coordinate token order (256,).

    n16_order: (N16,) physical N16 block indices in reveal order
    block_perm: (64,) block_perm[phys64] = model64
    Returns: (256,) token indices in model coordinates
    """
    N16 = n16_order.shape[0]
    T = N16 * sub_blocks * block_len
    token_order = torch.zeros(T, dtype=torch.long, device=n16_order.device)
    for t in range(N16):
        phys_n16 = n16_order[t].item()
        for a in range(sub_blocks):
            phys_n64 = phys_n16 * sub_blocks + a
            model_n64 = block_perm[phys_n64].item()
            for k in range(block_len):
                token_order[t * sub_blocks * block_len + a * block_len + k] = (
                    model_n64 * block_len + k
                )
    return token_order


def n16_batch_to_token_orders(
    n16_orders: torch.Tensor,
    block_perm: torch.Tensor,
    sub_blocks: int = 4,
    block_len: int = 4,
) -> torch.Tensor:
    """Convert batched N16 orders → batched token orders.

    n16_orders: (B, N16) physical N16 block indices
    Returns: (B, 256) token indices in model coordinates
    """
    B, N16 = n16_orders.shape
    T = N16 * sub_blocks * block_len
    token_orders = torch.zeros(B, T, dtype=torch.long, device=n16_orders.device)
    bp = block_perm.to(n16_orders.device)
    for t in range(N16):
        phys_n16 = n16_orders[:, t]  # (B,)
        for a in range(sub_blocks):
            phys_n64 = phys_n16 * sub_blocks + a
            model_n64 = bp[phys_n64]  # (B,)
            for k in range(block_len):
                token_orders[:, t * sub_blocks * block_len + a * block_len + k] = (
                    model_n64 * block_len + k
                )
    return token_orders


# ── Old ON Prior ───────────────────────────────────────────────────────────────

class OldONPrior:
    """Wraps frozen CrossAttentionOrderNetwork for safe feature extraction."""

    def __init__(self, model: CrossAttentionOrderNetwork, num_blocks: int = 16):
        self.model = model
        self.num_blocks = num_blocks

    @torch.no_grad()
    def get_logits(
        self, A: torch.Tensor, visited_masks: torch.Tensor, last_nodes: torch.Tensor
    ) -> torch.Tensor:
        """Raw logits from cross-attention ON. visited positions = -inf."""
        return self.model(A, visited_masks, last_nodes)

    @torch.no_grad()
    def get_logits_safe(
        self, A: torch.Tensor, visited_masks: torch.Tensor, last_nodes: torch.Tensor
    ) -> torch.Tensor:
        """Z-score normalized logits within unvisited candidates. Visited = 0."""
        logits = self.get_logits(A, visited_masks, last_nodes)  # (B, N), visited=-inf
        revealed_mask = masks_to_revealed_bool(visited_masks, self.num_blocks)  # (B, N)

        # Replace -inf with 0 for finite-only stats (avoid -inf * 0 = NaN)
        finite_mask = torch.isfinite(logits)
        logits_finite = torch.where(finite_mask, logits, torch.zeros_like(logits))
        count = finite_mask.float().sum(dim=-1, keepdim=True).clamp(min=1)
        mean = logits_finite.sum(dim=-1, keepdim=True) / count
        var = ((logits_finite - mean) * finite_mask.float()).pow(2).sum(dim=-1, keepdim=True) / count
        std = var.sqrt().clamp(min=1e-8)
        z_scored = (logits_finite - mean) / std
        z_scored[revealed_mask] = 0.0
        return z_scored

    @torch.no_grad()
    def get_full_order(self, A_batch: torch.Tensor) -> torch.Tensor:
        """Autoregressive greedy decode full order for each A matrix.

        Returns: (B, N) long tensor of block indices in reveal order.
        """
        B, N, _ = A_batch.shape
        device = A_batch.device

        orders = torch.zeros(B, N, dtype=torch.long, device=device)
        visited_masks = torch.zeros(B, dtype=torch.long, device=device)
        last_nodes = torch.zeros(B, dtype=torch.long, device=device)

        for step in range(N):
            logits = self.get_logits(A_batch, visited_masks, last_nodes)
            revealed = masks_to_revealed_bool(visited_masks, N)
            logits[revealed] = float("-inf")
            chosen = logits.argmax(dim=-1)
            orders[:, step] = chosen
            visited_masks = visited_masks | (1 << chosen)
            last_nodes = chosen

        return orders


# ── Reranker features ──────────────────────────────────────────────────────────

def build_reranker_features(
    A: torch.Tensor,
    visited_masks: torch.Tensor,
    last_nodes: torch.Tensor,
    old_on_prior: OldONPrior,
    sigma_old: Optional[torch.Tensor] = None,
    step_t: int = 0,
) -> torch.Tensor:
    """Build 7-dim per-candidate reranker features.

    Features:
      0: on_logit_i          — old ON raw logit (unvisited only, visited=0)
      1: on_rank_distance_i  — -|rank_old(i) - t| / N  (requires sigma_old)
      2: A_last_i            — edge last → candidate
      3: A_i_last            — edge candidate → last
      4: row_mean_i          — mean A[i,:] excl diag
      5: col_mean_i          — mean A[:,i] excl diag
      6: step_norm           — t / N

    Args:
        A: (B, N, N) attention matrices
        visited_masks: (B,) integer bitmask
        last_nodes: (B,) long tensor
        old_on_prior: OldONPrior instance
        sigma_old: (B, N) old ON full order (needed for rank_distance feature)
        step_t: current step index (0 to N-1)

    Returns:
        features: (B, N, 7), visited positions = 0.
    """
    B, N, _ = A.shape
    device = A.device

    # Feature 0: old ON logits (safe/z-scored)
    on_logits = old_on_prior.get_logits_safe(A, visited_masks, last_nodes)  # (B, N)
    revealed_mask = masks_to_revealed_bool(visited_masks, N)

    # Feature 1: on_rank_distance
    if sigma_old is not None:
        rank_old = torch.zeros(B, N, dtype=torch.float32, device=device)
        for b in range(B):
            for pos, blk in enumerate(sigma_old[b]):
                rank_old[b, blk] = float(pos)
        on_rank_distance = -torch.abs(rank_old - step_t) / N
    else:
        on_rank_distance = torch.zeros(B, N, device=device)

    # Feature 2, 3: A_last_i, A_i_last
    A_safe = torch.where(torch.isfinite(A), A, torch.zeros((), device=device, dtype=A.dtype))
    idx_last = last_nodes.view(B, 1, 1).expand(-1, N, 1)
    A_last_i = A_safe.gather(dim=-1, index=idx_last).squeeze(-1)  # (B, N)
    idx_from_last = last_nodes.view(B, 1, 1).expand(-1, 1, N)
    A_i_last = A_safe.gather(dim=1, index=idx_from_last).squeeze(1)  # (B, N)

    # Feature 4, 5: row_mean, col_mean (excl diag)
    not_self = (1 - torch.eye(N, device=device)).unsqueeze(0)
    row_sum = (A_safe * not_self).sum(dim=-1)
    col_sum = (A_safe * not_self).sum(dim=1)
    row_mean = row_sum / (N - 1)
    col_mean = col_sum / (N - 1)

    # Feature 6: step_norm
    step_norm = torch.full((B, N), step_t / max(N - 1, 1), device=device)

    features = torch.stack([
        on_logits,
        on_rank_distance,
        A_last_i,
        A_i_last,
        row_mean,
        col_mean,
        step_norm,
    ], dim=-1)  # (B, N, 7)

    features[revealed_mask] = 0.0
    return features


def build_reranker_features_v2(
    A: torch.Tensor,
    visited_masks: torch.Tensor,
    last_nodes: torch.Tensor,
    old_on_prior: OldONPrior,
    sigma_old: Optional[torch.Tensor] = None,
    step_t: int = 0,
    feature_set: str = "v2_basic",
) -> torch.Tensor:
    """Build per-candidate reranker features (v2, selective normalization).

    Normalization rule:
      - Z-score (per-step over unvisited): on_logit, A_last_i (continuous, cross-candidate)
      - No z-score: step_norm, remaining_count_norm, rank features (fixed scale / shared across candidates)

    v2_basic features (4 dims):
      0: on_logit_z            — z-scored ON logit from get_logits_safe
      1: A_last_i_z            — z-scored edge weight last→candidate
      2: remaining_count_norm  — K / N (current unvisited count)
      3: step_norm             — t / N

    v2_rank features (7 dims):
      0: on_logit_z
      1: A_last_i_z
      2: A_last_rank_norm      — rank(A_last_i among unvisited) / (K-1), 0=strongest
      3: old_on_rank_norm      — rank_old(i) / (N-1)
      4: rank_distance         — -abs(rank_old(i) - t) / N
      5: remaining_count_norm  — K / N
      6: step_norm             — t / N

    v3_qvalue features (11 dims) = v2_rank (7) + 1 prior + 3 future-graph features:
      7: is_old_on_next       — 1.0 if candidate == σ_old[t], else 0.0 (no z-score)
      8: cand_out_rem_mean     — mean_{r∈R} A[candidate, r], z-scored (outgoing to remaining)
      9: cand_out_rem_max      — max_{r∈R} A[candidate, r], z-scored (best outgoing edge)
     10: rem_to_cand_mean      — mean_{r∈R} A[r, candidate], z-scored (incoming from remaining)
      R = remaining = unvisited \ {candidate}. All z-scored over unvisited candidates.

    Args:
        A: (B, N, N) attention matrices
        visited_masks: (B,) integer bitmask
        last_nodes: (B,) long tensor
        old_on_prior: OldONPrior instance
        sigma_old: (B, N) old ON full order (needed for rank features)
        step_t: current step index (0 to N-1)
        feature_set: "v2_basic", "v2_rank", or "v3_qvalue"

    Returns:
        features: (B, N, F) where F=4 for v2_basic, F=7 for v2_rank, F=11 for v3_qvalue
    """
    B, N, _ = A.shape
    device = A.device
    revealed_mask = masks_to_revealed_bool(visited_masks, N)  # (B, N)
    candidates = ~revealed_mask  # (B, N)
    K = candidates.float().sum(dim=-1, keepdim=True)  # (B, 1)

    # Helper: per-sample z-score over unvisited candidates only
    def _zscore(v, mask):
        """Z-score v over mask=True entries per sample. mask=False entries get 0."""
        v_clean = torch.where(mask, v, torch.zeros_like(v))
        count = mask.float().sum(dim=-1, keepdim=True).clamp(min=1)
        mean = v_clean.sum(dim=-1, keepdim=True) / count  # (B, 1)
        # mean is (B,1), v is (B,N) — broadcasts correctly, no unsqueeze needed
        diff = (v - mean) * mask.float()
        var = diff.pow(2).sum(dim=-1, keepdim=True) / count
        std = var.sqrt().clamp(min=1e-8)
        z = (v - mean) / std
        return torch.where(mask, z, torch.zeros_like(v))

    # f0: on_logit (already z-scored within unvisited by get_logits_safe)
    on_logits = old_on_prior.get_logits_safe(A, visited_masks, last_nodes)  # (B, N)

    # f1: A_last_i (z-scored over unvisited)
    A_safe = torch.where(torch.isfinite(A), A, torch.zeros((), device=device, dtype=A.dtype))
    idx_last = last_nodes.view(B, 1, 1).expand(-1, N, 1)
    A_last_i = A_safe.gather(dim=-1, index=idx_last).squeeze(-1)  # (B, N)
    A_last_i_z = _zscore(A_last_i, candidates)

    # Common non-z-scored features
    step_norm = torch.full((B, N), step_t / max(N - 1, 1), device=device)
    remaining_count_norm = K.squeeze(-1).unsqueeze(-1).expand(-1, N) / N  # (B, N)

    # ── Rank features (shared by v2_rank, v3_qvalue) ──
    if feature_set in ("v2_rank", "v3_qvalue"):
        # A_last_rank_norm: rank by A_last_i descending, 0 = strongest
        A_last_rank = torch.zeros(B, N, device=device)
        for b in range(B):
            unvisited_idx = candidates[b].nonzero(as_tuple=True)[0]
            if len(unvisited_idx) > 0:
                vals = A_last_i[b, unvisited_idx]
                _, sort_idx = vals.sort(descending=True)
                ranks = torch.zeros(len(unvisited_idx), device=device)
                ranks[sort_idx] = torch.arange(len(unvisited_idx), dtype=torch.float32, device=device)
                A_last_rank[b, unvisited_idx] = ranks
        A_last_rank_norm = A_last_rank / (K.squeeze(-1).unsqueeze(-1) - 1).clamp(min=1)

        if sigma_old is not None:
            rank_old = torch.zeros(B, N, dtype=torch.float32, device=device)
            for b in range(B):
                for pos, blk in enumerate(sigma_old[b]):
                    rank_old[b, blk] = float(pos)
            old_on_rank_norm = rank_old / (N - 1)
            rank_distance = -torch.abs(rank_old - step_t) / N
        else:
            old_on_rank_norm = torch.zeros(B, N, device=device)
            rank_distance = torch.zeros(B, N, device=device)

    # ── Future remaining graph features (v3_qvalue only) ──
    if feature_set == "v3_qvalue":
        K_per_sample = K.squeeze(-1)  # (B,)
        K_rem = (K_per_sample - 1).clamp(min=1).unsqueeze(-1).expand(-1, N)

        # cand_out_rem: A[candidate, r] for r in remaining
        A_out_col_masked = A_safe * candidates.unsqueeze(-2)  # preserve cols where col∈remaining
        cand_out_sum = A_out_col_masked.sum(dim=-1)  # (B, N)
        diag_A = A_safe.diagonal(dim1=-2, dim2=-1)  # (B, N)
        cand_out_rem_sum = cand_out_sum - diag_A * candidates.float()
        cand_out_rem_mean_raw = cand_out_rem_sum / K_rem

        A_out_for_max = A_out_col_masked.clone()
        A_out_for_max.diagonal(dim1=-2, dim2=-1).fill_(float('-inf'))
        cand_out_rem_max_raw = A_out_for_max.max(dim=-1).values  # (B, N)

        # rem_to_cand_mean: A[r, candidate] for r in remaining
        A_in_row_masked = A_safe * candidates.unsqueeze(-1)  # preserve rows where row∈remaining
        rem_to_cand_sum = A_in_row_masked.sum(dim=-2)  # (B, N)
        rem_to_cand_rem_sum = rem_to_cand_sum - diag_A * candidates.float()
        rem_to_cand_mean_raw = rem_to_cand_rem_sum / K_rem

        # Z-score over unvisited
        cand_out_rem_mean = _zscore(cand_out_rem_mean_raw, candidates)
        cand_out_rem_max = _zscore(cand_out_rem_max_raw, candidates)
        rem_to_cand_mean = _zscore(rem_to_cand_mean_raw, candidates)

        # is_old_on_next: binary, 1.0 if candidate == σ_old[step_t]
        if sigma_old is not None:
            old_on_next = sigma_old[:, step_t]  # (B,) — block index old ON wants at step t
            is_old_on_next = (torch.arange(N, device=device).unsqueeze(0) == old_on_next.unsqueeze(1)).float()
        else:
            is_old_on_next = torch.zeros(B, N, device=device)

    # ── Assemble feature list ──
    if feature_set == "v2_basic":
        feature_list = [
            on_logits, A_last_i_z,
            remaining_count_norm, step_norm,
        ]
    elif feature_set == "v2_rank":
        feature_list = [
            on_logits, A_last_i_z,
            A_last_rank_norm, old_on_rank_norm, rank_distance,
            remaining_count_norm, step_norm,
        ]
    elif feature_set == "v3_qvalue":
        feature_list = [
            on_logits, A_last_i_z,
            A_last_rank_norm, old_on_rank_norm, rank_distance,
            remaining_count_norm, step_norm,
            is_old_on_next,
            cand_out_rem_mean, cand_out_rem_max, rem_to_cand_mean,
        ]
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    features = torch.stack(feature_list, dim=-1)  # (B, N, F)
    features[revealed_mask] = 0.0
    return features


# ── Reranker models ────────────────────────────────────────────────────────────

class StepWiseGreedyReranker:
    """No-training linear combination of normalized scores.

    score_i = α * norm(on_logit_i) + β * norm(A_last_i) + γ * norm(A_i_last)
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.25, gamma: float = 0.25):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    @torch.no_grad()
    def score(
        self,
        features: torch.Tensor,
        revealed_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined scores for all candidates.

        Args:
            features: (B, N, 7) from build_reranker_features
            revealed_mask: (B, N) bool

        Returns:
            scores: (B, N), visited = -inf
        """
        on_logit = features[..., 0]   # already z-scored in build step
        A_last_i = features[..., 2]
        A_i_last = features[..., 3]

        scores = torch.zeros_like(on_logit)
        candidates = ~revealed_mask

        if candidates.any():
            # Per-sample z-score over candidates for each feature
            def _zscore(v, mask):
                m = v * mask.float()
                mean = m.sum(dim=-1, keepdim=True) / mask.float().sum(dim=-1, keepdim=True).clamp(min=1)
                var = ((v - mean) * mask.float()).pow(2).sum(dim=-1, keepdim=True) / mask.float().sum(dim=-1, keepdim=True).clamp(min=1)
                std = var.sqrt().clamp(min=1e-8)
                return (v - mean) / std

            scores = (
                self.alpha * _zscore(on_logit, candidates)
                + self.beta * _zscore(A_last_i, candidates)
                + self.gamma * _zscore(A_i_last, candidates)
            )

        scores[revealed_mask] = float("-inf")
        return scores


class StepWiseMLPReranker(nn.Module):
    """MLP adapter: 7-dim features → 64 → 64 → 1 scalar score."""

    def __init__(
        self,
        feature_dim: int = 7,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        in_dim = feature_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        features: torch.Tensor,
        revealed_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Score all candidates. Visited = -inf.

        Args:
            features: (B, N, 7)
            revealed_mask: (B, N) bool

        Returns:
            scores: (B, N)
        """
        B, N, F = features.shape
        flat = features.reshape(B * N, F)
        scores = self.mlp(flat).squeeze(-1).reshape(B, N)
        scores = scores.masked_fill(revealed_mask, float("-inf"))
        return scores

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class StepWisePooledMLPReranker(nn.Module):
    """Candidate-set aware pooled MLP reranker.

    Architecture:
      1. h_i = MLP_local(f_i)              (K, D)  — per-candidate embedding
      2. g_mean = mean(h over unvisited)    (D,)    — pool over valid only
      3. g_max  = max(h over unvisited)     (D,)    — pool over valid only
      4. g = cat[g_mean, g_max]            (2D,)
      5. h_aug_i = cat[h_i, g]            (K, D+2D) — augmented per-candidate
      6. score_i = MLP_head(h_aug_i)       (K, 1)   — final score

    Pooling excludes visited candidates (not just score-masked).
    """

    def __init__(
        self,
        feature_dim: int = 7,
        local_hidden_dim: int = 32,
        local_num_layers: int = 1,
        head_hidden_dim: int = 64,
        head_num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        # MLP_local: feature_dim → local_hidden_dim
        local_layers = []
        in_dim = feature_dim
        for _ in range(local_num_layers):
            local_layers.append(nn.Linear(in_dim, local_hidden_dim))
            local_layers.append(nn.ReLU())
            local_layers.append(nn.Dropout(dropout))
            in_dim = local_hidden_dim
        self.mlp_local = nn.Sequential(*local_layers) if local_layers else nn.Identity()
        self.local_out_dim = local_hidden_dim if local_layers else feature_dim

        # MLP_head: (local_out_dim + 2*local_out_dim) → head_hidden_dim → 1
        head_input_dim = self.local_out_dim + 2 * self.local_out_dim
        head_layers = []
        in_dim = head_input_dim
        for i in range(head_num_layers):
            head_layers.append(nn.Linear(in_dim, head_hidden_dim))
            head_layers.append(nn.ReLU())
            head_layers.append(nn.Dropout(dropout))
            in_dim = head_hidden_dim
        head_layers.append(nn.Linear(head_hidden_dim, 1))
        self.mlp_head = nn.Sequential(*head_layers)

    def forward(
        self,
        features: torch.Tensor,
        revealed_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Score all candidates with set-pooled context.

        Args:
            features: (B, N, F)
            revealed_mask: (B, N) bool, True = visited

        Returns:
            scores: (B, N), visited = -inf
        """
        B, N, F = features.shape
        valid_mask = ~revealed_mask  # (B, N), True = unvisited

        # Per-candidate local embeddings
        flat = features.reshape(B * N, F)
        h_flat = self.mlp_local(flat)  # (B*N, local_out_dim)
        h = h_flat.reshape(B, N, self.local_out_dim)  # (B, N, D)

        # Pool only over unvisited candidates
        # Mean pool
        h_masked = h * valid_mask.unsqueeze(-1).float()  # (B, N, D)
        valid_count = valid_mask.float().sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
        g_mean = h_masked.sum(dim=1) / valid_count  # (B, D)

        # Max pool
        h_for_max = h.masked_fill(~valid_mask.unsqueeze(-1), float('-inf'))
        g_max = h_for_max.max(dim=1).values  # (B, D)

        # Global context
        g = torch.cat([g_mean, g_max], dim=-1)  # (B, 2D)
        g_expanded = g.unsqueeze(1).expand(-1, N, -1)  # (B, N, 2D)

        # Augmented per-candidate
        h_aug = torch.cat([h, g_expanded], dim=-1)  # (B, N, 3D)

        # Score head
        h_aug_flat = h_aug.reshape(B * N, -1)
        scores = self.mlp_head(h_aug_flat).squeeze(-1).reshape(B, N)

        scores = scores.masked_fill(revealed_mask, float('-inf'))
        return scores

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Sequential generation ──────────────────────────────────────────────────────

@torch.no_grad()
def sequential_generate(
    reranker,
    A: torch.Tensor,
    old_on_prior: OldONPrior,
    sigma_old: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    use_greedy_reranker: bool = False,
    feature_set: str = "v1_legacy",
) -> torch.Tensor:
    """Autoregressively generate a full ordering.

    At each step: get old ON features → build reranker features → score → sample.

    Args:
        reranker: StepWiseMLPReranker, StepWisePooledMLPReranker, or StepWiseGreedyReranker
        A: (B, N, N)
        old_on_prior: OldONPrior
        sigma_old: (B, N) pre-computed old ON full order (optional, for rank_distance)
        temperature: softmax temperature
        use_greedy_reranker: True if reranker is StepWiseGreedyReranker
        feature_set: "v1_legacy", "v2_basic", or "v2_rank"

    Returns:
        orders: (B, N) long tensor
    """
    B, N, _ = A.shape
    device = A.device

    orders = torch.zeros(B, N, dtype=torch.long, device=device)
    visited_masks = torch.zeros(B, dtype=torch.long, device=device)
    last_nodes = torch.zeros(B, dtype=torch.long, device=device)

    for step in range(N):
        if feature_set == "v1_legacy":
            features = build_reranker_features(
                A, visited_masks, last_nodes, old_on_prior,
                sigma_old=sigma_old, step_t=step,
            )
        else:
            features = build_reranker_features_v2(
                A, visited_masks, last_nodes, old_on_prior,
                sigma_old=sigma_old, step_t=step, feature_set=feature_set,
            )
        revealed_mask = masks_to_revealed_bool(visited_masks, N)

        if use_greedy_reranker:
            scores = reranker.score(features, revealed_mask)
        else:
            scores = reranker(features, revealed_mask)

        if temperature < 1e-8:
            chosen = scores.argmax(dim=-1)
        else:
            safe_scores = torch.where(
                torch.isfinite(scores),
                scores / temperature,
                torch.tensor(float('-inf'), device=device, dtype=scores.dtype),
            )
            probs = F.softmax(safe_scores, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            probs[revealed_mask] = 0.0
            prob_sum = probs.sum(dim=-1, keepdim=True)
            # If all probs zero (degenerate scores), fall back to uniform over unvisited
            uniform = (~revealed_mask).float()
            uniform = uniform / uniform.sum(dim=-1, keepdim=True).clamp(min=1)
            probs = torch.where(prob_sum > 1e-9, probs / prob_sum.clamp(min=1e-9), uniform)
            chosen = torch.multinomial(probs, num_samples=1).squeeze(-1)

        orders[:, step] = chosen
        visited_masks = visited_masks | (1 << chosen)
        last_nodes = chosen

    return orders


# ── Per-step candidate-local NLL (AOGPT label generation) ──────────────────────

@torch.no_grad()
def compute_candidate_prefix_nll(
    aogpt,
    idx: torch.Tensor,
    block_perm: torch.Tensor,
    prefix_order: torch.Tensor,
    candidate: torch.Tensor,
    rest_order: torch.Tensor,
    sub_blocks: int = 4,
    block_len: int = 4,
    k: int = 4,
) -> torch.Tensor:
    """Compute candidate-local NLL for ONE (seq, candidate) pair.

    Only the candidate block's first-k tokens contribute to loss.
    Rest tokens provide shape/context only.

    Args:
        aogpt: frozen AO-GPT model
        idx: (1, 256) token ids
        block_perm: (64,) block_perm tensor
        prefix_order: (prefix_len,) N16 blocks already placed
        candidate: scalar N16 block index
        rest_order: (N-prefix_len-1,) remaining N16 blocks (any valid order)
        sub_blocks: N64 sub-blocks per N16
        block_len: tokens per N64 sub-block
        k: first-k tokens of candidate block for loss

    Returns:
        nll: scalar tensor
    """
    prefix_len = prefix_order.shape[0]
    N = prefix_len + 1 + rest_order.shape[0]
    device = idx.device

    full_order = torch.cat([prefix_order, candidate.unsqueeze(0), rest_order])
    token_order = n16_order_to_token_order(full_order, block_perm, sub_blocks, block_len)
    token_order = token_order.unsqueeze(0)  # (1, 256)

    logits, _ = aogpt.forward_fn(idx, token_order)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_targets = aogpt.shuffle(idx, token_order)
    token_losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_targets.view(-1),
        ignore_index=-1,
        reduction="none",
    ).view(1, -1)  # (1, 256)

    c_start = prefix_len * sub_blocks * block_len
    c_end = c_start + sub_blocks * block_len
    candidate_losses = token_losses[:, c_start:c_end]  # (1, 16)
    return candidate_losses[:, :k].mean()


@torch.no_grad()
def compute_batched_candidate_nlls(
    aogpt,
    idx: torch.Tensor,
    block_perm: torch.Tensor,
    prefix_order: torch.Tensor,
    candidates: torch.Tensor,
    rest_order: torch.Tensor,
    chunk_size: int = 64,
    sub_blocks: int = 4,
    block_len: int = 4,
    k: int = 4,
) -> torch.Tensor:
    """Batch-evaluate candidate-local NLL for K candidates of one sequence.

    Processes in chunks to avoid OOM.

    Args:
        aogpt: frozen AO-GPT
        idx: (1, 256) token ids for this sequence
        block_perm: (64,) block_perm[phys64] = model64
        prefix_order: (prefix_len,) already-placed N16 blocks
        candidates: (K,) candidate N16 block indices to evaluate
        rest_order: (N-prefix_len-1,) remaining N16 blocks (L2R default)
        chunk_size: max candidates per forward pass

    Returns:
        nlls: (K,) mean candidate-local NLL per candidate
    """
    K = len(candidates)
    prefix_len = prefix_order.shape[0]
    device = idx.device

    all_nlls = []
    for chunk_start in range(0, K, chunk_size):
        chunk_end = min(chunk_start + chunk_size, K)
        chunk_candidates = candidates[chunk_start:chunk_end]
        C = chunk_end - chunk_start

        # Build full orders for chunk candidates (always N blocks)
        full_orders = torch.zeros(C, prefix_order.shape[0] + 1 + rest_order.shape[0] - 1, dtype=torch.long, device=device)
        for i, cand in enumerate(chunk_candidates):
            rest_filtered = rest_order[rest_order != cand]
            full_orders[i] = torch.cat([prefix_order, cand.unsqueeze(0), rest_filtered])

        # Expand to token orders (batched, avoids per-candidate Python loop)
        token_orders = n16_batch_to_token_orders(
            full_orders, block_perm, sub_blocks, block_len
        )

        idx_batch = idx.expand(C, -1)
        logits, _ = aogpt.forward_fn(idx_batch, token_orders)

        # Compute per-token losses manually
        shift_logits = logits[..., :-1, :].contiguous()
        shift_targets = aogpt.shuffle(idx_batch, token_orders)
        token_losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.view(-1),
            ignore_index=-1,
            reduction="none",
        ).view(C, -1)  # (C, 256)

        c_start = prefix_len * sub_blocks * block_len
        candidate_losses = token_losses[:, c_start:c_start + sub_blocks * block_len]
        chunk_nlls = candidate_losses[:, :k].mean(dim=1)  # (C,)
        all_nlls.append(chunk_nlls)

    return torch.cat(all_nlls)


# ── AO-GPT full-order NLL ──────────────────────────────────────────────────────

@torch.no_grad()
def compute_full_order_nll(
    aogpt,
    idx: torch.Tensor,
    block_perm: torch.Tensor,
    n16_order: torch.Tensor,
    sub_blocks: int = 4,
    block_len: int = 4,
) -> float:
    """Compute full-order AO-GPT NLL for a single sequence.

    Args:
        aogpt: frozen AO-GPT
        idx: (1, 256) token ids
        block_perm: (64,) block_perm tensor
        n16_order: (16,) N16 block order

    Returns:
        mean_nll: scalar float
    """
    token_order = n16_order_to_token_order(n16_order, block_perm, sub_blocks, block_len)
    token_order = token_order.unsqueeze(0)
    _, loss = aogpt.forward_fn(idx, token_order)
    return loss.item()


@torch.no_grad()
def compute_batch_full_order_nll(
    aogpt,
    idx_batch: torch.Tensor,
    block_perm: torch.Tensor,
    n16_orders: torch.Tensor,
    sub_blocks: int = 4,
    block_len: int = 4,
) -> torch.Tensor:
    """Batch full-order AOGPT NLL.

    Args:
        aogpt: frozen AO-GPT
        idx_batch: (B, 256) token ids
        block_perm: (64,) block_perm tensor
        n16_orders: (B, 16) N16 block orders

    Returns:
        nlls: (B,) mean NLL per sequence
    """
    token_orders = n16_batch_to_token_orders(n16_orders, block_perm, sub_blocks, block_len)
    logits, _ = aogpt.forward_fn(idx_batch, token_orders)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_targets = aogpt.shuffle(idx_batch, token_orders)
    token_losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_targets.view(-1),
        ignore_index=-1,
        reduction="none",
    ).view(idx_batch.shape[0], -1)
    return token_losses.mean(dim=1)


# ── Order quality metrics ──────────────────────────────────────────────────────

def compute_kendall_tau(pred_order: torch.Tensor, ref_order: torch.Tensor) -> float:
    """Kendall tau correlation between two orderings.

    Args:
        pred_order: (N,) predicted order
        ref_order: (N,) reference order

    Returns:
        tau: float in [-1, 1]
    """
    N = pred_order.shape[0]
    pred_rank = torch.zeros(N, dtype=torch.long)
    ref_rank = torch.zeros(N, dtype=torch.long)
    for i in range(N):
        pred_rank[pred_order[i]] = i
        ref_rank[ref_order[i]] = i

    concordant = 0
    discordant = 0
    for i in range(N):
        for j in range(i + 1, N):
            pi, pj = pred_rank[i], pred_rank[j]
            ri, rj = ref_rank[i], ref_rank[j]
            if (pi < pj and ri < rj) or (pi > pj and ri > rj):
                concordant += 1
            else:
                discordant += 1

    total = concordant + discordant
    return (concordant - discordant) / max(total, 1)


def compute_swap_metrics(
    pred_order: torch.Tensor,
    ref_order: torch.Tensor,
) -> Dict:
    """Compute pairwise order accuracy and adjacent swap repair rate.

    Args:
        pred_order: (N,) predicted order
        ref_order: (N,) reference order (A-DP teacher)

    Returns:
        dict with keys:
            pairwise_acc: fraction of pairs where relative order matches
            adjacent_repair: fraction of adjacent swaps in ref that are
                            repaired in pred (pred reverses the ref adjacent pair)
    """
    N = pred_order.shape[0]

    pred_rank = torch.zeros(N, dtype=torch.long)
    ref_rank = torch.zeros(N, dtype=torch.long)
    for i in range(N):
        pred_rank[pred_order[i]] = i
        ref_rank[ref_order[i]] = i

    # Pairwise accuracy
    correct = 0
    total = 0
    for i in range(N):
        for j in range(i + 1, N):
            total += 1
            if (pred_rank[i] < pred_rank[j]) == (ref_rank[i] < ref_rank[j]):
                correct += 1
    pairwise_acc = correct / max(total, 1)

    # Adjacent repair: find pairs that are adjacent in ref, check if pred reverses
    repairs = 0
    repair_total = 0
    for t in range(N - 1):
        a, b = ref_order[t].item(), ref_order[t + 1].item()
        repair_total += 1
        # Repair = pred places b before a (i.e., the reverse of ref)
        if pred_rank[b] < pred_rank[a]:
            repairs += 1
    adjacent_repair = repairs / max(repair_total, 1)

    return {
        "pairwise_acc": pairwise_acc,
        "adjacent_repair": adjacent_repair,
    }


def check_order_valid(order: torch.Tensor, N: int = 16) -> bool:
    """Verify order is a valid permutation: no duplicates, no omissions."""
    return len(set(order.tolist())) == N and min(order).item() == 0 and max(order).item() == N - 1


# ── Structural attention labels (NLL-free) ─────────────────────────────────────

def compute_prefix_score_np(
    A: "np.ndarray",
    prefix: "np.ndarray",
    candidate: int,
    mode: str = "edge",
    hybrid_lambdas: tuple = (1.0, 0.5, 0.3),
) -> float:
    """Structural score for one candidate given prefix.

    Args:
        A: (N, N) attention matrix, A[j,i] = j's attention to i
        prefix: (t,) already-placed blocks (empty for t=0)
        candidate: scalar block index
        mode: "edge" → A[candidate, prefix[-1]]
              "prefix_mean" → mean(A[candidate, prefix])
              "prefix_max" → max(A[candidate, prefix])
              "hybrid" → λ_last*A[last] + λ_mean*mean + λ_max*max
        hybrid_lambdas: (λ_last, λ_mean, λ_max) for hybrid mode

    Returns:
        scalar structural score
    """
    if len(prefix) == 0:
        # No predecessor: pick by max incoming attention from others
        not_self = np.ones(A.shape[0], dtype=bool)
        not_self[candidate] = False
        return float(A[not_self, candidate].mean())
    if mode == "edge":
        return float(A[candidate, prefix[-1]])
    elif mode == "prefix_mean":
        return float(A[candidate, prefix].mean())
    elif mode == "prefix_max":
        return float(A[candidate, prefix].max())
    elif mode == "hybrid":
        lam_last, lam_mean, lam_max = hybrid_lambdas
        return float(
            lam_last * A[candidate, prefix[-1]]
            + lam_mean * A[candidate, prefix].mean()
            + lam_max * A[candidate, prefix].max()
        )
    raise ValueError(f"Unknown mode: {mode}")


def build_attention_labels(
    A: "np.ndarray",
    prefix: "np.ndarray",
    candidates: "np.ndarray",
    mode: str = "edge",
    tau: float = 0.3,
) -> "np.ndarray":
    """Per-step soft label from attention structure (no NLL involved).

    struct_score_i = compute_prefix_score_np(A, prefix, i, mode)
    q_i = softmax(struct_score_i / tau)

    Args:
        A: (N, N) attention matrix
        prefix: (t,) already-placed blocks
        candidates: (K,) unvisited block indices
        mode: structural score mode
        tau: softmax temperature

    Returns:
        q: (K,) soft label distribution
    """
    scores = np.array([compute_prefix_score_np(A, prefix, int(c), mode) for c in candidates])
    scores = scores / tau
    scores = scores - scores.max()
    q = np.exp(scores)
    q = q / q.sum()
    return q


def compute_path_weight(
    A: "np.ndarray",
    order: "np.ndarray",
    mode: str = "edge",
) -> float:
    """Total structural path score for a full order.

    A[j,i] = j's attention to i.
    "edge" mode: W = Σ_t A[order_{t+1}, order_t]

    Args:
        A: (N, N)
        order: (N,) block order
        mode: "edge" or "prefix_mean"

    Returns:
        scalar path weight
    """
    N = len(order)
    total = 0.0
    for t in range(N - 1):
        if mode == "edge":
            total += A[order[t + 1], order[t]]
        elif mode == "prefix_mean":
            prefix = order[:t + 1]
            nxt = order[t + 1]
            total += float(A[nxt, prefix].mean())
        else:
            raise ValueError(f"Unknown mode: {mode}")
    return total


def generate_on_adapter_labels(
    A_batch: torch.Tensor,
    old_on_prior: "OldONPrior",
    feature_set: str = "v3_qvalue",
    include_edge_greedy_states: bool = True,
    include_random_states: bool = False,
) -> list:
    """Generate training labels for prefix-conditioned ON adapter.

    Teacher = old ON itself (σ_old). At each step t along a path, the MLP
    learns to predict σ_old[t] — "what does old ON want next from this state?"

    Because the MLP sees richer features (raw A-matrix, future graph stats)
    than old ON's internal representation, it may learn to override old ON
    when the A-matrix strongly suggests a better edge.

    Training states come from:
      - Old ON's own path (teacher-forcing on σ_old)
      - Edge-greedy path (off-ON-path states: "if we deviated, what would ON say?")

    No DP needed.  Cheap and self-consistent.

    Args:
        A_batch: (B, N, N) on device
        old_on_prior: OldONPrior instance
        feature_set: feature set for build_reranker_features_v2
        include_edge_greedy_states: also generate labels from edge-greedy path
        include_random_states: also sample random prefixes (optional)

    Returns:
        labels: list of dicts with seq_idx, step, features, struct_targets
    """
    B, N, _ = A_batch.shape
    device = A_batch.device

    all_labels = []

    for s in tqdm(range(B), desc="Generating ON-adapter labels"):
        A_s = A_batch[s:s + 1]  # (1, N, N)
        sigma_old = old_on_prior.get_full_order(A_s)[0]  # (N,) on device
        sigma_old_np = sigma_old.cpu().numpy()

        def process_path(path: np.ndarray, path_name: str):
            """Add teacher-forcing labels for one path. Teacher = σ_old[t]."""
            visited_mask = 0
            for t in range(N - 1):
                candidate_set = [i for i in range(N) if not (visited_mask & (1 << i))]
                if len(candidate_set) == 0:
                    break
                K = len(candidate_set)

                # Teacher: old ON's next block at step t
                teacher_next = int(sigma_old_np[t])

                # Build features (with prefix context from current path)
                last_val = int(path[t - 1]) if t > 0 else 0
                visited_t = torch.tensor([visited_mask], dtype=torch.long, device=device)
                last_t = torch.tensor([last_val], dtype=torch.long, device=device)

                if feature_set == "v1_legacy":
                    features = build_reranker_features(
                        A_s, visited_t, last_t, old_on_prior,
                        sigma_old=sigma_old.unsqueeze(0), step_t=t,
                    )
                else:
                    features = build_reranker_features_v2(
                        A_s, visited_t, last_t, old_on_prior,
                        sigma_old=sigma_old.unsqueeze(0), step_t=t,
                        feature_set=feature_set,
                    )
                candidate_features = features[0, candidate_set]  # (K, F)

                # One-hot target on teacher
                target = torch.zeros(K)
                try:
                    teacher_idx = candidate_set.index(teacher_next)
                    target[teacher_idx] = 1.0
                except ValueError:
                    # Teacher block already visited (shouldn't happen on ON's own path)
                    target[0] = 1.0

                all_labels.append({
                    "seq_idx": s,
                    "step": t,
                    "features": candidate_features.cpu(),
                    "struct_targets": target,
                })

                # Advance mask along the PATH (not the teacher)
                visited_mask |= (1 << int(path[t]))

        # 1. Old ON's own path (teacher-forcing)
        process_path(sigma_old_np, "old_on")

        # 2. Edge-greedy path (off-ON-path states)
        if include_edge_greedy_states:
            sigma_edge = generate_order_A_greedy(A_s, mode="edge")[0].cpu().numpy()
            if not np.array_equal(sigma_edge, sigma_old_np):
                process_path(sigma_edge, "edge")

        # 3. Random prefixes (optional, for broader state coverage)
        if include_random_states:
            for _ in range(3):
                rand_path = np.random.permutation(N)
                if (not np.array_equal(rand_path, sigma_old_np) and
                    not (include_edge_greedy_states and np.array_equal(rand_path, sigma_edge))):
                    process_path(rand_path, "rand")

    n_states = len(all_labels)
    n_seqs = len(set(lab["seq_idx"] for lab in all_labels))
    print(f"ON-adapter labels: {n_states} queries from {n_seqs} sequences")
    return all_labels


# Legacy wrapper — kept for backward compatibility
def generate_q_labels(
    A_batch: torch.Tensor,
    dp_results: list = None,
    old_on_prior: "OldONPrior" = None,
    feature_set: str = "v3_qvalue",
    teacher_mode: str = "on_hard",
    label_tau: float = 0.1,
    include_edge_greedy_states: bool = True,
) -> list:
    """Legacy wrapper. Use generate_on_adapter_labels() for old-ON teacher.

    If dp_results is None, delegates to generate_on_adapter_labels (no DP).
    """
    if dp_results is None:
        return generate_on_adapter_labels(
            A_batch, old_on_prior, feature_set=feature_set,
            include_edge_greedy_states=include_edge_greedy_states,
        )
    # Old DP path (kept for reference, not the default)
    raise NotImplementedError(
        "DP teacher removed. Use generate_on_adapter_labels() instead."
    )


@torch.no_grad()
def generate_order_A_greedy(
    A: torch.Tensor,
    mode: str = "edge",
    hybrid_lambdas: tuple = (1.0, 0.5, 0.3),
) -> torch.Tensor:
    """Greedy order by structural score (no training baseline).

    At each step: pick argmax compute_prefix_score_np(candidate | prefix).

    Args:
        A: (B, N, N)
        mode: "edge", "prefix_mean", "prefix_max", or "hybrid"
        hybrid_lambdas: (λ_last, λ_mean, λ_max) for hybrid mode

    Returns:
        orders: (B, N)
    """
    B, N, _ = A.shape
    device = A.device
    A_np = A.cpu().numpy()

    orders = torch.zeros(B, N, dtype=torch.long, device=device)
    for b in range(B):
        Ab = A_np[b]
        visited = set()
        prefix = np.array([], dtype=np.int64)
        for t in range(N):
            unvisited = np.array([i for i in range(N) if i not in visited], dtype=np.int64)
            scores = np.array([
                compute_prefix_score_np(Ab, prefix, int(i), mode, hybrid_lambdas)
                for i in unvisited
            ])
            best = unvisited[int(np.argmax(scores))]
            orders[b, t] = best
            visited.add(int(best))
            prefix = np.append(prefix, best)
    return orders
