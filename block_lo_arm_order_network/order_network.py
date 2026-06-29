"""P2: Order Network model + vectorized per-candidate feature extraction.

Architecture: Per-candidate 9-dim features → shared 2-layer MLP → 1 scalar score.
All 16 candidates scored independently, revealed positions masked to -inf.

Key design: extract_candidate_features is fully vectorized — no Python for-loops
over candidates. Called per training step, total calls = epochs × seqs × 15 steps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ── Vectorized feature extraction ───────────────────────────────────────────

def extract_candidate_features(
    A: torch.Tensor,
    revealed_mask: torch.Tensor,
    step_t: int,
    last_revealed: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Extract 9-dim per-candidate features for ALL N blocks in one vectorized pass.

    Args:
        A: (B, N, N) attention matrices, A[q, k] = query q attending to key k.
        revealed_mask: (B, N) bool, True = already revealed.
        step_t: int, current step index (0 to N-2).
        last_revealed: (B,) long tensor, index of most recently revealed block,
                       or None for step_t=0 (empty revealed set).

    Returns:
        features: (B, N, 9) per-candidate features.
                  Features 0-2: to-revealed (candidate→revealed).
                  Features 3-5: from-revealed (revealed→candidate).
                  Features 6-8: positional (pos, step, ratio).
    """
    B, N, _ = A.shape
    device = A.device

    num_revealed = revealed_mask.sum(dim=-1, keepdim=True).float()  # (B, 1)
    no_revealed = (num_revealed.squeeze(-1) == 0)  # (B,) — True if empty prefix
    safe_num = num_revealed.clamp(min=1)  # (B, 1), avoid div-by-zero

    # ── Candidate → Revealed (A[candidate, revealed_key]) ─────────────────
    # revealed_expanded: (B, 1, N) — broadcasts over candidate dim
    revealed_expanded = revealed_mask.unsqueeze(1).float()  # (B, 1, N)

    # Mean over revealed keys
    A_masked_to = A * revealed_expanded  # (B, N, N), unrevealed key cols → 0
    mean_A_to_revealed = A_masked_to.sum(dim=-1) / safe_num  # (B, N)

    # Max over revealed keys (set unrevealed to -inf)
    A_to_for_max = torch.where(
        revealed_expanded.bool(),
        A,
        torch.tensor(float('-inf'), device=device, dtype=A.dtype),
    )
    max_A_to_revealed = A_to_for_max.max(dim=-1).values  # (B, N)

    # A to last revealed block
    if last_revealed is not None:
        idx = last_revealed.view(B, 1, 1).expand(-1, N, 1)  # (B, N, 1)
        A_to_last = A.gather(dim=-1, index=idx).squeeze(-1)  # (B, N)
    else:
        A_to_last = torch.zeros(B, N, device=device, dtype=A.dtype)

    # ── Revealed → Candidate (A[revealed_query, candidate]) ───────────────
    from_revealed_expanded = revealed_mask.unsqueeze(-1).float()  # (B, N, 1)

    # Mean over revealed queries
    A_masked_from = A * from_revealed_expanded  # (B, N, N), unrevealed query rows → 0
    mean_A_from_revealed = A_masked_from.sum(dim=1) / safe_num  # (B, N)

    # Max over revealed queries
    A_from_for_max = torch.where(
        from_revealed_expanded.bool(),
        A,
        torch.tensor(float('-inf'), device=device, dtype=A.dtype),
    )
    max_A_from_revealed = A_from_for_max.max(dim=1).values  # (B, N)

    # A from last revealed block
    if last_revealed is not None:
        idx = last_revealed.view(B, 1, 1).expand(-1, 1, N)  # (B, 1, N)
        A_from_last = A.gather(dim=1, index=idx).squeeze(1)  # (B, N)
    else:
        A_from_last = torch.zeros(B, N, device=device, dtype=A.dtype)

    # ── Positional features ───────────────────────────────────────────────
    positions = torch.arange(N, device=device, dtype=A.dtype).unsqueeze(0).expand(B, -1) / N
    step_feat = torch.full((B, N), step_t / max(N - 1, 1), device=device, dtype=A.dtype)
    num_revealed_feat = num_revealed.expand(-1, N) / N  # (B, N)

    # ── Zero out affinity features when no blocks are revealed ────────────
    if no_revealed.any():
        zero_mask = no_revealed.unsqueeze(-1).expand(-1, N)  # (B, N)
        mean_A_to_revealed[zero_mask] = 0.0
        max_A_to_revealed[zero_mask] = 0.0
        A_to_last[zero_mask] = 0.0
        mean_A_from_revealed[zero_mask] = 0.0
        max_A_from_revealed[zero_mask] = 0.0
        A_from_last[zero_mask] = 0.0

    # ── Stack all 9 features ──────────────────────────────────────────────
    features = torch.stack([
        mean_A_to_revealed,       # 0: mean affinity candidate→revealed
        max_A_to_revealed,        # 1: max affinity candidate→revealed
        A_to_last,                # 2: affinity candidate→last revealed
        mean_A_from_revealed,     # 3: mean affinity revealed→candidate
        max_A_from_revealed,      # 4: max affinity revealed→candidate
        A_from_last,              # 5: affinity last revealed→candidate
        positions,                # 6: normalized physical position
        step_feat,                # 7: normalized step progress
        num_revealed_feat,        # 8: reveal ratio
    ], dim=-1)  # (B, N, 9)

    return features


def masks_to_revealed_bool(visited_masks: torch.Tensor, num_blocks: int) -> torch.Tensor:
    """Convert integer bitmasks to a (B, N) boolean revealed mask."""
    bits = torch.arange(num_blocks, device=visited_masks.device, dtype=torch.long)
    masks = visited_masks.to(torch.long).unsqueeze(-1)
    return ((masks >> bits.unsqueeze(0)) & 1).bool()


def extract_candidate_features_from_masks(
    A: torch.Tensor,
    visited_masks: torch.Tensor,
    last_nodes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract per-candidate features for arbitrary DP states encoded as bitmasks.

    Unlike extract_candidate_features, this supports mixed step indices in the
    same batch by deriving the progress feature from the popcount of each mask.
    """
    B, N, _ = A.shape
    device = A.device
    A = torch.where(
        torch.isfinite(A),
        A,
        torch.zeros((), device=device, dtype=A.dtype),
    )
    revealed_mask = masks_to_revealed_bool(visited_masks.to(device), N)
    num_revealed = revealed_mask.sum(dim=-1, keepdim=True).float()
    safe_num = num_revealed.clamp(min=1)

    revealed_expanded = revealed_mask.unsqueeze(1).float()
    A_masked_to = A * revealed_expanded
    mean_A_to_revealed = A_masked_to.sum(dim=-1) / safe_num
    A_to_for_max = torch.where(
        revealed_expanded.bool(),
        A,
        torch.tensor(float("-inf"), device=device, dtype=A.dtype),
    )
    max_A_to_revealed = A_to_for_max.max(dim=-1).values

    last_nodes = last_nodes.to(device)
    idx_to = last_nodes.view(B, 1, 1).expand(-1, N, 1)
    A_to_last = A.gather(dim=-1, index=idx_to).squeeze(-1)

    from_revealed_expanded = revealed_mask.unsqueeze(-1).float()
    A_masked_from = A * from_revealed_expanded
    mean_A_from_revealed = A_masked_from.sum(dim=1) / safe_num
    A_from_for_max = torch.where(
        from_revealed_expanded.bool(),
        A,
        torch.tensor(float("-inf"), device=device, dtype=A.dtype),
    )
    max_A_from_revealed = A_from_for_max.max(dim=1).values

    idx_from = last_nodes.view(B, 1, 1).expand(-1, 1, N)
    A_from_last = A.gather(dim=1, index=idx_from).squeeze(1)

    positions = torch.arange(N, device=device, dtype=A.dtype).unsqueeze(0).expand(B, -1) / N
    step_feat = (num_revealed / max(N - 1, 1)).expand(-1, N)
    num_revealed_feat = (num_revealed / N).expand(-1, N)

    features = torch.stack([
        mean_A_to_revealed,
        max_A_to_revealed,
        A_to_last,
        mean_A_from_revealed,
        max_A_from_revealed,
        A_from_last,
        positions,
        step_feat,
        num_revealed_feat,
    ], dim=-1)
    return features, revealed_mask


def extract_rich_candidate_features_from_masks(
    A: torch.Tensor,
    visited_masks: torch.Tensor,
    last_nodes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rich per-candidate features for learning DP labels from the full A graph.

    For each candidate block, features concatenate:
    A[candidate, :], A[:, candidate], revealed mask, last-node onehot,
    candidate onehot, normalized position, and reveal ratio.
    """
    B, N, _ = A.shape
    device = A.device
    A = torch.where(
        torch.isfinite(A),
        A,
        torch.zeros((), device=device, dtype=A.dtype),
    )
    revealed_mask = masks_to_revealed_bool(visited_masks.to(device), N)
    block_ids = torch.arange(N, device=device)

    rows = A
    cols = A.transpose(1, 2)
    revealed = revealed_mask.float().unsqueeze(1).expand(-1, N, -1)
    last_onehot = F.one_hot(last_nodes.to(device), num_classes=N).float()
    last_onehot = last_onehot.unsqueeze(1).expand(-1, N, -1)
    candidate_onehot = F.one_hot(block_ids, num_classes=N).float()
    candidate_onehot = candidate_onehot.unsqueeze(0).expand(B, -1, -1)

    positions = (block_ids.float() / N).view(1, N, 1).expand(B, -1, -1)
    reveal_ratio = (
        revealed_mask.sum(dim=-1, keepdim=True).float() / N
    ).view(B, 1, 1).expand(-1, N, -1)

    features = torch.cat([
        rows,
        cols,
        revealed,
        last_onehot,
        candidate_onehot,
        positions,
        reveal_ratio,
    ], dim=-1)
    return features, revealed_mask


# ── Order Network ──────────────────────────────────────────────────────────

class OrderNetwork(nn.Module):
    """
    Per-candidate shared MLP scorer.

    Each candidate's 9-dim features → shared MLP → 1 scalar score.
    All N candidate scores → logits, revealed positions masked to -inf.
    """

    def __init__(self, feature_dim: int = 9, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.1):
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

    def forward(self, candidate_features: torch.Tensor,
                revealed_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            candidate_features: (B, N, F) per-candidate features.
            revealed_mask: (B, N) bool, True = already revealed (excluded).

        Returns:
            logits: (B, N), revealed positions set to -inf.
        """
        B, N, F = candidate_features.shape
        flat = candidate_features.reshape(B * N, F)       # (B*N, F)
        scores = self.mlp(flat).squeeze(-1)               # (B*N,)
        scores = scores.reshape(B, N)                     # (B, N)
        scores = scores.masked_fill(revealed_mask, float('-inf'))
        return scores


class RouteATransformerOrderNetwork(nn.Module):
    """
    Route A state-conditioned Transformer policy.

    Inputs are compact DP/DAgger states:
    - visited_masks: uint/int bitmask, one bit per block.
    - last_nodes: index of the last revealed block.

    The network builds one token per block plus a learned policy token, then
    predicts the next block. Visited blocks are masked to -inf in the logits.
    """

    def __init__(
        self,
        num_blocks: int = 16,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 4096,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.d_model = d_model

        self.block_embedding = nn.Embedding(num_blocks, d_model)
        self.visited_embedding = nn.Embedding(2, d_model)
        self.last_embedding = nn.Embedding(2, d_model)
        self.policy_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.input_norm = nn.LayerNorm(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_blocks),
        )

        self.register_buffer(
            "block_ids",
            torch.arange(num_blocks, dtype=torch.long),
            persistent=False,
        )
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.policy_token, mean=0.0, std=0.02)

    def _visited_bool(self, visited_masks: torch.Tensor) -> torch.Tensor:
        bits = self.block_ids.to(visited_masks.device).unsqueeze(0)
        masks = visited_masks.to(torch.long).unsqueeze(-1)
        return ((masks >> bits) & 1).bool()

    def forward(
        self,
        visited_masks: torch.Tensor,
        last_nodes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            visited_masks: (B,) integer bitmask of revealed blocks.
            last_nodes: (B,) long tensor with last revealed block index.

        Returns:
            logits: (B, num_blocks), visited positions set to -inf.
        """
        if visited_masks.ndim != 1 or last_nodes.ndim != 1:
            raise ValueError("visited_masks and last_nodes must both be rank-1")
        if visited_masks.shape[0] != last_nodes.shape[0]:
            raise ValueError("visited_masks and last_nodes must have same batch size")

        B = visited_masks.shape[0]
        device = visited_masks.device
        block_ids = self.block_ids.to(device).unsqueeze(0).expand(B, -1)
        visited = self._visited_bool(visited_masks)
        is_last = block_ids.eq(last_nodes.to(device).view(B, 1))

        block_tokens = (
            self.block_embedding(block_ids)
            + self.visited_embedding(visited.long())
            + self.last_embedding(is_last.long())
        )
        policy_token = self.policy_token.expand(B, -1, -1)
        tokens = torch.cat([policy_token, block_tokens], dim=1)
        tokens = self.input_norm(tokens)

        hidden = self.transformer(tokens)
        policy_hidden = self.output_norm(hidden[:, 0])
        logits = self.output_head(policy_hidden)
        return logits.masked_fill(visited, float("-inf"))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── DAgger epsilon schedule ────────────────────────────────────────────────

def get_dagger_epsilon(epoch: int, config) -> float:
    """
    DAgger noise schedule: pure teacher forcing early, then linear ramp.

    Epoch < dagger_start_epoch:            ε = 0.0
    dagger_start_epoch ≤ epoch < ramp_end:  ε = linear 0 → max_epsilon
    epoch ≥ ramp_end:                       ε = max_epsilon
    """
    if epoch < config.dagger_start_epoch:
        return 0.0
    ramp_length = 20  # epochs 10→30 in default config
    ramp_end = config.dagger_start_epoch + ramp_length
    if epoch < ramp_end:
        progress = (epoch - config.dagger_start_epoch) / ramp_length
        return config.dagger_max_epsilon * progress
    return config.dagger_max_epsilon


# ── Topological Node Features (no position encoding) ────────────────────────

def extract_topological_node_features(
    A: torch.Tensor,
    visited_masks: torch.Tensor,
    last_nodes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pure topological/state features per node. No absolute position.

    Returns:
        features: (B, N, 7) — [is_visited, is_last, in_deg_w, out_deg_w,
                               in_deg_uw, out_deg_uw, reveal_ratio]
        revealed_mask: (B, N) bool
    """
    B, N, _ = A.shape
    device = A.device

    A_safe = torch.where(torch.isfinite(A), A, torch.zeros((), device=device, dtype=A.dtype))
    revealed_mask = masks_to_revealed_bool(visited_masks.to(device), N)

    is_visited = revealed_mask.float().unsqueeze(-1)  # (B, N, 1)
    is_last = torch.zeros(B, N, 1, device=device)
    last_idx = last_nodes.to(device).view(B, 1)
    is_last.scatter_(1, last_idx.unsqueeze(-1), 1.0)

    A_sp = F.softplus(A_safe)  # smooth positive edge weights
    in_deg_w = A_sp.sum(dim=1).unsqueeze(-1)   # (B, N, 1) — sum over queries (incoming to key)
    out_deg_w = A_sp.sum(dim=2).unsqueeze(-1)  # (B, N, 1) — sum over keys (outgoing from query)

    A_mean = A_safe.mean(dim=(1, 2), keepdim=True)
    in_deg_uw = (A_safe > A_mean).float().sum(dim=1).unsqueeze(-1)
    out_deg_uw = (A_safe > A_mean).float().sum(dim=2).unsqueeze(-1)

    reveal_ratio = revealed_mask.float().sum(dim=-1, keepdim=True).unsqueeze(-1) / N

    features = torch.cat([
        is_visited,
        is_last,
        in_deg_w,
        out_deg_w,
        in_deg_uw,
        out_deg_uw,
        reveal_ratio.expand(-1, N, -1),
    ], dim=-1)  # (B, N, 7)

    return features, revealed_mask


# ── Edge-Conditioned GNN Layer ──────────────────────────────────────────────

class EdgeGNNLayer(nn.Module):
    """One message-passing layer with learnable edge temperature."""

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_e = nn.Linear(1, d_model, bias=False)
        self.W_update = nn.Linear(2 * d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.log_tau = nn.Parameter(torch.zeros(1))  # temperature for tanh

    def forward(self, h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        B, N, _ = A.shape
        A_safe = torch.where(torch.isfinite(A), A, torch.zeros((), device=A.device, dtype=A.dtype))
        tau = F.softplus(self.log_tau) + 1e-3
        w = torch.tanh(A_safe / tau)  # (B, N, N), in [-1, 1]

        h_proj = self.W_v(h)  # (B, N, d)
        messages = torch.einsum('bij,bjd->bid', w, h_proj)  # (B, N, d)

        w_scalar = w.unsqueeze(-1)  # (B, N, N, 1)
        edge_feat = self.W_e(w_scalar).sum(dim=2)  # (B, N, d)

        agg = messages + edge_feat
        combined = torch.cat([h, agg], dim=-1)  # (B, N, 2d)
        h_new = self.norm(h + F.gelu(self.W_update(combined)))
        return h_new


# ── Graph Order Network ─────────────────────────────────────────────────────

class GraphOrderNetwork(nn.Module):
    """
    Permutation-equivariant GNN for block ordering.

    Pure topological node features → 3-layer edge-conditioned message-passing
    → readout MLP → per-node logits.
    """

    def __init__(self, num_blocks: int = 16, d_model: int = 64, num_layers: int = 3):
        super().__init__()
        self.num_blocks = num_blocks
        self.d_model = d_model

        self.feature_proj = nn.Linear(7, d_model, bias=False)
        self.gnn_layers = nn.ModuleList([EdgeGNNLayer(d_model) for _ in range(num_layers)])
        self.readout = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self, A: torch.Tensor, visited_masks: torch.Tensor, last_nodes: torch.Tensor
    ) -> torch.Tensor:
        features, revealed_mask = extract_topological_node_features(
            A, visited_masks, last_nodes
        )  # (B, N, 7)

        h = self.feature_proj(features)  # (B, N, d)
        for layer in self.gnn_layers:
            h = layer(h, A)

        scores = self.readout(h).squeeze(-1)  # (B, N)
        return scores.masked_fill(revealed_mask, float('-inf'))


# ── DeepSet Order Network (message-passing ablation) ─────────────────────────

class DeepSetOrderNetwork(nn.Module):
    """
    No message-passing baseline. Same features, same readout MLP.
    Any accuracy delta vs GraphOrderNetwork is purely from graph convolution.
    """

    def __init__(self, num_blocks: int = 16, d_model: int = 64):
        super().__init__()
        self.readout = nn.Sequential(
            nn.Linear(7, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self, A: torch.Tensor, visited_masks: torch.Tensor, last_nodes: torch.Tensor
    ) -> torch.Tensor:
        features, revealed_mask = extract_topological_node_features(
            A, visited_masks, last_nodes
        )  # (B, N, 7)
        scores = self.readout(features).squeeze(-1)  # (B, N)
        return scores.masked_fill(revealed_mask, float('-inf'))


# ── Cross-Attention Order Network ─────────────────────────────────────────────

class CrossAttentionOrderNetwork(nn.Module):
    """
    Cross-Attention A-Feature Network for block ordering.

    Q = candidate block c (unrevealed).
    K/V = revealed blocks r in S_t.

    For each (c, r) pair, edge features [A[c,r], A[r,c], is_last_r] are
    embedded via a shared edge MLP, then an attention MLP learns to weight
    revealed blocks. The weighted sum H_c captures the candidate's relationship
    to the entire revealed context. A final score MLP maps H_c + step progress
    to a scalar logit.

    Permutation-equivariant by construction: only set operations over S_t,
    no fixed-dimension position encoding. ~13K parameters at d_edge=64.
    """

    def __init__(self, num_blocks: int = 16, d_edge: int = 64, d_model: int = 64):
        super().__init__()
        self.num_blocks = num_blocks
        self.d_edge = d_edge

        self.edge_mlp = nn.Sequential(
            nn.Linear(3, d_edge),
            nn.ReLU(),
            nn.Linear(d_edge, d_edge),
            nn.ReLU(),
        )
        self.attn_mlp = nn.Linear(d_edge, 1)
        # d_edge (H_c) + 1 (step_ratio) + 2 (row_mean, col_mean global profile)
        self.score_mlp = nn.Sequential(
            nn.Linear(d_edge + 3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )
        self.start_embedding = nn.Parameter(torch.zeros(1, 1, d_edge))
        nn.init.normal_(self.start_embedding, mean=0.0, std=0.02)

    def forward(
        self, A: torch.Tensor, visited_masks: torch.Tensor, last_nodes: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            A: (B, N, N) attention/NLL matrix, A[q,k] = query q → key k.
            visited_masks: (B,) integer bitmask of revealed blocks.
            last_nodes: (B,) long tensor, index of most recently revealed block.

        Returns:
            logits: (B, N), revealed positions set to -inf.
        """
        B, N, _ = A.shape
        device = A.device

        A_safe = torch.where(
            torch.isfinite(A), A, torch.zeros((), device=device, dtype=A.dtype)
        )
        revealed_mask = masks_to_revealed_bool(visited_masks.to(device), N)  # (B, N)

        # Edge features: [A[c,r], A[r,c], is_last_r]  ── (B, N, N, 3)
        A_cr = A_safe.unsqueeze(-1)                    # (B, N, N, 1)
        A_rc = A_safe.transpose(1, 2).unsqueeze(-1)    # (B, N, N, 1)
        is_last_r = (
            F.one_hot(last_nodes.to(device), num_classes=N)
            .float()
            .unsqueeze(1)                              # (B, 1, N)
            .unsqueeze(-1)                             # (B, 1, N, 1)
            .expand(-1, N, -1, -1)                     # (B, N, N, 1)
        )
        edge_raw = torch.cat([A_cr, A_rc, is_last_r], dim=-1)  # (B, N, N, 3)

        # Per-edge embedding
        h_edge = self.edge_mlp(
            edge_raw.reshape(B * N * N, 3)
        ).reshape(B, N, N, self.d_edge)                # (B, N, N, d_edge)

        # Attention logits over revealed blocks
        attn_logits = self.attn_mlp(
            h_edge.reshape(B * N * N, self.d_edge)
        ).reshape(B, N, N, 1)                          # (B, N, N, 1)

        # Valid attention keys: revealed r, r != c
        valid = revealed_mask.float().unsqueeze(1).unsqueeze(-1)      # (B, 1, N, 1)
        not_self = (1 - torch.eye(N, device=device)).unsqueeze(0).unsqueeze(-1)  # (1, N, N, 1)
        attn_mask = valid * not_self                                   # (B, N, N, 1)
        has_keys = attn_mask.squeeze(-1).any(dim=2)                   # (B, N)

        # Init H_c with learned start embedding (used when no valid keys)
        H_c = self.start_embedding.to(device).expand(B, N, self.d_edge).clone()

        # Numerically stable masked softmax — handles all -inf rows safely
        logits_masked = attn_logits.masked_fill(attn_mask == 0, float('-inf'))
        logits_max = logits_masked.max(dim=2, keepdim=True).values   # (B, N, 1, 1)
        logits_max = torch.where(
            torch.isfinite(logits_max),
            logits_max,
            torch.zeros_like(logits_max),               # all -inf → use 0
        )
        logits_exp = torch.exp(logits_masked - logits_max)  # exp(-inf) = 0
        logits_sum = logits_exp.sum(dim=2, keepdim=True).clamp(min=1e-9)
        attn_w = logits_exp / logits_sum                # (B, N, N, 1) — 0 where no valid keys

        H_attn = (attn_w * h_edge).sum(dim=2)           # (B, N, d_edge)
        H_c = torch.where(
            has_keys.unsqueeze(-1), H_attn, H_c
        )

        # State feature: step progress
        num_revealed = revealed_mask.float().sum(dim=-1, keepdim=True)  # (B, 1)
        step_ratio = (num_revealed / max(N - 1, 1)).unsqueeze(-1)       # (B, 1, 1)
        step_ratio = step_ratio.expand(-1, N, -1)                        # (B, N, 1)

        # Global per-candidate profile (permutation-equivariant)
        not_self_mask = (1 - torch.eye(N, device=device)).unsqueeze(0)   # (1, N, N)
        row_sum = (A_safe * not_self_mask).sum(dim=-1)                   # (B, N)
        col_sum = (A_safe * not_self_mask).sum(dim=1)                    # (B, N)
        row_mean = row_sum / (N - 1)                                     # (B, N)
        col_mean = col_sum / (N - 1)                                     # (B, N)

        combined = torch.cat([
            H_c,
            step_ratio,
            row_mean.unsqueeze(-1),
            col_mean.unsqueeze(-1),
        ], dim=-1)  # (B, N, d_edge + 3)

        scores = self.score_mlp(
            combined.reshape(B * N, self.d_edge + 3)
        ).reshape(B, N)

        return scores.masked_fill(revealed_mask, float('-inf'))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
