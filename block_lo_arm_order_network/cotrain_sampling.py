"""Lightweight ON sampling helper for co-training.

Variant of train_grpo_on.sample_orders_from_on with K=1, no KL bookkeeping.
Used by train_aogpt_cotrain.py where ON is frozen and we just need orders.
"""
import torch
import torch.nn.functional as F

from order_network import masks_to_revealed_bool


def sample_one_order_per_seq(on_model, A, temperature: float):
    """Sample one full N16 order per sequence via sequential Categorical sampling.

    Args:
        on_model: CrossAttentionOrderNetwork (caller controls grad mode).
        A: (B, 16, 16) attention/score matrix.
        temperature: τ > 0. Lower → closer to argmax.

    Returns:
        orders: (B, 16) LongTensor of physical N16 block indices.
        log_probs: (B,) FloatTensor — sum_t log π(a_t | s_t).
        entropies: (B,) FloatTensor — mean per-step entropy of masked distribution.
    """
    B, N, _ = A.shape
    device = A.device

    visited_mask = torch.zeros(B, dtype=torch.long, device=device)
    last_node = torch.zeros(B, dtype=torch.long, device=device)

    order_list = []
    log_prob_sum = torch.zeros(B, device=device)
    ent_sum = torch.zeros(B, device=device)

    for step in range(N):
        logits = on_model(A, visited_mask, last_node)  # (B, N)
        logits = logits / max(temperature, 1e-6)
        logits = torch.clamp(logits, min=-50.0, max=50.0)

        visited_bool = masks_to_revealed_bool(visited_mask, N)
        logits = logits.masked_fill(visited_bool, float("-inf"))

        dist = torch.distributions.Categorical(logits=logits)
        chosen = dist.sample()
        log_prob_sum = log_prob_sum + dist.log_prob(chosen)
        ent_sum = ent_sum + dist.entropy()

        order_list.append(chosen)
        visited_mask = visited_mask | (1 << chosen)
        last_node = chosen

    orders = torch.stack(order_list, dim=1)
    return orders, log_prob_sum, ent_sum / N
