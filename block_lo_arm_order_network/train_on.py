"""P2: Order Network training with step-by-step teacher forcing + DAgger.

Key features:
- Per-step vectorized feature extraction (no Python loops over candidates)
- DAgger with per-sequence epsilon-probability, routing table O(1) lookup
- Per-epoch logging: loss, top-1 accuracy, top-3 accuracy
- Pure teacher forcing for first dagger_start_epoch epochs
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
from collections import defaultdict

from config import Config
from order_network import OrderNetwork, extract_candidate_features, get_dagger_epsilon
from dp_solver import lookup_next_step


def _mask_to_int(mask: torch.Tensor, N: int = 16) -> int:
    """Convert boolean mask (N,) to integer bitmask."""
    val = 0
    for i in range(N):
        if mask[i]:
            val |= (1 << i)
    return val


def train_order_network(
    order_network: OrderNetwork,
    A_matrices: np.ndarray,             # (num_sequences, N, N) float32
    optimal_orderings: List[List[int]],  # length num_sequences, each [16] ints
    routing_tables: List[np.ndarray],    # each (2^N, N) int8
    config: Optional[Config] = None,
) -> dict:
    """
    Train Order Network with teacher forcing + DAgger.

    Returns:
        metrics: dict with per-epoch 'loss', 'acc_top1', 'acc_top3' lists.
    """
    if config is None:
        config = Config()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N = config.num_blocks
    num_sequences = len(A_matrices)

    # Convert data to tensors
    A_tensor = torch.from_numpy(A_matrices).to(device)  # (S, N, N)
    opt_order = torch.tensor(optimal_orderings, dtype=torch.long, device=device)  # (S, N)

    order_network = order_network.to(device)
    optimizer = torch.optim.Adam(order_network.parameters(), lr=config.on_lr)

    metrics = defaultdict(list)

    for epoch in range(config.on_num_epochs):
        epsilon = get_dagger_epsilon(epoch, config)
        order_network.train()

        total_loss = 0.0
        total_correct_top1 = 0
        total_correct_top3 = 0
        total_steps = 0
        dagger_steps = 0

        # Shuffle sequences
        perm = torch.randperm(num_sequences, device=device)
        A_shuffled = A_tensor[perm]
        order_shuffled = opt_order[perm]

        for batch_start in range(0, num_sequences, config.on_batch_size):
            batch_end = min(batch_start + config.on_batch_size, num_sequences)
            B = batch_end - batch_start

            batch_A = A_shuffled[batch_start:batch_end]       # (B, N, N)
            batch_order = order_shuffled[batch_start:batch_end]  # (B, N)

            # Initialize per-sequence state
            revealed_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
            last_revealed = None  # (B,) long or None

            for step in range(N - 1):  # 0 .. 14, predict σ*[step]
                # Extract features for current state (vectorized over batch)
                features = extract_candidate_features(
                    batch_A, revealed_mask, step, last_revealed,
                )  # (B, N, 9)

                # Forward pass
                scores = order_network(features, revealed_mask)  # (B, N)

                # ── Determine label and actual_next ─────────────────────
                # actual_next: the block that gets revealed (follows the trajectory)
                # label:       the target for cross-entropy loss

                if step == 0 or epsilon == 0.0:
                    # Pure teacher forcing: follow σ*
                    label = batch_order[:, step]           # (B,)
                    actual_next = label

                else:
                    # Per-sequence DAgger decision
                    dagger_flags = torch.rand(B, device=device) < epsilon  # (B,)

                    if dagger_flags.any():
                        # Sample from ON for DAgger sequences
                        with torch.no_grad():
                            tau = config.dagger_temperature
                            dagger_scores = scores[dagger_flags] / tau
                            dagger_probs = F.softmax(dagger_scores, dim=-1)
                            sampled = torch.multinomial(dagger_probs, 1).squeeze(-1)  # (D,)

                        # Look up optimal next from routing table for deviated state
                        d_indices = torch.where(dagger_flags)[0]
                        batch_indices = [perm[batch_start + b].item()
                                         for b in range(B)]

                        # Build labels: teacher-forcing for non-dagger, routing for dagger
                        label = batch_order[:, step].clone()
                        actual_next = batch_order[:, step].clone()

                        for di, bi in enumerate(d_indices.tolist()):
                            # Current visited mask (before revealing sampled)
                            curr_mask_int = _mask_to_int(revealed_mask[bi], N)
                            new_mask_int = curr_mask_int | (1 << int(sampled[di]))
                            rt = routing_tables[batch_indices[bi]]
                            opt_next = lookup_next_step(rt, new_mask_int, int(sampled[di]))
                            if opt_next < 0:
                                # Routing table miss: fall back to teacher forcing
                                opt_next = int(batch_order[bi, step].item())
                            label[bi] = opt_next
                            actual_next[bi] = sampled[di]
                            dagger_steps += 1
                    else:
                        label = batch_order[:, step]
                        actual_next = label

                # ── Compute loss ───────────────────────────────────────
                loss_per = F.cross_entropy(scores, label, reduction='none',
                                           ignore_index=-1)
                # Skip inf/nan elements from edge cases (rare DAgger transition
                # where ON samples a block whose routing-table state is invalid)
                valid = loss_per.isfinite()
                if valid.any():
                    loss = loss_per[valid].mean()
                else:
                    loss = torch.tensor(0.0, device=device, requires_grad=True)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * B
                total_steps += B

                # Top-1 accuracy
                pred_top1 = scores.argmax(dim=-1)
                correct_top1 = (pred_top1 == label).float()
                total_correct_top1 += correct_top1.sum().item()

                # Top-3 accuracy
                _, pred_top3_indices = scores.topk(min(3, N), dim=-1)
                correct_top3 = (pred_top3_indices == label.unsqueeze(-1)).any(dim=-1).float()
                total_correct_top3 += correct_top3.sum().item()

                # ── Update state: reveal actual_next ────────────────────
                revealed_mask[torch.arange(B, device=device), actual_next] = True
                last_revealed = actual_next

        # ── Epoch-level metrics ───────────────────────────────────────────
        avg_loss = total_loss / max(total_steps, 1)
        acc1 = total_correct_top1 / max(total_steps, 1)
        acc3 = total_correct_top3 / max(total_steps, 1)
        random_baseline = 1.0 / N  # ~6.25%

        metrics['loss'].append(avg_loss)
        metrics['acc_top1'].append(acc1)
        metrics['acc_top3'].append(acc3)
        metrics['dagger_epsilon'].append(epsilon)
        metrics['dagger_steps'].append(dagger_steps)

        print(f"Epoch {epoch+1:3d}/{config.on_num_epochs} | "
              f"loss={avg_loss:.4f} | "
              f"top1={acc1:.3f} (rand={random_baseline:.3f}) | "
              f"top3={acc3:.3f} | "
              f"ε={epsilon:.2f} | dagger_steps={dagger_steps}")

    return dict(metrics)
