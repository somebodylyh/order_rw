"""
Compare two Order Network checkpoints (e.g., ON_round2 vs ON_round3).

Reports:
  1. Global greedy order on mean A (both before and after)
  2. Per-step entropy profiles
  3. Sampled order comparison: Hamming distance, cross-log-prob

Usage:
    python -u compare_on_versions.py \
      --on2 probe_results/grpo_on_round2.pt \
      --on3 probe_results/round0/grpo_on_round3.pt \
      --device cuda:0
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "AO-GPT-MDM"))

import numpy as np
import torch
import torch.nn.functional as F

from order_network import CrossAttentionOrderNetwork, masks_to_revealed_bool

N16 = 16


def load_on(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]
    d_edge = sd["edge_mlp.0.weight"].shape[0]
    d_model = sd["score_mlp.0.weight"].shape[0]
    model = CrossAttentionOrderNetwork(num_blocks=N16, d_edge=d_edge, d_model=d_model)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def greedy_order(on_model, A):
    B, N, _ = A.shape
    device = A.device
    visited_mask = torch.zeros(B, dtype=torch.long, device=device)
    last_node = torch.zeros(B, dtype=torch.long, device=device)
    order_list = []
    for _ in range(N):
        logits = on_model(A, visited_mask, last_node)
        visited_bool = masks_to_revealed_bool(visited_mask, N)
        logits = logits.masked_fill(visited_bool, float("-inf"))
        chosen = logits.argmax(dim=-1)
        order_list.append(chosen)
        visited_mask = visited_mask | (1 << chosen)
        last_node = chosen
    return torch.stack(order_list, dim=1)


@torch.no_grad()
def per_step_entropy(on_model, A):
    B, N, _ = A.shape
    device = A.device
    visited_mask = torch.zeros(B, dtype=torch.long, device=device)
    last_node = torch.zeros(B, dtype=torch.long, device=device)
    entropies = []
    for _ in range(N):
        logits = on_model(A, visited_mask, last_node)
        visited_bool = masks_to_revealed_bool(visited_mask, N)
        logits = logits.masked_fill(visited_bool, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        ent = -(probs * torch.log(probs + 1e-12)).sum(-1).mean().item()
        chosen = logits.argmax(dim=-1)
        visited_mask = visited_mask | (1 << chosen)
        last_node = chosen
    return entropies


@torch.no_grad()
def sample_and_log_prob(on_model, A, temperature, num_samples):
    B, N, _ = A.shape
    device = A.device
    SB = num_samples * B
    A_exp = A.unsqueeze(0).expand(num_samples, -1, -1, -1).reshape(SB, N, N)

    visited_mask = torch.zeros(SB, dtype=torch.long, device=device)
    last_node = torch.zeros(SB, dtype=torch.long, device=device)
    order_list = []
    log_prob_sum = torch.zeros(SB, device=device)

    for _ in range(N):
        logits = on_model(A_exp, visited_mask, last_node)
        logits = logits / max(temperature, 1e-6)
        logits = torch.clamp(logits, min=-50.0, max=50.0)
        visited_bool = masks_to_revealed_bool(visited_mask, N)
        logits = logits.masked_fill(visited_bool, float("-inf"))
        dist = torch.distributions.Categorical(logits=logits)
        chosen = dist.sample()
        log_prob_sum = log_prob_sum + dist.log_prob(chosen)
        order_list.append(chosen)
        visited_mask = visited_mask | (1 << chosen)
        last_node = chosen

    orders = torch.stack(order_list, dim=1).reshape(num_samples, B, N)
    log_probs = log_prob_sum.reshape(num_samples, B)
    return orders, log_probs


@torch.no_grad()
def log_prob_of_orders(on_model, A, orders):
    num_samples, B, N = orders.shape
    device = A.device
    SB = num_samples * B
    A_exp = A.unsqueeze(0).expand(num_samples, -1, -1, -1).reshape(SB, N, N)
    orders_flat = orders.reshape(SB, N)

    visited_mask = torch.zeros(SB, dtype=torch.long, device=device)
    last_node = torch.zeros(SB, dtype=torch.long, device=device)
    log_prob_sum = torch.zeros(SB, device=device)

    for step in range(N):
        logits = on_model(A_exp, visited_mask, last_node)
        visited_bool = masks_to_revealed_bool(visited_mask, N)
        logits = logits.masked_fill(visited_bool, float("-inf"))
        dist = torch.distributions.Categorical(logits=logits)
        chosen = orders_flat[:, step]
        log_prob_sum = log_prob_sum + dist.log_prob(chosen)
        visited_mask = visited_mask | (1 << chosen)
        last_node = chosen

    return log_prob_sum.reshape(num_samples, B)


def hamming_distance(o1, o2):
    return (o1 != o2).sum(dim=-1).float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--on2", default="probe_results/grpo_on_round2.pt")
    p.add_argument("--on3", default="probe_results/round0/grpo_on_round3.pt")
    p.add_argument("--a-before", default="probe_results/round0/A_before_2k.npy")
    p.add_argument("--a-after", default="probe_results/round0/A_after_2k.npy")
    p.add_argument("--n-compare", type=int, default=32)
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}", flush=True)

    # Load ONs
    print("Loading ON_round2...", flush=True)
    on2 = load_on(args.on2, device)
    ckpt2 = torch.load(args.on2, map_location="cpu", weights_only=False)
    print(f"  ON_round2: best_val_reward={ckpt2.get('best_val_reward','?'):.4f}, "
          f"best_epoch={ckpt2.get('best_epoch','?')}", flush=True)

    print("Loading ON_round3...", flush=True)
    on3 = load_on(args.on3, device)
    ckpt3 = torch.load(args.on3, map_location="cpu", weights_only=False)
    print(f"  ON_round3: best_val_reward={ckpt3.get('best_val_reward','?'):.4f}, "
          f"best_epoch={ckpt3.get('best_epoch','?')}", flush=True)

    # Load A matrices
    A_before = torch.as_tensor(np.load(args.a_before), dtype=torch.float32, device=device)
    A_after = torch.as_tensor(np.load(args.a_after), dtype=torch.float32, device=device)
    print(f"A_before: {A_before.shape}, A_after: {A_after.shape}", flush=True)

    # Subset for sampling comparisons
    n_cmp = min(args.n_compare, A_before.shape[0])
    rng = np.random.RandomState(42)
    cmp_idx = rng.choice(A_before.shape[0], n_cmp, replace=False)
    A_before_cmp = A_before[cmp_idx]
    A_after_cmp = A_after[cmp_idx]

    # ═══════════════════════════════════════════════════════════════════
    # 1. Global greedy orders on mean A
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("1. Global greedy orders on mean A")
    print("=" * 60)

    for label, A_mean in [("A_before", A_before.mean(0, keepdim=True)),
                            ("A_after", A_after.mean(0, keepdim=True))]:
        o2 = greedy_order(on2, A_mean)[0].tolist()
        o3 = greedy_order(on3, A_mean)[0].tolist()
        ham = sum(1 for a, b in zip(o2, o3) if a != b)
        print(f"  {label}:")
        print(f"    ON_round2: {o2}")
        print(f"    ON_round3: {o3}")
        print(f"    Hamming:   {ham}/16")

    # ═══════════════════════════════════════════════════════════════════
    # 2. Per-step entropy
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("2. Per-step entropy profiles (greedy path, on A_before[32])")
    print("=" * 60)

    ent2 = per_step_entropy(on2, A_before_cmp)
    ent3 = per_step_entropy(on3, A_before_cmp)

    print(f"  {'Step':>5s}  {'ON_r2':>8s}  {'ON_r3':>8s}  {'Δ':>8s}")
    for s in range(16):
        print(f"  {s:5d}  {ent2[s]:8.4f}  {ent3[s]:8.4f}  {ent3[s]-ent2[s]:+8.4f}")
    print(f"  {'Mean':>5s}  {np.mean(ent2):8.4f}  {np.mean(ent3):8.4f}  "
          f"{np.mean(ent3)-np.mean(ent2):+8.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # 3. Sampled order comparison
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"3. Sampled order comparison (tau={args.tau}, "
          f"n_samples={args.n_samples}, n_seqs={n_cmp})")
    print("=" * 60)

    orders2, lp2 = sample_and_log_prob(on2, A_before_cmp, args.tau, args.n_samples)
    orders3, lp3 = sample_and_log_prob(on3, A_before_cmp, args.tau, args.n_samples)

    lp2_given_3 = log_prob_of_orders(on3, A_before_cmp, orders2)
    lp3_given_2 = log_prob_of_orders(on2, A_before_cmp, orders3)

    print(f"\n  Self log-prob:")
    print(f"    ON_round2: {lp2.mean():.2f} ± {lp2.std():.2f}")
    print(f"    ON_round3: {lp3.mean():.2f} ± {lp3.std():.2f}")
    print(f"\n  Cross log-prob:")
    print(f"    ON_round3 scores ON_round2 orders: {lp2_given_3.mean():.2f} ± {lp2_given_3.std():.2f}")
    print(f"    ON_round2 scores ON_round3 orders: {lp3_given_2.mean():.2f} ± {lp3_given_2.std():.2f}")
    print(f"\n  Gap (cross - self):")
    gap_23 = lp2_given_3.mean() - lp2.mean()
    gap_32 = lp3_given_2.mean() - lp3.mean()
    print(f"    ON_round2: {gap_23:+.2f}  (how well ON3 explains ON2 orders vs ON2 itself)")
    print(f"    ON_round3: {gap_32:+.2f}  (how well ON2 explains ON3 orders vs ON3 itself)")

    # Hamming distances
    ham = hamming_distance(
        orders2.reshape(-1, 16), orders3.reshape(-1, 16)
    ).reshape(args.n_samples, n_cmp)

    print(f"\n  Pairwise Hamming distance (sampled orders, same A):")
    print(f"    Mean:   {ham.mean():.2f}/16")
    print(f"    Median: {ham.median():.2f}/16")
    print(f"    Min:    {ham.min():.0f}/16")
    print(f"    Max:    {ham.max():.0f}/16")

    # ── Decision ──
    print("\n" + "=" * 60)
    print("DECISION SUMMARY")
    print("=" * 60)

    o2_before = greedy_order(on2, A_before.mean(0, keepdim=True))[0]
    o3_before = greedy_order(on3, A_before.mean(0, keepdim=True))[0]
    o2_after = greedy_order(on2, A_after.mean(0, keepdim=True))[0]
    o3_after = greedy_order(on3, A_after.mean(0, keepdim=True))[0]

    ham_before = sum(1 for a, b in zip(o2_before.tolist(), o3_before.tolist()) if a != b)
    ham_after = sum(1 for a, b in zip(o2_after.tolist(), o3_after.tolist()) if a != b)

    print(f"  Greedy order diff (A_before): {ham_before}/16  "
          f"({'≈' if ham_before < 2 else '≠'} same)")
    print(f"  Greedy order diff (A_after):  {ham_after}/16  "
          f"({'≈' if ham_after < 2 else '≠'} same)")
    print(f"  Mean Hamming sampled pairs:    {ham.mean():.2f}/16")
    print(f"  Cross-logprob gap (R2→R3):     {gap_23:+.2f}")
    print(f"  Cross-logprob gap (R3→R2):     {gap_32:+.2f}")

    is_different = (
        ham_before >= 2 or ham_after >= 2
        or ham.mean() >= 2.0
        or gap_23 < -1.0 or gap_32 < -1.0
    )

    if is_different:
        print("\n  => ON_round3 MATERIALLY DIFFERS from ON_round2")
        print("  => A change carries useful signal => proceed to Round 1 (co-train)")
    else:
        print("\n  => ON_round3 == ON_round2 (functionally identical)")
        print("  => 10% Frobenius change does NOT carry learnable signal for ON")
        print("  => Go Plan C: restart Round 0 with tau_init=0.5, more iters")


if __name__ == "__main__":
    main()
