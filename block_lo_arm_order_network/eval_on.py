"""P3: Order Network evaluation with three metrics + plots.

Metric 1 — Per-step Top-1 Accuracy (teacher-forcing, per-step prediction)
Metric 2 — Path Coherence Q(sigma) (autoregressive sampling at multiple temps)
Metric 3 — Kendall tau (ON vs DP optimal, ON vs L2R)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional
from collections import defaultdict

from config import Config
from attention_extractor import generate_mock_dataset
from dp_solver import solve_dp_batch
from order_network import OrderNetwork, extract_candidate_features
from train_on import train_order_network
from utils import compute_path_coherence, kendall_tau


# ── Metric 1: Per-step top-1 accuracy ───────────────────────────────────────

def evaluate_per_step_accuracy(
    on: OrderNetwork,
    A_val: np.ndarray,
    optimal_orderings: List[List[int]],
    config: Config,
) -> dict:
    """
    For each step t (0 to N-2), show ON the optimal prefix sigma*[0:t+1]
    and check if top-1 prediction matches sigma*[t+1].

    Returns:
        dict with:
            per_step_acc: list of accuracy per step (relative to random baseline)
            random_baselines: list of 1/(N-t) per step
            overall_acc: mean accuracy across all steps
    """
    device = next(on.parameters()).device
    N = config.num_blocks
    A_tensor = torch.from_numpy(A_val).to(device)  # (S, N, N)
    opt_order = torch.tensor(optimal_orderings, dtype=torch.long, device=device)  # (S, N)

    on.eval()
    num_seqs = len(A_val)
    per_step_correct = np.zeros(N - 1, dtype=np.int64)

    with torch.no_grad():
        batch_size = config.on_batch_size
        for batch_start in range(0, num_seqs, batch_size):
            batch_end = min(batch_start + batch_size, num_seqs)
            B = batch_end - batch_start

            batch_A = A_tensor[batch_start:batch_end]
            batch_order = opt_order[batch_start:batch_end]

            revealed_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
            last_revealed = None

            for step in range(N - 1):
                features = extract_candidate_features(
                    batch_A, revealed_mask, step, last_revealed,
                )
                scores = on(features, revealed_mask)  # (B, N)
                pred = scores.argmax(dim=-1)  # (B,)
                label = batch_order[:, step]   # (B,)
                per_step_correct[step] += (pred == label).sum().item()

                # Reveal the optimal next block (teacher forcing)
                revealed_mask[torch.arange(B, device=device), label] = True
                last_revealed = label

    per_step_acc = per_step_correct / num_seqs
    random_baselines = [1.0 / (N - t) for t in range(N - 1)]
    overall_acc = float(np.mean(per_step_acc))

    return {
        'per_step_acc': per_step_acc.tolist(),
        'random_baselines': random_baselines,
        'overall_acc': overall_acc,
    }


# ── Metric 2: Path Coherence Q(sigma) ────────────────────────────────────────

@torch.no_grad()
def autoregressive_sample_order(
    on: OrderNetwork,
    A_single: torch.Tensor,  # (N, N)
    temperature: float = 1.0,
) -> List[int]:
    """
    Autoregressively sample a full ordering from ON (no teacher forcing).

    At each step, softmax ON scores with given temperature, sample next block,
    reveal it, and continue.
    """
    on.eval()
    device = next(on.parameters()).device
    N = A_single.shape[0]

    A_batch = A_single.unsqueeze(0).to(device)  # (1, N, N)
    revealed_mask = torch.zeros(1, N, dtype=torch.bool, device=device)
    last_revealed = None
    ordering = []

    for step in range(N - 1):
        features = extract_candidate_features(
            A_batch, revealed_mask, step, last_revealed,
        )
        scores = on(features, revealed_mask).squeeze(0)  # (N,)
        if temperature == 0.0:
            chosen = scores.argmax().item()
        else:
            probs = F.softmax(scores / temperature, dim=-1)
            chosen = torch.multinomial(probs, 1).item()
        ordering.append(chosen)
        revealed_mask[0, chosen] = True
        last_revealed = torch.tensor([chosen], device=device, dtype=torch.long)

    # Last remaining block
    remaining = torch.where(~revealed_mask[0])[0]
    ordering.append(remaining[0].item())

    return ordering


def evaluate_path_coherence(
    on: OrderNetwork,
    A_val: np.ndarray,
    dp_results: list,
    config: Config,
) -> dict:
    """
    Sample full orderings from ON at multiple temperatures, compute Q(sigma).

    Returns:
        dict mapping method -> list of Q values across sequences.
        Methods: 'random', 'dp_optimal', 'on_tau_X'
    """
    N = config.num_blocks
    num_seqs = len(A_val)
    temps = config.eval_temperatures
    num_samples = config.eval_num_samples

    results = defaultdict(list)

    for i in range(num_seqs):
        A_i = A_val[i]
        dp_path = dp_results[i]['optimal_path']
        dp_q = dp_results[i]['max_weight']
        results['dp_optimal'].append(dp_q)

        # Random baseline
        rng = np.random.default_rng(config.seed + i)
        rand_qs = []
        for _ in range(config.num_random_baselines):
            rand_ord = list(rng.permutation(N))
            rand_qs.append(compute_path_coherence(rand_ord, A_i))
        results['random'].append(float(np.mean(rand_qs)))

        # ON at each temperature
        A_tensor = torch.from_numpy(A_i)
        for tau in temps:
            tau_qs = []
            for s in range(num_samples):
                ord_sampled = autoregressive_sample_order(on, A_tensor, temperature=tau)
                tau_qs.append(compute_path_coherence(ord_sampled, A_i))
            results[f'on_tau_{tau}'].append(float(np.mean(tau_qs)))

        if (i + 1) % 100 == 0:
            print(f"  Q eval: {i + 1}/{num_seqs}")

    # Average across sequences
    summary = {}
    for key, vals in results.items():
        summary[key] = float(np.mean(vals))

    return {'per_sequence': dict(results), 'summary': summary}


# ── Metric 3: Kendall tau ────────────────────────────────────────────────────

def evaluate_kendall_tau(
    on: OrderNetwork,
    A_val: np.ndarray,
    dp_results: list,
    config: Config,
) -> dict:
    """
    Compute Kendall tau between ON-generated orderings and:
    - DP optimal path (sigma*)
    - L2R ordering [0, 1, ..., N-1]

    Returns:
        dict with per-temperature tau values (mean over sequences).
    """
    N = config.num_blocks
    num_seqs = len(A_val)
    temps = config.eval_temperatures
    num_samples = config.eval_num_samples
    l2r = list(range(N))

    tau_vs_opt = defaultdict(list)
    tau_vs_l2r = defaultdict(list)

    for i in range(num_seqs):
        dp_path = dp_results[i]['optimal_path']
        A_tensor = torch.from_numpy(A_val[i])

        for tau in temps:
            for _ in range(num_samples):
                ord_sampled = autoregressive_sample_order(on, A_tensor, temperature=tau)
                tau_vs_opt[tau].append(kendall_tau(ord_sampled, dp_path))
                tau_vs_l2r[tau].append(kendall_tau(ord_sampled, l2r))

        if (i + 1) % 100 == 0:
            print(f"  tau eval: {i + 1}/{num_seqs}")

    summary_opt = {float(tau): float(np.mean(vals)) for tau, vals in tau_vs_opt.items()}
    summary_l2r = {float(tau): float(np.mean(vals)) for tau, vals in tau_vs_l2r.items()}
    return {'tau_vs_optimal': summary_opt, 'tau_vs_l2r': summary_l2r}


# ── Main evaluation entry point ──────────────────────────────────────────────

def evaluate_order_network(
    on: Optional[OrderNetwork] = None,
    config: Optional[Config] = None,
    save_dir: str = 'eval_results',
) -> dict:
    """
    Full evaluation pipeline: train ON on mock data, then run all three metrics.

    Args:
        on: optionally provide a pre-trained ON; if None, trains from scratch.
        config: Config object.
        save_dir: directory for output plots.

    Returns:
        dict with per_step_acc, q_summary, tau_summary.
    """
    if config is None:
        config = Config()
    Config.set_seed()

    N = config.num_blocks
    n_train = config.num_train_sequences
    n_val = config.num_val_sequences
    total = n_train + n_val

    print("=" * 60)
    print("Phase 3: Order Network Evaluation")
    print("=" * 60)
    print(f"  N={N}, train={n_train}, val={n_val}")

    # ── Generate data ─────────────────────────────────────────────────────
    print("\n[1/5] Generating mock data...")
    A_all = generate_mock_dataset(num_sequences=total, num_blocks=N, seed=config.seed)
    A_train = A_all[:n_train]
    A_val = A_all[n_train:]

    # ── Run DP ─────────────────────────────────────────────────────────────
    print("\n[2/5] Running DP solver on all sequences...")
    dp_all = solve_dp_batch(A_all, num_blocks=N, show_progress=True)

    # ── Train ON ───────────────────────────────────────────────────────────
    print("\n[3/5] Training Order Network...")
    if on is None:
        on = OrderNetwork(
            feature_dim=config.on_feature_dim,
            hidden_dim=config.on_hidden_dim,
            num_layers=config.on_num_layers,
            dropout=config.on_dropout,
        )

    train_results = dp_all[:n_train]
    train_orderings = [r['optimal_path'] for r in train_results]
    train_routing = [r['routing_table'] for r in train_results]

    metrics = train_order_network(
        on, A_train, train_orderings, train_routing, config=config,
    )

    # ── Metric 1: Per-step accuracy ────────────────────────────────────────
    print("\n[4/5] Evaluating per-step accuracy (teacher forcing)...")
    val_orderings = [dp_all[n_train + i]['optimal_path'] for i in range(n_val)]
    acc_result = evaluate_per_step_accuracy(on, A_val, val_orderings, config)

    print(f"  Overall top-1 accuracy: {acc_result['overall_acc']:.3f}")
    print(f"  Random baseline:        1/16={1/N:.4f} (step 0), 1/2=0.5 (step 14)")

    # ── Metric 2: Path coherence Q ─────────────────────────────────────────
    print("\n[5/6] Evaluating path coherence Q(sigma)...")
    val_dp_results = dp_all[n_train:]
    q_result = evaluate_path_coherence(on, A_val, val_dp_results, config)

    print("\n  Q(sigma) Summary (avg over val set):")
    for key, val in sorted(q_result['summary'].items()):
        print(f"    {key:20s} = {val:.4f}")

    # ── Metric 3: Kendall tau ──────────────────────────────────────────────
    print("\n[6/6] Evaluating Kendall tau...")
    tau_result = evaluate_kendall_tau(on, A_val, val_dp_results, config)

    print("\n  Kendall tau Summary:")
    print("    vs DP optimal:")
    for tau, val in sorted(tau_result['tau_vs_optimal'].items()):
        print(f"      tau={tau:.1f}: {val:.4f}")
    print("    vs L2R:")
    for tau, val in sorted(tau_result['tau_vs_l2r'].items()):
        print(f"      tau={tau:.1f}: {val:.4f}")

    # ── Plots ───────────────────────────────────────────────────────────────
    print("\n[Plot] Generating evaluation figure...")
    os.makedirs(save_dir, exist_ok=True)
    _make_eval_plots(acc_result, q_result, tau_result, metrics, config, save_dir)

    return {
        'per_step_acc': acc_result,
        'q_summary': q_result['summary'],
        'tau_summary': tau_result,
        'training_metrics': metrics,
    }


# ── Plotting ─────────────────────────────────────────────────────────────────

def _make_eval_plots(acc_result, q_result, tau_result, training_metrics,
                      config, save_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.3)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    N = config.num_blocks
    temps = config.eval_temperatures

    # ── Subplot 1: Per-step accuracy ─────────────────────────────────────
    ax = axes[0]
    steps = np.arange(N - 1)
    ax.plot(steps, acc_result['per_step_acc'], 'o-', color='#2c7bb6', linewidth=2,
            markersize=6, label='ON (teacher forcing)')
    ax.plot(steps, acc_result['random_baselines'], '--', color='#d7191c',
            linewidth=1.5, label='Random baseline')
    ax.set_xlabel('Step t')
    ax.set_ylabel('Top-1 Accuracy')
    ax.set_title('Metric 1: Per-step Prediction Accuracy', fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)

    # ── Subplot 2: Path coherence Q ──────────────────────────────────────
    ax = axes[1]
    methods = ['random', 'dp_optimal'] + [f'on_tau_{t}' for t in temps]
    labels = ['Random', 'DP Optimal\n(upper bound)'] + [f'ON (τ={t})' for t in temps]
    values = [q_result['summary'][m] for m in methods]
    colors = ['#d7191c'] + ['#1a9641'] + sns.color_palette("Blues", len(temps))

    bars = ax.bar(range(len(methods)), values, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Q(σ)')
    ax.set_title('Metric 2: Path Coherence Q(σ)', fontweight='bold')
    # Annotate values on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=7)

    # ── Subplot 3: Kendall tau ───────────────────────────────────────────
    ax = axes[2]
    x = np.arange(len(temps))
    width = 0.35
    tau_opt = [tau_result['tau_vs_optimal'][t] for t in temps]
    tau_l2r = [tau_result['tau_vs_l2r'][t] for t in temps]

    bars1 = ax.bar(x - width / 2, tau_opt, width, color='#2c7bb6',
                   label='τ(ON, DP optimal)', edgecolor='white')
    bars2 = ax.bar(x + width / 2, tau_l2r, width, color='#fdae61',
                   label='τ(ON, L2R)', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels([f'τ={t}' for t in temps])
    ax.set_ylabel('Kendall τ')
    ax.set_title('Metric 3: Sequence Similarity', fontweight='bold')
    ax.legend(fontsize=8)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

    for bar, val in zip(bars1, tau_opt):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=7)
    for bar, val in zip(bars2, tau_l2r):
        ypos = bar.get_height() + 0.01 if val >= 0 else bar.get_height() - 0.06
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f'{val:.3f}', ha='center', fontsize=7)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'eval_report.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved evaluation plot to {save_path}")
    plt.close(fig)


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    config = Config()
    # Use smaller val set for faster iteration during development
    # config.num_val_sequences = 200  # uncomment for quick test
    evaluate_order_network(config=config)
