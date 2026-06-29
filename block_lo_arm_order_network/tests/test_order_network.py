"""Test Order Network forward pass, feature extraction, and training."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from collections import defaultdict

from config import Config
from order_network import (
    OrderNetwork,
    extract_candidate_features,
    extract_candidate_features_from_masks,
    extract_rich_candidate_features_from_masks,
    get_dagger_epsilon,
)
from attention_extractor import generate_mock_dataset
from dp_solver import solve_dp_batch


def test_feature_extraction_shape():
    """Feature extraction should output (B, N, 9) for all steps."""
    Config.set_seed()
    B, N = 4, 16
    A = torch.randn(B, N, N).abs()  # attention-like, non-negative
    for bi in range(B):
        A[bi].fill_diagonal_(0)

    # Step 0: empty revealed set
    revealed = torch.zeros(B, N, dtype=torch.bool)
    feats = extract_candidate_features(A, revealed, step_t=0, last_revealed=None)
    assert feats.shape == (B, N, 9), f"Expected (B,N,9), got {feats.shape}"

    # Affinity features should be 0 at step 0 (empty revealed)
    assert (feats[..., :6] == 0).all(), "Affinity features should be 0 at step 0"

    # Step 5: some revealed blocks
    for b in range(B):
        # Reveal a few blocks (different for each batch item)
        for j in range(5):
            revealed[b, (b * 3 + j) % N] = True
    last = torch.tensor([(b * 3) % N for b in range(B)], dtype=torch.long)
    feats2 = extract_candidate_features(A, revealed, step_t=5, last_revealed=last)
    assert feats2.shape == (B, N, 9)
    # Affinity features should now have nonzero values
    assert not (feats2[..., :6] == 0).all(), "Affinity features should be nonzero after reveals"

    print("  ✓ Feature extraction shape correct at both step=0 and step=5")


def test_feature_extraction_from_integer_masks_shape():
    Config.set_seed()
    B, N = 3, 16
    A = torch.randn(B, N, N).abs()
    visited_masks = torch.tensor([0b1, 0b1010, 0b111000], dtype=torch.long)
    last_nodes = torch.tensor([0, 3, 5], dtype=torch.long)

    feats, revealed = extract_candidate_features_from_masks(
        A,
        visited_masks,
        last_nodes,
    )

    assert feats.shape == (B, N, 9)
    assert revealed.shape == (B, N)
    assert revealed[0, 0]
    assert revealed[1, 1]
    assert revealed[1, 3]
    assert feats[0, :, 7].eq(1 / (N - 1)).all()


def test_feature_extraction_from_integer_masks_sanitizes_nonfinite_a():
    B, N = 2, 4
    A = torch.randn(B, N, N)
    A[:, torch.arange(N), torch.arange(N)] = float("-inf")
    visited_masks = torch.tensor([0b0001, 0b0110], dtype=torch.long)
    last_nodes = torch.tensor([0, 2], dtype=torch.long)

    feats, _ = extract_candidate_features_from_masks(A, visited_masks, last_nodes)

    assert torch.isfinite(feats).all()


def test_rich_feature_extraction_from_integer_masks_shape_and_finiteness():
    B, N = 2, 4
    A = torch.randn(B, N, N)
    A[:, torch.arange(N), torch.arange(N)] = float("-inf")
    visited_masks = torch.tensor([0b0001, 0b0110], dtype=torch.long)
    last_nodes = torch.tensor([0, 2], dtype=torch.long)

    feats, revealed = extract_rich_candidate_features_from_masks(
        A,
        visited_masks,
        last_nodes,
    )

    assert feats.shape == (B, N, 5 * N + 2)
    assert revealed.shape == (B, N)
    assert torch.isfinite(feats).all()


def test_feature_extraction_vectorized():
    """Feature extraction must be vectorized: no for-loops over candidates."""
    Config.set_seed()
    B, N = 64, 16  # batch size 64
    A = torch.randn(B, N, N).abs()
    for bi in range(B):
        A[bi].fill_diagonal_(0)
    revealed = torch.zeros(B, N, dtype=torch.bool)
    for b in range(B):
        revealed[b, :3] = True
    last = torch.zeros(B, dtype=torch.long)

    import time
    start = time.perf_counter()
    for _ in range(50):
        _ = extract_candidate_features(A, revealed, step_t=3, last_revealed=last)
    elapsed = time.perf_counter() - start
    avg_ms = elapsed / 50 * 1000
    print(f"  Feature extraction: {avg_ms:.2f} ms/call (B={B}, N={N})")
    assert avg_ms < 50, f"Too slow: {avg_ms:.1f} ms/call"
    print("  ✓ Feature extraction is fast (vectorized)")


def test_on_forward_shape():
    """ON forward pass: correct shapes and masking."""
    Config.set_seed()
    B, N, F = 8, 16, 9
    on = OrderNetwork(feature_dim=F, hidden_dim=64, num_layers=2, dropout=0.1)

    feats = torch.randn(B, N, F)
    revealed = torch.zeros(B, N, dtype=torch.bool)
    revealed[:, 0] = True  # first block revealed in all sequences

    scores = on(feats, revealed)  # (B, N)
    assert scores.shape == (B, N)
    # Revealed positions should be -inf
    assert (scores[:, 0] == float('-inf')).all(), "Revealed positions must be -inf"
    # Unrevealed positions should be finite
    assert torch.isfinite(scores[:, 1:]).all(), "Unrevealed positions must be finite"

    print("  ✓ ON forward pass: correct shapes and masking")


def test_dagger_epsilon_schedule():
    """DAgger epsilon schedule should follow the specified pattern."""
    class DummyCfg:
        dagger_start_epoch = 10
        dagger_max_epsilon = 0.3

    cfg = DummyCfg()

    # Epochs 0-9: pure teacher forcing
    for e in range(10):
        eps = get_dagger_epsilon(e, cfg)
        assert eps == 0.0, f"Epoch {e}: expected ε=0, got {eps}"

    # Epoch 10-30: linear ramp
    eps10 = get_dagger_epsilon(10, cfg)
    eps20 = get_dagger_epsilon(20, cfg)
    eps30 = get_dagger_epsilon(30, cfg)
    assert eps10 == 0.0, f"Epoch 10: {eps10}"
    assert 0.14 < eps20 < 0.16, f"Epoch 20: {eps20} (expected ~0.15)"
    assert eps30 == 0.3, f"Epoch 30: {eps30}"

    # Epoch 31+: max epsilon
    eps40 = get_dagger_epsilon(40, cfg)
    assert eps40 == 0.3

    print("  ✓ DAgger epsilon schedule correct")


def test_training_runs():
    """End-to-end: generate data → DP → train ON for 3 epochs, verify loss drops."""
    from train_on import train_order_network

    Config.set_seed()
    N = Config.num_blocks
    n_train = 100
    cfg = Config()
    cfg.on_num_epochs = 5
    cfg.on_batch_size = 32
    cfg.dagger_start_epoch = 3  # earlier DAgger for quick test

    # Generate mock data
    print("  Generating mock data...")
    A_batch = generate_mock_dataset(num_sequences=n_train, num_blocks=N)

    print("  Running DP...")
    results = solve_dp_batch(A_batch, num_blocks=N, show_progress=False)
    orderings = [r['optimal_path'] for r in results]
    routing_tables = [r['routing_table'] for r in results]

    # Create ON
    on = OrderNetwork(
        feature_dim=cfg.on_feature_dim,
        hidden_dim=128,
        num_layers=cfg.on_num_layers,
        dropout=cfg.on_dropout,
    )

    print("  Training...")
    metrics = train_order_network(
        on, A_batch, orderings, routing_tables, config=cfg,
    )

    # Check that loss decreases
    losses = metrics['loss']
    assert len(losses) == 5
    assert losses[-1] < losses[0] * 0.95, \
        f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
    # Accuracy should be above random baseline
    assert metrics['acc_top1'][-1] > 1.0 / N, \
        f"Accuracy {metrics['acc_top1'][-1]:.3f} below random {1.0/N:.3f}"

    print(f"  Loss: {losses[0]:.4f} → {losses[-1]:.4f}")
    print(f"  Top-1 acc: {metrics['acc_top1'][-1]:.3f}")
    print("  ✓ Training converges: loss decreases, accuracy > random")


if __name__ == '__main__':
    print("=" * 60)
    print("Order Network Tests")
    print("=" * 60)

    print("\n[Test 1] Feature extraction shape...")
    test_feature_extraction_shape()

    print("\n[Test 2] Feature extraction performance...")
    test_feature_extraction_vectorized()

    print("\n[Test 3] ON forward pass...")
    test_on_forward_shape()

    print("\n[Test 4] DAgger epsilon schedule...")
    test_dagger_epsilon_schedule()

    print("\n[Test 5] End-to-end training...")
    test_training_runs()

    print("\n" + "=" * 60)
    print("All Order Network tests passed!")
    print("=" * 60)
