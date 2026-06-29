"""Test mock attention data generation and full P0+P1 pipeline."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config import Config
from attention_extractor import (
    mock_attention_matrix,
    generate_mock_dataset,
    simulate_sparse_extraction,
    simulate_sparse_extraction_batch,
    check_attention_signal,
    print_signal_report,
)
from dp_solver import solve_dp_full, solve_dp_batch
from utils import compute_path_coherence


def test_mock_matrix_has_adjacent_signal():
    """Mock attention matrix should have higher values for adjacent blocks."""
    Config.set_seed()
    A = mock_attention_matrix(num_blocks=16)
    assert A.shape == (16, 16)
    assert np.all(A >= 0), "All values should be non-negative"
    assert np.all(np.diag(A) == 0), "Diagonal should be zero"

    # Check adjacent vs non-adjacent
    adj_vals = []
    nonadj_vals = []
    for i in range(16):
        for j in range(16):
            if i == j:
                continue
            if abs(i - j) == 1:
                adj_vals.append(A[i, j])
            else:
                nonadj_vals.append(A[i, j])

    adj_mean = np.mean(adj_vals)
    nonadj_mean = np.mean(nonadj_vals)
    ratio = adj_mean / max(nonadj_mean, 1e-8)

    print(f"  Adjacent mean:   {adj_mean:.4f}")
    print(f"  Non-adj mean:    {nonadj_mean:.4f}")
    print(f"  Ratio:           {ratio:.1f}x")

    assert adj_mean > nonadj_mean * 2, \
        f"Adjacent ({adj_mean}) should be much higher than non-adj ({nonadj_mean})"
    print("  ✓ Mock matrix has strong adjacent-block signal")


def test_generate_mock_dataset():
    """Batch generation of mock A matrices."""
    Config.set_seed()
    n_seq = 100
    A_batch = generate_mock_dataset(num_sequences=n_seq, num_blocks=16)
    assert A_batch.shape == (n_seq, 16, 16)
    assert A_batch.dtype == np.float32
    print(f"  Generated {n_seq} mock A matrices, shape={A_batch.shape}")
    print("  ✓ Mock dataset generation works")


def test_sparse_extraction():
    """Sparse extraction should reconstruct A with some fidelity."""
    Config.set_seed()
    A_true = mock_attention_matrix(num_blocks=16)
    A_sparse, vis_mask = simulate_sparse_extraction(A_true, M=3, seed=42)

    # With M=3, some pairs remain unobserved
    unobserved = (~vis_mask) & (~np.eye(16, dtype=bool))
    obs_rate = vis_mask.sum() / (16 * 15)  # excluding diagonal
    print(f"  Observation rate: {obs_rate:.2%}")

    # Observed pairs should match exactly (up to float precision)
    observed = vis_mask & (~np.eye(16, dtype=bool))
    if observed.any():
        obs_diff = np.abs(A_sparse[observed] - A_true[observed]).max()
        print(f"  Max diff on observed pairs: {obs_diff:.10f}")
        assert obs_diff < 1e-6, f"Observed pairs should match exactly"

    # Unobserved pairs should be filled with global mean (not 0)
    if unobserved.any():
        unobs_mean = A_sparse[unobserved].mean()
        obs_nondiag = A_sparse[observed].mean()
        print(f"  Unobserved fill mean: {unobs_mean:.6f}")
        print(f"  Observed mean:        {obs_nondiag:.6f}")
        assert unobs_mean > 0, "Unobserved pairs should not be 0 (global mean fill)"
    print("  ✓ Sparse extraction + global mean fill works correctly")


def test_cold_start_signal():
    """check_attention_signal should detect strong adjacent signal."""
    Config.set_seed()
    A_batch = generate_mock_dataset(num_sequences=50, num_blocks=16)

    stats = check_attention_signal(A_batch)
    print_signal_report(stats)

    assert stats['signal_ready'], \
        "Signal should be ready with exaggerated mock data"
    assert stats['delta_from_uniform'] > 0.01
    assert stats['adjacent_vs_nonadjacent'] > 0.1
    print("  ✓ Cold-start monitoring detects strong signal")


def test_full_pipeline_small_batch():
    """End-to-end: mock data → signal check → DP → verify Q."""
    Config.set_seed()
    n_seq = 20
    A_batch = generate_mock_dataset(num_sequences=n_seq, num_blocks=16)

    # Cold-start check
    stats = check_attention_signal(A_batch)
    assert stats['signal_ready'], "Signal not ready!"

    # Simulate sparse extraction
    A_sparse, _ = simulate_sparse_extraction_batch(A_batch, M=3)
    stats_sparse = check_attention_signal(A_sparse)
    print(f"  After sparse extraction (M=3): delta={stats_sparse['delta_from_uniform']:.6f}")

    # Run DP
    results = solve_dp_batch(A_sparse, num_blocks=16, show_progress=False)

    # Verify each sequence
    for i, res in enumerate(results):
        path = res['optimal_path']
        q = res['max_weight']

        # Random baseline for this sequence
        rng = np.random.default_rng(i)
        random_qs = [
            compute_path_coherence(list(rng.permutation(16)), A_sparse[i])
            for _ in range(50)
        ]
        mean_rand_q = np.mean(random_qs)

        assert q > mean_rand_q, \
            f"Seq {i}: DP Q ({q:.4f}) <= random Q ({mean_rand_q:.4f})"
        assert len(path) == 16
        assert len(set(path)) == 16

    print(f"  All {n_seq} sequences: DP Q > random Q")
    print("  ✓ Full P0+P1 pipeline works end-to-end")


if __name__ == '__main__':
    print("=" * 60)
    print("Mock Data & Pipeline Tests")
    print("=" * 60)

    print("\n[Test 1] Mock matrix adjacent signal...")
    test_mock_matrix_has_adjacent_signal()

    print("\n[Test 2] Mock dataset generation...")
    test_generate_mock_dataset()

    print("\n[Test 3] Sparse extraction + global mean fill...")
    test_sparse_extraction()

    print("\n[Test 4] Cold-start signal monitoring...")
    test_cold_start_signal()

    print("\n[Test 5] Full P0+P1 pipeline...")
    test_full_pipeline_small_batch()

    print("\n" + "=" * 60)
    print("All mock data & pipeline tests passed!")
    print("=" * 60)
