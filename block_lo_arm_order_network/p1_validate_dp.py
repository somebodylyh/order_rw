"""P1 Route A validation: run DP on N=16 A matrices and summarize paths."""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from dp_solver import solve_dp_full


DEFAULT_INPUT = "probe_results/A_n16_direct_100x5.npy"
DEFAULT_OUTPUT = "probe_results/dp_n16_direct_100x5.npz"


def resolve_worker_count(requested=None, num_sequences=1, cpu_count=None):
    """Choose a bounded worker count for DP validation."""
    if cpu_count is None:
        cpu_count = os.cpu_count() or 1
    if requested is None:
        requested = cpu_count
    return max(1, min(int(requested), int(num_sequences), int(cpu_count)))


def _solve_one(args):
    index, A_matrix, num_blocks = args
    result = solve_dp_full(A_matrix, num_blocks=num_blocks)
    return index, {
        "optimal_path": result["optimal_path"],
        "max_weight": result["max_weight"],
    }


def solve_dp_lightweight_batch(A_matrices, num_blocks=16, workers=None):
    """Run DP over a batch and keep only path/weight results."""
    n_seq = len(A_matrices)
    worker_count = resolve_worker_count(workers, n_seq)
    results = [None] * n_seq

    if worker_count == 1:
        for i, A_matrix in enumerate(A_matrices):
            if i % 10 == 0 or i == n_seq - 1:
                print(f"  DP: {i + 1}/{n_seq} sequences processed", flush=True)
            _, result = _solve_one((i, A_matrix, num_blocks))
            results[i] = result
        return results

    print(f"  DP: using {worker_count} workers", flush=True)
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(_solve_one, (i, A_matrices[i], num_blocks))
            for i in range(n_seq)
        ]
        completed = 0
        for future in as_completed(futures):
            i, result = future.result()
            results[i] = result
            completed += 1
            if completed % 10 == 0 or completed == n_seq:
                print(
                    f"  DP: {completed}/{n_seq} sequences processed",
                    flush=True,
                )

    return results


def summarize_paths(results, num_blocks=16):
    """Return Route A acceptance statistics for DP result paths."""
    l2r = list(range(num_blocks))
    r2l = list(range(num_blocks - 1, -1, -1))
    adj_counts = []
    l2r_count = 0
    r2l_count = 0

    for result in results:
        path = list(result["optimal_path"])
        if path == l2r:
            l2r_count += 1
        if path == r2l:
            r2l_count += 1
        adj_counts.append(sum(
            1 for a, b in zip(path, path[1:]) if abs(a - b) == 1
        ))

    avg_adj = float(np.mean(adj_counts)) if adj_counts else 0.0
    return {
        "num_sequences": len(results),
        "l2r_count": l2r_count,
        "r2l_count": r2l_count,
        "adj_counts": adj_counts,
        "avg_adj": avg_adj,
        "passes_route_a_acceptance": (
            l2r_count == 0 and r2l_count == 0 and avg_adj < 4.0
        ),
    }


def save_lightweight_results(results, summary, output_path):
    """Save paths and scalar metrics needed by the next Route A step."""
    paths = np.asarray(
        [result["optimal_path"] for result in results],
        dtype=np.int16,
    )
    weights = np.asarray(
        [result["max_weight"] for result in results],
        dtype=np.float32,
    )
    adj_counts = np.asarray(summary["adj_counts"], dtype=np.int16)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        paths=paths,
        weights=weights,
        adj_counts=adj_counts,
        l2r_count=np.int16(summary["l2r_count"]),
        r2l_count=np.int16(summary["r2l_count"]),
        avg_adj=np.float32(summary["avg_adj"]),
        passes_route_a_acceptance=np.bool_(summary["passes_route_a_acceptance"]),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DP validation for Route A N=16 pair score matrices."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input A .npy file")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output .npz for paths and summary metrics",
    )
    parser.add_argument("--num-blocks", type=int, default=16)
    parser.add_argument("--preview", type=int, default=5)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel DP workers; default uses available CPUs",
    )
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    A = np.load(args.input)
    if A.ndim != 3 or A.shape[1:] != (args.num_blocks, args.num_blocks):
        raise ValueError(
            f"Expected A shape (num_seq, {args.num_blocks}, {args.num_blocks}), "
            f"got {A.shape}"
        )

    print(f"Loaded {args.input}: shape={A.shape}, dtype={A.dtype}", flush=True)
    results = solve_dp_lightweight_batch(
        A,
        num_blocks=args.num_blocks,
        workers=args.workers,
    )
    summary = summarize_paths(results, num_blocks=args.num_blocks)

    for i, result in enumerate(results[:args.preview]):
        path = result["optimal_path"]
        adj = summary["adj_counts"][i]
        print(
            f"Seq {i}: path={path}, adj={adj}/{args.num_blocks - 1}, "
            f"weight={result['max_weight']:.2f}"
        )

    print(
        f"L2R={summary['l2r_count']}/{summary['num_sequences']}, "
        f"R2L={summary['r2l_count']}/{summary['num_sequences']}, "
        f"avg_adj={summary['avg_adj']:.1f}/{args.num_blocks - 1}"
    )
    print(f"Route A acceptance: {summary['passes_route_a_acceptance']}")

    if not args.no_save:
        save_lightweight_results(results, summary, args.output)
        print(f"Saved DP summary: {args.output}")


if __name__ == "__main__":
    main()
