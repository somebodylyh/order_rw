"""P2 Route A: build ON next-step training examples from DP paths."""

import argparse
from pathlib import Path

import numpy as np


DEFAULT_INPUT = "probe_results/dp_n16_direct_100x5.npz"
DEFAULT_OUTPUT = "probe_results/on_training_data.npz"


def _validate_paths(paths, num_blocks):
    if paths.ndim != 2 or paths.shape[1] != num_blocks:
        raise ValueError(
            f"Expected paths shape (num_seq, {num_blocks}), got {paths.shape}"
        )
    expected = set(range(num_blocks))
    for i, path in enumerate(paths):
        if set(int(x) for x in path) != expected:
            raise ValueError(f"Path {i} is not a permutation of 0..{num_blocks - 1}")


def build_training_examples(paths, num_blocks=16):
    """
    Convert DP paths to (visited_mask, last_node) -> next_node examples.

    For a path [p0, p1, ..., p15], examples are:
      mask={p0}, last=p0 -> p1
      mask={p0,p1}, last=p1 -> p2
      ...
      mask={p0..p14}, last=p14 -> p15
    """
    paths = np.asarray(paths, dtype=np.int16)
    _validate_paths(paths, num_blocks)

    num_sequences = paths.shape[0]
    num_examples = num_sequences * (num_blocks - 1)
    visited_masks = np.empty(num_examples, dtype=np.uint32)
    last_nodes = np.empty(num_examples, dtype=np.int16)
    next_nodes = np.empty(num_examples, dtype=np.int16)
    seq_indices = np.empty(num_examples, dtype=np.int32)
    step_indices = np.empty(num_examples, dtype=np.int16)

    out = 0
    for seq_idx, path in enumerate(paths):
        mask = 0
        for step_idx in range(num_blocks - 1):
            last = int(path[step_idx])
            nxt = int(path[step_idx + 1])
            mask |= (1 << last)

            visited_masks[out] = mask
            last_nodes[out] = last
            next_nodes[out] = nxt
            seq_indices[out] = seq_idx
            step_indices[out] = step_idx
            out += 1

    return {
        "visited_masks": visited_masks,
        "last_nodes": last_nodes,
        "next_nodes": next_nodes,
        "seq_indices": seq_indices,
        "step_indices": step_indices,
    }


def save_training_data(examples, paths, output_path, num_blocks=16):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        paths=np.asarray(paths, dtype=np.int16),
        num_blocks=np.int16(num_blocks),
        **examples,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build Route A ON next-step training examples from DP paths."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input DP .npz")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output data .npz")
    parser.add_argument("--num-blocks", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    data = np.load(args.input)
    paths = data["paths"]
    examples = build_training_examples(paths, num_blocks=args.num_blocks)
    save_training_data(
        examples,
        paths,
        args.output,
        num_blocks=args.num_blocks,
    )
    print(
        f"Saved {len(examples['next_nodes'])} examples "
        f"from {len(paths)} paths to {args.output}"
    )


if __name__ == "__main__":
    main()
