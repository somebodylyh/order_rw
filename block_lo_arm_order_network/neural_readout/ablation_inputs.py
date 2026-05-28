"""NR-1 Task 14: four input transforms for the MVP ablations (spec §6).

Each transform is applied to B at training time. Teacher labels (sigma, rank)
are ALWAYS generated from the ORIGINAL B and are NOT changed by the ablation:
the ablation studies whether the student can still recover the teacher's
order when its input is degraded in a specific, named way.

Transforms:
  identity     : no change (control)
  reverse      : B -> B^T per sample (test: does student use edge direction?)
  sym          : B -> 0.5 (B + B^T) per sample (test: does student need directedness?)
  row_shuffle  : apply one fixed random row-permutation to every sample
                 (test: does student use pairwise topology, or only readiness?)
  b_global     : replace every per-sample B[i] with the dataset-mean B_global
                 (test: does the per-sample dynamic signal matter, or is the
                  static slot prior alone enough?)

NR-1 supervision boundary: none of these transforms read NLL or any external
utility signal; they only restructure the model's input.
"""
import numpy as np


def identity(B, rng=None):
    return B


def reverse(B, rng=None):
    """B -> B^T per sample."""
    return np.transpose(B, (0, 2, 1)).copy()


def sym(B, rng=None):
    """B -> 0.5 (B + B^T) per sample."""
    return (0.5 * (B + np.transpose(B, (0, 2, 1)))).astype(B.dtype, copy=False)


def row_shuffle(B, rng):
    """Apply ONE fixed random row-permutation to every sample.

    Same permutation across the whole dataset so the model can in principle
    still learn the (now-shuffled) graph coordinate system; what is broken
    is the natural pairwise topology induced by B[i, :].
    """
    if rng is None:
        raise ValueError("row_shuffle requires an rng for deterministic permutation choice")
    N = B.shape[1]
    perm = rng.permutation(N)
    return B[:, perm, :].copy()


def b_global(B, rng=None):
    """Replace each per-sample B[i] with the dataset-mean B_global."""
    g = B.mean(axis=0, keepdims=True)
    return np.broadcast_to(g, B.shape).astype(B.dtype).copy()


TRANSFORMS = {
    "identity": identity,
    "reverse": reverse,
    "sym": sym,
    "row_shuffle": row_shuffle,
    "b_global": b_global,
}


def apply_input_transform(B, name, rng=None):
    """Dispatch on `name` ('identity' | 'reverse' | 'sym' | 'row_shuffle' | 'b_global')."""
    if name not in TRANSFORMS:
        raise ValueError(f"unknown ablation {name!r}; choose from {sorted(TRANSFORMS)}")
    return TRANSFORMS[name](B, rng)
