"""P1: Bitmask DP (Held-Karp) for maximum-weight Hamiltonian path.

Key design decisions:
- A[next][last] = attention of next (later/query) attending to last (earlier/key).
- routing_table[mask][last] = globally optimal next step from this state,
  computed via reverse DP, so DAgger can do O(1) lookups.
- N=16 → 2^16=65536 masks, runs in ~100-300ms per sequence with numpy.
"""

import numpy as np
from typing import List, Tuple


def _precompute_mask_info(N: int):
    """Precompute popcounts, in/out node lists for all masks up to 2^N."""
    total = 1 << N
    popcounts = np.zeros(total, dtype=np.int32)
    nodes_in = [[] for _ in range(total)]
    nodes_out = [[] for _ in range(total)]
    for mask in range(total):
        cnt = 0
        for i in range(N):
            if mask & (1 << i):
                cnt += 1
                nodes_in[mask].append(i)
            else:
                nodes_out[mask].append(i)
        popcounts[mask] = cnt
    # Group masks by popcount
    masks_by_pc = [[] for _ in range(N + 1)]
    for mask in range(total):
        masks_by_pc[popcounts[mask]].append(mask)
    return popcounts, nodes_in, nodes_out, masks_by_pc


def solve_dp_full(A_matrix: np.ndarray, num_blocks: int = 16) -> dict:
    """
    Run full Bitmask DP + reverse DP to produce optimal_path, max_weight,
    routing_table, and step_labels.

    Args:
        A_matrix: (num_blocks, num_blocks) float32, A[i,j] = attention of
                  i (query/later) attending to j (key/earlier).
        num_blocks: N, typically 16.

    Returns:
        dict with keys:
            optimal_path: list of int, length N, optimal ordering σ*.
            max_weight: float, Q(σ*).
            routing_table: np.ndarray (2^N, N) int8, routing_table[mask][last]
                           = best next node from state (mask, last), or -1.
            step_labels: dict {t: (mask, next_node)} for t in 0..N-2.
    """
    N = num_blocks
    total_masks = 1 << N
    full_mask = total_masks - 1

    A = np.asarray(A_matrix, dtype=np.float32)

    # Precompute helper structures
    popcounts, nodes_in, nodes_out, masks_by_pc = _precompute_mask_info(N)

    # ── Forward DP ─────────────────────────────────────────────────────
    # dp_fwd[mask][last] = max path weight from any start to (mask, last)
    dp_fwd = np.full((total_masks, N), -np.inf, dtype=np.float32)
    predecessor = np.full((total_masks, N), -1, dtype=np.int8)

    # Initialize: single-node masks
    for i in range(N):
        mask = 1 << i
        dp_fwd[mask][i] = 0.0

    # Forward pass: increasing popcount
    for pc in range(1, N):  # 1 to N-1
        for mask in masks_by_pc[pc]:
            last_nodes = nodes_in[mask]
            next_nodes = nodes_out[mask]
            if not next_nodes:
                continue
            for last in last_nodes:
                cur_val = dp_fwd[mask][last]
                if cur_val == -np.inf:
                    continue
                # Try extending to each possible next node
                for nxt in next_nodes:
                    val = cur_val + A[nxt][last]
                    new_mask = mask | (1 << nxt)
                    if val > dp_fwd[new_mask][nxt]:
                        dp_fwd[new_mask][nxt] = val
                        predecessor[new_mask][nxt] = last

    # Find optimal end state
    max_weight = -np.inf
    best_last = -1
    for i in range(N):
        if dp_fwd[full_mask][i] > max_weight:
            max_weight = dp_fwd[full_mask][i]
            best_last = i

    if best_last < 0:
        raise RuntimeError("DP failed: no valid path found (check A matrix).")

    # Reconstruct optimal path (backward from best_last)
    optimal_path = _reconstruct_path(predecessor, full_mask, best_last, N)

    # Build step_labels: for each prefix of the optimal path
    step_labels = {}
    mask = 0
    for t in range(N - 1):
        mask |= (1 << optimal_path[t])
        step_labels[t] = (mask, optimal_path[t + 1])

    # ── Reverse DP (for routing_table) ─────────────────────────────────
    # dp_rev[mask][last] = max additional weight from (mask, last) to any full mask
    dp_rev = np.full((total_masks, N), -np.inf, dtype=np.float32)
    routing = np.full((total_masks, N), -1, dtype=np.int8)

    # Base: full mask → 0 remaining weight
    for i in range(N):
        dp_rev[full_mask][i] = 0.0

    # Reverse pass: decreasing popcount
    for pc in range(N - 1, 0, -1):  # N-1 down to 1
        for mask in masks_by_pc[pc]:
            last_nodes = nodes_in[mask]
            next_nodes = nodes_out[mask]
            for last in last_nodes:
                best_val = -np.inf
                best_nxt = -1
                for nxt in next_nodes:
                    new_mask = mask | (1 << nxt)
                    rev_nxt = dp_rev[new_mask][nxt]
                    if rev_nxt == -np.inf:
                        continue
                    val = A[nxt][last] + rev_nxt
                    if val > best_val:
                        best_val = val
                        best_nxt = nxt
                if best_nxt >= 0:
                    dp_rev[mask][last] = best_val
                    routing[mask][last] = best_nxt

    return {
        'optimal_path': optimal_path,
        'max_weight': float(max_weight),
        'routing_table': routing,
        'step_labels': step_labels,
        'dp_rev': dp_rev,
    }


def _reconstruct_path(predecessor: np.ndarray, full_mask: int,
                      final_node: int, N: int) -> List[int]:
    """Backtrack from final state to reconstruct the optimal ordering."""
    path = [final_node]
    mask = full_mask
    cur = final_node
    for _ in range(N - 1):
        prev = predecessor[mask][cur]
        if prev < 0:
            break
        path.append(prev)
        mask ^= (1 << cur)
        cur = prev
    path.reverse()
    return path


def lookup_next_step(routing_table: np.ndarray, visited_mask: int,
                     last_node: int) -> int:
    """
    O(1) lookup: given any legal intermediate state (visited_mask, last_node),
    return the optimal next node.

    Returns -1 if mask is full or state is unreachable.
    """
    return int(routing_table[visited_mask][last_node])


def solve_dp_batch(A_matrices: np.ndarray, num_blocks: int = 16,
                   show_progress: bool = True) -> list:
    """
    Run DP on a batch of A matrices.

    Args:
        A_matrices: (num_sequences, N, N) float32.
        num_blocks: N.

    Returns:
        list of dicts (same format as solve_dp_full for each sequence).
    """
    results = []
    n_seq = len(A_matrices)
    for i in range(n_seq):
        if show_progress and (i % 100 == 0 or i == n_seq - 1):
            print(f"  DP: {i + 1}/{n_seq} sequences processed")
        res = solve_dp_full(A_matrices[i], num_blocks=num_blocks)
        results.append(res)
    return results


def verify_routing_table(A_matrix: np.ndarray, num_blocks: int = 6) -> dict:
    """
    Verify routing_table correctness for small N by comparing with
    brute-force re-solving from each state.

    Only call on small N (≤ 8) to keep brute force tractable.
    """
    N = num_blocks
    result = solve_dp_full(A_matrix, num_blocks=N)
    routing = result['routing_table']
    total_masks = 1 << N
    full_mask = total_masks - 1

    errors = []
    _, nodes_in, nodes_out, _ = _precompute_mask_info(N)

    for mask in range(total_masks):
        popcount = bin(mask).count('1')
        if popcount == 0 or popcount == N:
            continue
        for last in nodes_in[mask]:
            # Get routing table answer
            table_next = routing[mask][last]
            if table_next < 0:
                continue

            # Brute force: for each possible next, compute optimal continuation
            # by re-running sub-DP from new_mask
            brute_best_next = -1
            brute_best_val = -np.inf
            for nxt in nodes_out[mask]:
                new_mask = mask | (1 << nxt)
                # Compute max remaining from (new_mask, nxt) via mini DP
                remaining = _brute_force_remaining(A_matrix, new_mask, nxt, N,
                                                    nodes_in, nodes_out)
                if remaining is not None:
                    val = A_matrix[nxt][last] + remaining
                    if val > brute_best_val:
                        brute_best_val = val
                        brute_best_next = nxt

            if brute_best_next >= 0 and table_next != brute_best_next:
                errors.append({
                    'mask': mask,
                    'last': last,
                    'table_next': table_next,
                    'brute_next': brute_best_next,
                })

    return {
        'num_errors': len(errors),
        'errors': errors,
        'total_states_checked': sum(1 for m in range(total_masks)
                                     for l in nodes_in[m]
                                     if 0 < bin(m).count('1') < N
                                     and routing[m][l] >= 0),
    }


def _brute_force_remaining(A: np.ndarray, mask: int, last: int, N: int,
                           nodes_in: list, nodes_out: list) -> float:
    """
    Compute max remaining weight from (mask, last) to full mask
    by enumerating all possible continuations. Only for small N.
    """
    remaining_nodes = nodes_out[mask]
    if not remaining_nodes:
        return 0.0

    from itertools import permutations
    best = -np.inf
    for perm in permutations(remaining_nodes):
        total = 0.0
        prev = last
        for nxt in perm:
            total += A[nxt][prev]
            prev = nxt
        if total > best:
            best = total
    return best
