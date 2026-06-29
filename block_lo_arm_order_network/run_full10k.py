"""Ultra-tight full10k NN greedy diagnostics. No dispatch overhead."""
import os, sys, time
from collections import Counter
import numpy as np

_current = os.path.dirname(os.path.abspath(__file__))
if _current not in sys.path:
    sys.path.insert(0, _current)
from order_diagnostics import write_summary_tsv


def _kendall_tau(a, b):
    n = len(a)
    pos_a = np.empty(n, dtype=np.int64)
    pos_a[a] = np.arange(n)
    pairs = 0
    disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            pairs += 1
            if (pos_a[b[i]] - pos_a[b[j]]) * (j - i) > 0:
                disc += 1
    return (pairs - 2 * disc) / max(pairs, 1)

L2R = np.arange(64, dtype=np.int64)


def fast_diagnostics(orders, old_orders=None, n_pair_samples=5000):
    """Sampled diagnostics — avoids O(N^2) pairwise tau."""
    arr = orders.astype(np.int64)
    if arr.ndim == 1:
        arr = arr[None, :]
    n_orders, n = arr.shape

    valid = np.array([len(set(row)) == n and row.min() >= 0 and row.max() < n
                      for row in arr], dtype=bool)
    tau_l2r = np.array([_kendall_tau(row, L2R) for row in arr], dtype=np.float64)

    first_nodes = arr[:, 0]
    first_counts = Counter(first_nodes.tolist())
    probs = np.array(list(first_counts.values()), dtype=np.float64) / max(n_orders, 1)
    first_entropy = float(-(probs * np.log(probs + 1e-12)).sum())

    summary = {
        "n_orders": float(n_orders),
        "n_blocks": float(n),
        "valid_fraction": float(valid.mean()),
        "is_valid_permutation": float(valid.mean()),
        "tau_vs_l2r": float(tau_l2r.mean()),
        "tau_vs_l2r_std": float(tau_l2r.std()),
        "first_node_entropy": first_entropy,
    }
    for rank, (node, count) in enumerate(first_counts.most_common(5), start=1):
        summary[f"first_node_top{rank}"] = float(node)
        summary[f"first_node_top{rank}_freq"] = float(count / n_orders)

    # Sampled batch order agreement
    if n_orders > 1:
        np.random.seed(42)
        n_pairs = min(n_pair_samples, n_orders * (n_orders - 1) // 2)
        idx = np.random.choice(n_orders, size=(n_pairs, 2), replace=True)
        # ensure distinct pairs
        for p in range(n_pairs):
            while idx[p, 0] == idx[p, 1]:
                idx[p, 1] = np.random.randint(n_orders)
        pair_taus = [_kendall_tau(arr[i], arr[j]) for i, j in zip(idx[:, 0], idx[:, 1])]
        summary["batch_order_agreement_tau"] = float(np.mean(pair_taus))
    else:
        summary["batch_order_agreement_tau"] = 0.0

    # top8 overlap with old_orders
    if old_orders is not None:
        old = old_orders.astype(np.int64)
        if old.ndim == 1:
            old = old[None, :]
        overlaps = [len(set(arr[i, :8].tolist()) & set(old[i, :8].tolist())) / min(8, n)
                    for i in range(n_orders)]
        summary["top8_overlap_against_old"] = float(np.mean(overlaps))

    return summary

OUT = "block_lo_arm_order_network/probe_results/attention_curriculum_full10k"
A_PATH = "block_lo_arm_order_network/probe_results/A64_from_N64_50k_10k.npy"
G_PATH = "block_lo_arm_order_network/probe_results/attention_curriculum_real_diag/A_global_train_first8000_n64.npy"

os.makedirs(OUT, exist_ok=True)

print("Loading...", flush=True)
A_all = np.load(A_PATH).astype(np.float32)
A_global = np.load(G_PATH).astype(np.float32)
n_seqs, N = A_all.shape[0], A_all.shape[1]
print(f"  {n_seqs}x{N}x{N}", flush=True)

# Pre-compute sanitized W_global once (was done per-sample before!)
W_global = 0.5 * (A_global + A_global.T)
np.fill_diagonal(W_global, 0.0)


def compute_orders(A_all, lam, label):
    """Process all samples with tight loop. Returns orders (n_seqs,N) int16, weights (n_seqs,) float32."""
    orders = np.zeros((n_seqs, N), dtype=np.int16)
    weights = np.zeros(n_seqs, dtype=np.float32)

    use_global = lam is not None  # None means old_nn
    t0 = time.time()

    for s in range(n_seqs):
        # Copy one A matrix and sanitize in-place (avoid _as_square_attention overhead)
        A = A_all[s].astype(np.float64)
        np.fill_diagonal(A, 0.0)
        W = 0.5 * (A + A.T)
        np.fill_diagonal(W, 0.0)

        # Mix with global
        if use_global:
            W = lam * W + (1.0 - lam) * W_global

        # old_nn_greedy inlined
        starts = np.argsort(W.sum(axis=1))[:2]
        best_order = None
        best_w = -np.inf

        for start in starts:
            order = np.zeros(N, dtype=np.int16)
            visited = np.zeros(N, dtype=bool)
            cur = int(start)
            for t in range(N):
                order[t] = cur
                visited[cur] = True
                if t == N - 1:
                    break
                w_cur = W[cur].copy()
                w_cur[visited] = -np.inf
                cur = int(np.argmax(w_cur))

            pw = 0.0
            for i in range(N - 1):
                pw += float(W[order[i], order[i + 1]])
            if pw > best_w:
                best_w = pw
                best_order = order

        orders[s] = best_order
        weights[s] = best_w

        if (s + 1) % 2000 == 0:
            elapsed = time.time() - t0
            rate = (s + 1) / max(elapsed, 1e-9)
            eta = (n_seqs - s - 1) / max(rate, 1e-9)
            print(f"  [{label}] {s+1}/{n_seqs} | {rate:.1f} seq/s | ETA {eta:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"  [{label}] Done in {elapsed:.1f}s ({elapsed/n_seqs*1000:.1f} ms/seq)", flush=True)
    return orders, weights


def run_method(method, lam, label):
    orders, weights = compute_orders(A_all, lam, label)
    orders_i64 = orders.astype(np.int64)

    old_orders = None
    if method == "mixed":
        print(f"  [{label}] Computing old_nn reference...", flush=True)
        old_orders, _ = compute_orders(A_all, None, "old_nn_ref")

    summary = fast_diagnostics(
        orders_i64,
        old_orders=None if old_orders is None else old_orders.astype(np.int64),
    )
    summary["method"] = method
    summary["lambda_mix"] = float(lam if lam else 0.5)
    summary["mean_path_weight"] = float(weights.mean())

    npz = os.path.join(OUT, f"{label}.npz")
    kv = {"paths": orders, "orders": orders, "weights": weights}
    if old_orders is not None:
        kv["old_orders"] = old_orders
    np.savez(npz, **kv)

    tsv = os.path.join(OUT, f"{label}.tsv")
    write_summary_tsv(tsv, [summary])

    keys = ["method", "lambda_mix", "tau_vs_l2r", "tau_vs_l2r_std",
            "first_node_entropy", "top1_first_node_freq", "valid_fraction"]
    for k in keys:
        if k in summary:
            print(f"    {k}: {summary[k]}", flush=True)
    return summary


# ── Run ──
results = []

print("\n=== old_nn ===", flush=True)
results.append(run_method("old_nn", None, "old_nn"))

for lam in [0.25, 0.50, 0.75]:
    label = f"mixed_lam{str(lam).replace('.', '')}"
    print(f"\n=== {label} ===", flush=True)
    results.append(run_method("mixed", lam, label))

print(f"\n=== DONE ===")
print(f"Output: {OUT}/")
for f in sorted(os.listdir(OUT)):
    st = os.stat(os.path.join(OUT, f))
    print(f"  {f}  ({st.st_size/1024:.0f} KB)")
