r"""Task 1 — dynamic attention-state feature builder for the attention-conditioned
order policy.

Given a directed attention graph B (B[u, v] = edge u -> v, built via
`directed_graph_policy.build_directed_graph(A) = A.T` with zero diagonal) and the
current partial reveal state (S_t selected, U_t unselected candidates, last step),
produce a 12-d feature vector phi_t(v) per candidate v in U_t.

Feature layout (fixed order, see FEATURE_NAMES):
  A. selected set <-> candidate
     1. mean B[S_t, v]      2. max B[S_t, v]
     3. mean B[v, S_t]      4. max B[v, S_t]
  B. unselected set \ {v} <-> candidate
     5. mean B[U_t\{v}, v]  6. max B[U_t\{v}, v]
     7. mean B[v, U_t\{v}]  8. max B[v, U_t\{v}]
  C. last transition
     9. B[last, v]          10. B[v, last]
  D. progress
     11. t / N              12. |U_t| / N

Edge cases (return 0.0 for the affected block):
  - S_t empty            -> features 1-4 = 0
  - last is None         -> features 9, 10 = 0
  - U_t\{v} empty (|U|=1) -> features 5-8 = 0

No hidden global normalization. `normalize` is an opt-in hook (default None = identity).
"""
import numpy as np

FEATURE_NAMES = [
    "mean_B_S_to_v",     # 1
    "max_B_S_to_v",      # 2
    "mean_B_v_to_S",     # 3
    "max_B_v_to_S",      # 4
    "mean_B_U_to_v",     # 5  (U \ {v})
    "max_B_U_to_v",      # 6
    "mean_B_v_to_U",     # 7
    "max_B_v_to_U",      # 8
    "B_last_to_v",       # 9
    "B_v_to_last",       # 10
    "progress_t_over_N", # 11
    "frac_U_over_N",     # 12
]
NUM_FEATURES = 12


def build_features(B, S_t, U_t, last, t, N, normalize=None):
    """Build the [len(U_t), 12] feature matrix for candidates U_t at step t.

    Parameters
    ----------
    B : (N, N) array_like   directed attention graph, B[u, v] = edge u -> v.
    S_t : sequence of int   selected (revealed) node indices (may be empty).
    U_t : sequence of int   unselected candidate node indices.
    last : int or None      previously selected node, None at t=0.
    t : int                 current step.
    N : int                 number of nodes.
    normalize : callable or None
        Optional opt-in transform applied to the final X (X -> normalize(X)).
        Default None = no normalization. Exposed explicitly so any scaling is
        the caller's choice, never hidden.

    Returns
    -------
    X : (len(U_t), 12) float64
    candidates : (len(U_t),) int64   == np.asarray(U_t)
    feature_names : list[str]         length 12
    """
    B = np.asarray(B, dtype=np.float64)
    S = np.asarray(list(S_t), dtype=np.int64)
    U = np.asarray(list(U_t), dtype=np.int64)
    nU = U.shape[0]
    X = np.zeros((nU, NUM_FEATURES), dtype=np.float64)

    # A. selected set <-> candidate
    if S.shape[0] > 0 and nU > 0:
        BS_v = B[np.ix_(S, U)]           # [|S|, nU]: rows in S, columns are candidates -> B[u in S, v]
        X[:, 0] = BS_v.mean(axis=0)
        X[:, 1] = BS_v.max(axis=0)
        Bv_S = B[np.ix_(U, S)]           # [nU, |S|]: rows are candidates, columns in S -> B[v, u in S]
        X[:, 2] = Bv_S.mean(axis=1)
        X[:, 3] = Bv_S.max(axis=1)

    # B. unselected set \ {v} <-> candidate
    if nU > 1:
        M = B[np.ix_(U, U)]              # [nU, nU]: M[i, j] = B[U[i], U[j]]; diagonal = B[v, v] = 0
        # mean over U\{v}: column/row sums already exclude self because diag(B)=0; divide by (|U|-1)
        X[:, 4] = M.sum(axis=0) / (nU - 1)   # mean_{u in U\{v}} B[u, v]
        X[:, 6] = M.sum(axis=1) / (nU - 1)   # mean_{u in U\{v}} B[v, u]
        Mc = M.copy()
        np.fill_diagonal(Mc, -np.inf)        # exclude v itself from the max
        X[:, 5] = Mc.max(axis=0)             # max_{u in U\{v}} B[u, v]
        X[:, 7] = Mc.max(axis=1)             # max_{u in U\{v}} B[v, u]

    # C. last transition
    if last is not None and nU > 0:
        X[:, 8] = B[last, U]             # B[last, v]
        X[:, 9] = B[U, last]             # B[v, last]

    # D. progress
    if nU > 0:
        X[:, 10] = float(t) / float(N)
        X[:, 11] = float(nU) / float(N)

    if normalize is not None:
        X = np.asarray(normalize(X), dtype=np.float64)

    return X, U.copy(), list(FEATURE_NAMES)


def _self_test():
    rng = np.random.default_rng(0)
    N = 8
    B = rng.random((N, N))
    np.fill_diagonal(B, 0.0)
    print("feature_names:", FEATURE_NAMES)
    # t=0: empty S, no last
    X0, cand0, _ = build_features(B, [], list(range(N)), last=None, t=0, N=N)
    assert X0.shape == (N, 12) and not np.isnan(X0).any()
    assert np.all(X0[:, 0:4] == 0.0) and np.all(X0[:, 8:10] == 0.0)
    # t>0
    perm = rng.permutation(N)
    t = 3
    Xt, candt, _ = build_features(B, perm[:t].tolist(), perm[t:].tolist(),
                                  last=int(perm[t - 1]), t=t, N=N)
    assert Xt.shape == (N - t, 12) and not np.isnan(Xt).any()
    print("self-test OK: t=0 shape", X0.shape, "| t>0 shape", Xt.shape)


if __name__ == "__main__":
    _self_test()
