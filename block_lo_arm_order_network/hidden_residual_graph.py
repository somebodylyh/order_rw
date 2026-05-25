"""Phase-1 hidden-residual diagnostic: graph/teacher math (physical block frame).
See docs/superpowers/specs/2026-05-25-hidden-residual-order-diagnostic-design.md."""
import numpy as np
from directed_graph_policy import build_directed_graph

N_BLOCKS = 64
BLOCK_LEN = 4
T_LIST = [0, 16, 32, 48]
PHYS_FRAME = "physical_block"  # the ONLY frame the teacher is allowed to score in (guard 1)

def build_B_set(A_all, frame):
    """A_all: (n, N, N) attention in PHYSICAL block frame (extract_A_matrices already remaps).
    Returns (B_x_list, B_G, frame). B = build_directed_graph(A) = A.T with zero diag."""
    assert frame == PHYS_FRAME, f"B must be built in {PHYS_FRAME!r} frame, got {frame!r} (guard 1)"
    A_all = np.asarray(A_all, dtype=np.float64)
    assert A_all.ndim == 3 and A_all.shape[1] == A_all.shape[2], f"bad A_all shape {A_all.shape}"
    B_x_list = [build_directed_graph(A_all[i]) for i in range(A_all.shape[0])]
    B_G = build_directed_graph(A_all.mean(axis=0))
    return B_x_list, B_G, frame


from attn_order_teacher import rollout_order, teacher_scores

def canonical_states(B_G, t_list=T_LIST):
    """Guard 2: ONE global greedy C-D+L rollout over B_G defines the partial states used
    for EVERY sample. Returns {t: (S_t:tuple, U_t:tuple, last:int|None)}."""
    order = rollout_order(np.asarray(B_G, dtype=np.float64), mode="C-D+L",
                          greedy=True, standardize=True)
    order = [int(v) for v in order]
    states = {}
    for t in t_list:
        S = tuple(order[:t])
        U = tuple(order[t:])
        last = order[t - 1] if t > 0 else None
        states[t] = (S, U, last)
    return states


def residual_target(B_x, B_G, states, frame_x, frame_g, mode="C-D+L"):
    """r_x(v|S_t) = s_x(v|S_t) - s_G(v|S_t), scored by the same teacher at the SAME state.
    Guard 1: both graphs must be in physical block frame. Returns {t: {U, r, s_x, s_g}}."""
    assert frame_x == frame_g == PHYS_FRAME, \
        f"B_x/B_G frame mismatch (guard 1): {frame_x!r} vs {frame_g!r}"
    out = {}
    for t, (S, U, last) in states.items():
        q_x, U_x = teacher_scores(B_x, S, U, last, mode=mode)
        q_g, U_g = teacher_scores(B_G, S, U, last, mode=mode)
        assert np.array_equal(U_x, U_g), "candidate sets diverged (state not fixed)"
        out[t] = {"U": U_x, "r": (q_x - q_g), "s_x": q_x, "s_g": q_g}
    return out
