"""Phase-1 hidden-residual diagnostic: graph/teacher math (physical block frame).
See docs/superpowers/specs/2026-05-25-hidden-residual-order-diagnostic-design.md."""
import numpy as np
from directed_graph_policy import build_directed_graph
from attn_order_teacher import rollout_order, teacher_scores

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
        # Defensive: both teacher_scores calls get the same fixed (S,U,last) from
        # canonical_states, so U_x/U_g must match; this catches any future refactor break.
        assert np.array_equal(U_x, U_g), "candidate sets diverged (state not fixed)"
        out[t] = {"U": U_x, "r": (q_x - q_g), "s_x": q_x, "s_g": q_g}
    return out


def gate_metrics(A_all, A_half_a, A_half_b, t_list=T_LIST, frame=PHYS_FRAME, snr_threshold=1.0):
    """Pre-gate: does per-sample B_x vary, and is r_x above the split-pass noise floor?
    A_all: all-pass mean substrate; A_half_a/A_half_b: the two pass-halves (2-vs-2 at M=4).
    Returns dict of metrics + bool 'passed'."""
    B_x_list, B_G, fr = build_B_set(A_all, frame=frame)
    states = canonical_states(B_G, t_list=t_list)
    # signal: mean over samples of |r_x| at the largest probed t (most context)
    t_sig = max(t_list)
    r_sig = []
    for B_x in B_x_list:
        out = residual_target(B_x, B_G, states, fr, fr)
        r_sig.append(np.abs(out[t_sig]["r"]).mean())
    r_norm = float(np.mean(r_sig))
    # noise floor: same quantity from two pass-halves' graphs vs each other
    B_a_mean = build_directed_graph(np.asarray(A_half_a, np.float64).mean(0))
    B_b_mean = build_directed_graph(np.asarray(A_half_b, np.float64).mean(0))
    noise = np.abs(residual_target(B_a_mean, B_b_mean, states, frame, frame)[t_sig]["r"]).mean()
    noise_floor = float(noise)
    mean_abs = float(np.mean([np.abs(B_x - B_G).mean() for B_x in B_x_list]))
    def _safe_corr(a, b):
        if a.std() < 1e-12 or b.std() < 1e-12:
            return 0.0
        return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    corr = float(np.mean([_safe_corr(B_x, B_G) for B_x in B_x_list]))
    snr = r_norm / (noise_floor + 1e-9)
    return {"r_norm": r_norm, "noise_floor": noise_floor, "snr": snr,
            "mean_abs_Bx_minus_BG": mean_abs, "corr_Bx_BG": corr,
            "passed": bool(snr >= snr_threshold and r_norm > noise_floor)}
