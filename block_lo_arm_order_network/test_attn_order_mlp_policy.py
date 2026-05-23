r"""Tests for the batched Attn-Order MLP sampler (attn_order_mlp_policy).

Two layers:
  1. Correctness: the batched-torch features/scores match the canonical numpy
     build_features + MLP path (score_state_numpy) on random states. This is the
     feature-math gate — if it passes, the batched sampler scores the same policy.
  2. Orientation behaviour on the real clean ckpt20000 B + distilled MLP:
     original = reverse chain (tau<0), reversed = forward (tau>0), source_start
     anchors t=0 at the source node and is more forward than original.

Run:  python -m pytest block_lo_arm_order_network/test_attn_order_mlp_policy.py -q
  or: python block_lo_arm_order_network/test_attn_order_mlp_policy.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_HERE))

from directed_graph_policy import build_directed_graph
from train_attn_order_mlp import OrderMLP
from attn_order_features import NUM_FEATURES
import attn_order_mlp_policy as P

CKPT20K_A = _HERE / "probe_results/clean_method_graph_rw_v3_from20k/A_global_eval.npy"
CKPT20K_MLP = _REPO / "probe_results/attention_order_mlp/phase1_text_mlp_ckpt20k.pt"


def _rand_mlp(seed=0):
    torch.manual_seed(seed)
    m = OrderMLP(in_dim=NUM_FEATURES, hidden=64, layers=2, act="gelu")
    m.eval()
    return m


def _batched_scores_for_state(B_np, mlp, selected_mask, last, t, N, device):
    """Run the batched feature+MLP path for a single state; return raw scores (N,)."""
    sel = torch.tensor(selected_mask, dtype=torch.bool, device=device).unsqueeze(0)
    last_idx = torch.tensor([last if last is not None else -1], dtype=torch.long, device=device)
    B = torch.from_numpy(np.ascontiguousarray(B_np)).to(device=device, dtype=torch.float32)
    BT = B.t().contiguous()
    feat = P._batched_features(B, BT, sel, last_idx, t, N)
    with torch.no_grad():
        raw = mlp(feat)[0]
    return raw.cpu().numpy().astype(np.float64)


@pytest.mark.parametrize("t", [0, 1, 5, 11])
def test_batched_features_match_numpy(t):
    """Batched scores over candidates U equal the canonical build_features+MLP scores."""
    rng = np.random.default_rng(123)
    N = 12
    A = rng.random((N, N))
    B = build_directed_graph(A)            # float64, zero diagonal
    mlp = _rand_mlp(seed=7)
    device = torch.device("cpu")

    perm = rng.permutation(N)
    S = perm[:t].tolist()
    U = sorted(int(x) for x in perm[t:])   # candidate ordering we compare on
    last = int(perm[t - 1]) if t > 0 else None

    ref = P.score_state_numpy(B, mlp, S, U, last, t, N)   # aligned to U
    selected_mask = np.zeros(N, dtype=bool)
    selected_mask[S] = True
    batched = _batched_scores_for_state(B, mlp, selected_mask, last, t, N, device)
    batched_on_U = np.array([batched[u] for u in U])

    assert np.allclose(ref, batched_on_U, atol=1e-3, rtol=1e-3), (
        f"t={t}: max abs diff {np.abs(ref - batched_on_U).max():.2e}")


def test_sampled_orders_are_legal_permutations():
    rng = np.random.default_rng(0)
    N = 16
    B = build_directed_graph(rng.random((N, N)))
    mlp = _rand_mlp(seed=1)
    device = torch.device("cpu")
    for orient in P.ORIENTATIONS:
        orders = P.sample_orders_batched_mlp(
            B, batch_size=8, mlp=mlp, orientation=orient, base_seed=42, device=device,
            tau=0.5, tau_start=0.1, top_k=4, src_rho=0.3,
        ).cpu().numpy()
        assert orders.shape == (8, N)
        for row in orders:
            assert sorted(row.tolist()) == list(range(N)), f"{orient}: not a permutation"


def _kendall_vs_l2r(orders):
    from scipy.stats import kendalltau
    raster = np.arange(orders.shape[1])
    taus = np.array([kendalltau(o, raster).correlation for o in orders])
    return float(np.nanmean(taus)), float(np.nanmean(np.abs(taus)))


@pytest.mark.skipif(not (CKPT20K_A.exists() and CKPT20K_MLP.exists()),
                    reason="clean ckpt20000 B / distilled MLP not present")
def test_orientation_behaviour_on_clean_B():
    """On the real clean substrate: original=reverse, reversed=forward, source_start=anchored+forward."""
    device = torch.device("cpu")
    B = build_directed_graph(np.load(CKPT20K_A).astype(np.float64))
    N = B.shape[0]
    mlp = P.load_order_mlp(CKPT20K_MLP, device)
    readiness = P.readiness_vector(B)
    src_node = int(np.argmax(readiness))

    K = 128
    common = dict(batch_size=K, mlp=mlp, device=device, tau=0.5, tau_start=0.1, top_k=4)

    o_orig = P.sample_orders_batched_mlp(B, orientation="original", base_seed=100, **common).cpu().numpy()
    o_rev = P.sample_orders_batched_mlp(B, orientation="reversed", base_seed=100, **common).cpu().numpy()
    o_src = P.sample_orders_batched_mlp(B, orientation="source_start", base_seed=100, src_rho=0.3, **common).cpu().numpy()

    tau_orig, abs_orig = _kendall_vs_l2r(o_orig)
    tau_rev, abs_rev = _kendall_vs_l2r(o_rev)
    tau_src, abs_src = _kendall_vs_l2r(o_src)

    # original: reverse chain (negative tau, strong |tau|)
    assert tau_orig < 0, f"expected reverse chain, got tau={tau_orig:.3f}"
    assert abs_orig > 0.4, f"expected strong chain, got abs_tau={abs_orig:.3f}"
    # reversed: flips to forward
    assert tau_rev > 0, f"expected forward after reversal, got tau={tau_rev:.3f}"
    assert abs(tau_rev + tau_orig) < 0.05, "reversed tau should be ~ -original tau"
    # source_start: t=0 anchored at the source node, and clearly more forward than original
    start_mode = np.bincount(o_src[:, 0], minlength=N).argmax()
    assert start_mode == src_node, f"source_start should start at source node {src_node}, got {start_mode}"
    assert tau_src > tau_orig, f"source_start ({tau_src:.3f}) should be more forward than original ({tau_orig:.3f})"

    print(f"orig tau={tau_orig:.3f} |tau|={abs_orig:.3f} | rev tau={tau_rev:.3f} | "
          f"src tau={tau_src:.3f} start_mode={start_mode} (src_node={src_node})")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
