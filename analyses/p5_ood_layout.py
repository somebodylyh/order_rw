"""P5 OOD-layout diagnostic: evaluate H utility under alternate fixed permutations.

Uses the SAME frozen checkpoint but remaps data chunks to alternate layouts
(π_alt), then reruns the P5 utility pipeline. Answers: does H provide utility
correction when the attention scaffold is degraded by an unseen layout?

Key functions:
  remap_chunks_to_alt_perm  — model-token-space chunk remapping (π_train → π_alt)
  run_ood_phase0            — full P5 pipeline under a given layout
"""
import pathlib, sys, time, json as _json
import numpy as np, torch, torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
_BLOCK = ROOT / "block_lo_arm_order_network"
for _p in (str(ROOT), str(_BLOCK), str(ROOT / "analyses")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analyses.p5_utility_controller import (
    load_p5_ckpt, order_nll, utility_pool, candidate_orders,
    headroom_stats, soft_pref, pairwise_loss, BOnlyController,
    ScaffoldedController, block_b_features, sigma_from_B65,
    _apply_h_mode, _split_idx, _regret,
    predict_order, eval_controller_nll, p5_metrics, classify_p5,
    multi_layer_hidden_states, extract_h_by_context,
    SEQ_LEN, N, BLOCK_LEN,
)
from clean_training_protocol import (
    CleanPermutation, physical_blocks_to_model_token_order,
    phys_to_model_idx_clean, model_to_phys_idx_clean,
)
from analyses.physical_signal_source import random_reveal_orders, carrier_b65_per_text
from per_head_order_scan import _attn_to_A_block_loss_aligned_with_none_vec
from none_separated_block_graph import build_none_separated_B


# ── Layout utilities ─────────────────────────────────────────────────────────

def create_random_alt_perm(seed, n_blocks=N):
    """Create a random CleanPermutation (shuffled model→physical block mapping)."""
    rng = np.random.default_rng(seed)
    model_to_phys = rng.permutation(n_blocks).astype(np.int64)
    return CleanPermutation.from_model_to_phys(model_to_phys)


def remap_chunks_to_alt_perm(chunks_train, cp_train, cp_alt):
    """Remap model-token-space chunks from π_train to π_alt.

    chunks_train: (M, SEQ_LEN) LongTensor of vocabulary token IDs under π_train.
    Returns: (M, SEQ_LEN) LongTensor of vocabulary token IDs under π_alt.
    """
    # Step 1: model-space vocab IDs → physical-space vocab IDs (undo π_train)
    idx_phys = model_to_phys_idx_clean(chunks_train, cp_train)
    # Step 2: physical-space → new model-space (apply π_alt)
    idx_alt = phys_to_model_idx_clean(idx_phys, cp_alt)
    return idx_alt


# ── Multi-head B65 extraction under a specific layout ────────────────────────

@torch.no_grad()
def extract_multihead_b65_alt(model, chunks_alt, cp_alt, layer, heads, n_reveals=6,
                               fixed_reveal_seed=0, device="cpu"):
    """Extract per-text B65 (mean across heads) under an alternate layout.

    chunks_alt: (M, SEQ_LEN) in cp_alt's model-token space.
    cp_alt: CleanPermutation for the alternate layout.
    Returns: (B_list, sigma_B_list) where B_list[t] is (65,65) mean-B65,
             sigma_B_list[t] is (64,) physical-block order.
    """
    M = chunks_alt.shape[0]
    reveals = random_reveal_orders(n_reveals, fixed_reveal_seed)
    inv_arr = cp_alt.inv_perm_model_to_phys.cpu().numpy()
    B_list = []
    for t in range(M):
        # Accumulate A-matrices across reveals (all heads, one layer)
        A_acc = None  # (64, 65) float64
        for rev in reveals:
            # rev is already model-token indices (0..255), passed directly to forward_fn
            rev_t = torch.from_numpy(rev[None, :]).to(device)
            _, _, attn_list = model.forward_fn(
                chunks_alt[t:t+1].to(device), rev_t, return_attentions=True
            )
            attn = torch.stack(attn_list, 0).cpu().numpy()[:, 0]  # (L, HS, 257, 257)
            # Select specified layer, selected heads
            attn_per_head = attn[layer, heads]              # (n_heads, 257, 257)
            # Average attention across heads, then convert to A
            h_avg = attn_per_head.mean(axis=0)              # (257, 257)
            # _attn_to_A_block_loss_aligned_with_none_vec handles leading dims
            A = _attn_to_A_block_loss_aligned_with_none_vec(
                h_avg[None, None, :, :], rev, inv_arr,
                seq_len=SEQ_LEN, num_blocks=N, block_len=BLOCK_LEN
            )                                                # (1, 1, 64, 65)
            A_flat = A[0, 0].astype(np.float64)             # (64, 65)
            A_acc = A_flat if A_acc is None else A_acc + A_flat
        B = build_none_separated_B(A_acc / n_reveals)
        B_list.append(B)
    sig = [sigma_from_B65(B) for B in B_list]
    return B_list, sig


# ── OOD Phase 0 runner ──────────────────────────────────────────────────────

def run_ood_phase0(ckpt_path, M=32, n_reveals=6, T=0.3, split=(0.7, 0.15, 0.15),
                   epochs=100, layer=0, heads=None, layers=None, h_context="sigma_B",
                   alt_perm_seed=None, device="cpu"):
    """Run P5 under a specific layout (π_train or π_alt).

    If alt_perm_seed is not None, remaps data to an alternate layout first.
    Returns the result dict (same format as run_phase0).
    """
    # Load model + π_train data
    model, chunks_train, cp_train, dev = load_p5_ckpt(ckpt_path, M, device=device)

    if alt_perm_seed is not None:
        cp_alt = create_random_alt_perm(alt_perm_seed, N)
        chunks = remap_chunks_to_alt_perm(chunks_train, cp_train, cp_alt)
        cp_use = cp_alt
    else:
        cp_alt = None
        chunks = chunks_train
        cp_use = cp_train

    heads_use = heads if heads is not None else [1]
    layers_use = layers if layers is not None else [layer]

    # Extract multi-head B65
    B_list, sig_B = extract_multihead_b65_alt(
        model, chunks, cp_use, layer, heads_use, n_reveals=n_reveals,
        fixed_reveal_seed=0, device=dev
    )

    # Build samples
    rng = np.random.default_rng(0)  # fixed cand_seed for reproducibility
    samples = []
    for t in range(M):
        B65 = B_list[t]
        sigma_B = sig_B[t]
        cands = candidate_orders(sigma_B, rng, n_random=4, n_noisy=4)
        labels = list(cands)
        nlls = utility_pool(model, chunks[t:t+1], [cands[l] for l in labels], cp_use, dev)
        nll_by_label = dict(zip(labels, nlls))
        P = soft_pref([cands[l] for l in labels], nlls, T)
        H = extract_h_by_context(model, chunks[t:t+1], sigma_B, cp_use, layers_use, dev,
                                 h_context)
        samples.append({
            "B_feat": block_b_features(B65), "H": H, "P": P,
            "sigma_B": sigma_B, "idx_row": chunks[t:t+1],
            "nll_by_label": nll_by_label, "cands": cands,
            "clean_perm": cp_use, "model": model, "dev": dev,
            "h_context": h_context,
        })

    # Split, headroom, train, evaluate
    tr, va, te = _split_idx(len(samples), split)
    test = [samples[i] for i in te] or [samples[i] for i in tr]
    head_in = headroom_stats([s["nll_by_label"] for s in test])
    result = {
        "ckpt": ckpt_path, "layer": layer, "heads": heads_use,
        "layers": layers_use, "h_context": h_context,
        "alt_perm_seed": alt_perm_seed,
        "headroom": head_in,
    }
    if head_in["gate_pass"]:
        train = [samples[i] for i in tr]
        g_B = _train_b_only_quick(train, epochs)
        for p in g_B.parameters():
            p.requires_grad_(False)
        h_dim = samples[0]["H"].shape[1]
        sc_real = _train_residual_quick(train, g_B, h_dim, epochs, h_mode="real")
        sc_shuf = _train_residual_quick(train, g_B, h_dim, epochs, h_mode="shuffle")
        sc_zero = _train_residual_quick(train, g_B, h_dim, epochs, h_mode="zero")
        sc_mean = _train_residual_quick(train, g_B, h_dim, epochs, h_mode="mean")
        result["metrics"] = p5_metrics(test, g_B, sc_real, sc_shuf, sc_zero, sc_mean)
        result["residual_ratio"] = sc_real.residual_ratio(
            torch.tensor(test[0]["B_feat"]), torch.tensor(test[0]["H"])
        )
    return result


def _train_b_only_quick(samples, epochs=100, lr=1e-2):
    g_B = BOnlyController(b_dim=samples[0]["B_feat"].shape[1])
    opt = torch.optim.Adam(g_B.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = sum(pairwise_loss(g_B(torch.tensor(s["B_feat"])), s["P"]) for s in samples)
        (loss / len(samples)).backward()
        opt.step()
    return g_B


def _train_residual_quick(samples, g_B, h_dim, epochs=100, lr=1e-2, h_mode="real"):
    for p in g_B.parameters():
        p.requires_grad_(False)
    sc = ScaffoldedController(g_B, h_dim=h_dim)
    Hs = _apply_h_mode(samples, h_mode)
    opt = torch.optim.Adam([p for p in sc.parameters() if p.requires_grad], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = sum(
            pairwise_loss(sc(torch.tensor(s["B_feat"]), torch.tensor(H)), s["P"])
            for s, H in zip(samples, Hs)
        )
        (loss / len(samples)).backward()
        opt.step()
    return sc


# ── Main sweep ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    CKPT = str(ROOT / "runs/handoff_overnight/seed123/ckpt_step10000.pt")
    HEADS = [1, 2, 3, 4]
    LAYERS = [2]
    H_CTX = "sigma_B"

    # π_train (alt_perm_seed=None) + 4 alt perms
    perm_configs = [
        ("π_train", None),
        ("π_alt_1", 100),
        ("π_alt_2", 101),
        ("π_alt_3", 102),
        ("π_alt_4", 103),
    ]

    print(f"P5 OOD-layout sweep: multi-head B={HEADS}, H layer={LAYERS}, H ctx={H_CTX}")
    print(f"{'Layout':<14s} {'headroom':>9s} {'B-only':>7s} {'B+H':>7s} {'ΔNLL':>7s} "
          f"{'shuf_drop':>10s} {'res_ratio':>9s} {'regret_B':>9s} {'regret_BH':>9s} {'verdict':>12s}")
    print("-" * 115)

    all_results = []
    for name, alt_seed in perm_configs:
        t0 = time.time()
        r = run_ood_phase0(
            CKPT, M=32, n_reveals=6, epochs=100,
            heads=HEADS, layers=LAYERS, h_context=H_CTX,
            alt_perm_seed=alt_seed,
        )
        head = r["headroom"]
        m = r.get("metrics", {})
        rr = r.get("residual_ratio", float("nan"))
        t1 = time.time()

        print(f"{name:<14s} {head['abs_mean']:9.4f} "
              f"{m.get('nll_b_only', 0):7.4f} {m.get('nll_bh', 0):7.4f} "
              f"{m.get('delta_nll', 0):+7.4f} {m.get('h_shuffle_drop', 0):+10.4f} "
              f"{rr:9.4f} {m.get('regret_bonly', 0):9.4f} {m.get('regret_bh', 0):9.4f} "
              f"{m.get('verdict', 'no_train'):>12s}  ({t1-t0:.0f}s)")
        all_results.append(r)

    out_dir = ROOT / "runs/p5/seed123"
    out_dir.mkdir(parents=True, exist_ok=True)
    _json.dump(all_results, open(out_dir / "ood_layout.json", "w"), indent=2, default=float)
    print(f"\nSaved to {out_dir / 'ood_layout.json'}")
