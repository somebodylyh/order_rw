"""P5: attention-scaffolded utility controller (Phase 0).

H improves downstream reveal-order utility beyond the attention scaffold; it does
NOT recover physical order. Supervision = downstream teacher-forced AO NLL (never
physical-rank/CDL(B), which structurally ignore H under fixed layout).
See docs/superpowers/specs/2026-06-29-p5-attention-scaffolded-utility-controller-design.md.
"""
import json as _json
import pathlib, sys
import numpy as np
import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
_BLOCK = ROOT / "block_lo_arm_order_network"
_ANALYSES = ROOT / "analyses"
for _p in (str(ROOT), str(_BLOCK), str(_ANALYSES)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from neural_readout.extract_b import _load_model_and_chunks  # noqa: E402
from clean_training_protocol import physical_blocks_to_model_token_order  # noqa: E402
from physical_signal_source import carrier_b65_per_text  # noqa: E402
from none_separated_block_graph import rollout_by_method  # noqa: E402
from path_patch_handoff import capture_block_input  # noqa: E402

SEQ_LEN, N, BLOCK_LEN = 256, 64, 4


def load_p5_ckpt(ckpt_path, M, device="cpu"):
    model, chunks, clean_perm, dev, _ = _load_model_and_chunks(
        ckpt_path, M, seed=0, device=device, split="train")
    return model, chunks, clean_perm, dev


@torch.no_grad()
def order_nll(model, idx_row, sigma_phys, clean_perm, device):
    """Downstream teacher-forced AO NLL of revealing blocks in physical-block order
    sigma_phys (64,). Lower = better utility."""
    sigma = np.asarray(sigma_phys, dtype=np.int64)[None, :]            # (1,64)
    token_order = physical_blocks_to_model_token_order(
        torch.from_numpy(sigma), clean_perm, BLOCK_LEN).to(device)
    _, loss = model.forward_fn(idx_row.to(device), token_order)
    return float(loss)


def utility_pool(model, idx_row, sigmas, clean_perm, device):
    return [order_nll(model, idx_row, s, clean_perm, device) for s in sigmas]


# ── Task 2: Per-sample scaffold ──────────────────────────────────────────────

def block_b_features(B65):
    """Per non-None block features: its row (out-edges) and column (in-edges) of B65."""
    B = np.asarray(B65, dtype=np.float64)
    rows = B[1:, :]                                  # (64, 65) outgoing
    cols = B[:, 1:].T                                # (64, 65) incoming
    return np.concatenate([rows, cols], axis=1).astype(np.float32)   # (64, 130)


def sigma_from_B65(B65):
    """C-D+L rollout order of a B65 -> physical-block order (64,).

    rollout_by_method returns the order ndarray DIRECTLY (P2 wraps it as
    discovery_metrics(rollout_by_method(...))); it is NOT a dict."""
    order = rollout_by_method(np.asarray(B65), "C-D+L")
    return np.asarray(order, dtype=np.int64)


def sample_scaffold(ckpt_path, M, layer=0, head=1, n_reveals=8, fixed_reveal_seed=0,
                    device="cpu", heads=None):
    """Per-text B65 (seed123 carrier L0H1), its C-D+L order sigma_B, and the chunks.

    If heads is a list of ints, B65 is averaged element-wise across those heads
    before computing sigma_B and block_b_features."""
    model, chunks, clean_perm, dev = load_p5_ckpt(ckpt_path, M, device=device)
    if heads is not None and len(heads) > 0:
        # Multi-head: collect B65 per head, mean across heads per text
        B_per_head = []
        for h in heads:
            B_h, _ = carrier_b65_per_text(
                ckpt_path, layer, h, M=M, n_reveals=n_reveals,
                fixed_reveal_seed=fixed_reveal_seed, device=device)
            B_per_head.append([np.asarray(b, dtype=np.float64) for b in B_h])
        B_list = []
        for t in range(M):
            mean_B = np.mean([B_per_head[hi][t] for hi in range(len(heads))], axis=0)
            B_list.append(mean_B)
    else:
        B_list, _tau = carrier_b65_per_text(
            ckpt_path, layer, head, M=M, n_reveals=n_reveals,
            fixed_reveal_seed=fixed_reveal_seed, device=device)
    sig = [sigma_from_B65(B) for B in B_list]
    return {"B": B_list, "sigma_B": sig, "chunks": chunks,
            "clean_perm": clean_perm, "model": model, "dev": dev}


# ── Task 3: Candidate-order pool ─────────────────────────────────────────────

def _swap_perturb(sigma, rng, n_swaps):
    s = np.asarray(sigma, dtype=np.int64).copy()
    for _ in range(n_swaps):
        i, j = rng.integers(0, len(s), size=2)
        s[i], s[j] = s[j], s[i]
    return s


def candidate_orders(sigma_B, rng, n_random=4, n_noisy=4, noisy_swaps=3):
    sigma_B = np.asarray(sigma_B, dtype=np.int64)
    phys = np.arange(64, dtype=np.int64)
    pool = {"sigma_B": sigma_B.copy(),
            "phys": phys.copy(),
            "reverse_phys": phys[::-1].copy(),
            "local": phys.copy()}                       # local = identity adjacency baseline
    for k in range(n_random):
        pool[f"random_{k}"] = rng.permutation(64).astype(np.int64)
    for k in range(n_noisy):
        pool[f"noisy_B_{k}"] = _swap_perturb(sigma_B, rng, noisy_swaps)
    return pool


# ── Task 4: Utility-headroom gate ────────────────────────────────────────────

def headroom_stats(nll_by_label, sigma_b_label="sigma_B", n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    abs_h, rel_h, best_labels = [], [], []
    for d in nll_by_label:
        nb = d[sigma_b_label]
        best_label = min(d, key=lambda k: d[k])
        best = d[best_label]
        abs_h.append(nb - best)                       # >=0 by construction
        rel_h.append((nb - best) / (abs(nb) + 1e-9))
        best_labels.append(best_label)
    abs_h = np.asarray(abs_h)
    boot = np.array([rng.choice(abs_h, size=len(abs_h), replace=True).mean()
                     for _ in range(n_boot)])
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    labels = sorted({l for d in nll_by_label for l in d})
    best_dist = {l: float(np.mean([bl == l for bl in best_labels])) for l in labels}
    return {"abs_mean": float(abs_h.mean()), "abs_ci_low": lo, "abs_ci_high": hi,
            "rel_mean": float(np.mean(rel_h)), "gate_pass": bool(lo > 0.0),
            "best_dist": best_dist}


# ── Task 5: Block-level hidden-state extraction ──────────────────────────────

@torch.no_grad()
def block_hidden_states(model, idx_row, sigma_phys, clean_perm, layer, device):
    """Per model-block mean-pooled residual entering `layer`, under reveal sigma_phys."""
    sigma = np.asarray(sigma_phys, dtype=np.int64)[None, :]
    token_order = physical_blocks_to_model_token_order(
        torch.from_numpy(sigma), clean_perm, BLOCK_LEN).to(device)   # (1,256) model-token order
    handle, store = capture_block_input(model, layer)
    try:
        model.forward_fn(idx_row.to(device), token_order)
    finally:
        handle.remove()
    x = store["x"][0].cpu().numpy()                     # (257, d) incl leading [None]
    revealed = x[1:]                                    # (256, d) reveal-ordered positions
    tok = token_order[0].cpu().numpy()                  # model-token index at each position
    block_of_pos = tok // BLOCK_LEN                      # (256,) model block per position
    d_val = revealed.shape[1]
    H = np.zeros((N, d_val), dtype=np.float64)
    for m in range(N):
        sel = revealed[block_of_pos == m]
        H[m] = sel.mean(axis=0) if len(sel) else 0.0
    return H.astype(np.float32)


@torch.no_grad()
def multi_layer_hidden_states(model, idx_row, sigma_phys, clean_perm, layers, device):
    """Per model-block hidden states, mean-pooled across specified layers.

    layers: list of ints, e.g. [0,1,2,3] for L0–L3 mean."""
    Hs = [block_hidden_states(model, idx_row, sigma_phys, clean_perm, l, device)
          for l in layers]
    return np.mean(Hs, axis=0).astype(np.float32)   # (64, d)


@torch.no_grad()
def order_averaged_hidden_states(model, idx_row, sigmas, clean_perm, layers, device):
    """Per-block H extracted under each sigma, then averaged across sigma.

    Averages out order-specific reveal-context traces, retaining content component."""
    all_H = []
    for sigma in sigmas:
        all_H.append(multi_layer_hidden_states(
            model, idx_row, sigma, clean_perm, layers, device))
    return np.mean(all_H, axis=0).astype(np.float32)   # (64, d)


def extract_h_by_context(model, idx_row, sigma_B, clean_perm, layers, device, context):
    """Extract H under a named context.

    context: "sigma_B" | "phys" | "avg" (mean of sigma_B + phys + random)"""
    if context == "sigma_B":
        return multi_layer_hidden_states(model, idx_row, sigma_B, clean_perm, layers, device)
    elif context == "phys":
        phys = np.arange(N, dtype=np.int64)
        return multi_layer_hidden_states(model, idx_row, phys, clean_perm, layers, device)
    elif context == "avg":
        phys = np.arange(N, dtype=np.int64)
        rng = np.random.default_rng(0)
        rand = rng.permutation(N).astype(np.int64)
        sigmas = [sigma_B, phys, rand]
        return order_averaged_hidden_states(model, idx_row, sigmas, clean_perm, layers, device)
    else:
        raise ValueError(f"Unknown H context: {context!r}. Use sigma_B, phys, or avg.")


# ── Task 6: Soft pairwise utility teacher ────────────────────────────────────

def _order_to_rank(sigma):
    """sigma is a reveal order (positions->block). Return rank[block] = reveal position."""
    rank = np.empty(N, dtype=np.int64)
    rank[np.asarray(sigma, dtype=np.int64)] = np.arange(N)
    return rank


def soft_pref(sigmas, nlls, T):
    nlls = np.asarray(nlls, dtype=np.float64)
    w = np.exp(-(nlls - nlls.min()) / max(T, 1e-9))
    w = w / w.sum()
    P = np.zeros((N, N), dtype=np.float64)
    for wk, sig in zip(w, sigmas):
        rank = _order_to_rank(sig)
        before = (rank[:, None] < rank[None, :]).astype(np.float64)   # i before j
        P += wk * before
    return P


def pairwise_loss(z, P):
    P = torch.as_tensor(P, dtype=torch.float32, device=z.device)
    diff = z[:, None] - z[None, :]                      # z_i - z_j
    p_hat = torch.sigmoid(diff)
    c = (P - 0.5).abs()
    eps = 1e-6
    bce = -(P * torch.log(p_hat + eps) + (1 - P) * torch.log(1 - p_hat + eps))
    mask = ~torch.eye(N, dtype=torch.bool, device=z.device)
    return (c * bce)[mask].mean()


# ── Task 7: Controllers ─────────────────────────────────────────────────────

class BOnlyController(nn.Module):
    def __init__(self, b_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(b_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, B_feat):
        return self.net(B_feat).squeeze(-1)            # (64,)


class ScaffoldedController(nn.Module):
    def __init__(self, g_B, h_dim, hidden=64, alpha_init=0.01):
        super().__init__()
        self.g_B = g_B
        for p in self.g_B.parameters():
            p.requires_grad_(False)
        self.g_H = nn.Sequential(nn.Linear(h_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
        # softplus(a)=alpha_init  ->  a = log(exp(alpha)-1)
        self.a = nn.Parameter(torch.tensor(float(np.log(np.expm1(alpha_init)))))

    @property
    def alpha(self):
        return torch.nn.functional.softplus(self.a)

    def _delta(self, H):
        return self.g_H(H).squeeze(-1)

    def forward(self, B_feat, H):
        return self.g_B(B_feat) + self.alpha * self._delta(H)

    @torch.no_grad()
    def residual_ratio(self, B_feat, H):
        zb = self.g_B(B_feat)
        dh = self.alpha * self._delta(H)
        return float(dh.norm() / (zb.norm() + 1e-9))


# ── Task 8: Dataset assembly + train loops ───────────────────────────────────

def build_dataset(ckpt_path, M, layer=0, head=1, n_reveals=8, K_rand=4, K_noisy=4,
                  T=0.3, cand_seed=0, device="cpu", layers=None, heads=None,
                  h_context="sigma_B"):
    sc = sample_scaffold(ckpt_path, M, layer=layer, head=head, n_reveals=n_reveals,
                         device=device, heads=heads)
    model, chunks, clean_perm, dev = sc["model"], sc["chunks"], sc["clean_perm"], sc["dev"]
    if layers is None:
        layers = [layer]
    rng = np.random.default_rng(cand_seed)
    samples = []
    for t in range(M):
        B65 = sc["B"][t]; sigma_B = sc["sigma_B"][t]
        cands = candidate_orders(sigma_B, rng, n_random=K_rand, n_noisy=K_noisy)
        labels = list(cands)
        nlls = utility_pool(model, chunks[t:t+1], [cands[l] for l in labels], clean_perm, dev)
        nll_by_label = dict(zip(labels, nlls))
        P = soft_pref([cands[l] for l in labels], nlls, T)
        H = extract_h_by_context(model, chunks[t:t+1], sigma_B, clean_perm, layers, dev,
                                 h_context)
        samples.append({"B_feat": block_b_features(B65), "H": H, "P": P,
                        "sigma_B": sigma_B, "idx_row": chunks[t:t+1],
                        "nll_by_label": nll_by_label, "cands": cands,
                        "clean_perm": clean_perm, "model": model, "dev": dev,
                        "h_context": h_context})
    return samples


def _apply_h_mode(samples, h_mode):
    Hs = [s["H"] for s in samples]
    if h_mode == "real":
        return Hs
    if h_mode == "shuffle":
        return Hs[1:] + Hs[:1]                          # text-level mismatch (roll by 1)
    if h_mode == "zero":
        return [np.zeros_like(h) for h in Hs]
    if h_mode == "mean":
        m = np.mean(Hs, axis=0)
        return [m.copy() for _ in Hs]
    raise ValueError(h_mode)


def train_b_only(samples, epochs=200, lr=1e-2):
    g_B = BOnlyController(b_dim=samples[0]["B_feat"].shape[1])
    opt = torch.optim.Adam(g_B.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad(); loss = 0.0
        for s in samples:
            z = g_B(torch.tensor(s["B_feat"]))
            loss = loss + pairwise_loss(z, s["P"])
        (loss / len(samples)).backward(); opt.step()
    return g_B


def train_residual(samples, g_B, h_dim, epochs=200, lr=1e-2, h_mode="real"):
    for p in g_B.parameters():
        p.requires_grad_(False)
    sc = ScaffoldedController(g_B, h_dim=h_dim)
    Hs = _apply_h_mode(samples, h_mode)
    opt = torch.optim.Adam([p for p in sc.parameters() if p.requires_grad], lr=lr)
    for _ in range(epochs):
        opt.zero_grad(); loss = 0.0
        for s, H in zip(samples, Hs):
            z = sc(torch.tensor(s["B_feat"]), torch.tensor(H))
            loss = loss + pairwise_loss(z, s["P"])
        (loss / len(samples)).backward(); opt.step()
    return sc


# ── Task 9: Hard-order forward-NLL metrics + verdict ─────────────────────────

@torch.no_grad()
def predict_order(controller, B_feat, H=None):
    B = torch.tensor(B_feat)
    z = controller(B) if H is None else controller(B, torch.tensor(H))
    return np.argsort(-z.cpu().numpy()).astype(np.int64)


@torch.no_grad()
def eval_controller_nll(samples, controller, h_list=None):
    out = []
    for i, s in enumerate(samples):
        H = None if h_list is None else h_list[i]
        sigma = predict_order(controller, s["B_feat"], H)
        out.append(order_nll(s["model"], s["idx_row"], sigma, s["clean_perm"], s["dev"]))
    return out


def _regret(samples, nll_pred):
    return float(np.mean([nll_pred[i] - min(s["nll_by_label"].values())
                          for i, s in enumerate(samples)]))


def p5_metrics(samples, g_B, sc_real, sc_shuf, sc_zero, sc_mean):
    n_b = eval_controller_nll(samples, g_B)
    n_bh = eval_controller_nll(samples, sc_real, _apply_h_mode(samples, "real"))
    n_sh = eval_controller_nll(samples, sc_shuf, _apply_h_mode(samples, "shuffle"))
    n_zero = eval_controller_nll(samples, sc_zero, _apply_h_mode(samples, "zero"))
    n_mean = eval_controller_nll(samples, sc_mean, _apply_h_mode(samples, "mean"))
    mb, mbh = float(np.mean(n_b)), float(np.mean(n_bh))
    m = {"nll_b_only": mb, "nll_bh": mbh,
         "delta_nll": mbh - mb,
         "regret_bonly": _regret(samples, n_b), "regret_bh": _regret(samples, n_bh),
         "h_shuffle_drop": float(np.mean(n_sh)) - mbh,
         "zero_match": bool(abs(float(np.mean(n_zero)) - mbh) < 0.005),
         "mean_match": bool(abs(float(np.mean(n_mean)) - mbh) < 0.005)}
    m["verdict"] = classify_p5(m)
    return m


def classify_p5(m):
    gain = m["delta_nll"] < 0
    content = m["h_shuffle_drop"] > 0.5 * abs(m["delta_nll"])     # shuffle loses most gain
    not_capacity = (not m["zero_match"]) and (not m["mean_match"])
    return "utility_gain" if (gain and content and not_capacity) else "no_gain"


# ── Task 10: Phase-0 driver ─────────────────────────────────────────────────

def _split_idx(n, split, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_tr = int(split[0] * n); n_va = int(split[1] * n)
    return perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]


def run_phase0(ckpt_path, M=64, n_reveals=8, T=0.3, split=(0.7, 0.15, 0.15),
               epochs=200, layer=0, head=1, out_dir="runs/p5/seed123", device="cpu",
               layers=None, heads=None, h_context="sigma_B"):
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    samples = build_dataset(ckpt_path, M, layer=layer, head=head, n_reveals=n_reveals,
                            T=T, device=device, layers=layers, heads=heads,
                            h_context=h_context)
    tr, va, te = _split_idx(len(samples), split)
    test = [samples[i] for i in te] or [samples[i] for i in tr]
    head_in = headroom_stats([s["nll_by_label"] for s in test])
    result = {"ckpt": ckpt_path, "layer": layer, "head": head,
              "layers": layers, "heads": heads, "h_context": h_context,
              "headroom": head_in}
    if head_in["gate_pass"]:
        train = [samples[i] for i in tr]
        g_B = train_b_only(train, epochs=epochs)
        for p in g_B.parameters():
            p.requires_grad_(False)
        h_dim = samples[0]["H"].shape[1]
        sc_real = train_residual(train, g_B, h_dim, epochs=epochs, h_mode="real")
        sc_shuf = train_residual(train, g_B, h_dim, epochs=epochs, h_mode="shuffle")
        sc_zero = train_residual(train, g_B, h_dim, epochs=epochs, h_mode="zero")
        sc_mean = train_residual(train, g_B, h_dim, epochs=epochs, h_mode="mean")
        result["metrics"] = p5_metrics(test, g_B, sc_real, sc_shuf, sc_zero, sc_mean)
        result["residual_ratio"] = sc_real.residual_ratio(
            torch.tensor(test[0]["B_feat"]), torch.tensor(test[0]["H"]))
    _json.dump(result, open(out / "phase0.json", "w"), indent=2, default=float)
    return result


# ── Version A: direct-NLL soft-routing ───────────────────────────────────────
# Replace the pairwise-teacher imitation objective with NLL-weighted candidate
# routing on the SAME frozen scaffold. L_k are pre-computed constants; the frozen
# model is NOT in the gradient path. Spec:
#   docs/superpowers/specs/2026-06-30-p5-direct-nll-soft-routing-design.md

def candidate_priority(sigma, n=N):
    """Reveal-priority vector y for a candidate order: y[i] = 1 - rank_sigma(i)/(n-1).
    Block revealed first -> priority 1.0; revealed last -> 0.0."""
    sigma = np.asarray(sigma, dtype=np.int64)
    rank = np.empty(len(sigma), dtype=np.int64)
    rank[sigma] = np.arange(len(sigma))
    return (1.0 - rank / (len(sigma) - 1)).astype(np.float32)


def priority_matrix(cands):
    """cands: {label -> sigma}. Returns (labels, Y of shape (K, N)).

    Byte-identical candidate orders are deduplicated (keeping the first by sorted
    label). Without this, e.g. candidate_orders' 'phys'/'local' (both = identity)
    produce two equal logits and softmax floors at entropy ln(2)/max_p 0.5, pinning
    the argmax pool selection to a degenerate tie."""
    seen = {}
    labels = []
    for lab in sorted(cands):
        key = tuple(np.asarray(cands[lab], dtype=np.int64).tolist())
        if key in seen:
            continue
        seen[key] = lab
        labels.append(lab)
    Y = np.stack([candidate_priority(cands[l]) for l in labels], axis=0)
    return labels, Y.astype(np.float32)


def _routing_tensors(sample):
    """Per-sample (Y, L, labels) with a stable shared label ordering."""
    labels, Y = priority_matrix(sample["cands"])
    L = np.array([sample["nll_by_label"][l] for l in labels], dtype=np.float32)
    return torch.tensor(Y), torch.tensor(L), labels


def direct_nll_routing_loss(z, Y, L, tau):
    """L = sum_k softmax((z . y_k)/tau)_k * L_k. Returns (loss, p)."""
    a = Y @ z                                           # (K,)
    p = torch.softmax(a / max(tau, 1e-9), dim=0)
    return (p * L).sum(), p


def train_b_only_routing(samples, tau=0.3, epochs=300, lr=5e-2, return_history=False):
    g_B = BOnlyController(b_dim=samples[0]["B_feat"].shape[1])
    YL = [_routing_tensors(s) for s in samples]
    opt = torch.optim.Adam(g_B.parameters(), lr=lr)
    hist = []
    for _ in range(epochs):
        opt.zero_grad(); loss = 0.0
        for s, (Y, L, _) in zip(samples, YL):
            z = g_B(torch.tensor(s["B_feat"]))
            l, _p = direct_nll_routing_loss(z, Y, L, tau)
            loss = loss + l
        loss = loss / len(samples)
        loss.backward(); opt.step()
        hist.append(float(loss))
    return (g_B, hist) if return_history else g_B


def train_residual_routing(samples, g_B, h_dim, tau=0.3, epochs=300, lr=5e-2,
                           h_mode="real", return_history=False):
    for p in g_B.parameters():
        p.requires_grad_(False)
    sc = ScaffoldedController(g_B, h_dim=h_dim)
    Hs = _apply_h_mode(samples, h_mode)
    YL = [_routing_tensors(s) for s in samples]
    opt = torch.optim.Adam([p for p in sc.parameters() if p.requires_grad], lr=lr)
    hist = []
    for _ in range(epochs):
        opt.zero_grad(); loss = 0.0
        for s, H, (Y, L, _) in zip(samples, Hs, YL):
            z = sc(torch.tensor(s["B_feat"]), torch.tensor(H))
            l, _p = direct_nll_routing_loss(z, Y, L, tau)
            loss = loss + l
        loss = loss / len(samples)
        loss.backward(); opt.step()
        hist.append(float(loss))
    return (sc, hist) if return_history else sc


@torch.no_grad()
def routing_eval(samples, controller, h_mode=None, tau=0.3, with_model=True):
    """Two eval modes (spec): pool-selected NLL (primary, argmax_k p_k) and
    free-argsort NLL (diagnostic, argsort(-z), needs the model)."""
    Hs = _apply_h_mode(samples, h_mode) if h_mode is not None else None
    pool_nll, regret_pool, free_nll, regret_free = [], [], [], []
    ents, maxps = [], []
    sel = {}
    for i, s in enumerate(samples):
        B = torch.tensor(s["B_feat"])
        z = controller(B) if Hs is None else controller(B, torch.tensor(Hs[i]))
        labels, Y = priority_matrix(s["cands"])
        L = np.array([s["nll_by_label"][l] for l in labels], dtype=np.float64)
        a = torch.tensor(Y) @ z
        p = torch.softmax(a / max(tau, 1e-9), dim=0).cpu().numpy()
        kstar = int(p.argmax()); Lmin = float(L.min())
        pool_nll.append(float(L[kstar])); regret_pool.append(float(L[kstar]) - Lmin)
        sel[labels[kstar]] = sel.get(labels[kstar], 0) + 1
        ents.append(float(-(p * np.log(p + 1e-12)).sum())); maxps.append(float(p.max()))
        if with_model:
            sigma_free = np.argsort(-z.cpu().numpy()).astype(np.int64)
            nf = order_nll(s["model"], s["idx_row"], sigma_free, s["clean_perm"], s["dev"])
            free_nll.append(nf); regret_free.append(nf - Lmin)
    out = {"nll_pool": float(np.mean(pool_nll)), "regret_pool": float(np.mean(regret_pool)),
           "entropy": float(np.mean(ents)), "max_p": float(np.mean(maxps)),
           "best_candidate_frac": {k: v / len(samples) for k, v in sel.items()}}
    if with_model:
        out["nll_free"] = float(np.mean(free_nll))
        out["regret_free"] = float(np.mean(regret_free))
    return out


def run_routing_A(ckpt_path, M=64, out_dir="runs/p5/seed123/routing_A",
                  taus=(0.03, 0.1, 0.3, 1.0), epochs=300, lr=5e-2,
                  layer=0, head=1, heads=None, n_reveals=8,
                  split=(0.7, 0.15, 0.15), device="cpu", tag="main_L0H1"):
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    samples = build_dataset(ckpt_path, M, layer=layer, head=head,
                            n_reveals=n_reveals, device=device, heads=heads)
    tr, va, te = _split_idx(len(samples), split)
    train = [samples[i] for i in tr]
    test = [samples[i] for i in (list(te) or list(tr))]
    h_dim = samples[0]["H"].shape[1]
    head_in = headroom_stats([s["nll_by_label"] for s in test])

    # pairwise baseline (tau-independent training; evaluated per tau)
    pairwise_gB = train_b_only(train, epochs=epochs)
    for p in pairwise_gB.parameters():
        p.requires_grad_(False)

    per_tau = {}
    for tau in taus:
        d_gB, hist = train_b_only_routing(train, tau=tau, epochs=epochs, lr=lr,
                                          return_history=True)
        arms = {
            "pairwise_b_only": routing_eval(test, pairwise_gB, None, tau),
            "direct_b_only": routing_eval(test, d_gB, None, tau),
        }
        for mode in ("real", "shuffle", "zero", "mean"):
            sc = train_residual_routing(train, d_gB, h_dim, tau=tau,
                                        epochs=epochs, lr=lr, h_mode=mode)
            arms[f"direct_bh_{mode}"] = routing_eval(test, sc, mode, tau)
        per_tau[f"tau_{tau}"] = {"loss_hist_first_last": [hist[0], hist[-1]],
                                 "arms": arms}

    result = {"ckpt": ckpt_path, "tag": tag, "layer": layer, "head": head,
              "heads": heads, "M": M, "headroom": head_in, "per_tau": per_tau}
    _json.dump(result, open(out / f"{tag}.json", "w"), indent=2, default=float)
    return result
