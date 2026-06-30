"""P6 / Version B: online co-adaptive H-residual controller.

Wire the candidate-routing controller into AO-GPT training. No argsort(z) — uses
candidate soft-routing (Version A: direct-NLL learns a router, not a ranker) with
COSINE logits (Version A: unnormalized logits saturate softmax). This module is
the infrastructure (Task 1-5) + the B0 frozen-model code-path sanity.

RED LINE: B0 has no co-adaptation -> H is still frozen-posthoc -> a B0 null is
expected by construction and is NOT a scientific result. See
docs/superpowers/specs/2026-06-30-p6-online-coadaptive-controller-design.md.
"""
import json as _json
import pathlib, sys
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "analyses")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analyses.p5_utility_controller import (   # noqa: E402
    N, BLOCK_LEN, candidate_priority, priority_matrix, sigma_from_B65, order_nll,
    block_b_features, BOnlyController, ScaffoldedController, _swap_perturb,
    sample_scaffold, extract_h_by_context, utility_pool, headroom_stats,
    block_hidden_states, _split_idx, load_p5_ckpt,
)
from clean_training_protocol import physical_blocks_to_model_token_order  # noqa: E402
from analyses.canonical_reanalysis import random_reveal_orders  # noqa: E402
from per_head_order_scan import _attn_to_A_block_loss_aligned_with_none_vec  # noqa: E402
from none_separated_block_graph import build_none_separated_B  # noqa: E402


# ── Task 2: normalized (cosine) candidate logits ─────────────────────────────

def cosine_logits(z, Y, tau=0.3, eps=1e-8):
    """a_k = (z/||z||) . (y_k/||y_k||) / tau.  Scale-invariant in ||z|| so the
    softmax cannot saturate as z grows (the Version-A failure mode)."""
    zn = z / (z.norm() + eps)
    Yn = Y / (Y.norm(dim=1, keepdim=True) + eps)
    return (Yn @ zn) / max(tau, 1e-9)


def routing_p(z, Y, tau=0.3):
    return torch.softmax(cosine_logits(z, Y, tau), dim=0)


# ── Task 3: online routing loss ──────────────────────────────────────────────

def routing_loss_from_scores(z, Y, L, tau=0.3, beta_entropy=0.0):
    """L_route = sum_k p_k * L_k (- beta * H(p)).  L is a plain (constant) tensor:
    the controller gets gradient ONLY through p_k, never through the order/L_k."""
    a = cosine_logits(z, Y, tau)
    p = torch.softmax(a, dim=0)
    ent = -(p * (p + 1e-12).log()).sum()
    loss = (p * L).sum()
    if beta_entropy:
        loss = loss - beta_entropy * ent
    info = {"p": p, "entropy": float(ent), "max_p": float(p.max()),
            "selected": int(p.argmax())}
    return loss, info


# ── Task 4 + 5: controller scoring with detach_h config ──────────────────────

def controller_scores(controller, B_feat, H=None, detach_h=True):
    """z = g_B(B) [+ alpha g_H(H)].  detach_h=True stops controller-loss gradient
    from flowing back into the model's hidden states H (the safe B2a default)."""
    B = torch.as_tensor(B_feat, dtype=torch.float32)
    if H is None:
        return controller(B)
    Ht = torch.as_tensor(H, dtype=torch.float32)
    if detach_h:
        Ht = Ht.detach()
    return controller(B, Ht)


def h_mode_list(H_list, mode, seed=0):
    """real / zero / mean / shuffle (within-batch text roll, j != i)."""
    if mode in ("real", "b_only"):
        return H_list
    if mode == "zero":
        return [np.zeros_like(h) for h in H_list]
    if mode == "mean":
        m = np.mean(H_list, axis=0)
        return [m.copy() for _ in H_list]
    if mode == "shuffle":
        idx = np.roll(np.arange(len(H_list)), 1)
        return [H_list[j] for j in idx]
    raise ValueError(mode)


# ── Task 1: P6 candidate pool (fixed protocol, deduped downstream) ────────────

def _b_colsum_order(B65):
    """A direct multi-head attention readout distinct from the C-D+L rollout:
    order blocks by total incoming attention (descending)."""
    B = np.asarray(B65, dtype=np.float64)
    incoming = B[1:, 1:].sum(axis=0)                # (64,) per target block
    return np.argsort(-incoming).astype(np.int64)


def p6_candidate_pool(B65, sigma_B, rng, K=6):
    """Fixed protocol pool. Byte-identical orders are collapsed later by
    priority_matrix (Version A dedup)."""
    phys = np.arange(N, dtype=np.int64)
    pool = {"sigma_cdl_b": np.asarray(sigma_B, dtype=np.int64),
            "sigma_b_multihead": _b_colsum_order(B65),
            "phys": phys.copy(),
            "reverse_phys": phys[::-1].copy()}
    n_extra = max(K - 4, 0)
    n_rand = (n_extra + 1) // 2
    n_noisy = n_extra // 2
    for k in range(n_rand):
        pool[f"random_{k+1}"] = rng.permutation(N).astype(np.int64)
    for k in range(n_noisy):
        pool[f"noisy_B_{k+1}"] = _swap_perturb(sigma_B, rng, 3)
    return pool


# ── dataset + Y/L tensors ────────────────────────────────────────────────────

def build_dataset_p6(ckpt_path, M, layer=0, head=1, heads=None, n_reveals=8,
                     K=6, h_layer=1, h_context="sigma_B", cand_seed=0, device="cpu"):
    """Per-text scaffold using the P6 candidate pool + L1 hidden states."""
    sc = sample_scaffold(ckpt_path, M, layer=layer, head=head, n_reveals=n_reveals,
                         device=device, heads=heads)
    model, chunks, clean_perm, dev = sc["model"], sc["chunks"], sc["clean_perm"], sc["dev"]
    rng = np.random.default_rng(cand_seed)
    samples = []
    for t in range(M):
        B65 = sc["B"][t]; sigma_B = sc["sigma_B"][t]
        cands = p6_candidate_pool(B65, sigma_B, rng, K=K)
        labels = list(cands)
        nlls = utility_pool(model, chunks[t:t+1], [cands[l] for l in labels], clean_perm, dev)
        H = extract_h_by_context(model, chunks[t:t+1], sigma_B, clean_perm, [h_layer],
                                 dev, h_context)
        samples.append({"B_feat": block_b_features(B65), "H": H, "cands": cands,
                        "nll_by_label": dict(zip(labels, nlls)), "sigma_B": sigma_B,
                        "idx_row": chunks[t:t+1], "clean_perm": clean_perm,
                        "model": model, "dev": dev})
    return samples


def _yl(samples):
    """Per-sample (Y, L, labels) tensors with the deduped shared label ordering."""
    Ys, Ls, labs = [], [], []
    for s in samples:
        labels, Y = priority_matrix(s["cands"])
        L = np.array([s["nll_by_label"][l] for l in labels], dtype=np.float32)
        Ys.append(torch.tensor(Y)); Ls.append(torch.tensor(L)); labs.append(labels)
    return Ys, Ls, labs


# ── Task 6 (frozen variant): controller-only online training ─────────────────

def train_controller_online(samples, Y_list, L_list, mode="b_only", g_B=None,
                            tau=0.3, epochs=300, lr=5e-2, beta_entropy=0.0,
                            detach_h=True, h_dim=None, return_history=False):
    """B0: model frozen, L_k precomputed (constant). Trains a B-only controller
    (mode='b_only') or a residual g_H on a frozen g_B (mode in real/shuffle/zero/mean)."""
    if mode == "b_only":
        controller = BOnlyController(b_dim=samples[0]["B_feat"].shape[1])
        params = controller.parameters()
        Hs = None
    else:
        assert g_B is not None, "residual modes need a pre-trained g_B"
        for p in g_B.parameters():
            p.requires_grad_(False)
        h_dim = h_dim or samples[0]["H"].shape[1]
        controller = ScaffoldedController(g_B, h_dim=h_dim)
        params = [p for p in controller.parameters() if p.requires_grad]
        Hs = h_mode_list([s["H"] for s in samples], mode)
    opt = torch.optim.Adam(params, lr=lr)
    hist = []
    for _ in range(epochs):
        opt.zero_grad(); loss = 0.0
        for i, s in enumerate(samples):
            H = None if Hs is None else Hs[i]
            z = controller_scores(controller, s["B_feat"], H, detach_h=detach_h)
            l, _ = routing_loss_from_scores(z, Y_list[i], L_list[i], tau, beta_entropy)
            loss = loss + l
        loss = loss / len(samples)
        loss.backward(); opt.step(); hist.append(float(loss))
    return (controller, hist) if return_history else controller


# ── Task 7: three separately-named eval functions (risk #3) ──────────────────

@torch.no_grad()
def eval_soft_expected_nll(samples, controller, Y_list, L_list, tau=0.3,
                           h_list=None, detach_h=True):
    vals = []
    for i, s in enumerate(samples):
        H = None if h_list is None else h_list[i]
        z = controller_scores(controller, s["B_feat"], H, detach_h=detach_h)
        p = routing_p(z, Y_list[i], tau)
        vals.append(float((p * L_list[i]).sum()))
    return float(np.mean(vals))


@torch.no_grad()
def eval_hard_selected_nll(samples, controller, Y_list, L_list, labels_list,
                           tau=0.3, h_list=None, detach_h=True):
    vals, ents, maxps = [], [], []
    sel = {}
    for i, s in enumerate(samples):
        H = None if h_list is None else h_list[i]
        z = controller_scores(controller, s["B_feat"], H, detach_h=detach_h)
        p = routing_p(z, Y_list[i], tau).cpu().numpy()
        k = int(p.argmax())
        vals.append(float(L_list[i][k]))
        sel[labels_list[i][k]] = sel.get(labels_list[i][k], 0) + 1
        ents.append(float(-(p * np.log(p + 1e-12)).sum())); maxps.append(float(p.max()))
    return {"nll": float(np.mean(vals)), "entropy": float(np.mean(ents)),
            "max_p": float(np.mean(maxps)),
            "selection_frac": {k: v / len(samples) for k, v in sel.items()}}


@torch.no_grad()
def eval_fixed_order_nll(samples, sigma_fixed=None):
    """Model-quality val NLL under ONE fixed canonical order (default physical).
    Constant across controllers when the model is frozen (B0); the primary
    cross-arm comparison in B2."""
    if sigma_fixed is None:
        sigma_fixed = np.arange(N, dtype=np.int64)
    vals = [order_nll(s["model"], s["idx_row"], sigma_fixed, s["clean_perm"], s["dev"])
            for s in samples]
    return float(np.mean(vals))


# ── B0 frozen sanity runner ──────────────────────────────────────────────────

def run_b0_sanity(ckpt_path, M=64, layer=0, head=1, heads=None, K=6, h_layer=1,
                  tau=0.3, epochs=300, lr=5e-2, beta_entropy=0.0, detach_h=True,
                  split=(0.7, 0.15, 0.15), n_reveals=8, device="cpu",
                  out_dir="runs/p6/seed123/b0_sanity", tag="b0"):
    """Code-path sanity ONLY (no co-adaptation): candidate pool -> B/H extract ->
    cosine routing -> sum p_k L_k -> backward, on a frozen model."""
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    samples = build_dataset_p6(ckpt_path, M, layer=layer, head=head, heads=heads,
                               n_reveals=n_reveals, K=K, h_layer=h_layer, device=device)
    tr, va, te = _split_idx(len(samples), split)
    train = [samples[i] for i in tr]
    test = [samples[i] for i in (list(te) or list(tr))]
    Ytr, Ltr, _ = _yl(train)
    Yte, Lte, lab_te = _yl(test)
    head_in = headroom_stats([s["nll_by_label"] for s in test],
                             sigma_b_label="sigma_cdl_b")
    fixed_nll = eval_fixed_order_nll(test)
    sigma_b_nll = float(np.mean([s["nll_by_label"]["sigma_cdl_b"] for s in test]))

    g_B, hist_b = train_controller_online(train, Ytr, Ltr, mode="b_only", tau=tau,
                                          epochs=epochs, lr=lr, beta_entropy=beta_entropy,
                                          return_history=True)
    for p in g_B.parameters():
        p.requires_grad_(False)
    h_dim = samples[0]["H"].shape[1]

    arms = {}
    loss_down = {"b_only": hist_b[-1] < hist_b[0]}
    hard = eval_hard_selected_nll(test, g_B, Yte, Lte, lab_te, tau=tau)
    arms["b_only"] = {"nll_hard_selected": hard["nll"],
                      "nll_soft_expected": eval_soft_expected_nll(test, g_B, Yte, Lte, tau),
                      "nll_fixed_order": fixed_nll, "entropy": hard["entropy"],
                      "max_p": hard["max_p"], "selection_frac": hard["selection_frac"]}

    for mode in ("real", "shuffle", "zero", "mean"):
        sc, hist = train_controller_online(train, Ytr, Ltr, mode=mode, g_B=g_B,
                                           tau=tau, epochs=epochs, lr=lr,
                                           beta_entropy=beta_entropy, detach_h=detach_h,
                                           h_dim=h_dim, return_history=True)
        loss_down[mode] = hist[-1] < hist[0]
        h_te = h_mode_list([s["H"] for s in test], mode)
        hard = eval_hard_selected_nll(test, sc, Yte, Lte, lab_te, tau=tau,
                                      h_list=h_te, detach_h=detach_h)
        arms[mode] = {"nll_hard_selected": hard["nll"],
                      "nll_soft_expected": eval_soft_expected_nll(test, sc, Yte, Lte,
                                                                  tau, h_list=h_te,
                                                                  detach_h=detach_h),
                      "nll_fixed_order": fixed_nll, "entropy": hard["entropy"],
                      "max_p": hard["max_p"], "selection_frac": hard["selection_frac"],
                      "residual_ratio": sc.residual_ratio(
                          torch.tensor(test[0]["B_feat"]), torch.tensor(test[0]["H"]))}

    result = {"ckpt": ckpt_path, "tag": tag, "M": M, "K": K, "h_layer": h_layer,
              "tau": tau, "detach_h": detach_h, "heads": heads,
              "headroom": head_in, "sigma_b_nll": sigma_b_nll,
              "loss_down": loss_down, "arms": arms,
              "_note": "B0 has NO co-adaptation; H null here is expected by construction"}
    _json.dump(result, open(out / f"{tag}.json", "w"), indent=2, default=float)
    return result


# ════════════════════════════════════════════════════════════════════════════
# Task 6 + 9: B2 joint online co-adaptation (model + controller trainable)
# ════════════════════════════════════════════════════════════════════════════
# THE experiment: continue from an early ckpt; per step extract B/H from the LIVE
# model (no_grad), run K candidate forwards WITH model grad, route via cosine
# soft-routing, update model + controller on Sum_k p_k L_k. detach_h=True (B2a)
# keeps controller loss out of the model's hidden states.

@torch.no_grad()
def extract_b65_inmem(model, chunk_row, reveals, inv, layer, heads, dev):
    """Multi-head-mean B65 from the LIVE in-memory model (no reload, no grad)."""
    A_acc = None
    for rev in reveals:
        po = torch.from_numpy(rev[None, :]).to(dev)
        _, _, attn_list = model.forward_fn(chunk_row.to(dev), po, return_attentions=True)
        attn = torch.stack(attn_list, 0).cpu().numpy()[:, 0]      # (L,H,257,257)
        A = _attn_to_A_block_loss_aligned_with_none_vec(attn, rev, inv)
        A_acc = A.astype(np.float64) if A_acc is None else A_acc + A
    A_mean = A_acc / len(reveals)
    hs = heads if isinstance(heads, (list, tuple)) else [heads]
    return np.mean([build_none_separated_B(A_mean[layer, h]) for h in hs], axis=0)


def build_live_samples(model, chunks, clean_perm, dev, idxs, reveals, inv,
                       layer, heads, h_layer, K, rng, h_context="sigma_B"):
    """Rebuild per-text B/H/candidate scaffold from the current model (no_grad)."""
    samples = []
    for t in idxs:
        B65 = extract_b65_inmem(model, chunks[t:t+1], reveals, inv, layer, heads, dev)
        sigma_B = sigma_from_B65(B65)
        cands = p6_candidate_pool(B65, sigma_B, rng, K=K)
        H = extract_h_by_context(model, chunks[t:t+1], sigma_B, clean_perm, [h_layer],
                                 dev, h_context)
        samples.append({"B_feat": block_b_features(B65), "H": H, "cands": cands,
                        "sigma_B": sigma_B, "idx_row": chunks[t:t+1]})
    return samples


def make_b2_controller(b_dim, h_dim, mode, alpha_init=0.01, device="cpu"):
    """B-only -> BOnlyController. real/shuffle/etc -> ScaffoldedController with
    g_B TRAINABLE (B2 co-adapts g_B too, unlike the frozen B0 residual)."""
    if mode in ("b_only",):
        return BOnlyController(b_dim).to(device)
    sc = ScaffoldedController(BOnlyController(b_dim), h_dim=h_dim, alpha_init=alpha_init)
    for p in sc.g_B.parameters():
        p.requires_grad_(True)                                    # B2: g_B trainable
    return sc.to(device)


def b2_routing_step(model, controller, samples, clean_perm, dev, tau, beta_entropy,
                    detach_h, h_mode):
    """One joint step's loss. K candidate forwards retain model graph; controller
    grad flows only through p_k. Returns (loss tensor, mean entropy)."""
    use_H = h_mode not in (None, "b_only")
    Hs = h_mode_list([s["H"] for s in samples], h_mode) if use_H else None
    total, ents = 0.0, []
    for i, s in enumerate(samples):
        labels, Y = priority_matrix(s["cands"])
        Yt = torch.as_tensor(Y, dtype=torch.float32, device=dev)
        z = controller_scores(controller, s["B_feat"], None if Hs is None else Hs[i],
                              detach_h=detach_h)
        Lk = []
        for lab in labels:
            sigma = np.asarray(s["cands"][lab], dtype=np.int64)[None, :]
            tok = physical_blocks_to_model_token_order(
                torch.from_numpy(sigma), clean_perm, BLOCK_LEN).to(dev)
            _, loss_k = model.forward_fn(s["idx_row"].to(dev), tok)   # WITH model grad
            Lk.append(loss_k)
        Lk = torch.stack(Lk)
        a = cosine_logits(z, Yt, tau)
        p = torch.softmax(a, dim=0)
        ent = -(p * (p + 1e-12).log()).sum()
        total = total + (p * Lk).sum() - beta_entropy * ent
        ents.append(float(ent.detach()))
    return total / len(samples), float(np.mean(ents))


@torch.no_grad()
def b2_eval_fixed_order(model, chunks, clean_perm, dev, val_idxs, sigma_fixed=None):
    if sigma_fixed is None:
        sigma_fixed = np.arange(N, dtype=np.int64)
    return float(np.mean([order_nll(model, chunks[t:t+1], sigma_fixed, clean_perm, dev)
                          for t in val_idxs]))


def _set_trainable(model):
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    return model


def train_b2_arm(ckpt_path, mode, M_train=16, M_val=8, batch=4, steps=100,
                 n_reveals=4, K=6, h_layer=1, layer=0, heads=(1, 2, 3, 4),
                 tau=0.3, lr=1e-4, beta_entropy=0.0, detach_h=True, refresh_every=10,
                 eval_every=25, alpha_init=0.01, device="cpu", seed=0):
    """One B2 arm: continue training the model + controller from `ckpt_path`."""
    rng = np.random.default_rng(seed)
    model, chunks, clean_perm, dev = load_p5_ckpt(ckpt_path, M_train + M_val, device=device)
    _set_trainable(model)
    inv = clean_perm.inv_perm_model_to_phys.cpu().numpy()
    reveals = random_reveal_orders(n_reveals, 0)
    all_idx = np.arange(M_train + M_val)
    tr_idx, va_idx = all_idx[:M_train], all_idx[M_train:]

    b_dim = 2 * (N + 1)                                          # block_b_features width
    h_dim = block_hidden_states(model, chunks[0:1], np.arange(N), clean_perm, h_layer,
                                dev).shape[1]
    controller = make_b2_controller(b_dim, h_dim, mode, alpha_init=alpha_init, device=dev)
    params = [p for p in model.parameters() if p.requires_grad] + \
             [p for p in controller.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)

    hist = {"step": [], "loss": [], "entropy": [], "val_fixed_nll": []}
    live = None
    for step in range(steps):
        if step % refresh_every == 0:
            mb = rng.choice(tr_idx, size=min(batch, len(tr_idx)), replace=False)
            live = build_live_samples(model, chunks, clean_perm, dev, mb, reveals, inv,
                                      layer, list(heads), h_layer, K, rng)
        opt.zero_grad()
        loss, ent = b2_routing_step(model, controller, live, clean_perm, dev, tau,
                                    beta_entropy, detach_h, mode)
        loss.backward(); opt.step()
        if step % eval_every == 0 or step == steps - 1:
            vfn = b2_eval_fixed_order(model, chunks, clean_perm, dev, va_idx)
            hist["step"].append(step); hist["loss"].append(float(loss))
            hist["entropy"].append(ent); hist["val_fixed_nll"].append(vfn)
            _set_trainable(model)
    return {"mode": mode, "hist": hist}


def run_b2_smoke(ckpt_path, arms=("b_only", "real", "shuffle"), out_dir="runs/p6/seed123/b2_smoke",
                 tag="b2_smoke", **kw):
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    results = {}
    for mode in arms:
        results[mode] = train_b2_arm(ckpt_path, mode, **kw)
    payload = {"ckpt": ckpt_path, "tag": tag, "config": kw, "arms": results,
               "_note": "B2 smoke: validate joint chain (entropy/memory/loss/val), "
                        "not a science verdict at this step count"}
    _json.dump(payload, open(out / f"{tag}.json", "w"), indent=2, default=float)
    return payload
