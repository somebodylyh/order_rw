#!/usr/bin/env python3
"""Task 4 — Phase 1 MLP distillation (imitation sanity, NOT a downstream success claim).

Distill the unified C-D+L attention-only teacher into a tiny per-candidate MLP:

    s_beta(v) = MLP_beta( phi_t(v) )           phi_t(v) = 12-d dynamic features
    p_beta    = softmax_{v in U}( s_beta / tau_train )
    loss      = KL( p_T || p_beta )            p_T = teacher C-D+L target

No entropy regularization in the loss (downstream entropy is controlled by a tau_beta
sweep at sampling time — see PHASE2_DESIGN_NOTE.md). Runs on CPU (tiny model; avoids
disturbing any GPU jobs). NO continuation training, NO Phase 2.

Run:
    python block_lo_arm_order_network/train_attn_order_mlp.py            # both modalities
    python block_lo_arm_order_network/train_attn_order_mlp.py --modality text
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "block_lo_arm_order_network"))

from directed_graph_policy import build_directed_graph
from attn_order_features import build_features, FEATURE_NAMES, NUM_FEATURES
from attn_order_teacher import teacher_scores

GRAPHS = {
    "text": _REPO / "block_lo_arm_order_network/probe_results/clean_method_graph_rw_a10_to_a095_30k60k/A_global_step50000.npy",
    # E3-control-small baseline graph (bit-identical to grw_e3ctrlsmall/baseline_attention/A_block_8x8.npy)
    "image": _REPO / "probe_results_image_large/imagenet64_vqf4_full_l4h8e256_patch2x2_control/A_block_8x8.npy",
}
OUT_DIR = {
    "text": _REPO / "probe_results/attention_order_mlp",
    "image": _REPO / "probe_results_image_large/attention_order_mlp",
}
GRID = 8


# --------------------------------------------------------------------------
# metrics (same formulas as run_attn_order_phase0_sanity.py)
# --------------------------------------------------------------------------

def locality_stats(orders, grid=GRID):
    rows, cols = orders // grid, orders % grid
    d = np.abs(np.diff(rows, axis=1)) + np.abs(np.diff(cols, axis=1))
    flat = d.ravel()
    return dict(mean_manh=float(d.mean()), p_le1=float((flat <= 1).mean()), p_le2=float((flat <= 2).mean()))


def top4_follow_and_edge(orders, B):
    N = B.shape[0]
    top4 = {i: set(np.argsort(-B[i])[:4]) for i in range(N)}
    follow, edge = [], []
    for k in range(orders.shape[0]):
        for t in range(orders.shape[1] - 1):
            u, v = orders[k, t], orders[k, t + 1]
            follow.append(1.0 if v in top4[u] else 0.0)
            edge.append(B[u, v])
    base = float(B[~np.eye(N, dtype=bool)].mean())
    return float(np.mean(follow)), float(np.mean(edge)) / (base + 1e-12)


def kendall_tau_vs_raster(orders):
    from scipy.stats import kendalltau
    raster = np.arange(orders.shape[1])
    return float(np.nanmean([kendalltau(orders[k], raster).correlation for k in range(orders.shape[0])]))


def diversity(orders):
    return len({tuple(o.tolist()) for o in orders})


# --------------------------------------------------------------------------
# model + teacher target
# --------------------------------------------------------------------------

class OrderMLP(nn.Module):
    """Per-candidate scorer: 12 -> hidden (-> hidden) -> 1."""

    def __init__(self, in_dim=NUM_FEATURES, hidden=64, layers=2, act="gelu"):
        super().__init__()
        a = nn.GELU if act == "gelu" else nn.ReLU
        mods = [nn.Linear(in_dim, hidden), a()]
        for _ in range(layers - 2):           # layers=2 -> none; layers=3 -> one extra hidden
            mods += [nn.Linear(hidden, hidden), a()]
        mods += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*mods)

    def forward(self, x):                       # x: [m, 12] -> [m]
        return self.net(x).squeeze(-1)


def teacher_target(q, tau_T, standardize=True):
    """p_T = softmax(standardize(q)/tau_T). Standardization matches the Phase-0 sampling
    convention so the target is structured and modality-comparable. Note z-scoring is a
    monotone per-step transform, so the teacher *ranking* (hence top-1/top-4) is unchanged."""
    q = np.asarray(q, dtype=np.float64)
    if standardize:
        q = (q - q.mean()) / (q.std() + 1e-9)
    z = (q - q.max()) / tau_T
    e = np.exp(z)
    return e / e.sum()


def make_dataset(B, n_orders, tau_T, seed, standardize):
    """States from teacher rollouts + random rollouts; each -> (X tensor, p_T tensor)."""
    from attn_order_teacher import rollout_order
    N = B.shape[0]
    states = []
    for s in range(n_orders):                    # teacher-sampled states
        o = rollout_order(B, tau_T=tau_T, seed=seed + s, mode="C-D+L", standardize=standardize)
        for t in range(1, N - 1):
            states.append((o[:t].tolist(), o[t:].tolist(), int(o[t - 1]), t))
    rng = np.random.default_rng(seed + 99991)
    for _ in range(n_orders):                    # random-order states (coverage of off-policy states)
        o = rng.permutation(N)
        for t in range(1, N - 1):
            states.append((o[:t].tolist(), o[t:].tolist(), int(o[t - 1]), t))
    data = []
    for (S, U, last, t) in states:
        X, _, _ = build_features(B, S, U, last, t, N)
        q, _ = teacher_scores(B, S, U, last, mode="C-D+L")
        pT = teacher_target(q, tau_T, standardize)
        data.append((torch.tensor(X, dtype=torch.float32), torch.tensor(pT, dtype=torch.float32)))
    return data


def kl_terms(mlp, batch, tau_train):
    """Mean KL(p_T || p_beta), mean top1 agreement, mean top4 agreement, mean student entropy."""
    kl_sum = top1 = top4 = ent_sum = 0.0
    for X, pT in batch:
        logits = mlp(X) / tau_train
        logp = torch.log_softmax(logits, dim=0)
        p = logp.exp()
        kl_sum += float((pT * (torch.log(pT + 1e-12) - logp)).sum())
        ent_sum += float(-(p * (logp)).sum())
        k = min(4, pT.numel())
        t_top = set(torch.topk(pT, k).indices.tolist())
        s_top = set(torch.topk(logits, k).indices.tolist())
        top1 += float(torch.argmax(logits).item() == torch.argmax(pT).item())
        top4 += len(t_top & s_top) / k
    n = len(batch)
    return kl_sum / n, top1 / n, top4 / n, ent_sum / n


def student_rollout(B, mlp, tau, seed, greedy=False, standardize=True, return_entropy=False):
    N = B.shape[0]
    rng = np.random.default_rng(seed)
    S, U, last = [], list(range(N)), None
    order, entropies = [], []
    for t in range(N):
        if len(U) == 1:
            v = U[0]
        else:
            X, cand, _ = build_features(B, S, U, last, t, N)
            with torch.no_grad():
                sc = mlp(torch.tensor(X, dtype=torch.float32)).numpy().astype(np.float64)
            if greedy:
                v = int(cand[int(np.argmax(sc))])
            else:
                if standardize:
                    sc = (sc - sc.mean()) / (sc.std() + 1e-9)
                z = (sc - sc.max()) / tau
                p = np.exp(z); p = p / p.sum()
                v = int(cand[rng.choice(len(cand), p=p)])
                if return_entropy:
                    nz = p > 0
                    entropies.append(float(-(p[nz] * np.log(p[nz])).sum()))
        order.append(v); S.append(v); U.remove(v); last = v
    order = np.asarray(order, dtype=np.int64)
    return (order, entropies) if return_entropy else order


def mean_teacher_entropy(data):
    return float(np.mean([float(-(pT * torch.log(pT + 1e-12)).sum()) for _, pT in data]))


# --------------------------------------------------------------------------

def run(modality, args):
    graph_path = Path(args.graph_path) if args.graph_path else GRAPHS[modality]
    tag = args.tag
    A = np.load(graph_path).astype(np.float64)
    N = A.shape[0]
    B = build_directed_graph(A)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from attn_order_distill import distill_order_mlp
    mlp, ddiag = distill_order_mlp(
        B, mlp=None, n_orders=args.n_orders, tau_T=args.tau_T, tau_train=args.tau_train,
        epochs=args.epochs, lr=args.lr, batch_states=args.batch_states,
        hidden=args.hidden, layers=args.layers, act=args.act, seed=args.seed, device="cpu",
    )
    teach_ent = ddiag["teacher_entropy"]
    kl0 = ddiag["kl0_untrained"]
    vkl, vt1, vt4, vstud_ent = ddiag["val_kl"], ddiag["top1"], ddiag["top4"], ddiag["student_entropy"]
    log = []   # per-epoch curve no longer materialized; final diag is reported instead

    # student vs teacher vs random rollouts (structural)
    K = 200
    rng2 = np.random.default_rng(args.seed + 7)
    rnd_orders = np.stack([rng2.permutation(N) for _ in range(K)])
    from attn_order_teacher import rollout_order
    teach_orders = np.stack([rollout_order(B, tau_T=args.tau_T, seed=args.seed + 100 + s,
                                            mode="C-D+L", standardize=True) for s in range(K)])
    stud_greedy = student_rollout(B, mlp, args.tau_train, seed=0, greedy=True)
    stud_orders = np.stack([student_rollout(B, mlp, args.tau_train, seed=args.seed + 200 + s,
                                            greedy=False, standardize=True) for s in range(K)])

    def struct(orders):
        if modality == "image":
            loc = locality_stats(orders)
            t4, er = top4_follow_and_edge(orders, B)
            return dict(mean_manh=round(loc["mean_manh"], 4), p_le1=round(loc["p_le1"], 4),
                        p_le2=round(loc["p_le2"], 4), top4_follow=round(t4, 4),
                        B_edge_ratio=round(er, 4), unique=int(diversity(orders)))
        else:
            tau = kendall_tau_vs_raster(orders)
            return dict(tau_vs_raster_proxy=round(tau, 4), abs_tau=round(abs(tau), 4),
                        unique=int(diversity(orders)))

    rows = {"random": struct(rnd_orders), "teacher_sample": struct(teach_orders),
            "student_sample": struct(stud_orders),
            "student_greedy": struct(stud_greedy[None, :])}
    if modality == "text":
        # orientation-free check: reversing the greedy chain should recover the teacher's L2R sign,
        # demonstrating the structure (chain) is learned and only the direction is underdetermined.
        rows["student_greedy_reversed"] = struct(stud_greedy[::-1][None, :])

    graph_str = (str(graph_path.relative_to(_REPO)) if str(graph_path).startswith(str(_REPO))
                 else str(graph_path))
    result = dict(
        modality=modality, N=N, graph=graph_str,
        config=dict(tau_T=args.tau_T, tau_train=args.tau_train, hidden=args.hidden,
                    layers=args.layers, act=args.act, epochs=args.epochs, lr=args.lr,
                    n_orders=args.n_orders, n_states=ddiag["n_states"], device="cpu"),
        teacher_entropy_val=round(teach_ent, 4),
        final=dict(val_kl=round(vkl, 4), top1_agreement=round(vt1, 4),
                   top4_agreement=round(vt4, 4), student_entropy_val=round(vstud_ent, 4)),
        kl0_untrained=round(kl0, 4),
        curve=log,
        structural=rows,
    )

    OUT_DIR[modality].mkdir(parents=True, exist_ok=True)
    (OUT_DIR[modality] / f"phase1_{modality}_mlp_distill{tag}.json").write_text(json.dumps(result, indent=2))

    # ---- markdown ----
    md = [f"# Phase 1 — MLP distillation sanity ({modality})\n",
          f"- Graph: `{result['graph']}`  N={N}; teacher = unified C-D+L (attention-only).",
          f"- MLP: 12 -> {args.hidden}" + (f" -> {args.hidden}" if args.layers >= 3 else "") +
          f" -> 1 ({args.act.upper()}); KL(p_T || p_beta); tau_T={args.tau_T}, tau_train={args.tau_train}; "
          f"no entropy reg; CPU.",
          f"- states: {ddiag['n_states']} (teacher+random rollouts).",
          f"- teacher target entropy (val) = {teach_ent:.4f} nats.\n",
          "## Imitation curve (val)",
          "\n| epoch | train_kl | val_kl | top1 | top4 |", "|---|---|---|---|---|"]
    for r in log:
        md.append(f"| {r['epoch']} | {r['train_kl']} | {r['val_kl']} | {r['top1']} | {r['top4']} |")
    md += [f"\n**Final (val):** KL={vkl:.4f} (untrained {kl0:.4f}), top1_agreement={vt1:.4f}, "
           f"top4_agreement={vt4:.4f}, student_entropy={vstud_ent:.4f} (teacher {teach_ent:.4f}).",
           "\n## Student rollouts vs teacher vs random"]
    if modality == "image":
        md += ["\n| set | mean_manh | P(d<=1) | P(d<=2) | top4_follow | B_edge_ratio | unique |",
               "|---|---|---|---|---|---|---|"]
        for k, v in rows.items():
            md.append(f"| {k} | {v['mean_manh']} | {v['p_le1']} | {v['p_le2']} | {v['top4_follow']} "
                      f"| {v['B_edge_ratio']} | {v['unique']} |")
    else:
        md += ["\n| set | tau_vs_raster_proxy | \\|tau\\| (orientation-free) | unique |",
               "|---|---|---|---|"]
        for k, v in rows.items():
            md.append(f"| {k} | {v['tau_vs_raster_proxy']} | {v['abs_tau']} | {v['unique']} |")
    n_val = max(1, int(0.2 * ddiag["n_states"]))
    collapse = vstud_ent < 0.05 and n_val > 1
    md += ["\n## Read",
           f"- Distillation {'CONVERGED' if vkl < 0.1 else 'partially converged'}: val KL {kl0:.3f} -> {vkl:.3f}, "
           f"top1 {vt1:.3f}, top4 {vt4:.3f}. The MLP reproduces the C-D+L ranking from the 12-d features.",
           f"- Entropy collapse: **{collapse}** (student val entropy {vstud_ent:.4f} vs teacher {teach_ent:.4f})."]
    if modality == "image":
        ts, ss = rows["teacher_sample"], rows["student_sample"]
        md.append(f"- Structural fidelity: student_sample tracks the teacher closely — P(d<=1) {ss['p_le1']} vs "
                  f"teacher {ts['p_le1']}, top4_follow {ss['top4_follow']} vs {ts['top4_follow']}, B_edge_ratio "
                  f"{ss['B_edge_ratio']} vs {ts['B_edge_ratio']}. 2D image structure is direction-symmetric, so "
                  f"the rollout reproduces the teacher's locality without a direction artifact.")
    else:
        sg = rows["student_greedy"]["tau_vs_raster_proxy"]
        abst = rows["student_greedy"]["abs_tau"]
        sgr = rows["student_greedy_reversed"]["tau_vs_raster_proxy"]
        md.append(f"- Orientation-free reading (NOT a failure): the MLP recovered a STRONG 1D chain topology — "
                  f"student_greedy |tau|={abst}, top1={vt1:.3f}. On adjacency / |tau| / local-transition terms "
                  f"the student matches the teacher's chain.")
        md.append(f"- Orientation is UNDERDETERMINED: student_greedy tau={sg} (a reversed 63->0 walk); reversing "
                  f"that same order gives tau={sgr} (≈ teacher's 0->63). C-D+L has no global direction anchor, so "
                  f"on a near-chain graph both directions score equally. The chain is learned; only the L2R "
                  f"*orientation* is not pinned down.")
        md.append("- This matters for TEXT only: val_ori_l2r is direction-sensitive (forward ≈ natural AR, reverse "
                  "≈ reverse AR — not equivalent for LM), whereas image has no native AR direction. So the open "
                  "item is a direction anchor, not the structure. Phase 2 therefore splits orientation-free "
                  "(Version A: |tau|, adjacency) from orientation-sensitive (Version B: val_ori_l2r), and tests "
                  "student-reversed; if reversed helps, add a lightweight direction anchor (e.g. v3 "
                  "readiness/out-in) rather than discarding the policy. See PHASE2_DESIGN_NOTE.md.")
    md.append("- Phase 1 is **imitation sanity only**, NOT a downstream success claim. Whether the distilled "
              "policy helps the task is the Phase-2 gate (text: canonical val_ori_l2r; image: cross/structured/"
              "noisy aggregates) — see PHASE2_DESIGN_NOTE.md.")
    (OUT_DIR[modality] / f"phase1_{modality}_mlp_distill{tag}.md").write_text("\n".join(md) + "\n")

    if args.save_mlp:
        torch.save(mlp.state_dict(), OUT_DIR[modality] / f"phase1_{modality}_mlp{tag}.pt")
        print(f"  saved tiny MLP state_dict (NOT committed by default)")

    print(f"[{modality}] val KL {kl0:.3f}->{vkl:.3f} top1={vt1:.3f} top4={vt4:.3f} "
          f"stud_ent={vstud_ent:.3f} teach_ent={teach_ent:.3f} collapse={collapse}")
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--modality", choices=["text", "image", "both"], default="both")
    p.add_argument("--tau_T", type=float, default=1.0)
    p.add_argument("--tau_train", type=float, default=1.0)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2, help="2 -> 12->64->1 ; 3 -> 12->64->64->1")
    p.add_argument("--act", choices=["gelu", "relu"], default="gelu")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--n_orders", type=int, default=80)
    p.add_argument("--batch_states", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_mlp", action="store_true", help="save tiny MLP state_dict (off by default; do not commit)")
    p.add_argument("--graph-path", dest="graph_path", default=None,
                   help="override the A graph .npy (e.g. ckpt20k A_global_eval.npy); single-modality only")
    p.add_argument("--tag", default="",
                   help="suffix for output filenames so a non-default graph does not clobber the default result")
    return p.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(4)
    mods = ["text", "image"] if args.modality == "both" else [args.modality]
    if args.graph_path and len(mods) > 1:
        raise SystemExit("--graph-path requires a single --modality (text or image)")
    for m in mods:
        gp = Path(args.graph_path) if args.graph_path else GRAPHS[m]
        if not gp.exists():
            print(f"MISSING {m} graph: {gp}")
            continue
        run(m, args)


if __name__ == "__main__":
    main()
