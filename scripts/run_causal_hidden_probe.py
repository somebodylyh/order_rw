#!/usr/bin/env python3
"""Causal-Hidden Order Probe (frozen, no training). Per-checkpoint report: PRE-GATE causal
invariance; Level-0 baselines; Level-1 dynamic gamma-mix (cos-emb) vs A-only with position +
shuffled-hidden controls; WIN gate; decision. Two-path design:
  Path X (MAIN / verdict gate): candidate-conditioned c_t^{(v)} — different hidden per candidate.
  Path Y (CONTROL only — never gates): target-neutral pooled p_t — same vector for all candidates.
A Path-Y NULL does NOT close the line; only Path X gates the verdict.
See docs/superpowers/specs/2026-05-26-causal-hidden-order-probe-design.md."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
for p in ["block_lo_arm_order_network", "nanogpt-learned-order", "scripts"]:
    sys.path.insert(0, str(_REPO / p))

import causal_hidden_rollout as CHR
import position_graph as PG
import graph_order as GO
import hidden_graph_modelio as MIO


def build_text(args, device):
    from clean_training_protocol import build_clean_block_permutation, phys_to_model_idx_clean
    from training_utils import load_train_chunks
    from train_clean_aogpt import extract_A_matrices, physical_blocks_to_model_blocks
    import hidden_residual_graph as G
    from run_hidden_residual_diag import load_clean_ckpt

    model, perm_seed = load_clean_ckpt(args.ckpt, device)
    BL = int(model.block_order_block_len)
    clean_perm = build_clean_block_permutation(G.N_BLOCKS, seed=perm_seed)
    idx_phys = load_train_chunks(n_chunks=args.n_chunks)
    idx_model = phys_to_model_idx_clean(idx_phys, clean_perm)
    idx_roll = idx_model[:args.n_roll]

    A_all = extract_A_matrices(model, idx_model, clean_perm, device, n_chunks=args.n_chunks)
    _, B_A, _ = G.build_B_set(A_all, frame=G.PHYS_FRAME)
    B_pos = PG.text_position_graph(N=G.N_BLOCKS, tau=args.pos_tau)

    # phys-frame blocks (rollout/B_A frame) -> model-frame blocks (what the model is fed).
    # Reuse the SAME helper nll_fn uses, so framing is consistent (NOT inv_perm_model_to_phys,
    # which is the opposite direction — a classic coordinate bug in this codebase).
    def p2m(phys_blocks):
        return physical_blocks_to_model_blocks(
            torch.as_tensor(np.asarray(phys_blocks), dtype=torch.long), clean_perm).cpu().numpy()

    # Path X (MAIN): per-candidate c_t^{(v)} -> (M, E)
    # ctx_fn must return one row per candidate in the same order as cand_phys.
    def ctxX_fn(S_phys, cand_phys):
        prefix_m = p2m(S_phys).tolist()
        cols = []
        for v in cand_phys:
            others = [u for u in cand_phys if u != v]
            next_m = int(p2m([v])[0])
            other_m = p2m(others).tolist() if others else []
            c = CHR.candidate_conditioned_hidden(model, idx_roll, prefix_m, next_m, other_m, BL, device)
            cols.append(c.mean(axis=0))    # mean over n_roll samples -> (E,)
        return np.stack(cols, axis=0)      # (M, E)

    # Path Y (CONTROL): shared pooled p_t -> (E,)
    def ctxY_fn(S_phys, cand_phys):
        prefix_m = p2m(S_phys).tolist()
        comp_m = p2m(cand_phys).tolist()
        p = CHR.pooled_context_hidden(model, idx_roll, prefix_m, comp_m, BL, device)
        return p.mean(axis=0)              # (E,)

    # Candidate embeddings -> (M, E)
    def cand_fn(cand_phys):
        E_c = CHR.candidate_embeddings(model, idx_roll, p2m(cand_phys).tolist(), BL,
                                       mode=args.ev_mode, device=device)
        return E_c.mean(axis=0)            # (M, E)

    nll_fn = lambda phys_order: MIO.nll_under_order_text(
        model, idx_model, physical_blocks_to_model_blocks(
            torch.as_tensor(phys_order, dtype=torch.long), clean_perm).cpu().numpy(),
        BL, device, batch_size=args.eval_batch_size)

    pregate = CHR.causal_invariance_check(model, idx_roll, G.N_BLOCKS, BL, device,
                                          n_patterns=args.n_patterns)
    print(f"[PRE-GATE] passed={pregate['passed']} min_cosine={pregate['min_cosine']:.6f} "
          f"n_compare={pregate['n_compare']} n_fail={pregate['n_fail']}", flush=True)

    return dict(model=model, B_A=B_A, B_pos=B_pos,
                ctxX_fn=ctxX_fn, ctxY_fn=ctxY_fn, cand_fn=cand_fn,
                nll_fn=nll_fn, pregate=pregate, N=G.N_BLOCKS)


def build_image(args, device):
    raise NotImplementedError("CT7: image path not yet implemented")


def run_levels(ctx, args):
    B_A, B_pos = ctx["B_A"], ctx["B_pos"]
    cand_fn, nll_fn, N = ctx["cand_fn"], ctx["nll_fn"], ctx["N"]
    ctxX_fn, ctxY_fn = ctx["ctxX_fn"], ctx["ctxY_fn"]
    l2r = np.arange(N, dtype=np.int64)

    # Level-0 baselines
    static_cdl = GO.cdl_order(B_A, greedy=True)
    rng = np.random.default_rng(0)
    base = {
        "static_BA_CDL": nll_fn(static_cdl),
        "random": nll_fn(rng.permutation(N).astype(np.int64)),
        "L2R_or_raster": nll_fn(l2r),
    }
    print(f"[baselines] static_CDL={base['static_BA_CDL']:.4f}  "
          f"random={base['random']:.4f}  L2R={base['L2R_or_raster']:.4f}", flush=True)

    # Level-1: gamma sweep, both paths
    gammas = [0.0, 0.5, 1.0]   # cost-controlled: 3 points (gamma0 = A-only anchor)

    def sweep(path, ctx_fn):
        d = {}
        for g in gammas:
            print(f"  [{path}] gamma={g} rollout ...", flush=True)
            o = CHR.causal_score_mix_rollout(B_A, B_pos, g, path, ctx_fn, cand_fn,
                                             hidden_off_at_t0=True)
            nll = float(nll_fn(o))
            tau = float(CHR._kendall_tau(o, l2r))
            d[f"gamma{g}"] = {"nll": nll, "tau_vs_L2R": tau}
            print(f"  [{path}] gamma={g} nll={nll:.4f}  tau_vs_L2R={tau:.3f}", flush=True)
        return d

    print("[Level-1] Path X (MAIN: candidate-conditioned) ...", flush=True)
    pathX = sweep("X", ctxX_fn)

    print("[Level-1] Path Y (CONTROL: pooled target-neutral) ...", flush=True)
    pathY = sweep("Y", ctxY_fn)

    # Controls
    print("[controls] position_control ...", flush=True)
    pos_nll = float(nll_fn(CHR.position_control_rollout(B_A, B_pos, gamma=1.0)))
    print(f"[controls] position nll={pos_nll:.4f}", flush=True)

    print("[controls] shuffled_hidden_X ...", flush=True)
    shuf_nll = float(nll_fn(
        CHR.shuffled_hidden_rollout(B_A, B_pos, 1.0, "X", ctxX_fn, cand_fn, seed=0)))
    print(f"[controls] shuffled_hidden_X nll={shuf_nll:.4f}", flush=True)

    controls = {
        "position": pos_nll,
        "shuffled_hidden_X": shuf_nll,
    }
    return {"baselines": base, "pathX": pathX, "pathY": pathY,
            "controls": controls, "gammas": gammas}


def verdict(results):
    """6-condition WIN gate — evaluated on Path X (MAIN) only.
    Path Y is reported but never gates the verdict."""
    px = results["pathX"]
    a_only = px["gamma0.0"]["nll"]
    improved = {g: a_only - v["nll"] for g, v in px.items()}
    nonzero = [g for g in px if g != "gamma0.0"]
    best = max(improved.values())
    best_g = max(px, key=lambda k: a_only - px[k]["nll"])
    adj = [improved[g] for g in nonzero]
    # "adjacent gammas don't collapse": require the non-best nonzero gammas to not fall far below A-only.
    # With 2 nonzero gammas, at least 1 must be non-negative (i.e. >=1 out of max(1, len-1)=1).
    stable = (best > 0.01) and (sum(d >= -1e-3 for d in adj) >= max(1, len(adj) - 1))
    beats_pos = (a_only - results["controls"]["position"]) < best
    beats_shuf = (a_only - results["controls"]["shuffled_hidden_X"]) < best
    not_l2r = abs(px[best_g]["tau_vs_L2R"]) < 0.95
    win = all([best > 0.01, stable, beats_pos, beats_shuf, not_l2r])
    return {
        "verdict": "WIN" if win else "NULL",
        "improved_vs_Aonly": improved,
        "best_gamma": best_g,
        "stable": stable,
        "beats_position": beats_pos,
        "beats_shuffled": beats_shuf,
        "not_L2R_degenerate": not_l2r,
    }


def write_report(out_dir, modality, ckpt, pregate, results, verd, ev_mode):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    blob = {
        "modality": modality,
        "ckpt": str(ckpt),
        "pregate": pregate,
        "results": results,
        "decision": verd,
    }
    (out / "report.json").write_text(json.dumps(blob, indent=2, default=float))

    content_token_caveat = ""
    if ev_mode == "content_token":
        content_token_caveat = (
            "\n> CAVEAT: content_token e_v sees candidate content -> "
            "NOT a deployable generation-time controller\n"
        )

    px = results["pathX"]
    py = results["pathY"]
    base = results["baselines"]
    ctrl = results["controls"]

    lines = [
        f"# Causal-Hidden Probe — {modality}",
        f"",
        f"- ckpt: `{ckpt}`",
        f"- ev_mode: `{ev_mode}`",
        f"- PRE-GATE passed: {pregate['passed']} (min_cosine={pregate['min_cosine']:.6f}, "
        f"n_compare={pregate['n_compare']}, n_fail={pregate['n_fail']})",
        f"",
        content_token_caveat,
        f"## Verdict: **{verd['verdict']}**  (best gamma: {verd['best_gamma']})",
        f"",
        f"| condition | value |",
        f"|-----------|-------|",
        f"| best improvement vs A-only | {max(verd['improved_vs_Aonly'].values()):.4f} |",
        f"| stable (adj gammas don't collapse) | {verd['stable']} |",
        f"| beats position control | {verd['beats_position']} |",
        f"| beats shuffled-hidden control | {verd['beats_shuffled']} |",
        f"| not L2R-degenerate | {verd['not_L2R_degenerate']} |",
        f"",
        f"## Level-0 Baselines",
        f"",
        f"| order | NLL |",
        f"|-------|-----|",
        f"| static_BA_CDL | {base['static_BA_CDL']:.4f} |",
        f"| random | {base['random']:.4f} |",
        f"| L2R_or_raster | {base['L2R_or_raster']:.4f} |",
        f"",
        f"## Level-1: Path X (MAIN — candidate-conditioned c_t^(v))",
        f"",
        f"| gamma | NLL | tau_vs_L2R |",
        f"|-------|-----|------------|",
    ] + [
        f"| {g} | {v['nll']:.4f} | {v['tau_vs_L2R']:.3f} |"
        for g, v in px.items()
    ] + [
        f"",
        f"## Level-1: Path Y (CONTROL — pooled target-neutral p_t)",
        f"",
        f"| gamma | NLL | tau_vs_L2R |",
        f"|-------|-----|------------|",
    ] + [
        f"| {g} | {v['nll']:.4f} | {v['tau_vs_L2R']:.3f} |"
        for g, v in py.items()
    ] + [
        f"",
        f"## Controls",
        f"",
        f"| control | NLL |",
        f"|---------|-----|",
        f"| position (B_pos slice) | {ctrl['position']:.4f} |",
        f"| shuffled_hidden_X | {ctrl['shuffled_hidden_X']:.4f} |",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(f"[report] {out / 'report.json'}  verdict={verd['verdict']}", flush=True)


def main():
    p = argparse.ArgumentParser(
        description="Causal-Hidden Order Probe: frozen diagnostic, text or image.")
    p.add_argument("--modality", required=True, choices=["text", "image"])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-chunks", type=int, default=16)
    p.add_argument("--n-roll", type=int, default=16)
    p.add_argument("--ev-mode", default="content_token",
                   choices=["content_token", "content_free"])
    p.add_argument("--pos-tau", type=float, default=2.0)
    p.add_argument("--n-patterns", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda:0")
    # Image-only args (CT7 wires these; unused for text)
    p.add_argument("--a-global-path", default=None)
    p.add_argument("--data-val", default=None)
    p.add_argument("--meta", default=None)
    p.add_argument("--n-eval", type=int, default=256)
    args = p.parse_args()

    device = torch.device(args.device)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.modality == "text":
        ctx = build_text(args, device)
    else:
        ctx = build_image(args, device)

    if not ctx["pregate"]["passed"]:
        (out / "PREGATE_FAILED.md").write_text(
            f"# PRE-GATE FAILED ({args.modality})\n"
            f"Future completion leaks into predictor hidden; "
            f"dynamic-hidden probe invalid.\n"
            f"min_cosine={ctx['pregate']['min_cosine']:.6f}\n"
            f"n_fail={ctx['pregate']['n_fail']}\n"
            f"fails={json.dumps(ctx['pregate']['fails'], indent=2)}\n")
        print(f"[PRE-GATE FAILED] min_cosine={ctx['pregate']['min_cosine']:.6f} "
              f"n_fail={ctx['pregate']['n_fail']} -> abort", flush=True)
        return

    results = run_levels(ctx, args)
    verd = verdict(results)
    write_report(args.out_dir, args.modality, args.ckpt, ctx["pregate"],
                 results, verd, args.ev_mode)


if __name__ == "__main__":
    main()
