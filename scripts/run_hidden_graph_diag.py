#!/usr/bin/env python3
"""Hidden Graph Diagnostic (frozen, no training). Per-checkpoint report:
B_A vs B_H vs B_pos structure, position residualization, C-D+L readout orders,
graph-level lambda-mix + score-level gamma-mix, frozen NLL-under-order."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
for p in ["block_lo_arm_order_network", "nanogpt-learned-order", "scripts"]:
    sys.path.insert(0, str(_REPO / p))

import hidden_graph as HG
import position_graph as PG
import graph_normalize as GN
import graph_structure_metrics as SM
import graph_order as GO
import hidden_graph_modelio as MIO


def AOGPT_from_ckpt(ckpt, device):
    from AOGPT import AOGPT, AOGPTConfig
    ma = ckpt["model_args"]
    model = AOGPT(AOGPTConfig(**ma)).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model


def diagnose_text(args, device):
    from AOGPT import AOGPT
    from clean_training_protocol import build_clean_block_permutation, phys_to_model_idx_clean
    from training_utils import load_train_chunks
    from train_clean_aogpt import extract_A_matrices, physical_blocks_to_model_blocks
    import hidden_residual_graph as G
    import hidden_residual_hidden as Hd
    from run_hidden_residual_diag import load_clean_ckpt

    model, perm_seed = load_clean_ckpt(args.ckpt, device)
    block_len = int(model.block_order_block_len)
    clean_perm = build_clean_block_permutation(G.N_BLOCKS, seed=perm_seed)
    idx_phys = load_train_chunks(n_chunks=args.n_chunks)
    idx_model = phys_to_model_idx_clean(idx_phys, clean_perm)

    # B_A = global attention graph (physical frame), same as the residual line's B_G
    A_all = extract_A_matrices(model, idx_model, clean_perm, device, n_chunks=args.n_chunks)
    _, B_A, _ = G.build_B_set(A_all, frame=G.PHYS_FRAME)

    # B_H = cosine of full-context block hidden (physical frame), Stage-1 oracle
    order_model_full = torch.stack([clean_perm.block_perm_phys_to_model] * idx_model.shape[0])
    H = Hd.extract_oracle_hidden(model, idx_model, clean_perm, device, order_model_full)  # (n,N,E)
    _, B_H = HG.cosine_graph(H)

    B_pos = PG.text_position_graph(N=G.N_BLOCKS, tau=args.pos_tau)
    phys_to_model = lambda o: physical_blocks_to_model_blocks(
        torch.as_tensor(o, dtype=torch.long), clean_perm).cpu().numpy()
    nll_fn = lambda phys_order: MIO.nll_under_order_text(
        model, idx_model, phys_to_model(phys_order), block_len, device,
        batch_size=args.eval_batch_size)
    return B_A, B_H, B_pos, nll_fn


def analyze(B_A, B_H, B_pos, nll_fn, args):
    """Run the full diagnostic on three raw graphs + a per-order frozen NLL fn."""
    B_H_resid = GN.residualize(GN.offdiag_zscore(B_H), GN.offdiag_zscore(B_pos))
    graphs = {"B_A": B_A, "B_H_raw": B_H, "B_H_resid": B_H_resid, "B_pos": B_pos}

    structure = {name: {m.__name__: SM.structure_vs_null(g, metric=m, n_shuffle=args.n_shuffle)
                        for m in (SM.sharpness, SM.spectral_gap)}
                 for name, g in graphs.items()}
    corr = {
        "BH_vs_Bpos": GN.offdiag_corr(B_H, B_pos),
        "BH_vs_BA": GN.offdiag_corr(B_H, B_A),
        "BA_vs_Bpos": GN.offdiag_corr(B_A, B_pos),
        "BHresid_vs_BA": GN.offdiag_corr(B_H_resid, B_A),
        "BHresid_vs_Bpos": GN.offdiag_corr(B_H_resid, B_pos),  # sanity ~0
    }

    orders = {name: GO.cdl_order(g, greedy=True) for name, g in graphs.items()}
    lam_orders = {f"mix_lam{lam}": GO.graph_mix_order(B_A, B_H_resid, lam, greedy=True)
                  for lam in (0.0, 0.25, 0.5, 0.75, 1.0)}
    # score-level mix: report mean +/- spread over seeds for the noise floor (sampled)
    gamma_nll = {}
    for gamma in (0.0, 0.25, 0.5, 1.0, 2.0):
        seeds = [GO.score_mix_order(B_A, B_H_resid, gamma, greedy=False, seed=s)
                 for s in range(args.n_seed)]
        nlls = [nll_fn(o) for o in seeds]
        gamma_nll[f"gamma{gamma}"] = {"mean": float(np.mean(nlls)), "std": float(np.std(nlls))}
    base_nll = {name: nll_fn(o) for name, o in orders.items()}
    mix_nll = {name: nll_fn(o) for name, o in lam_orders.items()}
    return {"structure": structure, "corr": corr, "nll_orders": base_nll,
            "nll_graph_mix": mix_nll, "nll_score_mix": gamma_nll}


def control_and_verdict(B_A, B_H, B_pos, nll_fn, score_mix_nll, args):
    B_H_resid = GN.residualize(GN.offdiag_zscore(B_H), GN.offdiag_zscore(B_pos))
    # matched-random residual control (same multiset), evaluated at gamma=1.0
    ctrl_nlls = []
    for s in range(args.n_seed):
        Bc = GN.matched_random_residual(B_H_resid, seed=1000 + s)
        o = GO.score_mix_order(B_A, Bc, gamma=1.0, greedy=False, seed=s)
        ctrl_nlls.append(nll_fn(o))
    a_only = score_mix_nll["gamma0.0"]["mean"]
    floor = max(score_mix_nll["gamma0.0"]["std"], 1e-9)
    improved = {g: (a_only - v["mean"]) for g, v in score_mix_nll.items()}
    # stable improvement: adjacent gammas in {.25,.5,1} not worse, one clearly better
    adj = [improved[f"gamma{g}"] for g in (0.25, 0.5, 1.0)]
    stable = sum(d >= -floor for d in adj) >= 2 and max(adj) > 2 * floor
    beats_ctrl = improved["gamma1.0"] > (a_only - float(np.mean(ctrl_nlls))) + floor
    bh_resid_structured = (SM.structure_vs_null(B_H_resid, metric=SM.sharpness,
                                                n_shuffle=args.n_shuffle)["z"] > 2.0)
    not_clone = abs(GN.offdiag_corr(B_H_resid, B_A)) < 0.95
    not_pos = abs(GN.offdiag_corr(B_H_resid, B_pos)) < 0.1
    verdict = ("WIN" if all([bh_resid_structured, not_clone, not_pos, stable, beats_ctrl])
               else "NULL")
    return {"verdict": verdict, "improved_vs_Aonly": improved, "stable": stable,
            "beats_matched_control": beats_ctrl, "ctrl_nll_mean": float(np.mean(ctrl_nlls)),
            "bh_resid_structured": bh_resid_structured, "not_BA_clone": not_clone,
            "not_position_artifact": not_pos}


def write_report(out_dir, modality, ckpt, results, verdict):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    blob = {"modality": modality, "ckpt": str(ckpt), **results, "decision": verdict}
    (out / "report.json").write_text(json.dumps(blob, indent=2, default=float))
    lines = [f"# Hidden Graph Diagnostic — {modality}", f"- ckpt: `{ckpt}`",
             f"- **verdict: {verdict['verdict']}**", "",
             f"- corr(B_H,B_pos)={results['corr']['BH_vs_Bpos']:.3f}  "
             f"corr(B_H,B_A)={results['corr']['BH_vs_BA']:.3f}  "
             f"corr(B_Hresid,B_A)={results['corr']['BHresid_vs_BA']:.3f}",
             f"- score-mix NLL vs A-only: {verdict['improved_vs_Aonly']}",
             f"- stable={verdict['stable']} beats_control={verdict['beats_matched_control']} "
             f"bh_resid_structured={verdict['bh_resid_structured']}"]
    (out / "report.md").write_text("\n".join(lines))
    print(f"[report] {out/'report.json'}  verdict={verdict['verdict']}")


def diagnose_image(args, device):
    import pickle
    from AOGPT import AOGPT
    from directed_graph_policy import build_directed_graph
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = AOGPT_from_ckpt(ckpt, device)                       # see Step 5
    with open(args.meta, "rb") as f: meta = pickle.load(f)
    block_len = int(meta.get("block_order_block_len", 1))
    n_blocks = 64
    val = np.fromfile(args.data_val, dtype=np.uint16).reshape(-1, n_blocks * block_len)
    tokens = torch.from_numpy(val[:args.n_eval].astype(np.int64))
    B_A = build_directed_graph(np.load(args.a_global_path).astype(np.float32))
    H = MIO.extract_image_block_hidden(model, tokens, block_len, n_blocks, device)
    _, B_H = HG.cosine_graph(H)
    B_pos = PG.image_manhattan_graph(side=8, tau=args.pos_tau)
    nll_fn = lambda phys_order: MIO.nll_under_order_image(
        model, tokens, np.asarray(phys_order), block_len, device,
        batch_size=args.eval_batch_size)
    return B_A, B_H, B_pos, nll_fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modality", required=True, choices=["text", "image"])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-chunks", type=int, default=16)            # text
    p.add_argument("--n-eval", type=int, default=256)             # image
    p.add_argument("--a-global-path"); p.add_argument("--data-val"); p.add_argument("--meta")
    p.add_argument("--pos-tau", type=float, default=2.0)
    p.add_argument("--n-shuffle", type=int, default=100)
    p.add_argument("--n-seed", type=int, default=5)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    device = torch.device(args.device)
    if args.modality == "text":
        B_A, B_H, B_pos, nll_fn = diagnose_text(args, device)
    else:
        B_A, B_H, B_pos, nll_fn = diagnose_image(args, device)
    results = analyze(B_A, B_H, B_pos, nll_fn, args)
    verdict = control_and_verdict(B_A, B_H, B_pos, nll_fn, results["nll_score_mix"], args)
    write_report(args.out_dir, args.modality, args.ckpt, results, verdict)


if __name__ == "__main__":
    main()
