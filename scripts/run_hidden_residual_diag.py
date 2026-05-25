#!/usr/bin/env python3
"""Phase-1 hidden-residual order diagnostic — orchestrator.
Runs the staged pipeline (pre-gate -> 1a -> 1b -> order-effects) on a clean-line text ckpt.
NO training. See docs/superpowers/specs/2026-05-25-hidden-residual-order-diagnostic-design.md."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "block_lo_arm_order_network"))
sys.path.insert(0, str(_REPO / "nanogpt-learned-order"))

from AOGPT import AOGPT, AOGPTConfig
from clean_training_protocol import (build_clean_block_permutation,
                                     phys_to_model_idx_clean)
from training_utils import load_train_chunks
from train_clean_aogpt import extract_A_matrices
import hidden_residual_graph as G
import hidden_residual_hidden as Hd
import hidden_residual_probe as P
from attn_order_teacher import teacher_components


def load_clean_ckpt(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    ma = ckpt["model_args"]
    sig = set(AOGPTConfig.__init__.__code__.co_varnames)
    model = AOGPT(AOGPTConfig(**{k: v for k, v in ma.items() if k in sig})).to(device)
    sd = ckpt["model"]
    if all(k.startswith("_orig_mod.") for k in sd):
        sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
    incompatible = model.load_state_dict(sd, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(f"WARNING load_state_dict: missing={incompatible.missing_keys[:5]} "
              f"unexpected={incompatible.unexpected_keys[:5]}", flush=True)
    model.eval()
    # Try several places the permute_seed may be stored
    if isinstance(ckpt.get("config"), dict) and "permute_seed" in ckpt["config"]:
        seed = int(ckpt["config"]["permute_seed"])
    elif isinstance(ckpt.get("args"), dict) and "permute_seed" in ckpt["args"]:
        seed = int(ckpt["args"]["permute_seed"])
    else:
        seed = 42
    return model, seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--n-chunks", type=int, default=256)
    p.add_argument("--m-passes", type=int, default=4, help="attention extraction passes; >=2 required, even preferred (split into 2 halves for noise floor)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-root", default=str(_REPO / "block_lo_arm_order_network/probe_results/hidden_residual_diag"))
    args = p.parse_args()
    if args.m_passes < 2:
        p.error("--m-passes must be >= 2 (need >=1 pass per half for the split-pass noise floor)")
    dev = torch.device(args.device)
    out = Path(args.out_root) / args.tag; out.mkdir(parents=True, exist_ok=True)

    model, perm_seed = load_clean_ckpt(args.ckpt, dev)
    clean_perm = build_clean_block_permutation(G.N_BLOCKS, seed=perm_seed)
    idx_phys = load_train_chunks(n_chunks=args.n_chunks)
    idx_model = phys_to_model_idx_clean(idx_phys, clean_perm)

    # M passes of attention -> per-sample A (physical frame) averaged over passes; keep halves
    passes = [extract_A_matrices(model, idx_model, clean_perm, dev, n_chunks=args.n_chunks)
              for _ in range(args.m_passes)]
    A_stack = np.stack(passes)                       # (M, n, N, N)
    A_all = A_stack.mean(0)                           # (n, N, N) all-pass per-sample
    half = args.m_passes // 2
    A_half_a = A_stack[:half].mean(0); A_half_b = A_stack[half:].mean(0)

    gate = G.gate_metrics(A_all, A_half_a, A_half_b, frame=G.PHYS_FRAME)
    (out / "gate.json").write_text(json.dumps(gate, indent=2))
    print("GATE:", json.dumps(gate, indent=2), flush=True)
    if not gate["passed"]:
        (out / "SUMMARY.md").write_text(
            f"# {args.tag}\n\nPRE-GATE FAILED (weak-model preliminary null). "
            f"r_norm={gate['r_norm']:.4g} <= noise_floor={gate['noise_floor']:.4g}. "
            "Per-sample B_x has no usable order residual on this ckpt. Not a final refutation.\n")
        print("Gate failed -> recorded preliminary null, stopping.", flush=True); return

    B_x_list, B_G, fr = G.build_B_set(A_all, frame=G.PHYS_FRAME)
    states = G.canonical_states(B_G)
    canon_order = G.rollout_order(B_G, mode="C-D+L", greedy=True, standardize=True)
    canon_model = clean_perm.block_perm_phys_to_model.numpy()[canon_order.astype(int)]
    canon_model_t = torch.tensor(canon_model, dtype=torch.long).unsqueeze(0).expand(args.n_chunks, -1).contiguous()
    order_model_full = torch.arange(G.N_BLOCKS).unsqueeze(0).expand(args.n_chunks, -1).contiguous()

    # residual target r_x per sample at each t
    targets = [G.residual_target(B_x, B_G, states, fr, fr) for B_x in B_x_list]

    # hidden
    h_oracle = Hd.extract_oracle_hidden(model, idx_model, clean_perm, dev, order_model_full)  # (n,N,E)
    h_causal = Hd.extract_causal_hidden(model, idx_model, dev, canon_model_t, list(states.keys()))

    # ---- 1a representation probe: predict residual sign at t=max over candidates (oracle h_v) ----
    t_top = max(states.keys()); U_top = np.asarray(states[t_top][1])
    H_uv = h_oracle[:, U_top, :].reshape(-1, h_oracle.shape[-1])             # (n*|U|, E)
    r_uv = np.concatenate([np.sign(t[t_top]["r"]) for t in targets]).astype(int)
    phase1a = {"residual_sign_oracle": P.representation_probe(H_uv, (r_uv > 0).astype(int),
                                                              kind="classification")}
    (out / "phase1a.json").write_text(json.dumps(phase1a, indent=2))
    print("PHASE1A:", json.dumps(phase1a, indent=2), flush=True)

    # ---- 1b residual probe (oracle + causal) at t_top ----
    r_mat = np.stack([t[t_top]["r"] for t in targets])                       # (n, |U|)
    Ho = h_oracle[:, U_top, :]                                               # (n, |U|, E)
    # B1 global phi: C-D+L components of B_G over U (broadcast across samples)
    S, U, last = states[t_top]
    C, D, L, _ = teacher_components(B_G, S, U, last)
    phi_g = np.stack([C, D, L], axis=1)[None].repeat(len(targets), 0)        # (n,|U|,3)
    pos_id = np.eye(len(U))[None].repeat(len(targets), 0)                    # (n,|U|,|U|)
    res1b = P.residual_probe(r_mat, H_oracle=Ho, phi_global=phi_g, pos_id=pos_id)
    cand_emb = np.eye(len(U))[None].repeat(len(targets), 0)                  # candidate id emb
    feats_c = P.build_causal_features(h_causal[t_top], cand_emb)
    res1b.update(P.residual_probe_causal(r_mat, feats_c))
    res1b["R2_causal_over_oracle"] = (res1b["R2_causal"] / res1b["R2_oracle"]
                                      if res1b["R2_oracle"] > 1e-6 else None)
    (out / "phase1b.json").write_text(json.dumps(res1b, indent=2))
    print("PHASE1B:", json.dumps(res1b, indent=2), flush=True)

    # ---- order effect: oracle residual head prediction added to s_g ----
    # (use planted-free check: measure with the per-sample r itself as the upper-bound delta)
    eff = [P.order_effect(targets[i][t_top]["s_g"], targets[i][t_top]["r"]) for i in range(len(targets))]
    order_eff = {k: float(np.mean([e[k] for e in eff])) for k in eff[0]}
    (out / "order_effects.json").write_text(json.dumps(order_eff, indent=2))
    print("ORDER_EFFECTS:", json.dumps(order_eff, indent=2), flush=True)

    (out / "SUMMARY.md").write_text(
        f"# {args.tag}\n\nGate PASSED (snr={gate['snr']:.3g}).\n"
        f"- 1a residual-sign probe (oracle): score={phase1a['residual_sign_oracle']['score']:.3f} "
        f"vs control {phase1a['residual_sign_oracle']['shuffled_control']:.3f}\n"
        f"- 1b R2_oracle={res1b['R2_oracle']:.3f} (B1={res1b['R2_B1']:.3f}, B2={res1b['R2_B2']:.3f}); "
        f"R2_causal={res1b['R2_causal']:.3f}; causal/oracle={res1b['R2_causal_over_oracle']}\n"
        f"- order-effect (oracle-r upper bound): tau_vs_global={order_eff['tau_vs_global']:.3f}, "
        f"argmax_changed={order_eff['argmax_changed']:.3f}\n\n"
        "Preliminary (weak ckpt); null not a final refutation.\n")
    print("Wrote", out, flush=True)


if __name__ == "__main__":
    main()
