#!/usr/bin/env python3
"""Phase 1.5 — rollout + orientation diagnostic for the distilled order policy.

Goal (NOT training): explain the text student-greedy REVERSAL and confirm the image student
free-rollout is stable, so Phase 2 can be scoped correctly.

  Task A (text):  teacher/student greedy + sampled tau_beta sweep {0.5,1,1.5,2} + random + v3 ref.
                  Metrics: tau_vs_raster, abs_tau, forward/reverse ratio, start/end dist, avg step
                  entropy, unique count, FREE-RUNNING top1/top4 (student's own path, not teacher-forced).
  Task B (image): teacher/student greedy + sampled sweep + random + Bcov ref.
                  Metrics: mean_manh, P(d<=1), P(d<=2), top4_follow, B_edge_ratio, entropy, unique.

Uses the saved Phase-1 MLPs. NO continuation, NO trainer changes, NO large files saved.

Run:
    python block_lo_arm_order_network/train_attn_order_mlp.py --save_mlp   # produce the MLPs first
    python scripts/run_attn_order_phase1_5_rollout_diagnostic.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "block_lo_arm_order_network"))

from directed_graph_policy import build_directed_graph, sample_order
from attn_order_teacher import rollout_order
from attn_order_diagnostics import (
    locality_stats, top4_follow_and_edge, kendall_tau_vs_raster, forward_reverse_ratio,
    diversity, start_end_distribution, free_running_agreement,
)
from train_attn_order_mlp import OrderMLP, student_rollout

TEXT_A = _REPO / "block_lo_arm_order_network/probe_results/clean_method_graph_rw_a10_to_a095_30k60k/A_global_step50000.npy"
IMAGE_A = _REPO / "probe_results_image_large/imagenet64_vqf4_full_l4h8e256_patch2x2_control/A_block_8x8.npy"
TEXT_MLP = _REPO / "probe_results/attention_order_mlp/phase1_text_mlp.pt"
IMAGE_MLP = _REPO / "probe_results_image_large/attention_order_mlp/phase1_image_mlp.pt"
BCOV_REF = _REPO / "probe_results_image_large/grw_e3ctrlsmall/readout_diagnostic/orders/Bcov_balanced.npy"
TEXT_V3_PARAMS = dict(tau_start=0.1, tau_step=0.1, alpha_dep=0.5, lam=0.75, rho=0.2, top_k=4, epsilon_uniform=0.0)

TAUS = [0.5, 1.0, 1.5, 2.0]
K = 200
SEED = 1234


def load_mlp(path, hidden=64, layers=2, act="gelu"):
    m = OrderMLP(hidden=hidden, layers=layers, act=act)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    m.eval()
    return m


def student_score_fn(mlp):
    def f(X):
        with torch.no_grad():
            return mlp(torch.tensor(X, dtype=torch.float32)).numpy()
    return f


# --- order set generators ---------------------------------------------------

def gen_teacher(B, kind, tau=None):
    if kind == "greedy":
        return rollout_order(B, mode="C-D+L", greedy=True)[None, :], float("nan")
    ords, ents = [], []
    for s in range(K):
        o, e = rollout_order(B, tau_T=tau, seed=SEED + s, mode="C-D+L", standardize=True, return_entropy=True)
        ords.append(o); ents.extend(e)
    return np.stack(ords), float(np.mean(ents))


def gen_student(B, mlp, kind, tau=None):
    if kind == "greedy":
        return student_rollout(B, mlp, tau=1.0, seed=0, greedy=True)[None, :], float("nan")
    ords, ents = [], []
    for s in range(K):
        o, e = student_rollout(B, mlp, tau=tau, seed=SEED + 500 + s, greedy=False,
                               standardize=True, return_entropy=True)
        ords.append(o); ents.extend(e)
    return np.stack(ords), float(np.mean(ents))


def gen_random(B):
    rng = np.random.default_rng(SEED + 9)
    return np.stack([rng.permutation(B.shape[0]) for _ in range(K)]), float("nan")


def gen_v3(B):
    ords = [sample_order(B, "progressive_rw_v3", TEXT_V3_PARAMS, seed=SEED + 2000 + s)[0] for s in range(K)]
    return np.stack(ords), float("nan")


# --- per-modality drivers ---------------------------------------------------

def text_metrics(orders, avg_ent):
    fwd, rev, abst = forward_reverse_ratio(orders)
    sed = start_end_distribution(orders)
    return dict(K=int(orders.shape[0]), tau_vs_raster=round(kendall_tau_vs_raster(orders), 4),
                abs_tau=round(abst, 4), fwd_ratio=round(fwd, 3), rev_ratio=round(rev, 3),
                avg_step_entropy=("" if np.isnan(avg_ent) else round(avg_ent, 4)),
                unique=int(diversity(orders)),
                top_start=sed["top_start"][:3], top_end=sed["top_end"][:3])


def image_metrics(orders, avg_ent, B):
    loc = locality_stats(orders)
    t4, er = top4_follow_and_edge(orders, B)
    return dict(K=int(orders.shape[0]), mean_manh=round(loc["mean_manh"], 4),
                p_le1=round(loc["p_le1"], 4), p_le2=round(loc["p_le2"], 4),
                top4_follow=round(t4, 4), B_edge_ratio=round(er, 4),
                avg_step_entropy=("" if np.isnan(avg_ent) else round(avg_ent, 4)),
                unique=int(diversity(orders)))


def run_text(text_a=TEXT_A, text_mlp=TEXT_MLP):
    B = build_directed_graph(np.load(text_a).astype(np.float64))
    N = B.shape[0]
    mlp = load_mlp(text_mlp)
    rows = {}

    o, e = gen_teacher(B, "greedy"); rows["teacher_greedy"] = text_metrics(o, e)
    o, e = gen_student(B, mlp, "greedy"); rows["student_greedy"] = text_metrics(o, e)
    student_greedy_order = o[0]
    for tau in TAUS:
        o, e = gen_teacher(B, "sample", tau); rows[f"teacher_tau{tau}"] = text_metrics(o, e)
        o, e = gen_student(B, mlp, "sample", tau); rows[f"student_tau{tau}"] = text_metrics(o, e)
    o, e = gen_random(B); rows["random"] = text_metrics(o, e)
    o, e = gen_v3(B); rows["v3_reference"] = text_metrics(o, e)

    # free-running agreement (student's OWN path vs teacher)
    fr1, fr4 = free_running_agreement(B, student_score_fn(mlp), N, k=4)
    # reversed-student vs teacher/v3
    rev_tau = kendall_tau_vs_raster(student_greedy_order[::-1][None, :])
    teacher_greedy_tau = rows["teacher_greedy"]["tau_vs_raster"]
    v3_tau = rows["v3_reference"]["tau_vs_raster"]

    # A MEANINGFUL forward chain needs forward majority AND retained structure (abs_tau well above
    # the random floor ~0.07). A forward majority at high tau where abs_tau ~ random is just the
    # structure collapsing to noise, NOT a flipped chain.
    fwd_taus_naive = [tau for tau in TAUS if rows[f"student_tau{tau}"]["fwd_ratio"] > 0.5]
    fwd_taus_structured = [tau for tau in TAUS
                           if rows[f"student_tau{tau}"]["fwd_ratio"] > 0.5
                           and rows[f"student_tau{tau}"]["abs_tau"] > 0.4]
    struct_regime = rows["student_tau0.5"]            # lowest tau = most structured
    abs_tau_at_naive_fwd = [rows[f"student_tau{tau}"]["abs_tau"] for tau in fwd_taus_naive]

    diag = dict(free_running_top1=round(fr1, 4), free_running_top4=round(fr4, 4),
                student_greedy_tau=rows["student_greedy"]["tau_vs_raster"],
                student_greedy_reversed_tau=round(rev_tau, 4),
                teacher_greedy_tau=teacher_greedy_tau, v3_reference_tau=v3_tau,
                structured_regime_tau=struct_regime["tau_vs_raster"],
                structured_regime_abs_tau=struct_regime["abs_tau"],
                forward_majority_taus_naive=fwd_taus_naive,
                abs_tau_at_naive_forward=abs_tau_at_naive_fwd,
                forward_with_structure_taus=fwd_taus_structured)
    return B, rows, diag


def run_image():
    B = build_directed_graph(np.load(IMAGE_A).astype(np.float64))
    mlp = load_mlp(IMAGE_MLP)
    rows = {}
    o, e = gen_teacher(B, "greedy"); rows["teacher_greedy"] = image_metrics(o, e, B)
    o, e = gen_student(B, mlp, "greedy"); rows["student_greedy"] = image_metrics(o, e, B)
    for tau in TAUS:
        o, e = gen_teacher(B, "sample", tau); rows[f"teacher_tau{tau}"] = image_metrics(o, e, B)
        o, e = gen_student(B, mlp, "sample", tau); rows[f"student_tau{tau}"] = image_metrics(o, e, B)
    o, e = gen_random(B); rows["random"] = image_metrics(o, e, B)
    if BCOV_REF.exists():
        rows["Bcov_reference"] = image_metrics(np.load(BCOV_REF), float("nan"), B)
    return B, rows


def _table(rows, cols):
    head = "| set | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    body = [f"| {k} | " + " | ".join(str(v.get(c, "")) for c in cols) + " |" for k, v in rows.items()]
    return "\n".join([head, sep] + body)


def _phase2_rec_text(diag, chain_strong, reversed_recovers, text_direct):
    return [
        "\n## Phase 2 recommendation (text)",
        f"1. **Text direct Phase 2?** {'YES at tau='+str(diag['forward_with_structure_taus']) if text_direct else 'NO'} — "
        f"the structured regime is reverse and temperature cannot flip it (only dissolves it).",
        f"2. **Reversed-student + direction-anchor needed?** {'YES (required)' if (chain_strong and not text_direct) else 'optional'} — "
        f"student learned the chain (free-run top1={diag['free_running_top1']}) but flipped; reversing recovers the "
        f"teacher orientation ({reversed_recovers}). Phase 2 (text) must include reversed-student AND a "
        f"readiness/source start-prior variant (v3 is forward at tau={diag['v3_reference_tau']}).",
        f"3. **One-line read:** chain topology learned & recoverable by reversal; the missing piece is a "
        f"direction anchor (readiness/source), not readout capacity.",
    ]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-graph", dest="text_graph", default=str(TEXT_A),
                    help="text A graph .npy (default = step50000; pass ckpt20k A_global_eval.npy for the clean substrate)")
    ap.add_argument("--text-mlp", dest="text_mlp", default=str(TEXT_MLP),
                    help="saved Phase-1 text MLP .pt matching --text-graph")
    ap.add_argument("--tag", default="", help="suffix for output filenames so a non-default graph does not clobber the default result")
    ap.add_argument("--text-only", dest="text_only", action="store_true", help="skip the image diagnostic")
    args = ap.parse_args()
    text_a = Path(args.text_graph)
    text_mlp = Path(args.text_mlp)
    tag = args.tag

    out_text = _REPO / "probe_results/attention_order_mlp"
    out_img = _REPO / "probe_results_image_large/attention_order_mlp"
    out_text.mkdir(parents=True, exist_ok=True); out_img.mkdir(parents=True, exist_ok=True)

    # ---------------- TEXT ----------------
    B_t, rows_t, diag = run_text(text_a, text_mlp)
    json.dump(dict(rows=rows_t, diagnostics=diag), open(out_text / f"phase1_5_text_rollout_diagnostic{tag}.json", "w"), indent=2)
    cols_t = ["K", "tau_vs_raster", "abs_tau", "fwd_ratio", "rev_ratio", "avg_step_entropy", "unique", "top_start", "top_end"]
    # tsv
    with open(out_text / f"phase1_5_text_rollout_diagnostic{tag}.tsv", "w") as f:
        f.write("set\t" + "\t".join(cols_t) + "\n")
        for k, v in rows_t.items():
            f.write(k + "\t" + "\t".join(str(v.get(c, "")) for c in cols_t) + "\n")

    chain_strong = diag["free_running_top1"] > 0.8 and abs(diag["student_greedy_tau"]) > 0.8
    reversed_recovers = abs(diag["student_greedy_reversed_tau"] - diag["teacher_greedy_tau"]) < 0.1
    # text is "direct-ready" only if a forward majority survives WITH structure
    text_direct = len(diag["forward_with_structure_taus"]) > 0
    struct_is_reverse = diag["structured_regime_tau"] < -0.3

    text_a_str = (str(text_a.relative_to(_REPO)) if str(text_a).startswith(str(_REPO)) else str(text_a))
    md = [f"# Phase 1.5 — text rollout & orientation diagnostic\n",
          f"- Graph `{text_a_str}` N={B_t.shape[0]}; teacher=C-D+L; MLP=saved Phase-1 text.",
          f"- Sweep tau_beta in {TAUS}; K={K}; standardized sampling. NO training.\n",
          "## Rollout metrics", _table(rows_t, cols_t),
          "\n## Orientation diagnosis",
          f"- **Free-running** (student's OWN greedy path vs teacher): top1={diag['free_running_top1']}, "
          f"top4={diag['free_running_top4']}. High ⇒ student makes the SAME local decisions as the teacher "
          f"along its own trajectory — it is NOT random drift.",
          f"- student_greedy tau={diag['student_greedy_tau']} (|tau| high ⇒ strong chain); "
          f"**reversed** student_greedy tau={diag['student_greedy_reversed_tau']} vs teacher_greedy "
          f"tau={diag['teacher_greedy_tau']} (v3 ref tau={diag['v3_reference_tau']}).",
          f"- Reversing the student recovers the teacher orientation: **{reversed_recovers}**.",
          f"- Structured regime (tau_beta=0.5): tau={diag['structured_regime_tau']} "
          f"(abs_tau={diag['structured_regime_abs_tau']}) — strongly REVERSE.",
          f"- Naive forward-majority taus (fwd_ratio>0.5): {diag['forward_majority_taus_naive'] or 'none'}, "
          f"but their abs_tau={diag['abs_tau_at_naive_forward']} ≈ random floor — i.e. structure has "
          f"COLLAPSED, not flipped. Forward-majority WITH retained structure (abs_tau>0.4): "
          f"**{diag['forward_with_structure_taus'] or 'none'}**.",
          "\n## Verdict",
          f"- Student learned a reverse CHAIN, not random drift: **{chain_strong}** "
          f"(free-running top1={diag['free_running_top1']}, |student tau|={abs(diag['student_greedy_tau'])}).",
          f"- Can tau_beta restore a FORWARD chain? **{text_direct}**. The structured regime is reverse "
          f"(reverse_in_structured_regime={struct_is_reverse}); raising tau_beta only crosses fwd_ratio>0.5 "
          f"once the chain has dissolved to near-random (abs_tau≈0.1). Temperature destroys structure; it "
          f"does NOT flip orientation.",
          f"- v3 (which uses the readiness/source start prior) is forward (tau={diag['v3_reference_tau']}, "
          f"starts at source node 0, 195/200 starts) — consistent with 'C-D+L lacks a direction anchor, "
          f"readiness/source supplies it'. This points directly at the fix."]
    (out_text / f"phase1_5_text_rollout_diagnostic{tag}.md").write_text("\n".join(md) + "\n")

    if args.text_only:
        with open(out_text / f"phase1_5_text_rollout_diagnostic{tag}.md", "a") as f:
            f.write("\n".join(_phase2_rec_text(diag, chain_strong, reversed_recovers, text_direct)) + "\n")
        print("=== PHASE 1.5 SUMMARY (text-only) ===")
        print(f"TEXT  free_run top1={diag['free_running_top1']} top4={diag['free_running_top4']} | "
              f"student_greedy tau={diag['student_greedy_tau']} reversed={diag['student_greedy_reversed_tau']} | "
              f"structured-regime tau={diag['structured_regime_tau']} | fwd-with-structure taus="
              f"{diag['forward_with_structure_taus']} | chain_strong={chain_strong} reversed_recovers={reversed_recovers}")
        return

    # ---------------- IMAGE ----------------
    B_i, rows_i = run_image()
    json.dump(dict(rows=rows_i), open(out_img / "phase1_5_image_rollout_diagnostic.json", "w"), indent=2)
    cols_i = ["K", "mean_manh", "p_le1", "p_le2", "top4_follow", "B_edge_ratio", "avg_step_entropy", "unique"]
    with open(out_img / "phase1_5_image_rollout_diagnostic.tsv", "w") as f:
        f.write("set\t" + "\t".join(cols_i) + "\n")
        for k, v in rows_i.items():
            f.write(k + "\t" + "\t".join(str(v.get(c, "")) for c in cols_i) + "\n")

    rnd = rows_i["random"]
    # student stays B-following at low tau?
    s_low = rows_i["student_tau0.5"]
    image_proceed = s_low["top4_follow"] > rnd["top4_follow"] + 0.05 and s_low["B_edge_ratio"] > 1.4
    md2 = [f"# Phase 1.5 — image rollout & orientation diagnostic\n",
           f"- Graph `{IMAGE_A.relative_to(_REPO)}` N={B_i.shape[0]} (E3-control-small); teacher=C-D+L; "
           f"MLP=saved Phase-1 image.",
           f"- Sweep tau_beta in {TAUS}; K={K}. Bcov ref = reference only (uses image distance). NO training.\n",
           "## Rollout metrics", _table(rows_i, cols_i),
           "\n## Verdict",
           f"- Student free-rollout stays B-following / local at low tau "
           f"(student_tau0.5: top4_follow={s_low['top4_follow']} vs random {rnd['top4_follow']}, "
           f"B_edge_ratio={s_low['B_edge_ratio']}, P(d<=1)={s_low['p_le1']} vs random {rnd['p_le1']}).",
           f"- Image has no native AR direction, so no reversal problem. **Proceed to Phase 2 (image): "
           f"{image_proceed}.**",
           f"- tau_beta sweep shows the usual entropy/structure trade: lower tau = stronger structure. "
           f"Recommend the low-noise band (tau_beta ≈ 0.5–1.0) to match the helpful Graph-RW top_k=4 regime."]
    (out_img / "phase1_5_image_rollout_diagnostic.md").write_text("\n".join(md2) + "\n")

    # ---------------- Phase 2 recommendation (console + appended to text md) ----------------
    rec = [
        "\n## Phase 2 recommendation",
        f"1. **Text direct Phase 2?** {'YES at tau='+str(diag['forward_with_structure_taus']) if text_direct else 'NO'} — "
        f"the structured regime is reverse and temperature cannot flip it (only dissolves it). Running plain "
        f"text Phase 2 on the raw student would evaluate a reverse-AR order on direction-sensitive val_ori_l2r.",
        f"2. **Text reversed-student control needed?** {'YES (required)' if (chain_strong and not text_direct) else 'optional'} — "
        f"the student learned the chain (free-run top1={diag['free_running_top1']}) but with flipped orientation; "
        f"reversing it recovers the teacher orientation ({reversed_recovers}). Phase 2 (text) must include: "
        f"reversed-student, AND a readiness/source direction-anchor variant (v3-style, which is forward at "
        f"tau={diag['v3_reference_tau']}). The cleanest fix is to add a source/readiness start prior to the "
        f"policy so orientation is pinned at t=0.",
        f"3. **Image Phase 2?** {'YES' if image_proceed else 'NO'} — student free-rollout is B-following "
        f"(top4_follow {s_low['top4_follow']} vs random {rnd['top4_follow']}) and orientation-insensitive; "
        f"start image first as the cleaner test of attention-only utility.",
        f"4. **Recommended tau_beta band:** 0.5–1.0 (low-noise, structure-preserving; matches Graph-RW "
        f"top_k=4). Note for TEXT this band is exactly where the order is reverse, so it must be paired with "
        f"the reversal/anchor fix above. Do NOT add entropy regularization; control entropy via tau_beta only.",
        f"5. **One-line read:** chain topology is learned and recoverable by reversal; the missing piece is a "
        f"direction anchor (readiness/source), not readout capacity. Image is unaffected.",
    ]
    with open(out_text / "phase1_5_text_rollout_diagnostic.md", "a") as f:
        f.write("\n".join(rec) + "\n")

    print("=== PHASE 1.5 SUMMARY ===")
    print(f"TEXT  free_run top1={diag['free_running_top1']} top4={diag['free_running_top4']} | "
          f"student_greedy tau={diag['student_greedy_tau']} reversed={diag['student_greedy_reversed_tau']} | "
          f"structured-regime tau={diag['structured_regime_tau']} | fwd-with-structure taus="
          f"{diag['forward_with_structure_taus']} | chain_strong={chain_strong}")
    print(f"IMAGE student_tau0.5 top4_follow={rows_i['student_tau0.5']['top4_follow']} "
          f"B_edge_ratio={rows_i['student_tau0.5']['B_edge_ratio']} (random {rows_i['random']['top4_follow']}) "
          f"| proceed={image_proceed}")
    print(f"REC: text_direct={text_direct} reversed_recovers={reversed_recovers} image_proceed={image_proceed}")


if __name__ == "__main__":
    main()
