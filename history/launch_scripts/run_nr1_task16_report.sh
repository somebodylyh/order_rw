#!/bin/bash
# NR-1 Task 16: aggregate all results into REPORT.md per spec §9/§10.
set -euo pipefail

cd /home/admin/lyuyuhuan/order_lyu

OUT_DIR="analyses/neural_readout_nr1_2026-05-28"
mkdir -p "${OUT_DIR}"

python - <<'PY'
import json, pathlib, torch

OUT = pathlib.Path("analyses/neural_readout_nr1_2026-05-28")
CKPT_DIR = pathlib.Path("block_lo_arm_order_network/neural_readout/checkpoints")
DATA_DIR = pathlib.Path("block_lo_arm_order_network/neural_readout/data")


def load_json(p):
    return json.loads(pathlib.Path(p).read_text()) if pathlib.Path(p).exists() else None

def load_ckpt_metrics(p):
    if not pathlib.Path(p).exists():
        return None
    s = torch.load(p, map_location="cpu", weights_only=False)
    return s["metrics"]


def fmt_float(x, n=4):
    return "n/a" if x is None else f"{x:.{n}f}"

def fmt_bool(b):
    return "PASS" if b else "FAIL"


# Load everything
full_metrics = load_ckpt_metrics(CKPT_DIR / "full_10k" / "g_beta_best.pt")
gate_json = load_json(CKPT_DIR / "full_10k" / "hard_gate_check.json") or {}
nll_diag = load_json(CKPT_DIR / "full_10k" / "frozen_nll_diag.json") or {}
diversity_train = load_json(DATA_DIR / "text_5k_full_10k.diversity.json") or {}
ablation_summary = load_json(CKPT_DIR / "ablation_summary.json") or {}
cross_ckpt = load_json(CKPT_DIR / "full_10k" / "cross_ckpt_eval.json") or {}

# Hard gates
tau   = (full_metrics or {}).get("kendall_tau")
pair  = (full_metrics or {}).get("pairwise_precedence_acc")
spear = (full_metrics or {}).get("spearman_rho")
gate_tau   = tau is not None and tau >= 0.80
gate_pair  = pair is not None and pair >= 0.90
gate_spear = spear is not None and spear >= 0.85
all_gates_pass = gate_tau and gate_pair and gate_spear

# Honest claim per §10
if all_gates_pass:
    nll_gap = (nll_diag or {}).get("nll_gap")
    if nll_gap is not None and abs(nll_gap) > 0.02:
        verdict = (
            "**Gates PASS but frozen-θ NLL gap is large (>0.02).** Per spec §10: "
            "\"Imitation succeeds, utility is not guaranteed.\" The NLL gap is the empirical wedge "
            "that motivates NR-5 (frozen plug-in utility) and NR-7 (NLL/ranking fine-tune). "
            "Do not retrain g_β to chase NLL — that would silently turn NR-1 into NR-7 and "
            "break the supervision boundary in §1."
        )
    else:
        verdict = (
            "**Gates PASS.** Per spec §10: A small graph-attention readout, with no rollout state, "
            "recovers the order structure encoded by the source-start C-D+L teacher from raw "
            "per-sample attention graphs at the 5k checkpoint of a random-permutation-trained "
            "any-order text model. The hand-designed attention-to-order mapping can be amortized "
            "into a single neural pass. This does **not** yet show that learned readouts beat "
            "hand readouts, nor that the imitated order is NLL-optimal — those are NR-5 and NR-7 "
            "questions."
        )
else:
    verdict = (
        "**Gates FAILED.** Per spec §10: reduce scope (smaller M for sanity), or relax the "
        "readout to include B_global as an input (Strategy B fallback), or escalate to NR-3 "
        "architecture variants. Do not redefine the gates after seeing the numbers, and in "
        "particular do not add NLL into the gates."
    )

# Compose report
lines = []
lines.append("# NR-1 Neural Attention-to-Order Readout — Final Report")
lines.append("")
lines.append(f"**Date:** 2026-05-28")
lines.append("**Spec:** `docs/superpowers/specs/2026-05-28-neural-attention-to-order-readout-nr1-design.md`")
lines.append("**Plan:** `docs/superpowers/plans/2026-05-28-neural-attention-to-order-readout-nr1.md`")
lines.append("")
lines.append("---")
lines.append("")
lines.append("## 1. §5.1 hard gates (same-ckpt held-out val, full 10k)")
lines.append("")
lines.append("| Metric | Value | Threshold | Verdict |")
lines.append("|---|---|---|---|")
lines.append(f"| Kendall τ                  | {fmt_float(tau)}   | ≥ 0.80 | {fmt_bool(gate_tau)} |")
lines.append(f"| Pairwise precedence acc    | {fmt_float(pair)}  | ≥ 0.90 | {fmt_bool(gate_pair)} |")
lines.append(f"| Spearman ρ                 | {fmt_float(spear)} | ≥ 0.85 | {fmt_bool(gate_spear)} |")
lines.append("")
lines.append(f"**NR-1 §5.1 hard gates: {fmt_bool(all_gates_pass)}**")
lines.append("")

# Teacher diversity
lines.append("## 2. §5.1b teacher diversity diagnostic (full 10k training set)")
lines.append("")
lines.append("| Stat | Value |")
lines.append("|---|---|")
lines.append(f"| unique_sigma_count            | {diversity_train.get('unique_sigma_count', 'n/a')} |")
lines.append(f"| first_node_entropy (nats)     | {fmt_float(diversity_train.get('first_node_entropy'))} |")
lines.append(f"| distinct_first3_prefix_count  | {diversity_train.get('distinct_first3_prefix_count', 'n/a')} |")
lines.append(f"| mean_pairwise_tau among σ_T   | {fmt_float(diversity_train.get('mean_pairwise_tau'))} |")
lines.append("")
mean_tau = diversity_train.get("mean_pairwise_tau")
if mean_tau is not None and mean_tau >= 0.9:
    lines.append("**Interpretation (per §5.1b):** teacher diversity is LOW (mean pairwise τ ≥ 0.9 across samples). "
                 "A high student τ in this regime would mean the student learned a global attention-induced order prior, "
                 "NOT a per-sample attention-to-order mapping.")
elif mean_tau is not None:
    lines.append("**Interpretation (per §5.1b):** teacher diversity is HIGH (mean pairwise τ < 0.9). "
                 "A high student τ in this regime supports the claim that the student learned a per-sample mapping.")
lines.append("")

# Diagnostics
lines.append("## 3. §5.2 diagnostic metrics (NOT pass/fail; reported only)")
lines.append("")
lines.append("| Metric | Value | Notes |")
lines.append("|---|---|---|")
lines.append(f"| top1_first_node_match | {fmt_float((full_metrics or {}).get('top1_first_node_match'))} | inheritance of source-start anchor |")
lines.append(f"| first3_set_match      | {fmt_float((full_metrics or {}).get('first3_set_match'))}      | early-order quality |")
lines.append(f"| frozen-θ NLL teacher  | {fmt_float(nll_diag.get('nll_teacher'))}  | mean per-token NLL under σ_T |")
lines.append(f"| frozen-θ NLL student  | {fmt_float(nll_diag.get('nll_student'))}  | mean per-token NLL under σ̂ |")
lines.append(f"| **frozen-θ NLL gap**  | {fmt_float(nll_diag.get('nll_gap'))}      | σ̂ - σ_T; **diagnostic only, never selects g_β** |")
lines.append("")

# Cross-ckpt
lines.append("## 4. Cross-ckpt diagnostic (NR-4 generalization; NOT NR-1 gate)")
lines.append("")
if cross_ckpt:
    lines.append("| ckpt step | τ | pairwise acc | Spearman ρ | top1 | unique σ | first-node entropy | mean pairwise τ |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for step in sorted(cross_ckpt.keys(), key=int):
        e = cross_ckpt[step]
        m = e["metrics"]
        d = e["teacher_diversity"]
        lines.append(
            f"| {step} | {fmt_float(m['kendall_tau'])} | {fmt_float(m['pairwise_precedence_acc'])} | "
            f"{fmt_float(m['spearman_rho'])} | {fmt_float(m['top1_first_node_match'])} | "
            f"{d['unique_sigma_count']} | {fmt_float(d['first_node_entropy'])} | "
            f"{fmt_float(d['mean_pairwise_tau'])} |"
        )
else:
    lines.append("_(cross-ckpt eval not produced)_")
lines.append("")

# Ablations
lines.append("## 5. §6 MVP ablations (input transforms; same training recipe)")
lines.append("")
if ablation_summary:
    lines.append("| Ablation     | τ | pairwise acc | Spearman ρ |")
    lines.append("|---|---|---|---|")
    for ab in ["identity", "reverse", "sym", "row_shuffle", "b_global"]:
        m = ablation_summary.get(ab)
        if m is None:
            lines.append(f"| {ab} | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {ab} | {fmt_float(m['kendall_tau'])} | "
            f"{fmt_float(m['pairwise_precedence_acc'])} | {fmt_float(m['spearman_rho'])} |"
        )
else:
    lines.append("_(ablation summary not produced)_")
lines.append("")
lines.append("**Per §6 rubric (text substrate):**")
lines.append("- `reverse`: τ should flip sign if the model uses edge direction.")
lines.append("- `sym`: τ should drop significantly if directedness matters.")
lines.append("- `row_shuffle`: text load-bearing signal is in-out readiness, not pairwise topology — τ may barely drop.")
lines.append("- `b_global`: if τ ≥ 0.80 from B_global alone, static slot prior dominates → per-sample dynamic claim weakens.")
lines.append("")

# Verdict
lines.append("## 6. §9 / §10 interpretation")
lines.append("")
lines.append(verdict)
lines.append("")

# Footer
lines.append("---")
lines.append("")
lines.append("**Supervision boundary reminder (spec §1):** no NLL was used as a training signal, label, or hard gate. "
             "The frozen-θ NLL number above is a §5.2 diagnostic and did NOT participate in g_β selection (enforced by "
             "`tests/test_neural_readout_selection_policy.py`).")

(OUT / "REPORT.md").write_text("\n".join(lines))
print(f"wrote {OUT / 'REPORT.md'}")
PY

echo "=== Task 16 DONE ==="
