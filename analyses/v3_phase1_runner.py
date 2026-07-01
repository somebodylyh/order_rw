"""Phase-1 orchestration + trajectory-based two-tier gate.  ``decide_gate()`` is a
pure function (unit-tested); the CLI runs the 4 arms on GPU and writes the gate
report.  Gate is ADVISORY — final go/no-go is a human call, but the report
auto-computes it.

**Do not launch the GPU sweep without explicit user approval.**
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Dict, List

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
BLOCK_ROOT = ROOT / "block_lo_arm_order_network"
for path in (ROOT, BLOCK_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

EPS_LOSS = 0.01  # nat — gate threshold for "catastrophically worse"


def _auc(curve: List[float]) -> float:
    """Trapezoidal area under the loss curve (lower = better)."""
    return float(np.trapz(curve))


def decide_gate(m: dict) -> dict:
    """Pure-function two-tier gate on Phase-1 metrics.

    Parameters
    ----------
    m : dict with keys:
        nan, entropy_ok, orderhead_param_delta,
        loss_joint_group_final, loss_frozen_gbeta_final,
        delta_probe_group_best_joint, delta_probe_group_best_frozen,
        real_beats_controls, loss_curve_joint, loss_curve_frozen

    Returns
    -------
    dict with keys: tier1_pass, tier2_pass, gate ("PASS"|"FAIL"), reasons
    """
    reasons: List[str] = []

    # ── Tier 1 — hard health gate ─────────────────────────────────────────
    tier1 = True

    if m["nan"]:
        tier1 = False
        reasons.append("health: NaN detected in training log")
    if not m["entropy_ok"]:
        tier1 = False
        reasons.append("health: entropy unhealthy (collapsed or exploded)")
    if m["orderhead_param_delta"] <= 0:
        tier1 = False
        reasons.append("health: OrderHead parameters did not move")

    loss_gap = m["loss_joint_group_final"] - m["loss_frozen_gbeta_final"]
    if loss_gap > EPS_LOSS:
        tier1 = False
        reasons.append(
            f"joint_group catastrophically worse than frozen_gbeta "
            f"(Δloss={loss_gap:.4f} > eps={EPS_LOSS})"
        )

    # ── Tier 2 — life-sign gate (any one) ─────────────────────────────────
    mech_life = (
        m["delta_probe_group_best_joint"]
        < m["delta_probe_group_best_frozen"]
        and m["real_beats_controls"]
    )
    payoff_life = (
        m["loss_joint_group_final"] <= m["loss_frozen_gbeta_final"]
        or _auc(m["loss_curve_joint"]) <= _auc(m["loss_curve_frozen"])
    )
    tier2 = bool(mech_life or payoff_life)

    if mech_life:
        reasons.append(
            "mechanism life: group probe improves vs frozen under real-B guard"
        )
    if payoff_life:
        reasons.append(
            "payoff life: fixed-order PPL/AUC matches or beats frozen_gbeta"
        )
    if not tier2:
        reasons.append("no life sign: neither mechanism nor payoff shows improvement")

    gate = "PASS" if (tier1 and tier2) else "FAIL"
    return {
        "tier1_pass": bool(tier1),
        "tier2_pass": tier2,
        "gate": gate,
        "reasons": reasons,
    }


# ── evaluator closure (Task 3 + Task 4) ──────────────────────────────────────


def _build_evaluator(held_chunks, clean_perm, device, *, m=16):
    """Closure over the FIXED held-out chunks/grouping.

    Identical for every arm and checkpoint (Global Constraints: fixed held-out
    grouping).  Called by ``train_arm`` at each eval step.
    """
    import torch
    from analyses.v3_group_probe import group_probe
    from analyses.v3_fixed_order_eval import fixed_order_val_loss

    held_stack = torch.stack(list(held_chunks))

    def evaluator(step: int, model, wrap):
        loss = fixed_order_val_loss(model, list(held_chunks), clean_perm, device)
        gp = group_probe(wrap, held_stack, m=m, seed=0, device=str(device))
        return {
            "val_loss": loss,
            "ppl": float(math.exp(loss)),
            "delta_probe_group": gp["delta_probe_group"],
            "real_beats_controls": gp["real_beats_controls"],
            "tau_to_l2r": gp["tau_to_l2r"],
            "tau_consensus": gp["tau_consensus"],
        }

    return evaluator


# ── runner ───────────────────────────────────────────────────────────────────


def run_phase1(
    arms,
    n_steps,
    eval_steps,
    device,
    *,
    out_root="runs/v3_sweep/phase1_seed123",
    ckpt="runs/handoff_overnight/seed123/ckpt_step10000.pt",
    n_held=32,
    batch_size=64,
    lr_backbone=None,
    lr_orderhead=3e-4,
    tau=1.0,
    beta=3e-3,
):
    """Run the Phase-1 four-arm sweep and write the gate report.

    ``eval_steps`` are **global** steps (e.g. 15000, 20000, …).  The runner
    converts them to 1-based local continuation steps before passing them to
    ``train_arm``.

    Returns the summary dict.  The gate report is written to
    ``<out_root>/phase1_gate_report.{json,md}``.
    """
    import torch
    from analyses.v3_group_trainer import train_arm
    from analyses.p5_utility_controller import load_p5_ckpt

    out = pathlib.Path(out_root)
    out.mkdir(parents=True, exist_ok=True)

    # Fixed held-out chunks + clean_perm — built ONCE and reused for every arm.
    _model, chunks, clean_perm, dev = load_p5_ckpt(ckpt, n_held, device=device)
    held_chunks = [chunks[i] for i in range(n_held)]
    evaluator = _build_evaluator(held_chunks, clean_perm, str(dev), m=16)

    # Convert global eval steps → local continuation steps.
    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    start_step = int(raw.get("global_step", raw.get("iter_num", 10000)))
    del raw
    local_eval_steps: List[int] = []
    for step in sorted(eval_steps):
        local = int(step) - start_step
        if 1 <= local <= n_steps:
            local_eval_steps.append(local)
    if not local_eval_steps:
        raise ValueError(
            f"no eval_steps {eval_steps} fall within local [1, {n_steps}] "
            f"(start_step={start_step})"
        )

    summary: Dict[str, dict] = {}
    for arm in arms:
        res = train_arm(
            ckpt,
            arm,
            n_steps=n_steps,
            batch_size=batch_size,
            device=str(dev),
            out_dir=str(out),
            tag="phase1_seed123",
            eval_steps=local_eval_steps,
            evaluator=evaluator,
            lr_backbone=lr_backbone,
            lr_orderhead=lr_orderhead,
            tau=tau,
            beta=beta,
        )
        summary[arm] = res

    # Assemble gate metrics from frozen_gbeta (baseline) + joint_group (main).
    fz = summary.get("frozen_gbeta")
    jg = summary.get("joint_group")
    if fz is not None and jg is not None:
        jg_ev, fz_ev = jg.get("evals", []), fz.get("evals", [])
        if not jg_ev or not fz_ev:
            raise RuntimeError("eval steps produced no evaluator output")

        entropy_ok = all(
            np.isfinite(x["entropy"]) for x in jg["log"]
        ) and jg["log"][-1]["entropy"] > 0.0

        metrics = {
            "nan": jg["nan"],
            "entropy_ok": entropy_ok,
            "orderhead_param_delta": jg["orderhead_param_delta"],
            "loss_joint_group_final": jg_ev[-1]["val_loss"],
            "loss_frozen_gbeta_final": fz_ev[-1]["val_loss"],
            "delta_probe_group_best_joint": min(
                e["delta_probe_group"] for e in jg_ev
            ),
            "delta_probe_group_best_frozen": min(
                e["delta_probe_group"] for e in fz_ev
            ),
            "real_beats_controls": any(e["real_beats_controls"] for e in jg_ev),
            "loss_curve_joint": [e["val_loss"] for e in jg_ev],
            "loss_curve_frozen": [e["val_loss"] for e in fz_ev],
        }
        gate = decide_gate(metrics)

        (out / "phase1_gate_report.json").write_text(
            json.dumps({"gate": gate, "metrics": metrics}, indent=2, default=float),
            encoding="utf-8",
        )
        (out / "phase1_gate_report.md").write_text(
            f"# Phase-1 gate: {gate['gate']}\n\n"
            f"Tier 1 (health): {'PASS' if gate['tier1_pass'] else 'FAIL'}\n"
            f"Tier 2 (life-sign): {'PASS' if gate['tier2_pass'] else 'FAIL'}\n\n"
            f"## Reasons\n\n"
            + "\n".join(f"- {r}" for r in gate["reasons"])
            + "\n\n## Metrics\n\n"
            + f"```json\n{json.dumps(metrics, indent=2, default=float)}\n```\n",
            encoding="utf-8",
        )

    (out / "phase1_summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8"
    )
    return summary


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="V3 Phase-1 four-arm sweep")
    ap.add_argument(
        "--arms", nargs="+",
        default=["l2r", "frozen_gbeta", "joint_group", "joint_batch"],
    )
    ap.add_argument("--n-steps", type=int, default=20000)
    ap.add_argument(
        "--eval-steps", nargs="+", type=int,
        default=[15000, 20000, 25000, 30000],
        help="Global steps at which to evaluate (converted to local continuation steps)",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--ckpt", default="runs/handoff_overnight/seed123/ckpt_step10000.pt")
    ap.add_argument("--out-root", default="runs/v3_sweep/phase1_seed123")
    ap.add_argument("--n-held", type=int, default=32)
    ap.add_argument("--lr-backbone", type=float, default=None)
    ap.add_argument("--lr-orderhead", type=float, default=3e-4)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=3e-3)
    a = ap.parse_args()
    run_phase1(
        a.arms, a.n_steps, a.eval_steps, a.device,
        out_root=a.out_root, ckpt=a.ckpt, n_held=a.n_held,
        batch_size=a.batch_size,
        lr_backbone=a.lr_backbone, lr_orderhead=a.lr_orderhead,
        tau=a.tau, beta=a.beta,
    )
