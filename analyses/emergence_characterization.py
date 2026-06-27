"""Spec A: characterize order-carrier emergence (diffuse -> pruning) from the
existing every-200-step trajectories.

See docs/superpowers/specs/2026-06-27-emergence-characterization-design.md
and docs/superpowers/plans/2026-06-27-emergence-characterization.md.

After Pillar 3 falsified the cross-layer handoff, the mechanism question becomes:
how is the carrier layer/head set *selected*? The order signal first appears
diffusely (all layers/heads, step 0-1200), then a short winner-take-all pruning
event (~1200-2000) leaves a seed-specific layer-local redundant carrier.
"""
import glob
import pathlib
import sys

import numpy as np

# Make block_lo_arm_order_network modules importable regardless of CWD.
_BLOCK_DIR = pathlib.Path(__file__).resolve().parents[1] / "block_lo_arm_order_network"
if str(_BLOCK_DIR) not in sys.path:
    sys.path.insert(0, str(_BLOCK_DIR))

TRAJ_ROOT = "runs/handoff_overnight"
_METHOD_I = 0  # C-D+L


# ── Task 1: tau-trajectory loader + winner determination ─────────────────────

def load_tau_trajectory(seed, root=TRAJ_ROOT):
    """Stack the 51 per-step tau_table.npz into steps + signed tau[S,4,8] (C-D+L)."""
    pat = f"{root}/seed{seed}/attention_trajectory/raw/step_*/tau_table.npz"
    files = sorted(glob.glob(pat))
    steps, taus = [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        step = int(f.split("step_")[1].split("/")[0])
        steps.append(step)
        taus.append(d["tau"][:, :, _METHOD_I])  # (4,8) signed C-D+L
    steps = np.array(steps, dtype=int)
    tau = np.stack(taus).astype(float)           # (S,4,8)
    order = np.argsort(steps)
    return {"steps": steps[order], "tau": tau[order], "abs_tau": np.abs(tau[order])}


def winner(seed_traj, seed, strong=0.95):
    """Winning layer = argmax_L sum_h|tau| at the final step. Winner set = that
    layer's |tau|>=strong heads, or its frozen weak carrier set if none (seed42)."""
    from analyses.handoff_carrier_config import load_carrier_sets
    final = seed_traj["abs_tau"][-1]             # (4,8)
    winning_layer = int(final.sum(axis=1).argmax())
    strong_heads = [h for h in range(8) if final[winning_layer, h] >= strong]
    if strong_heads:
        return {"winning_layer": winning_layer, "winner_heads": strong_heads,
                "tier": "strong"}
    weak = load_carrier_sets(seed)[winning_layer]["weak"]
    return {"winning_layer": winning_layer, "winner_heads": list(weak), "tier": "weak"}
