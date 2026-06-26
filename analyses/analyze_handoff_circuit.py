"""Offline analysis of handoff-circuit trajectories: flow + handoff timing."""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np

_STEP_RE = re.compile(r"step_(\d+)")


def load_trajectory(run_dir) -> dict:
    raw = Path(run_dir) / "raw"
    step_dirs = sorted(raw.glob("step_*"), key=lambda p: int(_STEP_RE.search(p.name).group(1)))
    steps, taus, comps = [], [], {}
    methods = None
    for d in step_dirs:
        step = int(_STEP_RE.search(d.name).group(1))
        tt = np.load(d / "tau_table.npz", allow_pickle=True)
        steps.append(step)
        taus.append(tt["tau"])
        methods = list(tt["methods"])
        cpath = d / "composition.npz"
        if cpath.exists():
            cz = np.load(cpath)
            for key in cz.files:
                i, j = map(int, key.split("_"))
                comps.setdefault((i, j), []).append(cz[key])
    return {
        "steps": np.array(steps),
        "tau": np.stack(taus, axis=0),
        "composition": {k: np.stack(v, axis=0) for k, v in comps.items()},
        "methods": methods,
    }


def handoff_timing(traj: dict) -> dict:
    tau = traj["tau"]            # (T,L,H,M)
    steps = traj["steps"]
    max_abs = np.abs(tau).max(axis=(2, 3))   # (T,L)
    peak_idx = max_abs.argmax(axis=0)         # (L,)
    layer_peak_step = steps[peak_idx]
    edge_peak_step = {}
    for (i, j), arr in traj["composition"].items():  # (T,H,H,3)
        edge_peak_step[(i, j)] = int(steps[arr.max(axis=(1, 2, 3)).argmax()])
    return {"layer_peak_step": layer_peak_step, "edge_peak_step": edge_peak_step}


def main():
    import argparse, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    traj = load_trajectory(args.run_dir)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tau = traj["tau"]; steps = traj["steps"]
    for label, data in [("max_abs", np.abs(tau).max(axis=2)),
                        ("max_signed", tau.max(axis=2))]:
        for mi, method in enumerate(traj["methods"]):
            fig, axp = plt.subplots(figsize=(8, 3))
            im = axp.imshow(data[:, :, mi].T, aspect="auto", origin="lower",
                            extent=[steps[0], steps[-1], -0.5, tau.shape[1] - 0.5],
                            cmap="RdBu_r", vmin=-1, vmax=1)
            axp.set_xlabel("step"); axp.set_ylabel("layer")
            axp.set_title(f"{label} tau ({method})")
            fig.colorbar(im, ax=axp)
            fig.savefig(out / f"flow_{label}_{method}.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
    print(f"wrote flow heatmaps to {out}")


if __name__ == "__main__":
    main()
