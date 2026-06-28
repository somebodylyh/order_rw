"""Plot the canonical 65-node emergence sweep."""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_sweep(sweep_json, out_dir):
    s = json.load(open(sweep_json))
    out = pathlib.Path(out_dir)
    steps = sorted(int(k) for k in s["by_step"])
    best_tau = [s["by_step"][str(k)]["best_tau"] for k in steps]
    n_strong = [s["by_step"][str(k)]["n_strong"] for k in steps]
    floor = [s["by_step"][str(k)]["destroyed_floor"] for k in steps]
    carrier = s["by_step"][str(steps[-1])]["best_head"]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(steps, best_tau, "-o", color="C0", label="best |τ| (C-D+L)")
    ax1.plot(steps, floor, ":", color="gray", label="destroyed floor")
    ax1.set_xlabel("training step")
    ax1.set_ylabel("|τ| vs physical L2R", color="C0")
    ax1.set_ylim(-0.05, 1.05)
    ax2 = ax1.twinx()
    ax2.plot(steps, n_strong, "-s", color="C3", alpha=0.6, label="# strong-pass heads")
    ax2.set_ylabel("# strong-pass (C-D+L)", color="C3")
    ax1.set_title(f"seed{s['seed']} canonical 65-node emergence "
                  f"(step10k carrier L{carrier[0]}H{carrier[1]})")
    ax1.legend(loc="center right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "emergence.png", dpi=120)
    plt.close(fig)


def plot_emergence_curves(curves_json, out_dir):
    """Two panels: per-layer max|tau_physical| vs step, and L0 per-head signed tau vs step."""
    s = json.load(open(curves_json))
    out = pathlib.Path(out_dir)
    steps = sorted(int(k) for k in s["by_step"])
    fig, (axL, axH) = plt.subplots(1, 2, figsize=(12, 4))
    # left: per-layer max|tau|
    for L in range(4):
        axL.plot(steps, [s["by_step"][str(k)]["layer_max"][str(L)] for k in steps],
                 "-o", label=f"L{L} max|τ|")
    axL.plot(steps, [s["by_step"][str(k)]["floor"] for k in steps], ":", color="gray", label="floor")
    axL.set_xlabel("step"); axL.set_ylabel("max |τ_physical| (C-D+L)"); axL.set_ylim(-0.05, 1.05)
    axL.set_title(f"seed{s['seed']} per-layer physical τ"); axL.legend(fontsize=8)
    # right: L0 per-head signed tau
    for h in range(8):
        axH.plot(steps, [s["by_step"][str(k)]["l0_heads"][str(h)] for k in steps],
                 "-", alpha=0.8, label=f"L0H{h}")
    axH.axhline(0, color="k", lw=0.5); axH.set_ylim(-1.05, 1.05)
    axH.set_xlabel("step"); axH.set_ylabel("signed τ_physical")
    axH.set_title(f"seed{s['seed']} L0 per-head (carrier {s['carrier_L0_step10000']})")
    axH.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "physical_emergence.png", dpi=120)
    plt.close(fig)
