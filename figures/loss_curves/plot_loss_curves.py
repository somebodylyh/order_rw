"""Plot text + image loss curves for the advisor meeting.

text: train_loss (col=train_loss) for 3 arms — random / α=0.9 / α=1
image: train_loss for 4 arms — fixed_random / fixed_raster / α=0.9 / α=1
y-axis upper limit 4 on text (user request); image needs its own range (NLL ~7+).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

REPO = Path("/home/admin/lyuyuhuan/order_lyu")
OUT = REPO / "figures/loss_curves"
OUT.mkdir(parents=True, exist_ok=True)


def load_tsv(path: Path, col: str, max_step: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    rows = path.read_text().strip().splitlines()
    header = rows[0].split("\t")
    idx_step = header.index("step")
    idx_y = header.index(col)
    steps, ys = [], []
    for line in rows[1:]:
        parts = line.split("\t")
        try:
            s = int(parts[idx_step])
            v = float(parts[idx_y])
        except (ValueError, IndexError):
            continue
        if np.isnan(v):
            continue
        if max_step is not None and s > max_step:
            continue
        steps.append(s)
        ys.append(v)
    return np.array(steps), np.array(ys)


# ---------- TEXT ----------
text_paths = {
    "random (clean_base)": REPO / "block_lo_arm_order_network/probe_results/clean_base_random_perm/eval_curve.tsv",
    "alternating MLP, α=0.9": REPO / "probe_results/attention_order_mlp/alt_from0_mlp_finetune/eval_curve.tsv",
    "alternating MLP, α=1.0": REPO / "probe_results/attention_order_mlp/alt_from0_mlp_alpha1/eval_curve.tsv",
}
text_colors = {"random (clean_base)": "#888888",
               "alternating MLP, α=0.9": "#1f77b4",
               "alternating MLP, α=1.0": "#d62728"}

fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=140)
for name, p in text_paths.items():
    xs, ys = load_tsv(p, "val_ori_l2r_block")
    ax.plot(xs, ys, label=name, color=text_colors[name], linewidth=1.8)
ax.set_xlabel("training step")
ax.set_ylabel("val loss on canonical L2R order (NLL, nats)")
ax.set_title("Text — loss curve (from-0 alternating, attention-only MLP)")
ax.set_ylim(3.3, 4.0)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", frameon=True)
fig.tight_layout()
fig.savefig(OUT / "text_loss_curve.png", dpi=140)
plt.close(fig)
print("Saved", OUT / "text_loss_curve.png")


# ---------- IMAGE ----------
# Each arm uses the val on its own training order:
#   fixed random  -> val_random
#   fixed raster  -> val_raster
#   alt (α=0.9/1) -> val_mlp_order (the order the alt model was trained on)
image_specs = [
    ("fixed random",            REPO / "probe_results_image/vq64_fixed_random_l8h8e512/eval_curve.tsv",                       "val_random",    "#888888"),
    ("fixed raster",            REPO / "probe_results_image/vq64_fixed_raster_l8h8e512/eval_curve.tsv",                       "val_raster",    "#2ca02c"),
    ("alternating MLP, α=0.9",  REPO / "probe_results_image/vq64_alt_from0_mlp_patch2x2_l8h8e512/eval_curve.tsv",              "val_mlp_order", "#1f77b4"),
    ("alternating MLP, α=1.0",  REPO / "probe_results_image/vq64_alt_from0_mlp_patch2x2_l8h8e512_alpha1/eval_curve.tsv",       "val_mlp_order", "#d62728"),
]


def ema(ys: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    out = np.empty_like(ys)
    out[0] = ys[0]
    for i in range(1, len(ys)):
        out[i] = alpha * ys[i] + (1 - alpha) * out[i - 1]
    return out


fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=140)
for name, p, col, color in image_specs:
    xs, ys = load_tsv(p, col)
    ys_smooth = ema(ys, alpha=0.35)
    ax.plot(xs, ys_smooth, label=name, color=color, linewidth=2.0)
ax.set_xlabel("training step")
ax.set_ylabel("val loss on each arm's training order (NLL, nats)")
ax.set_title("Image (ImageNet64 VQ-f4, l8h8e512 patch2×2) — loss curve")
ax.set_ylim(6.8, 8.5)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", frameon=True)
fig.tight_layout()
fig.savefig(OUT / "image_loss_curve.png", dpi=140)
plt.close(fig)
print("Saved", OUT / "image_loss_curve.png")
