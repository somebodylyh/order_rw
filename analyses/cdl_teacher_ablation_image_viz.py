"""Visualize actual teacher traversal paths on the 8×8 grid.

Shows step-by-step order for: argmax, local-greedy, CDL variants, raster.
Numbers = step index (0..63), arrows show direction of traversal.
"""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AOGPT_DIR = os.path.join(_SCRIPT_DIR, "..", "block_lo_arm_order_network")
_REPO = os.path.join(_SCRIPT_DIR, "..")
_NANOGPT_DIR = os.path.join(_REPO, "nanogpt-learned-order")
sys.path.insert(0, _AOGPT_DIR)
sys.path.insert(0, _NANOGPT_DIR)

import torch
from AOGPT import AOGPTConfig, AOGPT
from attn_order_teacher import teacher_components, _softmax

# ═══════════════════════════════════════════════════════════════════
# Setup (same as ablation script)
# ═══════════════════════════════════════════════════════════════════

GRID = 8
N_BLOCKS = 64
TAU_T = 1.0

CKPT_PATH = os.path.join(_REPO, "probe_results_image/vq64_fixed_raster_l8h8e512/ckpt_step30000.pt")
DATA_PATH = os.path.join(_NANOGPT_DIR, "data/Imagenet64VQ_f4_800k_full_rowmajor/val.bin")
OUT_DIR = os.path.join(_SCRIPT_DIR, "cdl_teacher_ablation", "traversal_viz")
DEVICE = "cuda:0"
N_IMAGES = 200
M_PASSES = 3
TOKEN_GRID = 16


def _make_manh_table(grid):
    N = grid * grid
    rows = np.arange(N) // grid
    cols = np.arange(N) % grid
    return (np.abs(rows[:, None] - rows[None, :]) +
            np.abs(cols[:, None] - cols[None, :])).astype(np.int64)


def token_to_patch_indices():
    PATCH_GRID = 8
    patch_idx = np.zeros(TOKEN_GRID * TOKEN_GRID, dtype=np.int64)
    for br in range(PATCH_GRID):
        for bc in range(PATCH_GRID):
            p = br * PATCH_GRID + bc
            i_tl = 2 * br * TOKEN_GRID + 2 * bc
            i_tr = i_tl + 1
            i_bl = i_tl + TOKEN_GRID
            i_br = i_bl + 1
            for tok in (i_tl, i_tr, i_bl, i_br):
                patch_idx[tok] = p
    return patch_idx


def aggregate_token_to_block(A_token):
    assert A_token.shape == (256, 256)
    p_idx = token_to_patch_indices()
    A_block = np.zeros((64, 64), dtype=np.float64)
    counts = np.zeros((64, 64), dtype=np.int64)
    for i in range(256):
        for j in range(256):
            A_block[p_idx[i], p_idx[j]] += A_token[i, j]
            counts[p_idx[i], p_idx[j]] += 1
    A_block /= np.maximum(counts, 1)
    np.fill_diagonal(A_block, 0.0)
    return A_block.astype(np.float64)


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    m_args = ckpt.get("model_args", {})
    cfg = ckpt.get("config", {})
    keys = ["block_size", "vocab_size", "n_layer", "n_head", "n_embd",
            "dropout", "bias", "block_order_block_len", "order_impl"]
    model_args = {}
    for k in keys:
        if k in m_args:
            model_args[k] = m_args[k]
        elif k in cfg:
            model_args[k] = cfg[k]
    model = AOGPT(AOGPTConfig(**model_args))
    sd = ckpt["model"]
    if all(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.crop_block_size(model_args["block_size"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, model_args


def extract_block_B(model, data_tokens, n_images, m_passes, device, target_l=2, target_h=4):
    """Extract block-level B for a specific head."""
    T = model.config.block_size
    n_available = len(data_tokens) // T
    n_images = min(n_images, n_available)

    A_sum = torch.zeros(256, 256, device=device)
    cnt = 0

    for img_idx in range(n_images):
        start = img_idx * T
        tok = torch.from_numpy(
            data_tokens[start:start + T].astype(np.int64)
        ).to(device).unsqueeze(0)

        for _ in range(m_passes):
            rand_order = torch.randperm(T, device=device).unsqueeze(0)
            with torch.no_grad():
                _, _, attn_list = model.forward_fn(tok, rand_order, return_attentions=True)
            attn = attn_list[target_l][0, target_h]          # (T+1, T+1)
            content = attn[1:, 1:]                            # (T, T) model order
            inv = torch.argsort(rand_order[0])
            A_sum += content[inv][:, inv]                     # physical coords
            cnt += 1

        if (img_idx + 1) % 50 == 0:
            print(f"  {img_idx + 1}/{n_images}", flush=True)

    A_token = (A_sum / cnt).float().cpu().numpy()
    A_block = aggregate_token_to_block(A_token.astype(np.float64))
    B_block = A_block.T.copy()
    np.fill_diagonal(B_block, 0.0)
    return B_block


# ═══════════════════════════════════════════════════════════════════
# Order generators
# ═══════════════════════════════════════════════════════════════════

def order_argmax(B):
    """Per-row argmax: for each row i, pick j with max B[i,j]."""
    # This gives a "next" for each position, but not necessarily a valid permutation.
    # We use a greedy sequential: start at 0, always go to argmax of current row among unseen.
    N = B.shape[0]
    Bc = B.copy()
    current = 0
    order = [current]
    seen = {current}
    for _ in range(N - 1):
        row = Bc[current].copy()
        row[list(seen)] = -np.inf
        nxt = int(np.argmax(row))
        order.append(nxt)
        seen.add(nxt)
        current = nxt
    return np.array(order, dtype=np.int64)


def order_local_greedy(B, start=0):
    """Local-greedy: at each step, go to the unseen neighbor with max B[current, j]."""
    N = B.shape[0]
    Bc = B.copy()
    order = [start]
    seen = {start}
    current = start
    for _ in range(N - 1):
        row = Bc[current].copy()
        row[list(seen)] = -np.inf
        nxt = int(np.argmax(row))
        order.append(nxt)
        seen.add(nxt)
        current = nxt
    return np.array(order, dtype=np.int64)


def order_cdl(B, w_c, w_d, w_l, seed=42):
    """CDL teacher rollout (greedy sequential with score = w_c*C − w_d*D + w_l*L)."""
    N = B.shape[0]
    Bn = np.asarray(B, dtype=np.float64)
    rng = np.random.default_rng(seed)
    S, U, last = [], list(range(N)), None
    order = []
    for _ in range(N):
        if len(U) == 1:
            v = U[0]
        else:
            C, D, L, cand = teacher_components(Bn, S, U, last)
            q = w_c * C + w_d * D + w_l * L
            q = (q - q.mean()) / (q.std() + 1e-9)
            p = _softmax(q, TAU_T)
            v = int(cand[rng.choice(len(cand), p=p)])
        order.append(v)
        S.append(v)
        U.remove(v)
        last = v
    return np.array(order, dtype=np.int64)


def order_raster():
    return np.arange(64, dtype=np.int64)


# ═══════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════

def draw_traversal(order, ax, title, manh_table, cmap='plasma'):
    """Draw the traversal path as numbered steps on an 8×8 grid."""
    grid = np.full((GRID, GRID), -1, dtype=int)
    for step, block_idx in enumerate(order):
        r, c = block_idx // GRID, block_idx % GRID
        grid[r, c] = step

    # Color by step index
    im = ax.imshow(grid, cmap=cmap, aspect='equal', vmin=0, vmax=63)

    # Draw step numbers
    for step, block_idx in enumerate(order):
        r, c = block_idx // GRID, block_idx % GRID
        color = 'white' if step > 31 else 'black'
        ax.text(c, r, str(step), ha='center', va='center', fontsize=6,
                color=color, fontweight='bold')

    # Draw arrows between consecutive steps
    prev_r, prev_c = order[0] // GRID, order[0] % GRID
    for step in range(1, len(order)):
        curr_r, curr_c = order[step] // GRID, order[step] % GRID
        dr, dc = curr_r - prev_r, curr_c - prev_c
        dist = manh_table[order[step-1], order[step]]
        # Arrow from prev to curr
        color = 'red' if dist > 2 else ('lime' if dist == 1 else 'yellow')
        alpha = 0.9 if dist > 2 else 0.5
        ax.arrow(prev_c, prev_r, dc * 0.8, dr * 0.8,
                 head_width=0.2, head_length=0.2, fc=color, ec=color,
                 alpha=alpha, linewidth=0.5)
        prev_r, prev_c = curr_r, curr_c

    # Grid lines
    ax.set_xticks(np.arange(-0.5, GRID, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, GRID, 1), minor=True)
    ax.grid(which='minor', color='gray', linewidth=0.5, alpha=0.5)
    ax.set_xticks([]); ax.set_yticks([])

    # Title with metrics
    steps = np.array([manh_table[order[t], order[t+1]] for t in range(len(order)-1)])
    d_manh = steps.mean()
    jumps = (steps > 2).sum()
    long_jumps = (steps > 4).sum()
    ax.set_title(f"{title}\nD_manh={d_manh:.3f}  P(d≤1)={(steps<=1).mean():.3f}  "
                 f"jumps>2={jumps}  long>4={long_jumps}",
                 fontsize=8)


def draw_jump_map(order, ax, title, manh_table):
    """Draw where long jumps occur — highlight the blocks that are jumped FROM."""
    jump_map = np.zeros((GRID, GRID))
    for t in range(len(order) - 1):
        d = manh_table[order[t], order[t+1]]
        if d > 2:
            r, c = order[t] // GRID, order[t] % GRID
            jump_map[r, c] = d

    im = ax.imshow(jump_map, cmap='Reds', aspect='equal', vmin=0, vmax=manh_table.max())
    ax.set_title(f"{title}\n(jump distance > 2)", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    return im


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    manh_table = _make_manh_table(GRID)

    # Load everything
    print("Loading model...", flush=True)
    model, model_args = load_model(CKPT_PATH, DEVICE)
    data = np.memmap(DATA_PATH, dtype=np.uint16, mode="r")

    # Extract B for L2H4 (best) and L0H5 (best CDL performer)
    print("Extracting L2H4 B...", flush=True)
    B_L2H4 = extract_block_B(model, data, N_IMAGES, M_PASSES, DEVICE,
                             target_l=2, target_h=4)
    print("Extracting L0H5 B...", flush=True)
    B_L0H5 = extract_block_B(model, data, N_IMAGES, M_PASSES, DEVICE,
                             target_l=0, target_h=5)

    del model
    torch.cuda.empty_cache()

    # Generate orders for each head
    heads = [("L2H4", B_L2H4), ("L0H5", B_L0H5)]

    for head_name, B in heads:
        print(f"\n{'='*60}")
        print(f"Visualizing {head_name}")
        print(f"{'='*60}")

        # Generate all orders
        orders = {}
        orders["argmax (per-row top-1)"] = order_argmax(B)
        orders["local-greedy (from 0)"] = order_local_greedy(B, start=0)
        orders["local-greedy (from 32)"] = order_local_greedy(B, start=32)
        orders["CDL full (C−D+L)"] = order_cdl(B, 1.0, -1.0, 1.0, seed=42)
        orders["CDL L only"] = order_cdl(B, 0.0, 0.0, 1.0, seed=42)
        orders["CDL no-local (C−D)"] = order_cdl(B, 1.0, -1.0, 0.0, seed=42)
        orders["CDL −D only"] = order_cdl(B, 0.0, -1.0, 0.0, seed=42)
        orders["CDL wrong-D (C+D+L)"] = order_cdl(B, 1.0, 1.0, 1.0, seed=42)
        orders["raster (row-major)"] = order_raster()

        # Print all orders for detailed inspection
        print("\nActual orders:")
        for name, o in orders.items():
            d_manh = float(manh_table[o[:-1], o[1:]].mean())
            steps = manh_table[o[:-1], o[1:]]
            long_jumps = [(t, o[t], o[t+1], manh_table[o[t], o[t+1]])
                          for t in range(len(o)-1) if manh_table[o[t], o[t+1]] > 3]
            print(f"\n  {name}  (D_manh={d_manh:.3f})")
            print(f"  {list(o)}")
            if long_jumps:
                print(f"  LONG JUMPS (>3):")
                for t, fr, to, d in long_jumps:
                    fr_rc = (fr // 8, fr % 8)
                    to_rc = (to // 8, to % 8)
                    print(f"    step {t}: {fr}({fr_rc}) -> {to}({to_rc})  dist={d}")

        # Plot: 3×3 grid of traversals
        plot_orders = [
            "argmax (per-row top-1)",
            "local-greedy (from 0)",
            "raster (row-major)",
            "CDL L only",
            "CDL full (C−D+L)",
            "CDL −D only",
            "CDL no-local (C−D)",
            "CDL wrong-D (C+D+L)",
            "local-greedy (from 32)",
        ]

        fig, axes = plt.subplots(3, 3, figsize=(16, 16))
        for ax, name in zip(axes.flat, plot_orders):
            if name in orders:
                draw_traversal(orders[name], ax, name, manh_table)
        fig.suptitle(f"Traversal Paths — {head_name}\nNumbers = step index, red arrows = jumps > 2",
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"traversal_{head_name}.png")
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {out_path}")

        # Second figure: jump maps (where long jumps originate)
        fig2, axes2 = plt.subplots(3, 3, figsize=(16, 14))
        for ax, name in zip(axes2.flat, plot_orders):
            if name in orders:
                draw_jump_map(orders[name], ax, name, manh_table)
        fig2.suptitle(f"Long-Jump Origins (distance > 2) — {head_name}",
                      fontsize=12, fontweight='bold')
        plt.tight_layout()
        out_path2 = os.path.join(OUT_DIR, f"jumps_{head_name}.png")
        fig2.savefig(out_path2, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f"Saved {out_path2}")

    print(f"\nDone. All outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
