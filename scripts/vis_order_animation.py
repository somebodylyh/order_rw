#!/usr/bin/env python3
"""Generate HTML animation of MLP order selection walking through the 8×8 block grid.

Usage:
  python scripts/vis_order_animation.py \
    --a-global probe_results_image/vq64_alt_from0_mlp_patch2x2/A_global_step3000.npy \
    --beta probe_results_image/vq64_alt_from0_mlp_patch2x2/beta_step3000.pt \
    --output /tmp/order_anim.html
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "block_lo_arm_order_network"))
sys.path.insert(0, str(_REPO / "nanogpt-learned-order"))
sys.path.insert(0, str(_REPO / "scripts"))

from train_attn_order_mlp import OrderMLP
from attn_order_mlp_policy import sample_orders_batched_mlp
from attn_order_teacher import rollout_order

GRID = 8
N_BLOCKS = 64


def sample_orders(B, mlp, n_samples, device="cpu", tau=0.5, top_k=4):
    orders_np = sample_orders_batched_mlp(
        B, n_samples, mlp, "original", base_seed=42,
        device=torch.device(device), tau=tau, top_k=top_k,
    ).numpy()
    return orders_np


def sample_teacher(B, n_samples, tau=0.5):
    return np.stack([
        rollout_order(B, tau_T=tau, seed=s, mode="C-D+L", standardize=True)
        for s in range(n_samples)
    ])


def order_to_coords(order):
    return [(int(i) // GRID, int(i) % GRID) for i in order]


def build_html(orders, B, title="MLP Order Animation"):
    """Generate self-contained HTML with animated order visualization."""
    orders_json = json.dumps([order_to_coords(o) for o in orders])
    # B edge weights for display (normalized)
    B_disp = B.copy()
    np.fill_diagonal(B_disp, 0)
    B_max = np.abs(B_disp).max()
    B_norm = (B_disp / B_max).tolist() if B_max > 0 else B_disp.tolist()

    html = f'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #f0f4f3; color: #2d4a3e;
       display: flex; flex-direction: column; align-items: center; padding: 20px; }}
h1 {{ font-size: 1.3em; margin-bottom: 8px; color: #1a6b4a; }}
.controls {{ display: flex; gap: 16px; align-items: center; margin: 12px 0; flex-wrap: wrap; justify-content: center; }}
.controls button {{ background: #e8f5ee; color: #1a6b4a; border: 1px solid #b7d7c5; padding: 8px 18px;
                    border-radius: 6px; cursor: pointer; font-size: 14px; transition: all .15s; }}
.controls button:hover {{ background: #c8e6d9; border-color: #5b9a7a; }}
.controls button.active {{ background: #5b9a7a; color: #fff; border-color: #3d7a58; }}
.controls button:disabled {{ opacity: 0.4; cursor: default; }}
.info {{ display: flex; gap: 24px; margin: 8px 0; font-size: 14px; color: #5b7a6a; }}
.info span {{ color: #1a6b4a; font-weight: 600; }}
#grid-container {{ position: relative; margin: 12px 0; }}
canvas {{ border: 1px solid #c8dcd0; border-radius: 8px; background: #fafdfb; }}
.slider-row {{ display: flex; gap: 8px; align-items: center; margin: 4px 0; }}
.slider-row label {{ font-size: 13px; color: #6b8a78; }}
input[type=range] {{ width: 200px; accent-color: #3d8b6e; }}
#step-slider {{ width: 520px; }}
.legend {{ font-size: 12px; color: #8aaa98; margin-top: 4px; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="info">
  <div>样本 <span id="sample-num">1</span> / <span id="total-samples">{len(orders)}</span></div>
  <div>步数 <span id="step-num">0</span> / 64</div>
  <div>曼哈顿累积 <span id="manh-sum">0</span></div>
  <div>距中心 <span id="center-dist">-</span></div>
</div>
<div class="controls">
  <button id="btn-prev" onclick="prevSample()" title="上一个样本">◀ 上一样本</button>
  <button id="btn-play" onclick="togglePlay()">▶ 播放</button>
  <button id="btn-next" onclick="nextSample()" title="下一个样本">下一样本 ▶</button>
  <button id="btn-reset" onclick="resetAnim()">↺ 重置</button>
</div>
<div>
  <div class="slider-row">
    <label>步:</label>
    <input type="range" id="step-slider" min="0" max="64" value="0" oninput="jumpTo(this.value)">
  </div>
  <div class="slider-row">
    <label>速:</label>
    <input type="range" id="speed-slider" min="50" max="1000" value="200" oninput="setSpeed(this.value)">
    <span id="speed-label" style="font-size:12px;color:#aaa">200ms</span>
  </div>
</div>
<div id="grid-container">
  <canvas id="grid" width="560" height="560"></canvas>
</div>
<div class="legend">
  方块颜色: 浅蓝=未访问 | 橙→红=已访问(越早越暖) | 亮黄=当前 | 箭头=方向 | 透明度=B边权
</div>

<script>
const GRID = 8, BLOCK = 64, GAP = 4;
const ALL_ORDERS = {orders_json};
const B_NORM = {json.dumps(B_norm)};
let currentSample = 0;
let currentStep = 0;
let playing = false;
let playTimer = null;
let speedMs = 200;

const canvas = document.getElementById('grid');
const ctx = canvas.getContext('2d');

function drawGrid() {{
    const order = ALL_ORDERS[currentSample];
    const total = order.length;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Precompute positions
    const pos = [];
    for (let r = 0; r < GRID; r++) {{
        for (let c = 0; c < GRID; c++) {{
            pos.push({{r, c, x: GAP + c * BLOCK, y: GAP + r * BLOCK}});
        }}
    }}

    // Draw B edges as faint background lines
    const stepLimit = Math.min(currentStep, total);
    const visited = new Set();
    for (let s = 0; s < stepLimit; s++) {{
        const blockIdx = order[s][0] * GRID + order[s][1];
        visited.add(blockIdx);
    }}

    // Draw visited blocks
    for (let s = 0; s < stepLimit; s++) {{
        const [r, c] = order[s];
        const blockIdx = r * GRID + c;
        const p = pos[blockIdx];
        const t = s / Math.max(total - 1, 1);
        // Color gradient: cyan (early) → deep teal (late)
        const red = Math.floor(20 + 15 * (1 - t));
        const green = Math.floor(180 - 50 * t);
        const blue = Math.floor(170 - 60 * t);
        ctx.fillStyle = `rgb(${{red}},${{green}},${{blue}})`;
        ctx.fillRect(p.x, p.y, BLOCK - GAP, BLOCK - GAP);
        // Step number
        ctx.fillStyle = t > 0.5 ? '#e8f5ee' : '#0a3d2a';
        ctx.font = '9px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(s + 1, p.x + BLOCK / 2, p.y + BLOCK / 2 + 3);
    }}

    // Highlight current position
    if (currentStep > 0 && currentStep <= total) {{
        const [cr, cc] = order[currentStep - 1];
        const ci = cr * GRID + cc;
        const cp = pos[ci];
        ctx.strokeStyle = '#00c896';
        ctx.lineWidth = 3;
        ctx.strokeRect(cp.x - 1, cp.y - 1, BLOCK - GAP + 2, BLOCK - GAP + 2);
    }}

    // Draw edge arrows between consecutive steps
    if (stepLimit > 1) {{
        ctx.strokeStyle = 'rgba(45,120,90,0.55)';
        ctx.lineWidth = 1.5;
        ctx.fillStyle = '#2d785a';
        for (let s = 0; s < stepLimit - 1; s++) {{
            const [r1, c1] = order[s];
            const [r2, c2] = order[s + 1];
            const p1 = pos[r1 * GRID + c1];
            const p2 = pos[r2 * GRID + c2];
            const cx1 = p1.x + BLOCK / 2, cy1 = p1.y + BLOCK / 2;
            const cx2 = p2.x + BLOCK / 2, cy2 = p2.y + BLOCK / 2;
            ctx.beginPath();
            ctx.moveTo(cx1, cy1);
            ctx.lineTo(cx2, cy2);
            ctx.stroke();
            // Arrowhead
            const dx = cx2 - cx1, dy = cy2 - cy1;
            const len = Math.sqrt(dx * dx + dy * dy);
            if (len > 1) {{
                const ux = dx / len * 8, uy = dy / len * 8;
                const mx = cx2 - ux, my = cy2 - uy;
                ctx.beginPath();
                ctx.moveTo(mx - uy * 0.5, my + ux * 0.5);
                ctx.lineTo(cx2, cy2);
                ctx.lineTo(mx + uy * 0.5, my - ux * 0.5);
                ctx.fill();
            }}
        }}
    }}

    // Draw unvisited blocks (light green outline)
    for (let i = 0; i < N_BLOCKS; i++) {{
        if (!visited.has(i)) {{
            const p = pos[i];
            ctx.strokeStyle = '#c8dcd0';
            ctx.lineWidth = 0.5;
            ctx.strokeRect(p.x, p.y, BLOCK - GAP, BLOCK - GAP);
            // Block index label
            ctx.fillStyle = '#bfd5c8';
            ctx.font = '8px monospace';
            ctx.textAlign = 'center';
            ctx.fillText(i, p.x + BLOCK / 2, p.y + BLOCK / 2 + 2);
        }}
    }}

    // Grid coordinates
    ctx.fillStyle = '#8aaa98';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    for (let r = 0; r < GRID; r++) {{
        ctx.fillText(r, 8, GAP + r * BLOCK + BLOCK / 2 + 4);
    }}
    for (let c = 0; c < GRID; c++) {{
        ctx.fillText(c, GAP + c * BLOCK + BLOCK / 2, 14);
    }}
}}

// Calculate center distance
function centerDist(coord) {{
    return Math.sqrt((coord[0] - 3.5) ** 2 + (coord[1] - 3.5) ** 2);
}}

function updateDisplay() {{
    const order = ALL_ORDERS[currentSample];
    const limit = Math.min(currentStep, order.length);
    let manh = 0;
    for (let s = 1; s < limit; s++) {{
        manh += Math.abs(order[s][0] - order[s - 1][0]) + Math.abs(order[s][1] - order[s - 1][1]);
    }}
    document.getElementById('sample-num').textContent = currentSample + 1;
    document.getElementById('step-num').textContent = limit;
    document.getElementById('manh-sum').textContent = manh;
    const cd = limit > 0 ? centerDist(order[limit - 1]).toFixed(2) : '-';
    document.getElementById('center-dist').textContent = cd;
    document.getElementById('step-slider').value = currentStep;
    drawGrid();
}}

function jumpTo(step) {{
    currentStep = parseInt(step);
    updateDisplay();
}}

function nextSample() {{
    if (currentSample < ALL_ORDERS.length - 1) {{
        currentSample++;
        currentStep = 0;
        updateDisplay();
    }}
}}

function prevSample() {{
    if (currentSample > 0) {{
        currentSample--;
        currentStep = 0;
        updateDisplay();
    }}
}}

function resetAnim() {{
    currentStep = 0;
    updateDisplay();
}}

function togglePlay() {{
    if (playing) {{
        clearInterval(playTimer);
        playing = false;
        document.getElementById('btn-play').textContent = '▶ 播放';
    }} else {{
        if (currentStep >= ALL_ORDERS[currentSample].length) currentStep = 0;
        playing = true;
        document.getElementById('btn-play').textContent = '⏸ 暂停';
        playTimer = setInterval(() => {{
            if (currentStep < ALL_ORDERS[currentSample].length) {{
                currentStep++;
                updateDisplay();
            }} else {{
                clearInterval(playTimer);
                playing = false;
                document.getElementById('btn-play').textContent = '▶ 播放';
            }}
        }}, speedMs);
    }}
}}

function setSpeed(val) {{
    speedMs = parseInt(val);
    document.getElementById('speed-label').textContent = speedMs + 'ms';
    if (playing) {{
        clearInterval(playTimer);
        playTimer = setInterval(() => {{
            if (currentStep < ALL_ORDERS[currentSample].length) {{
                currentStep++;
                updateDisplay();
            }} else {{
                clearInterval(playTimer);
                playing = false;
                document.getElementById('btn-play').textContent = '▶ 播放';
            }}
        }}, speedMs);
    }}
}}

// Keyboard controls
document.addEventListener('keydown', (e) => {{
    if (e.key === 'ArrowRight') jumpTo(Math.min(currentStep + 1, 64));
    else if (e.key === 'ArrowLeft') jumpTo(Math.max(currentStep - 1, 0));
    else if (e.key === ' ') {{ e.preventDefault(); togglePlay(); }}
    else if (e.key === 'n' || e.key === 'N') nextSample();
    else if (e.key === 'p' || e.key === 'P') prevSample();
    else if (e.key === 'r' || e.key === 'R') resetAnim();
}});

// Initial draw
updateDisplay();
</script>
</body>
</html>'''
    return html


def main():
    p = argparse.ArgumentParser(description="Generate HTML order animation")
    p.add_argument("--a-global", required=True, help="A_global_step{N}.npy (saved B.T)")
    p.add_argument("--beta", required=True, help="beta_step{N}.pt MLP checkpoint")
    p.add_argument("--output", default="/tmp/order_anim.html", help="output HTML path")
    p.add_argument("--n-samples", type=int, default=20, help="number of orders to sample")
    p.add_argument("--mode", choices=["mlp", "teacher", "both"], default="mlp")
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--top-k", type=int, default=4)
    args = p.parse_args()

    # Load B (saved as B.T = A, so B = A_global.T)
    A_saved = np.load(args.a_global)
    B = A_saved.T.copy()
    np.fill_diagonal(B, 0.0)

    # Load MLP
    mlp = OrderMLP()
    mlp.load_state_dict(torch.load(args.beta, map_location="cpu"))
    mlp.eval()

    orders = sample_orders(B, mlp, args.n_samples, tau=args.tau, top_k=args.top_k)
    title = f"MLP Order — 2×2 Patch 8×8 Grid ({args.n_samples} samples)"

    html = build_html(orders, B, title=title)
    Path(args.output).write_text(html, encoding="utf-8")
    print(f"Wrote {args.n_samples} orders to {args.output}")
    print(f"Open with:  open {args.output}  or browse to file://{args.output}")


if __name__ == "__main__":
    main()
