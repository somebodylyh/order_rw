"""Sample MLP orders from alpha=1 step50000 ckpt and emit an HTML visualization.

8x8 grid (N=64 patches, raster-flattened). For each sampled order we render a heatmap
where cell color encodes when the patch was decoded (early=light, late=dark) and the
cell label is the decoding step (1..64).
"""
from __future__ import annotations
import sys
from pathlib import Path
import json
import numpy as np
import torch

REPO = Path("/home/admin/lyuyuhuan/order_lyu")
sys.path.insert(0, str(REPO / "block_lo_arm_order_network"))
sys.path.insert(0, str(REPO / "scripts"))

from train_attn_order_mlp import OrderMLP
import attn_order_mlp_policy as P

RUN_DIR = REPO / "probe_results_image/vq64_alt_from0_mlp_patch2x2_l8h8e512_alpha1"
A_PATH = RUN_DIR / "A_global_step48000.npy"
BETA_PATH = RUN_DIR / "beta_step48000.pt"
OUT_HTML = REPO / "figures/loss_curves/image_orders_alpha1_step50000.html"
OUT_ANIM = REPO / "figures/loss_curves/image_orders_alpha1_step50000_anim.html"
N_SAMPLES = 8           # how many independent orders to render
GRID = 8                # 8x8 patches
TAU = 0.5
TOP_K = 4
SEED_BASE = 20260528

device = "cpu"

# Load B and MLP
B = np.load(A_PATH).T.copy()
np.fill_diagonal(B, 0.0)
B = B.astype(np.float32)
mlp = OrderMLP()
mlp.load_state_dict(torch.load(BETA_PATH, map_location=device, weights_only=False))
mlp.eval()

# Sample N orders. sample_orders_batched_mlp returns LongTensor [bs, N]
orders = P.sample_orders_batched_mlp(
    B, N_SAMPLES, mlp, "original",
    base_seed=SEED_BASE, device=device, tau=TAU, top_k=TOP_K,
).cpu().numpy()  # shape [N_SAMPLES, 64]

# Diagnostics: simple locality stats per order (mean manhattan step-to-step in 8x8 grid)
def manhattan_step(order):
    coords = np.array([(i // GRID, i % GRID) for i in order])
    diffs = np.abs(np.diff(coords, axis=0)).sum(axis=1)
    return float(diffs.mean())

stats = [{"manhattan": manhattan_step(o), "start": int(o[0])} for o in orders]

# Emit HTML
html = []
html.append("<!doctype html>")
html.append("<html lang='zh'><head><meta charset='utf-8'>")
html.append("<title>MLP order — image α=1 ckpt step50000 (β/A from step48000)</title>")
html.append("""<style>
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background:#fafafa; color:#222; padding:24px; margin:0; }
h1 { font-size: 20px; margin: 0 0 6px 0; }
.sub { color:#555; font-size:13px; margin-bottom: 20px; }
.row { display:flex; flex-wrap:wrap; gap:24px; }
.card { background:white; border:1px solid #ddd; border-radius:6px; padding:12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
.card h3 { margin:0 0 8px 0; font-size:13px; color:#333; font-weight:600; }
.card .meta { font-size:11px; color:#666; margin-bottom:8px; }
.grid { display:grid; grid-template-columns: repeat(8, 36px); grid-template-rows: repeat(8, 36px);
        gap:1px; background:#bbb; padding:1px; border-radius:3px; }
.cell { display:flex; align-items:center; justify-content:center; font-size:11px;
        font-family: ui-monospace, 'SF Mono', Menlo, monospace; font-weight:600; }
.legend { display:flex; align-items:center; gap:8px; margin-top:14px; font-size:12px; color:#555; }
.legend .bar { width:200px; height:12px;
       background: linear-gradient(to right, #fff7bc, #fec44f, #d95f0e, #7f2704); border:1px solid #999; }
.note { margin-top:20px; padding:12px; background:#fff8e6; border-left:3px solid #f0b429;
        font-size:13px; color:#444; max-width:900px; }
.note code { background:#f0f0f0; padding:1px 4px; border-radius:2px; }
</style></head><body>""")

html.append("<h1>Image alternating-MLP order @ step 50000 (α=1)</h1>")
html.append(f"<div class='sub'>每张 8×8 grid 表示一次采样得到的 decoding order — "
            f"格子里的数字是解码步序(1=最先,64=最后),颜色亮→暗代表早→晚。"
            f"参数 τ={TAU}, top_k={TOP_K},采样自 ckpt_step50000.pt + β/A from step48000(最近一次 refresh)。</div>")

html.append("<div class='row'>")

# colormap: from light-yellow → dark-red (sequential)
# stops at fractions [0.0, 0.33, 0.66, 1.0]
def lerp(a, b, t):
    return int(a + (b - a) * t)

def color_for(step01: float) -> str:
    # interpolate over: #fff7bc (cream) -> #fec44f (gold) -> #d95f0e (orange) -> #7f2704 (deep red)
    stops = [
        (0.00, (255, 247, 188)),
        (0.33, (254, 196,  79)),
        (0.66, (217,  95,  14)),
        (1.00, (127,  39,   4)),
    ]
    for i in range(len(stops) - 1):
        x0, c0 = stops[i]; x1, c1 = stops[i + 1]
        if step01 <= x1:
            t = (step01 - x0) / (x1 - x0) if x1 > x0 else 0.0
            r = lerp(c0[0], c1[0], t)
            g = lerp(c0[1], c1[1], t)
            b = lerp(c0[2], c1[2], t)
            return f"rgb({r},{g},{b})"
    return "rgb(127,39,4)"

for k, order in enumerate(orders):
    # order[t] = patch index decoded at step t  (0..63)
    # We need: for each patch index p, what step was it decoded at?
    step_of_patch = np.empty(GRID * GRID, dtype=int)
    for t, p in enumerate(order):
        step_of_patch[p] = t  # 0-based

    html.append("<div class='card'>")
    html.append(f"<h3>sample {k+1} (seed {SEED_BASE + k})</h3>")
    html.append(f"<div class='meta'>start patch = {stats[k]['start']} "
                f"({stats[k]['start']//GRID},{stats[k]['start']%GRID}) "
                f"&nbsp;|&nbsp; mean Δmanhattan = {stats[k]['manhattan']:.2f}</div>")
    html.append("<div class='grid'>")
    for p in range(GRID * GRID):
        t = step_of_patch[p]
        frac = t / (GRID * GRID - 1)
        bg = color_for(frac)
        # text color: light bg → black, dark bg → white
        fg = "#222" if frac < 0.55 else "#fff"
        html.append(f"<div class='cell' style='background:{bg};color:{fg}' "
                    f"title='patch {p} (row {p//GRID}, col {p%GRID}) decoded at step {t+1}/64'>{t+1}</div>")
    html.append("</div>")
    html.append("</div>")

html.append("</div>")  # row

html.append("<div class='legend'><span>解码步序</span>"
            "<span>step 1</span><div class='bar'></div><span>step 64</span></div>")

# Random baseline comparison: just one random order to show the contrast
rng = np.random.default_rng(SEED_BASE)
rand_order = rng.permutation(GRID * GRID)
rand_step_of_patch = np.empty(GRID * GRID, dtype=int)
for t, p in enumerate(rand_order):
    rand_step_of_patch[p] = t
rand_manhattan = manhattan_step(rand_order)

# Raster baseline (deterministic, just shows row-major sweep — for reference)
raster_order = np.arange(GRID * GRID)
raster_step_of_patch = np.empty(GRID * GRID, dtype=int)
for t, p in enumerate(raster_order):
    raster_step_of_patch[p] = t

html.append("<h2 style='margin-top:32px; font-size:18px;'>对照参考</h2>")
html.append("<div class='row'>")
for label, sop, mh in [("raster (reference)", raster_step_of_patch, manhattan_step(raster_order)),
                      ("random (seed match)", rand_step_of_patch, rand_manhattan)]:
    html.append("<div class='card'>")
    html.append(f"<h3>{label}</h3>")
    html.append(f"<div class='meta'>mean Δmanhattan = {mh:.2f}</div>")
    html.append("<div class='grid'>")
    for p in range(GRID * GRID):
        t = sop[p]
        frac = t / (GRID * GRID - 1)
        bg = color_for(frac)
        fg = "#222" if frac < 0.55 else "#fff"
        html.append(f"<div class='cell' style='background:{bg};color:{fg}'>{t+1}</div>")
    html.append("</div></div>")
html.append("</div>")

html.append("<div class='note'>"
            "<b>读法</b>:8×8 grid 中每格代表一个图像 patch(2×2 像素聚合后,共 64 patch),"
            "格内数字 = 该 patch 在 AR 解码中是第几步生成。"
            "颜色越亮(米黄)= 越早解码,颜色越深(暗红)= 越晚。"
            "<br><b>看点</b>:相邻 step 的格子应该在空间上靠近(mean Δmanhattan 接近 1 = 严格局部遍历;"
            f"raster={manhattan_step(raster_order):.2f},random≈5.3,MLP 期望在 1.x–3.x 之间反映自学到的局部 traversal)。"
            "</div>")

html.append("</body></html>")

OUT_HTML.write_text("\n".join(html), encoding="utf-8")
print("Wrote", OUT_HTML)
print("MLP orders mean Δmanhattan:", np.mean([s['manhattan'] for s in stats]).round(2),
      " | start patches:", [s['start'] for s in stats])

# =============== Animated HTML ===============
all_orders = {
    "mlp": [o.tolist() for o in orders],
    "raster": [raster_order.tolist()],
    "random": [rand_order.tolist()],
}
all_stats = {
    "mlp": stats,
    "raster": [{"manhattan": manhattan_step(raster_order), "start": int(raster_order[0])}],
    "random": [{"manhattan": rand_manhattan, "start": int(rand_order[0])}],
}

anim = []
anim.append("<!doctype html>")
anim.append("<html lang='zh'><head><meta charset='utf-8'>")
anim.append("<title>MLP order animation — image α=1 step50000</title>")
anim.append("""<style>
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background:#fafafa; color:#222; padding:20px; margin:0; }
h1 { font-size: 19px; margin: 0 0 4px 0; }
.sub { color:#555; font-size:13px; margin-bottom: 14px; }
.controls { display:flex; align-items:center; gap:14px; padding:12px 16px;
            background:white; border:1px solid #ddd; border-radius:6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06); margin-bottom:18px;
            position:sticky; top:0; z-index:10; }
.controls button { font-size:13px; padding:6px 14px; border-radius:4px;
                   border:1px solid #888; background:#fff; cursor:pointer; }
.controls button:hover { background:#eef; }
.controls button.primary { background:#1f77b4; color:#fff; border-color:#1f77b4; }
.controls button.primary:hover { background:#1465a0; }
.controls input[type=range] { width:160px; }
.controls .label { font-size:12px; color:#555; }
.controls .step-display { font-family: ui-monospace, Menlo, monospace;
                          font-size:13px; color:#222; min-width:80px; }
.section-title { font-size:14px; color:#333; margin: 18px 0 8px 4px;
                 font-weight:600; letter-spacing:0.02em; }
.row { display:flex; flex-wrap:wrap; gap:18px; }
.card { background:white; border:1px solid #ddd; border-radius:6px; padding:10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
.card h3 { margin:0 0 6px 0; font-size:12px; color:#333; font-weight:600; }
.card .meta { font-size:10.5px; color:#666; margin-bottom:6px; }
.grid { display:grid; grid-template-columns: repeat(8, 32px); grid-template-rows: repeat(8, 32px);
        gap:1px; background:#bbb; padding:1px; border-radius:3px; }
.cell { display:flex; align-items:center; justify-content:center; font-size:10.5px;
        font-family: ui-monospace, Menlo, monospace; font-weight:600;
        background:#eee; color:#aaa;
        transition: background 250ms ease-out, color 250ms ease-out; }
.cell.filled { /* set inline */ }
.note { margin-top:18px; padding:12px; background:#fff8e6; border-left:3px solid #f0b429;
        font-size:13px; color:#444; max-width:900px; }
</style></head><body>""")

anim.append("<h1>Image alternating-MLP order animation @ step 50000 (α=1)</h1>")
anim.append(f"<div class='sub'>每个 grid 按 MLP 采样的解码顺序逐步填充("
            f"τ={TAU}, top_k={TOP_K},β/A from step48000)。"
            f"颜色亮→暗 = 早→晚解码。点 Play 同步播放所有 grid。</div>")

anim.append("""<div class='controls'>
  <button id='play' class='primary'>▶ Play</button>
  <button id='pause'>⏸ Pause</button>
  <button id='reset'>⟲ Reset</button>
  <span class='label'>speed</span>
  <input type='range' id='speed' min='20' max='400' value='110'>
  <span class='step-display' id='stepDisplay'>step 0 / 64</span>
  <label class='label' style='margin-left:auto;'>
    <input type='checkbox' id='loop' checked> loop
  </label>
</div>""")

def emit_section(title, group_key, color_fn_inline):
    anim.append(f"<div class='section-title'>{title}</div>")
    anim.append("<div class='row'>")
    for k, order in enumerate(all_orders[group_key]):
        s = all_stats[group_key][k]
        title_label = (f"{group_key} sample {k+1}" if group_key == "mlp"
                       else group_key)
        anim.append(f"<div class='card'><h3>{title_label}</h3>"
                    f"<div class='meta'>start patch {s['start']} "
                    f"({s['start']//GRID},{s['start']%GRID}) · "
                    f"Δmanhattan={s['manhattan']:.2f}</div>")
        anim.append(f"<div class='grid' data-group='{group_key}' data-idx='{k}'>")
        for p in range(GRID * GRID):
            anim.append(f"<div class='cell' data-patch='{p}'></div>")
        anim.append("</div></div>")
    anim.append("</div>")

emit_section("MLP samples (α=1, step 50000)", "mlp", None)
emit_section("Reference: raster (row-major)", "raster", None)
emit_section("Reference: random", "random", None)

orders_json = json.dumps(all_orders)
anim.append(f"<script>const ALL_ORDERS = {orders_json}; const N = {GRID*GRID};</script>")
anim.append("""<script>
const stops = [
  [0.00, [255,247,188]],
  [0.33, [254,196, 79]],
  [0.66, [217, 95, 14]],
  [1.00, [127, 39,  4]],
];
function colorFor(f) {
  for (let i = 0; i < stops.length - 1; i++) {
    const [x0, c0] = stops[i], [x1, c1] = stops[i+1];
    if (f <= x1) {
      const t = x1 > x0 ? (f - x0) / (x1 - x0) : 0;
      const r = Math.round(c0[0] + (c1[0]-c0[0]) * t);
      const g = Math.round(c0[1] + (c1[1]-c0[1]) * t);
      const b = Math.round(c0[2] + (c1[2]-c0[2]) * t);
      return `rgb(${r},${g},${b})`;
    }
  }
  return 'rgb(127,39,4)';
}
const grids = Array.from(document.querySelectorAll('.grid'));
const stepDisplay = document.getElementById('stepDisplay');
const speedSlider = document.getElementById('speed');
const loopChk = document.getElementById('loop');
let step = 0, timer = null;

function resetAll() {
  step = 0;
  grids.forEach(g => {
    Array.from(g.children).forEach(c => {
      c.style.background = '#eee';
      c.style.color = '#aaa';
      c.textContent = '';
    });
  });
  stepDisplay.textContent = `step 0 / ${N}`;
}

function tick() {
  if (step >= N) {
    if (loopChk.checked) {
      stop();
      setTimeout(() => { resetAll(); play(); }, 600);
    } else {
      stop();
    }
    return;
  }
  const frac = step / (N - 1);
  const bg = colorFor(frac);
  const fg = frac < 0.55 ? '#222' : '#fff';
  grids.forEach(g => {
    const group = g.dataset.group;
    const idx = parseInt(g.dataset.idx);
    const order = ALL_ORDERS[group][idx];
    const p = order[step];
    const cell = g.children[p];
    cell.style.background = bg;
    cell.style.color = fg;
    cell.textContent = step + 1;
  });
  step += 1;
  stepDisplay.textContent = `step ${step} / ${N}`;
}

function play() {
  if (timer) return;
  if (step >= N) resetAll();
  timer = setInterval(tick, parseInt(speedSlider.value));
}
function stop() { if (timer) { clearInterval(timer); timer = null; } }

document.getElementById('play').addEventListener('click', play);
document.getElementById('pause').addEventListener('click', stop);
document.getElementById('reset').addEventListener('click', () => { stop(); resetAll(); });
speedSlider.addEventListener('input', () => {
  if (timer) { stop(); play(); }
});
resetAll();
// auto-play on load
setTimeout(play, 400);
</script>""")

anim.append("<div class='note'>"
            "<b>读法</b>:点 Play 后,所有 8×8 grid 同步按各自的解码顺序逐步亮起。"
            "拖 speed 改变每步间隔(20–400ms)。勾选 loop 后自动循环。"
            "<br><b>看点</b>:MLP samples 应该出现"
            "<i>蛇形局部遍历</i>(相邻 step 在空间上紧邻),起点偏好角落。"
            "raster 是行优先扫描的参考,random 几乎跳跃。"
            "</div>")

anim.append("</body></html>")
OUT_ANIM.write_text("\n".join(anim), encoding="utf-8")
print("Wrote", OUT_ANIM)
