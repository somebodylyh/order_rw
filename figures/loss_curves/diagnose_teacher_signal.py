"""Diagnose how the teacher signal evolves with the AOGPT checkpoint.

No downstream NLL — purely analyzes the raw attention graph B = Aᵀ and the
C-D+L teacher built on it. Outputs:
  - per-(run, step) row with C/D/L scale, B locality, teacher rollout stats
  - ablation: |τ| under each teacher mode (C, L, C-D, C+L, C-D+L)
  - markdown summary and TSV (saved next to this script)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

REPO = Path("/home/admin/lyuyuhuan/order_lyu")
sys.path.insert(0, str(REPO / "block_lo_arm_order_network"))

from attn_order_teacher import teacher_components, rollout_order, MODES
from scipy.stats import kendalltau

OUT = REPO / "figures/loss_curves"
OUT.mkdir(parents=True, exist_ok=True)

# (run_label, ckpt_dir, glob, [steps to skip])
RUNS = [
    ("alt α=1",   REPO / "probe_results/attention_order_mlp/alt_from0_mlp_alpha1",
     "A_global_step*.npy", {50000}),  # step50000 is a manual cp of step48000
    ("alt α=0.9", REPO / "probe_results/attention_order_mlp/alt_from0_mlp_finetune",
     "A_global_step*.npy", set()),
    ("random",    REPO / "block_lo_arm_order_network/probe_results/clean_base_random_perm",
     "A_global_step*_seed42.npy", set()),
]

K_ROLLOUT = 64           # rollouts per teacher mode
N_STATES = 100           # random states for C/D/L stats
SEED = 42


def load_B(npy_path: Path) -> np.ndarray:
    A = np.load(npy_path).astype(np.float64)
    B = A.T.copy()
    np.fill_diagonal(B, 0.0)
    return B


def b_stats(B: np.ndarray) -> dict:
    """Diagonal-locality, neighbor preference, spectral spread."""
    N = B.shape[0]
    total = B.sum()
    # adjacent (|i-j|=1) mass fraction
    adj_idx = np.abs(np.arange(N)[:, None] - np.arange(N)[None, :]) == 1
    adj_mass_frac = float(B[adj_idx].sum() / (total + 1e-12))
    # argmax neighbor fraction: how often argmax over j of B[i,:] is |j-i|=1
    argmax_j = B.argmax(axis=1)
    is_neighbor = np.abs(argmax_j - np.arange(N)) == 1
    neighbor_argmax_frac = float(is_neighbor.mean())
    # spectral
    w = np.linalg.eigvalsh((B + B.T) / 2)
    spectral_spread = float(w.max() - w.min())
    return {
        "adj_mass_frac": round(adj_mass_frac, 4),
        "argmax_neighbor_frac": round(neighbor_argmax_frac, 4),
        "spectral_spread": round(spectral_spread, 4),
    }


def cdl_stats(B: np.ndarray, n_states: int, seed: int) -> dict:
    """Mean/std of C, D, L over random partial states."""
    N = B.shape[0]
    rng = np.random.default_rng(seed)
    Cs, Ds, Ls = [], [], []
    for k in range(n_states):
        t = rng.integers(2, N - 1)
        perm = rng.permutation(N)
        S = perm[:t].tolist()
        U = perm[t:].tolist()
        last = int(perm[t - 1])
        C, D, L, _ = teacher_components(B, S, U, last)
        Cs.append(C); Ds.append(D); Ls.append(L)
    Cf = np.concatenate(Cs); Df = np.concatenate(Ds); Lf = np.concatenate(Ls)
    return {
        "C_mean": round(float(Cf.mean()), 4),
        "C_std":  round(float(Cf.std()),  4),
        "D_mean": round(float(Df.mean()), 4),
        "D_std":  round(float(Df.std()),  4),
        "L_mean": round(float(Lf.mean()), 4),
        "L_std":  round(float(Lf.std()),  4),
        "CmDpL_mean":  round(float((Cf - Df + Lf).mean()), 4),
        "CmDpL_std":   round(float((Cf - Df + Lf).std()),  4),
    }


def teacher_rollout_stats(B: np.ndarray, mode: str, K: int, seed: int,
                          tau_T: float = 0.5) -> dict:
    """Run K teacher rollouts under given mode, report τ/|τ|/entropy/diversity."""
    raster = np.arange(B.shape[0])
    orders = []
    ents_per_run = []
    for k in range(K):
        o, ent = rollout_order(B, tau_T=tau_T, seed=seed + k, mode=mode,
                               standardize=True, return_entropy=True)
        orders.append(o)
        ents_per_run.append(float(np.mean(ent)) if ent else 0.0)
    orders = np.stack(orders)
    taus = []
    for k in range(K):
        r = kendalltau(orders[k], raster).correlation
        if np.isfinite(r):
            taus.append(r)
    tau_mean = float(np.mean(taus)) if taus else float("nan")
    abs_tau  = float(np.mean(np.abs(taus))) if taus else float("nan")
    starts = orders[:, 0]
    unique_orders = len({tuple(o.tolist()) for o in orders})
    start_top = int(np.bincount(starts, minlength=B.shape[0]).max())
    return {
        f"{mode}_tau":     round(tau_mean, 4),
        f"{mode}_abs_tau": round(abs_tau, 4),
        f"{mode}_entropy": round(float(np.mean(ents_per_run)), 4),
        f"{mode}_unique":  unique_orders,
        f"{mode}_start_top_count": start_top,
    }


# ---------------- main ----------------
def parse_step(stem: str) -> int:
    # "A_global_step12000" or "A_global_step12000_seed42"
    s = stem.split("step")[-1].split("_")[0]
    return int(s)


rows = []
for run_label, run_dir, glob_pat, skip_steps in RUNS:
    A_files = sorted(run_dir.glob(glob_pat), key=lambda p: parse_step(p.stem))
    for A_path in A_files:
        step = parse_step(A_path.stem)
        if step in skip_steps:
            continue
        B = load_B(A_path)
        row = {"run": run_label, "step": step}
        row.update(b_stats(B))
        row.update(cdl_stats(B, N_STATES, SEED))
        for m in MODES:
            row.update(teacher_rollout_stats(B, m, K_ROLLOUT, SEED))
        rows.append(row)
        print(f"  [{run_label}] step {step}: adj_mass={row['adj_mass_frac']:.3f} "
              f"arg_nbr={row['argmax_neighbor_frac']:.3f} | "
              f"C-D+L τ={row['C-D+L_tau']:.3f} |τ|={row['C-D+L_abs_tau']:.3f} "
              f"ent={row['C-D+L_entropy']:.3f}")

# emit TSV
keys = list(rows[0].keys())
with (OUT / "teacher_signal_evolution.tsv").open("w") as f:
    f.write("\t".join(keys) + "\n")
    for r in rows:
        f.write("\t".join(str(r[k]) for k in keys) + "\n")
print("Wrote", OUT / "teacher_signal_evolution.tsv")

# emit compact markdown
md = ["# Teacher signal evolution (no downstream NLL)\n",
      "B = Aᵀ (zero diag). teacher modes scored on K=64 standardize rollouts, τ_T=0.5. "
      "C/D/L stats from 100 random states.\n",
      "| run | step | adj_mass | argmax_nbr | C_mean | D_mean | L_mean | "
      "C-D+L |τ| | C-D+L ent | C only |τ| | L only |τ| | C-D |τ| | C+L |τ| |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    md.append(f"| {r['run']} | {r['step']} | {r['adj_mass_frac']} | "
              f"{r['argmax_neighbor_frac']} | {r['C_mean']} | {r['D_mean']} | "
              f"{r['L_mean']} | **{r['C-D+L_abs_tau']}** | "
              f"{r['C-D+L_entropy']} | "
              f"{r['C_abs_tau']} | {r['L_abs_tau']} | "
              f"{r['C-D_abs_tau']} | {r['C+L_abs_tau']} |")
md.append("")
md.append("**读法**:")
md.append("- `adj_mass`(B 邻接 mass 比例)、`argmax_nbr`(argmax 是邻居的比例)= B 自身的 chain-ness。")
md.append("- `|τ|` = teacher rollout 的 orientation-free structure 强度(1=完美 chain,0=随机)。")
md.append("- ablation 比较:看 C / L / C-D / C+L / C-D+L 哪个模式 |τ| 最高 → 哪一项 / 哪种组合最贡献结构。")
md.append("- 若 C-D+L |τ| < C+L |τ| ⇒ D 项有害;若 C-D+L > C+L ⇒ D 项有用。")
md.append("- C/D/L mean 比例反映 (1,1,1) 是否数值平衡。")

(OUT / "teacher_signal_evolution.md").write_text("\n".join(md) + "\n")
print("Wrote", OUT / "teacher_signal_evolution.md")
