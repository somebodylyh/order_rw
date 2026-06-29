"""Scan ALL heads for content-only tau vs L2R — batched, fast."""
import sys, json, numpy as np, torch
from scipy.stats import kendalltau

_HERE = "/home/admin/lyuyuhuan/order_lyu/block_lo_arm_order_network"
sys.path.insert(0, _HERE)

from neural_readout.extract_b import _load_model_and_chunks
from per_head_order_scan import _per_sample_A, _batch_mean_B, extract_per_head_and_heavy_A
from neural_readout.teacher_labels import generate_teacher_label
from batch_readout.diversity_batch import teacher_diversity_stats

ckpt = f"{_HERE}/probe_results/clean_base_random_perm/ckpt_step5000.pt"
M, batch_size = 100, 16
seed, device = 42, "cuda:1"

print(f"Loading model + {M*batch_size} chunks...")
model, chunks, clean_perm, dev, _ = _load_model_and_chunks(ckpt, M*batch_size, seed, device, "train")
inv_perm = clean_perm.inv_perm_model_to_phys.cpu().numpy()
l2r = np.arange(64)

print(f"Extracting ALL heads (none_mode=content, fwd_batch=32)...")
A_lh, _ = extract_per_head_and_heavy_A(
    model, chunks, clean_perm, dev, seed,
    n_top=4, fwd_batch=32, none_mode="content",
)
# A_lh: (n_chunks, L, H, N, N) where n_chunks = M * batch_size
L, H = A_lh.shape[1], A_lh.shape[2]
print(f"A_lh shape: {A_lh.shape}")

results = []
for ell in range(L):
    for hh in range(H):
        A_sel = A_lh[:, ell, hh]  # (n, N, N)
        B_batch = _batch_mean_B(A_sel, M, batch_size)

        sigmas = np.zeros((M, 64), dtype=np.int64)
        for m in range(M):
            sigmas[m], _, _ = generate_teacher_label(B_batch[m], alpha_dep=0.5)

        unique = len(set(tuple(s.tolist()) for s in sigmas))
        div = teacher_diversity_stats(sigmas)

        taus = []
        for s in sigmas:
            s_phys = inv_perm[s]
            t, _ = kendalltau(s_phys, l2r)
            if not np.isnan(t): taus.append(t)
        taus = np.array(taus)

        row_negs = []
        for m in range(M):
            for i in range(64):
                row = np.abs(B_batch[m, i]) + 1e-12
                p = row / row.sum()
                H_r = -np.sum(p * np.log(p))
                row_negs.append(np.log(64) - H_r)
        row_conc = float(np.mean(row_negs))

        r = {
            "layer": ell, "head": hh,
            "tau_mean": float(taus.mean()), "tau_std": float(taus.std()),
            "abs_tau_mean": float(np.abs(taus).mean()),
            "frac_gt_05": float((np.abs(taus) > 0.5).mean()),
            "frac_gt_07": float((np.abs(taus) > 0.7).mean()),
            "frac_pos": float((taus > 0).mean()),
            "pw_tau": float(div["mean_pairwise_tau"]),
            "unique_ratio": float(div["unique_sigma_ratio"]),
            "first_H": float(div["first_step_entropy"]),
            "row_conc": row_conc,
        }
        results.append(r)
        print(f"  L{ell}H{hh}: tau={r['tau_mean']:+.3f}±{r['tau_std']:.2f} |tau|>0.5={r['frac_gt_05']:.2f} pw={r['pw_tau']:.3f} row_conc={row_conc:.2f} unique={unique}/{M}", flush=True)

results.sort(key=lambda r: abs(r["tau_mean"]), reverse=True)
print(f"\n{'='*90}")
print(f"RANKED by |tau| (content-only)")
print(f"{'='*90}")
for i, r in enumerate(results):
    print(f"  {i:2d} L{r['layer']}H{r['head']}: tau={r['tau_mean']:+.4f}±{r['tau_std']:.3f} |tau|>0.5={r['frac_gt_05']:.3f} |tau|>0.7={r['frac_gt_07']:.3f} %pos={r['frac_pos']:.1%} row_conc={r['row_conc']:.4f} pw={r['pw_tau']:.4f}")

out = f"{_HERE}/analyses/content_head_scan_5k.json"
json.dump(results, open(out, "w"), indent=2)
print(f"\nSaved -> {out}")
