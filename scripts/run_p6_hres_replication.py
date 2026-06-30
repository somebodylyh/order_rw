"""Replicate the P6 H-residual-modulation result across seeds & ckpts.

Focus metric: does the penalized residual arm (b+H_res+pen) show a STABLE
real<shuffle gap (sample-specific content) with B/position redundancy removed,
across CDL runs (seeds 123/2/124, their own teacher heads) and ckpts?
"""
import json, pathlib, sys
sys.path.insert(0, ".")
from analyses.p6_h_residual_modulation import run_h_residual

PR = "block_lo_arm_order_network/probe_results"
# (run_dir, layer, head, [ckpt steps])
MATRIX = [
    ("cdl_from10k_L1H7_seed123", 1, 7, [20000, 30000, 40000]),       # s123, multi-ckpt
    ("cdl_teacher_seed123_from10k_l0h2", 0, 2, [25000, 40000]),       # s123, other head
    ("cdl_teacher_seed2_from10k_l0h2", 0, 2, [25000, 40000]),         # s2
    ("cdl_teacher_from20k_seed124", 0, 2, [30000, 50000]),           # s124
]

rows = []
for run_dir, layer, head, steps in MATRIX:
    for st in steps:
        ck = f"{PR}/{run_dir}/ckpt_step{st}.pt"
        if not pathlib.Path(ck).exists():
            print(f"SKIP missing {ck}", flush=True); continue
        tag = f"{run_dir}_{st}"
        try:
            r = run_h_residual(ck, M=64, layer=layer, head=head, epochs=300,
                               out_dir="runs/p6/h_residual_repl", tag=tag)
        except Exception as e:
            print(f"FAIL {tag}: {e}", flush=True); continue
        pen = r["arms"]["b_plus_Hres_pen"]
        raw = r["arms"]["b_plus_rawH"]
        rows.append({
            "run": run_dir, "seed": run_dir, "step": st, "layer": layer, "head": head,
            "pen_real_vs_shuffle": pen["real_vs_shuffle_delta"],
            "pen_delta_vs_bonly": pen["delta_vs_bonly"],
            "pen_corr_zH_zB": pen["corr_zH_zB"], "pen_corr_zH_phys": pen["corr_zH_phys"],
            "pen_norm_ratio": pen["norm_ratio"], "pen_lambda": pen["lambda"],
            "rawH_real_vs_shuffle": raw["real_vs_shuffle_delta"],
        })
        print(f"[{tag}] pen real-shuf={pen['real_vs_shuffle_delta']:+.4f} "
              f"dNLL={pen['delta_vs_bonly']:+.4f} corr_zB={pen['corr_zH_zB']:+.3f} "
              f"corr_phys={pen['corr_zH_phys']:+.3f} normR={pen['norm_ratio']:.3f} "
              f"| rawH real-shuf={raw['real_vs_shuffle_delta']:+.4f}", flush=True)

out = pathlib.Path("runs/p6/h_residual_repl"); out.mkdir(parents=True, exist_ok=True)
json.dump(rows, open(out / "summary.json", "w"), indent=2, default=float)

# verdict
import numpy as np
rs = np.array([r["pen_real_vs_shuffle"] for r in rows])
print("\n=== SUMMARY ===", flush=True)
print(f"n={len(rs)}  pen real_vs_shuffle: mean={rs.mean():+.4f} "
      f"frac_negative(real<shuffle)={(rs<0).mean():.2f} min={rs.min():+.4f} max={rs.max():+.4f}",
      flush=True)
print("REPLICATION DONE", flush=True)
