"""What order does H represent across CDL-from10k ckpts (20k..60k)?

The CDL run keeps training on the CDL teacher order. Track, per ckpt, the order
the H-only readout represents vs physical and vs the (batch-mean) CDL order, with
the B readout as a reference, plus across-ckpt stability of the H order.
"""
import json, pathlib, sys
import numpy as np
from scipy.stats import kendalltau

sys.path.insert(0, ".")
from analyses.p6_online_controller import (
    build_dataset_p6, _yl, train_controller_online, controller_scores, h_mode_list,
)
from analyses.p5_utility_controller import sigma_from_B65, N

PR = "block_lo_arm_order_network/probe_results/cdl_from10k_L1H7_seed123"
STEPS = [20000, 30000, 40000, 50000, 60000]
phys = np.arange(N)


def _consensus_order(controller, samples, Hs=None):
    Z = []
    for i, s in enumerate(samples):
        H = None if Hs is None else Hs[i]
        Z.append(controller_scores(controller, s["B_feat"], H).detach().cpu().numpy())
    return np.argsort(-np.mean(Z, axis=0)).astype(np.int64)


def _tau(a, b):
    return float(kendalltau(a, b)[0])


rows, prev_h = [], None
for st in STEPS:
    ck = f"{PR}/ckpt_step{st}.pt"
    if not pathlib.Path(ck).exists():
        print(f"SKIP {ck}", flush=True); continue
    samples = build_dataset_p6(ck, M=64, layer=1, head=7, heads=None, K=6, h_layer=1)
    Ys, Ls, _ = _yl(samples)
    sigma_cdl = sigma_from_B65(np.mean([s["B65"] for s in samples], axis=0))

    g_B = train_controller_online(samples, Ys, Ls, mode="b_only", tau=0.3, epochs=300)
    for p in g_B.parameters():
        p.requires_grad_(False)
    h_dim = samples[0]["H"].shape[1]
    h_only = train_controller_online(samples, Ys, Ls, mode="h_only", tau=0.3,
                                     epochs=300, h_dim=h_dim)
    Hs = h_mode_list([s["H"] for s in samples], "real")

    H_order = _consensus_order(h_only, samples, Hs)
    B_order = _consensus_order(g_B, samples)
    row = {"step": st,
           "H_vs_phys": _tau(H_order, phys), "H_vs_cdl": _tau(H_order, sigma_cdl),
           "B_vs_phys": _tau(B_order, phys), "B_vs_cdl": _tau(B_order, sigma_cdl),
           "cdl_vs_phys": _tau(sigma_cdl, phys),
           "H_vs_prev_ckpt": _tau(H_order, prev_h) if prev_h is not None else None}
    prev_h = H_order
    rows.append(row)
    print(f"[{st}] H~phys={row['H_vs_phys']:+.3f} H~cdl={row['H_vs_cdl']:+.3f} | "
          f"B~phys={row['B_vs_phys']:+.3f} B~cdl={row['B_vs_cdl']:+.3f} | "
          f"cdl~phys={row['cdl_vs_phys']:+.3f} | H~prev={row['H_vs_prev_ckpt']}",
          flush=True)

out = pathlib.Path("runs/p6/h_order_evolution"); out.mkdir(parents=True, exist_ok=True)
json.dump(rows, open(out / "summary.json", "w"), indent=2, default=float)
print("H ORDER EVOLUTION DONE", flush=True)
