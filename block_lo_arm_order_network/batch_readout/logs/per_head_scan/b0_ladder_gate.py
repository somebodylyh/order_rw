"""§3.0 gate: compare OLD vs B0 ladders on row-concentration health, top-k pool
precision, best+ stability, and late-stage winner identity. Reads both dirs'
ckpt*_seed*.json graph dumps. Pass = B0 rowconc median not collapsed, high pool
precision, best+ stable across seeds."""
import json, glob, re, os
import numpy as np
from collections import defaultdict
import statistics as st

OLD_DIR = os.path.dirname(os.path.abspath(__file__))
B0_DIR = os.path.join(os.path.dirname(OLD_DIR), "per_head_scan_b0")
EPS = 1e-12
TOPK = 5
TAU_STRONG = 0.8


def c_row(A):  # A:(L,H,N,N) -> (L*H,) row-concentration of B=A^T
    L, H, N, _ = A.shape
    B = A.transpose(0, 1, 3, 2).copy()
    di = np.arange(N); B[:, :, di, di] = 0.0
    rs = B.sum(-1, keepdims=True); P = B / (rs + EPS)
    lP = np.where(P > 0, np.log(P + EPS), 0.0); Hr = -(P * lP).sum(-1)
    Cn = 1.0 - Hr / np.log(N - 1)
    Cn = np.where(rs[..., 0] <= EPS, 0.0, Cn)
    return Cn.mean(-1).reshape(-1)


def load(d):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "ckpt*_seed*.json"))):
        o = json.load(open(f))
        g = o.get("graphs")
        if not g or "A_mean" not in g:
            continue
        step = int(re.search(r"ckpt(\d+)_seed", f).group(1))
        seed = int(re.search(r"_seed(\d+)", f).group(1))
        A = np.array(g["A_mean"], np.float64); L, H = A.shape[:2]
        C = c_row(A)
        tau = np.zeros(L * H)
        for h in o["per_head_layer_sorted_by_abs_tau_vs_l2r"]:
            tau[h["layer"] * H + h["head"]] = h["tau_vs_l2r"]
        topk = np.argsort(-C)[:TOPK]
        rows.append(dict(step=step, seed=seed, med=float(np.median(C)),
                         winconc=float(C[np.argmax(np.abs(tau))]),
                         pool_prec=float(np.mean(np.abs(tau[topk]) >= 0.9)),
                         bestpos_recall=int(np.any(tau[topk] >= TAU_STRONG)),
                         maxtau_topk=float(np.max(np.abs(tau[topk]))),
                         top1=int(topk[0])))
    return rows


def report(name, rows):
    by = defaultdict(list)
    for r in rows: by[r["step"]].append(r)
    print(f"\n=== {name} ({len(rows)} points) ===")
    print(f"{'step':>6} | med C | poolP@5 | best+rec | maxτ@5 | top1-head mode")
    for s in sorted(by):
        rs = by[s]
        from collections import Counter
        mode = Counter(r["top1"] for r in rs).most_common(1)[0]
        print(f"{s:>6} | {st.median([r['med'] for r in rs]):.3f} | "
              f"{st.mean([r['pool_prec'] for r in rs]):.2f}    | "
              f"{st.mean([r['bestpos_recall'] for r in rs]):.2f}     | "
              f"{st.median([r['maxtau_topk'] for r in rs]):.2f}   | "
              f"L{mode[0]//8}H{mode[0]%8} ({mode[1]}/{len(rs)})")


old, b0 = load(OLD_DIR), load(B0_DIR)
report("OLD (0.1-sink)", old)
report("B0 (none->block0)", b0)

# §3.0 PASS check on B0
conv = [r for r in b0 if r["step"] >= 5000]
med_min = min(r["med"] for r in conv) if conv else 0.0
poolP = st.mean([r["pool_prec"] for r in conv]) if conv else 0.0
recall = st.mean([r["bestpos_recall"] for r in conv]) if conv else 0.0
print(f"\n§3.0 GATE (B0, step>=5000): min median C={med_min:.3f} (want >0.02), "
      f"mean poolP@5={poolP:.2f}, mean best+recall@5={recall:.2f} (want ~1.0)")
print("PASS" if (med_min > 0.02 and recall > 0.9) else "REVIEW")
