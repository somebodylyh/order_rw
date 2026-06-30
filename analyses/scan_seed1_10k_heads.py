"""Quick per-head CDL scan for seed1 @ 10k — find which head has L2R signal."""
import sys, os, time
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AOGPT_DIR = os.path.join(_SCRIPT_DIR, "..", "block_lo_arm_order_network")
sys.path.insert(0, _AOGPT_DIR)

from attn_order_teacher import rollout_order
from per_head_order_scan import extract_per_head_and_heavy_A
from neural_readout.extract_b import _load_model_and_chunks

N_BLOCKS = 64; L2R = np.arange(N_BLOCKS); TAU_T = 1.0

def kendall_tau(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n = len(a); conc = disc = 0
    for i in range(n):
        for j in range(i+1, n):
            da, db = a[i]-a[j], b[i]-b[j]
            if da==0 and db==0: continue
            if da*db>0: conc+=1
            elif da*db<0: disc+=1
    d=conc+disc; return (conc-disc)/d if d>0 else 0.0

def _A_to_B(A):
    B=np.asarray(A.T,dtype=np.float64).copy(); np.fill_diagonal(B,0.0); return B

CKPT = ("/home/admin/lyuyuhuan/order_lyu/block_lo_arm_order_network/"
        "probe_results/random_baseline_continuous_jun09_seed1/ckpt_step10000.pt")

t0 = time.time()
print(f"Loading {CKPT}")
model, chunks, clean_perm, dev, ci = _load_model_and_chunks(
    ckpt_path=CKPT, M=200, seed=42, device="cuda:0", split="train")

A_lh, _ = extract_per_head_and_heavy_A(
    model, chunks, clean_perm, dev, seed=42, fwd_batch=64, none_mode="b0")
n, L, H, N, _ = A_lh.shape
print(f"Extracted {n} graphs, {L}x{H} heads in {time.time()-t0:.1f}s\n")

# Per-head CDL scan
print(f"{'Head':>8s}  {'τ(CDL,L2R)':>12s}  {'C-D+L':>10s}  {'C-D':>10s}  {'-D only':>10s}")
print("-"*60)
results = []
for l in range(L):
    for h in range(H):
        B = _A_to_B(A_lh[:,l,h].mean(axis=0))
        # C-D+L
        o1 = rollout_order(B, tau_T=TAU_T, seed=42, mode="C-D+L", standardize=True)
        t1 = kendall_tau(o1, L2R)
        # C-D
        o2 = rollout_order(B, tau_T=TAU_T, seed=42, mode="C-D", standardize=True)
        t2 = kendall_tau(o2, L2R)
        # -D only
        from attn_order_teacher import teacher_components, _softmax, _entropy
        Br = np.asarray(B, dtype=np.float64); rng = np.random.default_rng(42)
        S,U,last=[],list(range(N)),None; order=[]
        for _ in range(N):
            if len(U)==1: v=U[0]
            else:
                _,D,_,cand=teacher_components(Br,S,U,last)
                q=-D; q=(q-q.mean())/(q.std()+1e-9); p=_softmax(q,TAU_T)
                v=int(cand[rng.choice(len(cand),p=p)])
            order.append(v); S.append(v); U.remove(v); last=v
        t3 = kendall_tau(np.array(order,dtype=np.int64), L2R)

        results.append((l,h,t1,t2,t3))
        marker = " ★" if t1 > 0.3 else ""
        print(f"  L{l}H{h}    {t1:+.4f}        {t2:+.4f}      {t3:+.4f}{marker}")

# Best heads
results.sort(key=lambda x: -x[1])  # sort by C-D+L τ
print(f"\nBest heads (C-D+L τ):")
for l,h,t1,t2,t3 in results[:5]:
    print(f"  L{l}H{h}: C-D+L={t1:+.4f}  C-D={t2:+.4f}  -D={t3:+.4f}")

# Summary: how many heads have τ > 0.2?
good = [r for r in results if r[2] > 0.2]
print(f"\n{len(good)}/{L*H} heads have C-D+L τ > 0.2")
if good:
    for l,h,t1,t2,t3 in good:
        print(f"  L{l}H{h}: τ={t1:+.4f}")
else:
    print("  NONE — seed1 @ 10k has no CDL-readable L2R head!")
    print("  The attention hasn't crystallized yet at this checkpoint.")
