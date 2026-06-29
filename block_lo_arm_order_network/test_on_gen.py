"""Quick ON generalization test: τ(ON, teacher) at 16k vs known 8k baseline (~0.92)."""
import os, sys, time, json
import numpy as np
import torch
import torch.nn.functional as F

_current = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _current)
from order_network import CrossAttentionOrderNetwork

N = 64
LAM = 0.25
DEVICE = "cuda:0"
EPOCHS = 20
LR = 1e-3


def kendall_tau(a, b):
    n = len(a)
    pos_a = np.empty(n, dtype=np.int64); pos_a[a] = np.arange(n)
    p = d = 0
    for i in range(n):
        for j in range(i+1, n):
            p += 1
            if (pos_a[b[i]] - pos_a[b[j]]) * (j - i) > 0: d += 1
    return (p - 2*d) / max(p, 1)


def compute_teachers(A, Wg):
    """NN greedy mixed025 teacher orders for all samples."""
    ns, N = A.shape[:2]
    orders = np.zeros((ns, N), dtype=np.int64)
    t0 = time.time()
    for s in range(ns):
        As = A[s].astype(np.float64); np.fill_diagonal(As, 0.0)
        W = 0.5*(As + As.T); np.fill_diagonal(W, 0.0)
        W = LAM*W + (1-LAM)*Wg
        starts = np.argsort(W.sum(axis=1))[:2]
        best_o, best_w = None, -np.inf
        for st in starts:
            o = np.zeros(N, dtype=np.int64); v = np.zeros(N, dtype=bool)
            cur = int(st)
            for t in range(N):
                o[t] = cur; v[cur] = True
                if t == N-1: break
                wc = W[cur].copy(); wc[v] = -np.inf; cur = int(np.argmax(wc))
            pw = sum(float(W[o[i], o[i+1]]) for i in range(N-1))
            if pw > best_w: best_w, best_o = pw, o
        orders[s] = best_o
        if (s+1) % 5000 == 0:
            print(f"  teacher {s+1}/{ns} ({ns/max(time.time()-t0,1e-9):.0f} seq/s)", flush=True)
    print(f"  teachers done in {time.time()-t0:.1f}s", flush=True)
    return orders


def compute_amix(A, Wg):
    """A_mix = λ * W_sym + (1-λ) * Wg for all samples."""
    ns, N = A.shape[:2]
    am = np.zeros_like(A)
    for s in range(ns):
        As = A[s].astype(np.float64); np.fill_diagonal(As, 0.0)
        W = 0.5*(As + As.T); np.fill_diagonal(W, 0.0)
        am[s] = (LAM*W + (1-LAM)*Wg).astype(np.float32)
    return am


@torch.no_grad()
def compute_tau(on_model, A_t, teacher_t):
    """τ(ON_greedy, teacher) mean over batch."""
    B, N = A_t.shape[:2]
    vis = torch.zeros(B, dtype=torch.long, device=DEVICE)
    last = torch.zeros(B, dtype=torch.long, device=DEVICE)
    on_o = np.zeros((B, N), dtype=np.int64)
    for t in range(N):
        logits = on_model(A_t, vis, last)
        pred = logits.argmax(dim=-1)
        on_o[:, t] = pred.cpu().numpy()
        vis = vis | (1 << pred); last = pred
    tn = teacher_t.cpu().numpy()
    return float(np.mean([kendall_tau(on_o[i], tn[i]) for i in range(B)]))


def train(on_model, A_t, p_t, lr):
    """1 epoch BC training. Returns model in eval mode."""
    n_tr = A_t.shape[0]; bs = 64
    opt = torch.optim.AdamW(on_model.parameters(), lr=lr)
    for ep in range(EPOCHS):
        on_model.train()
        perm = torch.randperm(n_tr)
        for st in range(0, n_tr, bs):
            idx = perm[st:st+bs]; Ab = A_t[idx]; pb = p_t[idx]; B = Ab.shape[0]
            vis = torch.zeros(B, dtype=torch.long, device=DEVICE)
            last = torch.zeros(B, dtype=torch.long, device=DEVICE)
            for t in range(N):
                logits = on_model(Ab, vis, last)
                tg = pb[:, t]
                loss = F.cross_entropy(logits, tg, reduction='sum') / B
                vis = vis | (1 << tg); last = tg
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(on_model.parameters(), 1.0)
                opt.step()
    on_model.eval()
    return on_model


def test_size(train_size, A_all, teacher_all, n_val=4000):
    """Train ON on train_size samples, test τ on last n_val."""
    n_max = len(A_all) - n_val
    ts = min(train_size, n_max)
    print(f"\n--- train_size={ts} ---", flush=True)

    # Compute A_global from training set
    Ag_train = A_all[:ts].mean(axis=0)
    Wg = 0.5*(Ag_train + Ag_train.T); np.fill_diagonal(Wg, 0.0)

    # A_mix for train and val
    Am_train = compute_amix(A_all[:ts], Wg)
    Am_val = compute_amix(A_all[n_max:], Wg)

    Amt = torch.from_numpy(Am_train).float().to(DEVICE)
    Amv = torch.from_numpy(Am_val).float().to(DEVICE)
    pt = torch.from_numpy(teacher_all[:ts]).to(DEVICE)
    pv = torch.from_numpy(teacher_all[n_max:]).to(DEVICE)

    on_model = CrossAttentionOrderNetwork(
        feature_dim=9, hidden_dim=128, num_blocks=N,
        d_model=256, nhead=8, num_layers=4, dropout=0.1,
    ).to(DEVICE)
    n_param = sum(p.numel() for p in on_model.parameters())

    t0 = time.time()
    on_model = train(on_model, Amt, pt, LR)
    dt = time.time() - t0

    tau_tr = compute_tau(on_model, Amt[:min(500, ts)], pt[:min(500, ts)])
    tau_v = compute_tau(on_model, Amv, pv)

    print(f"  params={n_param:,}  τ_train={tau_tr:.4f}  τ_val={tau_v:.4f}  time={dt:.0f}s", flush=True)
    return {"train_size": ts, "tau_train": tau_tr, "tau_val": tau_v,
            "time_s": dt, "n_params": n_param}


if __name__ == "__main__":
    torch.manual_seed(42); np.random.seed(42)

    A_PATH = "probe_results/A_train_n64_10k.npy"  # actually 20k
    print(f"Loading {A_PATH}...", flush=True)
    A_all = np.load(A_PATH).astype(np.float32)
    print(f"  shape={A_all.shape}", flush=True)

    # Compute teacher for ALL samples once
    print("\nComputing teachers (mixed025)...", flush=True)
    Ag_full = A_all.mean(axis=0)
    Wg_full = 0.5*(Ag_full + Ag_full.T); np.fill_diagonal(Wg_full, 0.0)
    teacher_all = compute_teachers(A_all, Wg_full)

    # Test multiple training sizes
    results = []
    for ts in [2000, 5000, 8000, 12000, 16000]:
        results.append(test_size(ts, A_all, teacher_all))

    print(f"\n{'='*50}")
    print(f"{'Size':>8s}  {'τ_train':>8s}  {'τ_val':>8s}  {'Time':>8s}")
    for r in results:
        print(f"{r['train_size']:>8d}  {r['tau_train']:>8.4f}  {r['tau_val']:>8.4f}  {r['time_s']:>7.0f}s")

    out = os.path.join(os.path.dirname(A_PATH), "on_gen_results.json")
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")
