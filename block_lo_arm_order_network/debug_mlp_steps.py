"""Step-by-step MLP adapter decision analysis."""
import sys, os, torch, numpy as np
sys.path.insert(0, '.')
from config import RerankerConfig
from reranker import (
    load_old_on, OldONPrior,
    build_reranker_features_v2, StepWiseMLPReranker,
    generate_order_A_greedy, compute_path_weight,
)
cfg = RerankerConfig()
device = 'cuda:0'
torch.manual_seed(cfg.seed)

old_on = load_old_on(cfg.old_on_ckpt, device)
old_on_prior = OldONPrior(old_on, num_blocks=cfg.num_blocks)

ckpt = torch.load('probe_results/reranker/mlp_reranker_on_adapter_v3_qvalue.pt', map_location=device)
mlp = StepWiseMLPReranker(feature_dim=11, hidden_dim=64, num_layers=2, dropout=0.0).to(device)
mlp.load_state_dict(ckpt['model_state_dict'])
mlp.eval()

A_all = np.load(cfg.data_path, mmap_mode='r')
A_batch = torch.from_numpy(A_all[100:103].copy()).float().to(device)

for s in range(3):
    A_s = A_batch[s:s+1]
    sigma_old = old_on_prior.get_full_order(A_s)[0]
    sigma_edge = generate_order_A_greedy(A_s, mode='edge')[0]
    A_np = A_s[0].cpu().numpy()

    print()
    print('=' * 70)
    on_w = compute_path_weight(A_np, sigma_old.cpu().numpy())
    edge_w = compute_path_weight(A_np, sigma_edge.cpu().numpy())
    print(f'Seq {s}:  old ON W={on_w:.4f}  edge-greedy W={edge_w:.4f}')
    print('=' * 70)

    visited_mask = 0
    mlp_order = []
    agree_on = 0
    agree_edge = 0

    for t in range(16):
        candidates = [i for i in range(16) if not (visited_mask & (1 << i))]
        if not candidates:
            break

        last_val = mlp_order[-1] if t > 0 else 0
        visited_t = torch.tensor([visited_mask], dtype=torch.long, device=device)
        last_t = torch.tensor([last_val], dtype=torch.long, device=device)

        features = build_reranker_features_v2(
            A_s, visited_t, last_t, old_on_prior,
            sigma_old=sigma_old.unsqueeze(0), step_t=t,
            feature_set='v3_qvalue',
        )
        cand_feat = features[0, candidates]

        with torch.no_grad():
            scores = mlp.mlp(cand_feat).squeeze(-1)

        sorted_idx = scores.argsort(descending=True)
        best = candidates[sorted_idx[0].item()]
        old_on_next = sigma_old[t].item()
        edge_best = max(candidates, key=lambda i: A_np[i, last_val] if t > 0 else A_np[:, i].mean())

        if best == old_on_next:
            agree_on += 1
        if best == edge_best:
            agree_edge += 1

        print(f'  t={t:2d} K={len(candidates):2d}  MLP->{best:2d}  ON->{old_on_next:2d}  edge->{edge_best:2d}  '
              f'on={"Y" if best==old_on_next else "N"}  edge={"Y" if best==edge_best else "N"}  '
              f'score={scores[sorted_idx[0]].item():.3f}')

        if t < 4:
            for k in range(min(3, len(sorted_idx))):
                idx = sorted_idx[k].item()
                cand = candidates[idx]
                tags = []
                if cand == old_on_next:
                    tags.append('ON')
                if cand == edge_best:
                    tags.append('edge')
                tag = ','.join(tags) if tags else ''
                print(f'       #{k+1}: blk={cand:2d} score={scores[idx].item():.4f} {tag}')

        mlp_order.append(best)
        visited_mask |= (1 << best)

    mlp_order_t = torch.tensor(mlp_order)
    mlp_w = compute_path_weight(A_np, mlp_order_t.cpu().numpy())
    print(f'  ---')
    print(f'  Final: MLP={mlp_w:.4f}  ON={on_w:.4f}  edge={edge_w:.4f}')
    print(f'  MLP==ON: {agree_on}/15  MLP==edge: {agree_edge}/15')
