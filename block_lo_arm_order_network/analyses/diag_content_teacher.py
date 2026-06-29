"""Quick diagnostic: content-only teacher signal for L0H0 at step 5000."""
import sys, json, numpy as np, torch
from collections import Counter
from scipy.stats import kendalltau

_HERE = "/home/admin/lyuyuhuan/order_lyu/block_lo_arm_order_network"
sys.path.insert(0, _HERE)

from neural_readout.extract_b import _load_model_and_chunks
from per_head_order_scan import _per_sample_A, _batch_mean_B
from neural_readout.teacher_labels import generate_teacher_label
from batch_readout.diversity_batch import teacher_diversity_stats
from clean_training_protocol import expand_model_blocks_to_token_order

ckpt = f"{_HERE}/probe_results/clean_base_random_perm/ckpt_step5000.pt"
M, batch_size = 200, 16
seed, head = 42, (0, 0)
device = "cuda:1"

print(f"Loading model + {M*batch_size} chunks...")
model, chunks, clean_perm, dev, _ = _load_model_and_chunks(ckpt, M*batch_size, seed, device, "train")
inv_perm = clean_perm.inv_perm_model_to_phys.cpu().numpy()

print(f"Extracting content-only L0H0 ({M*batch_size} forwards)...")
A_list = []
model.eval()
for i in range(M * batch_size):
    tokens = chunks[i:i+1].to(dev)
    gen = torch.Generator(device='cpu'); gen.manual_seed(int(seed) + i)
    rand_blocks = torch.randperm(64, generator=gen, device='cpu')
    token_order = expand_model_blocks_to_token_order(rand_blocks.unsqueeze(0), 4).to(dev)
    with torch.no_grad():
        _, _, attn_list = model.forward_fn(tokens, token_order, return_attentions=True)
    attn_stack = torch.stack(attn_list).detach().squeeze(1).cpu().numpy()
    reveal_tokens = token_order[0].cpu().numpy()
    A_i, _ = _per_sample_A(attn_stack, reveal_tokens, inv_perm, n_top=4, none_mode='content', head=head)
    A_list.append(A_i[0, 0])
    if (i+1) % 400 == 0:
        print(f"  forward {i+1}/{M*batch_size}", flush=True)

A_all = np.stack(A_list)
B_batch = _batch_mean_B(A_all, M, batch_size)

print(f"\nCDL on {M} content-only B graphs...")
sigmas = np.zeros((M, 64), dtype=np.int64)
for m in range(M):
    sigmas[m], _, _ = generate_teacher_label(B_batch[m], alpha_dep=0.5)

unique = len(set(tuple(s.tolist()) for s in sigmas))
print(f"\n=== RESULTS ===")
print(f"Unique orderings: {unique}/{M} ({unique/M:.3f})")
div = teacher_diversity_stats(sigmas)
print(f"Teacher pw_tau: {div['mean_pairwise_tau']:.4f}  first_H: {div['first_step_entropy']:.2f}")

# Post-hoc remap to physical for tau vs L2R
l2r = np.arange(64)
taus = []
for s in sigmas:
    s_phys = inv_perm[s]
    t, _ = kendalltau(s_phys, l2r)
    if not np.isnan(t):
        taus.append(t)
taus = np.array(taus)
print(f"\nTau vs L2R (post-hoc remap to physical):")
print(f"  mean={taus.mean():.4f}  std={taus.std():.4f}")
print(f"  min={taus.min():.4f}  max={taus.max():.4f}")
print(f"  |tau|>0.3: {(np.abs(taus)>0.3).mean():.3f}")
print(f"  |tau|>0.5: {(np.abs(taus)>0.5).mean():.3f}")
print(f"  |tau|>0.7: {(np.abs(taus)>0.7).mean():.3f}")

# First block distribution
first = Counter([s[0] for s in sigmas])
print(f"\nFirst block (top 10):")
for blk, cnt in first.most_common(10):
    print(f"  model_blk={blk:2d} (phys={inv_perm[blk]:2d}): {cnt:3d}/{M} ({cnt/M:.2f})")

# Consensus order
phys_ranks = np.array([inv_perm[s] for s in sigmas])
mean_pos = phys_ranks.mean(axis=0)
order_by_pos = np.argsort(mean_pos)
consensus_tau, _ = kendalltau(order_by_pos, l2r)
print(f"\nConsensus order tau vs L2R: {consensus_tau:.4f}")
print(f"Consensus first 10: {order_by_pos[:10].tolist()}")
print(f"Consensus last  10: {order_by_pos[-10:][::-1].tolist()}")

# Anti-L2R check
neg_tau_count = (taus < -0.3).sum()
print(f"\nStrongly anti-L2R (tau<-0.3): {neg_tau_count}/{M}")
print(f"Strongly pro-L2R  (tau>+0.3): {(taus>0.3).sum()}/{M}")
