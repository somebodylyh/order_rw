"""
Train ON32 (CrossAttentionOrderNetwork, num_blocks=32) with soft A-edge teacher.

Teacher:
  Step 0: q_i = softmax(row_mean_i / τ_start)  — prefer blocks with high mean attention
  Step t>0: q_i = softmax(A[i, last] / τ_teacher) for unvisited i

Training: CE(q, log_softmax(on_logits)) — distribution matching, NOT argmax imitation.

Success criteria (not step accuracy):
  1. Sampled order path weight > random
  2. Entropy / effective subset size reasonable
  3. Downstream ON-mixed training improves AO-GPT vs random continual

Usage:
    python -u train_on32.py --device cuda:0
"""

import argparse, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from order_network import CrossAttentionOrderNetwork

NUM_BLOCKS = 32
A32_PATH = "probe_results/A_train_n32_10k.npy"
OUTPUT_DIR = "probe_results"


# ── Soft label generation ──

def generate_soft_labels(A_batch, tau_teacher=0.1, tau_start=0.5):
    """Generate per-step soft label distributions for ON training.

    A_batch: (N_seq, 32, 32) in model coordinates.
    Returns: list of dicts with A_idx, step, visited_mask, last_node, soft_targets.
    """
    N_seq, N, _ = A_batch.shape
    labels = []

    for s in range(N_seq):
        A = A_batch[s]  # (32, 32)
        visited_mask = 0
        last = 0

        for t in range(N - 1):  # last step has only 1 candidate, no learning signal
            candidates = [i for i in range(N) if not (visited_mask & (1 << i))]
            K = len(candidates)

            if t == 0:
                # Step 0: softmax over row means
                row_mean = A.mean(dim=1)  # (32,)
                raw = row_mean[candidates] / tau_start
            else:
                # Step t>0: softmax over A[candidate, last]
                raw = A[candidates, last] / tau_teacher

            q = F.softmax(raw, dim=0)  # (K,)

            labels.append({
                "seq_idx": s,
                "step": t,
                "visited_mask": visited_mask,
                "last_node": last,
                "soft_targets": q,  # (K,)
                "candidates": candidates,  # list of block indices
            })

            # Advance to next step (use argmax for state transition, not sampling)
            best = candidates[q.argmax().item()]
            visited_mask |= (1 << best)
            last = best

    return labels


# ── Dataset ──

class SoftLabelDataset(Dataset):
    """Pre-tensorized dataset: (A, visited_mask, last_node, soft_targets).

    soft_targets: (32,) already-normalized probabilities, 0 for visited positions.
    """

    def __init__(self, labels, A_all, subset_indices):
        self.A_all = A_all  # (N_seq, 32, 32) tensor
        # Pre-tensorize all examples
        self.A_list = []
        self.visited_list = []
        self.last_list = []
        self.targets_list = []

        for lab in labels:
            if lab["seq_idx"] in subset_indices:
                N = A_all.shape[1]
                soft_targets = torch.zeros(N)
                for i, cand in enumerate(lab["candidates"]):
                    soft_targets[cand] = lab["soft_targets"][i]
                self.A_list.append(A_all[lab["seq_idx"]])
                self.visited_list.append(torch.tensor(lab["visited_mask"], dtype=torch.long))
                self.last_list.append(torch.tensor(lab["last_node"], dtype=torch.long))
                self.targets_list.append(soft_targets)

    def __len__(self):
        return len(self.A_list)

    def __getitem__(self, idx):
        return (
            self.A_list[idx],
            self.visited_list[idx],
            self.last_list[idx],
            self.targets_list[idx],
        )


# ── Training ──

def train_on32(model, train_loader, val_loader, epochs, lr, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for A, visited, last, targets in train_loader:
            A = A.to(device)
            visited = visited.to(device)
            last = last.to(device)
            targets = targets.to(device)  # (B, 32), already softmaxed, 0 for visited

            logits = model(A, visited, last)  # (B, 32), visited=-inf
            log_probs = F.log_softmax(logits, dim=-1)
            # CE: -sum(q * log(p)), mask out visited (targets==0 → 0*-inf=NaN)
            safe_log_probs = torch.where(targets > 0, log_probs, torch.zeros_like(log_probs))
            loss = -(targets * safe_log_probs).sum(dim=-1).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        # Validation
        if (epoch + 1) % 20 == 0:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for A, visited, last, targets in val_loader:
                    A = A.to(device)
                    visited = visited.to(device)
                    last = last.to(device)
                    targets = targets.to(device)
                    logits = model(A, visited, last)
                    log_probs = F.log_softmax(logits, dim=-1)
                    safe_log_probs = torch.where(targets > 0, log_probs, torch.zeros_like(log_probs))
                    loss = -(targets * safe_log_probs).sum(dim=-1).mean()
                    val_loss += loss.item()
                    val_batches += 1
            val_loss /= max(val_batches, 1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                    "num_blocks": NUM_BLOCKS,
                }, os.path.join(OUTPUT_DIR, "on32_soft_edge_best.pt"))

            print(f"  Epoch {epoch+1}/{epochs}: train_loss={avg_loss:.6f}, "
                  f"val_loss={val_loss:.6f}, best={best_val_loss:.6f}")

    return model


# ── Evaluation ──

@torch.no_grad()
def evaluate_on32(model, A_batch, temperature=1.0, num_random_mc=20):
    """Evaluate ON32 with structural path weight metrics."""
    from reranker import compute_path_weight, OldONPrior

    device = A_batch.device
    N_seq = len(A_batch)
    A_np = A_batch.cpu().numpy()

    on_weights = []
    greedy_weights = []
    random_weights = []

    # Edge-greedy orders for comparison
    for s in range(N_seq):
        A = A_np[s]
        # ON32 sampled order
        visited = torch.zeros(1, dtype=torch.long, device=device)
        last = torch.zeros(1, dtype=torch.long, device=device)
        A_s = A_batch[s:s + 1]
        order = []
        for t in range(NUM_BLOCKS):
            logits = model(A_s, visited, last)
            if temperature < 1e-8:
                chosen = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                chosen = torch.multinomial(probs, 1).squeeze(-1)
            order.append(chosen.item())
            visited |= (1 << chosen)
            last = chosen
        on_weights.append(compute_path_weight(A, np.array(order)))

        # Edge-greedy
        from train_on_mixed import sample_edge_greedy_order
        eg_order = sample_edge_greedy_order(A_s).squeeze(0).cpu().numpy()
        greedy_weights.append(compute_path_weight(A, eg_order))

        # Random MC
        rw = [compute_path_weight(A, np.random.permutation(NUM_BLOCKS))
              for _ in range(num_random_mc)]
        random_weights.append(np.mean(rw))

    return {
        "on_weight": float(np.mean(on_weights)),
        "greedy_weight": float(np.mean(greedy_weights)),
        "random_weight": float(np.mean(random_weights)),
        "delta_vs_random": float(np.mean(on_weights) - np.mean(random_weights)),
        "delta_vs_greedy": float(np.mean(on_weights) - np.mean(greedy_weights)),
    }


# ── Main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tau-teacher", type=float, default=0.01,
                        help="Temperature for soft edge teacher (A values ~0.01-0.1, lower=sharper)")
    parser.add_argument("--tau-start", type=float, default=0.3,
                        help="Temperature for step-0 softmax over row means")
    parser.add_argument("--train-seqs", type=int, default=8000)
    parser.add_argument("--val-seqs", type=int, default=1000)
    parser.add_argument("--eval-seqs", type=int, default=200)
    parser.add_argument("--d-edge", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.multiprocessing.set_sharing_strategy('file_system')  # avoid CUDA sharing issues

    # ── Load A32 ──
    print("Loading A32...")
    A_all = np.load(A32_PATH, mmap_mode="r")
    n_avail = min(A_all.shape[0], args.train_seqs + args.val_seqs + args.eval_seqs)
    A_all = torch.from_numpy(A_all[:n_avail].copy()).float()

    # ── Generate soft labels ──
    print(f"Generating soft labels (τ_teacher={args.tau_teacher}, τ_start={args.tau_start})...")
    t0 = time.time()
    n_label_seqs = min(args.train_seqs + args.val_seqs, n_avail)
    labels = generate_soft_labels(A_all[:n_label_seqs], args.tau_teacher, args.tau_start)
    print(f"  {len(labels)} per-step queries from {n_label_seqs} seqs, {time.time() - t0:.1f}s")

    # ── Split train/val by sequence ──
    rng = np.random.RandomState(42)
    seq_order = rng.permutation(n_label_seqs)
    train_seq_set = set(seq_order[:args.train_seqs].tolist())
    val_seq_set = set(seq_order[args.train_seqs:args.train_seqs + args.val_seqs].tolist())

    train_ds = SoftLabelDataset(labels, A_all, train_seq_set)
    val_ds = SoftLabelDataset(labels, A_all, val_seq_set)
    print(f"  Train: {len(train_ds)} queries, Val: {len(val_ds)} queries")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    # ── Model ──
    model = CrossAttentionOrderNetwork(
        num_blocks=NUM_BLOCKS, d_edge=args.d_edge, d_model=args.d_model
    ).to(device)
    print(f"ON32 params: {model.count_parameters():,}")

    # ── Train ──
    print(f"\nTraining ON32 ({args.epochs} epochs, lr={args.lr})...")
    t0 = time.time()
    model = train_on32(model, train_loader, val_loader, args.epochs, args.lr, device)
    print(f"Training took {time.time() - t0:.1f}s")

    # ── Evaluate ──
    print("\nEvaluating ON32 (structural path weight)...")
    eval_A = A_all[n_label_seqs:n_label_seqs + args.eval_seqs].to(device)
    results = evaluate_on32(model, eval_A, temperature=1.0)
    print(f"  ON32 sampled (τ=1.0):    {results['on_weight']:.4f}")
    print(f"  Edge-greedy:             {results['greedy_weight']:.4f}")
    print(f"  Random:                  {results['random_weight']:.4f}")
    print(f"  Δ vs Random:             {results['delta_vs_random']:+.4f}")
    print(f"  Δ vs Greedy:             {results['delta_vs_greedy']:+.4f}")

    # Save final ckpt
    ckpt_path = os.path.join(args.output_dir, "on32_soft_edge_final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_blocks": NUM_BLOCKS,
        "d_edge": args.d_edge,
        "d_model": args.d_model,
        "eval_results": results,
    }, ckpt_path)
    print(f"Saved: {ckpt_path}")


if __name__ == "__main__":
    main()
