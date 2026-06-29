"""Train prefix-conditioned MLP adapter with old ON as teacher.

Old ON gives a complete order prior σ_old.
At each step t, the MLP learns to predict σ_old[t] — "what does old ON want next?"
Because the MLP sees richer features (raw A-matrix) than old ON's internal representation,
it may learn to override old ON when A-matrix strongly suggests a better edge.

Core question: does old ON's complete order prior help in step-wise decision making?
  MLP > old ON greedy → prior useful, prefix conditioning works
  MLP ≤ old ON greedy → prior doesn't help, need true V1 sequential ON
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RerankerConfig
from reranker import (
    load_old_on,
    OldONPrior,
    build_reranker_features,
    build_reranker_features_v2,
    StepWiseMLPReranker,
    StepWisePooledMLPReranker,
    sequential_generate,
    check_order_valid,
    compute_path_weight,
    generate_order_A_greedy,
    generate_on_adapter_labels,
)


# ── Training ──────────────────────────────────────────────────────────────────

def train_mlp_adapter(labels, cfg, device):
    """Train MLP to predict old ON's next block."""
    feature_dim = cfg.adapter_feature_dim

    if cfg.use_pooled_context:
        model = StepWisePooledMLPReranker(
            feature_dim=feature_dim,
            local_hidden_dim=cfg.adapter_pooled_hidden_dim,
            local_num_layers=cfg.adapter_pooled_num_layers,
            head_hidden_dim=cfg.adapter_hidden_dim,
            head_num_layers=cfg.adapter_num_layers,
            dropout=cfg.adapter_dropout,
        ).to(device)
    else:
        model = StepWiseMLPReranker(
            feature_dim=feature_dim,
            hidden_dim=cfg.adapter_hidden_dim,
            num_layers=cfg.adapter_num_layers,
            dropout=cfg.adapter_dropout,
        ).to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.adapter_lr,
        weight_decay=cfg.adapter_weight_decay,
    )

    n_queries = len(labels)
    print(f"Training on {n_queries} per-step queries from "
          f"{len(set(l['seq_idx'] for l in labels))} sequences")
    print(f"  Model: {'PooledMLP' if cfg.use_pooled_context else 'MLP'}, "
          f"features={cfg.reranker_feature_set}, feature_dim={feature_dim}, "
          f"loss={cfg.loss_type}")

    for epoch in range(cfg.adapter_epochs):
        perm = torch.randperm(n_queries)
        total_loss = 0.0
        n_batches = 0

        for batch_start in range(0, n_queries, cfg.adapter_batch_size):
            batch_end = min(batch_start + cfg.adapter_batch_size, n_queries)
            batch_indices = perm[batch_start:batch_end]

            optimizer.zero_grad()
            batch_losses = []

            for idx in batch_indices:
                lab = labels[idx]
                features = lab["features"].to(device)  # (K, F)
                struct_targets = lab["struct_targets"].to(device)  # (K,)
                K = features.shape[0]

                if cfg.use_pooled_context:
                    revealed_mask = torch.zeros(1, K, dtype=torch.bool, device=device)
                    scores = model(features.unsqueeze(0), revealed_mask).squeeze(0)
                else:
                    scores = model.mlp(features).squeeze(-1)  # (K,)

                if cfg.loss_type == "hard_ce":
                    target = struct_targets.argmax(dim=0)
                    loss = F.cross_entropy(scores.unsqueeze(0), target.unsqueeze(0))
                else:
                    eps = 1e-6
                    q = (1 - eps) * struct_targets + eps / K
                    loss = -(q * F.log_softmax(scores, dim=0)).sum()

                batch_losses.append(loss)

            if len(batch_losses) == 0:
                continue
            batch_loss = torch.stack(batch_losses).mean()
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1}/{cfg.adapter_epochs}, loss={avg_loss:.6f}")

    return model


# ── Sanity overfit ─────────────────────────────────────────────────────────────

def sanity_overfit_adapter(labels, cfg, device, n_seqs=10):
    """Train MLP on tiny subset to verify it can fit old ON labels."""
    subset_indices = [i for i, lab in enumerate(labels) if lab["seq_idx"] < n_seqs]
    subset_labels = [labels[i] for i in subset_indices]

    feature_dim = cfg.adapter_feature_dim
    print(f"Sanity overfit: {len(subset_labels)} queries from {n_seqs} seqs, "
          f"feature_dim={feature_dim}, loss={cfg.loss_type}")

    if cfg.use_pooled_context:
        model = StepWisePooledMLPReranker(
            feature_dim=feature_dim,
            local_hidden_dim=cfg.adapter_pooled_hidden_dim,
            local_num_layers=cfg.adapter_pooled_num_layers,
            head_hidden_dim=cfg.adapter_hidden_dim,
            head_num_layers=cfg.adapter_num_layers,
            dropout=0.0,
        ).to(device)
    else:
        model = StepWiseMLPReranker(
            feature_dim=feature_dim,
            hidden_dim=cfg.adapter_hidden_dim,
            num_layers=cfg.adapter_num_layers,
            dropout=0.0,
        ).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)

    for epoch in range(500):
        total_loss = 0.0
        for lab in subset_labels:
            features = lab["features"].to(device)
            struct_targets = lab["struct_targets"].to(device)
            K = features.shape[0]

            if cfg.use_pooled_context:
                revealed_mask = torch.zeros(1, K, dtype=torch.bool, device=device)
                scores = model(features.unsqueeze(0), revealed_mask).squeeze(0)
            else:
                scores = model.mlp(features).squeeze(-1)

            if cfg.loss_type == "hard_ce":
                target = struct_targets.argmax(dim=0)
                loss = F.cross_entropy(scores.unsqueeze(0), target.unsqueeze(0))
            else:
                eps = 1e-6
                q = (1 - eps) * struct_targets + eps / K
                loss = -(q * F.log_softmax(scores, dim=0)).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 100 == 0:
            correct = 0
            total = 0
            with torch.no_grad():
                for lab in subset_labels:
                    features = lab["features"].to(device)
                    struct_targets = lab["struct_targets"].to(device)
                    K = features.shape[0]
                    if cfg.use_pooled_context:
                        revealed_mask = torch.zeros(1, K, dtype=torch.bool, device=device)
                        scores = model(features.unsqueeze(0), revealed_mask).squeeze(0)
                    else:
                        scores = model.mlp(features).squeeze(-1)
                    pred_best = scores.argmax().item()
                    true_best = struct_targets.argmax().item()
                    total += 1
                    if pred_best == true_best:
                        correct += 1
            acc = correct / max(total, 1)
            print(f"  Epoch {epoch + 1}: loss={total_loss / len(subset_labels):.6f}, "
                  f"top1_acc={acc:.3f}")

    return model


# ── Validation (structural metrics) ────────────────────────────────────────────

@torch.no_grad()
def validate_adapter(
    reranker, old_on_prior, A_batch, cfg
):
    """Validate adapter with structural path weight metrics. No NLL."""
    device = A_batch.device
    num_seqs = len(A_batch)
    A_np = A_batch.cpu().numpy()

    mlp_weights = []
    old_on_weights = []
    greedy_weights = []
    random_weights = []

    for s in tqdm(range(num_seqs), desc="Validating adapter"):
        A_s = A_batch[s:s + 1]
        sigma_old = old_on_prior.get_full_order(A_s)[0]

        # MLP adapter order
        mlp_order = sequential_generate(
            reranker, A_s, old_on_prior,
            sigma_old=sigma_old.unsqueeze(0),
            temperature=0.0,
            use_greedy_reranker=False,
            feature_set=cfg.reranker_feature_set,
        )[0]
        assert check_order_valid(mlp_order), f"Invalid MLP order"

        # A-greedy order
        greedy_order = generate_order_A_greedy(A_s, mode="edge")[0]

        # Compute path weights (edge mode)
        A = A_np[s]
        mlp_weights.append(compute_path_weight(A, mlp_order.cpu().numpy(), mode="edge"))
        old_on_weights.append(compute_path_weight(A, sigma_old.cpu().numpy(), mode="edge"))
        greedy_weights.append(compute_path_weight(A, greedy_order.cpu().numpy(), mode="edge"))

        # Random MC
        rand_ws = []
        for _ in range(cfg.num_random_mc):
            rand_order = np.random.permutation(cfg.num_blocks)
            rand_ws.append(compute_path_weight(A, rand_order, mode="edge"))
        random_weights.append(np.mean(rand_ws))

    mlp_mean = np.mean(mlp_weights)
    old_on_mean = np.mean(old_on_weights)
    greedy_mean = np.mean(greedy_weights)
    random_mean = np.mean(random_weights)

    results = {
        "mlp_weight": mlp_mean,
        "old_on_weight": old_on_mean,
        "greedy_A_weight": greedy_mean,
        "random_weight": random_mean,
        "delta_vs_old_on": mlp_mean - old_on_mean,
        "delta_vs_greedy_A": mlp_mean - greedy_mean,
        "delta_vs_random": mlp_mean - random_mean,
    }
    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--skip-labels", action="store_true")
    parser.add_argument("--labels-file", type=str, default="")
    parser.add_argument("--skip-sanity", action="store_true")
    args = parser.parse_args()

    cfg = RerankerConfig()
    if args.output_dir:
        cfg.output_dir = args.output_dir
    os.makedirs(cfg.output_dir, exist_ok=True)
    device = args.device

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    cfg.reranker_feature_set = "v3_qvalue"
    cfg.loss_type = "hard_ce"

    # ── Load old ON ──
    print("Loading Old ON...")
    old_on = load_old_on(cfg.old_on_ckpt, device)
    old_on_prior = OldONPrior(old_on, num_blocks=cfg.num_blocks)

    # ── Load A matrices ──
    print("Loading A matrices...")
    A_all = np.load(cfg.data_path, mmap_mode="r")
    total_seqs = min(A_all.shape[0], cfg.num_train_seqs + cfg.num_tune_seqs + cfg.num_test_seqs)
    A_all = torch.from_numpy(A_all[:total_seqs].copy()).float().to(device)

    n_train = cfg.num_train_seqs
    n_tune = cfg.num_tune_seqs

    A_train = A_all[:n_train]
    A_tune = A_all[n_train:n_train + n_tune]

    print(f"Data: train={n_train}, tune={n_tune}")

    # ── Generate or load ON-adapter labels (no DP) ──
    if not args.labels_file:
        args.labels_file = os.path.join(
            cfg.output_dir, "labels_on_adapter_v3_qvalue.pt"
        )
    if args.skip_labels and os.path.exists(args.labels_file):
        print(f"Loading cached ON-adapter labels from {args.labels_file}")
        labels_train = torch.load(args.labels_file, weights_only=False)
    else:
        print("Generating ON-adapter labels (teacher = old ON)...")
        t0 = time.time()
        labels_train = generate_on_adapter_labels(
            A_train, old_on_prior,
            feature_set=cfg.reranker_feature_set,
            include_edge_greedy_states=True,
        )
        print(f"Label generation took {time.time() - t0:.1f}s, {len(labels_train)} queries")
        torch.save(labels_train, args.labels_file)
        print(f"Saved labels to {args.labels_file}")

    # Auto-detect feature_dim
    cfg.adapter_feature_dim = labels_train[0]["features"].shape[-1]
    print(f"Auto-detected feature_dim={cfg.adapter_feature_dim}")

    # ── Sanity overfit ──
    if not args.skip_sanity:
        print("\n=== Sanity Overfit (old ON teacher) ===")
        sanity_overfit_adapter(labels_train, cfg, device, n_seqs=10)

    # ── Train ──
    print("\n=== Training MLP Adapter (old ON teacher) ===")
    t0 = time.time()
    mlp_adapter = train_mlp_adapter(labels_train, cfg, device)
    print(f"Training took {time.time() - t0:.1f}s")
    print(f"MLP params: {mlp_adapter.count_parameters():,}")

    suffix = "_on_adapter_v3_qvalue"
    ckpt_path = os.path.join(cfg.output_dir, f"mlp_reranker{suffix}.pt")
    torch.save({
        "model_state_dict": mlp_adapter.state_dict(),
        "config": cfg,
    }, ckpt_path)
    print(f"Saved to {ckpt_path}")

    # ── Validate ──
    print("\n=== Validation (Structural Metrics) ===")
    mlp_adapter.eval()
    results = validate_adapter(
        mlp_adapter, old_on_prior, A_tune, cfg
    )

    print(f"\nVal results ({n_tune} seqs, edge path weight):")
    print(f"  MLP adapter weight:      {results['mlp_weight']:.4f}")
    print(f"  A-greedy weight:         {results['greedy_A_weight']:.4f}")
    print(f"  Old ON weight:           {results['old_on_weight']:.4f}")
    print(f"  Random weight:           {results['random_weight']:.4f}")
    print(f"  Δ vs Old ON:             {results['delta_vs_old_on']:+.4f}")
    print(f"  Δ vs A-greedy:           {results['delta_vs_greedy_A']:+.4f}")
    print(f"  Δ vs Random:             {results['delta_vs_random']:+.4f}")

    results_path = os.path.join(cfg.output_dir, f"train_results{suffix}.pt")
    torch.save(results, results_path)
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
