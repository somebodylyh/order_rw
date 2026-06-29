"""Train Step-Wise MLP Reranker with AOGPT per-step candidate-local NLL labels.

Flow:
1. Load A matrices, old ON (frozen), AO-GPT (frozen)
2. Load wikitext token chunks matching A matrices
3. Generate per-step candidate-local NLL labels (old ON greedy prefix)
4. Train MLP reranker with listwise soft CE loss
5. Validate on held-out val set
6. Sanity overfit on 10 sequences

Usage:
    python -u train_reranker.py --device cuda:0
"""

import argparse
import os
import sys
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
from tqdm import tqdm

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RerankerConfig
from reranker import (
    load_old_on,
    load_aogpt,
    OldONPrior,
    build_reranker_features,
    build_reranker_features_v2,
    StepWiseMLPReranker,
    StepWisePooledMLPReranker,
    sequential_generate,
    compute_batched_candidate_nlls,
    compute_batch_full_order_nll,
    n16_order_to_token_order,
    masks_to_revealed_bool,
    check_order_valid,
)


def load_token_chunks(n_chunks: int, tokenizer_dir: str, wikitext_dir: str):
    """Load token chunks from wikitext-103 TRAIN set. Deterministic chunking."""
    from datasets import Dataset
    from transformers import GPT2TokenizerFast

    tok = GPT2TokenizerFast.from_pretrained(tokenizer_dir, local_files_only=True)
    chunks = []
    buffer_ids = []

    shards = ["wikitext-train-00000-of-00002.arrow", "wikitext-train-00001-of-00002.arrow"]
    for shard in shards:
        ds = Dataset.from_file(os.path.join(wikitext_dir, shard))
        for ex in tqdm(ds, desc=f"Chunking {shard}", unit=" texts"):
            ids = tok.encode(ex["text"])
            if len(ids) < 256:
                buffer_ids.extend(ids)
                while len(buffer_ids) >= 256:
                    chunks.append(torch.tensor(buffer_ids[:256], dtype=torch.long))
                    buffer_ids = buffer_ids[256:]
                    if len(chunks) >= n_chunks:
                        break
            else:
                for start in range(0, len(ids) - 256 + 1, 128):
                    chunks.append(torch.tensor(ids[start:start + 256], dtype=torch.long))
                    if len(chunks) >= n_chunks:
                        break
            if len(chunks) >= n_chunks:
                break
        if len(chunks) >= n_chunks:
            break
    print(f"Loaded {len(chunks)} chunks")
    return torch.stack(chunks)


def phys_to_model_idx(idx_phys, inv_perm, blk_size=4):
    """Convert token sequences from physical to model coordinates."""
    B, T = idx_phys.shape
    device = idx_phys.device
    idx_model = torch.zeros_like(idx_phys)
    for s in range(T):
        model_block = inv_perm[s // blk_size].item()
        offset = s % blk_size
        model_pos = model_block * blk_size + offset
        idx_model[:, model_pos] = idx_phys[:, s]
    return idx_model


# ── Label generation ──────────────────────────────────────────────────────────

def generate_labels(
    aogpt, old_on_prior, idx_batch, block_perm, inv_perm, A_batch, cfg
):
    """Generate per-step candidate-local NLL labels for training sequences.

    For each sequence, runs old ON autoregressively to get prefix,
    then evaluates each candidate's AOGPT NLL at each step.

    Returns:
        labels: list of dicts, each dict contains:
            seq_idx, step, features (K,7), nll_targets (K,)
    """
    num_seqs = len(A_batch)
    N = cfg.num_blocks
    device = idx_batch.device

    # Convert idx to model coordinates
    idx_model = phys_to_model_idx(idx_batch, inv_perm, blk_size=cfg.block_len)

    all_labels = []

    for s in tqdm(range(num_seqs), desc="Generating labels"):
        A_s = A_batch[s:s + 1]  # (1, N, N)

        # Get old ON full order (for prefix distribution and rank_distance feature)
        sigma_old = old_on_prior.get_full_order(A_s)[0]  # (N,)

        # Determine rest order (L2R of remaining blocks, used for NLL computation)
        all_blocks = torch.arange(N, device=device)

        idx_s = idx_model[s:s + 1]  # (1, 256)

        for t in range(N):
            prefix = sigma_old[:t]  # t blocks already placed (t=0 → empty)
            candidate_set = all_blocks[~torch.isin(all_blocks, prefix)]  # (N-t,)

            if len(candidate_set) == 0:
                break

            # Build state
            visited_mask = 0
            for blk in prefix:
                visited_mask |= (1 << int(blk.item()))
            visited_mask = torch.tensor([visited_mask], dtype=torch.long, device=device)
            last_node = prefix[-1:] if t > 0 else torch.zeros(1, dtype=torch.long, device=device)
            if t == 0:
                last_node = torch.zeros(1, dtype=torch.long, device=device)

            # Build features for all candidates
            if cfg.reranker_feature_set == "v1_legacy":
                features = build_reranker_features(
                    A_s, visited_mask, last_node, old_on_prior,
                    sigma_old=sigma_old.unsqueeze(0), step_t=t,
                )  # (1, N, 7)
            else:
                features = build_reranker_features_v2(
                    A_s, visited_mask, last_node, old_on_prior,
                    sigma_old=sigma_old.unsqueeze(0), step_t=t,
                    feature_set=cfg.reranker_feature_set,
                )  # (1, N, F)
            candidate_features = features[0, candidate_set]  # (K, F)

            # Rest order base: all blocks not in prefix (each candidate will be excluded individually)
            rest_base = all_blocks[~torch.isin(all_blocks, prefix)]

            # Compute AOGPT NLL for each candidate (batched)
            nlls = compute_batched_candidate_nlls(
                aogpt, idx_s, block_perm,
                prefix_order=prefix,
                candidates=candidate_set,
                rest_order=rest_base,
                chunk_size=cfg.label_chunk_size,
                sub_blocks=cfg.sub_blocks,
                block_len=cfg.block_len,
                k=cfg.label_k,
            )  # (K,)

            all_labels.append({
                "seq_idx": s,
                "step": t,
                "features": candidate_features.cpu(),
                "nll_targets": nlls.cpu(),
            })

    return all_labels


# ── Training ──────────────────────────────────────────────────────────────────

def train_mlp_reranker(labels, cfg, device):
    """Train StepWiseMLPReranker or StepWisePooledMLPReranker.

    Supports hard CE (target = argmin NLL) and soft CE loss.
    """
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
    print(f"Training on {n_queries} per-step queries from {len(set(l['seq_idx'] for l in labels))} sequences")
    print(f"  Model: {'PooledMLP' if cfg.use_pooled_context else 'MLP'}, features={cfg.reranker_feature_set}, "
          f"feature_dim={feature_dim}, loss={cfg.loss_type}")

    for epoch in range(cfg.adapter_epochs):
        # Shuffle queries
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
                nll_targets = lab["nll_targets"].to(device)  # (K,)
                K = features.shape[0]

                # MLP scores — need revealed_mask for pooled MLP (all False for training)
                if cfg.use_pooled_context:
                    revealed_mask = torch.zeros(1, K, dtype=torch.bool, device=device)
                    scores = model(features.unsqueeze(0), revealed_mask).squeeze(0)  # (K,)
                else:
                    scores = model.mlp(features).squeeze(-1)  # (K,)

                if cfg.loss_type == "hard_ce":
                    # Target = argmin NLL (the oracle best candidate)
                    target = (-nll_targets).argmax(dim=0)  # scalar
                    loss = F.cross_entropy(scores.unsqueeze(0), target.unsqueeze(0))
                else:
                    # Soft CE: q_i = softmax(-NLL_i / τ_label), with label smoothing
                    q_raw = F.softmax(-nll_targets / cfg.label_tau, dim=0)
                    eps = 1e-6
                    q = (1 - eps) * q_raw + eps / K
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


# ── Validation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate_reranker(
    reranker, aogpt, old_on_prior, idx_batch, block_perm, A_batch, cfg
):
    """Validate MLP reranker: full-order AOGPT NLL on val set."""
    device = idx_batch.device
    num_seqs = len(A_batch)
    N = cfg.num_blocks

    mlp_nlls = []
    old_on_nlls = []
    random_nlls = []

    for s in tqdm(range(num_seqs), desc="Validating"):
        A_s = A_batch[s:s + 1]
        sigma_old = old_on_prior.get_full_order(A_s)[0]

        # MLP reranker order
        mlp_order = sequential_generate(
            reranker, A_s, old_on_prior,
            sigma_old=sigma_old.unsqueeze(0),
            temperature=0.0,  # greedy
            use_greedy_reranker=False,
            feature_set=cfg.reranker_feature_set,
        )[0]
        assert check_order_valid(mlp_order, N), f"Invalid MLP order: {mlp_order}"

        # Old ON order
        old_on_order = sigma_old

        # Random N16 orders
        rand_orders = torch.stack([
            torch.randperm(N, device=device) for _ in range(cfg.num_random_mc)
        ])

        # Compute NLLs
        idx_s = idx_batch[s:s + 1]

        mlp_nll = compute_batch_full_order_nll(
            aogpt, idx_s.expand(1, -1), block_perm, mlp_order.unsqueeze(0),
        ).item()

        old_on_nll = compute_batch_full_order_nll(
            aogpt, idx_s.expand(1, -1), block_perm, old_on_order.unsqueeze(0),
        ).item()

        rand_batch = compute_batch_full_order_nll(
            aogpt, idx_s.expand(cfg.num_random_mc, -1), block_perm, rand_orders,
        )
        rand_nll = rand_batch.mean().item()

        mlp_nlls.append(mlp_nll)
        old_on_nlls.append(old_on_nll)
        random_nlls.append(rand_nll)

    mlp_mean = np.mean(mlp_nlls)
    old_on_mean = np.mean(old_on_nlls)
    random_mean = np.mean(random_nlls)

    results = {
        "mlp_nll": mlp_mean,
        "old_on_nll": old_on_mean,
        "random_n16_nll": random_mean,
        "delta_vs_old_on": old_on_mean - mlp_mean,
        "delta_vs_random": random_mean - mlp_mean,
    }
    return results


# ── Sanity overfit ─────────────────────────────────────────────────────────────

def sanity_overfit(labels, cfg, device, n_seqs=10):
    """Train MLP on a tiny subset to verify it can fit AOGPT labels.

    Supports hard CE / soft CE, v1/v2 features, and pooled MLP.
    """
    subset_indices = [i for i, lab in enumerate(labels) if lab["seq_idx"] < n_seqs]
    subset_labels = [labels[i] for i in subset_indices]

    feature_dim = cfg.adapter_feature_dim
    print(f"Sanity overfit: {len(subset_labels)} queries from {n_seqs} seqs, "
          f"feature_dim={feature_dim}, loss={cfg.loss_type}, "
          f"pooled={cfg.use_pooled_context}")

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

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-2, weight_decay=0.0
    )

    for epoch in range(500):
        total_loss = 0.0
        for lab in subset_labels:
            features = lab["features"].to(device)  # (K, F)
            nll_targets = lab["nll_targets"].to(device)  # (K,)
            K = features.shape[0]

            if cfg.use_pooled_context:
                revealed_mask = torch.zeros(1, K, dtype=torch.bool, device=device)
                scores = model(features.unsqueeze(0), revealed_mask).squeeze(0)  # (K,)
            else:
                scores = model.mlp(features).squeeze(-1)  # (K,)

            if cfg.loss_type == "hard_ce":
                target = (-nll_targets).argmax(dim=0)
                loss = F.cross_entropy(scores.unsqueeze(0), target.unsqueeze(0))
            else:
                q_raw = F.softmax(-nll_targets / cfg.label_tau, dim=0)
                eps = 1e-6
                q = (1 - eps) * q_raw + eps / K
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
                    nll_targets = lab["nll_targets"].to(device)
                    K = features.shape[0]
                    if cfg.use_pooled_context:
                        revealed_mask = torch.zeros(1, K, dtype=torch.bool, device=device)
                        scores = model(features.unsqueeze(0), revealed_mask).squeeze(0)
                    else:
                        scores = model.mlp(features).squeeze(-1)
                    pred_best = scores.argmax().item()
                    true_best = (-nll_targets).argmax().item()
                    total += 1
                    if pred_best == true_best:
                        correct += 1
            acc = correct / max(total, 1)
            print(f"  Epoch {epoch + 1}: loss={total_loss / len(subset_labels):.6f}, top1_acc_vs_AOGPT={acc:.3f}")

    return model


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--skip-labels", action="store_true",
                        help="Skip label generation, load from file")
    parser.add_argument("--labels-file", type=str,
                        default="")  # auto-set from feature_set
    parser.add_argument("--skip-sanity", action="store_true")
    args = parser.parse_args()

    cfg = RerankerConfig()
    if args.output_dir:
        cfg.output_dir = args.output_dir
    os.makedirs(cfg.output_dir, exist_ok=True)
    device = args.device

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ── Paths ──
    aogpt_ckpt = os.path.expanduser(cfg.aogpt_ckpt)
    tokenizer_dir = os.path.expanduser(
        "~/.cache/huggingface/hub/models--gpt2/snapshots/"
        "607a30d783dfa663caf39e06633721c8d4cfcd7e"
    )
    wikitext_dir = os.path.expanduser(
        "~/.cache/huggingface/datasets/wikitext/wikitext-103-raw-v1/0.0.0/"
        "b08601e04326c79dfdd32d625aee71d232d685c3"
    )

    # ── Load models ──
    print("Loading Old ON...")
    old_on = load_old_on(cfg.old_on_ckpt, device)
    old_on_prior = OldONPrior(old_on, num_blocks=cfg.num_blocks)

    print("Loading AO-GPT...")
    aogpt, block_perm, inv_perm = load_aogpt(aogpt_ckpt, device)
    block_perm = block_perm.to(device)
    inv_perm = inv_perm.to(device)

    # ── Load data ──
    print("Loading A matrices...")
    A_all = np.load(cfg.data_path, mmap_mode="r")
    total_seqs = min(A_all.shape[0], cfg.num_train_seqs + cfg.num_tune_seqs + cfg.num_test_seqs)
    A_all = torch.from_numpy(A_all[:total_seqs].copy()).float().to(device)

    print("Loading token chunks...")
    idx_all = load_token_chunks(total_seqs, tokenizer_dir, wikitext_dir).to(device)

    assert len(idx_all) >= total_seqs, f"Only {len(idx_all)} chunks, need {total_seqs}"

    # ── Data splits ──
    n_train = cfg.num_train_seqs
    n_tune = cfg.num_tune_seqs
    n_test = cfg.num_test_seqs

    A_train = A_all[:n_train]
    A_tune = A_all[n_train:n_train + n_tune]
    A_test = A_all[n_train + n_tune:n_train + n_tune + n_test]

    idx_train = idx_all[:n_train]
    idx_tune = idx_all[n_train:n_train + n_tune]
    idx_test = idx_all[n_train + n_tune:n_train + n_tune + n_test]

    print(f"Data: train={n_train}, tune={n_tune}, test={n_test}")

    # ── Generate or load labels ──
    if not args.labels_file:
        args.labels_file = os.path.join(cfg.output_dir, f"labels_train_{cfg.reranker_feature_set}.pt")
    if args.skip_labels and os.path.exists(args.labels_file):
        print(f"Loading cached labels from {args.labels_file}")
        labels_train = torch.load(args.labels_file, weights_only=False)
    else:
        print("Generating per-step candidate-local NLL labels...")
        t0 = time.time()
        labels_train = generate_labels(
            aogpt, old_on_prior, idx_train, block_perm, inv_perm, A_train, cfg
        )
        print(f"Label generation took {time.time() - t0:.1f}s, {len(labels_train)} queries")

        os.makedirs(os.path.dirname(args.labels_file), exist_ok=True)
        torch.save(labels_train, args.labels_file)
        print(f"Saved labels to {args.labels_file}")

    # Auto-detect feature_dim from labels
    if cfg.reranker_feature_set != "v1_legacy":
        cfg.adapter_feature_dim = labels_train[0]["features"].shape[-1]
        print(f"Auto-detected feature_dim={cfg.adapter_feature_dim} from labels")
    else:
        cfg.adapter_feature_dim = 7

    # ── Sanity overfit ──
    if not args.skip_sanity:
        print("\n=== Sanity Overfit ===")
        sanity_overfit(labels_train, cfg, device, n_seqs=10)
        print("Sanity check: MLP can fit AOGPT labels on small subset")

    # ── Train MLP reranker ──
    print("\n=== Training MLP Reranker ===")
    t0 = time.time()
    mlp_reranker = train_mlp_reranker(labels_train, cfg, device)
    print(f"Training took {time.time() - t0:.1f}s")
    print(f"MLP params: {mlp_reranker.count_parameters():,}")

    # Save model with feature_set in filename
    suffix = f"_{cfg.reranker_feature_set}_{cfg.loss_type}"
    if cfg.use_pooled_context:
        suffix += "_pooled"
    ckpt_path = os.path.join(cfg.output_dir, f"mlp_reranker{suffix}.pt")
    torch.save({
        "model_state_dict": mlp_reranker.state_dict(),
        "config": cfg,
    }, ckpt_path)
    print(f"Saved MLP reranker to {ckpt_path}")

    # Also save results with same suffix
    results_path = os.path.join(cfg.output_dir, f"train_results{suffix}.pt")

    # ── Validate ──
    print("\n=== Validation ===")
    mlp_reranker.eval()
    results = validate_reranker(
        mlp_reranker, aogpt, old_on_prior, idx_tune, block_perm, A_tune, cfg
    )

    print(f"\nVal results ({n_tune} seqs):")
    print(f"  MLP reranker NLL:   {results['mlp_nll']:.4f}")
    print(f"  Old ON NLL:         {results['old_on_nll']:.4f}")
    print(f"  Random N16 NLL:     {results['random_n16_nll']:.4f}")
    print(f"  Δ vs Old ON:        {results['delta_vs_old_on']:+.4f}")
    print(f"  Δ vs Random:        {results['delta_vs_random']:+.4f}")

    # Check success criteria
    improved_over_old_on = results['delta_vs_old_on'] > 0
    improved_over_random = results['delta_vs_random'] > 0
    print(f"\n  MLP > Old ON:  {'PASS' if improved_over_old_on else 'FAIL'}")
    print(f"  MLP > Random:  {'PASS' if improved_over_random else 'FAIL'}")

    torch.save(results, results_path)
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
