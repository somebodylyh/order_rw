"""All hyperparameters for Block-LO-ARM Order Network MVP."""

import torch


class Config:
    # === Data ===
    num_blocks: int = 16
    block_len: int = 16
    seq_len: int = 256  # num_blocks * block_len

    # === P0: Attention Extraction ===
    M_forward_passes: int = 3        # random-order forward passes per sequence
    num_train_sequences: int = 2000  # sequences for training ON
    num_val_sequences: int = 500     # sequences for evaluation

    # === P0: Mock Data (exaggerated signal for debugging) ===
    mock_adj_mean: float = 0.5       # adjacent block pair mean attention
    mock_nonadj_mean: float = 0.01   # non-adjacent block pair mean attention
    mock_noise_std: float = 0.02     # Gaussian noise std

    # === P0: Cold Start Monitoring ===
    signal_threshold: float = 0.01   # min delta_from_uniform to proceed

    # === P1: DP Solver ===
    use_numba: bool = False          # numba not available, use numpy

    # === P2: Order Network ===
    on_feature_dim: int = 9
    on_hidden_dim: int = 128
    on_num_layers: int = 2
    on_dropout: float = 0.1
    on_lr: float = 1e-3
    on_num_epochs: int = 50
    on_batch_size: int = 64

    # === P2: DAgger ===
    dagger_start_epoch: int = 10
    dagger_max_epsilon: float = 0.3
    dagger_temperature: float = 1.0

    # === P3: Evaluation ===
    eval_temperatures: list = [0.3, 0.5, 1.0, 2.0]
    eval_num_samples: int = 10
    num_random_baselines: int = 100

    # === Reproducibility ===
    seed: int = 42

    @classmethod
    def set_seed(cls):
        torch.manual_seed(cls.seed)
        import numpy as np
        np.random.seed(cls.seed)
        import random
        random.seed(cls.seed)


# ── Reranker Configuration ────────────────────────────────────────────────────

from dataclasses import dataclass, field


@dataclass
class RerankerConfig:
    """Configuration for Frozen Old ON Prior + AO-GPT Local-Utility Reranker."""

    # Paths
    old_on_ckpt: str = "probe_results/crossattn_on_best.pt"
    data_path: str = "probe_results/A_train_10k.npy"
    aogpt_ckpt: str = (
        "~/ych/nanogpt-learned-order/out/base/permute/seq256/block64/"
        "out-wikitext103-seq256-random-b64-permute-block-50000-iters/ckpt.pt"
    )
    output_dir: str = "probe_results/reranker"

    # Dimensions
    num_blocks: int = 16       # N16 policy blocks
    sub_blocks: int = 4        # N64 sub-blocks per N16 block
    block_len: int = 4         # tokens per N64 sub-block
    label_k: int = 4           # first-k tokens of candidate block for NLL label

    # Data splits
    num_train_seqs: int = 100
    num_tune_seqs: int = 200    # grid search / early stop
    num_test_seqs: int = 500

    # Label generation
    label_tau: float = 0.3      # τ for softmax(score/τ) → soft target
    label_mode: str = "edge"    # structural score mode: "edge", "prefix_mean", "prefix_max"
    label_chunk_size: int = 64  # max candidates per AOGPT forward (NLL mode only)

    # Feature builder
    reranker_feature_set: str = "v2_basic"  # "v1_legacy", "v2_basic", "v2_rank"
    adapter_feature_dim: int = 2  # auto-set by feature_set (v1_legacy=7, v2_basic=4, v2_rank=7)

    # MLP adapter architecture
    adapter_hidden_dim: int = 64
    adapter_num_layers: int = 2  # feature_dim → hidden → hidden → 1
    adapter_dropout: float = 0.1

    # Pooled MLP (Step 3)
    use_pooled_context: bool = False
    adapter_pooled_hidden_dim: int = 32   # local embedding dim D
    adapter_pooled_num_layers: int = 1    # layers in MLP_local

    # MLP training
    adapter_lr: float = 1e-3
    adapter_epochs: int = 200
    adapter_batch_size: int = 256
    adapter_weight_decay: float = 1e-4
    loss_type: str = "hard_ce"  # "hard_ce" or "soft_ce"

    # Stage 1 grid search
    grid_alphas: list = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    grid_betas: list = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    grid_gammas: list = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])

    # Evaluation
    num_random_mc: int = 20     # MC samples for random_N16 baseline
    eval_temperatures: list = field(default_factory=lambda: [0.3, 0.5, 1.0, 2.0])

    # Reproducibility
    seed: int = 42
