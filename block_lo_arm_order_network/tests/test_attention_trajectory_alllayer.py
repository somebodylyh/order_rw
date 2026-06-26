import numpy as np
import torch
from pathlib import Path
from attention_trajectory import extract_all_layer_B, AttentionTrajectoryLogger


def test_log_snapshot_saves_tau_and_composition(tmp_path):
    from model_AOGPT_AdaLN6_NoRep_cond_128_trunc_qknorm import AOGPT, AOGPTConfig
    cfg = AOGPTConfig(n_layer=4, n_head=8, n_embd=64, block_size=256)
    model = AOGPT(cfg)
    logger = AttentionTrajectoryLogger(
        output_root=tmp_path, run_name="t", n_attention_samples=4,
        composition_pairs=[(0, 1), (1, 2), (2, 3)], seed=2,
    )
    eval_tokens = torch.randint(0, cfg.vocab_size, (8, 256))
    logger.log_snapshot(model, eval_tokens, global_step=0, clean_perm=None,
                        device=torch.device("cpu"))
    step_dir = tmp_path / "raw" / "step_000000"
    tau = np.load(step_dir / "tau_table.npz")
    assert tau["tau"].shape == (4, 8, 2)  # (L,H,methods)
    comp = np.load(step_dir / "composition.npz")
    assert comp["0_1"].shape == (8, 8, 3)


def test_extract_all_layer_B_shape():
    L, S, H, T = 4, 2, 8, 256
    rng = np.random.default_rng(0)
    # causal-ish attention over T+1 tokens; rows sum ~1 not required for shape
    attn_list = [rng.random((S, H, T + 1, T + 1)).astype(np.float32) for _ in range(L)]
    probe_orders = np.tile(np.arange(T, dtype=np.int64), (S, 1))
    B = extract_all_layer_B(attn_list, probe_orders)
    assert B.shape == (L, S, H, 65, 65)
    # diagonal must be zero (none-separated convention)
    assert np.allclose(np.diagonal(B, axis1=3, axis2=4), 0.0)
