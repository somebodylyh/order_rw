import sys, pathlib, numpy as np, torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))

import pytest


@pytest.fixture(scope="module")
def loaded_model_chunks():
    """Load 5k ckpt + a few chunks once for all seeding tests."""
    from train_clean_aogpt import build_model, CleanPermutation, phys_to_model_idx_clean
    from training_utils import load_train_chunks, SEQ_LEN

    ckpt_dir = ROOT / "block_lo_arm_order_network/probe_results/clean_base_random_perm"
    ckpt = torch.load(ckpt_dir / "ckpt_step5000.pt", map_location="cpu", weights_only=False)
    protocol = ckpt["clean_protocol"]
    model_args = ckpt["model_args"]
    model_args["block_size"] = SEQ_LEN

    clean_perm = CleanPermutation(
        block_perm_phys_to_model=torch.tensor(protocol["block_perm_phys_to_model"], dtype=torch.long),
        inv_perm_model_to_phys=torch.tensor(protocol["inv_perm_model_to_phys"], dtype=torch.long),
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(model_args, device, compile_model=False)
    state_dict = ckpt.get("model") or ckpt.get("model_state_dict")
    clean_sd = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_sd)
    model.to(device)
    model.eval()

    idx_phys = load_train_chunks(n_chunks=None)
    idx_model = phys_to_model_idx_clean(idx_phys, clean_perm)
    return model, idx_model[:4], clean_perm, device


def test_same_seed_bit_identical(loaded_model_chunks):
    from train_clean_aogpt import extract_A_matrices
    model, idx_chunks, clean_perm, device = loaded_model_chunks
    A1 = extract_A_matrices(model, idx_chunks, clean_perm, device, seed=42)
    A2 = extract_A_matrices(model, idx_chunks, clean_perm, device, seed=42)
    np.testing.assert_array_equal(A1, A2)


def test_different_seed_different_A(loaded_model_chunks):
    from train_clean_aogpt import extract_A_matrices
    model, idx_chunks, clean_perm, device = loaded_model_chunks
    A1 = extract_A_matrices(model, idx_chunks, clean_perm, device, seed=1)
    A2 = extract_A_matrices(model, idx_chunks, clean_perm, device, seed=2)
    assert not np.array_equal(A1, A2)


def test_seed_none_preserves_old_behavior(loaded_model_chunks):
    """seed=None must not crash and must produce a valid (n,64,64) array."""
    from train_clean_aogpt import extract_A_matrices
    model, idx_chunks, clean_perm, device = loaded_model_chunks
    A = extract_A_matrices(model, idx_chunks, clean_perm, device, seed=None)
    assert A.shape == (4, 64, 64)
    assert np.all(np.diagonal(A, axis1=1, axis2=2) == 0.0)
