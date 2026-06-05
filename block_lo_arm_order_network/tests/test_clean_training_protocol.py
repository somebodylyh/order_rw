import numpy as np
import torch

from clean_training_protocol import (
    CleanPermutation,
    batch_indices_for_step,
    build_clean_block_permutation,
    build_fixed_split_and_shuffle,
    build_phys_to_model_token_gather,
    expand_model_blocks_to_token_order,
    load_token_stream,
    model_blocks_to_physical_blocks,
    phys_to_model_idx_clean,
    physical_blocks_to_model_blocks,
    physical_blocks_to_model_token_order,
    sample_stream_batch,
    verify_clean_coordinate_round_trip,
)


def test_clean_block_permutation_uses_physical_to_model_convention():
    model_to_phys = torch.tensor([2, 0, 3, 1])
    perm = CleanPermutation.from_model_to_phys(model_to_phys)

    assert perm.block_perm_phys_to_model.tolist() == [1, 3, 0, 2]
    assert perm.inv_perm_model_to_phys.tolist() == [2, 0, 3, 1]
    assert physical_blocks_to_model_blocks(torch.tensor([0, 1, 2, 3]), perm).tolist() == [1, 3, 0, 2]
    assert verify_clean_coordinate_round_trip(perm, block_len=2)


def test_build_clean_block_permutation_is_reproducible_and_valid():
    first = build_clean_block_permutation(num_blocks=8, seed=42)
    second = build_clean_block_permutation(num_blocks=8, seed=42)

    assert torch.equal(first.block_perm_phys_to_model, second.block_perm_phys_to_model)
    assert torch.equal(first.inv_perm_model_to_phys, second.inv_perm_model_to_phys)
    assert verify_clean_coordinate_round_trip(first, block_len=4)


def test_shuffled_l2r_train_traversal_matches_val_model_order_eval():
    # shuffled-L2R control: train AR along the data's shuffled layout (model_ascending),
    # NOT the recovered original order. The training arm feeds
    #   fixed_phys = model_blocks_to_physical_blocks(arange(N))
    # through order_loss, which must reproduce EXACTLY the token traversal that the
    # eval's `val_model_order` column scores (model_ascending = arange(N) expanded).
    # If these diverge, the control is silently wrong. perm ∘ inv_perm = identity guarantees it.
    N, block_len = 8, 4
    clean_perm = build_clean_block_permutation(num_blocks=N, seed=42)

    model_ascending = torch.arange(N, dtype=torch.long).unsqueeze(0)
    fixed_phys = model_blocks_to_physical_blocks(model_ascending, clean_perm)

    train_token_order = physical_blocks_to_model_token_order(fixed_phys, clean_perm, block_len)
    eval_token_order = expand_model_blocks_to_token_order(model_ascending, block_len)

    assert torch.equal(train_token_order, eval_token_order)
    # and it must be the plain ascending token order [0,1,...,N*block_len-1]
    assert torch.equal(eval_token_order[0], torch.arange(N * block_len, dtype=torch.long))
    # sanity: the shuffled layout is genuinely NOT the recovered original order
    ori_phys = torch.arange(N, dtype=torch.long).unsqueeze(0)
    assert not torch.equal(fixed_phys, ori_phys)


def test_fixed_split_shuffle_is_reproducible_disjoint_and_permuted_train_indices():
    split = build_fixed_split_and_shuffle(total_chunks=20, seed=42, val_fraction=0.2)
    split_again = build_fixed_split_and_shuffle(total_chunks=20, seed=42, val_fraction=0.2)

    assert np.array_equal(split.train_indices, split_again.train_indices)
    assert np.array_equal(split.val_indices, split_again.val_indices)
    assert np.array_equal(split.train_shuffle_order, split_again.train_shuffle_order)
    assert set(split.train_indices).isdisjoint(set(split.val_indices))
    assert sorted(split.train_shuffle_order.tolist()) == sorted(split.train_indices.tolist())


def test_batch_indices_for_step_wraps_and_aligns_continuation_cursor():
    order = np.array([10, 11, 12, 13, 14], dtype=np.int64)

    baseline_next = batch_indices_for_step(order, global_step=3, micro_step=0, batch_size=2, grad_accum=1)
    resumed_next = batch_indices_for_step(order, global_step=3, micro_step=0, batch_size=2, grad_accum=1)
    wrapped_batch = batch_indices_for_step(order, global_step=1, micro_step=0, batch_size=2, grad_accum=2)

    assert baseline_next.tolist() == [11, 12]
    assert resumed_next.tolist() == baseline_next.tolist()
    assert wrapped_batch.tolist() == [14, 10]


def test_expand_model_blocks_to_token_order_keeps_l2r_inside_block():
    model_blocks = torch.tensor([[2, 0, 1]])

    token_order = expand_model_blocks_to_token_order(model_blocks, block_len=3)

    assert token_order.tolist() == [[6, 7, 8, 0, 1, 2, 3, 4, 5]]


def test_build_phys_to_model_token_gather_matches_per_position_loop():
    perm = build_clean_block_permutation(num_blocks=8, seed=7)
    block_len = 4
    g = build_phys_to_model_token_gather(perm, block_len)

    idx_phys = torch.randint(0, 50000, (5, 8 * block_len))
    expected = phys_to_model_idx_clean(idx_phys, perm)

    assert g.shape == (8 * block_len,)
    assert torch.equal(idx_phys[:, g], expected)


def test_sample_stream_batch_is_deterministic_shaped_and_contiguous():
    stream = np.arange(1000, dtype=np.uint16)

    b1 = sample_stream_batch(stream, batch_size=4, block_size=16, seed=42, step=3, micro=0)
    b2 = sample_stream_batch(stream, batch_size=4, block_size=16, seed=42, step=3, micro=0)

    assert b1.shape == (4, 16)
    assert b1.dtype == torch.long
    assert torch.equal(b1, b2)  # deterministic in (seed, step, micro)

    # Each row is a contiguous window of the stream (arange -> consecutive diffs of 1).
    diffs = b1[:, 1:].long() - b1[:, :-1].long()
    assert torch.all(diffs == 1)
    assert int(b1.min()) >= 0 and int(b1.max()) < 1000


def test_sample_stream_batch_varies_with_micro_and_step():
    stream = np.arange(1000, dtype=np.uint16)

    base = sample_stream_batch(stream, batch_size=4, block_size=16, seed=42, step=3, micro=0)
    other_micro = sample_stream_batch(stream, batch_size=4, block_size=16, seed=42, step=3, micro=1)
    other_step = sample_stream_batch(stream, batch_size=4, block_size=16, seed=42, step=4, micro=0)

    assert not torch.equal(base, other_micro)
    assert not torch.equal(base, other_step)


def test_load_token_stream_reads_uint16_bin(tmp_path):
    data = np.arange(64, dtype=np.uint16)
    bin_path = tmp_path / "toks.bin"
    data.tofile(bin_path)

    stream = load_token_stream(str(bin_path))

    assert len(stream) == 64
    assert int(stream[0]) == 0 and int(stream[63]) == 63
