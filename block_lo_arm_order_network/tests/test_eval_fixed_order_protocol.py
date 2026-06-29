import torch

from eval_fixed_order_protocol import (
    block_orders_to_token_orders,
    describe_block_order,
    original_l2r_block_order,
)


def test_original_l2r_uses_inverse_permutation_for_model_positions():
    # checkpoint convention: block_perm[model] = physical, inv_perm[physical] = model
    block_perm = torch.tensor([2, 0, 3, 1])
    inv_perm = torch.empty_like(block_perm)
    inv_perm[block_perm] = torch.arange(block_perm.numel())

    order = original_l2r_block_order(inv_perm)

    assert order.physical.tolist() == [0, 1, 2, 3]
    assert order.model.tolist() == [1, 3, 0, 2]


def test_block_orders_expand_to_model_token_order_with_l2r_inside_block():
    model_blocks = torch.tensor([[1, 3, 0, 2]])

    token_orders = block_orders_to_token_orders(model_blocks, block_len=2)

    assert token_orders.tolist() == [[2, 3, 6, 7, 0, 1, 4, 5]]


def test_describe_block_order_records_both_coordinate_frames():
    block_perm = torch.tensor([2, 0, 3, 1])
    model_order = torch.tensor([1, 3, 0, 2])

    desc = describe_block_order(model_order, block_perm, block_len=2)

    assert desc["model_order_first16"] == [1, 3, 0, 2]
    assert desc["physical_order_first16"] == [0, 1, 2, 3]
    assert desc["token_ranges_first8"] == ["phys[0:2]", "phys[2:4]", "phys[4:6]", "phys[6:8]"]
