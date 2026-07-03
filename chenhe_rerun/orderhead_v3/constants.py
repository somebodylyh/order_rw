"""Layout constants + guard for the V3 frozen-gβ order policy.

The V3 port supports ONLY the chenhe mainline seq256/permute/block64 layout
(N=64 content blocks, block_len=4, 8 heads, 65-node strict65). Any other layout
must hard-fail until explicitly extended.
"""

SEQ_LEN = 256
N = 64            # number of content blocks
BLOCK_LEN = 4     # tokens per block (256 / 64 = 4)
HEADS = 8
STRICT_NODES = 65  # 64 content + 1 None node (index 0)
PERMUTE_SEED = 42


def assert_layout(num_blocks: int, block_len: int, n_head: int) -> None:
    """Hard-fail on any layout other than seq256/block64."""
    if (int(num_blocks), int(block_len), int(n_head)) != (N, BLOCK_LEN, HEADS):
        raise ValueError(
            f"orderhead_v3 supports only seq256/block64 "
            f"(N={N}, block_len={BLOCK_LEN}, heads={HEADS}); got "
            f"num_blocks={num_blocks}, block_len={block_len}, n_head={n_head}."
        )
