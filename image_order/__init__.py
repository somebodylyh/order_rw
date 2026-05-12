"""image_order — CIFAR-10 image-patch extension of AOGPT + Graph-RW v2.

Research question: does AOGPT's internal attention on image-patch data encode
dataset-level shared structure (spatial locality, region grouping, center vs
edge tendency)? If yes, the same Graph-RW order policy that recovers L2R-like
order from WikiText attention should produce spatially-coherent reveal orders
here, AND using those orders as an any-order curriculum should match or beat
random-order training.

Design contract (do not violate):
- No block_perm / inv_perm. There is no checkpoint permutation for image patches.
- No L2R oracle. Raster scan is reported as a diagnostic baseline only, never
  used as a training label.
- A_global is averaged over all heads, layers, and images.
- N=64 patches (8x8 grid), patch_dim=48 (4x4x3), grid index r*8+c.

Reuses the text-pipeline policy code in
block_lo_arm_order_network/directed_graph_policy.py verbatim — that module is
pure NumPy and N-agnostic.
"""
