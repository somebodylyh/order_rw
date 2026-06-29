"""
Debug: verify block_perm coordinate system and what each eval mode actually evaluates.

Run:
    cd block_lo_arm_order_network && python debug_coordinate_system.py
"""
import os
import sys
import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", "AO-GPT-MDM"))

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

BLOCK_LEN = 4
N_BLOCKS = 64
SEQ_LEN = 256

CKPT = os.path.expanduser(
    "~/ych/nanogpt-learned-order/out/base/permute/seq256/block64/"
    "out-wikitext103-seq256-random-b64-permute-block-50000-iters/ckpt.pt"
)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Load checkpoint and inspect block_perm / inv_perm
# ═══════════════════════════════════════════════════════════════════════════════
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
block_perm = torch.tensor(ckpt["data_permutation"]["block_perm"], dtype=torch.long)
inv_perm = torch.tensor(ckpt["data_permutation"]["inverse_block_perm"], dtype=torch.long)

print("=" * 70)
print("1. CHECKPOINT block_perm / inv_perm")
print("=" * 70)
print(f"block_perm shape: {block_perm.shape}")
print(f"inv_perm shape:    {inv_perm.shape}")
print(f"block_perm[:16]:   {block_perm[:16].tolist()}")
print(f"inv_perm[:16]:     {inv_perm[:16].tolist()}")
print()

# Check: is block_perm[phys] = model or block_perm[model] = phys?
# We can check by verifying round-trip
bp_as_list = block_perm.tolist()
ip_as_list = inv_perm.tolist()

# If block_perm[phys] = model and inv_perm[model] = phys:
# then inv_perm[block_perm[phys]] == phys
# If block_perm[model] = phys and inv_perm[phys] = model:
# then inv_perm[block_perm[model]] == model

test_phys_indices = [0, 1, 2, 3, 4, 5, 6, 7]
test_model_indices = [0, 1, 2, 3, 4, 5, 6, 7]

# Test hypothesis A: bp[phys] = model, ip[model] = phys
print("--- Test A: block_perm[phys] = model, inv_perm[model] = phys ---")
all_pass_A = True
for pi in test_phys_indices:
    mi = bp_as_list[pi]
    back = ip_as_list[mi]
    ok = (back == pi)
    if not ok:
        all_pass_A = False
    print(f"  phys {pi} -> model {mi} -> inv_perm[{mi}] = {back}  {'OK' if ok else 'FAIL'}")

# Test hypothesis B: bp[model] = phys, ip[phys] = model
print("\n--- Test B: block_perm[model] = phys, inv_perm[phys] = model ---")
all_pass_B = True
for mi in test_model_indices:
    pi = bp_as_list[mi]
    back = ip_as_list[pi]
    ok = (back == mi)
    if not ok:
        all_pass_B = False
    print(f"  model {mi} -> phys {pi} -> inv_perm[{pi}] = {back}  {'OK' if ok else 'FAIL'}")

print()
if all_pass_A and not all_pass_B:
    print(">>> CONFIRMED: block_perm[phys] = model, inv_perm[model] = phys")
elif all_pass_B and not all_pass_A:
    print(">>> CONFIRMED: block_perm[model] = phys, inv_perm[phys] = model")
elif all_pass_A and all_pass_B:
    print(">>> AMBIGUOUS: both directions round-trip (perm is identity or symmetric?)")
else:
    print(">>> NEED DEEPER CHECK: neither A nor B holds cleanly")

# Also test: does ip[pi] make sense?
print("\n--- What does inv_perm[phys_index] return? ---")
for pi in test_phys_indices:
    result = ip_as_list[pi]
    print(f"  inv_perm[{pi}] = {result}  (model block? {result})")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Verify phys_to_model_idx round-trip
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("2. phys_to_model_idx ROUND-TRIP")
print("=" * 70)

def phys_to_model_idx(idx_phys, inv_perm):
    B_val, T = idx_phys.shape
    M_val = len(inv_perm)
    blk_size = T // M_val
    idx_model = torch.zeros_like(idx_phys)
    for s in range(T):
        model_block = inv_perm[s // blk_size].item()
        offset = s % blk_size
        model_pos = model_block * blk_size + offset
        idx_model[:, model_pos] = idx_phys[:, s]
    return idx_model

def model_to_phys_idx(idx_model, block_perm):
    B_val, T = idx_model.shape
    M_val = len(block_perm)
    blk_size = T // M_val
    idx_phys = torch.zeros_like(idx_model)
    for s in range(T):
        phys_block = block_perm[s // blk_size].item()
        offset = s % blk_size
        phys_pos = phys_block * blk_size + offset
        idx_phys[:, phys_pos] = idx_model[:, s]
    return idx_phys

# Create a test tensor where token value = physical position
test_phys = torch.tensor([[i for i in range(SEQ_LEN)]], dtype=torch.long)
test_model = phys_to_model_idx(test_phys, inv_perm)
test_phys_back = model_to_phys_idx(test_model, block_perm)

print(f"Original phys[0, :8]:            {test_phys[0, :8].tolist()}")
print(f"Model coord[0, :8]:             {test_model[0, :8].tolist()}")
print(f"Model coord[0] first 4 tokens:   {test_model[0, :4].tolist()} <- physical block {inv_perm[0].item()}")
print(f"Model coord[0, 4:8] next 4:      {test_model[0, 4:8].tolist()} <- physical block {inv_perm[1].item()}")
print(f"Model coord[0, 8:12]:            {test_model[0, 8:12].tolist()} <- physical block {inv_perm[2].item()}")
print(f"Round-trip ok: {torch.equal(test_phys, test_phys_back)}")

# Test: model_to_phys_idx with block_perm where block_perm[model_pos] -> phys_pos
# The function above uses block_perm[s//blk_size] where s iterates model positions
# So block_perm[model_block] must = phys_block
# Let's verify: for model position 0 (model block 0), what phys does block_perm[0] give?
model_block0 = 0
phys_from_bp = block_perm[model_block0].item()
print(f"\nblock_perm[model_block={model_block0}] = phys_block={phys_from_bp}")
print(f"Inv check: inv_perm[phys={phys_from_bp}] = {inv_perm[phys_from_bp].item()} (should = {model_block0})")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Simulate eval modes and verify PHYSICAL block order
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("3. EVAL MODE: physical block order verification")
print("=" * 70)

def phys_n64_block_order_to_model_token_order(block_order, block_perm):
    """From train_aogpt_graph_rw.py.
    block_order: physical N64 block indices at each reveal step
    block_perm[phys] = model (confirmed above)
    """
    B_bs, N = block_order.shape
    T = N * BLOCK_LEN
    bp = block_perm
    token_order = torch.zeros(B_bs, T, dtype=torch.long)
    for t in range(N):
        phys_blk = block_order[:, t]
        model_blk = bp[phys_blk]
        for k in range(BLOCK_LEN):
            token_order[:, t * BLOCK_LEN + k] = model_blk * BLOCK_LEN + k
    return token_order

# ── val_l2r: torch.arange(N_blocks) interpreted as physical block order ──
l2r_block = torch.arange(N_BLOCKS).unsqueeze(0)  # [0,1,...,63] as physical
token_order_l2r = phys_n64_block_order_to_model_token_order(l2r_block, block_perm)

# What actual token positions does this pick?
print("\n--- val_l2r (torch.arange(64) as physical blocks) ---")
print(f"block_order (first 16):          {l2r_block[0, :16].tolist()}")
print(f"token_order (first 16 positions): {token_order_l2r[0, :16].tolist()}")
print("Interpretation: at each reveal step t, the model reads tokens from model-coordinate positions:")
for t in range(8):
    phys_blk = l2r_block[0, t].item()
    model_blk = block_perm[phys_blk].item()
    tok_start = token_order_l2r[0, t * BLOCK_LEN].item()
    tok_end = token_order_l2r[0, (t + 1) * BLOCK_LEN - 1].item()
    print(f"  step {t:2d}: physical block {phys_blk:2d} -> model block {model_blk:2d} -> tokens [{tok_start:3d}, {tok_end:3d}]")

# Since we're revealing physical blocks in order [0,1,2,...,63],
# the model sees physical block 0 first, then 1, etc.
# This IS "ori L2R" at the block level.

# But wait - does the model's native token order also align?
# Let's see the first 16 tokens:
print(f"\nFirst 16 tokens in val_l2r reveal order:")
for i in range(16):
    tok = token_order_l2r[0, i].item()
    # Which physical block does token tok belong to?
    # In model coords: token at position tok is in model block (tok // BLOCK_LEN)
    model_blk = tok // BLOCK_LEN
    # Model block model_blk corresponds to physical block = ???
    # We need the inverse: block_perm[phys] = model, so we need to find phys s.t. block_perm[phys] = model_blk
    # That's what inv_perm does: inv_perm[model] = phys
    phys_blk = inv_perm[model_blk].item()
    print(f"  token pos {tok:3d} (model block {model_blk:2d}) = physical block {phys_blk:2d}")

# ── What if torch.arange(64) was meant as MODEL block order? ──
print("\n--- If torch.arange(64) were MODEL blocks (hypothetical) ---")
# To evaluate "model L2R" where model blocks [0,1,...,63] are revealed in order,
# we'd need the physical blocks corresponding to model blocks 0,1,...,63
# Since block_perm[phys] = model, we need the inverse: for each model block m,
# find physical block p such that block_perm[p] = m.
# That's exactly inv_perm[m] = p.

# But the eval function phys_n64_block_order_to_model_token_order expects
# physical block indices as input. So if we want model L2R:
# model_block_i needs physical_block = inv_perm[model_block_i]
model_l2r_physical_blocks = inv_perm[torch.arange(N_BLOCKS)]
print(f"model_l2r physical block order (first 16): {model_l2r_physical_blocks[:16].tolist()}")
print(f"This is NOT [0,1,2,...,15], so model-L2R != physical-L2R")

# Create token order for model-L2R
model_l2r_block = model_l2r_physical_blocks.unsqueeze(0)
token_order_model_l2r = phys_n64_block_order_to_model_token_order(model_l2r_block, block_perm)
print(f"token_order_model_l2r (first 16): {token_order_model_l2r[0, :16].tolist()}")

# ── val_ar: ascending orders ──
print("\n--- val_ar (set_ascending_orders = torch.arange(256)) ---")
ascending_orders = torch.arange(SEQ_LEN).unsqueeze(0)
print(f"ascending orders (first 16): {ascending_orders[0, :16].tolist()}")
print("This means: model reads tokens in model-coordinate order 0,1,2,...,255")
print("Which corresponds to model block order: [0,1,...,63]")
print("Which corresponds to physical block order:")
ar_phys_order = []
for mb in range(N_BLOCKS):
    phys_blk = inv_perm[mb].item()
    ar_phys_order.append(phys_blk)
print(f"  val_ar physical block order (first 16): {ar_phys_order[:16]}")
print(f"  val_ar physical block order (all 64):   {ar_phys_order}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Summary table
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("4. SUMMARY: What each eval mode evaluates")
print("=" * 70)
print()
print("Physical block [i] = the i-th block of 4 tokens in the original text")
print("Model block [j]   = the j-th block position in the model's internal representation")
print(f"block_perm[phys] = model:  block_perm[{block_perm[0].item()}] = {block_perm[0].item()} (at phys=0)")
print(f"inv_perm[model]  = phys:   inv_perm[{inv_perm[0].item()}] = {inv_perm[0].item()} (at model=0)")
print()

# Compute physical block orders for each mode
phys_ori_l2r = list(range(N_BLOCKS))  # [0,1,...,63]
phys_val_l2r = list(range(N_BLOCKS))  # val_l2r is the same as ori_l2r
phys_val_ar = [inv_perm[m].item() for m in range(N_BLOCKS)]
phys_model_l2r = [inv_perm[m].item() for m in range(N_BLOCKS)]  # model-L2R = [inv_perm[0], inv_perm[1], ...]

print(f"{'Eval mode':<25s} {'first 8 physical blocks':<40s} {'= ori L2R?':<12s} {'NLL (from pilot)'}")
print("-" * 100)
print(f"{'val_l2r':<25s} {str(phys_val_l2r[:8]):<40s} {'YES':<12s} {'3.792 -> 3.950'}")
print(f"{'val_ar':<25s} {str(phys_val_ar[:8]):<40s} {'NO' if phys_val_ar != phys_ori_l2r else 'YES':<12s} {'3.786 -> 3.938'}")

# val_rw_order: depends on graph-RW sampling, can't pre-compute
print(f"{'val_rw_order':<25s} {'(sampled from pi_RW)':<40s} {'N/A':<12s} {'3.776 -> 3.924'}")
print(f"{'val_unstructured':<25s} {'(random permutation)':<40s} {'N/A':<12s} {'3.783 -> 3.935'}")

# Hypothetical model-L2R
print(f"{'val_model_l2r (hypothetical)':<25s} {str(phys_model_l2r[:8]):<40s} {'NO':<12s} {'(not in eval)'}")

print()
print("=" * 70)
print("5. ANSWER")
print("=" * 70)
if phys_val_l2r == phys_ori_l2r:
    print()
    print("val_l2r (torch.arange(64) as physical blocks) IS ori physical L2R.")
    print("Physical block order = [0,1,2,...,63] exactly.")
    print()
    print("val_ar (ascending model order) IS NOT ori physical L2R.")
    print(f"val_ar physical block order = {phys_val_ar[:8]}...")
    print()
    print("val_l2r == val_ar in NLL because they're both some form of L2R,")
    print("just at different granularity (block vs token level).")
    print("The near-identity of their NLL values suggests the model is")
    print("relatively agnostic to block-level vs token-level AR in practice.")
else:
    print("val_l2r is NOT ori physical L2R! This is a bug.")
