"""P2: disambiguate the L0 physical-order signal source (B fixed-layout map vs
C content-dependent). Measures content-dependence of the carrier's attention/B,
NOT the order output (tautological under fixed layout). See spec
docs/superpowers/specs/2026-06-28-physical-signal-source-disambiguation-design.md."""
import pathlib, sys
import numpy as np, torch

_BLOCK_DIR = pathlib.Path(__file__).resolve().parents[1] / "block_lo_arm_order_network"
if str(_BLOCK_DIR) not in sys.path:
    sys.path.insert(0, str(_BLOCK_DIR))

from neural_readout.extract_b import _load_model_and_chunks
from analyses.canonical_reanalysis import random_reveal_orders
from per_head_order_scan import _attn_to_A_block_loss_aligned_with_none_vec
from none_separated_block_graph import build_none_separated_B, rollout_by_method, discovery_metrics

@torch.no_grad()
def carrier_b65_per_text(ckpt_path, layer, head, M=24, fixed_reveal_seed=0, n_reveals=8,
                         device="cpu"):
    model, chunks, clean_perm, dev, _ = _load_model_and_chunks(
        ckpt_path, M, seed=0, device=device, split="train")
    inv = clean_perm.inv_perm_model_to_phys.cpu().numpy()
    reveals = random_reveal_orders(n_reveals, fixed_reveal_seed)   # SHARED across texts
    B_list, tau_list = [], []
    for t in range(M):
        A_acc = None
        for rev in reveals:
            po = torch.from_numpy(rev[None, :]).to(dev)
            _, _, attn_list = model.forward_fn(chunks[t:t+1].to(dev), po, return_attentions=True)
            attn = torch.stack(attn_list, 0).cpu().numpy()[:, 0]   # (L,H,257,257)
            A = _attn_to_A_block_loss_aligned_with_none_vec(attn, rev, inv)  # (L,H,64,65)
            A_acc = A.astype(np.float64) if A_acc is None else A_acc + A
        B = build_none_separated_B((A_acc / n_reveals)[layer, head])
        B_list.append(B)
        tau_list.append(float(discovery_metrics(rollout_by_method(B, "C-D+L"))["tau_vs_l2r"]))
    return B_list, tau_list
