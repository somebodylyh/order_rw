# Collaborator Checkpoint B1 Signal Scan

Date: 2026-06-16

Checkpoint:

- `/home/admin/ych/nanogpt-learned-order/ckpt/ckpt.pt`
- iter_num: 50000
- model: 4L/8H/384d, seq256, block64, block_len=4
- config: `dataset=wikitext103`, `data_record_mode=stream`, `permute_data=True`, `permute_seed=42`, `aogpt_train_mode=Random`

Adapter:

- Script: `scripts/scan_collaborator_ckpt_b1.py`
- Purpose: read-only adapter for collaborator checkpoint format (`config` + `data_permutation`) into our per-head B1/predictor extraction.
- No training was launched.
- No checkpoint or training code was modified.

## Main Result

Main scan:

- Protocol: B1/predictor-aligned diagnostic (`none_mode=predictor`)
- Input coordinate assumption: `data_permutation.block_perm[physical]=model`, `inverse_block_perm[model]=physical`
- Samples: `M=40`, `batch_size=8`, total 320 Wikitext-103 train chunks
- Device: CPU
- Source JSON: `reports/collaborator_ckpt_b1_scan_20260616/b1_predictor_phys_to_model_M40_seed0.json`

Summary:

| Metric | Value |
|---|---:|
| best head | L1H7 |
| best tau vs L2R | 0.805952 |
| best mean pairwise tau | 0.726061 |
| `|tau|>0.9` | 0/32 |
| `|tau|>0.7` | 5/32 |
| heavy tau vs L2R | 0.400992 |
| heavy mean pairwise tau | 0.299163 |

Interpretation:

> The collaborator checkpoint contains a clear but moderate B1/predictor-frame order signal. The strongest head is around tau=0.806, with five heads above 0.7, but no head above 0.9 under this lightweight M=40 scan.

## Top Heads

Source: `b1_predictor_phys_to_model_M40_seed0.json`.

| Rank | Head | tau vs L2R | mean pairwise tau | first-step entropy | tau vs heavy |
|---:|---|---:|---:|---:|---:|
| 1 | L1H7 | 0.805952 | 0.726061 | -0.000 | 0.384325 |
| 2 | L0H6 | 0.800843 | 0.707005 | 0.117 | 0.380407 |
| 3 | L2H5 | 0.761657 | 0.656097 | -0.000 | 0.390427 |
| 4 | L0H7 | 0.738864 | 0.635802 | -0.000 | 0.367684 |
| 5 | L3H0 | 0.728522 | 0.614705 | -0.000 | 0.378026 |
| 6 | L1H0 | 0.690774 | 0.571762 | -0.000 | 0.365923 |
| 7 | L3H2 | 0.681250 | 0.564370 | -0.000 | 0.364484 |
| 8 | L1H2 | 0.679365 | 0.563197 | -0.000 | 0.373661 |
| 9 | L3H7 | 0.655779 | 0.536125 | -0.000 | 0.372991 |
| 10 | L1H1 | 0.648760 | 0.521039 | -0.000 | 0.372024 |
| 11 | L3H5 | 0.637624 | 0.535352 | -0.000 | 0.371453 |
| 12 | L3H4 | 0.616394 | 0.484149 | -0.000 | 0.363715 |

## Controls / Sensitivity

| Scan | Protocol | Perm orientation | M | Best head | Best tau | `|tau|>0.9` | `|tau|>0.7` | Source |
|---|---|---|---:|---|---:|---:|---:|---|
| Main | B1/predictor | phys_to_model | 40 | L1H7 | 0.805952 | 0/32 | 5/32 | `b1_predictor_phys_to_model_M40_seed0.json` |
| Lightweight | B1/predictor | phys_to_model | 20 | L1H7 | 0.797371 | 0/32 | 5/32 | `b1_predictor_phys_to_model_M20_seed0.json` |
| Orientation sensitivity | B1/predictor | model_to_phys | 20 | L0H6 | 0.804315 | 0/32 | 4/32 | `b1_predictor_model_to_phys_M20_seed0.json` |
| Physical-remap enum | `none_mode=b1` | phys_to_model | 20 | L2H0 | 0.073760 | 0/32 | 0/32 | `b1_physical_phys_to_model_M20_seed0.json` |
| Legacy extraction | B0 | phys_to_model | 20 | L2H0 | 0.066468 | 0/32 | 0/32 | `b0_legacy_phys_to_model_M20_seed0.json` |
| Physical-remap enum | `none_mode=b1` | model_to_phys | 20 | L0H3 | 0.796131 | 0/32 | 1/32 | `b1_physical_model_to_phys_M20_seed0.json` |
| Legacy extraction | B0 | model_to_phys | 20 | L0H3 | 0.796131 | 0/32 | 2/32 | `b0_legacy_model_to_phys_M20_seed0.json` |

Key protocol conclusion:

> On this collaborator checkpoint, the original physical L2R order is known from the Wikitext chunks, but the checkpoint's `data_permutation` lacks an explicit convention string. The B1/predictor-frame diagnostic is robust to this ambiguity and gives best tau around 0.80. Physical-remap paths (`none_mode=b1` and B0) are convention-sensitive: under the clean-protocol assumption `block_perm[physical]=model` they look near-zero, while under the opposite interpretation `block_perm[model]=physical` they recover a similar best tau around 0.796.

## Caveats

- This is a diagnostic scan, not a hook or training result.
- The scan uses Wikitext-103 chunks from the local Hugging Face cache and maps physical chunks into model frame using the checkpoint's `data_permutation`.
- The collaborator checkpoint does not include our internal `clean_protocol` / `args` fields, so `scripts/scan_collaborator_ckpt_b1.py` adapts the checkpoint format.
- The main result is M=40. A full M=100 scan could reduce noise, but the M=20 and M=40 B1/predictor results are already consistent.
- The checkpoint's `data_permutation` does not include an explicit convention string. I tested both orientation assumptions. For B1/predictor, both gave a similar best tau around 0.80. For physical-remap B1/B0, only the `model_to_phys` interpretation gave a strong signal.

## Attention Map Figure

Figure:

- `reports/collaborator_ckpt_b1_scan_20260616/figures/collaborator_ckpt_attention_maps_M20.png`

Panels:

- top-left: B1/predictor raw map, `L1H7`, no physical remap, `tau=0.789`.
- top-right: B1 physical remap, `L0H3`, correct collaborator checkpoint convention, `tau=0.796`.
- bottom-left: B0 legacy remap, `L0H3`, correct collaborator checkpoint convention, `tau=0.796`.
- bottom-right: B1 physical remap with the wrong clean-protocol orientation, `L0H3`, `tau=0.032`.

This directly shows the coordinate issue: the original physical L2R order is known, and remapping attention back to that order is our job. The near-zero result appears only when the checkpoint permutation field is interpreted in the wrong direction.
