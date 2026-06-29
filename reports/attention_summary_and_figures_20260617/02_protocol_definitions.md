# Protocol Definitions

## Attention / Graph Protocol Table

| Protocol | Nodes | None handling | Query axis | Key/source axis | Used for | Status |
|---|---|---|---|---|---|---|
| B0 legacy controller extraction | 64 physical blocks | `[None]` folded into physical block0 | legacy block query labels | physical/source block labels | Existing frozen `g_beta` / hook controller path | Legacy acceleration evidence; not strict discovery. |
| B1 predictor frame | 64 predictor-frame blocks | Existing result files use `none_mode=predictor`; `[None]` stays in predictor frame position 0 and is averaged by reshape | `attn[:-1, :-1]` predictor frame | same predictor frame | Collaborator-aligned diagnostic | Strong diagnostic signal; not current hook path. |
| B1 predictor + content-only | 64 content nodes | None removed | predictor/content-only readout | content-only graph | Tests whether order axis survives without None anchor | Supporting diagnostic; can lose anchor. |
| loss-aligned AR extraction | 64 targets plus None/source column when with-none | None retained as separate source column in with-none variant | target/query positions aligned to predicted tokens | source/content positions aligned to context tokens | Basis for strict 65-node graph | Current strict mechanism protocol. |
| `[None]->phys0` anchored protocol | 64 physical blocks | None manually folded into phys0 | block-level folded query | block-level folded key/source | Anchored controller / older diagnostics | Not label-free start discovery. |
| content-only graph | 64 content nodes | None removed | content-only | content-only | Tests order axis without BOS | Mechanism diagnostic; anchor unstable. |
| strict 65-node label-free graph | 65 nodes: node0=None, node1..64=physical content blocks | None independent BOS/start node | target/query block with None-separated graph | source/content block plus None node | Main mechanism evidence | Current preferred discovery protocol. |
| oracle-remapped graph | 65 nodes after label remap | None fixed, content labels remapped for evaluation or construction | remapped labels | remapped labels | Equivariance check / comparison | Matches strict LF; not a leak. |
| destroyed controls | Same 65-node shape | None fixed | same as source graph | edge weights shuffled or content labels permuted | Tests tie-break / graph-statistic confounds | Near random; supports real structure. |

## `inv_perm` Boundary

| Stage | inv_perm allowed? | Reason |
|---|---:|---|
| graph construction | yes | graph labels may be remapped; edge weights unchanged |
| CDL rollout | no | rollout must rely only on edge weights |
| posthoc scoring | yes | only translate predicted sigma for evaluation |

Interpretation:

Strict label-free and oracle-remapped consistency is not a leak. It is permutation equivariance: if the graph labels are consistently permuted and rollout uses only graph edges, the recovered order translates back the same way. The leak would be using `inv_perm` inside rollout decisions; the current strict LF framing excludes that.

## Source Pointers

- B1 predictor convention: `reports/b1_attention_extraction_summary_20260615/01_b1_protocol_definition.md`
- strict 65-node report: `reports/strict_65node_discovery_ckpt_verification_20260617/02_existing_result_summary.md`
- loss-aligned / strict extraction code: `block_lo_arm_order_network/per_head_order_scan.py`
- strict LF search script: `scripts/search_strict_label_free_65.py`
- None-separated graph code: `block_lo_arm_order_network/none_separated_block_graph.py`

