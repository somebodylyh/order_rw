# Protocol Evolution

| Version | Protocol | Problem found | Current status |
|---|---|---|---|
| B0 legacy | Original controller extraction, with `[None]` folded into physical block0 under selected-head controller path. | Useful for frozen `g_beta` acceleration, but not a strict block-discovery proof because the start anchor is injected. | Legacy acceleration evidence. |
| B1 predictor | Collaborator-aligned predictor frame, result-bearing files use `none_mode=predictor` and `attn[:-1, :-1]`. | Signal survives, but content-only variants can recover cyclic order while losing the correct anchor. | Diagnostic evidence. |
| content-only | Remove None and roll out from content graph. | Can recover order axis, but start is unstable. For L0H1/L0H2/L0H4 the order starts at physical block 6 in B1 content-only. | Mechanism diagnostic, not anchored discovery proof. |
| `[None]->phys0` | Anchored physical start by folding None into block0. | Leaks canonical start for discovery; okay for controller, not for label-free start discovery. | Only anchored controller evidence. |
| loss-aligned AR + None-separated | Query/target and key/source alignment is handled asymmetrically; None is separate. | More strict and more semantically faithful, but extraction-frame-specific. | Current strict discovery protocol. |
| strict 65-node LF | Node0=None, nodes1..64=physical blocks, rollout from None. | Head-specific and extraction-frame-dependent, but successful in selected heads. | Current preferred discovery protocol and main mechanism evidence. |

## Current Preferred Wording

Use:

> The token-level AR model's attention can be aggregated into a strict 65-node block graph in which selected early heads recover L2R from None without manually assigning block0.

Avoid:

> The model is trained with a true block-level loss.

