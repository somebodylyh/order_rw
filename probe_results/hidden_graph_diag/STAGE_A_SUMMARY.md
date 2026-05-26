# Hidden-Graph Diagnostic — Stage A Summary

Frozen, no-training diagnostic (spec `docs/superpowers/specs/2026-05-26-hidden-graph-diagnostic-design.md`,
plan `docs/superpowers/plans/2026-05-26-hidden-graph-diagnostic.md`). Question: does the
hidden-state cosine graph `B_H = cos(H,H)`, **after residualizing out position** (`B_H_resid`),
carry order structure complementary to the attention graph `B_A` — enough to justify the
two-graph policy `s = f_A(φ_A) + γ·f_H(φ_H_resid)`? Run via `scripts/run_hidden_graph_diag.py`.

## Verdict: NULL on all three checkpoints

| ckpt | n | corr(B_H,B_pos) | corr(B_Hresid,B_A) | B_Hresid sharpness z | best γ Δ vs A-only (std) | beats matched-ctrl | **verdict** |
|---|---|---|---|---|---|---|---|
| text `clean_base_random_perm`@30k (primary) | 16 chunks | 0.616 | −0.299 | **−0.04** | +0.019 @γ0.25 (±0.046) | No | **NULL** |
| text `alt_from0_mlp_finetune`@30k (contrast) | 16 chunks | 0.629 | −0.306 | **+0.95** | +0.042 @γ2.0 (±0.012) | Yes | **NULL** |
| image `vq64_alt_from0_mlp_patch2x2`@30k (primary) | 256 | 0.758 | +0.124 | **−5.35** | +0.002 @γ0.5 (±0.009) | No | **NULL** |

Anti-fooling gate (ALL five required for WIN): `bh_resid_structured (z>2)` ∧ `not_BA_clone` ∧
`not_position_artifact` ∧ `stable γ-sweep` ∧ `beats_matched_control`. The decisive failure on
every ckpt is **`bh_resid_structured = False`** — once position is removed, the hidden graph has
no peak-to-mean structure above its own shuffled null. `stable` is also False everywhere (no γ
gives an improvement > 2× the A-only seed noise floor).

## What the numbers say

1. **The hidden cosine graph IS position.** `corr(B_H, B_pos)` = 0.62 / 0.63 / 0.76 across the
   three ckpts. The apparent structure in raw `B_H` (e.g. image `B_H_raw` spectral structure) is
   the locality/index-distance graph in disguise. Residualization sanity holds exactly:
   `corr(B_Hresid, B_pos)` ≈ 0 (−7e-16 / −7e-16 / −2e-16).

2. **After removing position, nothing remains.** `B_H_resid` sharpness z = −0.04 (text clean),
   +0.95 (text mlp), −5.35 (image) — all below the z>2 structure bar; the image value is strongly
   *negative* (the residual is flatter than a shuffled null). `B_H_resid` is also not a clone of
   `B_A` (|corr| ≤ 0.31) and not a position artifact — i.e. it is genuinely *unstructured noise*,
   not a relabeled copy of either reference graph.

3. **The score-level γ-mix (the load-bearing two-graph-policy proxy) buys nothing stable.** Adding
   the residual hidden branch never produces an NLL gain over A-only that survives the seed noise
   floor and the matched-random control simultaneously. text-mlp shows a weak positive trend
   (γ-sweep all ≥0, beats matched control) but the largest gain (+0.042) is below 2× its A-only
   noise floor (2×0.031 = 0.062), so `stable = False`. Not enough to act on.

4. **Where the order signal actually lives (consistent with prior lines).** Image `B_A` is the only
   graph that is genuinely structured (sharpness z = +11.6), and in the image substrate everything
   co-varies with locality (`corr(B_A, B_pos)` = 0.90) — order signal = attention + position, not a
   position-independent hidden graph. Text `B_A` sharpness is *below* its null (z = −29 to −61),
   matching the known "text B_A has collapsed ≈ L2R / diffuse" observation.

## Decision (scoped — this closes the *static param-free* hidden route, not "hidden is useless")

Across primary (clean) + sculpted (mlp) text and the image primary, the frozen evidence does **not**
support a position-independent **static cosine** hidden relational graph. Two scoping caveats keep
this from being read as "hidden has no order signal":

1. **Symmetry is a confound, not just a result.** `B_H = cos(h_u,h_v)` is symmetric; order needs
   direction (`u→v ≠ v→u`). The C-D+L rollout *does* produce a directed order, but all direction is
   injected by the readout heuristic (start + greedy), identically for `B_A` and `B_H` — the *edge
   weights* of cosine carry no direction. So cosine can only say "which blocks cluster," never "which
   way to traverse." We have falsified the **symmetric** part of hidden geometry (post-position), not
   hidden order content in general.
2. **Attention already IS the frozen asymmetric hidden transition.** A single-step attention logit is
   `q(h_i)·k(h_j)` — exactly the asymmetric hidden-transition score one would build by hand, using the
   model's own projections. `B_A` is derived from it. So any param-free asymmetric hidden probe with
   the model's q,k reduces to (a slice of) `B_A`, which the diagnostic already uses as the primary
   graph. There is no *additional* param-free hidden signal to find.

**Falsified (closed) — the two param-free hidden-geometry probes:**
- hidden as **scalar residual predictor** (`s = s_G + Δs_H(v,x)`) → null ([[hidden_residual_diag_line]]).
- hidden as **static symmetric cosine graph** (`B_H = cos`) → null (this Stage A).

**NOT falsified (open) — all require training, i.e. they are the research program, not a diagnostic:**
- learned **bilinear** edge `h_u^⊤ W h_v` (asymmetric) / **edge-MLP** `g([h_u,h_v,h_u−h_v,h_u⊙h_v])`.
- **context-dependent dynamic controller** `s(v,x)=f(φ(B_A,·), h_v, h_{S_t}, h_{last})` — uses the
  *un-collapsed* per-step attention/hidden rather than the single collapsed `B_A` (text `B_A` has
  collapsed ≈ L2R), so it is **not** subsumed by static `B_A`. This is the most promising next line.
- larger-model hidden geometry.

Other similarity variants (dot-product, RBF) stay symmetric → expected to behave like cosine; **not
prioritized**. Confidence/uncertainty inputs **de-prioritized** (the attention-only MLP is already
saturated by structural signal — see [[text_phase0_residual_gate_result.md]]).

**Paper framing:** *Stage-A hidden-graph diagnostic shows that a static cosine-similarity graph over
hidden states is dominated by positional structure and provides no complementary order signal beyond
the attention graph. This closes the static, parameter-free hidden-graph route at the current scale,
but leaves open a more expressive context-dependent controller in which hidden states condition the
order policy dynamically, rather than being compressed into a fixed symmetric pairwise graph.*

**Gated next steps (NOT triggered):**
- **Stage-2 causal** (`extract_causal_hidden`, t∈{0,16,32,48}): gate = any Stage-A oracle WIN. All
  NULL → **skipped per spec** (oracle `B_H_resid` unstructured ⇒ no need to test the weaker causal
  hidden).
- **Stage B image contrast** (`vq64_fixed_random_l8h8e512`@30k): ckpt currently only at step 15000
  (baseline still training) — **pending**. Given the three NULLs, expected to confirm NULL; it is a
  control, not a decision point.

Reports: `probe_results/hidden_graph_diag/{text_clean_random_30k,text_alt_mlp_30k}/report.{json,md}`,
`probe_results_image/hidden_graph_diag/image_mlp_alt_30k/report.{json,md}`.
