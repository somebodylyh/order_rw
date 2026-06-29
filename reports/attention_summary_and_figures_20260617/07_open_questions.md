# Open Questions

| Question | Why it matters | Current evidence | Suggested action |
|---|---|---|---|
| Has strict 65-node teacher been distilled to `g_beta`? | Needed to connect mechanism discovery to controller training. | No strict-teacher `g_beta` result found in requested sources. | Train/distill strict-teacher `g_beta` as next P0. |
| Has strict 65-node hook been run? | Needed before claiming strict teacher acceleration. | Existing acceleration is legacy B0 controller path. | Run short hook smoke with strict teacher. |
| Can strong heads be selected by label-free audition? | Avoids oracle tau head selection. | Strong heads exist, but end-to-end label-free audition not closed. | Define audition score and test whether it selects strong-pass heads. |
| Is 317M strict protocol verified? | Needed for scale claim under final protocol. | 317M evidence found only as older B0 diagnostic, not strict 65-node. | Run strict 65-node diagnostic on 317M if scale claim is needed. |
| Is multi-seed strict discovery verified? | Needed for robustness beyond collaborator/clean-base runs. | Clean-base and collaborator both show selected strong heads, but head identity drifts. | Add more seeds/checkpoints under strict LF. |
| Do we need a strict 65-node schematic figure? | Boss update needs a clean visual of the final protocol. | No existing schematic found. | Generate schematic manually or with a small plotting script. |
| Do we need unified L0H1-L0H4 heatmaps? | Strong-pass figure would make mechanism evidence easier to communicate. | A_with_none data exists, but no unified L0H1-L0H4 PNG found. | Generate a 4-panel B65 heatmap from saved `A_with_none_lh_mean_MODEL_FRAME.npy`. |
| Should token-level maps be shown? | Helps explain "bidirectional-looking original-frame map" confusion. | Existing model-frame and original-L2R token maps are available. | Use only if the boss asks about token-level attention directionality. |

