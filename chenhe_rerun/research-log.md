# Research Log

This log has been compacted after the language-only cleanup. The removed
non-language history remains recoverable from Git history and the remote backup
branch.

| ID | Date | Topic | Note |
| --- | --- | --- | --- |
| 1 | 2026-05 to 2026-06 | language recovery | Developed AO-GPT learned-order experiments around WikiText103 random reveal-order training, current-frame pair mining, and curriculum recovery. |
| 2 | 2026-06 | online order | Added online spectral fixed-head and distribution order teachers with current-frame attention/loss reranking. |
| 3 | 2026-06 | diagnostics | Established that original-frame views, original tau, and `OriginalL2R` are diagnostics/oracle baselines only under permutation. |
| 4 | 2026-06 | Attn-MLP | Shifted the active plan toward distilling validated attention/loss teachers into MLP order policies. |
| 5 | 2026-06-25 | cleanup | Backed up the mixed repository to `language-graph-laplacian-decomposition-20260624`, then removed non-language data, outputs, configs, dependencies, reports, and prompts from the working tree. |
| 6 | 2026-06-26 | current mainline | Re-centered docs on the L0 layer-mean pairwise-max Fiedler distribution teacher, EMA/no-EMA ablation, and Fiedler-teacher MLP distillation/insertion tests. |

Current detailed status lives in:

```text
docs/findings/findings_language.md
docs/prompts/prompt_language_block_current_task.md
Report/language/wikitext103/order_teacher_distribution/head_signal_stability/current_distribution_methodology_summary.md
Report/language/wikitext103/mlp/distillation/README.md
Report/language/wikitext103/mlp/distillation/HISTORY_ARCHIVE_20260624.md
Report/language/wikitext103/
```
