# Evidence Package — 2026-06-13

> AO-GPT / Block-LO-ARM / frozen g_β — text-side evidence consolidation.
> **No new experiments. Read-only audit of existing results.**

---

## Purpose

This package consolidates all text-side experimental results into a unified, verifiable format suitable for:
- Advisor review / boss update
- Paper writing
- Decision on next experimental direction

## Structure

| File | Content |
|------|------|
| `00_README.md` | This index |
| `01_claims_lock.md` | CAN CLAIM / CANNOT CLAIM / NEED DECISION |
| `02_unified_tables.md` | 7 unified tables (A–G) with source paths |
| `03_result_inventory.md` | All relevant files with trust levels |
| `04_metric_definitions.md` | Precise definitions for all metrics |
| `05_conflicts_and_todo.md` | Unresolved conflicts and verification items |
| `06_paper_ready_summary.md` | 2–3 page paper-ready summary |
| `07_boss_update_outline.md` | 10–15 minute advisor presentation outline |
| `figures_index.md` | All figures with quality check |

## Key Conventions

1. **ori-L2R = reference, NOT upper bound.** Recovery > 100% is possible and meaningful.
2. **Per-head τ, not heavy τ.** Heavy τ averages positive and negative heads — meaningless.
3. **All numbers from primary sources** (eval_curve.tsv, head_scan JSON). Secondary sources (memory/*.md, analyses/*.py output) verified against primaries.
4. **Seed/baseline groups kept separate.** seed2 and seed42 have different random baselines.
5. **Global step must match** for all Recovery calculations. @50k used throughout.

## Generation Method

- Primary sources: `probe_results/*/eval_curve.tsv`, `probe_results/*/head_scan_*.json`, `probe_results/*/config.json`
- Cross-verified against: `analyses/*.py` outputs, `memory/*.md` summaries
- No new training runs, no code modifications
- Conflicts flagged in `05_conflicts_and_todo.md`

## Status

**Advisor-ready**: Yes — all key claims have multi-line evidence, all numbers traceable to source files.

**Paper-ready**: Mostly — tables and figures exist, but need final formatting. See `05_conflicts_and_todo.md` for remaining verification items.
