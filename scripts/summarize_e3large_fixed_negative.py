from pathlib import Path
import csv, sys
seed = sys.argv[1] if len(sys.argv) > 1 else '42'
root=Path(f'/home/admin/lyuyuhuan/order_lyu/probe_results_image_large/e3large_fixed_round2_negative_seed{seed}')
arms=['cont_random','cont_hilbert','cont_Bcov_balanced','cont_distance_only_coverage','cont_shuffled_Bcov_balanced']
common=['val_random','val_raster','val_hilbert','val_Bcov_balanced','val_rw_top4_eps0','val_rw_eps015','val_rw_topk8']
struct=['val_raster','val_hilbert','val_Bcov_balanced','val_rw_top4_eps0','val_rw_topk8']
noisy=['val_random','val_rw_eps015']
matched={'cont_random':'val_random','cont_hilbert':'val_hilbert','cont_Bcov_balanced':'val_Bcov_balanced','cont_distance_only_coverage':'val_distance_only_coverage','cont_shuffled_Bcov_balanced':'val_Bcov_balanced'}
vals={}
for arm in arms:
    rows=list(csv.DictReader((root/arm/'eval_curve.tsv').open(), delimiter='\t'))
    if int(rows[-1]['step']) < 5000:
        raise RuntimeError(f'{arm} incomplete')
    r=rows[-1]
    vals[arm]={
        'cross':sum(float(r[c]) for c in common)/len(common),
        'struct':sum(float(r[c]) for c in struct)/len(struct),
        'noisy':sum(float(r[c]) for c in noisy)/len(noisy),
        'matched':float(r[matched[arm]]) if matched[arm] in r else float('nan'),
    }
lines=['# E3-large Fixed Negative/Fallback Validation','', 'Diagnostic regime: `random_or_no_structure_fallback`. Single-seed validation. Lower is better.', '', '| arm | cross_avg | structured_avg | noisy_avg | matched/control |', '|---|---:|---:|---:|---:|']
for arm in arms:
    v=vals[arm]
    lines.append(f'| {arm} | {v["cross"]:.4f} | {v["struct"]:.4f} | {v["noisy"]:.4f} | {v["matched"]:.4f} |')
lines += ['', '## Key deltas']
for target in ['cont_random','cont_hilbert','cont_distance_only_coverage','cont_shuffled_Bcov_balanced']:
    lines.append(f'- Bcov_balanced - {target}: cross={vals["cont_Bcov_balanced"]["cross"]-vals[target]["cross"]:+.4f}, structured={vals["cont_Bcov_balanced"]["struct"]-vals[target]["struct"]:+.4f}, noisy={vals["cont_Bcov_balanced"]["noisy"]-vals[target]["noisy"]:+.4f}')
lines += ['', '## Interpretation', '', 'This validates the graph-regime diagnostic on a graph diagnosed as no-structure/fallback. The expected outcome is that Bcov should not produce the same robust advantage seen on E3-control-small.']
(root/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
print(root/'SUMMARY.md')
for arm,v in vals.items():
    print(f'{arm}\tcross={v["cross"]:.4f}\tstruct={v["struct"]:.4f}\tnoisy={v["noisy"]:.4f}\tmatched={v["matched"]:.4f}')
