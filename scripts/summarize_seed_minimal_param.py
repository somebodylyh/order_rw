import sys
from pathlib import Path
import csv
seed = sys.argv[1]
root = Path(f'/home/admin/lyuyuhuan/order_lyu/probe_results_image_large/grw_e3ctrlsmall_round2_seed{seed}_minimal')
arms = ['cont_random','cont_Bcov_balanced','cont_distance_only_coverage','cont_shuffled_Bcov_balanced']
common = ['val_random','val_raster','val_hilbert','val_Bcov_balanced','val_rw_top4_eps0','val_rw_eps015','val_rw_topk8']
struct = ['val_raster','val_hilbert','val_Bcov_balanced','val_rw_top4_eps0','val_rw_topk8']
noisy = ['val_random','val_rw_eps015']
matched = {
  'cont_random':'val_random',
  'cont_Bcov_balanced':'val_Bcov_balanced',
  'cont_distance_only_coverage':'val_distance_only_coverage',
  'cont_shuffled_Bcov_balanced':'val_Bcov_balanced',
}
vals = {}
for arm in arms:
    path = root/arm/'eval_curve.tsv'
    rows = list(csv.DictReader(path.open(), delimiter='\t'))
    if int(rows[-1]['step']) < 5000:
        raise RuntimeError(f'{arm} not complete: step={rows[-1]["step"]}')
    r = rows[-1]
    vals[arm] = {
        'cross': sum(float(r[c]) for c in common)/len(common),
        'struct': sum(float(r[c]) for c in struct)/len(struct),
        'noisy': sum(float(r[c]) for c in noisy)/len(noisy),
        'matched': float(r[matched[arm]]) if matched[arm] in r else float('nan'),
    }
lines = []
lines.append(f'# Round-2 Seed {seed} Minimal Replication')
lines.append('')
lines.append('Additional seed for minimal stability check. Lower is better. Aggregates use common eval columns: random, raster, hilbert, Bcov_balanced, rw_top4, rw_eps015, rw_topk8.')
lines.append('')
lines.append('| arm | cross_avg | structured_avg | noisy_avg | matched/control |')
lines.append('|---|---:|---:|---:|---:|')
for arm in arms:
    v = vals[arm]
    lines.append(f'| {arm} | {v["cross"]:.4f} | {v["struct"]:.4f} | {v["noisy"]:.4f} | {v["matched"]:.4f} |')
lines.append('')
lines.append('## Key deltas')
for target in ['cont_random','cont_distance_only_coverage','cont_shuffled_Bcov_balanced']:
    lines.append(f'- Bcov_balanced - {target}: cross={vals["cont_Bcov_balanced"]["cross"]-vals[target]["cross"]:+.4f}, structured={vals["cont_Bcov_balanced"]["struct"]-vals[target]["struct"]:+.4f}, noisy={vals["cont_Bcov_balanced"]["noisy"]-vals[target]["noisy"]:+.4f}')
(root/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
print(root/'SUMMARY.md')
for arm in arms:
    v = vals[arm]
    print(f'{arm}\tcross={v["cross"]:.4f}\tstruct={v["struct"]:.4f}\tnoisy={v["noisy"]:.4f}\tmatched={v["matched"]:.4f}')
