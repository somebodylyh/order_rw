from pathlib import Path
import csv, math, statistics

BASE = Path('/home/admin/lyuyuhuan/order_lyu/probe_results_image_large')
SEED_ROOTS = {
    42: BASE/'grw_e3ctrlsmall_round2_5arm',
    123: BASE/'grw_e3ctrlsmall_round2_seed123_minimal',
    314: BASE/'grw_e3ctrlsmall_round2_seed314_minimal',
    777: BASE/'grw_e3ctrlsmall_round2_seed777_minimal',
    2024: BASE/'grw_e3ctrlsmall_round2_seed2024_minimal',
}
ARMS = ['cont_random','cont_Bcov_balanced','cont_distance_only_coverage','cont_shuffled_Bcov_balanced']
COMMON = ['val_random','val_raster','val_hilbert','val_Bcov_balanced','val_rw_top4_eps0','val_rw_eps015','val_rw_topk8']
STRUCT = ['val_raster','val_hilbert','val_Bcov_balanced','val_rw_top4_eps0','val_rw_topk8']
NOISY = ['val_random','val_rw_eps015']
MATCHED = {
    'cont_random':'val_random',
    'cont_Bcov_balanced':'val_Bcov_balanced',
    'cont_distance_only_coverage':'val_distance_only_coverage',
    'cont_shuffled_Bcov_balanced':'val_Bcov_balanced',
}
OUT = BASE/'grw_e3ctrlsmall_round2_multiseed_minimal'
OUT.mkdir(parents=True, exist_ok=True)

def read_arm(root, arm):
    path = root/arm/'eval_curve.tsv'
    rows = list(csv.DictReader(path.open(), delimiter='\t'))
    if int(rows[-1]['step']) < 5000:
        raise RuntimeError(f'{path} incomplete: step={rows[-1]["step"]}')
    r = rows[-1]
    return {
        'cross': sum(float(r[c]) for c in COMMON)/len(COMMON),
        'structured': sum(float(r[c]) for c in STRUCT)/len(STRUCT),
        'noisy': sum(float(r[c]) for c in NOISY)/len(NOISY),
        'matched': float(r[MATCHED[arm]]) if MATCHED[arm] in r else float('nan'),
    }

def mean_std(xs):
    return statistics.mean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0

rows = []
vals = {}
for seed, root in SEED_ROOTS.items():
    vals[seed] = {}
    for arm in ARMS:
        vals[seed][arm] = read_arm(root, arm)
        v = vals[seed][arm]
        rows.append([seed, arm, v['cross'], v['structured'], v['noisy'], v['matched']])

with (OUT/'per_seed.tsv').open('w') as f:
    f.write('seed\tarm\tcross\tstructured\tnoisy\tmatched\n')
    for row in rows:
        f.write('\t'.join([str(row[0]), row[1]] + [f'{x:.6f}' for x in row[2:]]) + '\n')

summary = {}
for arm in ARMS:
    summary[arm] = {}
    for metric in ['cross','structured','noisy','matched']:
        xs = [vals[s][arm][metric] for s in SEED_ROOTS]
        summary[arm][metric] = mean_std(xs)

deltas = []
for seed in SEED_ROOTS:
    b = vals[seed]['cont_Bcov_balanced']
    for target in ['cont_random','cont_distance_only_coverage','cont_shuffled_Bcov_balanced']:
        t = vals[seed][target]
        deltas.append([seed, 'Bcov_minus_'+target, b['cross']-t['cross'], b['structured']-t['structured'], b['noisy']-t['noisy']])
with (OUT/'deltas.tsv').open('w') as f:
    f.write('seed\tcomparison\tdelta_cross\tdelta_structured\tdelta_noisy\n')
    for row in deltas:
        f.write('\t'.join([str(row[0]), row[1]] + [f'{x:.6f}' for x in row[2:]]) + '\n')

lines = []
lines.append('# Round-2 Minimal Multi-Seed Summary')
lines.append('')
lines.append('Seeds: ' + ', '.join(map(str, SEED_ROOTS.keys())) + '. Lower is better. This table covers the minimal stability matrix only: random, Bcov_balanced, distance_only_coverage, shuffled_Bcov_balanced.')
lines.append('')
lines.append('| arm | cross mean±std | structured mean±std | noisy mean±std | matched mean±std |')
lines.append('|---|---:|---:|---:|---:|')
for arm in ARMS:
    c=summary[arm]['cross']; st=summary[arm]['structured']; no=summary[arm]['noisy']; m=summary[arm]['matched']
    lines.append(f'| {arm} | {c[0]:.4f} ± {c[1]:.4f} | {st[0]:.4f} ± {st[1]:.4f} | {no[0]:.4f} ± {no[1]:.4f} | {m[0]:.4f} ± {m[1]:.4f} |')
lines.append('')
lines.append('## Delta Summary')
lines.append('')
lines.append('| comparison | delta_cross mean±std | delta_structured mean±std | delta_noisy mean±std |')
lines.append('|---|---:|---:|---:|')
for target in ['cont_random','cont_distance_only_coverage','cont_shuffled_Bcov_balanced']:
    name='Bcov_minus_'+target
    dx=[r[2] for r in deltas if r[1]==name]; ds=[r[3] for r in deltas if r[1]==name]; dn=[r[4] for r in deltas if r[1]==name]
    mx,sx=mean_std(dx); ms,ss=mean_std(ds); mn,sn=mean_std(dn)
    lines.append(f'| Bcov - {target} | {mx:+.4f} ± {sx:.4f} | {ms:+.4f} ± {ss:.4f} | {mn:+.4f} ± {sn:.4f} |')
lines.append('')
lines.append('## Interpretation')
lines.append('')
lines.append('Use this as the minimal multi-seed stability check for the Round-2 pilot. A negative delta means Bcov_balanced is better. This file is generated only after every listed seed reaches step 5000.')
(OUT/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
print(OUT/'SUMMARY.md')
