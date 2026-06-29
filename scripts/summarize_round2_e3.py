#!/usr/bin/env python3
import csv
from pathlib import Path

OUT_ROOT = Path('/home/admin/lyuyuhuan/order_lyu/probe_results_image_large/grw_e3ctrlsmall_round2_5arm')
ARMS = ['cont_random', 'cont_v1_graph_rw', 'cont_hilbert', 'cont_Bcov_balanced', 'cont_raster']
EVAL_COLS = ['val_random', 'val_raster', 'val_hilbert', 'val_Bcov_balanced', 'val_rw_top4_eps0', 'val_rw_eps015', 'val_rw_topk8']
STRUCT = ['val_raster', 'val_hilbert', 'val_Bcov_balanced', 'val_rw_top4_eps0', 'val_rw_topk8']
NOISY = ['val_random', 'val_rw_eps015']
MATCHED = {
    'cont_random': 'val_random',
    'cont_v1_graph_rw': 'val_rw_top4_eps0',
    'cont_hilbert': 'val_hilbert',
    'cont_Bcov_balanced': 'val_Bcov_balanced',
    'cont_raster': 'val_raster',
}

def read_endpoint(arm):
    path = OUT_ROOT / arm / 'eval_curve.tsv'
    if not path.exists():
        raise FileNotFoundError(path)
    rows = list(csv.DictReader(path.open(), delimiter='\t'))
    if not rows:
        raise RuntimeError(f'no rows in {path}')
    return rows[-1]

def f(x):
    return float(x)

def md_table(headers, rows):
    out = ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join(['---'] * len(headers)) + '|']
    for row in rows:
        vals = []
        for v in row:
            vals.append(f'{v:.4f}' if isinstance(v, float) else str(v))
        out.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join(out)

def main():
    endpoints = {arm: read_endpoint(arm) for arm in ARMS}
    missing_done = [arm for arm, row in endpoints.items() if int(row['step']) < 5000]
    if missing_done:
        raise RuntimeError(f'not all arms reached step 5000: {missing_done}')

    endpoint_rows = []
    for arm in ARMS:
        row = endpoints[arm]
        endpoint_rows.append([arm] + [f(row[c]) for c in EVAL_COLS])

    base = endpoints['cont_random']
    delta_rows = []
    for arm in ARMS:
        if arm == 'cont_random':
            continue
        row = endpoints[arm]
        delta_rows.append([arm] + [f(row[c]) - f(base[c]) for c in EVAL_COLS])

    agg_rows = []
    for arm in ARMS:
        row = endpoints[arm]
        cross = sum(f(row[c]) for c in EVAL_COLS) / len(EVAL_COLS)
        struct = sum(f(row[c]) for c in STRUCT) / len(STRUCT)
        noisy = sum(f(row[c]) for c in NOISY) / len(NOISY)
        matched = f(row[MATCHED[arm]])
        agg_rows.append([arm, cross, struct, noisy, matched])

    agg_base = agg_rows[0]
    agg_delta_rows = []
    for row in agg_rows[1:]:
        agg_delta_rows.append([row[0], row[1] - agg_base[1], row[2] - agg_base[2], row[3] - agg_base[3], row[4]])

    bcov = next(r for r in agg_rows if r[0] == 'cont_Bcov_balanced')
    hil = next(r for r in agg_rows if r[0] == 'cont_hilbert')
    v1 = next(r for r in agg_rows if r[0] == 'cont_v1_graph_rw')
    rnd = agg_base
    ras = next(r for r in agg_rows if r[0] == 'cont_raster')

    lines = []
    lines.append('# Round-2 E3 5-arm SUMMARY')
    lines.append('')
    lines.append('Lower is better. Physical orders are remapped to model frame via `inverse_block_perm` inside `train_imagelarge_round2.py`. Bcov_balanced uses the frozen score `minmax(B[last,v]) - minmax(manh(last,v))` with `gamma_B=gamma_d=1.0`; training uses cyclic starts for the deterministic readout family.')
    lines.append('')
    lines.append('## Endpoint Eval Table')
    lines.append('')
    lines.append(md_table(['arm'] + EVAL_COLS, endpoint_rows))
    lines.append('')
    lines.append('## Delta vs cont_random')
    lines.append('')
    lines.append(md_table(['arm'] + EVAL_COLS, delta_rows))
    lines.append('')
    lines.append('## Aggregates')
    lines.append('')
    lines.append(md_table(['arm', 'cross_avg', 'structured_avg', 'noisy_avg', 'matched_order'], agg_rows))
    lines.append('')
    lines.append('## Aggregate Delta vs cont_random')
    lines.append('')
    lines.append(md_table(['arm', 'delta_cross', 'delta_structured', 'delta_noisy', 'matched_order'], agg_delta_rows))
    lines.append('')
    lines.append('## Primary Comparison')
    lines.append('')
    lines.append(f'Bcov_balanced vs Hilbert: cross_avg delta = {bcov[1] - hil[1]:+.4f}; structured_avg delta = {bcov[2] - hil[2]:+.4f}; noisy_avg delta = {bcov[3] - hil[3]:+.4f}.')
    lines.append(f'Secondary: Hilbert vs random cross delta = {hil[1] - rnd[1]:+.4f}; Bcov_balanced vs v1_graph_rw cross delta = {bcov[1] - v1[1]:+.4f}; raster remains specialization reference with matched_order={ras[4]:.4f}.')
    lines.append('')
    lines.append('## Interpretation Frame')
    lines.append('')
    lines.append('This tests the graph-regime diagnostic framing: E3 B is diagnosed as proximity-dominant/hybrid, so the relevant question is whether B-guided proximity coverage improves validation loss, cross-order robustness, or sample quality beyond generic Hilbert and failed-readout v1 Graph-RW. Mean Manhattan distance is not the final objective.')
    lines.append('')
    lines.append('Negative, neutral, and positive outcomes are all interpretable under the frozen Stage-1 conclusion; do not treat this as changing Stage-1 unless a concrete bug is found.')
    (OUT_ROOT / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')
    print(OUT_ROOT / 'SUMMARY.md')

if __name__ == '__main__':
    main()
