#!/usr/bin/env python3
"""Compute-normalized saving analysis: wall-clock to target ori_l2r.

Usage:
    python wall_clock_saving.py --v3 <v3_log> --random <random_log> --l2r <l2r_log> [--gbeta-cost 2100]

Outputs:
    1. Table: wall-clock to each ori_l2r threshold with saving %
    2. Tab-separated data for plotting: (run, wall_seconds, ori_l2r)
"""
import re, argparse, sys
import numpy as np

def parse_wallclock(log_path):
    """Extract (step, ori_l2r, wall_seconds) from training log."""
    points = []
    with open(log_path) as f:
        for line in f:
            m = re.search(r'\[Eval @ (\d+)\].*?ori_l2r=([\d.]+)', line)
            if m:
                step = int(m.group(1))
                ori = float(m.group(2))
                points.append({'step': step, 'ori_l2r': ori, 'wall': None})

            m2 = re.search(r'step\s+\d+->\d+/\d+.*?total\s+(\d+)s', line)
            if m2 and points:
                points[-1]['wall'] = int(m2.group(1))

    # Fill missing wall times by linear interpolation
    for i, p in enumerate(points):
        if p['wall'] is None:
            for j in range(i-1, -1, -1):
                if points[j]['wall'] is not None:
                    prev_wall, prev_step = points[j]['wall'], points[j]['step']
                    for k in range(i+1, len(points)):
                        if points[k]['wall'] is not None:
                            next_wall, next_step = points[k]['wall'], points[k]['step']
                            frac = (p['step'] - prev_step) / max(1, next_step - prev_step)
                            points[i]['wall'] = int(prev_wall + frac * (next_wall - prev_wall))
                            break
                    break

    return [(p['wall'], p['ori_l2r']) for p in points if p['wall'] is not None]


def wall_to_target(data, target):
    """First wall time (seconds) where ori_l2r < target."""
    for wall, ori in data:
        if ori < target:
            return wall
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--v3', required=True, help='V3/frozen_beta train log')
    ap.add_argument('--random', required=True, help='random baseline train log')
    ap.add_argument('--l2r', default=None, help='ori-L2R train log (optional)')
    ap.add_argument('--gbeta-cost', type=int, default=2100,
                    help='g_beta pretraining wall-clock cost in seconds (default 2100)')
    ap.add_argument('--targets', type=float, nargs='*',
                    default=[3.60, 3.55, 3.50, 3.47, 3.45, 3.42])
    ap.add_argument('--plot-data', default=None, help='output TSV for plotting')
    ap.add_argument('--step-target', type=float, default=3.47,
                    help='target ori_l2r for main saving number')
    ap.add_argument('--random-step-target', type=int, default=None,
                    help='step count for random to reach target (for step-saving calc)')
    ap.add_argument('--v3-step-target', type=int, default=None,
                    help='step count for V3 to reach target (for step-saving calc)')
    args = ap.parse_args()

    v3 = parse_wallclock(args.v3)
    rnd = parse_wallclock(args.random)
    l2r = parse_wallclock(args.l2r) if args.l2r else None

    print(f"V3: {len(v3)} eval points, wall range [{v3[0][0]}, {v3[-1][0]}]s")
    print(f"Random: {len(rnd)} eval points, wall range [{rnd[0][0]}, {rnd[-1][0]}]s")
    if l2r:
        print(f"L2R: {len(l2r)} eval points")
    print()

    GB = args.gbeta_cost

    # Header
    header = f"{'Target':<10} {'Random wall':>14} {'V3 wall':>14} {'V3+gb wall':>14} {'saving':>8} {'saving+gb':>8}"
    print(header)
    print("-" * len(header))

    for t in args.targets:
        w_rnd = wall_to_target(rnd, t)
        w_v3 = wall_to_target(v3, t)
        if w_rnd and w_v3:
            sv = (w_rnd - w_v3) / w_rnd * 100
            sv_gb = (w_rnd - w_v3 - GB) / w_rnd * 100
            print(f"ori<{t:<7} {w_rnd:>7}s ({w_rnd/3600:.1f}h) {w_v3:>7}s ({w_v3/3600:.1f}h) {w_v3+GB:>7}s ({(w_v3+GB)/3600:.1f}h) {sv:>7.1f}% {sv_gb:>7.1f}%")
        elif w_v3:
            print(f"ori<{t:<7} {'--':>14} {w_v3:>7}s ({w_v3/3600:.1f}h) --")
        elif w_rnd:
            print(f"ori<{t:<7} {w_rnd:>7}s ({w_rnd/3600:.1f}h) {'--':>14} --")

    # Main number
    t_main = args.step_target
    w_rnd_main = wall_to_target(rnd, t_main)
    w_v3_main = wall_to_target(v3, t_main)
    if w_rnd_main and w_v3_main:
        sv_wall = (w_rnd_main - w_v3_main) / w_rnd_main * 100
        sv_wall_gb = (w_rnd_main - w_v3_main - GB) / w_rnd_main * 100
        print(f"\n=== Main result: wall-clock to ori_l2r < {t_main} ===")
        print(f"Random: {w_rnd_main}s = {w_rnd_main/3600:.1f}h ({w_rnd_main/60:.0f}min)")
        print(f"V3:     {w_v3_main}s = {w_v3_main/3600:.1f}h ({w_v3_main/60:.0f}min)")
        print(f"V3+gb:  {w_v3_main+GB}s = {(w_v3_main+GB)/3600:.1f}h ({(w_v3_main+GB)/60:.0f}min)")
        print(f"Wall saving: {sv_wall:.0f}%  (incl g_beta: {sv_wall_gb:.0f}%)")

        if args.random_step_target and args.v3_step_target:
            sv_step = (args.random_step_target - args.v3_step_target) / args.random_step_target * 100
            print(f"Step saving: {sv_step:.0f}%  (random={args.random_step_target}, V3={args.v3_step_target})")

    # Latest ori_l2r
    print(f"\n=== Latest eval ===")
    print(f"V3:     wall={v3[-1][0]}s  ori_l2r={v3[-1][1]:.4f}")
    print(f"Random: wall={rnd[-1][0]}s  ori_l2r={rnd[-1][1]:.4f}")
    if l2r:
        print(f"L2R:    wall={l2r[-1][0]}s  ori_l2r={l2r[-1][1]:.4f}")

    # Optional: output plot data
    if args.plot_data:
        with open(args.plot_data, 'w') as f:
            f.write("run\twall_s\tori_l2r\n")
            for wall, ori in rnd:
                f.write(f"random\t{wall}\t{ori:.6f}\n")
            for wall, ori in v3:
                f.write(f"V3_matched\t{wall}\t{ori:.6f}\n")
            if l2r:
                for wall, ori in l2r:
                    f.write(f"ori_L2R\t{wall}\t{ori:.6f}\n")
        print(f"\nPlot data saved to {args.plot_data}")


if __name__ == '__main__':
    main()
