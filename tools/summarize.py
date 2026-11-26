#!/usr/bin/env python3
"""Summarise scored runs into paper-style tables (mean ± std over seeds).

Run folders are named <config>_s<seed> (e.g. ours_s42, ours_s1, ours_s2).
Requires tools/score_metrics.py and tools/compute_f05.py to have been run.

Usage:
  python tools/summarize.py --root outputs [--metric Sm] [--csv outputs/summary.csv]
"""
import argparse
import csv
import os
import re
import statistics as st
from collections import defaultdict

DATASETS = ['DUTS-TE', 'DUT-O', 'ECSSD', 'HKU-IS', 'PASCAL-S-full', 'SIP-full']
HEADER = ['DUTS-TE', 'DUT-OMRON', 'ECSSD', 'HKU-IS', 'PASCAL-S', 'SIP']


def load(run_dir):
    p = os.path.join(run_dir, 'unified_metrics.csv')
    if not os.path.exists(p):
        return None
    r = {x['dataset']: {k: float(v) for k, v in x.items() if k != 'dataset'}
         for x in csv.DictReader(open(p))}
    q = os.path.join(run_dir, 'f05_fixed.csv')
    if os.path.exists(q):
        for x in csv.DictReader(open(q)):
            r.setdefault(x['dataset'], {})['F05'] = float(x['F05'])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='outputs')
    ap.add_argument('--metric', default='Sm', help='Sm, F05, meanE, maxF, MAE, ...')
    ap.add_argument('--csv', default='')
    a = ap.parse_args()

    groups = defaultdict(list)
    for name in sorted(os.listdir(a.root)):
        d = os.path.join(a.root, name)
        r = load(d) if os.path.isdir(d) else None
        if r:
            groups[re.sub(r'_s\d+$', '', name)].append(r)

    rows = []
    print(f"{'config':32s} n  " + '  '.join(f'{h:>13s}' for h in HEADER))
    for cfg, runs in groups.items():
        cells = []
        for d in DATASETS:
            v = [r[d][a.metric] for r in runs if d in r and a.metric in r[d]]
            if not v:
                cells.append('-')
            elif len(v) == 1:
                cells.append(f'{v[0]:.4f}')
            else:
                cells.append(f'{st.mean(v):.4f}±{st.stdev(v):.4f}')
        print(f'{cfg:32s} {len(runs)}  ' + '  '.join(f'{c:>13s}' for c in cells))
        rows.append([cfg, len(runs)] + cells)
    if a.csv:
        with open(a.csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['config', 'runs'] + HEADER)
            w.writerows(rows)


if __name__ == '__main__':
    main()
