#!/usr/bin/env python3
"""Score saved continuous saliency maps with PySODMetrics.

Metric definitions (pysodmetrics, beta^2 = 0.3):
  Sm      S-measure (alpha=0.5) on the continuous map
  maxF    max over 256 thresholds of the dataset-mean F-measure curve
  meanF   mean of that curve
  adpF    adaptive F (threshold = 2 x mean saliency, per image)
  wF      weighted F-measure
  meanE   mean over 256 thresholds of the dataset-mean E-measure curve
  maxE    max of that curve
  adpE    adaptive E-measure
  MAE     mean absolute error on the continuous map

Usage:
  python tools/score_metrics.py --root outputs             # every run folder in outputs/
  python tools/score_metrics.py --root outputs --variants ours_s42
Writes <run>/unified_metrics.csv and <root>/UNIFIED_SUMMARY.csv.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
import argparse
import csv
import os

import numpy as np
from PIL import Image
import py_sod_metrics as M

import clipsam

DATASETS = ['DUTS-TE', 'DUT-O', 'ECSSD', 'HKU-IS', 'PASCAL-S-full', 'SIP', 'SIP-full']
FIELDS = ['Sm', 'maxF', 'meanF', 'adpF', 'wF', 'meanE', 'maxE', 'adpE', 'MAE', 'n']


def gt_lookup(name):
    """Return a function stem -> uint8 GT mask (0/255)."""
    reg = clipsam.BENCHMARK_REGISTRY[name]
    if reg['type'] == 'npy':
        masks = np.load(reg['masks_npy'], allow_pickle=True)

        def get(stem):
            m = clipsam._as_mask_float(masks[int(stem)])
            return ((m > 0.5) * 255).astype(np.uint8)
        return get
    mdir = reg['masks']
    same = os.path.abspath(reg['images']) == os.path.abspath(mdir)

    def get(stem):
        p = (os.path.join(mdir, stem + '.png') if same
             else clipsam.find_mask_path(mdir, stem + '.jpg'))
        if p is None or not os.path.exists(p):
            return None
        g = np.array(Image.open(p).convert('L'))
        return ((g > 127) * 255).astype(np.uint8)
    return get


def score_dir(map_dir, name):
    get_gt = gt_lookup(name)
    sm, em, fm, wfm, mae = M.Smeasure(), M.Emeasure(), M.Fmeasure(), M.WeightedFmeasure(), M.MAE()
    n = 0
    for fn in sorted(os.listdir(map_dir)):
        if not fn.endswith('.png'):
            continue
        gt = get_gt(os.path.splitext(fn)[0])
        if gt is None:
            continue
        pred = np.array(Image.open(os.path.join(map_dir, fn)).convert('L'))
        if pred.shape != gt.shape:
            pred = np.array(Image.fromarray(pred).resize(gt.shape[::-1], Image.BILINEAR))
        for m in (sm, em, fm, wfm, mae):
            m.step(pred=pred, gt=gt)
        n += 1
    if n == 0:
        return None
    e, f = em.get_results()['em'], fm.get_results()['fm']
    return {'Sm': float(sm.get_results()['sm']), 'maxF': float(f['curve'].max()),
            'meanF': float(f['curve'].mean()), 'adpF': float(f['adp']),
            'wF': float(wfm.get_results()['wfm']), 'meanE': float(e['curve'].mean()),
            'maxE': float(e['curve'].max()), 'adpE': float(e['adp']),
            'MAE': float(mae.get_results()['mae']), 'n': n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--variants', default='', help='comma list; default = all subfolders')
    ap.add_argument('--force', action='store_true', help='rescore even if CSV exists')
    args = ap.parse_args()
    variants = ([v for v in args.variants.split(',') if v] or
                sorted(d for d in os.listdir(args.root)
                       if os.path.isdir(os.path.join(args.root, d, 'saliency_maps'))))
    summary = []
    for v in variants:
        vdir = os.path.join(args.root, v)
        out_csv = os.path.join(vdir, 'unified_metrics.csv')
        rows = []
        if os.path.exists(out_csv) and not args.force:
            with open(out_csv) as f:
                rows = list(csv.DictReader(f))
        else:
            for name in DATASETS:
                d = os.path.join(vdir, 'saliency_maps', clipsam._safe_split_dirname(name))
                if not os.path.isdir(d):
                    continue
                r = score_dir(d, name)
                if r:
                    rows.append({'dataset': name, **{k: (r[k] if k == 'n' else f"{r[k]:.4f}")
                                                     for k in FIELDS}})
                    print(f"{v:22s} {name:14s} " + "  ".join(f"{k}={rows[-1][k]}" for k in FIELDS),
                          flush=True)
            if rows:
                with open(out_csv, 'w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=['dataset', *FIELDS])
                    w.writeheader(); w.writerows(rows)
        summary += [{'variant': v, **r} for r in rows]
    if summary:
        with open(os.path.join(args.root, 'UNIFIED_SUMMARY.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['variant', 'dataset', *FIELDS])
            w.writeheader(); w.writerows(summary)
        print(f"wrote {os.path.join(args.root, 'UNIFIED_SUMMARY.csv')} ({len(summary)} rows)")


if __name__ == '__main__':
    main()
