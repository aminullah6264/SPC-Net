#!/usr/bin/env python3
"""Mean per-image F_beta (beta^2=0.3) at a FIXED threshold 0.5, computed from saved
continuous saliency maps (uint8 PNG = prob*255). Independent of the threshold used
during evaluation. Writes <run>/f05_fixed.csv.

Usage: python tools/compute_f05.py --roots outputs
"""
import argparse, csv, os, sys
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import score_metrics as SU
import clipsam


def fb(pred, gt, b2=0.3):
    tp = np.logical_and(pred, gt).sum(dtype=np.float64)
    ps, gs = pred.sum(dtype=np.float64), gt.sum(dtype=np.float64)
    p = tp / ps if ps > 0 else 0.0
    r = tp / gs if gs > 0 else 0.0
    return 0.0 if p == 0 and r == 0 else (1 + b2) * p * r / (b2 * p + r)


def one(args):
    vdir, name = args
    d = os.path.join(vdir, 'saliency_maps', clipsam._safe_split_dirname(name))
    if not os.path.isdir(d):
        return None
    get_gt = SU.gt_lookup(name)
    vals = []
    for f in sorted(os.listdir(d)):
        if not f.endswith('.png'):
            continue
        g = get_gt(os.path.splitext(f)[0])
        if g is None:
            continue
        m = Image.open(os.path.join(d, f)).convert('L')
        if m.size != (g.shape[1], g.shape[0]):
            m = m.resize((g.shape[1], g.shape[0]), Image.BILINEAR)
        vals.append(fb(np.array(m) >= 128, g > 127))
    return (vdir, name, float(np.mean(vals)), len(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--roots', nargs='+', required=True)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    jobs = []
    for root in a.roots:
        for v in sorted(os.listdir(root)):
            vd = os.path.join(root, v)
            if os.path.isdir(os.path.join(vd, 'saliency_maps')):
                jobs += [(vd, n) for n in SU.DATASETS]
    res = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(one, jobs):
            if r:
                res.setdefault(r[0], []).append(r[1:])
    for vd, rows in res.items():
        with open(os.path.join(vd, 'f05_fixed.csv'), 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['dataset', 'F05', 'n']); w.writerows(rows)
    print('wrote', len(res), 'variants')


if __name__ == '__main__':
    main()
