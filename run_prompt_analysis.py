#!/usr/bin/env python3
"""Prompt bank analysis experiments for SPC-Net.

Runs variants: K sweep (1,5,10,20), aggregation (max/mean),
wording (curated/random), and point-loss weight (0, 0.3, 1.0).

Usage:
    python run_prompt_analysis.py --gpu 1 --epochs 60 --seed 42
    python run_prompt_analysis.py --gpu 1 --variants K1,K5,K20  # subset
    python run_prompt_analysis.py --smoke  # verify output format
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path


# Variant definitions: name -> extra flags (relative to full model defaults)
ANALYSIS_VARIANTS = OrderedDict([
    ('K1', ['--prompt-bank-size', '1']),
    ('K5', ['--prompt-bank-size', '5']),
    ('K10_default', []),  # Full model default — skip if already run
    ('K20', ['--prompt-bank-size', '20']),
    ('agg_mean', ['--prompt-agg', 'mean']),
    ('random_bank', ['--random-prompts']),
    ('gamma0', ['--pp-loss-weight', '0']),
    ('gamma03_default', []),  # Full model default — skip if already run
    ('gamma1', ['--pp-loss-weight', '1.0']),
])

# Skip these (they are the full model default)
SKIP_VARIANTS = {'K10_default', 'gamma03_default'}

METRIC_COLUMNS = ['s_measure', 'max_f', 'mean_e', 'weighted_f', 'mae']


def parse_args():
    p = argparse.ArgumentParser(description='Run SPC-Net prompt bank analysis')
    p.add_argument('--gpu', type=int, default=1, help='GPU device ID')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output-dir', default='outputs/prompt_analysis')
    p.add_argument('--variants', type=str, default=None,
                   help='Comma-separated subset of variants to run')
    p.add_argument('--smoke', action='store_true',
                   help='Write dummy outputs without training')
    p.add_argument('--clip-model', default='ViT-L-14')
    return p.parse_args()


def read_metrics(run_dir):
    """Read metrics from results_summary.md or results.yaml."""
    import yaml
    yaml_path = Path(run_dir) / 'results.yaml'
    if yaml_path.exists():
        with open(yaml_path) as f:
            doc = yaml.safe_load(f)
        # Try to get DUTS-TE metrics from benchmarks
        if 'benchmarks' in doc and 'DUTS-TE' in doc['benchmarks']:
            m = doc['benchmarks']['DUTS-TE']
            return {
                's_measure': m.get('s_measure', m.get('Sm', 0)),
                'max_f': m.get('max_f', m.get('maxFm', 0)),
                'mean_e': m.get('mean_e', m.get('meanEm', 0)),
                'weighted_f': m.get('weighted_f', m.get('wFm', 0)),
                'mae': m.get('mae', m.get('MAE', 0)),
            }
    # Fallback: check eval_results.json from PySODEvalToolkit
    json_path = Path(run_dir) / 'saliency_maps' / 'eval_results.json'
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        if 'DUTS-TE' in data:
            m = data['DUTS-TE']
            return {
                's_measure': m.get('Sm', 0),
                'max_f': m.get('maxFm', 0),
                'mean_e': m.get('meanEm', 0),
                'weighted_f': m.get('wFm', 0),
                'mae': m.get('MAE', 0),
            }
    raise FileNotFoundError(f"No metrics found in {run_dir}")


def smoke_rows():
    rows = []
    for i, (name, flags) in enumerate(ANALYSIS_VARIANTS.items()):
        if name in SKIP_VARIANTS:
            continue
        rows.append({
            'variant': name,
            's_measure': 0.880 + i * 0.005,
            'max_f': 0.850 + i * 0.005,
            'mean_e': 0.910 + i * 0.005,
            'weighted_f': 0.820 + i * 0.005,
            'mae': 0.045 - i * 0.002,
            'flags': ' '.join(flags) if flags else '(default)',
        })
    return rows


def write_results_md(rows, output_path):
    """Write results as a markdown table."""
    lines = ['# Prompt Bank Analysis Results\n',
             '| Variant | Sm ↑ | maxFβ ↑ | meanEm ↑ | wFβ ↑ | MAE ↓ |',
             '|---------|------|---------|----------|-------|-------|']
    for r in rows:
        lines.append(
            f"| {r['variant']} | {float(r['s_measure']):.4f} | {float(r['max_f']):.4f} | "
            f"{float(r['mean_e']):.4f} | {float(r['weighted_f']):.4f} | {float(r['mae']):.4f} |")
    lines.append('')
    Path(output_path).write_text('\n'.join(lines))


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        rows = smoke_rows()
        write_results_md(rows, out_dir / 'prompt_analysis_results.md')
        print("Smoke test complete.")
        print((out_dir / 'prompt_analysis_results.md').read_text())
        return

    # Determine which variants to run
    if args.variants:
        selected = [v.strip() for v in args.variants.split(',')]
    else:
        selected = [k for k in ANALYSIS_VARIANTS if k not in SKIP_VARIANTS]

    rows = []
    for variant in selected:
        if variant not in ANALYSIS_VARIANTS:
            print(f"[WARN] Unknown variant: {variant}, skipping")
            continue
        flags = ANALYSIS_VARIANTS[variant]
        run_dir = out_dir / variant
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, 'clipsam.py',
            '--clip-model', args.clip_model,
            '--epochs', str(args.epochs),
            '--seed', str(args.seed),
            '--batch-size', '8',
            '--num-workers', '8',
            '--mse-lambda', '0.1',
            '--output-dir', str(run_dir),
            '--eval-datasets', 'DUTS-TE',
            *flags,
        ]
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

        print(f"\n{'='*60}")
        print(f"[PROMPT ANALYSIS] {variant}: {' '.join(cmd)}")
        print(f"{'='*60}")

        log_path = out_dir / f'{variant}.log'
        with open(log_path, 'w') as log_f:
            proc = subprocess.run(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
        
        if proc.returncode != 0:
            print(f"[ERROR] {variant} failed (exit {proc.returncode}). See {log_path}")
            continue

        try:
            metrics = read_metrics(run_dir)
            rows.append({'variant': variant, **metrics, 'flags': ' '.join(flags)})
            print(f"  Sm={metrics['s_measure']:.4f} maxF={metrics['max_f']:.4f} "
                  f"meanE={metrics['mean_e']:.4f} wF={metrics['weighted_f']:.4f} MAE={metrics['mae']:.4f}")
        except Exception as e:
            print(f"[ERROR] Could not read metrics for {variant}: {e}")

    if rows:
        write_results_md(rows, out_dir / 'prompt_analysis_results.md')
        print(f"\nResults saved to {out_dir / 'prompt_analysis_results.md'}")


if __name__ == '__main__':
    main()
