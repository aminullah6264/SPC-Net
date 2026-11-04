#!/usr/bin/env python3
"""Collect all experiment results into unified RESULTS.md and LaTeX tables.

Usage:
    python collect_results.py [--output-dir outputs]
"""
import argparse
import json
import os
from pathlib import Path


def load_json_metrics(json_path):
    """Load metrics from eval_results.json."""
    if not os.path.exists(json_path):
        return None
    with open(json_path) as f:
        return json.load(f)


def format_table_row(values, bold_indices=None):
    """Format a row with optional bold for best values."""
    cells = []
    for i, v in enumerate(values):
        if isinstance(v, float):
            text = f"{v:.4f}"
            if bold_indices and i in bold_indices:
                text = f"**{text}**"
        else:
            text = str(v)
        cells.append(text)
    return '| ' + ' | '.join(cells) + ' |'


def generate_main_table(output_dir):
    """Generate main comparison table from full model results."""
    json_path = Path(output_dir) / 'full_model' / 'saliency_maps' / 'eval_results.json'
    metrics = load_json_metrics(json_path)
    if not metrics:
        return "*Full model results not yet available.*\n"
    
    lines = [
        '## Main Results: SPC-Net (ViT-L/14) on Standard SOD Benchmarks\n',
        'Training: DUTS-TR (10,553 images), 60 epochs, batch 8, seed 42.',
        'Evaluation: Standard protocol — no TTA, no post-processing, no threshold calibration.\n',
        '| Dataset | Sm ↑ | maxFβ ↑ | meanEm ↑ | wFβ ↑ | MAE ↓ | N |',
        '|---------|------|---------|----------|-------|-------|---|',
    ]
    for ds in ['DUTS-TE', 'DUT-O', 'ECSSD', 'HKU-IS', 'PASCAL-S', 'SIP']:
        if ds in metrics:
            m = metrics[ds]
            lines.append(
                f"| {ds} | {m['Sm']:.4f} | {m['maxFm']:.4f} | {m['meanEm']:.4f} | "
                f"{m['wFm']:.4f} | {m['MAE']:.4f} | {m['n_samples']} |")
        else:
            lines.append(f"| {ds} | — | — | — | — | — | — |")
    lines.append('')
    return '\n'.join(lines)


def generate_ablation_table(output_dir):
    """Generate ablation table."""
    ablation_dir = Path(output_dir) / 'ablation'
    md_path = ablation_dir / 'ablation_results.md'
    if md_path.exists():
        return f"## Ablation Study (DUTS-TE)\n\n" + md_path.read_text()
    return "*Ablation results not yet available.*\n"


def generate_prompt_analysis_table(output_dir):
    """Generate prompt analysis table."""
    pa_dir = Path(output_dir) / 'prompt_analysis'
    md_path = pa_dir / 'prompt_analysis_results.md'
    if md_path.exists():
        return md_path.read_text()
    return "*Prompt analysis results not yet available.*\n"


def generate_latex_tables(output_dir):
    """Generate LaTeX table bodies."""
    json_path = Path(output_dir) / 'full_model' / 'saliency_maps' / 'eval_results.json'
    metrics = load_json_metrics(json_path)
    if not metrics:
        return "% No results available yet\n"
    
    lines = ['% Main results table body (paste into \\begin{tabular})',
             '% Columns: Dataset & Sm & maxFb & meanEm & wFb & MAE \\\\']
    for ds in ['DUTS-TE', 'DUT-O', 'ECSSD', 'HKU-IS', 'PASCAL-S', 'SIP']:
        if ds in metrics:
            m = metrics[ds]
            lines.append(
                f"{ds} & {m['Sm']:.4f} & {m['maxFm']:.4f} & "
                f"{m['meanEm']:.4f} & {m['wFm']:.4f} & {m['MAE']:.4f} \\\\")
    lines.append('')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Collect all SPC-Net experiment results')
    parser.add_argument('--output-dir', default='outputs', help='Root output directory')
    args = parser.parse_args()
    
    output_dir = args.output_dir
    results_path = os.path.join(output_dir, 'RESULTS.md')
    latex_path = os.path.join(output_dir, 'results_tables.tex')
    
    sections = [
        '# SPC-Net Experiment Results\n',
        f'Generated from experiments in `{output_dir}/`\n',
        '---\n',
        generate_main_table(output_dir),
        '---\n',
        generate_ablation_table(output_dir),
        '---\n',
        generate_prompt_analysis_table(output_dir),
        '---\n',
        '## Experimental Setup\n',
        '- **Backbone**: CLIP ViT-L/14 + SAM ViT-B',
        '- **Training**: DUTS-TR (10,553 images), 60 epochs, batch 8',
        '- **Optimizer**: AdamW, dual LR (2e-6 pretrained, 2e-5 new modules)',
        '- **Loss**: 0.6×Tversky + 0.4×BoundaryFocalBCE + 0.1×MSE + 0.3×PointCls',
        '- **Evaluation**: Standard SOD protocol (no TTA, no post-processing)',
        '- **Metrics**: Sm, maxFβ (255 thresholds), meanEm, wFβ, MAE',
        '- **Seed**: 42',
        '',
    ]
    
    os.makedirs(output_dir, exist_ok=True)
    with open(results_path, 'w') as f:
        f.write('\n'.join(sections))
    print(f"Results written to {results_path}")
    
    # LaTeX tables
    latex_content = generate_latex_tables(output_dir)
    with open(latex_path, 'w') as f:
        f.write(latex_content)
    print(f"LaTeX tables written to {latex_path}")


if __name__ == '__main__':
    main()
