#!/usr/bin/env python3
"""Evaluate saved saliency maps using py-sod-metrics (PySODEvalToolkit compatible).

Usage:
    python run_eval_toolbox.py --pred-root outputs/full_model/saliency_maps \
                               --datasets DUTS-TE,DUT-O,ECSSD,HKU-IS,PASCAL-S,SIP
"""
import argparse
import json
import os
import numpy as np
from pathlib import Path
from PIL import Image

# Dataset GT paths
SOD_DATASET_ROOT = '/mnt/data/projects/CAD_Adaptation/SOD_Datasets'
GT_REGISTRY = {
    'DUTS-TE': os.path.join(SOD_DATASET_ROOT, 'DUTS_Dataset/DUTS-TE/DUTS-TE-Mask'),
    'DUT-O': os.path.join(SOD_DATASET_ROOT, 'DUT-O/pixelwiseGT-new-PNG'),
    'ECSSD': os.path.join(SOD_DATASET_ROOT, 'ECSSD/ground_truth_mask'),
    'HKU-IS': None,  # NPY — handled separately
    'PASCAL-S': None,  # NPY — handled separately
    'SIP': os.path.join(SOD_DATASET_ROOT, 'SIP_Dataset/Test/Masks'),
}

NPY_GT = {
    'HKU-IS': os.path.join(SOD_DATASET_ROOT, 'HKU-IS/gt.npy'),
    'PASCAL-S': os.path.join(SOD_DATASET_ROOT, 'pascals/pascals_test_masks.npy'),
}


def compute_metrics_for_pair(pred_prob, gt_bin):
    """Compute standard SOD metrics for a single prediction-GT pair."""
    from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
    
    mae_calc = MAE()
    sm_calc = Smeasure()
    em_calc = Emeasure()
    fm_calc = Fmeasure()
    wfm_calc = WeightedFmeasure()
    
    mae_calc.step(pred=pred_prob, gt=gt_bin)
    sm_calc.step(pred=pred_prob, gt=gt_bin)
    em_calc.step(pred=pred_prob, gt=gt_bin)
    fm_calc.step(pred=pred_prob, gt=gt_bin)
    wfm_calc.step(pred=pred_prob, gt=gt_bin)
    
    return mae_calc, sm_calc, em_calc, fm_calc, wfm_calc


def evaluate_directory_dataset(pred_dir, gt_dir, dataset_name):
    """Evaluate a directory-based dataset."""
    from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
    
    mae_calc = MAE()
    sm_calc = Smeasure()
    em_calc = Emeasure()
    fm_calc = Fmeasure()
    wfm_calc = WeightedFmeasure()
    
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.png')])
    print(f"  [{dataset_name}] Found {len(pred_files)} predictions")
    
    matched = 0
    for fn in pred_files:
        # Try to find matching GT
        stem = os.path.splitext(fn)[0]
        gt_path = None
        for ext in ['.png', '.jpg', '.bmp']:
            candidate = os.path.join(gt_dir, stem + ext)
            if os.path.exists(candidate):
                gt_path = candidate
                break
        if gt_path is None:
            continue
        
        pred = np.array(Image.open(os.path.join(pred_dir, fn)).convert('L'), dtype=np.float64) / 255.0
        gt = np.array(Image.open(gt_path).convert('L'), dtype=np.float64) / 255.0
        gt_bin = (gt > 0.5).astype(np.uint8) * 255  # py-sod-metrics expects 0/255
        pred_u8 = (pred * 255).astype(np.uint8)
        
        mae_calc.step(pred=pred_u8, gt=gt_bin)
        sm_calc.step(pred=pred_u8, gt=gt_bin)
        em_calc.step(pred=pred_u8, gt=gt_bin)
        fm_calc.step(pred=pred_u8, gt=gt_bin)
        wfm_calc.step(pred=pred_u8, gt=gt_bin)
        matched += 1
    
    print(f"  [{dataset_name}] Matched {matched}/{len(pred_files)} predictions to GTs")
    
    results = {
        'Sm': sm_calc.get_results()['sm'],
        'maxFm': fm_calc.get_results()['fm']['curve'].max(),
        'meanEm': em_calc.get_results()['em']['curve'].mean(),
        'wFm': wfm_calc.get_results()['wfm'],
        'MAE': mae_calc.get_results()['mae'],
        'adpFm': fm_calc.get_results()['fm']['adp'],
        'adpEm': em_calc.get_results()['em']['adp'],
        'n_samples': matched,
    }
    return results


def evaluate_npy_dataset(pred_dir, gt_npy_path, dataset_name):
    """Evaluate an NPY-based dataset."""
    from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
    
    mae_calc = MAE()
    sm_calc = Smeasure()
    em_calc = Emeasure()
    fm_calc = Fmeasure()
    wfm_calc = WeightedFmeasure()
    
    gt_data = np.load(gt_npy_path)
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.png')])
    print(f"  [{dataset_name}] Found {len(pred_files)} predictions, {len(gt_data)} GT masks")
    
    matched = 0
    for i, fn in enumerate(pred_files):
        if i >= len(gt_data):
            break
        
        pred = np.array(Image.open(os.path.join(pred_dir, fn)).convert('L'), dtype=np.uint8)
        
        gt_mask = gt_data[i]
        if gt_mask.ndim == 1:
            # Flattened — infer square size
            side = int(np.sqrt(gt_mask.shape[0]))
            gt_mask = gt_mask.reshape(side, side)
        if gt_mask.max() <= 1.0:
            gt_mask = (gt_mask * 255).astype(np.uint8)
        else:
            gt_mask = gt_mask.astype(np.uint8)
        gt_bin = ((gt_mask > 127).astype(np.uint8)) * 255
        
        # Resize pred to match GT if needed
        if pred.shape != gt_bin.shape:
            pred = np.array(Image.fromarray(pred).resize(
                (gt_bin.shape[1], gt_bin.shape[0]), Image.BILINEAR), dtype=np.uint8)
        
        mae_calc.step(pred=pred, gt=gt_bin)
        sm_calc.step(pred=pred, gt=gt_bin)
        em_calc.step(pred=pred, gt=gt_bin)
        fm_calc.step(pred=pred, gt=gt_bin)
        wfm_calc.step(pred=pred, gt=gt_bin)
        matched += 1
    
    print(f"  [{dataset_name}] Evaluated {matched} samples")
    
    results = {
        'Sm': sm_calc.get_results()['sm'],
        'maxFm': fm_calc.get_results()['fm']['curve'].max(),
        'meanEm': em_calc.get_results()['em']['curve'].mean(),
        'wFm': wfm_calc.get_results()['wfm'],
        'MAE': mae_calc.get_results()['mae'],
        'adpFm': fm_calc.get_results()['fm']['adp'],
        'adpEm': em_calc.get_results()['em']['adp'],
        'n_samples': matched,
    }
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate saliency maps with PySODEvalToolkit metrics')
    parser.add_argument('--pred-root', required=True, help='Root dir containing dataset subdirs with predicted saliency maps')
    parser.add_argument('--datasets', default='DUTS-TE,DUT-O,ECSSD,HKU-IS,PASCAL-S,SIP',
                       help='Comma-separated list of datasets')
    parser.add_argument('--output', default=None, help='Output JSON path (default: <pred-root>/eval_results.json)')
    args = parser.parse_args()
    
    datasets = [d.strip() for d in args.datasets.split(',')]
    output_path = args.output or os.path.join(args.pred_root, 'eval_results.json')
    
    all_results = {}
    for ds in datasets:
        pred_dir = os.path.join(args.pred_root, ds.replace('-', '_'))
        if not os.path.isdir(pred_dir):
            # Try alternative naming
            pred_dir = os.path.join(args.pred_root, ds)
        if not os.path.isdir(pred_dir):
            pred_dir = os.path.join(args.pred_root, ds.lower().replace('-', '_'))
        if not os.path.isdir(pred_dir):
            print(f"  [SKIP] {ds}: prediction dir not found at {pred_dir}")
            continue
        
        print(f"\nEvaluating {ds}...")
        if ds in NPY_GT and GT_REGISTRY.get(ds) is None:
            results = evaluate_npy_dataset(pred_dir, NPY_GT[ds], ds)
        else:
            gt_dir = GT_REGISTRY[ds]
            if not os.path.isdir(gt_dir):
                print(f"  [SKIP] {ds}: GT dir not found at {gt_dir}")
                continue
            results = evaluate_directory_dataset(pred_dir, gt_dir, ds)
        
        all_results[ds] = results
        print(f"  Sm={results['Sm']:.4f}  maxFm={results['maxFm']:.4f}  "
              f"meanEm={results['meanEm']:.4f}  wFm={results['wFm']:.4f}  MAE={results['MAE']:.4f}")
    
    # Save results
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Convert numpy types to float for JSON serialization
    json_results = {}
    for ds, metrics in all_results.items():
        json_results[ds] = {k: float(v) for k, v in metrics.items()}
    
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    # Print summary table
    print("\n" + "="*80)
    print(f"{'Dataset':<12} {'Sm':>8} {'maxFm':>8} {'meanEm':>8} {'wFm':>8} {'MAE':>8}")
    print("-"*80)
    for ds, r in all_results.items():
        print(f"{ds:<12} {r['Sm']:>8.4f} {r['maxFm']:>8.4f} {r['meanEm']:>8.4f} {r['wFm']:>8.4f} {r['MAE']:>8.4f}")
    print("="*80)


if __name__ == '__main__':
    main()
