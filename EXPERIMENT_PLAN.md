# SPC-Net Revision — Experiment Plan

## Overview
Address WACV 2027 reviewer criticisms by upgrading backbone, adopting standard SOD evaluation, and running comprehensive experiments.

## Environment
- **Conda env**: `open_clip`
- **GPUs available**: 1, 2, 3 (A100 80GB each; GPU 0 in use by teammate)
- **Estimated wall-clock**: ~7 hours

## Key Changes from WACV Submission
1. **Backbone upgrade**: CLIP ViT-B/16 → ViT-L/14 (keep SAM ViT-B)
2. **Drop inference tricks**: No TTA, no connected-component post-processing, no threshold calibration
3. **Standard SOD evaluation**: Max-Fβ (255-threshold sweep), Mean E-measure, S-measure, Weighted-F, MAE
4. **Train on DUTS-TR**: Standard SOD training set (10,553 images)
5. **Evaluate on 6 benchmarks**: DUTS-TE, DUT-O, HKU-IS, ECSSD, PASCAL-S, SIP
6. **Full ablation + prompt analysis**: Address every reviewer's request

## Dataset Paths
| Dataset | Images | Masks | Count |
|---------|--------|-------|-------|
| DUTS-TR (train) | `/mnt/data/projects/CAD_Adaptation/SOD_Datasets/DUTS_Dataset/DUTS-TR/DUTS-TR-Image` | `/mnt/data/projects/CAD_Adaptation/SOD_Datasets/DUTS_Dataset/DUTS-TR/DUTS-TR-Mask` | 10,553 |
| DUTS-TE (eval) | `/mnt/data/projects/CAD_Adaptation/SOD_Datasets/DUTS_Dataset/DUTS-TE/DUTS-TE-Image` | `/mnt/data/projects/CAD_Adaptation/SOD_Datasets/DUTS_Dataset/DUTS-TE/DUTS-TE-Mask` | 5,019 |
| DUT-O (eval) | `/mnt/data/projects/CAD_Adaptation/SOD_Datasets/DUT-O/DUT-OMRON-image` | `/mnt/data/projects/CAD_Adaptation/SOD_Datasets/DUT-O/pixelwiseGT-new-PNG` | 5,168 |
| ECSSD (eval) | `/mnt/data/projects/CAD_Adaptation/SOD_Datasets/ECSSD/images` | `/mnt/data/projects/CAD_Adaptation/SOD_Datasets/ECSSD/ground_truth_mask` | 1,000 |
| HKU-IS (eval) | NPY: `/mnt/data/projects/CAD_Adaptation/SOD_Datasets/HKU-IS/images.npy` | NPY: `/mnt/data/projects/CAD_Adaptation/SOD_Datasets/HKU-IS/gt.npy` | 4,447 |
| PASCAL-S (eval) | NPY: `/mnt/data/projects/CAD_Adaptation/SOD_Datasets/pascals/pascals_test_images.npy` | NPY: `/mnt/data/projects/CAD_Adaptation/SOD_Datasets/pascals/pascals_test_masks.npy` | ~256 |
| SIP (eval) | `/mnt/data/projects/CAD_Adaptation/SOD_Datasets/SIP_Dataset/Test/Images` | `/mnt/data/projects/CAD_Adaptation/SOD_Datasets/SIP_Dataset/Test/Masks` | 129 |

## ViT-L/14 Architecture Changes
| Parameter | ViT-B/16 (old) | ViT-L/14 (new) |
|-----------|---------------|----------------|
| Hidden dim | 768 | 1024 |
| Projection dim | 512 | 768 |
| Transformer blocks | 12 | 24 |
| Hook indices | [2, 5, 8, 11] | [5, 11, 17, 23] |
| Patch size | 16 | 14 |
| Grid size (@ 336px) | 21×21 | 24×24 |
| vis_proj_w shape | (768, 512) | (1024, 768) |

## GPU Allocation
| GPU | Phase 1 (Ablation/Full Model) | Phase 2 (Prompt Analysis) |
|-----|-------------------------------|---------------------------|
| 1 | Full model training (60 epochs) | K=1, K=5, K=20 |
| 2 | Ablation: baseline, +mp, +af | mean-aggregation, random-bank |
| 3 | Ablation: +bisam, +dec, +lpt | γ=0, γ=1.0 |

## Experiment Matrix

### Ablation Study (6 runs on DUTS-TE)
| Variant | Flags |
|---------|-------|
| baseline | `--no-prompt-bank --no-adaptive-fallback --no-bidirectional-sam --simple-decoder --plain-bce-loss --pp-loss-weight 0 --ds-weight 0` |
| +mp | baseline minus `--no-prompt-bank` |
| +af | +mp minus `--no-adaptive-fallback` |
| +bisam | +af minus `--no-bidirectional-sam` |
| +dec | +bisam minus `--simple-decoder --plain-bce-loss`, set `--ds-weight 0.3` |
| +lpt (full) | +dec plus `--pp-loss-weight 0.3` |

### Prompt Bank Analysis (7 new runs on DUTS-TE)
| Factor | Setting | How |
|--------|---------|-----|
| Bank size K=1 | Single prompt | `--prompt-bank-size 1` |
| Bank size K=5 | 5 prompts | `--prompt-bank-size 5` |
| Bank size K=10 | Default (= full model, skip) | — |
| Bank size K=20 | 20 prompts | `--prompt-bank-size 20` |
| Aggregation: mean | Mean instead of max | `--prompt-agg mean` |
| Wording: random | Random unrelated sentences | `--random-prompts` |
| γ=0 | No point-loss | `--pp-loss-weight 0` |
| γ=0.3 | Default (= full model, skip) | — |
| γ=1.0 | Heavy point-loss | `--pp-loss-weight 1.0` |

## Running Experiments

```bash
# Launch all experiments (dry run first)
bash launch_all.sh --dry-run

# Launch for real
bash launch_all.sh

# Monitor progress
bash monitor.sh

# After completion, collect results
python collect_results.py
```

## Output Structure
```
outputs/
├── full_model/
│   ├── escnet_best.pth
│   ├── escnet_final.pth
│   ├── training_curves.png
│   ├── training_diagnostics.csv
│   ├── results_summary.md
│   └── saliency_maps/
│       ├── DUTS-TE/
│       ├── DUT-O/
│       ├── ECSSD/
│       ├── HKU-IS/
│       ├── PASCAL-S/
│       └── SIP/
├── ablation/
│   ├── baseline/
│   ├── plus_mp/
│   ├── plus_af/
│   ├── plus_bisam/
│   ├── plus_dec/
│   ├── plus_lpt/
│   └── ablation_results.md
├── prompt_analysis/
│   ├── K1/ K5/ K20/
│   ├── mean_agg/
│   ├── random_bank/
│   ├── gamma0/ gamma1/
│   └── prompt_analysis_results.md
├── logs/
│   ├── gpu1_*.log
│   ├── gpu2_*.log
│   └── gpu3_*.log
└── RESULTS.md  (unified results with LaTeX tables)
```
