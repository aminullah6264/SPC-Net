# CLIPSAM — SIP Dataset Branch

This branch is configured for CLIP+SAM segmentation experiments on the **SIP dataset** and reports the SIP / salient-object segmentation metrics required for experiment tables:

| Metric | Direction | Output key |
| --- | --- | --- |
| E-Measure | ↑ higher is better | `e_measure` |
| S-Measure | ↑ higher is better | `s_measure` |
| Weighted-F | ↑ higher is better | `weighted_f` |
| F-Measure | ↑ higher is better | `f_measure` |
| MAE | ↓ lower is better | `mae` |

Dice and IoU are still reported for debugging and comparison, but the SIP table should use **E-Measure**, **S-Measure**, **Weighted-F**, **F-Measure**, and **MAE**.

## Repository files

- `clipsam.py` — main training, validation, test, visualization, and SIP evaluation pipeline.
- `CLIPUNet.py` — alternative CLIP-UNet implementation retained from the original repository.
- `setup_dataset.py` — dataset setup helper.
- `requirements.txt` — Python dependencies.

## Expected SIP dataset layout

By default, `clipsam.py` expects the dataset at:

```text
/mnt/data/users/av543f/datasets/SIP_Dataset/
├── Train/
│   ├── Images/
│   └── Masks/
├── Test/
│   ├── Images/
│   └── Masks/
└── Test1/
    ├── Images/
    └── Masks/
```

The default mapping in `clipsam.py` is:

- training images: `Train/Images`
- training masks: `Train/Masks`
- validation images: `Test/Images`
- validation masks: `Test/Masks`
- test images: `Test1/Images`
- test masks: `Test1/Masks`
- outputs: `outputs_clipSAM/`

You can override any path from the command line with `--img-path`, `--mask-path`, `--val-img`, `--val-mask`, `--test-img`, `--test-mask`, and `--output-dir`.

## Installation

Create an environment with Python and install the dependencies:

```bash
pip install -r requirements.txt
```

`segment-anything` is installed from GitHub by `requirements.txt`. The first run of `clipsam.py` downloads the SAM ViT-B checkpoint into the output directory if it is not already present.

> Note: OpenCV may require system libraries such as `libGL.so.1` in minimal Linux containers.

## Training on SIP

Run training with the default SIP paths:

```bash
python clipsam.py \
  --epochs 60 \
  --batch-size 8 \
  --num-workers 4 \
  --output-dir outputs_clipSAM
```

Common options:

```bash
python clipsam.py \
  --img-path /path/to/SIP_Dataset/Train/Images \
  --mask-path /path/to/SIP_Dataset/Train/Masks \
  --val-img /path/to/SIP_Dataset/Test/Images \
  --val-mask /path/to/SIP_Dataset/Test/Masks \
  --test-img /path/to/SIP_Dataset/Test1/Images \
  --test-mask /path/to/SIP_Dataset/Test1/Masks \
  --output-dir outputs_clipSAM \
  --epochs 60 \
  --batch-size 8
```

During evaluation, the pipeline calibrates a threshold on the validation split and applies that calibrated threshold to validation and test reporting.

## Test-only evaluation

To evaluate a trained checkpoint without retraining:

```bash
python clipsam.py \
  --test-only \
  --resume outputs_clipSAM/escnet_best.pth \
  --output-dir outputs_clipSAM
```

If `--resume` is omitted, test-only mode looks for `escnet_best.pth` inside `--output-dir`.

## SIP evaluation outputs

After validation/test evaluation, the output directory contains:

- `results.yaml` — training configuration, threshold information, and metric summaries.
- `sip_results.csv` — compact table with validation/test aggregate SIP metrics:
  - `e_measure`
  - `s_measure`
  - `weighted_f`
  - `f_measure`
  - `mae`
  - plus Dice, IoU, threshold, and sample count.
- `validation_sip_metrics.csv` — per-image validation metrics.
- `test_sip_metrics.csv` — per-image test metrics when test masks are available.
- `test_outputs/` — side-by-side visualizations for test images.

For the SIP experiment table, report the `test` row in `sip_results.csv`:

```text
E-Measure ↑ | S-Measure ↑ | Weighted-F ↑ | F-Measure ↑ | MAE ↓
```

## Threshold sweep

To additionally sweep test thresholds and save Dice/IoU threshold diagnostics:

```bash
python clipsam.py \
  --test-only \
  --resume outputs_clipSAM/escnet_best.pth \
  --output-dir outputs_clipSAM \
  --threshold-sweep
```

This writes `threshold_sweep_test.csv`. The SIP metrics in `sip_results.csv` still use the validation-calibrated threshold, not a test-tuned threshold.
