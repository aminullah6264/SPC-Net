#!/bin/bash
# Evaluate a trained checkpoint on all six benchmarks (no training).
#
# Usage: bash scripts/evaluate.sh <config> <checkpoint.pth> <output_dir> [gpu]
#   e.g. bash scripts/evaluate.sh ours outputs/ours_s42/escnet_best.pth outputs/eval_ours 0
# The config must match the one used for training (it sets the architecture flags).
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/configs.sh

CONFIG="$1"; CKPT="$2"; OUT="$3"; GPU="${4:-0}"
[ -n "${CFG[$CONFIG]+x}" ] && [ -f "$CKPT" ] && [ -n "$OUT" ] || {
  echo "Usage: bash scripts/evaluate.sh <config> <checkpoint.pth> <output_dir> [gpu]"; exit 1; }

mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES="$GPU" python -u clipsam.py --clip-model ViT-L-14 --test-only \
  --resume "$CKPT" --output-dir "$OUT" --eval-datasets "$EVAL_DATASETS" \
  --no-calibrate --threshold 0.5 --skip-default-eval ${CFG[$CONFIG]}
python tools/score_metrics.py --root "$(dirname "$OUT")" --variants "$(basename "$OUT")" --force
python tools/compute_f05.py --roots "$(dirname "$OUT")"
