#!/bin/bash
# Train one configuration with one seed, then evaluate on all six benchmarks.
#
# Usage: bash scripts/train.sh <config> <seed> [gpu] [extra clipsam.py flags]
#   e.g. bash scripts/train.sh ours 42 0
# Output: outputs/<config>_s<seed>/ (checkpoint, saliency maps, per-image metrics)
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/configs.sh

CONFIG="$1"; SEED="$2"; GPU="${3:-0}"; shift $(( $# < 3 ? $# : 3 ))
[ -n "${CFG[$CONFIG]+x}" ] || { echo "Unknown config '$CONFIG'. Options: ${!CFG[*]}"; exit 1; }
[ -n "$SEED" ] || { echo "Usage: bash scripts/train.sh <config> <seed> [gpu]"; exit 1; }

OUT="outputs/${CONFIG}_s${SEED}"
mkdir -p "$OUT" logs
echo "flags: ${CFG[$CONFIG]} --seed ${SEED} $*" > "$OUT/run_flags.txt"
echo "[train] ${CONFIG} seed ${SEED} on GPU ${GPU} -> ${OUT}"
CUDA_VISIBLE_DEVICES="$GPU" python -u clipsam.py ${COMMON} ${CFG[$CONFIG]} \
  --seed "$SEED" --output-dir "$OUT" "$@" 2>&1 | tee "logs/${CONFIG}_s${SEED}.log"
