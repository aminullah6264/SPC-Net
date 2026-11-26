#!/bin/bash
# Reproduce the SPC-Net results of the paper.
#
# Usage: bash scripts/reproduce_paper.sh <main|table3|table4|all> [gpu ...]
#   main    SPC-Net, seed 42 (Tables 1 and 2)
#   table3  prompt-side study (Table 3)
#   table4  architecture and training study (Table 4)
#   all     everything (35 runs, ~5 h each on one A100)
# With several GPUs (e.g. "0 1 2 3"), runs are distributed round-robin, one per GPU
# at a time. Finished runs (results.yaml present) are skipped, so the script can be
# restarted. Scoring and the summary table are produced at the end.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/configs.sh

WHAT="${1:-main}"; shift || true
GPUS=("${@:-0}")
case "$WHAT" in
  main)   LIST="ours"; ONLY_SEED=42 ;;
  table3) LIST="$TABLE3" ;;
  table4) LIST="ours $TABLE4" ;;
  all)    LIST="$TABLE3 $TABLE4" ;;
  *) echo "Usage: bash scripts/reproduce_paper.sh <main|table3|table4|all> [gpu ...]"; exit 1 ;;
esac

JOBS=()
for c in $LIST; do
  for s in ${ONLY_SEED:-${SEEDS[$c]}}; do
    [ -f "outputs/${c}_s${s}/results.yaml" ] && { echo "[skip] ${c}_s${s}"; continue; }
    JOBS+=("$c $s")
  done
done
echo "[reproduce] ${#JOBS[@]} runs on GPU(s): ${GPUS[*]}"

worker() {
  local slot=$1 gpu=${GPUS[$1]} i=0
  for job in "${JOBS[@]}"; do
    if [ $(( i % ${#GPUS[@]} )) -eq "$slot" ]; then
      set -- $job
      bash scripts/train.sh "$1" "$2" "$gpu" > /dev/null || echo "[failed] $1_s$2 (see logs/)"
    fi
    i=$((i + 1))
  done
}
for k in "${!GPUS[@]}"; do worker "$k" & done
wait

python tools/score_metrics.py --root outputs
python tools/compute_f05.py --roots outputs
python tools/summarize.py --root outputs --metric Sm --csv outputs/summary_Sm.csv
