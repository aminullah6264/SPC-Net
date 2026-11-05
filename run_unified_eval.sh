#!/bin/bash
# ============================================================
# Unified re-evaluation of EXISTING best checkpoints (camera-ready, 2026-11)
# Test-only inference on 6 benchmarks; saves continuous saliency maps that
# score_unified.py scores with PySODMetrics.
#
# Usage: bash run_unified_eval.sh streamA|streamB|launch|dry
# Two streams share GPUs 2/3 with training; threads are capped to limit
# CPU contention. Variants with an existing unified_eval/<name>/DONE are skipped.
# ============================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
CONDA_SH="/home/av543f/anaconda3/etc/profile.d/conda.sh"
R="/mnt/data/projects/CAD_Adaptation/CLIPSAM_new_results"
SRC="$R/outputs"
DST="$R/unified_eval"
EVAL="DUTS-TE,DUT-O,ECSSD,HKU-IS,PASCAL-S-full,SIP"

BASELINE="--no-prompt-bank --no-adaptive-fallback --no-bidirectional-sam --simple-decoder --plain-bce-loss --pp-loss-weight 0 --ds-weight 0"
PLUS_MP="--no-adaptive-fallback --no-bidirectional-sam --simple-decoder --plain-bce-loss --pp-loss-weight 0 --ds-weight 0"
PLUS_AF="--no-bidirectional-sam --simple-decoder --plain-bce-loss --pp-loss-weight 0 --ds-weight 0"
PLUS_BISAM="--simple-decoder --plain-bce-loss --pp-loss-weight 0 --ds-weight 0"
PLUS_DEC="--pp-loss-weight 0 --ds-weight 0.3"
FULL="--pp-loss-weight 0.3 --ds-weight 0.3"

# name | checkpoint dir (under outputs/) | stream | flags
# K20 omitted: the bank has 10 prompts, so it duplicated K=10.
LIST=(
  "plus_af|ablation/plus_af|A|$PLUS_AF"
  "baseline|ablation/baseline|A|$BASELINE"
  "plus_mp|ablation/plus_mp|A|$PLUS_MP"
  "plus_bisam|ablation/plus_bisam|A|$PLUS_BISAM"
  "plus_dec|ablation/plus_dec|A|$PLUS_DEC"
  "plus_lpt|ablation/plus_lpt|A|$FULL"
  "full_model|full_model|A|$FULL"
  "mp_attention|prompt_modes/mp_attention|A|$PLUS_MP --prompt-agg attention"
  "baseline_attention|prompt_modes/baseline_attention|A|$PLUS_MP --prompt-agg attention"
  "attention|prompt_modes/attention|A|$FULL --prompt-agg attention"
  "full_attention|prompt_modes/full_attention|B|$FULL --prompt-agg attention"
  "learnable|prompt_modes/learnable|B|$FULL --prompt-agg learnable"
  "gated|prompt_modes/gated|B|$FULL --prompt-agg gated"
  "K1|prompt_analysis/K1|B|$FULL --prompt-bank-size 1"
  "K5|prompt_analysis/K5|B|$FULL --prompt-bank-size 5"
  "gamma0|prompt_analysis/gamma0|B|--pp-loss-weight 0 --ds-weight 0.3"
  "gamma1|prompt_analysis/gamma1|B|--pp-loss-weight 1.0 --ds-weight 0.3"
  "agg_mean|prompt_analysis/agg_mean|B|$FULL --prompt-agg mean"
  "random_bank|prompt_analysis/random_bank|B|$FULL --random-prompts"
  "diverse_bank|prompt_analysis/diverse_bank|B|$FULL --diverse-prompts"
)

run_stream() {
  local s="$1" gpu="$2"
  source "$CONDA_SH"; conda activate open_clip
  export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
  for e in "${LIST[@]}"; do
    IFS='|' read -r name ck st flags <<< "$e"
    [ "$st" = "$s" ] || continue
    out="$DST/$name"
    [ -f "$out/DONE" ] && { echo "SKIP $name"; continue; }
    mkdir -p "$out"
    echo "EVAL $name $(date)"
    if python -u clipsam.py --clip-model ViT-L-14 --test-only \
        --resume "$SRC/$ck/escnet_best.pth" --output-dir "$out" \
        --eval-datasets "$EVAL" --no-calibrate --threshold 0.5 --skip-default-eval \
        $flags > "$out/eval.log" 2>&1; then
      echo "$flags" > "$out/flags.txt"; date > "$out/DONE"
    else
      echo "FAILED $name (see $out/eval.log)"
    fi
  done
  echo "stream $s finished $(date)"
}

case "${1:-}" in
  streamA) run_stream A 2 2>&1 | tee -a "$DST/streamA.log" ;;
  streamB) run_stream B 3 2>&1 | tee -a "$DST/streamB.log" ;;
  launch)
    mkdir -p "$DST"
    screen -dmS ue_A bash -c "cd $SCRIPT_DIR && nice -n 10 bash run_unified_eval.sh streamA"
    screen -dmS ue_B bash -c "cd $SCRIPT_DIR && nice -n 10 bash run_unified_eval.sh streamB"
    echo "launched ue_A (GPU2) and ue_B (GPU3)" ;;
  dry) for e in "${LIST[@]}"; do IFS='|' read -r n c s f <<< "$e"; echo "$s $n <- $c : $f"; done ;;
  *) echo "Usage: bash run_unified_eval.sh {streamA|streamB|launch|dry}"; exit 1 ;;
esac
