#!/bin/bash
# ============================================================
# SPC-Net camera-ready: point loss removed from the method (2026-11)
# Real CLIP text encoder, fine-tuned encoders, NO point loss. 8 slots (2 per GPU).
# Existing and reused: baseline, +MP, +AF (3 seeds), plus_dec_s42 (= dec_max_s42).
# Usage: bash run_nolpt.sh {launch_all|slot0..slot7|dry}
# ============================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
CONDA_SH="/home/av543f/anaconda3/etc/profile.d/conda.sh"
ROOT="/mnt/data/projects/CAD_Adaptation/CLIPSAM_new_results/outputs_textfix"
LOGS="/mnt/data/projects/CAD_Adaptation/CLIPSAM_new_results/logs_textfix"
EVAL="DUTS-TE,DUT-O,ECSSD,HKU-IS,PASCAL-S-full,SIP,SIP-full"
COMMON="--clip-model ViT-L-14 --epochs 30 --batch-size 8 --num-workers 4 --mse-lambda 0.1 --save-best-only --eval-datasets $EVAL --no-calibrate --threshold 0.5"

BASE_AF="--no-prompt-bank --no-bidirectional-sam --simple-decoder --plain-bce-loss --pp-loss-weight 0 --ds-weight 0"
PLUS_AF="--no-bidirectional-sam --simple-decoder --plain-bce-loss --pp-loss-weight 0 --ds-weight 0"
DEC="--pp-loss-weight 0 --ds-weight 0.3"   # correlation-aware decoder, no point loss

# name | slot (gpu = slot % 4) | flags
QUEUE=(
  "base_af_s42|0|$BASE_AF --seed 42"
  "af_K1_s42|1|$PLUS_AF --prompt-bank-size 1 --seed 42"
  "af_K5_s42|2|$PLUS_AF --prompt-bank-size 5 --seed 42"
  "af_random_s42|3|$PLUS_AF --random-prompts --seed 42"
  "af_diverse_s42|4|$PLUS_AF --diverse-prompts --seed 42"
  "dec_max_s1|5|$DEC --seed 1"
  "dec_max_s2|6|$DEC --seed 2"
  "dec_max_noresid_s42|7|$DEC --no-prompt-residual --seed 42"
  "dec_max_noresid_s1|0|$DEC --no-prompt-residual --seed 1"
  "dec_max_noresid_s2|1|$DEC --no-prompt-residual --seed 2"
  "dec_attention_s42|2|$DEC --prompt-agg attention --seed 42"
  "dec_attention_s1|3|$DEC --prompt-agg attention --seed 1"
  "dec_attention_s2|4|$DEC --prompt-agg attention --seed 2"
  "dec_attention_noresid_s42|5|$DEC --prompt-agg attention --no-prompt-residual --seed 42"
  "dec_attention_noresid_s1|6|$DEC --prompt-agg attention --no-prompt-residual --seed 1"
  "dec_attention_noresid_s2|7|$DEC --prompt-agg attention --no-prompt-residual --seed 2"
)

is_done() { local d="$1"; [ -f "$d/results.yaml" ] || return 1
  local n=$(( $(wc -l < "$d/training_diagnostics.csv" 2>/dev/null || echo 1) - 1 )); [ "$n" -ge 28 ]; }

run_slot() {
  local slot="$1"; local gpu=$(( slot % 4 ))
  source "$CONDA_SH"; conda activate open_clip
  export CUDA_VISIBLE_DEVICES="$gpu"
  echo "[slot $slot / GPU $gpu] start $(date)"
  for entry in "${QUEUE[@]}"; do
    IFS='|' read -r name s flags <<< "$entry"
    [ "$s" = "$slot" ] || continue
    out="$ROOT/$name"
    if is_done "$out"; then echo "[slot $slot] SKIP $name (done)"; continue; fi
    rm -rf "$out"; mkdir -p "$out"; echo "flags: $flags" > "$out/run_flags.txt"
    echo "[slot $slot] RUN $name  $(date)"
    python -u clipsam.py $COMMON --output-dir "$out" $flags > "$LOGS/${name}.log" 2>&1 \
      || echo "[slot $slot] FAILED $name"
    echo "[slot $slot] END $name  $(date)"
  done
  echo "[slot $slot] queue finished $(date)"
}

mkdir -p "$ROOT" "$LOGS"
case "${1:-}" in
  slot[0-7]) run_slot "${1#slot}" 2>&1 | tee -a "$LOGS/queue_nolpt_${1}.log" ;;
  launch_all) for s in 0 1 2 3 4 5 6 7; do
      screen -dmS "nl_slot$s" bash -c "cd $SCRIPT_DIR && bash run_nolpt.sh slot$s"; echo "launched nl_slot$s"; done ;;
  dry) for e in "${QUEUE[@]}"; do IFS='|' read -r n s f <<< "$e"; echo "slot $s (GPU $((s%4)))  $n"; done ;;
  *) echo "Usage: bash run_nolpt.sh {launch_all|slot0..slot7|dry}"; exit 1 ;;
esac
