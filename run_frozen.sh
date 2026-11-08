#!/bin/bash
# ============================================================
# SPC-Net camera-ready, option b: frozen-encoder ablation set (2026-11)
# Real CLIP text encoder + --freeze-pretrained. 8 slots (2 per GPU).
# Already run separately: frozen +AF (seeds 42,1,2), frozen full+attention s42.
# Usage: bash run_frozen.sh {launch_all|slot0..slot7|dry}
# ============================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
CONDA_SH="/home/av543f/anaconda3/etc/profile.d/conda.sh"
ROOT="/mnt/data/projects/CAD_Adaptation/CLIPSAM_new_results/outputs_textfix"
LOGS="/mnt/data/projects/CAD_Adaptation/CLIPSAM_new_results/logs_textfix"
EVAL="DUTS-TE,DUT-O,ECSSD,HKU-IS,PASCAL-S-full,SIP"
COMMON="--clip-model ViT-L-14 --epochs 30 --batch-size 8 --num-workers 4 --mse-lambda 0.1 --save-best-only --eval-datasets $EVAL --freeze-pretrained"

BASELINE="--no-prompt-bank --no-adaptive-fallback --no-bidirectional-sam --simple-decoder --plain-bce-loss --pp-loss-weight 0 --ds-weight 0"
PLUS_MP="--no-adaptive-fallback --no-bidirectional-sam --simple-decoder --plain-bce-loss --pp-loss-weight 0 --ds-weight 0"
PLUS_BISAM="--simple-decoder --plain-bce-loss --pp-loss-weight 0 --ds-weight 0"
PLUS_DEC="--pp-loss-weight 0 --ds-weight 0.3"
FULL="--pp-loss-weight 0.3 --ds-weight 0.3"

# name | slot (gpu = slot % 4) | flags
QUEUE=(
  "frozen_baseline_s42|0|$BASELINE --seed 42"
  "frozen_baseline_s1|1|$BASELINE --seed 1"
  "frozen_baseline_s2|2|$BASELINE --seed 2"
  "frozen_plus_mp_s42|3|$PLUS_MP --seed 42"
  "frozen_full_max_s42|4|$FULL --seed 42"
  "frozen_plus_bisam_s42|5|$PLUS_BISAM --seed 42"
  "frozen_plus_dec_s42|6|$PLUS_DEC --seed 42"
  "frozen_full_max_noresid_s42|7|$FULL --no-prompt-residual --seed 42"
  "frozen_full_attention_noresid_s42|0|$FULL --prompt-agg attention --no-prompt-residual --seed 42"
  "frozen_K1_s42|1|$FULL --prompt-bank-size 1 --seed 42"
  "frozen_K5_s42|2|$FULL --prompt-bank-size 5 --seed 42"
  "frozen_random_bank_s42|3|$FULL --random-prompts --seed 42"
  "frozen_diverse_bank_s42|4|$FULL --diverse-prompts --seed 42"
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
    rm -rf "$out"; mkdir -p "$out"; echo "flags: $flags --freeze-pretrained" > "$out/run_flags.txt"
    echo "[slot $slot] RUN $name  $(date)"
    python -u clipsam.py $COMMON --output-dir "$out" $flags > "$LOGS/${name}.log" 2>&1 \
      || echo "[slot $slot] FAILED $name"
    echo "[slot $slot] END $name  $(date)"
  done
  echo "[slot $slot] queue finished $(date)"
}

mkdir -p "$ROOT" "$LOGS"
case "${1:-}" in
  slot[0-7]) run_slot "${1#slot}" 2>&1 | tee -a "$LOGS/queue_frozen_${1}.log" ;;
  launch_all) for s in 0 1 2 3 4 5 6 7; do
      screen -dmS "fz_slot$s" bash -c "cd $SCRIPT_DIR && bash run_frozen.sh slot$s"; echo "launched fz_slot$s"; done ;;
  dry) for e in "${QUEUE[@]}"; do IFS='|' read -r n s f <<< "$e"; echo "slot $s (GPU $((s%4)))  $n"; done ;;
  *) echo "Usage: bash run_frozen.sh {launch_all|slot0..slot7|dry}"; exit 1 ;;
esac
