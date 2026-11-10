#!/bin/bash
# ============================================================
# SPC-Net camera-ready: retrain with the REAL CLIP text encoder (2026-11)
# The accepted-paper code skipped the CLIP text transformer, so prompt
# embeddings depended only on token length. The default is now the real
# (frozen) encoder; --legacy-text-encoder reproduces the old numbers.
#
#   30 epochs, --save-best-only, --num-workers 4; every run is evaluated on
#   all 6 benchmarks with continuous maps saved.
#
# Usage:
#   bash run_textfix.sh launch_all          # 4 detached screens (GPUs 0-3)
#   bash run_textfix.sh gpu0|gpu1|gpu2|gpu3
#   bash run_textfix.sh dry
# Safe to relaunch: runs with results.yaml + >=28 epochs are skipped.
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
CONDA_SH="/home/av543f/anaconda3/etc/profile.d/conda.sh"
ROOT="/mnt/data/projects/CAD_Adaptation/CLIPSAM_new_results/outputs_textfix"
LOGS="/mnt/data/projects/CAD_Adaptation/CLIPSAM_new_results/logs_textfix"
EVAL="DUTS-TE,DUT-O,ECSSD,HKU-IS,PASCAL-S-full,SIP"
COMMON="--clip-model ViT-L-14 --epochs 30 --batch-size 8 --num-workers 4 --mse-lambda 0.1 --save-best-only --eval-datasets $EVAL"

# Configuration flags (same definitions as the accepted paper)
BASELINE="--no-prompt-bank --no-adaptive-fallback --no-bidirectional-sam --simple-decoder --plain-bce-loss --pp-loss-weight 0 --ds-weight 0"
PLUS_MP="--no-adaptive-fallback --no-bidirectional-sam --simple-decoder --plain-bce-loss --pp-loss-weight 0 --ds-weight 0"
PLUS_AF="--no-bidirectional-sam --simple-decoder --plain-bce-loss --pp-loss-weight 0 --ds-weight 0"
PLUS_BISAM="--simple-decoder --plain-bce-loss --pp-loss-weight 0 --ds-weight 0"
PLUS_DEC="--pp-loss-weight 0 --ds-weight 0.3"
FULL="--pp-loss-weight 0.3 --ds-weight 0.3"

# name | gpu | flags   (ordered by priority within each GPU)
QUEUE=(
  "plus_af_s42|0|$PLUS_AF --seed 42"
  "plus_mp_s42|0|$PLUS_MP --seed 42"
  "baseline_s2|0|$BASELINE --seed 2"
  "diverse_bank_s42|0|$FULL --diverse-prompts --seed 42"

  "full_max_s42|1|$FULL --seed 42"
  "plus_af_s1|1|$PLUS_AF --seed 1"
  "K1_s42|1|$FULL --prompt-bank-size 1 --seed 42"
  "plus_bisam_s42|1|$PLUS_BISAM --seed 42"

  "full_attention_s42|2|$FULL --prompt-agg attention --seed 42"
  "plus_af_s2|2|$PLUS_AF --seed 2"
  "K5_s42|2|$FULL --prompt-bank-size 5 --seed 42"
  "plus_dec_s42|2|$PLUS_DEC --seed 42"

  "baseline_s42|3|$BASELINE --seed 42"
  "baseline_s1|3|$BASELINE --seed 1"
  "random_bank_s42|3|$FULL --random-prompts --seed 42"

  # Residual on/off (residual only reaches the loss through the correlation-aware
  # decoder, so it is tested on the full model). "_s42" ON runs exist above.
  "full_max_noresid_s42|0|$FULL --no-prompt-residual --seed 42"
  "full_max_noresid_s1|0|$FULL --no-prompt-residual --seed 1"
  "full_max_s1|0|$FULL --seed 1"
  "full_attention_noresid_s42|1|$FULL --prompt-agg attention --no-prompt-residual --seed 42"
  "full_attention_noresid_s1|1|$FULL --prompt-agg attention --no-prompt-residual --seed 1"
  "full_attention_s1|1|$FULL --prompt-agg attention --seed 1"
  "full_max_noresid_s2|2|$FULL --no-prompt-residual --seed 2"
  "full_max_s2|2|$FULL --seed 2"
  "full_attention_noresid_s2|3|$FULL --prompt-agg attention --no-prompt-residual --seed 2"
  "full_attention_s2|3|$FULL --prompt-agg attention --seed 2"
)

is_done() {
  local d="$1"
  [ -f "$d/results.yaml" ] || return 1
  local n=$(( $(wc -l < "$d/training_diagnostics.csv" 2>/dev/null || echo 1) - 1 ))
  [ "$n" -ge 28 ]
}

run_gpu() {
  local gpu="$1"
  source "$CONDA_SH"; conda activate open_clip
  export CUDA_VISIBLE_DEVICES="$gpu"
  echo "[GPU $gpu] start $(date)"
  for entry in "${QUEUE[@]}"; do
    IFS='|' read -r name g flags <<< "$entry"
    [ "$g" = "$gpu" ] || continue
    out="$ROOT/$name"
    if is_done "$out"; then echo "[GPU $gpu] SKIP $name (done)"; continue; fi
    rm -rf "$out"; mkdir -p "$out"
    echo "[GPU $gpu] RUN $name  $(date)"
    echo "flags: $flags" > "$out/run_flags.txt"
    python -u clipsam.py $COMMON --output-dir "$out" $flags \
      > "$LOGS/${name}.log" 2>&1 || echo "[GPU $gpu] FAILED $name (see $LOGS/${name}.log)"
    echo "[GPU $gpu] END $name  $(date)"
  done
  echo "[GPU $gpu] queue finished $(date)"
}

mkdir -p "$ROOT" "$LOGS"
case "${1:-}" in
  gpu0|gpu1|gpu2|gpu3) run_gpu "${1#gpu}" 2>&1 | tee -a "$LOGS/queue_${1}.log" ;;
  launch_all)
    for g in 0 1 2 3; do
      screen -dmS "tf_gpu$g" bash -c "cd $SCRIPT_DIR && bash run_textfix.sh gpu$g"
      echo "launched screen tf_gpu$g"
    done ;;
  dry)
    for entry in "${QUEUE[@]}"; do IFS='|' read -r name g flags <<< "$entry"; echo "GPU $g  $name  $flags"; done ;;
  *) echo "Usage: bash run_textfix.sh {launch_all|gpu0..gpu3|dry}"; exit 1 ;;
esac
# (2026-11-10) VLF multi-token runs on ours were launched by hand as screens vm_*:
#   vlfmt_af_s{42,1,2}: $PLUS_AF --vlf-multi-token --vlf-residual ; vlfmt_nores_af_s42: $PLUS_AF --vlf-multi-token
