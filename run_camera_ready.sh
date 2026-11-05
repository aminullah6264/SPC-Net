#!/bin/bash
# ============================================================
# SPC-Net camera-ready experiments (2026-11)
#   Matched residual ablation, seeds, and the VLF fix.
#   30 epochs, --save-best-only, --num-workers 4.
#   Every run evaluates on all 6 benchmarks and saves continuous maps.
#
# Usage:
#   bash run_camera_ready.sh launch_all     # 4 detached screens (GPUs 0-3)
#   bash run_camera_ready.sh gpu0|gpu1|gpu2|gpu3
#   bash run_camera_ready.sh dry            # print the queue
# Safe to relaunch: runs with results.yaml + >=28 epochs are skipped.
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
CONDA_SH="/home/av543f/anaconda3/etc/profile.d/conda.sh"
ROOT="/mnt/data/projects/CAD_Adaptation/CLIPSAM_new_results/outputs_camera_ready"
LOGS="/mnt/data/projects/CAD_Adaptation/CLIPSAM_new_results/logs_camera_ready"
EVAL="DUTS-TE,DUT-O,ECSSD,HKU-IS,PASCAL-S-full,SIP"
COMMON="--clip-model ViT-L-14 --epochs 30 --batch-size 8 --num-workers 4 --mse-lambda 0.1 --save-best-only --eval-datasets $EVAL"

# Configuration flags (same definitions as the accepted paper)
BASELINE="--no-prompt-bank --no-adaptive-fallback --no-bidirectional-sam --simple-decoder --plain-bce-loss --pp-loss-weight 0 --ds-weight 0"
PLUS_MP="--no-adaptive-fallback --no-bidirectional-sam --simple-decoder --plain-bce-loss --pp-loss-weight 0 --ds-weight 0"
PLUS_AF="--no-bidirectional-sam --simple-decoder --plain-bce-loss --pp-loss-weight 0 --ds-weight 0"
FULL="--pp-loss-weight 0.3 --ds-weight 0.3"
VLF_FIX="--vlf-multi-token --vlf-residual"

# name | gpu | flags
QUEUE=(
  # GPU 0: VLF fix on the recommended config, then matched residual-off (full+attention)
  "vlffix_plus_af_s42|0|$PLUS_AF $VLF_FIX --seed 42"
  "full_attention_noresid_s42|0|$FULL --prompt-agg attention --no-prompt-residual --seed 42"
  "plus_af_s2|0|$PLUS_AF --seed 2"
  # GPU 1: VLF fix on full model (max and attention)
  "vlffix_full_max_s42|1|$FULL $VLF_FIX --seed 42"
  "full_max_noresid_s42|1|$FULL --no-prompt-residual --seed 42"
  "baseline_s2|1|$BASELINE --seed 2"
  # GPU 2: VLF fix full+attention, then seeds
  "vlffix_full_attention_s42|2|$FULL --prompt-agg attention $VLF_FIX --seed 42"
  "baseline_s1|2|$BASELINE --seed 1"
  # GPU 3: VLF fix on +MP, then +AF seed 1
  "vlffix_plus_mp_s42|3|$PLUS_MP $VLF_FIX --seed 42"
  "plus_af_s1|3|$PLUS_AF --seed 1"
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
      screen -dmS "cr_gpu$g" bash -c "cd $SCRIPT_DIR && bash run_camera_ready.sh gpu$g"
      echo "launched screen cr_gpu$g"
    done ;;
  dry)
    for entry in "${QUEUE[@]}"; do IFS='|' read -r name g flags <<< "$entry"; echo "GPU $g  $name  $flags"; done ;;
  *) echo "Usage: bash run_camera_ready.sh {launch_all|gpu0..gpu3|dry}"; exit 1 ;;
esac
