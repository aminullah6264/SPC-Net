#!/bin/bash
# Evaluate every outputs_textfix checkpoint on full SIP (929 images).
# Usage: bash run_sipfull_eval.sh {launch_all|stream0..stream3}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"
SRC=/mnt/data/projects/CAD_Adaptation/CLIPSAM_new_results/outputs_textfix
DST=/mnt/data/projects/CAD_Adaptation/CLIPSAM_new_results/outputs_sipfull
run_stream() {
  local s=$1; source /home/av543f/anaconda3/etc/profile.d/conda.sh; conda activate open_clip
  export CUDA_VISIBLE_DEVICES=$s OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
  local i=0
  for d in "$SRC"/*/; do
    name=$(basename "$d"); [ -f "$d/escnet_best.pth" ] || continue
    i=$((i+1)); [ $(( i % 4 )) -eq "$s" ] || continue
    out="$DST/$name"; [ -f "$out/DONE" ] && { echo "SKIP $name"; continue; }
    mkdir -p "$out"; flags=$(sed 's/^flags: //' "$d/run_flags.txt" | sed 's/--seed [0-9]*//')
    echo "EVAL $name $(date)"
    if python -u clipsam.py --clip-model ViT-L-14 --test-only --resume "$d/escnet_best.pth" \
        --output-dir "$out" --eval-datasets SIP-full --no-calibrate --threshold 0.5 \
        --skip-default-eval $flags > "$out/eval.log" 2>&1; then
      echo "$flags" > "$out/flags.txt"; date > "$out/DONE"
    else echo "FAILED $name"; fi
  done; echo "stream $s finished $(date)"
}
mkdir -p "$DST"
case "${1:-}" in
  stream[0-3]) run_stream "${1#stream}" 2>&1 | tee -a "$DST/stream_${1}.log" ;;
  launch_all) for s in 0 1 2 3; do screen -dmS "sf_$s" bash -c "cd $SCRIPT_DIR && bash run_sipfull_eval.sh stream$s"; done ;;
  *) echo "Usage: $0 {launch_all|stream0..3}"; exit 1 ;;
esac
