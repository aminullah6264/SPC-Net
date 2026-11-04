#!/bin/bash
# GPU 0: Pick up experiments from other GPUs' queues to maximize utilization
# Takes: K5 (from GPU1 queue), K20 (from GPU2 queue), Base+Attention (from GPU3 queue)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_SH="/home/av543f/anaconda3/etc/profile.d/conda.sh"
RESULTS_ROOT="/mnt/data/projects/CAD_Adaptation/CLIPSAM_new_results/outputs"
EPOCHS=30
SEED=42
COMMON="--clip-model ViT-L-14 --epochs $EPOCHS --seed $SEED --batch-size 8 --num-workers 8 --mse-lambda 0.1 --save-best-only"

source "$CONDA_SH"
conda activate open_clip
export CUDA_VISIBLE_DEVICES=0

echo "=========================================="
echo " GPU 0 — Starting at $(date)"
echo " Taking queued work from GPUs 1, 2, 3"
echo "=========================================="

# 1. K=5 rerun (was queued on GPU1 as 3rd run)
echo ""
echo "============================================================"
echo " EXPERIMENT: K=5 (prompt bank size) — taken from GPU1 queue"
echo " Output: $RESULTS_ROOT/prompt_analysis/K5"
echo " Time: $(date)"
echo "============================================================"
rm -rf "$RESULTS_ROOT/prompt_analysis/K5"
mkdir -p "$RESULTS_ROOT/prompt_analysis/K5"
python clipsam.py $COMMON \
    --output-dir "$RESULTS_ROOT/prompt_analysis/K5" \
    --eval-datasets DUTS-TE \
    --prompt-bank-size 5 \
    --pp-loss-weight 0.3 --ds-weight 0.3
echo "[DONE] K=5 completed at $(date)"

# 2. K=20 (was queued on GPU2 as 2nd run)
echo ""
echo "============================================================"
echo " EXPERIMENT: K=20 (prompt bank size) — taken from GPU2 queue"
echo " Output: $RESULTS_ROOT/prompt_analysis/K20"
echo " Time: $(date)"
echo "============================================================"
rm -rf "$RESULTS_ROOT/prompt_analysis/K20"
mkdir -p "$RESULTS_ROOT/prompt_analysis/K20"
python clipsam.py $COMMON \
    --output-dir "$RESULTS_ROOT/prompt_analysis/K20" \
    --eval-datasets DUTS-TE \
    --prompt-bank-size 20 \
    --pp-loss-weight 0.3 --ds-weight 0.3
echo "[DONE] K=20 completed at $(date)"

# 3. Baseline + Attention (was queued on GPU3 as 3rd run)
echo ""
echo "============================================================"
echo " EXPERIMENT: Baseline + Attention agg — taken from GPU3 queue"
echo " Output: $RESULTS_ROOT/prompt_modes/baseline_attention"
echo " Time: $(date)"
echo "============================================================"
rm -rf "$RESULTS_ROOT/prompt_modes/baseline_attention"
mkdir -p "$RESULTS_ROOT/prompt_modes/baseline_attention"
python clipsam.py $COMMON \
    --output-dir "$RESULTS_ROOT/prompt_modes/baseline_attention" \
    --eval-datasets DUTS-TE,DUT-O,SIP \
    --prompt-agg attention \
    --no-adaptive-fallback --no-bidirectional-sam \
    --simple-decoder --plain-bce-loss \
    --pp-loss-weight 0 --ds-weight 0
echo "[DONE] Baseline+Attention completed at $(date)"

echo ""
echo "=========================================="
echo " GPU 0 — ALL COMPLETE at $(date)"
echo "=========================================="
