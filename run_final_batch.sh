#!/bin/bash
# Phase 1: Evaluate interrupted runs (test-only)
# Phase 2: Launch remaining 7 experiments with 30 epochs
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")
CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"
source "$CONDA_SH"
conda activate open_clip

mkdir -p outputs/logs

echo "============================================================"
echo " Phase 1: Evaluate interrupted checkpoints (test-only)"
echo " Time: $(date)"
echo "============================================================"

# Evaluate +af
echo "[EVAL] +af checkpoint..."
CUDA_VISIBLE_DEVICES=1 python clipsam.py \
    --clip-model ViT-L-14 --test-only \
    --resume outputs/ablation/ablation_sip/plus_af/escnet_best.pth \
    --output-dir outputs/ablation/ablation_sip/plus_af \
    --eval-datasets DUTS-TE \
    --no-bidirectional-sam --simple-decoder --plain-bce-loss \
    --pp-loss-weight 0 --ds-weight 0 \
    --no-calibrate --threshold 0.5

# Evaluate +dec
echo "[EVAL] +dec checkpoint..."
CUDA_VISIBLE_DEVICES=1 python clipsam.py \
    --clip-model ViT-L-14 --test-only \
    --resume outputs/ablation/ablation_sip/plus_dec/escnet_best.pth \
    --output-dir outputs/ablation/ablation_sip/plus_dec \
    --eval-datasets DUTS-TE \
    --pp-loss-weight 0 --ds-weight 0.3 \
    --no-calibrate --threshold 0.5

# Evaluate gamma0
echo "[EVAL] gamma0 checkpoint..."
CUDA_VISIBLE_DEVICES=1 python clipsam.py \
    --clip-model ViT-L-14 --test-only \
    --resume outputs/prompt_analysis/gamma0/escnet_best.pth \
    --output-dir outputs/prompt_analysis/gamma0 \
    --eval-datasets DUTS-TE \
    --pp-loss-weight 0 \
    --no-calibrate --threshold 0.5

echo ""
echo "============================================================"
echo " Phase 2: Launch remaining experiments (30 epochs)"
echo " Time: $(date)"
echo "============================================================"

# ============================================================
# GPU 1: +lpt → K1 → K5 → K20
# ============================================================
cat > /tmp/spcnet_final_gpu1.sh << GPUSCRIPT
#!/bin/bash
set -e
source $CONDA_SH
conda activate open_clip
cd $SCRIPT_DIR
export CUDA_VISIBLE_DEVICES=1

echo "[GPU 1] \$(date) — +lpt (full model ablation endpoint, 30 epochs)..."
python clipsam.py \
    --clip-model ViT-L-14 --epochs 30 --seed 42 \
    --batch-size 8 --num-workers 8 --mse-lambda 0.1 \
    --output-dir outputs/ablation/ablation_sip/plus_lpt \
    --eval-datasets DUTS-TE \
    --pp-loss-weight 0.3 --ds-weight 0.3

echo "[GPU 1] \$(date) — K=1 (single prompt)..."
python clipsam.py \
    --clip-model ViT-L-14 --epochs 30 --seed 42 \
    --batch-size 8 --num-workers 8 --mse-lambda 0.1 \
    --output-dir outputs/prompt_analysis/K1 \
    --eval-datasets DUTS-TE \
    --prompt-bank-size 1

echo "[GPU 1] \$(date) — K=5..."
python clipsam.py \
    --clip-model ViT-L-14 --epochs 30 --seed 42 \
    --batch-size 8 --num-workers 8 --mse-lambda 0.1 \
    --output-dir outputs/prompt_analysis/K5 \
    --eval-datasets DUTS-TE \
    --prompt-bank-size 5

echo "[GPU 1] \$(date) — K=20..."
python clipsam.py \
    --clip-model ViT-L-14 --epochs 30 --seed 42 \
    --batch-size 8 --num-workers 8 --mse-lambda 0.1 \
    --output-dir outputs/prompt_analysis/K20 \
    --eval-datasets DUTS-TE \
    --prompt-bank-size 20

echo "[GPU 1] \$(date) — ALL COMPLETE."
GPUSCRIPT
chmod +x /tmp/spcnet_final_gpu1.sh

# ============================================================
# GPU 2: mean-agg → random-bank
# ============================================================
cat > /tmp/spcnet_final_gpu2.sh << GPUSCRIPT
#!/bin/bash
set -e
source $CONDA_SH
conda activate open_clip
cd $SCRIPT_DIR
export CUDA_VISIBLE_DEVICES=2

echo "[GPU 2] \$(date) — mean-aggregation..."
python clipsam.py \
    --clip-model ViT-L-14 --epochs 30 --seed 42 \
    --batch-size 8 --num-workers 8 --mse-lambda 0.1 \
    --output-dir outputs/prompt_analysis/agg_mean \
    --eval-datasets DUTS-TE \
    --prompt-agg mean

echo "[GPU 2] \$(date) — random-bank..."
python clipsam.py \
    --clip-model ViT-L-14 --epochs 30 --seed 42 \
    --batch-size 8 --num-workers 8 --mse-lambda 0.1 \
    --output-dir outputs/prompt_analysis/random_bank \
    --eval-datasets DUTS-TE \
    --random-prompts

echo "[GPU 2] \$(date) — ALL COMPLETE."
GPUSCRIPT
chmod +x /tmp/spcnet_final_gpu2.sh

# ============================================================
# GPU 3: gamma=1.0
# ============================================================
cat > /tmp/spcnet_final_gpu3.sh << GPUSCRIPT
#!/bin/bash
set -e
source $CONDA_SH
conda activate open_clip
cd $SCRIPT_DIR
export CUDA_VISIBLE_DEVICES=3

echo "[GPU 3] \$(date) — gamma=1.0 (heavy point loss)..."
python clipsam.py \
    --clip-model ViT-L-14 --epochs 30 --seed 42 \
    --batch-size 8 --num-workers 8 --mse-lambda 0.1 \
    --output-dir outputs/prompt_analysis/gamma1 \
    --eval-datasets DUTS-TE \
    --pp-loss-weight 1.0

echo "[GPU 3] \$(date) — ALL COMPLETE."
GPUSCRIPT
chmod +x /tmp/spcnet_final_gpu3.sh

# Clean up old interrupted data for reruns
rm -rf outputs/prompt_analysis/K1 outputs/prompt_analysis/K5 outputs/prompt_analysis/K20
rm -rf outputs/prompt_analysis/agg_mean outputs/prompt_analysis/random_bank
rm -rf outputs/prompt_analysis/gamma1
rm -rf outputs/ablation/ablation_sip/plus_lpt

# Launch
nohup bash /tmp/spcnet_final_gpu1.sh > outputs/logs/gpu1_final.log 2>&1 &
PID1=$!

nohup bash /tmp/spcnet_final_gpu2.sh > outputs/logs/gpu2_final.log 2>&1 &
PID2=$!

nohup bash /tmp/spcnet_final_gpu3.sh > outputs/logs/gpu3_final.log 2>&1 &
PID3=$!

echo ""
echo "  GPU 1: PID $PID1 (+lpt → K1 → K5 → K20)"
echo "  GPU 2: PID $PID2 (mean-agg → random-bank)"
echo "  GPU 3: PID $PID3 (gamma=1.0)"
echo ""
echo "$PID1 $PID2 $PID3" > outputs/logs/.pids_final
echo "  PIDs saved. Monitor: tail -f outputs/logs/gpu1_final.log"
echo "  Expected: ~7h per GPU (30 epochs × ~14min/epoch)"
