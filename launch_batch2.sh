#!/bin/bash
# ============================================================
# SPC-Net Architecture B — BATCH 2: Prompt Analysis
# Run after Batch 1 completes (or in parallel on free GPUs)
#
#   GPU 1: Attention prompts → Gated prompts
#   GPU 2: K=1 → K=5 → K=20
#   GPU 3: gamma=0 → gamma=1 → mean-agg → random-bank
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_SH="/home/av543f/anaconda3/etc/profile.d/conda.sh"
EPOCHS=30
SEED=42

mkdir -p outputs/logs

echo "============================================================"
echo " SPC-Net Architecture B — BATCH 2: Prompt Analysis"
echo " Epochs: $EPOCHS | Seed: $SEED | Residual: ON"
echo " Time: $(date)"
echo "============================================================"

# ── GPU 1: Attention → Gated ─────────────────────────────────────────────
cat > /tmp/archB_b2_gpu1.sh << SCRIPT
#!/bin/bash
set -e
source $CONDA_SH
conda activate open_clip
cd $SCRIPT_DIR
export CUDA_VISIBLE_DEVICES=1

echo "[GPU1] \$(date) — Attention aggregation..."
python clipsam.py \\
    --clip-model ViT-L-14 --epochs $EPOCHS --seed $SEED \\
    --batch-size 8 --num-workers 8 --mse-lambda 0.1 \\
    --output-dir outputs/prompt_modes/attention \\
    --eval-datasets DUTS-TE,DUT-O,SIP \\
    --prompt-agg attention \\
    --pp-loss-weight 0.3 --ds-weight 0.3

echo "[GPU1] \$(date) — Gated prompts..."
python clipsam.py \\
    --clip-model ViT-L-14 --epochs $EPOCHS --seed $SEED \\
    --batch-size 8 --num-workers 8 --mse-lambda 0.1 \\
    --output-dir outputs/prompt_modes/gated \\
    --eval-datasets DUTS-TE,DUT-O,SIP \\
    --prompt-agg gated \\
    --pp-loss-weight 0.3 --ds-weight 0.3

echo "[GPU1] \$(date) — BATCH 2 GPU1 COMPLETE."
SCRIPT
chmod +x /tmp/archB_b2_gpu1.sh

# ── GPU 2: K=1 → K=5 → K=20 ─────────────────────────────────────────────
cat > /tmp/archB_b2_gpu2.sh << SCRIPT
#!/bin/bash
set -e
source $CONDA_SH
conda activate open_clip
cd $SCRIPT_DIR
export CUDA_VISIBLE_DEVICES=2

echo "[GPU2] \$(date) — K=1 (single prompt)..."
python clipsam.py \\
    --clip-model ViT-L-14 --epochs $EPOCHS --seed $SEED \\
    --batch-size 8 --num-workers 8 --mse-lambda 0.1 \\
    --output-dir outputs/prompt_analysis/K1 \\
    --eval-datasets DUTS-TE \\
    --prompt-bank-size 1 \\
    --pp-loss-weight 0.3 --ds-weight 0.3

echo "[GPU2] \$(date) — K=5..."
python clipsam.py \\
    --clip-model ViT-L-14 --epochs $EPOCHS --seed $SEED \\
    --batch-size 8 --num-workers 8 --mse-lambda 0.1 \\
    --output-dir outputs/prompt_analysis/K5 \\
    --eval-datasets DUTS-TE \\
    --prompt-bank-size 5 \\
    --pp-loss-weight 0.3 --ds-weight 0.3

echo "[GPU2] \$(date) — K=20..."
python clipsam.py \\
    --clip-model ViT-L-14 --epochs $EPOCHS --seed $SEED \\
    --batch-size 8 --num-workers 8 --mse-lambda 0.1 \\
    --output-dir outputs/prompt_analysis/K20 \\
    --eval-datasets DUTS-TE \\
    --prompt-bank-size 20 \\
    --pp-loss-weight 0.3 --ds-weight 0.3

echo "[GPU2] \$(date) — BATCH 2 GPU2 COMPLETE."
SCRIPT
chmod +x /tmp/archB_b2_gpu2.sh

# ── GPU 3: gamma=0 → gamma=1 → mean-agg → random-bank ───────────────────
cat > /tmp/archB_b2_gpu3.sh << SCRIPT
#!/bin/bash
set -e
source $CONDA_SH
conda activate open_clip
cd $SCRIPT_DIR
export CUDA_VISIBLE_DEVICES=3

echo "[GPU3] \$(date) — gamma=0 (no point loss)..."
python clipsam.py \\
    --clip-model ViT-L-14 --epochs $EPOCHS --seed $SEED \\
    --batch-size 8 --num-workers 8 --mse-lambda 0.1 \\
    --output-dir outputs/prompt_analysis/gamma0 \\
    --eval-datasets DUTS-TE \\
    --pp-loss-weight 0 --ds-weight 0.3

echo "[GPU3] \$(date) — gamma=1.0 (heavy point loss)..."
python clipsam.py \\
    --clip-model ViT-L-14 --epochs $EPOCHS --seed $SEED \\
    --batch-size 8 --num-workers 8 --mse-lambda 0.1 \\
    --output-dir outputs/prompt_analysis/gamma1 \\
    --eval-datasets DUTS-TE \\
    --pp-loss-weight 1.0 --ds-weight 0.3

echo "[GPU3] \$(date) — Mean aggregation..."
python clipsam.py \\
    --clip-model ViT-L-14 --epochs $EPOCHS --seed $SEED \\
    --batch-size 8 --num-workers 8 --mse-lambda 0.1 \\
    --output-dir outputs/prompt_analysis/agg_mean \\
    --eval-datasets DUTS-TE \\
    --prompt-agg mean \\
    --pp-loss-weight 0.3 --ds-weight 0.3

echo "[GPU3] \$(date) — Random prompts..."
python clipsam.py \\
    --clip-model ViT-L-14 --epochs $EPOCHS --seed $SEED \\
    --batch-size 8 --num-workers 8 --mse-lambda 0.1 \\
    --output-dir outputs/prompt_analysis/random_bank \\
    --eval-datasets DUTS-TE \\
    --random-prompts \\
    --pp-loss-weight 0.3 --ds-weight 0.3

echo "[GPU3] \$(date) — BATCH 2 GPU3 COMPLETE."
SCRIPT
chmod +x /tmp/archB_b2_gpu3.sh

# ── LAUNCH ────────────────────────────────────────────────────────────────
nohup bash /tmp/archB_b2_gpu1.sh > outputs/logs/batch2_gpu1.log 2>&1 &
PID1=$!
nohup bash /tmp/archB_b2_gpu2.sh > outputs/logs/batch2_gpu2.log 2>&1 &
PID2=$!
nohup bash /tmp/archB_b2_gpu3.sh > outputs/logs/batch2_gpu3.log 2>&1 &
PID3=$!

echo "$PID1 $PID2 $PID3" > outputs/logs/.pids_batch2
echo ""
echo "  GPU 1: PID $PID1 (attention → gated)"
echo "  GPU 2: PID $PID2 (K=1 → K=5 → K=20)"
echo "  GPU 3: PID $PID3 (γ=0 → γ=1 → mean → random)"
echo ""
echo "  Logs: outputs/logs/batch2_gpu{1,2,3}.log"
