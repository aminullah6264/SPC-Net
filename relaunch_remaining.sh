#!/bin/bash
# SPC-Net: Relaunch remaining experiments
# Interrupted runs restart from scratch (60 epochs)
# GPUs 1, 2, 3 all available
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")
CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"

mkdir -p outputs/logs

# ============================================================
# GPU 1: Resume +af (ablation) → +lpt (ablation) → K1, K5, K20 (prompt)
# ============================================================
cat > /tmp/spcnet_relaunch_gpu1.sh << GPUSCRIPT
#!/bin/bash
set -e
source $CONDA_SH
conda activate open_clip
cd $SCRIPT_DIR
export CUDA_VISIBLE_DEVICES=1

echo "[GPU 1] \$(date) — Starting +af ablation (fresh 60 epochs)..."
python clipsam.py \\
    --clip-model ViT-L-14 --epochs 60 --seed 42 \\
    --batch-size 8 --num-workers 8 --mse-lambda 0.1 \\
    --output-dir outputs/ablation/ablation_sip/plus_af \\
    --eval-datasets DUTS-TE \\
    --no-bidirectional-sam --simple-decoder --plain-bce-loss \\
    --pp-loss-weight 0 --ds-weight 0

echo "[GPU 1] \$(date) — +af done. Starting +lpt (full model = ablation endpoint)..."
python clipsam.py \\
    --clip-model ViT-L-14 --epochs 60 --seed 42 \\
    --batch-size 8 --num-workers 8 --mse-lambda 0.1 \\
    --output-dir outputs/ablation/ablation_sip/plus_lpt \\
    --eval-datasets DUTS-TE \\
    --pp-loss-weight 0.3 --ds-weight 0.3

echo "[GPU 1] \$(date) — +lpt done. Starting prompt K=1..."
python clipsam.py \\
    --clip-model ViT-L-14 --epochs 60 --seed 42 \\
    --batch-size 8 --num-workers 8 --mse-lambda 0.1 \\
    --output-dir outputs/prompt_analysis/K1 \\
    --eval-datasets DUTS-TE \\
    --prompt-bank-size 1

echo "[GPU 1] \$(date) — K1 done. Starting prompt K=5..."
python clipsam.py \\
    --clip-model ViT-L-14 --epochs 60 --seed 42 \\
    --batch-size 8 --num-workers 8 --mse-lambda 0.1 \\
    --output-dir outputs/prompt_analysis/K5 \\
    --eval-datasets DUTS-TE \\
    --prompt-bank-size 5

echo "[GPU 1] \$(date) — K5 done. Starting prompt K=20..."
python clipsam.py \\
    --clip-model ViT-L-14 --epochs 60 --seed 42 \\
    --batch-size 8 --num-workers 8 --mse-lambda 0.1 \\
    --output-dir outputs/prompt_analysis/K20 \\
    --eval-datasets DUTS-TE \\
    --prompt-bank-size 20

echo "[GPU 1] \$(date) — ALL TASKS COMPLETE."
GPUSCRIPT
chmod +x /tmp/spcnet_relaunch_gpu1.sh

# ============================================================
# GPU 2: Resume +dec (ablation) → mean-agg → random-bank (prompt)
# ============================================================
cat > /tmp/spcnet_relaunch_gpu2.sh << GPUSCRIPT
#!/bin/bash
set -e
source $CONDA_SH
conda activate open_clip
cd $SCRIPT_DIR
export CUDA_VISIBLE_DEVICES=2

echo "[GPU 2] \$(date) — Starting +dec ablation (fresh 60 epochs)..."
python clipsam.py \\
    --clip-model ViT-L-14 --epochs 60 --seed 42 \\
    --batch-size 8 --num-workers 8 --mse-lambda 0.1 \\
    --output-dir outputs/ablation/ablation_sip/plus_dec \\
    --eval-datasets DUTS-TE \\
    --pp-loss-weight 0 --ds-weight 0.3

echo "[GPU 2] \$(date) — +dec done. Starting mean-aggregation..."
python clipsam.py \\
    --clip-model ViT-L-14 --epochs 60 --seed 42 \\
    --batch-size 8 --num-workers 8 --mse-lambda 0.1 \\
    --output-dir outputs/prompt_analysis/agg_mean \\
    --eval-datasets DUTS-TE \\
    --prompt-agg mean

echo "[GPU 2] \$(date) — mean-agg done. Starting random-bank..."
python clipsam.py \\
    --clip-model ViT-L-14 --epochs 60 --seed 42 \\
    --batch-size 8 --num-workers 8 --mse-lambda 0.1 \\
    --output-dir outputs/prompt_analysis/random_bank \\
    --eval-datasets DUTS-TE \\
    --random-prompts

echo "[GPU 2] \$(date) — ALL TASKS COMPLETE."
GPUSCRIPT
chmod +x /tmp/spcnet_relaunch_gpu2.sh

# ============================================================
# GPU 3: γ=0, γ=1.0 (prompt analysis)
# ============================================================
cat > /tmp/spcnet_relaunch_gpu3.sh << GPUSCRIPT
#!/bin/bash
set -e
source $CONDA_SH
conda activate open_clip
cd $SCRIPT_DIR
export CUDA_VISIBLE_DEVICES=3

echo "[GPU 3] \$(date) — Starting gamma=0 (no point loss)..."
python clipsam.py \\
    --clip-model ViT-L-14 --epochs 60 --seed 42 \\
    --batch-size 8 --num-workers 8 --mse-lambda 0.1 \\
    --output-dir outputs/prompt_analysis/gamma0 \\
    --eval-datasets DUTS-TE \\
    --pp-loss-weight 0

echo "[GPU 3] \$(date) — gamma0 done. Starting gamma=1.0..."
python clipsam.py \\
    --clip-model ViT-L-14 --epochs 60 --seed 42 \\
    --batch-size 8 --num-workers 8 --mse-lambda 0.1 \\
    --output-dir outputs/prompt_analysis/gamma1 \\
    --eval-datasets DUTS-TE \\
    --pp-loss-weight 1.0

echo "[GPU 3] \$(date) — ALL TASKS COMPLETE."
GPUSCRIPT
chmod +x /tmp/spcnet_relaunch_gpu3.sh

# ============================================================
# Launch
# ============================================================
echo "============================================================"
echo " SPC-Net Remaining Experiments — Relaunching on GPUs 1, 2, 3"
echo " Time: $(date)"
echo " Logs: outputs/logs/gpu{1,2,3}_relaunch.log"
echo "============================================================"
echo ""
echo " GPU 1: +af → +lpt → K1 → K5 → K20 (5 runs)"
echo " GPU 2: +dec → mean-agg → random-bank (3 runs)"
echo " GPU 3: γ=0 → γ=1.0 (2 runs)"
echo ""

nohup bash /tmp/spcnet_relaunch_gpu1.sh > outputs/logs/gpu1_relaunch.log 2>&1 &
PID1=$!
echo "  GPU 1: PID $PID1"

nohup bash /tmp/spcnet_relaunch_gpu2.sh > outputs/logs/gpu2_relaunch.log 2>&1 &
PID2=$!
echo "  GPU 2: PID $PID2"

nohup bash /tmp/spcnet_relaunch_gpu3.sh > outputs/logs/gpu3_relaunch.log 2>&1 &
PID3=$!
echo "  GPU 3: PID $PID3"

echo ""
echo "$PID1 $PID2 $PID3" > outputs/logs/.pids_relaunch
echo "  PIDs saved to outputs/logs/.pids_relaunch"
echo "  Monitor: tail -f outputs/logs/gpu1_relaunch.log"
