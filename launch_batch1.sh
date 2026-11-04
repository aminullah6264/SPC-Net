#!/bin/bash
# ============================================================
# SPC-Net Architecture B: Full experiment re-run
# Correlation residual enabled, 30 epochs (converges by ep 25)
# 
# System has ~12h walltime limit, so each GPU runs max 2 full runs
# per launch batch. We'll need multiple launch batches.
#
# BATCH 1 (this script): Core experiments
#   GPU 1: Full model (all benchmarks) → Learnable prompts
#   GPU 2: Ablation baseline → +MP → +AF
#   GPU 3: +BiSAM → +Dec → +Lpt
#
# BATCH 2 (launch_batch2.sh): Prompt analysis
#   GPU 1: Attention → Gated
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
echo " SPC-Net Architecture B — BATCH 1: Core Experiments"
echo " Epochs: $EPOCHS | Seed: $SEED | Residual: ON"
echo " Time: $(date)"
echo "============================================================"

# ── GPU 1: Full model (all benchmarks) + Learnable prompts ────────────────
cat > /tmp/archB_gpu1.sh << SCRIPT
#!/bin/bash
set -e
source $CONDA_SH
conda activate open_clip
cd $SCRIPT_DIR
export CUDA_VISIBLE_DEVICES=1

echo "[GPU1] \$(date) — Full model (max prompts, all benchmarks)..."
python clipsam.py \\
    --clip-model ViT-L-14 --epochs $EPOCHS --seed $SEED \\
    --batch-size 8 --num-workers 8 --mse-lambda 0.1 \\
    --output-dir outputs/full_model \\
    --eval-datasets DUTS-TE,DUT-O,ECSSD,SIP \\
    --prompt-agg max \\
    --pp-loss-weight 0.3 --ds-weight 0.3

echo "[GPU1] \$(date) — Learnable prompts (K=10)..."
python clipsam.py \\
    --clip-model ViT-L-14 --epochs $EPOCHS --seed $SEED \\
    --batch-size 8 --num-workers 8 --mse-lambda 0.1 \\
    --output-dir outputs/prompt_modes/learnable \\
    --eval-datasets DUTS-TE,DUT-O,SIP \\
    --prompt-agg learnable \\
    --pp-loss-weight 0.3 --ds-weight 0.3

echo "[GPU1] \$(date) — BATCH 1 COMPLETE."
SCRIPT
chmod +x /tmp/archB_gpu1.sh

# ── GPU 2: Ablation baseline → +MP → +AF ─────────────────────────────────
cat > /tmp/archB_gpu2.sh << SCRIPT
#!/bin/bash
set -e
source $CONDA_SH
conda activate open_clip
cd $SCRIPT_DIR
export CUDA_VISIBLE_DEVICES=2

echo "[GPU2] \$(date) — Ablation: baseline..."
python clipsam.py \\
    --clip-model ViT-L-14 --epochs $EPOCHS --seed $SEED \\
    --batch-size 8 --num-workers 8 --mse-lambda 0.1 \\
    --output-dir outputs/ablation/baseline \\
    --eval-datasets DUTS-TE,DUT-O,SIP \\
    --no-prompt-bank --no-adaptive-fallback --no-bidirectional-sam \\
    --simple-decoder --plain-bce-loss --pp-loss-weight 0 --ds-weight 0

echo "[GPU2] \$(date) — Ablation: +MP (multi-prompt)..."
python clipsam.py \\
    --clip-model ViT-L-14 --epochs $EPOCHS --seed $SEED \\
    --batch-size 8 --num-workers 8 --mse-lambda 0.1 \\
    --output-dir outputs/ablation/plus_mp \\
    --eval-datasets DUTS-TE,DUT-O,SIP \\
    --no-adaptive-fallback --no-bidirectional-sam \\
    --simple-decoder --plain-bce-loss --pp-loss-weight 0 --ds-weight 0

echo "[GPU2] \$(date) — Ablation: +AF (adaptive fallback)..."
python clipsam.py \\
    --clip-model ViT-L-14 --epochs $EPOCHS --seed $SEED \\
    --batch-size 8 --num-workers 8 --mse-lambda 0.1 \\
    --output-dir outputs/ablation/plus_af \\
    --eval-datasets DUTS-TE,DUT-O,SIP \\
    --no-bidirectional-sam \\
    --simple-decoder --plain-bce-loss --pp-loss-weight 0 --ds-weight 0

echo "[GPU2] \$(date) — BATCH 1 COMPLETE."
SCRIPT
chmod +x /tmp/archB_gpu2.sh

# ── GPU 3: +BiSAM → +Dec → +Lpt ─────────────────────────────────────────
cat > /tmp/archB_gpu3.sh << SCRIPT
#!/bin/bash
set -e
source $CONDA_SH
conda activate open_clip
cd $SCRIPT_DIR
export CUDA_VISIBLE_DEVICES=3

echo "[GPU3] \$(date) — Ablation: +BiSAM..."
python clipsam.py \\
    --clip-model ViT-L-14 --epochs $EPOCHS --seed $SEED \\
    --batch-size 8 --num-workers 8 --mse-lambda 0.1 \\
    --output-dir outputs/ablation/plus_bisam \\
    --eval-datasets DUTS-TE,DUT-O,SIP \\
    --simple-decoder --plain-bce-loss --pp-loss-weight 0 --ds-weight 0

echo "[GPU3] \$(date) — Ablation: +Dec (correlation decoder + compound loss)..."
python clipsam.py \\
    --clip-model ViT-L-14 --epochs $EPOCHS --seed $SEED \\
    --batch-size 8 --num-workers 8 --mse-lambda 0.1 \\
    --output-dir outputs/ablation/plus_dec \\
    --eval-datasets DUTS-TE,DUT-O,SIP \\
    --pp-loss-weight 0 --ds-weight 0.3

echo "[GPU3] \$(date) — Ablation: +Lpt (point loss = full model)..."
python clipsam.py \\
    --clip-model ViT-L-14 --epochs $EPOCHS --seed $SEED \\
    --batch-size 8 --num-workers 8 --mse-lambda 0.1 \\
    --output-dir outputs/ablation/plus_lpt \\
    --eval-datasets DUTS-TE,DUT-O,SIP \\
    --pp-loss-weight 0.3 --ds-weight 0.3

echo "[GPU3] \$(date) — BATCH 1 COMPLETE."
SCRIPT
chmod +x /tmp/archB_gpu3.sh

# ── LAUNCH ────────────────────────────────────────────────────────────────
nohup bash /tmp/archB_gpu1.sh > outputs/logs/batch1_gpu1.log 2>&1 &
PID1=$!
nohup bash /tmp/archB_gpu2.sh > outputs/logs/batch1_gpu2.log 2>&1 &
PID2=$!
nohup bash /tmp/archB_gpu3.sh > outputs/logs/batch1_gpu3.log 2>&1 &
PID3=$!

echo "$PID1 $PID2 $PID3" > outputs/logs/.pids_batch1
echo ""
echo "  GPU 1: PID $PID1 (full_model → learnable)"
echo "  GPU 2: PID $PID2 (baseline → +mp → +af)"
echo "  GPU 3: PID $PID3 (+bisam → +dec → +lpt)"
echo ""
echo "  Logs: outputs/logs/batch1_gpu{1,2,3}.log"
echo "  Monitor: tail -f outputs/logs/batch1_gpu1.log"
echo ""
echo "  ~30 min/epoch × 30 epochs = ~15h per run"
echo "  GPU 1: 2 runs = ~30h (needs batch 2 for learnable if killed)"
echo "  GPU 2: 3 runs = ~45h (will need multiple batches)" 
echo "  GPU 3: 3 runs = ~45h (will need multiple batches)"
echo ""
echo "  NOTE: System kills after ~12h. Monitor and relaunch as needed."
