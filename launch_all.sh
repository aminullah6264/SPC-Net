#!/bin/bash
# SPC-Net experiment launcher — distributes across GPUs 1, 2, 3
# Usage: bash launch_all.sh [--dry-run]
set -e

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN — commands will be printed but not executed ==="
    echo ""
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Conda setup
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")
CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"

# Common args
EPOCHS=60
SEED=42
BATCH_SIZE=8
NUM_WORKERS=8
CLIP_MODEL="ViT-L-14"
DATASET_ROOT="/mnt/data/projects/CAD_Adaptation/SOD_Datasets"

# Create output directories
mkdir -p outputs/logs outputs/full_model outputs/ablation outputs/prompt_analysis

# ============================================================
# GPU 1: Full model → then prompt analysis (K=1, K=5, K=20)
# ============================================================
cat > /tmp/spcnet_gpu1.sh << 'GPUSCRIPT'
#!/bin/bash
set -e
source CONDA_SH_PLACEHOLDER
conda activate open_clip
cd SCRIPT_DIR_PLACEHOLDER

export CUDA_VISIBLE_DEVICES=1

echo "[GPU 1] $(date) — Starting full model training..."
python clipsam.py \
    --clip-model ViT-L-14 \
    --epochs 60 --batch-size 8 --num-workers 8 \
    --mse-lambda 0.1 --seed 42 \
    --output-dir outputs/full_model \
    --eval-datasets DUTS-TE,DUT-O,ECSSD,HKU-IS,PASCAL-S,SIP \
    --img-path /mnt/data/projects/CAD_Adaptation/SOD_Datasets/DUTS_Dataset/DUTS-TR/DUTS-TR-Image \
    --mask-path /mnt/data/projects/CAD_Adaptation/SOD_Datasets/DUTS_Dataset/DUTS-TR/DUTS-TR-Mask \
    --val-img /mnt/data/projects/CAD_Adaptation/SOD_Datasets/DUTS_Dataset/DUTS-TE/DUTS-TE-Image \
    --val-mask /mnt/data/projects/CAD_Adaptation/SOD_Datasets/DUTS_Dataset/DUTS-TE/DUTS-TE-Mask

echo "[GPU 1] $(date) — Full model done. Starting prompt analysis..."
python run_prompt_analysis.py \
    --gpu 1 --epochs 60 --seed 42 \
    --clip-model ViT-L-14 \
    --variants K1,K5,K20 \
    --output-dir outputs/prompt_analysis

echo "[GPU 1] $(date) — All tasks complete."
GPUSCRIPT

sed -i "s|CONDA_SH_PLACEHOLDER|$CONDA_SH|g" /tmp/spcnet_gpu1.sh
sed -i "s|SCRIPT_DIR_PLACEHOLDER|$SCRIPT_DIR|g" /tmp/spcnet_gpu1.sh
chmod +x /tmp/spcnet_gpu1.sh

# ============================================================
# GPU 2: Ablation (baseline, +mp, +af) → then prompt analysis (mean, random)
# ============================================================
cat > /tmp/spcnet_gpu2.sh << 'GPUSCRIPT'
#!/bin/bash
set -e
source CONDA_SH_PLACEHOLDER
conda activate open_clip
cd SCRIPT_DIR_PLACEHOLDER

export CUDA_VISIBLE_DEVICES=2

echo "[GPU 2] $(date) — Starting ablation (baseline, +mp, +af)..."
python run_ablation.py \
    --gpu 2 --epochs 60 --seed 42 \
    --clip-model ViT-L-14 \
    --variants baseline,+mp,+af \
    --output-dir outputs/ablation

echo "[GPU 2] $(date) — Ablation done. Starting prompt analysis..."
python run_prompt_analysis.py \
    --gpu 2 --epochs 60 --seed 42 \
    --clip-model ViT-L-14 \
    --variants agg_mean,random_bank \
    --output-dir outputs/prompt_analysis

echo "[GPU 2] $(date) — All tasks complete."
GPUSCRIPT

sed -i "s|CONDA_SH_PLACEHOLDER|$CONDA_SH|g" /tmp/spcnet_gpu2.sh
sed -i "s|SCRIPT_DIR_PLACEHOLDER|$SCRIPT_DIR|g" /tmp/spcnet_gpu2.sh
chmod +x /tmp/spcnet_gpu2.sh

# ============================================================
# GPU 3: Ablation (+bisam, +dec, +lpt) → then prompt analysis (γ=0, γ=1)
# ============================================================
cat > /tmp/spcnet_gpu3.sh << 'GPUSCRIPT'
#!/bin/bash
set -e
source CONDA_SH_PLACEHOLDER
conda activate open_clip
cd SCRIPT_DIR_PLACEHOLDER

export CUDA_VISIBLE_DEVICES=3

echo "[GPU 3] $(date) — Starting ablation (+bisam, +dec, +lpt)..."
python run_ablation.py \
    --gpu 3 --epochs 60 --seed 42 \
    --clip-model ViT-L-14 \
    --variants +bisam,+dec,+lpt \
    --output-dir outputs/ablation

echo "[GPU 3] $(date) — Ablation done. Starting prompt analysis..."
python run_prompt_analysis.py \
    --gpu 3 --epochs 60 --seed 42 \
    --clip-model ViT-L-14 \
    --variants gamma0,gamma1 \
    --output-dir outputs/prompt_analysis

echo "[GPU 3] $(date) — All tasks complete."
GPUSCRIPT

sed -i "s|CONDA_SH_PLACEHOLDER|$CONDA_SH|g" /tmp/spcnet_gpu3.sh
sed -i "s|SCRIPT_DIR_PLACEHOLDER|$SCRIPT_DIR|g" /tmp/spcnet_gpu3.sh
chmod +x /tmp/spcnet_gpu3.sh

# ============================================================
# Launch
# ============================================================
if [ "$DRY_RUN" = true ]; then
    echo "=== GPU 1 script ==="
    cat /tmp/spcnet_gpu1.sh
    echo ""
    echo "=== GPU 2 script ==="
    cat /tmp/spcnet_gpu2.sh
    echo ""
    echo "=== GPU 3 script ==="
    cat /tmp/spcnet_gpu3.sh
else
    echo "============================================================"
    echo " SPC-Net Experiments — Launching on GPUs 1, 2, 3"
    echo " Time: $(date)"
    echo " Logs: outputs/logs/gpu{1,2,3}.log"
    echo "============================================================"
    echo ""

    # Launch in background with nohup for persistence
    nohup bash /tmp/spcnet_gpu1.sh > outputs/logs/gpu1.log 2>&1 &
    PID1=$!
    echo "  GPU 1: PID $PID1 (full model → prompt K sweep)"

    nohup bash /tmp/spcnet_gpu2.sh > outputs/logs/gpu2.log 2>&1 &
    PID2=$!
    echo "  GPU 2: PID $PID2 (ablation baseline/+mp/+af → mean/random)"

    nohup bash /tmp/spcnet_gpu3.sh > outputs/logs/gpu3.log 2>&1 &
    PID3=$!
    echo "  GPU 3: PID $PID3 (ablation +bisam/+dec/+lpt → γ sweep)"

    echo ""
    echo "  PIDs saved to outputs/logs/.pids"
    echo "$PID1 $PID2 $PID3" > outputs/logs/.pids

    echo ""
    echo "  Monitor with: bash monitor.sh"
    echo "  Tail logs:    tail -f outputs/logs/gpu1.log"
    echo ""
    echo "  Estimated completion: ~7 hours ($(date -d '+7 hours' 2>/dev/null || date -v+7H 2>/dev/null || echo 'check manually'))"
fi
