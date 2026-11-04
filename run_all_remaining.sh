#!/bin/bash
# ============================================================
# SPC-Net: Complete all remaining experiments
# Architecture B (correlation residual), ViT-L/14, 30 epochs
#
# Designed for GNU screen: survives terminal close
#
# Usage:
#   screen -S gpu1 bash run_all_remaining.sh gpu1
#   screen -S gpu2 bash run_all_remaining.sh gpu2
#   screen -S gpu3 bash run_all_remaining.sh gpu3
#
# Or launch all at once:
#   bash run_all_remaining.sh launch_all
#
# Results saved to: /mnt/data/projects/CAD_Adaptation/CLIPSAM_new_results/outputs/
# Only best checkpoint saved (--save-best-only) to conserve disk
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_SH="/home/av543f/anaconda3/etc/profile.d/conda.sh"
RESULTS_ROOT="/mnt/data/projects/CAD_Adaptation/CLIPSAM_new_results/outputs"
EPOCHS=30
SEED=42
COMMON="--clip-model ViT-L-14 --epochs $EPOCHS --seed $SEED --batch-size 8 --num-workers 8 --mse-lambda 0.1 --save-best-only"

run_experiment() {
    local name="$1"
    local outdir="$2"
    shift 2
    local flags="$@"
    
    echo ""
    echo "============================================================"
    echo " EXPERIMENT: $name"
    echo " Output: $outdir"
    echo " Time: $(date)"
    echo "============================================================"
    
    # Skip if already completed (has results.yaml with 30 epochs)
    if [ -f "$outdir/results.yaml" ]; then
        local ep_count=$(wc -l < "$outdir/training_diagnostics.csv" 2>/dev/null | xargs -I{} expr {} - 1 2>/dev/null || echo 0)
        if [ "$ep_count" -ge 28 ]; then
            echo "[SKIP] Already completed ($ep_count epochs, results.yaml exists)"
            return 0
        fi
    fi
    
    # Clean old incomplete data
    rm -rf "$outdir"
    mkdir -p "$outdir"
    
    python clipsam.py $COMMON --output-dir "$outdir" $flags
    
    echo "[DONE] $name completed at $(date)"
}

# ============================================================
# GPU 1: Ablation reruns + K5 fix
# ============================================================
run_gpu1() {
    source "$CONDA_SH"
    conda activate open_clip
    cd "$SCRIPT_DIR"
    export CUDA_VISIBLE_DEVICES=1
    
    echo "=========================================="
    echo " GPU 1 — Starting at $(date)"
    echo "=========================================="
    
    # 1. +AF ablation (was 18/30, redo full)
    run_experiment "+AF (adaptive fallback)" \
        "$RESULTS_ROOT/ablation/plus_af" \
        --eval-datasets DUTS-TE,DUT-O,SIP \
        --no-bidirectional-sam --simple-decoder --plain-bce-loss \
        --pp-loss-weight 0 --ds-weight 0
    
    # 2. +Dec ablation (was 19/30, redo full)
    run_experiment "+Dec (correlation decoder)" \
        "$RESULTS_ROOT/ablation/plus_dec" \
        --eval-datasets DUTS-TE,DUT-O,SIP \
        --pp-loss-weight 0 --ds-weight 0.3
    
    # 3. K=5 (anomalous result, rerun)
    run_experiment "K=5 (prompt bank size)" \
        "$RESULTS_ROOT/prompt_analysis/K5" \
        --eval-datasets DUTS-TE \
        --prompt-bank-size 5 \
        --pp-loss-weight 0.3 --ds-weight 0.3
    
    echo ""
    echo "=========================================="
    echo " GPU 1 — ALL COMPLETE at $(date)"
    echo "=========================================="
}

# ============================================================
# GPU 2: Gated redo + K20 + new combo experiment
# ============================================================
run_gpu2() {
    source "$CONDA_SH"
    conda activate open_clip
    cd "$SCRIPT_DIR"
    export CUDA_VISIBLE_DEVICES=2
    
    echo "=========================================="
    echo " GPU 2 — Starting at $(date)"
    echo "=========================================="
    
    # 1. Gated prompts (was 22/30, redo full)
    run_experiment "Gated prompts" \
        "$RESULTS_ROOT/prompt_modes/gated" \
        --eval-datasets DUTS-TE,DUT-O,SIP \
        --prompt-agg gated \
        --pp-loss-weight 0.3 --ds-weight 0.3
    
    # 2. K=20 (never completed)
    run_experiment "K=20 (prompt bank size)" \
        "$RESULTS_ROOT/prompt_analysis/K20" \
        --eval-datasets DUTS-TE \
        --prompt-bank-size 20 \
        --pp-loss-weight 0.3 --ds-weight 0.3
    
    echo ""
    echo "=========================================="
    echo " GPU 2 — ALL COMPLETE at $(date)"
    echo "=========================================="
}

# ============================================================
# GPU 3: Best combo experiments (new — not run before)
# ============================================================
run_gpu3() {
    source "$CONDA_SH"
    conda activate open_clip
    cd "$SCRIPT_DIR"
    export CUDA_VISIBLE_DEVICES=3
    
    echo "=========================================="
    echo " GPU 3 — Starting at $(date)"
    echo "=========================================="
    
    # 1. +MP with Attention aggregation (best component + best prompt mode)
    #    Previous ablation showed +MP is the best single component
    #    and Attention is the best aggregation mode
    #    But they were never tested together without the other components
    run_experiment "+MP with Attention agg (no BiSAM/Dec/Lpt)" \
        "$RESULTS_ROOT/prompt_modes/mp_attention" \
        --eval-datasets DUTS-TE,DUT-O,SIP \
        --prompt-agg attention \
        --no-adaptive-fallback --no-bidirectional-sam \
        --simple-decoder --plain-bce-loss \
        --pp-loss-weight 0 --ds-weight 0
    
    # 2. Full model with Attention (all components + attention agg)
    #    vs the existing full model with fixed max
    run_experiment "Full model + Attention agg" \
        "$RESULTS_ROOT/prompt_modes/full_attention" \
        --eval-datasets DUTS-TE,DUT-O,SIP \
        --prompt-agg attention \
        --pp-loss-weight 0.3 --ds-weight 0.3
    
    # 3. Baseline with Attention (no extra components, just attention agg)
    #    Tests if attention alone improves over single-prompt baseline
    run_experiment "Baseline + Attention agg (no prompt bank components)" \
        "$RESULTS_ROOT/prompt_modes/baseline_attention" \
        --eval-datasets DUTS-TE,DUT-O,SIP \
        --prompt-agg attention \
        --no-adaptive-fallback --no-bidirectional-sam \
        --simple-decoder --plain-bce-loss \
        --pp-loss-weight 0 --ds-weight 0
    
    echo ""
    echo "=========================================="
    echo " GPU 3 — ALL COMPLETE at $(date)"
    echo "=========================================="
}

# ============================================================
# Launch mode
# ============================================================
case "${1:-}" in
    gpu1)
        run_gpu1 2>&1 | tee "$RESULTS_ROOT/../logs/gpu1_$(date +%Y%m%d_%H%M).log"
        ;;
    gpu2)
        run_gpu2 2>&1 | tee "$RESULTS_ROOT/../logs/gpu2_$(date +%Y%m%d_%H%M).log"
        ;;
    gpu3)
        run_gpu3 2>&1 | tee "$RESULTS_ROOT/../logs/gpu3_$(date +%Y%m%d_%H%M).log"
        ;;
    launch_all)
        echo "============================================================"
        echo " Launching all 3 GPUs in screen sessions"
        echo " Time: $(date)"
        echo "============================================================"
        
        mkdir -p "$RESULTS_ROOT/../logs"
        
        # Launch each GPU in a detached screen session
        screen -dmS spcnet_gpu1 bash -c "cd $SCRIPT_DIR && bash run_all_remaining.sh gpu1"
        echo "  GPU 1: screen -r spcnet_gpu1  (+AF, +Dec, K5)"
        
        screen -dmS spcnet_gpu2 bash -c "cd $SCRIPT_DIR && bash run_all_remaining.sh gpu2"
        echo "  GPU 2: screen -r spcnet_gpu2  (Gated, K20)"
        
        screen -dmS spcnet_gpu3 bash -c "cd $SCRIPT_DIR && bash run_all_remaining.sh gpu3"
        echo "  GPU 3: screen -r spcnet_gpu3  (MP+Attn, Full+Attn, Base+Attn)"
        
        echo ""
        echo "  Monitor: screen -ls"
        echo "  Attach:  screen -r spcnet_gpu1"
        echo "  Detach:  Ctrl+A then D"
        echo ""
        echo "  Logs: $RESULTS_ROOT/../logs/"
        echo "============================================================"
        ;;
    *)
        echo "Usage: bash run_all_remaining.sh {gpu1|gpu2|gpu3|launch_all}"
        echo ""
        echo "GPU assignments:"
        echo "  gpu1: +AF (redo), +Dec (redo), K=5 (rerun)          — 3 runs"
        echo "  gpu2: Gated (redo), K=20                            — 2 runs"
        echo "  gpu3: MP+Attention, Full+Attention, Base+Attention  — 3 runs (NEW)"
        echo ""
        echo "launch_all: Start all 3 in detached screen sessions"
        exit 1
        ;;
esac
