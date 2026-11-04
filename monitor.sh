#!/bin/bash
# Monitor SPC-Net experiment progress
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== SPC-Net Experiment Monitor ==="
echo "Time: $(date)"
echo ""

# Check PIDs
if [ -f outputs/logs/.pids ]; then
    read -r PID1 PID2 PID3 < outputs/logs/.pids
    echo "Process status:"
    for pid in $PID1 $PID2 $PID3; do
        if kill -0 $pid 2>/dev/null; then
            echo "  PID $pid: RUNNING"
        else
            echo "  PID $pid: FINISHED"
        fi
    done
else
    echo "No .pids file found. Are experiments running?"
fi

echo ""
echo "=== GPU Utilization ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
    awk -F', ' '{printf "  GPU %s: %s%% util, %s/%s MB\n", $1, $3, $4, $5}'

echo ""
echo "=== Latest log lines ==="
for gpu in 1 2 3; do
    log="outputs/logs/gpu${gpu}.log"
    if [ -f "$log" ]; then
        echo "--- GPU $gpu ($(wc -l < $log) lines) ---"
        tail -3 "$log"
        echo ""
    fi
done

echo ""
echo "=== Completed experiments ==="
for dir in outputs/full_model outputs/ablation/*/  outputs/prompt_analysis/*/; do
    if [ -f "${dir}escnet_best.pth" ] 2>/dev/null || [ -f "${dir}results.yaml" ] 2>/dev/null; then
        echo "  ✓ $dir"
    fi
done
