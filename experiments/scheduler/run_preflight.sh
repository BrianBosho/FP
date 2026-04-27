#!/bin/bash
# Pre-flight GPU check script
# Runs a quick Cora experiment to verify GPU is working

set -e

echo "=============================================="
echo "🚀 PRE-FLIGHT GPU CHECK"
echo "=============================================="
echo ""

# Check GPU is available
if ! nvidia-smi > /dev/null 2>&1; then
    echo "❌ ERROR: nvidia-smi not found. GPU not available."
    exit 1
fi

echo "✅ GPU detected"
echo ""

# Record start time
START_TIME=$(date +%s)

# Run pre-flight experiment
cd /home/bosho/FP
/home/bosho/.conda/envs/fedgnn/bin/python -m src.experiments.run_experiments \
    --config experiments/configs/preflight_cora.yaml \
    --datasets Cora \
    --data_loading full \
    --beta 1 \
    --models GCN \
    --clients 10 \
    --results_dir experiments/results/preflight \
    --save_results \
    --hop 1

# Record end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "=============================================="
echo "📊 PRE-FLIGHT RESULTS"
echo "=============================================="
echo "Duration: ${DURATION} seconds"

if [ $DURATION -lt 30 ]; then
    echo "✅ PASS: Completed in < 30 seconds"
else
    echo "⚠️  WARNING: Took ${DURATION} seconds (expected < 30)"
fi

# Check GPU utilization during run
GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
echo "Current GPU utilization: ${GPU_UTIL}%"

if [ "$GPU_UTIL" -gt 80 ]; then
    echo "✅ PASS: GPU utilization > 80%"
else
    echo "⚠️  WARNING: GPU utilization only ${GPU_UTIL}%"
fi

echo ""
echo "=============================================="
echo "🎉 PRE-FLIGHT COMPLETE"
echo "=============================================="
