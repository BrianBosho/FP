#!/bin/bash
# Quick health check for R1 experiment run
LOG="/tmp/run_R1.log"
PID=$(pgrep -f "run.py --track fedprop --result R1" | head -1)

echo "=== FedProp R1 Health Check ==="
echo "Time: $(date)"

if [ -n "$PID" ]; then
    echo "Status: RUNNING (PID $PID)"
    echo "Log size: $(ls -lh $LOG 2>/dev/null | awk '{print $5}')"
    echo "GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null)"
    echo "--- Recent log tail ---"
    tail -8 $LOG 2>/dev/null
    echo "--- Result count ---"
    find /home/bosho/FP/experiments/results/R1 -name "*.json" 2>/dev/null | wc -l
else
    echo "Status: NOT RUNNING"
    echo "--- Last log lines ---"
    tail -20 $LOG 2>/dev/null
fi
