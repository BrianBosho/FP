#!/bin/bash
# Monitor GPU and CPU during experiment run

LOGFILE="experiments/results/single_test/profiling.log"
mkdir -p experiments/results/single_test

echo "timestamp,gpu_util,gpu_mem_used,gpu_mem_total,cpu_load" > "$LOGFILE"

# Monitor in background
while true; do
    TS=$(date +%s)
    GPU_DATA=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    CPU_LOAD=$(cat /proc/loadavg | cut -d' ' -f1)
    echo "${TS},${GPU_DATA},${CPU_LOAD}" >> "$LOGFILE"
    sleep 2
done &
MONITOR_PID=$!

# Run experiment
echo "Starting experiment at $(date)"
cd /home/bosho/FP
/home/bosho/.conda/envs/fedgnn/bin/python -m src.experiments.run_experiments \
    --config experiments/configs/single_cora_test.yaml \
    --datasets Cora \
    --data_loading full \
    --beta 1 \
    --models GCN \
    --clients 1 \
    --results_dir experiments/results/single_test \
    --save_results \
    --hop 1 \
    2>&1 | tee experiments/results/single_test/experiment.log

# Stop monitoring
kill $MONITOR_PID 2>/dev/null
wait $MONITOR_PID 2>/dev/null

echo ""
echo "Experiment complete. Profiling data:"
echo "GPU/CPU log: $LOGFILE"
echo "Experiment log: experiments/results/single_test/experiment.log"
