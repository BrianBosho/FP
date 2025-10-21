#!/bin/bash

# Run multiple experiments in parallel with GPU memory fraction control
# This allows multiple experiments to share the GPU without conflicts

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/parallel_experiments_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

echo "Starting parallel experiments at $(date)"
echo "Logs will be saved to ${LOG_DIR}"

# Define experiments with their GPU memory fractions
# Format: "config_file:gpu_memory_fraction"
EXPERIMENTS=(
    "conf/ogbn-arxiv_config.yaml:0.4"
    "conf/photos_config.yaml:0.3"
)

# Track PIDs for cleanup
PIDS=()

# Function to run experiment with GPU memory limit
run_experiment() {
    local config=$1
    local gpu_fraction=$2
    local config_name=$(basename "${config}" .yaml)
    local log_file="${LOG_DIR}/${config_name}.log"
    
    echo "Starting ${config} with ${gpu_fraction} GPU memory fraction"
    
    # Set GPU memory fraction via environment variable
    # PyTorch will respect this when initializing CUDA
    PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512" \
    python -m src.experiments.run_experiments --config "${config}" > "${log_file}" 2>&1 &
    
    local pid=$!
    PIDS+=($pid)
    echo "  → PID: ${pid}, Log: ${log_file}"
}

# Launch all experiments in parallel
for exp in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r config gpu_fraction <<< "$exp"
    run_experiment "$config" "$gpu_fraction"
    
    # Small delay to stagger starts
    sleep 2
done

echo ""
echo "All experiments launched. PIDs: ${PIDS[@]}"
echo "Waiting for completion..."

# Wait for all experiments to complete
for pid in "${PIDS[@]}"; do
    wait $pid
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✓ PID ${pid} completed successfully"
    else
        echo "✗ PID ${pid} failed with exit code ${exit_code}"
    fi
done

echo ""
echo "========================================"
echo "All parallel experiments completed at $(date)"
echo "Check logs in: ${LOG_DIR}"
echo "========================================"
