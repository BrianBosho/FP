#!/bin/bash

# Queue-based experiment runner
# Runs N experiments in parallel, queuing the rest
# Each experiment uses a unique Ray port to avoid conflicts

MAX_PARALLEL=2  # Number of experiments to run simultaneously
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/queued_experiments_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

echo "Starting queued experiment runner at $(date)"
echo "Max parallel experiments: ${MAX_PARALLEL}"
echo "Logs will be saved to ${LOG_DIR}"

# List of all experiments to run
EXPERIMENTS=(
    "conf/ogbn-arxiv_config.yaml"
    "conf/photos_config.yaml"
    "conf/cora_config.yaml"
    "conf/citeseer_config.yaml"
    "conf/pubmed_config.yaml"
    "conf/computers_config.yaml"
)

# Function to get next available Ray port
get_ray_port() {
    # Start from 6379 (default) and increment
    local base_port=6379
    local port=$((base_port + $1))
    echo $port
}

# Function to run a single experiment
run_single_experiment() {
    local config=$1
    local exp_id=$2
    local config_name=$(basename "${config}" .yaml)
    local log_file="${LOG_DIR}/${config_name}_${exp_id}.log"
    local ray_port=$(get_ray_port $exp_id)
    
    echo "[$(date)] Starting: ${config} (Ray port: ${ray_port})"
    
    # Set unique Ray port to avoid conflicts
    RAY_PORT=${ray_port} \
    python -m src.experiments.run_experiments --config "${config}" > "${log_file}" 2>&1
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[$(date)] ✓ Completed: ${config}"
    else
        echo "[$(date)] ✗ Failed: ${config} (exit code: ${exit_code})"
    fi
    
    return $exit_code
}

# Main execution loop with queue management
running_jobs=0
completed=0
failed=0
exp_id=0

for config in "${EXPERIMENTS[@]}"; do
    # Wait if we've reached max parallel limit
    while [ $running_jobs -ge $MAX_PARALLEL ]; do
        sleep 5
        # Count currently running background jobs
        running_jobs=$(jobs -r | wc -l)
    done
    
    # Launch experiment in background
    run_single_experiment "$config" $exp_id &
    exp_id=$((exp_id + 1))
    running_jobs=$((running_jobs + 1))
    
    echo "  → Active jobs: ${running_jobs}/${MAX_PARALLEL}"
    
    # Small delay to stagger starts
    sleep 3
done

# Wait for all remaining jobs to complete
echo ""
echo "All experiments queued. Waiting for completion..."
wait

# Count results
for config in "${EXPERIMENTS[@]}"; do
    config_name=$(basename "${config}" .yaml)
    if grep -q "✓" "${LOG_DIR}/${config_name}"*.log 2>/dev/null; then
        completed=$((completed + 1))
    else
        failed=$((failed + 1))
    fi
done

echo ""
echo "========================================"
echo "Experiment Queue Complete at $(date)"
echo "Total: ${#EXPERIMENTS[@]}"
echo "Completed: ${completed}"
echo "Failed: ${failed}"
echo "Logs: ${LOG_DIR}"
echo "========================================"
