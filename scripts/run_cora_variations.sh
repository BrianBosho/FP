#!/bin/bash

# Run Cora experiment variations: GAT/GCN, beta=1/10000, PE=True (set in config)
# Runs 4 experiments in parallel batches

MAX_PARALLEL=4  # Run up to 4 at a time
BASE_RAY_PORT=6379
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/cora_variations_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

echo "========================================"
echo "Cora Experiment Variations Runner"
echo "========================================"
echo "Started at: $(date)"
echo "Max parallel: ${MAX_PARALLEL}"
echo "Logs: ${LOG_DIR}"
echo ""

# Kill any existing Ray processes to avoid conflicts
echo "Killing any existing Ray processes..."
pkill -f ray || true
sleep 2

BASE_CONFIG="conf/cora_config.yaml"
MODELS=("GCN" "GAT")
BETAS=(1 10000)
USE_PE="True"  # Fixed in config

# Track jobs
declare -A RUNNING_JOBS
declare -A JOB_PORTS
NEXT_PORT=${BASE_RAY_PORT}

# Function to get next available port
get_next_port() {
    local port=${NEXT_PORT}
    NEXT_PORT=$((NEXT_PORT + 1))
    echo ${port}
}

# Function to run experiment with overrides
run_experiment() {
    local model=$1
    local beta=$2
    local ray_port=$3
    local exp_name="${model}_beta${beta}_pe${USE_PE}"
    local log_file="${LOG_DIR}/${exp_name}_port${ray_port}.log"
    
    echo "[$(date '+%H:%M:%S')] Starting: ${exp_name}"
    echo "  → Ray port: ${ray_port}, Log: ${log_file}"
    
    # Set Ray port via environment variable for this process
    RAY_REDIS_ADDRESS="127.0.0.1:${ray_port}" \
    python -m src.experiments.run_experiments \
        --config "${BASE_CONFIG}" \
        --models "${model}" \
        --beta "${beta}" \
        --ray_port ${ray_port} \
        > "${log_file}" 2>&1 &
    
    local pid=$!
    RUNNING_JOBS[${pid}]="${exp_name}"
    JOB_PORTS[${pid}]=${ray_port}
    echo "  → PID: ${pid}"
}

# Wait for slot
wait_for_slot() {
    while [ ${#RUNNING_JOBS[@]} -ge ${MAX_PARALLEL} ]; do
        for pid in "${!RUNNING_JOBS[@]}"; do
            if ! kill -0 ${pid} 2>/dev/null; then
                local exp_name="${RUNNING_JOBS[${pid}]}"
                local port="${JOB_PORTS[${pid}]}"
                wait ${pid}
                local exit_code=$?
                echo "[$(date '+%H:%M:%S')] ${exit_code:+✗}✓ Completed: ${exp_name} (PID: ${pid}, Port: ${port})"
                unset RUNNING_JOBS[${pid}] JOB_PORTS[${pid}]
            fi
        done
        [ ${#RUNNING_JOBS[@]} -ge ${MAX_PARALLEL} ] && sleep 5
    done
}

# Run experiments: 2 models × 2 betas = 4 total
echo "Starting experiments with PE=${USE_PE}"
for model in "${MODELS[@]}"; do
    for beta in "${BETAS[@]}"; do
        wait_for_slot
        ray_port=$(get_next_port)
        run_experiment "${model}" "${beta}" ${ray_port}
        sleep 3  # Stagger starts
    done
done

# Wait for all to finish
while [ ${#RUNNING_JOBS[@]} -gt 0 ]; do
    for pid in "${!RUNNING_JOBS[@]}"; do
        if ! kill -0 ${pid} 2>/dev/null; then
            local exp_name="${RUNNING_JOBS[${pid}]}"
            local port="${JOB_PORTS[${pid}]}"
            wait ${pid}
            local exit_code=$?
            echo "[$(date '+%H:%M:%S')] ${exit_code:+✗}✓ Completed: ${exp_name} (PID: ${pid}, Port: ${port})"
            unset RUNNING_JOBS[${pid}] JOB_PORTS[${pid}]
        fi
    done
    [ ${#RUNNING_JOBS[@]} -gt 0 ] && sleep 5
done
echo "All experiments complete."

echo ""
echo "========================================"
echo "All Cora Variations Complete"
echo "========================================"
echo "Finished at: $(date)"
echo "Logs: ${LOG_DIR}"
echo "Total experiments: 4 (GCN/GAT × beta=1/10000, PE=True)"
echo "========================================"