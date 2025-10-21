#!/bin/bash

# Advanced parallel experiment runner with custom Ray ports
# This version assigns unique Ray ports to each parallel experiment

MAX_PARALLEL=2  # Number of experiments to run simultaneously
BASE_RAY_PORT=6379  # Starting port for Ray
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/parallel_ray_experiments_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

echo "========================================"
echo "Advanced Parallel Experiment Runner"
echo "========================================"
echo "Started at: $(date)"
echo "Max parallel experiments: ${MAX_PARALLEL}"
echo "Base Ray port: ${BASE_RAY_PORT}"
echo "Logs directory: ${LOG_DIR}"
echo ""

# List of experiments to run
EXPERIMENTS=(
    "conf/ogbn-arxiv_config.yaml"
    "conf/photos_config.yaml"
    "conf/cora_config.yaml"
    "conf/citeseer_config.yaml"
)

# Track running jobs
declare -A RUNNING_JOBS  # PID -> config mapping
declare -A JOB_PORTS     # PID -> Ray port mapping
NEXT_PORT=${BASE_RAY_PORT}

# Function to get next available port
get_next_port() {
    local port=${NEXT_PORT}
    NEXT_PORT=$((NEXT_PORT + 1))
    echo ${port}
}

# Function to run a single experiment with custom Ray port
run_experiment_with_port() {
    local config=$1
    local ray_port=$2
    local config_name=$(basename "${config}" .yaml)
    local log_file="${LOG_DIR}/${config_name}_port${ray_port}.log"
    
    echo "[$(date '+%H:%M:%S')] Starting: ${config}"
    echo "  → Ray port: ${ray_port}"
    echo "  → Log file: ${log_file}"
    
    # Run experiment with custom Ray port
    python -m src.experiments.run_experiments \
        --config "${config}" \
        --ray_port ${ray_port} \
        > "${log_file}" 2>&1 &
    
    local pid=$!
    echo "  → PID: ${pid}"
    echo ""
    
    # Store job info
    RUNNING_JOBS[${pid}]="${config}"
    JOB_PORTS[${pid}]=${ray_port}
}

# Function to wait for a job slot to become available
wait_for_slot() {
    while [ ${#RUNNING_JOBS[@]} -ge ${MAX_PARALLEL} ]; do
        # Check which jobs are still running
        for pid in "${!RUNNING_JOBS[@]}"; do
            if ! kill -0 ${pid} 2>/dev/null; then
                # Job finished
                local config="${RUNNING_JOBS[${pid}]}"
                local port="${JOB_PORTS[${pid}]}"
                local config_name=$(basename "${config}" .yaml)
                
                # Check exit status
                wait ${pid}
                local exit_code=$?
                
                if [ ${exit_code} -eq 0 ]; then
                    echo "[$(date '+%H:%M:%S')] ✓ Completed: ${config_name} (PID: ${pid}, Port: ${port})"
                else
                    echo "[$(date '+%H:%M:%S')] ✗ Failed: ${config_name} (PID: ${pid}, Port: ${port}, Exit: ${exit_code})"
                fi
                
                # Remove from tracking
                unset RUNNING_JOBS[${pid}]
                unset JOB_PORTS[${pid}]
            fi
        done
        
        # If still at max capacity, wait a bit
        if [ ${#RUNNING_JOBS[@]} -ge ${MAX_PARALLEL} ]; then
            sleep 5
        fi
    done
}

# Main execution loop
echo "Queueing ${#EXPERIMENTS[@]} experiments..."
echo ""

for config in "${EXPERIMENTS[@]}"; do
    # Wait for an available slot
    wait_for_slot
    
    # Get next port and launch experiment
    ray_port=$(get_next_port)
    run_experiment_with_port "${config}" ${ray_port}
    
    # Small delay to stagger starts
    sleep 3
done

# Wait for all remaining jobs
echo "All experiments queued. Waiting for completion..."
echo ""

while [ ${#RUNNING_JOBS[@]} -gt 0 ]; do
    for pid in "${!RUNNING_JOBS[@]}"; do
        if ! kill -0 ${pid} 2>/dev/null; then
            local config="${RUNNING_JOBS[${pid}]}"
            local port="${JOB_PORTS[${pid}]}"
            local config_name=$(basename "${config}" .yaml)
            
            wait ${pid}
            local exit_code=$?
            
            if [ ${exit_code} -eq 0 ]; then
                echo "[$(date '+%H:%M:%S')] ✓ Completed: ${config_name} (PID: ${pid}, Port: ${port})"
            else
                echo "[$(date '+%H:%M:%S')] ✗ Failed: ${config_name} (PID: ${pid}, Port: ${port}, Exit: ${exit_code})"
            fi
            
            unset RUNNING_JOBS[${pid}]
            unset JOB_PORTS[${pid}]
        fi
    done
    
    if [ ${#RUNNING_JOBS[@]} -gt 0 ]; then
        sleep 5
    fi
done

# Final summary
echo ""
echo "========================================"
echo "All Experiments Complete"
echo "========================================"
echo "Finished at: $(date)"
echo "Logs saved to: ${LOG_DIR}"
echo ""
echo "Summary:"
completed=0
failed=0
for config in "${EXPERIMENTS[@]}"; do
    config_name=$(basename "${config}" .yaml)
    log_files=("${LOG_DIR}/${config_name}"_port*.log)
    if [ -f "${log_files[0]}" ]; then
        if grep -q "Experiment completed successfully" "${log_files[0]}" 2>/dev/null || \
           grep -q "Results saved" "${log_files[0]}" 2>/dev/null; then
            echo "  ✓ ${config_name}"
            completed=$((completed + 1))
        else
            echo "  ✗ ${config_name} (check log for details)"
            failed=$((failed + 1))
        fi
    else
        echo "  ? ${config_name} (log not found)"
        failed=$((failed + 1))
    fi
done

echo ""
echo "Total: ${#EXPERIMENTS[@]}"
echo "Completed: ${completed}"
echo "Failed: ${failed}"
echo "========================================"
